"""
FastAPI Application Module.
Provides REST API endpoints and serves the web UI.
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from typing import Optional
import uvicorn

from agent import agent_runner
from config import config
from databricks_client import databricks_client
from observability import get_logger, setup_logging, metrics, get_tracer
from auth import EntraAuthConfig, get_current_user, User
from sessions import session_manager

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)
tracer = get_tracer()


# Initialize FastAPI app
app = FastAPI(
    title="System Health Analyst",
    description="An AI-powered system health analyzer for business users - queries Azure Databricks and provides insights",
    version="1.0.0"
)

# Setup templates
templates = Jinja2Templates(directory="templates")


# Request/Response models
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    message: str
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str
    success: bool
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """Response model for health check."""
    status: str
    databricks_connected: bool
    config_valid: bool
    llm_model: str
    kong_route: str


class SessionRenameRequest(BaseModel):
    """Request model for renaming a session."""
    title: str


# API Endpoints
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Serve the main chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})


# Auth endpoints
@app.get("/api/auth/status")
async def auth_status(user: User = Depends(get_current_user)):
    """Get current authentication status."""
    return {
        "auth_enabled": EntraAuthConfig.ENABLED,
        "user": user.to_dict()
    }


@app.get("/api/auth/config")
async def auth_config():
    """Get auth configuration (for frontend MSAL setup)."""
    return {
        "enabled": EntraAuthConfig.ENABLED,
        "client_id": EntraAuthConfig.CLIENT_ID if EntraAuthConfig.ENABLED else None,
        "tenant_id": EntraAuthConfig.TENANT_ID if EntraAuthConfig.ENABLED else None,
        "authority": EntraAuthConfig.get_authority() if EntraAuthConfig.ENABLED else None
    }


# Session endpoints
@app.get("/api/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    """Get all sessions for the current user."""
    sessions = session_manager.get_user_sessions(user.id)
    return {"sessions": [s.to_dict() for s in sessions]}


@app.post("/api/sessions")
async def create_session(user: User = Depends(get_current_user)):
    """Create a new chat session."""
    session = session_manager.create_session(user.id)
    return {"session": session.to_dict()}


@app.get("/api/sessions/{session_id}/messages")
async def get_session_messages(session_id: str, user: User = Depends(get_current_user)):
    """Get all messages for a session."""
    session = session_manager.get_session(session_id)
    if not session or session.user_id != user.id:
        raise HTTPException(status_code=404, detail="Session not found")
    
    messages = [{"role": m.role, "content": m.content, "timestamp": m.timestamp} 
                for m in session.messages]
    return {"messages": messages}


@app.delete("/api/sessions/{session_id}")
async def delete_session(session_id: str, user: User = Depends(get_current_user)):
    """Delete a session."""
    if session_manager.delete_session(session_id, user.id):
        return {"status": "deleted"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.patch("/api/sessions/{session_id}")
async def rename_session(
    session_id: str, 
    request: SessionRenameRequest,
    user: User = Depends(get_current_user)
):
    """Rename a session."""
    if session_manager.rename_session(session_id, user.id, request.title):
        return {"status": "renamed"}
    raise HTTPException(status_code=404, detail="Session not found")


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, user: User = Depends(get_current_user)):
    """
    Chat endpoint - sends user message to AI agent and returns response.
    
    Args:
        request: ChatRequest containing the user's message
        
    Returns:
        ChatResponse with the agent's response
    """
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    # Ensure session exists
    session_id = request.session_id
    if not session_id:
        session = session_manager.create_session(user.id)
        session_id = session.id
    
    # Store user message in session
    session_manager.add_message(session_id, "user", request.message)
    
    # Start trace for this request
    trace = tracer.start_trace("chat_request", {"user_message_length": len(request.message)})
    
    try:
        with metrics.time_request("chat"):
            logger.info("Processing chat request", extra={"extra_data": {
                "message_preview": request.message[:100],
                "session_id": session_id,
                "user_id": user.id
            }})
            
            with tracer.span("agent_chat"):
                response = await agent_runner.chat(request.message)
            
            # Store assistant response in session
            session_manager.add_message(session_id, "assistant", response)
            
            logger.info("Chat response generated", extra={"extra_data": {"response_length": len(response)}})
            return ChatResponse(response=response, success=True, session_id=session_id)
            
    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}", exc_info=True)
        metrics.request_errors.inc(endpoint="chat")
        return ChatResponse(response=f"Error: {str(e)}", success=False)
    finally:
        tracer.end_trace()


@app.post("/api/clear")
async def clear_conversation():
    """Clear the conversation history."""
    logger.info("Clearing conversation history")
    await agent_runner.clear_history()
    return {"status": "cleared"}


@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """
    Health check endpoint - verifies configuration and connectivity.
    
    Returns:
        HealthResponse with system status
    """
    logger.debug("Health check requested")
    config_errors = config.validate()
    databricks_ok = False
    
    if not config_errors:
        databricks_ok = databricks_client.test_connection()
    
    llm_config = config.get_llm_config()
    
    return HealthResponse(
        status="healthy" if databricks_ok else "degraded",
        databricks_connected=databricks_ok,
        config_valid=len(config_errors) == 0,
        llm_model=llm_config["model"],
        kong_route=llm_config["route"]
    )


@app.get("/api/tables")
async def get_tables():
    """Get list of available tables (for debugging/testing)."""
    try:
        with metrics.time_query():
            tables = databricks_client.get_tables()
        return {"success": True, "tables": tables}
    except Exception as e:
        logger.error(f"Failed to get tables: {str(e)}")
        return {"success": False, "error": str(e)}


@app.get("/api/metrics")
async def get_metrics():
    """Get current application metrics."""
    return metrics.get_summary()


@app.get("/api/metrics/full")
async def get_full_metrics():
    """Get full metrics export."""
    return metrics.export_metrics()


# Run the application
if __name__ == "__main__":
    # Validate configuration on startup
    errors = config.validate()
    if errors:
        logger.error("Configuration errors detected")
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        print("\nPlease set the required environment variables in .env file")
    else:
        llm_config = config.get_llm_config()
        logger.info("Starting System Health Analyst", extra={"extra_data": {
            "kong_gateway": config.KONG_AI_GATEWAY_BASE_URL,
            "kong_route": llm_config['route'],
            "llm_model": llm_config['model'],
            "databricks_host": config.DATABRICKS_SERVER_HOSTNAME
        }})
        print("Starting System Health Analyst...")
        print(f"Kong AI Gateway: {config.KONG_AI_GATEWAY_BASE_URL}")
        print(f"Kong Route: /{llm_config['route']}")
        print(f"LLM Model: {llm_config['model']}")
        print(f"Databricks host: {config.DATABRICKS_SERVER_HOSTNAME}")
        print(f"Logs directory: logs/")
        
    uvicorn.run(app, host="0.0.0.0", port=8000)
