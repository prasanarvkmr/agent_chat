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
from observability import get_logger, setup_logging, metrics, get_tracer, get_agent_tracer
from auth import EntraAuthConfig, get_current_user, User
from sessions import session_manager
from metadata_cache import metadata_cache

# Setup logging
setup_logging(level="INFO")
logger = get_logger(__name__)
tracer = get_tracer()
agent_tracer = get_agent_tracer()


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


# Client Error Logging Endpoint
class ClientErrorLog(BaseModel):
    """Model for client-side error logs."""
    message: str
    error: Optional[str] = None
    stack: Optional[str] = None
    timestamp: Optional[str] = None
    url: Optional[str] = None
    userAgent: Optional[str] = None


@app.post("/api/log/error")
async def log_client_error(error_log: ClientErrorLog, user: User = Depends(get_current_user)):
    """
    Log client-side errors to the observability system.
    This enables tracking of frontend errors alongside backend errors.
    """
    logger.error(
        f"Client Error: {error_log.message}",
        extra={
            "extra_data": {
                "error": error_log.error,
                "stack": error_log.stack,
                "url": error_log.url,
                "user_agent": error_log.userAgent,
                "user_id": user.id,
                "timestamp": error_log.timestamp,
                "source": "frontend"
            }
        }
    )
    
    # Record in metrics
    metrics.request_errors.inc(endpoint="frontend")
    
    return {"success": True, "logged": True}


# Agent Observability Endpoints
@app.get("/api/agent/traces")
async def get_agent_traces():
    """
    Get recent agent trace history.
    Shows how the agent thinks, what tools it calls, and event flow.
    """
    traces = agent_tracer.get_trace_history()
    return {
        "success": True,
        "trace_count": len(traces),
        "traces": traces
    }


@app.get("/api/agent/traces/{trace_id}")
async def get_agent_trace(trace_id: str):
    """
    Get detailed information about a specific agent trace.
    Includes all events, thinking steps, tool calls, and LLM interactions.
    """
    summary = agent_tracer.get_trace_summary(trace_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"success": True, "trace": summary}


@app.get("/api/agent/traces/current")
async def get_current_agent_trace():
    """
    Get the current active agent trace (if any).
    Useful for real-time monitoring of agent execution.
    """
    current_trace = agent_tracer.get_current_trace()
    if not current_trace:
        return {"success": True, "active": False, "trace": None}
    return {
        "success": True,
        "active": True,
        "trace": current_trace.to_dict()
    }


@app.get("/api/agent/thinking/{trace_id}")
async def get_agent_thinking(trace_id: str):
    """
    Get the thinking/reasoning flow for a specific trace.
    Shows how the agent decided what to do.
    """
    summary = agent_tracer.get_trace_summary(trace_id)
    if not summary:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {
        "success": True,
        "trace_id": trace_id,
        "thinking_steps": summary.get("thinking_flow", []),
        "tool_decisions": summary.get("tool_calls_detail", [])
    }


# Metadata Cache Endpoints
@app.get("/api/metadata/info")
async def get_metadata_info():
    """Get information about the metadata cache status."""
    return {
        "success": True,
        "cache": metadata_cache.get_cache_info()
    }


@app.post("/api/metadata/refresh")
async def refresh_metadata(force: bool = False):
    """
    Refresh the table metadata cache.
    Use force=true to refresh even if cache is not stale.
    """
    logger.info(f"Metadata cache refresh requested (force={force})")
    result = metadata_cache.refresh_cache(force=force)
    return {"success": True, "result": result}


@app.get("/api/metadata/tables")
async def get_cached_tables(domain: str = None, schema: str = None):
    """
    Get cached tables, optionally filtered by domain or schema.
    """
    if domain:
        tables = metadata_cache.get_tables_by_domain(domain)
    elif schema:
        tables = metadata_cache.get_tables_by_schema(schema)
    else:
        tables = metadata_cache.get_all_tables()
    
    return {
        "success": True,
        "count": len(tables),
        "tables": tables
    }


@app.get("/api/metadata/tables/{table_name}")
async def get_cached_table(table_name: str, schema: str = None, catalog: str = None):
    """Get metadata for a specific table."""
    table = metadata_cache.get_table(table_name, schema, catalog)
    if not table:
        raise HTTPException(status_code=404, detail="Table not found in cache")
    return {"success": True, "table": table}


@app.get("/api/metadata/search")
async def search_tables(q: str):
    """Search tables by name, description, or column names."""
    if not q or len(q) < 2:
        raise HTTPException(status_code=400, detail="Query must be at least 2 characters")
    
    results = metadata_cache.search_tables(q)
    return {
        "success": True,
        "query": q,
        "count": len(results),
        "results": results
    }


@app.get("/api/metadata/summary")
async def get_metadata_summary():
    """Get a summary of available tables formatted for display."""
    return {
        "success": True,
        "summary": metadata_cache.get_metadata_summary_for_agent()
    }


# User Role Endpoint (fetches from Databricks)
@app.get("/api/user/role")
async def get_user_role(user: User = Depends(get_current_user)):
    """
    Get the user's role from Databricks.
    Roles determine what data domains the user can access.
    """
    try:
        # Query the user_roles table in Databricks
        query = f"""
        SELECT role, permissions, domains
        FROM user_roles 
        WHERE user_id = '{user.id}' OR email = '{user.email}'
        LIMIT 1
        """
        results = databricks_client.execute_query(query)
        
        if results:
            role_data = results[0]
            return {
                "success": True,
                "user_id": user.id,
                "role": role_data.get("role", "viewer"),
                "permissions": role_data.get("permissions", []),
                "domains": role_data.get("domains", [])
            }
        else:
            # Default role if not found
            return {
                "success": True,
                "user_id": user.id,
                "role": "viewer",
                "permissions": ["read"],
                "domains": ["general"]
            }
    except Exception as e:
        logger.warning(f"Failed to fetch user role: {e}")
        return {
            "success": True,
            "user_id": user.id,
            "role": "viewer",
            "permissions": ["read"],
            "domains": ["general"],
            "note": "Default role (role lookup failed)"
        }


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
