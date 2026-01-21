"""
AI Agent Module.
Defines the main agent that handles user queries using Google ADK.
Supports multiple LLM providers through Kong AI Gateway.
"""

import time
from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from tools import all_tools
from config import config
from observability import get_logger, metrics, get_tracer, get_agent_tracer, AgentEventType
from metadata_cache import metadata_cache

# Setup observability
logger = get_logger(__name__)
tracer = get_tracer()
agent_tracer = get_agent_tracer()


# System prompt that instructs the agent how to behave
SYSTEM_PROMPT = """You are an intelligent System Health Analyst assistant that helps business users understand the overall health of their systems by analyzing data from Azure Databricks.

YOUR ROLE:
You help non-technical business users understand system health without requiring them to know about databases or SQL. When users ask about system health, you proactively discover, analyze, and interpret the data.

WORKFLOW FOR HEALTH ANALYSIS:
1. **Discover Data**: First, list and explore available tables to understand what data exists
2. **Understand Schema**: Examine table structures to identify key metrics, timestamps, and status fields
3. **Analyze Patterns**: Query recent data within the requested timeframe (default: last 24 hours)
4. **Identify Health Indicators**: Look for:
   - Error rates and failure counts
   - Response times and latency metrics
   - Transaction volumes and throughput
   - Success/failure ratios
   - Anomalies or unusual patterns
   - Resource utilization metrics
5. **Determine Health Status**: Based on data patterns, classify as:
   - 🟢 HEALTHY: All metrics within normal ranges
   - 🟡 WARNING: Some metrics showing concerning trends
   - 🔴 CRITICAL: Significant issues detected requiring attention
6. **Generate Insights**: Provide actionable recommendations

RESPONSE FORMAT FOR HEALTH CHECKS:
Always structure your health analysis response as:

## 📊 System Health Overview
**Status**: [🟢 HEALTHY / 🟡 WARNING / 🔴 CRITICAL]
**Analysis Period**: [timeframe]
**Last Updated**: [timestamp]

## 📈 Key Performance Indicators (KPIs)
- List the most important metrics with current values
- Compare to expected/normal ranges where possible
- Highlight any concerning trends

## 🔍 Detailed Findings
- Summarize what you found in each analyzed table
- Explain what the data means in business terms
- Identify any patterns or anomalies

## ⚠️ Issues Detected (if any)
- List specific problems found
- Severity and potential impact

## 💡 Recommendations
- Actionable steps to improve or maintain health
- Priority order for addressing issues
- Preventive measures

GUIDELINES:
- Always explain technical findings in simple business language
- If no timeframe specified, analyze the last 24 hours
- Proactively look for problems - don't wait to be asked
- When data is limited, clearly state assumptions
- If you can't determine health status, explain what additional data would help
- Be concise but thorough
- Use emojis and formatting to make reports easy to scan

IMPORTANT: You can only READ data. You analyze and advise, but cannot make changes to the system."""


def create_agent() -> Agent:
    """
    Create and configure the AI agent with tools and LLM.
    Connects to LLM through Kong AI Gateway.
    
    Returns:
        Configured Agent instance
    """
    # Get LLM configuration from config (handles route selection)
    llm_config = config.get_llm_config()
    
    logger.info("Creating AI agent", extra={"extra_data": {
        "route": llm_config['route'],
        "model": llm_config['model'],
        "api_base": llm_config['api_base']
    }})
    
    # Configure the LLM using LiteLLM (routes through Kong AI Gateway)
    # Use 'openai/' prefix to tell LiteLLM to use OpenAI-compatible API format
    llm = LiteLlm(
        model=f"openai/{llm_config['model']}",
        api_key=llm_config["api_key"],
        api_base=llm_config["api_base"],
        extra_headers={
            "Authorization": f"Bearer {llm_config['api_key']}",
            "apikey": llm_config["api_key"]
        }
    )
    
    # Get cached metadata to include in system prompt (non-blocking)
    metadata_summary = ""
    try:
        # Trigger background refresh if needed (non-blocking)
        metadata_cache.ensure_fresh_cache()
        # Get compact summary to reduce token consumption
        metadata_summary = metadata_cache.get_metadata_summary_for_agent(compact=True)
        logger.info("Loaded metadata context for agent", extra={"extra_data": {
            "summary_length": len(metadata_summary),
            "is_refreshing": metadata_cache.is_refreshing()
        }})
    except Exception as e:
        logger.warning(f"Could not load metadata cache: {e}")
        metadata_summary = "*Use list_tables and describe_table tools to discover data.*"
    
    # Build enhanced system prompt with metadata (compact version)
    # Include actual catalog and schema from config
    catalog = config.DATABRICKS_CATALOG
    schema = config.DATABRICKS_SCHEMA
    
    enhanced_prompt = f"""{SYSTEM_PROMPT}

---

## DATABRICKS CONFIGURATION

**Catalog:** `{catalog}`
**Schema:** `{schema}`

**CRITICAL:** When writing SQL queries, ALWAYS use fully qualified table names in the format:
`{catalog}.{schema}.<table_name>`

Example: SELECT * FROM {catalog}.{schema}.your_table LIMIT 10

---

## DATA CATALOG

{metadata_summary}

Use describe_table tool for column details before writing queries.
"""
    
    # Create the agent with tools
    agent = Agent(
        name="databricks_analyst",
        model=llm,
        description="An AI assistant that queries Azure Databricks to answer data questions",
        instruction=enhanced_prompt,
        tools=all_tools
    )
    
    logger.info("AI agent created successfully", extra={"extra_data": {
        "agent_name": "databricks_analyst",
        "tools_count": len(all_tools)
    }})
    metrics.agent_initialized.inc()
    
    return agent


class AgentRunner:
    """Runner class to manage agent conversations using Google ADK."""
    
    def __init__(self):
        """Initialize the agent runner with session management."""
        logger.info("Initializing AgentRunner")
        
        with tracer.span("agent_runner_init"):
            self.agent = create_agent()
            self.session_service = InMemorySessionService()
            self.runner = Runner(
                agent=self.agent,
                app_name="system_health_analyst",
                session_service=self.session_service
            )
            self.user_id = "default_user"
            self.session_id = None
            self._session_created = False
        
        logger.info("AgentRunner initialized successfully")
    
    async def _ensure_session(self):
        """Ensure a session exists, creating one if needed."""
        if not self._session_created or self.session_id is None:
            import uuid
            self.session_id = str(uuid.uuid4())
            
            logger.debug("Creating new agent session", extra={"extra_data": {
                "session_id": self.session_id,
                "user_id": self.user_id
            }})
            
            # Create the session in the session service (async)
            await self.session_service.create_session(
                app_name="system_health_analyst",
                user_id=self.user_id,
                session_id=self.session_id
            )
            self._session_created = True
            metrics.agent_sessions_created.inc()
    
    async def chat(self, user_message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            user_message: The user's question or message
            
        Returns:
            The agent's response as a string
        """
        start_time = time.time()
        
        logger.info("Processing agent chat request", extra={"extra_data": {
            "message_length": len(user_message),
            "message_preview": user_message[:100],
            "session_id": self.session_id
        }})
        
        metrics.agent_requests.inc()
        
        # Start agent trace for detailed observability
        agent_tracer.start_trace(
            session_id=self.session_id,
            user_message=user_message
        )
        
        try:
            # Ensure session exists
            with tracer.span("ensure_session"):
                await self._ensure_session()
            
            # Create content for the message
            from google.genai import types
            content = types.Content(
                role="user",
                parts=[types.Part(text=user_message)]
            )
            
            # Record LLM request
            llm_config = config.get_llm_config()
            agent_tracer.record_llm_call(
                prompt=user_message,
                model=llm_config['model'],
                is_request=True
            )
            
            # Run the agent and collect response
            response_text = ""
            event_count = 0
            function_call_count = 0
            thinking_detected = False
            
            with tracer.span("agent_run", {"session_id": self.session_id}):
                async for event in self.runner.run_async(
                    user_id=self.user_id,
                    session_id=self.session_id,
                    new_message=content
                ):
                    event_count += 1
                    
                    # Record the ADK event for detailed tracing
                    agent_tracer.record_adk_event(
                        event=event,
                        event_name=f"ADK Event #{event_count}"
                    )
                    
                    # Extract and analyze event content
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                # Capture text responses
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                                
                                # Capture function/tool calls
                                if hasattr(part, 'function_call') and part.function_call:
                                    fc = part.function_call
                                    function_call_count += 1
                                    tool_name = getattr(fc, 'name', 'unknown')
                                    tool_args = dict(getattr(fc, 'args', {})) if hasattr(fc, 'args') else {}
                                    
                                    logger.debug(f"Agent calling tool: {tool_name}", extra={"extra_data": {
                                        "tool_name": tool_name,
                                        "tool_args_preview": str(tool_args)[:200]
                                    }})
                                    
                                    # Record tool selection thinking
                                    agent_tracer.record_event(
                                        event_type=AgentEventType.TOOL_SELECTION,
                                        name=f"Selected tool: {tool_name}",
                                        thought=f"Agent decided to use {tool_name} to help answer the query",
                                        tool_name=tool_name,
                                        tool_input=tool_args
                                    )
                                
                                # Capture function responses
                                if hasattr(part, 'function_response') and part.function_response:
                                    fr = part.function_response
                                    tool_name = getattr(fr, 'name', 'unknown')
                                    tool_response = getattr(fr, 'response', None)
                                    
                                    logger.debug(f"Tool response received: {tool_name}")
                                
                                # Check for thinking/reasoning content
                                if hasattr(part, 'thought') and part.thought:
                                    thinking_detected = True
                                    agent_tracer.record_thinking(
                                        thought=part.thought,
                                        reasoning="Agent internal reasoning"
                                    )
                    
                    # Check for model-level thinking
                    if hasattr(event, 'thinking') and event.thinking:
                        thinking_detected = True
                        agent_tracer.record_thinking(
                            thought=event.thinking,
                            reasoning="Model thinking step"
                        )
                    
                    # Capture any text directly on event
                    elif hasattr(event, 'text') and event.text:
                        response_text += event.text
            
            duration_ms = (time.time() - start_time) * 1000
            
            # Record LLM response
            agent_tracer.record_llm_call(
                completion=response_text[:1000] if response_text else None,
                model=llm_config['model'],
                duration_ms=duration_ms,
                is_request=False
            )
            
            # End the agent trace
            agent_tracer.end_trace(
                agent_response=response_text,
                status="success"
            )
            
            logger.info("Agent chat completed successfully", extra={"extra_data": {
                "response_length": len(response_text),
                "event_count": event_count,
                "function_calls": function_call_count,
                "thinking_detected": thinking_detected,
                "duration_ms": round(duration_ms, 2),
                "session_id": self.session_id
            }})
            
            metrics.agent_response_time.observe(duration_ms)
            metrics.agent_successes.inc()
            
            return response_text if response_text else "I couldn't generate a response. Please try again."
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            
            # End trace with error
            agent_tracer.end_trace(
                status="error",
                error=str(e)
            )
            
            logger.error(f"Agent chat failed: {str(e)}", exc_info=True, extra={"extra_data": {
                "error_type": type(e).__name__,
                "duration_ms": round(duration_ms, 2),
                "session_id": self.session_id
            }})
            
            metrics.agent_errors.inc(error_type=type(e).__name__)
            
            error_msg = f"Error processing your request: {str(e)}"
            return error_msg
    
    async def clear_history(self):
        """Clear the conversation history by creating a new session."""
        logger.info("Clearing agent conversation history", extra={"extra_data": {
            "previous_session_id": self.session_id
        }})
        
        self._session_created = False
        self.session_id = None
        
        metrics.agent_history_cleared.inc()


# Create a singleton agent runner
logger.info("Creating singleton AgentRunner instance")
agent_runner = AgentRunner()
