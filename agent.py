"""
AI Agent Module.
Defines the main agent that handles user queries using Google ADK.
Supports multiple LLM providers through Kong AI Gateway.
"""

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm
from tools import all_tools
from config import config


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
    
    print(f"[LLM Config] Route: {llm_config['route']}, Model: {llm_config['model']}")
    print(f"[LLM Config] API Base: {llm_config['api_base']}")
    
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
    
    # Create the agent with tools
    agent = Agent(
        name="databricks_analyst",
        model=llm,
        description="An AI assistant that queries Azure Databricks to answer data questions",
        instruction=SYSTEM_PROMPT,
        tools=all_tools
    )
    
    return agent


class AgentRunner:
    """Runner class to manage agent conversations using Google ADK."""
    
    def __init__(self):
        """Initialize the agent runner with session management."""
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
    
    async def _ensure_session(self):
        """Ensure a session exists, creating one if needed."""
        if not self._session_created or self.session_id is None:
            import uuid
            self.session_id = str(uuid.uuid4())
            # Create the session in the session service (async)
            await self.session_service.create_session(
                app_name="system_health_analyst",
                user_id=self.user_id,
                session_id=self.session_id
            )
            self._session_created = True
    
    async def chat(self, user_message: str) -> str:
        """
        Send a message to the agent and get a response.
        
        Args:
            user_message: The user's question or message
            
        Returns:
            The agent's response as a string
        """
        try:
            # Ensure session exists
            await self._ensure_session()
            
            # Create content for the message
            from google.genai import types
            content = types.Content(
                role="user",
                parts=[types.Part(text=user_message)]
            )
            
            # Run the agent and collect response
            response_text = ""
            async for event in self.runner.run_async(
                user_id=self.user_id,
                session_id=self.session_id,
                new_message=content
            ):
                # Extract text from agent response events
                if hasattr(event, 'content') and event.content:
                    if hasattr(event.content, 'parts'):
                        for part in event.content.parts:
                            if hasattr(part, 'text') and part.text:
                                response_text += part.text
                elif hasattr(event, 'text'):
                    response_text += event.text
            
            return response_text if response_text else "I couldn't generate a response. Please try again."
            
        except Exception as e:
            error_msg = f"Error processing your request: {str(e)}"
            return error_msg
    
    async def clear_history(self):
        """Clear the conversation history by creating a new session."""
        self._session_created = False
        self.session_id = None


# Create a singleton agent runner
agent_runner = AgentRunner()
