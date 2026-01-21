"""
Base Agent Module.
Provides common functionality for all agents in the multi-agent system.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio

from google.adk.agents import Agent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.models.lite_llm import LiteLlm

from config import config
from observability import get_logger, metrics, get_tracer

logger = get_logger(__name__)
tracer = get_tracer()


class AgentType(Enum):
    """Types of agents in the system."""
    ORCHESTRATOR = "orchestrator"
    DATA = "data"
    IT = "it"
    MANAGER = "manager"
    EXECUTIVE = "executive"


@dataclass
class AgentMessage:
    """Message passed between agents."""
    content: str
    sender: AgentType
    recipient: AgentType
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "sender": self.sender.value,
            "recipient": self.recipient.value,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }


@dataclass
class AgentResult:
    """Result from an agent execution."""
    success: bool
    content: str
    agent_type: AgentType
    raw_data: Optional[dict] = None
    duration_ms: float = 0
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "success": self.success,
            "content": self.content,
            "agent_type": self.agent_type.value,
            "raw_data": self.raw_data,
            "duration_ms": self.duration_ms,
            "error": self.error
        }


class BaseAgent(ABC):
    """
    Base class for all agents in the multi-agent system.
    Provides common LLM configuration and execution patterns.
    """
    
    def __init__(self, agent_type: AgentType, name: str, description: str):
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self._agent: Optional[Agent] = None
        self._runner: Optional[Runner] = None
        self._session_service = InMemorySessionService()
        self._session_id: Optional[str] = None
        self._initialized = False
        
        logger.info(f"BaseAgent initialized: {name}", extra={"extra_data": {
            "agent_type": agent_type.value,
            "name": name
        }})
    
    def _get_llm(self) -> LiteLlm:
        """Get configured LLM instance."""
        llm_config = config.get_llm_config()
        return LiteLlm(
            model=f"openai/{llm_config['model']}",
            api_key=llm_config["api_key"],
            api_base=llm_config["api_base"],
            extra_headers={
                "Authorization": f"Bearer {llm_config['api_key']}",
                "apikey": llm_config["api_key"]
            }
        )
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _get_tools(self) -> list:
        """Get the tools available to this agent. Must be implemented by subclasses."""
        pass
    
    def _initialize(self):
        """Initialize the agent with LLM and tools."""
        if self._initialized:
            return
            
        logger.info(f"Initializing agent: {self.name}")
        
        with tracer.span(f"agent_init_{self.agent_type.value}"):
            self._agent = Agent(
                name=self.name,
                model=self._get_llm(),
                description=self.description,
                instruction=self._get_system_prompt(),
                tools=self._get_tools()
            )
            
            self._runner = Runner(
                agent=self._agent,
                app_name=f"multi_agent_{self.name}",
                session_service=self._session_service
            )
            
            self._initialized = True
            
        logger.info(f"Agent initialized: {self.name}", extra={"extra_data": {
            "tools_count": len(self._get_tools())
        }})
    
    async def _ensure_session(self, user_id: str = "default"):
        """Ensure a session exists for this agent."""
        if self._session_id is None:
            import uuid
            self._session_id = str(uuid.uuid4())
            await self._session_service.create_session(
                app_name=f"multi_agent_{self.name}",
                user_id=user_id,
                session_id=self._session_id
            )
    
    async def process(self, message: str, context: dict = None) -> AgentResult:
        """
        Process a message and return the result.
        
        Args:
            message: The input message to process
            context: Optional context dictionary
            
        Returns:
            AgentResult with the response
        """
        import time
        start_time = time.time()
        
        self._initialize()
        
        logger.info(f"Agent {self.name} processing message", extra={"extra_data": {
            "message_preview": message[:100],
            "has_context": context is not None
        }})
        
        try:
            await self._ensure_session()
            
            from google.genai import types
            content = types.Content(
                role="user",
                parts=[types.Part(text=message)]
            )
            
            response_text = ""
            
            with tracer.span(f"agent_run_{self.agent_type.value}"):
                async for event in self._runner.run_async(
                    user_id="default",
                    session_id=self._session_id,
                    new_message=content
                ):
                    if hasattr(event, 'content') and event.content:
                        if hasattr(event.content, 'parts'):
                            for part in event.content.parts:
                                if hasattr(part, 'text') and part.text:
                                    response_text += part.text
                    elif hasattr(event, 'text') and event.text:
                        response_text += event.text
            
            duration_ms = (time.time() - start_time) * 1000
            
            logger.info(f"Agent {self.name} completed", extra={"extra_data": {
                "response_length": len(response_text),
                "duration_ms": round(duration_ms, 2)
            }})
            
            return AgentResult(
                success=True,
                content=response_text,
                agent_type=self.agent_type,
                duration_ms=duration_ms
            )
            
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Agent {self.name} failed: {e}", exc_info=True)
            
            return AgentResult(
                success=False,
                content="",
                agent_type=self.agent_type,
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def reset_session(self):
        """Reset the agent's session for a fresh conversation."""
        self._session_id = None
        logger.info(f"Agent {self.name} session reset")
