"""
Agent-specific tracing module for observing AI agent behavior.
Captures thinking, tool calls, events, and LLM interactions.
"""

import os
import json
import uuid
import time
from datetime import datetime
from typing import Any, Optional, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum
from contextlib import contextmanager


# Create agent traces directory
AGENT_TRACES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "agent_traces")
os.makedirs(AGENT_TRACES_DIR, exist_ok=True)


class AgentEventType(str, Enum):
    """Types of agent events."""
    # Lifecycle events
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    
    # Thinking/Reasoning events
    THINKING_START = "thinking_start"
    THINKING_STEP = "thinking_step"
    THINKING_END = "thinking_end"
    
    # LLM events
    LLM_REQUEST = "llm_request"
    LLM_RESPONSE = "llm_response"
    LLM_STREAM_CHUNK = "llm_stream_chunk"
    LLM_ERROR = "llm_error"
    
    # Tool events
    TOOL_SELECTION = "tool_selection"
    TOOL_CALL_START = "tool_call_start"
    TOOL_CALL_END = "tool_call_end"
    TOOL_ERROR = "tool_error"
    
    # Agent events from ADK
    ADK_EVENT = "adk_event"
    AGENT_RESPONSE = "agent_response"
    FUNCTION_CALL = "function_call"
    FUNCTION_RESPONSE = "function_response"
    
    # State events
    STATE_CHANGE = "state_change"
    CONTEXT_UPDATE = "context_update"
    
    # Error events
    ERROR = "error"
    WARNING = "warning"


@dataclass
class AgentEvent:
    """Represents a single agent event."""
    event_id: str
    event_type: AgentEventType
    timestamp: str
    trace_id: str
    session_id: Optional[str] = None
    
    # Event details
    name: Optional[str] = None
    data: dict = field(default_factory=dict)
    
    # Timing
    duration_ms: Optional[float] = None
    
    # For thinking events
    thought: Optional[str] = None
    reasoning: Optional[str] = None
    
    # For tool events
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[Any] = None
    
    # For LLM events
    prompt: Optional[str] = None
    completion: Optional[str] = None
    model: Optional[str] = None
    tokens_used: Optional[int] = None
    
    # Parent event for nesting
    parent_event_id: Optional[str] = None
    
    # Status
    status: str = "success"
    error: Optional[str] = None
    
    def to_dict(self) -> dict:
        """Convert event to dictionary."""
        result = {
            "event_id": self.event_id,
            "event_type": self.event_type.value if isinstance(self.event_type, AgentEventType) else self.event_type,
            "timestamp": self.timestamp,
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "status": self.status
        }
        
        # Add optional fields only if they have values
        if self.name:
            result["name"] = self.name
        if self.data:
            result["data"] = self.data
        if self.duration_ms is not None:
            result["duration_ms"] = round(self.duration_ms, 2)
        if self.thought:
            result["thought"] = self.thought
        if self.reasoning:
            result["reasoning"] = self.reasoning
        if self.tool_name:
            result["tool_name"] = self.tool_name
        if self.tool_input is not None:
            result["tool_input"] = self._safe_serialize(self.tool_input)
        if self.tool_output is not None:
            result["tool_output"] = self._safe_serialize(self.tool_output)
        if self.prompt:
            result["prompt"] = self.prompt[:1000] if len(self.prompt) > 1000 else self.prompt
        if self.completion:
            result["completion"] = self.completion
        if self.model:
            result["model"] = self.model
        if self.tokens_used:
            result["tokens_used"] = self.tokens_used
        if self.parent_event_id:
            result["parent_event_id"] = self.parent_event_id
        if self.error:
            result["error"] = self.error
            
        return result
    
    def _safe_serialize(self, obj: Any) -> Any:
        """Safely serialize an object for JSON."""
        try:
            if isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            elif isinstance(obj, dict):
                return {k: self._safe_serialize(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [self._safe_serialize(item) for item in obj]
            else:
                return str(obj)
        except Exception:
            return f"<unserializable: {type(obj).__name__}>"


@dataclass
class AgentTrace:
    """Represents a complete agent execution trace."""
    trace_id: str
    session_id: Optional[str] = None
    user_message: Optional[str] = None
    agent_response: Optional[str] = None
    
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    
    events: list[AgentEvent] = field(default_factory=list)
    
    # Summary stats
    tool_calls_count: int = 0
    llm_calls_count: int = 0
    thinking_steps_count: int = 0
    total_tokens: int = 0
    
    # Status
    status: str = "in_progress"
    error: Optional[str] = None
    
    def add_event(self, event: AgentEvent):
        """Add an event to the trace."""
        self.events.append(event)
        
        # Update counters
        if event.event_type == AgentEventType.TOOL_CALL_START:
            self.tool_calls_count += 1
        elif event.event_type == AgentEventType.LLM_REQUEST:
            self.llm_calls_count += 1
        elif event.event_type == AgentEventType.THINKING_STEP:
            self.thinking_steps_count += 1
        if event.tokens_used:
            self.total_tokens += event.tokens_used
    
    def end(self, status: str = "success", error: str = None):
        """Mark the trace as complete."""
        self.end_time = time.time()
        self.status = status
        self.error = error
    
    def to_dict(self) -> dict:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "session_id": self.session_id,
            "user_message": self.user_message[:500] if self.user_message and len(self.user_message) > 500 else self.user_message,
            "agent_response_preview": self.agent_response[:500] if self.agent_response and len(self.agent_response) > 500 else self.agent_response,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() + "Z",
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() + "Z" if self.end_time else None,
            "duration_ms": round((self.end_time - self.start_time) * 1000, 2) if self.end_time else None,
            "status": self.status,
            "error": self.error,
            "summary": {
                "event_count": len(self.events),
                "tool_calls": self.tool_calls_count,
                "llm_calls": self.llm_calls_count,
                "thinking_steps": self.thinking_steps_count,
                "total_tokens": self.total_tokens
            },
            "events": [e.to_dict() for e in self.events]
        }
    
    def get_thinking_summary(self) -> list[dict]:
        """Get a summary of the agent's thinking process."""
        thinking_events = [
            e for e in self.events 
            if e.event_type in [
                AgentEventType.THINKING_START,
                AgentEventType.THINKING_STEP,
                AgentEventType.THINKING_END,
                AgentEventType.TOOL_SELECTION
            ]
        ]
        return [e.to_dict() for e in thinking_events]
    
    def get_tool_calls_summary(self) -> list[dict]:
        """Get a summary of all tool calls."""
        tool_events = [
            e for e in self.events
            if e.event_type in [
                AgentEventType.TOOL_CALL_START,
                AgentEventType.TOOL_CALL_END,
                AgentEventType.TOOL_ERROR,
                AgentEventType.FUNCTION_CALL,
                AgentEventType.FUNCTION_RESPONSE
            ]
        ]
        return [e.to_dict() for e in tool_events]


class AgentTracer:
    """
    Tracer specifically designed for observing AI agent behavior.
    Captures thinking, reasoning, tool calls, and LLM interactions.
    """
    
    def __init__(self, service_name: str = "system-health-analyst"):
        self.service_name = service_name
        self._current_trace: Optional[AgentTrace] = None
        self._event_stack: list[AgentEvent] = []
        self._traces_history: list[AgentTrace] = []
        self._max_history = 100
        
    def start_trace(
        self, 
        session_id: str = None, 
        user_message: str = None
    ) -> AgentTrace:
        """Start a new agent trace."""
        trace = AgentTrace(
            trace_id=str(uuid.uuid4()),
            session_id=session_id,
            user_message=user_message
        )
        self._current_trace = trace
        self._event_stack = []
        
        # Add session start event
        self.record_event(
            event_type=AgentEventType.SESSION_START,
            name="Agent trace started",
            data={
                "service": self.service_name,
                "session_id": session_id,
                "message_length": len(user_message) if user_message else 0
            }
        )
        
        return trace
    
    def end_trace(
        self, 
        agent_response: str = None, 
        status: str = "success", 
        error: str = None,
        save: bool = True
    ) -> Optional[AgentTrace]:
        """End the current trace."""
        if not self._current_trace:
            return None
            
        self._current_trace.agent_response = agent_response
        self._current_trace.end(status=status, error=error)
        
        # Add session end event
        self.record_event(
            event_type=AgentEventType.SESSION_END,
            name="Agent trace ended",
            data={
                "status": status,
                "response_length": len(agent_response) if agent_response else 0,
                "total_events": len(self._current_trace.events)
            }
        )
        
        if save:
            self._save_trace(self._current_trace)
            
        # Store in history
        self._traces_history.append(self._current_trace)
        if len(self._traces_history) > self._max_history:
            self._traces_history.pop(0)
            
        trace = self._current_trace
        self._current_trace = None
        self._event_stack = []
        
        return trace
    
    def record_event(
        self,
        event_type: AgentEventType,
        name: str = None,
        data: dict = None,
        thought: str = None,
        reasoning: str = None,
        tool_name: str = None,
        tool_input: dict = None,
        tool_output: Any = None,
        prompt: str = None,
        completion: str = None,
        model: str = None,
        tokens_used: int = None,
        duration_ms: float = None,
        status: str = "success",
        error: str = None
    ) -> AgentEvent:
        """Record an agent event."""
        if not self._current_trace:
            # Auto-start trace if none exists
            self.start_trace()
            
        parent_event_id = self._event_stack[-1].event_id if self._event_stack else None
        
        event = AgentEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=datetime.utcnow().isoformat() + "Z",
            trace_id=self._current_trace.trace_id,
            session_id=self._current_trace.session_id,
            name=name,
            data=data or {},
            thought=thought,
            reasoning=reasoning,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            prompt=prompt,
            completion=completion,
            model=model,
            tokens_used=tokens_used,
            duration_ms=duration_ms,
            parent_event_id=parent_event_id,
            status=status,
            error=error
        )
        
        self._current_trace.add_event(event)
        return event
    
    def record_thinking(self, thought: str, reasoning: str = None, step: int = None):
        """Record an agent thinking/reasoning step."""
        return self.record_event(
            event_type=AgentEventType.THINKING_STEP,
            name=f"Thinking step {step}" if step else "Thinking",
            thought=thought,
            reasoning=reasoning,
            data={"step": step} if step else {}
        )
    
    def record_tool_call(
        self,
        tool_name: str,
        tool_input: dict,
        tool_output: Any = None,
        duration_ms: float = None,
        success: bool = True,
        error: str = None
    ):
        """Record a tool call event."""
        return self.record_event(
            event_type=AgentEventType.TOOL_CALL_END if tool_output is not None else AgentEventType.TOOL_CALL_START,
            name=f"Tool: {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            duration_ms=duration_ms,
            status="success" if success else "error",
            error=error
        )
    
    def record_llm_call(
        self,
        prompt: str = None,
        completion: str = None,
        model: str = None,
        tokens_used: int = None,
        duration_ms: float = None,
        is_request: bool = True
    ):
        """Record an LLM call event."""
        return self.record_event(
            event_type=AgentEventType.LLM_REQUEST if is_request else AgentEventType.LLM_RESPONSE,
            name=f"LLM {'Request' if is_request else 'Response'}",
            prompt=prompt,
            completion=completion,
            model=model,
            tokens_used=tokens_used,
            duration_ms=duration_ms
        )
    
    def record_adk_event(self, event: Any, event_name: str = None):
        """
        Record a Google ADK event.
        Parses the ADK event structure to extract relevant information.
        """
        event_data = {}
        event_type = AgentEventType.ADK_EVENT
        tool_name = None
        tool_input = None
        tool_output = None
        thought = None
        completion = None
        
        # Extract event type name
        event_class = type(event).__name__
        event_data["event_class"] = event_class
        
        # Try to extract content from various ADK event structures
        if hasattr(event, 'content') and event.content:
            content = event.content
            
            # Check for text parts (agent responses)
            if hasattr(content, 'parts'):
                text_parts = []
                for part in content.parts:
                    if hasattr(part, 'text') and part.text:
                        text_parts.append(part.text)
                    # Check for function calls
                    if hasattr(part, 'function_call') and part.function_call:
                        fc = part.function_call
                        event_type = AgentEventType.FUNCTION_CALL
                        tool_name = getattr(fc, 'name', None)
                        tool_input = dict(getattr(fc, 'args', {})) if hasattr(fc, 'args') else None
                        event_data["function_name"] = tool_name
                    # Check for function responses
                    if hasattr(part, 'function_response') and part.function_response:
                        fr = part.function_response
                        event_type = AgentEventType.FUNCTION_RESPONSE
                        tool_name = getattr(fr, 'name', None)
                        tool_output = getattr(fr, 'response', None)
                        event_data["function_name"] = tool_name
                        
                if text_parts:
                    completion = "".join(text_parts)
                    event_data["text_content"] = completion[:500]
                    if event_type == AgentEventType.ADK_EVENT:
                        event_type = AgentEventType.AGENT_RESPONSE
            
            # Check for role
            if hasattr(content, 'role'):
                event_data["role"] = content.role
        
        # Check for model output
        if hasattr(event, 'model') and event.model:
            event_data["model"] = str(event.model)
            
        # Check for thinking/reasoning in model response
        if hasattr(event, 'thinking') and event.thinking:
            thought = event.thinking
            event_type = AgentEventType.THINKING_STEP
            
        # Check for usage stats
        if hasattr(event, 'usage_metadata'):
            usage = event.usage_metadata
            if hasattr(usage, 'total_token_count'):
                event_data["tokens"] = usage.total_token_count
        
        return self.record_event(
            event_type=event_type,
            name=event_name or event_class,
            data=event_data,
            tool_name=tool_name,
            tool_input=tool_input,
            tool_output=tool_output,
            thought=thought,
            completion=completion
        )
    
    @contextmanager
    def trace_context(
        self, 
        session_id: str = None, 
        user_message: str = None
    ):
        """Context manager for tracing an agent execution."""
        self.start_trace(session_id=session_id, user_message=user_message)
        try:
            yield self._current_trace
        except Exception as e:
            self.end_trace(status="error", error=str(e))
            raise
    
    @contextmanager
    def thinking_context(self, description: str = "Agent reasoning"):
        """Context manager for tracking a thinking/reasoning phase."""
        start_time = time.time()
        
        self.record_event(
            event_type=AgentEventType.THINKING_START,
            name=description
        )
        
        start_event = self._current_trace.events[-1] if self._current_trace else None
        if start_event:
            self._event_stack.append(start_event)
        
        try:
            yield
        finally:
            duration_ms = (time.time() - start_time) * 1000
            if start_event:
                self._event_stack.pop()
            
            self.record_event(
                event_type=AgentEventType.THINKING_END,
                name=f"{description} complete",
                duration_ms=duration_ms
            )
    
    @contextmanager
    def tool_context(self, tool_name: str, tool_input: dict = None):
        """Context manager for tracking a tool call."""
        start_time = time.time()
        
        self.record_event(
            event_type=AgentEventType.TOOL_CALL_START,
            name=f"Calling {tool_name}",
            tool_name=tool_name,
            tool_input=tool_input
        )
        
        start_event = self._current_trace.events[-1] if self._current_trace else None
        if start_event:
            self._event_stack.append(start_event)
        
        result = {"output": None, "error": None}
        
        try:
            yield result
        except Exception as e:
            result["error"] = str(e)
            raise
        finally:
            duration_ms = (time.time() - start_time) * 1000
            if start_event:
                self._event_stack.pop()
            
            self.record_event(
                event_type=AgentEventType.TOOL_CALL_END if not result["error"] else AgentEventType.TOOL_ERROR,
                name=f"{tool_name} {'complete' if not result['error'] else 'failed'}",
                tool_name=tool_name,
                tool_input=tool_input,
                tool_output=result["output"],
                duration_ms=duration_ms,
                status="success" if not result["error"] else "error",
                error=result["error"]
            )
    
    def get_current_trace(self) -> Optional[AgentTrace]:
        """Get the current active trace."""
        return self._current_trace
    
    def get_trace_history(self) -> list[dict]:
        """Get recent trace history."""
        return [t.to_dict() for t in self._traces_history]
    
    def get_trace_summary(self, trace_id: str = None) -> Optional[dict]:
        """Get a summary of a specific trace or the current trace."""
        trace = None
        
        if trace_id:
            for t in self._traces_history:
                if t.trace_id == trace_id:
                    trace = t
                    break
        else:
            trace = self._current_trace
            
        if not trace:
            return None
            
        return {
            "trace_id": trace.trace_id,
            "status": trace.status,
            "duration_ms": round((trace.end_time - trace.start_time) * 1000, 2) if trace.end_time else None,
            "thinking_steps": trace.thinking_steps_count,
            "tool_calls": trace.tool_calls_count,
            "llm_calls": trace.llm_calls_count,
            "total_events": len(trace.events),
            "thinking_flow": trace.get_thinking_summary(),
            "tool_calls_detail": trace.get_tool_calls_summary()
        }
    
    def _save_trace(self, trace: AgentTrace):
        """Save trace to file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        trace_file = os.path.join(AGENT_TRACES_DIR, f"agent_traces_{timestamp}.jsonl")
        
        with open(trace_file, "a") as f:
            f.write(json.dumps(trace.to_dict(), default=str) + "\n")


# Global agent tracer instance
_agent_tracer = AgentTracer()


def get_agent_tracer() -> AgentTracer:
    """Get the global agent tracer instance."""
    return _agent_tracer
