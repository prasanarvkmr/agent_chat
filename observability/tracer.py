"""
Tracing module for the System Health Analyst agent.
Provides request tracing and span tracking for debugging and performance analysis.
"""

import os
import json
import uuid
import time
import functools
import asyncio
from datetime import datetime
from typing import Any, Callable, Optional
from contextlib import contextmanager
from dataclasses import dataclass, field, asdict


# Create traces directory
TRACES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "traces")
os.makedirs(TRACES_DIR, exist_ok=True)


@dataclass
class Span:
    """Represents a single span in a trace."""
    span_id: str
    name: str
    trace_id: str
    parent_span_id: Optional[str] = None
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    duration_ms: Optional[float] = None
    status: str = "OK"
    attributes: dict = field(default_factory=dict)
    events: list = field(default_factory=list)
    error: Optional[str] = None
    
    def end(self, status: str = "OK", error: str = None):
        """Mark the span as complete."""
        self.end_time = time.time()
        self.duration_ms = (self.end_time - self.start_time) * 1000
        self.status = status
        self.error = error
        
    def add_event(self, name: str, attributes: dict = None):
        """Add an event to the span."""
        self.events.append({
            "name": name,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "attributes": attributes or {}
        })
        
    def set_attribute(self, key: str, value: Any):
        """Set an attribute on the span."""
        self.attributes[key] = value
        
    def to_dict(self) -> dict:
        """Convert span to dictionary."""
        return {
            "span_id": self.span_id,
            "trace_id": self.trace_id,
            "parent_span_id": self.parent_span_id,
            "name": self.name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() + "Z",
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() + "Z" if self.end_time else None,
            "duration_ms": round(self.duration_ms, 2) if self.duration_ms else None,
            "status": self.status,
            "attributes": self.attributes,
            "events": self.events,
            "error": self.error
        }


@dataclass
class Trace:
    """Represents a complete trace with multiple spans."""
    trace_id: str
    name: str
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None
    spans: list[Span] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def add_span(self, span: Span):
        """Add a span to the trace."""
        self.spans.append(span)
        
    def end(self):
        """Mark the trace as complete."""
        self.end_time = time.time()
        
    def to_dict(self) -> dict:
        """Convert trace to dictionary."""
        return {
            "trace_id": self.trace_id,
            "name": self.name,
            "start_time": datetime.fromtimestamp(self.start_time).isoformat() + "Z",
            "end_time": datetime.fromtimestamp(self.end_time).isoformat() + "Z" if self.end_time else None,
            "total_duration_ms": round((self.end_time - self.start_time) * 1000, 2) if self.end_time else None,
            "span_count": len(self.spans),
            "metadata": self.metadata,
            "spans": [span.to_dict() for span in self.spans]
        }


class Tracer:
    """
    Tracer for creating and managing traces and spans.
    """
    
    _current_trace: Optional[Trace] = None
    _current_span: Optional[Span] = None
    _span_stack: list[Span] = []
    
    def __init__(self, service_name: str = "system-health-analyst"):
        self.service_name = service_name
        self._traces: list[Trace] = []
        
    def start_trace(self, name: str, metadata: dict = None) -> Trace:
        """Start a new trace."""
        trace = Trace(
            trace_id=str(uuid.uuid4()),
            name=name,
            metadata=metadata or {}
        )
        trace.metadata["service"] = self.service_name
        self._current_trace = trace
        self._span_stack = []
        return trace
    
    def end_trace(self, save: bool = True) -> Optional[Trace]:
        """End the current trace and optionally save it."""
        if self._current_trace:
            self._current_trace.end()
            if save:
                self._save_trace(self._current_trace)
            trace = self._current_trace
            self._current_trace = None
            self._current_span = None
            self._span_stack = []
            return trace
        return None
    
    @contextmanager
    def span(self, name: str, attributes: dict = None):
        """Context manager for creating spans."""
        span = self.start_span(name, attributes)
        try:
            yield span
            span.end(status="OK")
        except Exception as e:
            span.end(status="ERROR", error=str(e))
            raise
        finally:
            self.end_span()
    
    def start_span(self, name: str, attributes: dict = None) -> Span:
        """Start a new span."""
        if not self._current_trace:
            # Auto-create trace if none exists
            self.start_trace(name)
            
        parent_span_id = self._span_stack[-1].span_id if self._span_stack else None
        
        span = Span(
            span_id=str(uuid.uuid4()),
            trace_id=self._current_trace.trace_id,
            parent_span_id=parent_span_id,
            name=name,
            attributes=attributes or {}
        )
        
        self._current_trace.add_span(span)
        self._span_stack.append(span)
        self._current_span = span
        
        return span
    
    def end_span(self, status: str = None, error: str = None):
        """End the current span."""
        if self._span_stack:
            span = self._span_stack.pop()
            if not span.end_time:  # Only end if not already ended
                span.end(status=status or span.status, error=error or span.error)
            self._current_span = self._span_stack[-1] if self._span_stack else None
            
    def get_current_span(self) -> Optional[Span]:
        """Get the current active span."""
        return self._current_span
    
    def get_current_trace(self) -> Optional[Trace]:
        """Get the current active trace."""
        return self._current_trace
    
    def _save_trace(self, trace: Trace):
        """Save trace to file."""
        timestamp = datetime.now().strftime("%Y%m%d")
        trace_file = os.path.join(TRACES_DIR, f"traces_{timestamp}.jsonl")
        
        with open(trace_file, "a") as f:
            f.write(json.dumps(trace.to_dict(), default=str) + "\n")


# Global tracer instance
_tracer = Tracer()


def get_tracer() -> Tracer:
    """Get the global tracer instance."""
    return _tracer


def trace_async(name: str = None, attributes: dict = None):
    """Decorator for tracing async functions."""
    def decorator(func: Callable):
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            tracer = get_tracer()
            
            with tracer.span(span_name, attributes):
                span = tracer.get_current_span()
                span.set_attribute("args_count", len(args))
                span.set_attribute("kwargs_keys", list(kwargs.keys()))
                
                result = await func(*args, **kwargs)
                
                # Add result info if it's a dict with success status
                if isinstance(result, dict) and "success" in result:
                    span.set_attribute("success", result["success"])
                    
                return result
        return wrapper
    return decorator


def trace_sync(name: str = None, attributes: dict = None):
    """Decorator for tracing sync functions."""
    def decorator(func: Callable):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            span_name = name or f"{func.__module__}.{func.__name__}"
            tracer = get_tracer()
            
            with tracer.span(span_name, attributes):
                span = tracer.get_current_span()
                span.set_attribute("args_count", len(args))
                span.set_attribute("kwargs_keys", list(kwargs.keys()))
                
                result = func(*args, **kwargs)
                
                if isinstance(result, dict) and "success" in result:
                    span.set_attribute("success", result["success"])
                    
                return result
        return wrapper
    return decorator
