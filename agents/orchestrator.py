"""
Orchestrator Agent Module.
Master agent that coordinates the multi-agent system.
Routes requests to appropriate agents and manages workflow.
Supports progress callbacks and cancellation.
"""

import asyncio
from typing import Optional, List, Callable, AsyncGenerator
from enum import Enum
from dataclasses import dataclass, field
import time

from .base_agent import BaseAgent, AgentType, AgentResult
from .data_agent import DataAgent, data_agent
from .persona_agents import (
    PersonaType, 
    ITAgent, 
    ManagerAgent, 
    ExecutiveAgent,
    get_persona_agent
)
from observability import get_logger, metrics, get_tracer

logger = get_logger(__name__)
tracer = get_tracer()


class ExecutionMode(Enum):
    """Agent execution modes."""
    SEQUENTIAL = "sequential"  # Run agents one after another
    PARALLEL = "parallel"      # Run persona agents in parallel


@dataclass
class ProgressUpdate:
    """Progress update during multi-agent execution."""
    stage: str  # "orchestrator", "data_agent", "persona_agent"
    status: str  # "starting", "running", "completed", "error"
    message: str
    agent_name: Optional[str] = None
    progress_pct: int = 0
    elapsed_ms: float = 0
    
    def to_dict(self) -> dict:
        return {
            "stage": self.stage,
            "status": self.status,
            "message": self.message,
            "agent_name": self.agent_name,
            "progress_pct": self.progress_pct,
            "elapsed_ms": round(self.elapsed_ms, 2)
        }


@dataclass
class OrchestratorConfig:
    """Configuration for the orchestrator."""
    default_persona: PersonaType = PersonaType.MANAGER
    execution_mode: ExecutionMode = ExecutionMode.SEQUENTIAL
    enable_all_personas: bool = False  # Generate reports for all personas
    default_time_range: str = "24h"


class CancellationToken:
    """Token to signal cancellation of an operation."""
    def __init__(self):
        self._cancelled = False
    
    def cancel(self):
        self._cancelled = True
    
    @property
    def is_cancelled(self) -> bool:
        return self._cancelled


class OrchestratorAgent:
    """
    Master Orchestrator Agent.
    Coordinates the multi-agent system by:
    1. Understanding user intent
    2. Routing to data agent for data retrieval
    3. Passing data to appropriate persona agent(s) for formatting
    4. Returning the final tailored report
    """
    
    def __init__(self, config: OrchestratorConfig = None):
        self.config = config or OrchestratorConfig()
        self._data_agent = data_agent
        self._persona_agents: dict[PersonaType, BaseAgent] = {}
        
        logger.info("Orchestrator initialized", extra={"extra_data": {
            "default_persona": self.config.default_persona.value,
            "execution_mode": self.config.execution_mode.value
        }})
    
    def _get_persona_agent(self, persona: PersonaType) -> BaseAgent:
        """Get or create a persona agent."""
        if persona not in self._persona_agents:
            self._persona_agents[persona] = get_persona_agent(persona)
        return self._persona_agents[persona]
    
    def _detect_persona_from_query(self, query: str) -> PersonaType:
        """
        Detect the appropriate persona based on query content.
        
        Args:
            query: User's query text
            
        Returns:
            Detected PersonaType
        """
        query_lower = query.lower()
        
        # IT persona indicators
        it_keywords = [
            "error", "log", "trace", "debug", "cpu", "memory", "disk",
            "latency", "timeout", "exception", "stack", "query", "sql",
            "technical", "root cause", "troubleshoot", "config", "server"
        ]
        
        # Executive persona indicators
        exec_keywords = [
            "executive", "summary", "brief", "ceo", "cto", "leadership",
            "board", "strategic", "risk", "bottom line", "quick", "overview"
        ]
        
        # Manager persona indicators (default if others don't match strongly)
        manager_keywords = [
            "team", "operational", "impact", "trend", "week", "month",
            "compare", "progress", "status update", "report"
        ]
        
        # Score each persona
        it_score = sum(1 for kw in it_keywords if kw in query_lower)
        exec_score = sum(1 for kw in exec_keywords if kw in query_lower)
        manager_score = sum(1 for kw in manager_keywords if kw in query_lower)
        
        # Determine persona based on scores
        if exec_score >= 2 or ("executive" in query_lower or "brief" in query_lower):
            return PersonaType.EXECUTIVE
        elif it_score >= 2 or any(kw in query_lower for kw in ["technical", "debug", "error log"]):
            return PersonaType.IT
        elif manager_score >= 1:
            return PersonaType.MANAGER
        else:
            return self.config.default_persona
    
    async def process(
        self,
        query: str,
        persona: PersonaType = None,
        time_range: str = None,
        execution_mode: ExecutionMode = None
    ) -> AgentResult:
        """
        Process a user query through the multi-agent pipeline.
        
        Args:
            query: User's natural language query
            persona: Target persona (auto-detected if not specified)
            time_range: Time range for data (default: 24h)
            execution_mode: Sequential or parallel execution
            
        Returns:
            AgentResult with the final formatted report
        """
        import time
        start_time = time.time()
        
        # Determine execution parameters
        target_persona = persona or self._detect_persona_from_query(query)
        data_time_range = time_range or self.config.default_time_range
        mode = execution_mode or self.config.execution_mode
        
        logger.info("Orchestrator processing query", extra={"extra_data": {
            "query_preview": query[:100],
            "persona": target_persona.value,
            "time_range": data_time_range,
            "mode": mode.value
        }})
        
        metrics.tool_calls.inc(tool="orchestrator_process")
        
        try:
            with tracer.span("orchestrator_pipeline"):
                # Step 1: Get data from data agent
                logger.info("Step 1: Fetching data via Data Agent")
                
                with tracer.span("data_agent_fetch"):
                    data_result = await self._data_agent.process(query)
                
                if not data_result.success:
                    logger.error(f"Data agent failed: {data_result.error}")
                    # Return user-friendly error with agent's formatted message
                    error_content = data_result.content if data_result.content else f"Error retrieving data: {data_result.error}"
                    return AgentResult(
                        success=False,
                        content=error_content,
                        agent_type=AgentType.ORCHESTRATOR,
                        duration_ms=(time.time() - start_time) * 1000,
                        error=data_result.error
                    )
                
                raw_data = data_result.content
                logger.info(f"Data agent returned {len(raw_data)} chars")
                
                # Step 2: Transform data through persona agent(s)
                context = {
                    "user_query": query,
                    "time_range": data_time_range,
                    "raw_data_length": len(raw_data)
                }
                
                if self.config.enable_all_personas and mode == ExecutionMode.PARALLEL:
                    # Generate reports for all personas in parallel
                    logger.info("Step 2: Generating reports for all personas (parallel)")
                    final_result = await self._process_all_personas_parallel(raw_data, context)
                else:
                    # Generate report for single persona
                    logger.info(f"Step 2: Generating report for {target_persona.value}")
                    persona_agent = self._get_persona_agent(target_persona)
                    
                    with tracer.span(f"persona_agent_{target_persona.value}"):
                        final_result = await persona_agent.transform_report(raw_data, context)
                
                # Calculate total duration
                duration_ms = (time.time() - start_time) * 1000
                
                logger.info("Orchestrator completed", extra={"extra_data": {
                    "success": final_result.success,
                    "total_duration_ms": round(duration_ms, 2),
                    "persona": target_persona.value
                }})
                
                return AgentResult(
                    success=final_result.success,
                    content=final_result.content,
                    agent_type=AgentType.ORCHESTRATOR,
                    raw_data={"persona": target_persona.value, "data_agent_output": raw_data[:500]},
                    duration_ms=duration_ms,
                    error=final_result.error
                )
                
        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            logger.error(f"Orchestrator failed: {e}", exc_info=True)
            
            return AgentResult(
                success=False,
                content=f"Error processing request: {str(e)}",
                agent_type=AgentType.ORCHESTRATOR,
                duration_ms=duration_ms,
                error=str(e)
            )
    
    async def _process_all_personas_parallel(
        self, 
        raw_data: str, 
        context: dict
    ) -> AgentResult:
        """
        Process data through all persona agents in parallel.
        
        Args:
            raw_data: Data from data agent
            context: Context dictionary
            
        Returns:
            Combined AgentResult with all persona reports
        """
        personas = [PersonaType.IT, PersonaType.MANAGER, PersonaType.EXECUTIVE]
        
        async def process_persona(persona: PersonaType):
            agent = self._get_persona_agent(persona)
            return persona, await agent.transform_report(raw_data, context)
        
        # Run all persona agents in parallel
        tasks = [process_persona(p) for p in personas]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Combine results
        combined_content = []
        errors = []
        
        for result in results:
            if isinstance(result, Exception):
                errors.append(str(result))
                continue
            
            persona, agent_result = result
            if agent_result.success:
                combined_content.append(f"## {persona.value.upper()} VIEW\n\n{agent_result.content}")
            else:
                errors.append(f"{persona.value}: {agent_result.error}")
        
        final_content = "\n\n---\n\n".join(combined_content)
        
        return AgentResult(
            success=len(combined_content) > 0,
            content=final_content,
            agent_type=AgentType.ORCHESTRATOR,
            error="; ".join(errors) if errors else None
        )
    
    async def process_sequential(
        self,
        query: str,
        personas: List[PersonaType]
    ) -> dict[PersonaType, AgentResult]:
        """
        Process query sequentially through specified personas.
        
        Args:
            query: User's query
            personas: List of personas to generate reports for
            
        Returns:
            Dictionary mapping personas to their results
        """
        # First get data
        data_result = await self._data_agent.process(query)
        
        if not data_result.success:
            return {p: AgentResult(
                success=False,
                content="",
                agent_type=AgentType.ORCHESTRATOR,
                error=data_result.error
            ) for p in personas}
        
        # Then process through each persona sequentially
        results = {}
        context = {"user_query": query}
        
        for persona in personas:
            agent = self._get_persona_agent(persona)
            results[persona] = await agent.transform_report(data_result.content, context)
        
        return results
    
    def set_default_persona(self, persona: PersonaType):
        """Set the default persona for queries."""
        self.config.default_persona = persona
        logger.info(f"Default persona set to: {persona.value}")
    
    def set_execution_mode(self, mode: ExecutionMode):
        """Set the execution mode."""
        self.config.execution_mode = mode
        logger.info(f"Execution mode set to: {mode.value}")
    
    async def process_with_progress(
        self,
        query: str,
        persona: PersonaType = None,
        time_range: str = None,
        cancellation_token: CancellationToken = None
    ) -> AsyncGenerator[ProgressUpdate | AgentResult, None]:
        """
        Process a query with streaming progress updates.
        Yields ProgressUpdate objects during execution, then final AgentResult.
        
        Args:
            query: User's natural language query
            persona: Target persona (auto-detected if not specified)
            time_range: Time range for data
            cancellation_token: Token to cancel the operation
            
        Yields:
            ProgressUpdate objects, then final AgentResult
        """
        start_time = time.time()
        cancel_token = cancellation_token or CancellationToken()
        
        def elapsed() -> float:
            return (time.time() - start_time) * 1000
        
        # Determine execution parameters
        target_persona = persona or self._detect_persona_from_query(query)
        data_time_range = time_range or self.config.default_time_range
        
        try:
            # Stage 1: Orchestrator starting
            yield ProgressUpdate(
                stage="orchestrator",
                status="starting",
                message="Analyzing your query...",
                agent_name="Orchestrator",
                progress_pct=5,
                elapsed_ms=elapsed()
            )
            
            if cancel_token.is_cancelled:
                yield ProgressUpdate(stage="orchestrator", status="cancelled", message="Request cancelled", elapsed_ms=elapsed())
                return
            
            await asyncio.sleep(0.1)  # Small delay for UI to update
            
            yield ProgressUpdate(
                stage="orchestrator",
                status="running",
                message=f"Detected persona: {target_persona.value.title()}",
                agent_name="Orchestrator",
                progress_pct=10,
                elapsed_ms=elapsed()
            )
            
            # Stage 2: Data Agent
            yield ProgressUpdate(
                stage="data_agent",
                status="starting",
                message="Initializing Data Agent...",
                agent_name="Data Agent",
                progress_pct=15,
                elapsed_ms=elapsed()
            )
            
            if cancel_token.is_cancelled:
                yield ProgressUpdate(stage="data_agent", status="cancelled", message="Request cancelled", elapsed_ms=elapsed())
                return
            
            yield ProgressUpdate(
                stage="data_agent",
                status="running",
                message="Querying Databricks for system health data...",
                agent_name="Data Agent",
                progress_pct=25,
                elapsed_ms=elapsed()
            )
            
            # Actually run data agent
            data_result = await self._data_agent.process(query)
            
            if cancel_token.is_cancelled:
                yield ProgressUpdate(stage="data_agent", status="cancelled", message="Request cancelled", elapsed_ms=elapsed())
                return
            
            if not data_result.success:
                yield ProgressUpdate(
                    stage="data_agent",
                    status="error",
                    message=f"Data retrieval failed: {data_result.error}",
                    agent_name="Data Agent",
                    progress_pct=30,
                    elapsed_ms=elapsed()
                )
                yield AgentResult(
                    success=False,
                    content=f"Error retrieving data: {data_result.error}",
                    agent_type=AgentType.ORCHESTRATOR,
                    duration_ms=elapsed(),
                    error=data_result.error
                )
                return
            
            yield ProgressUpdate(
                stage="data_agent",
                status="completed",
                message=f"Retrieved {len(data_result.content)} characters of data",
                agent_name="Data Agent",
                progress_pct=50,
                elapsed_ms=elapsed()
            )
            
            raw_data = data_result.content
            
            # Stage 3: Persona Agent
            persona_name = f"{target_persona.value.title()} Agent"
            
            yield ProgressUpdate(
                stage="persona_agent",
                status="starting",
                message=f"Initializing {persona_name}...",
                agent_name=persona_name,
                progress_pct=55,
                elapsed_ms=elapsed()
            )
            
            if cancel_token.is_cancelled:
                yield ProgressUpdate(stage="persona_agent", status="cancelled", message="Request cancelled", elapsed_ms=elapsed())
                return
            
            yield ProgressUpdate(
                stage="persona_agent",
                status="running",
                message=f"Formatting report for {target_persona.value} audience...",
                agent_name=persona_name,
                progress_pct=65,
                elapsed_ms=elapsed()
            )
            
            # Run persona agent
            persona_agent = self._get_persona_agent(target_persona)
            context = {
                "user_query": query,
                "time_range": data_time_range,
                "raw_data_length": len(raw_data)
            }
            
            final_result = await persona_agent.transform_report(raw_data, context)
            
            if cancel_token.is_cancelled:
                yield ProgressUpdate(stage="persona_agent", status="cancelled", message="Request cancelled", elapsed_ms=elapsed())
                return
            
            yield ProgressUpdate(
                stage="persona_agent",
                status="completed",
                message="Report formatting complete",
                agent_name=persona_name,
                progress_pct=90,
                elapsed_ms=elapsed()
            )
            
            # Final stage
            yield ProgressUpdate(
                stage="orchestrator",
                status="completed",
                message="Analysis complete!",
                agent_name="Orchestrator",
                progress_pct=100,
                elapsed_ms=elapsed()
            )
            
            # Yield final result
            yield AgentResult(
                success=final_result.success,
                content=final_result.content,
                agent_type=AgentType.ORCHESTRATOR,
                raw_data={"persona": target_persona.value, "data_agent_output": raw_data[:500]},
                duration_ms=elapsed(),
                error=final_result.error
            )
            
        except Exception as e:
            logger.error(f"Orchestrator failed: {e}", exc_info=True)
            yield ProgressUpdate(
                stage="orchestrator",
                status="error",
                message=f"Error: {str(e)}",
                elapsed_ms=elapsed()
            )
            yield AgentResult(
                success=False,
                content=f"Error processing request: {str(e)}",
                agent_type=AgentType.ORCHESTRATOR,
                duration_ms=elapsed(),
                error=str(e)
            )


# Active cancellation tokens for ongoing requests
_active_requests: dict[str, CancellationToken] = {}


def create_cancellation_token(request_id: str) -> CancellationToken:
    """Create and register a cancellation token for a request."""
    token = CancellationToken()
    _active_requests[request_id] = token
    return token


def cancel_request(request_id: str) -> bool:
    """Cancel an active request by ID."""
    if request_id in _active_requests:
        _active_requests[request_id].cancel()
        del _active_requests[request_id]
        return True
    return False


def cleanup_request(request_id: str):
    """Clean up a completed request."""
    _active_requests.pop(request_id, None)


# Singleton orchestrator instance
orchestrator = OrchestratorAgent()
