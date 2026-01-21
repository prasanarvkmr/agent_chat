"""
Persona Agents Module.
Specialized agents for different user personas: IT, Manager, Executive.
Each agent tailors reports and communication style for their target audience.
"""

from typing import Optional
from enum import Enum
from .base_agent import BaseAgent, AgentType, AgentResult
from observability import get_logger

logger = get_logger(__name__)


class PersonaType(Enum):
    """User persona types."""
    IT = "it"
    MANAGER = "manager"
    EXECUTIVE = "executive"


# ============================================================================
# IT PERSONA AGENT
# ============================================================================

IT_AGENT_PROMPT = """You are an IT Technical Analyst Agent that transforms system health data into detailed technical reports for IT professionals.

YOUR AUDIENCE: IT Engineers, DevOps, System Administrators

COMMUNICATION STYLE:
- Technical and precise language
- Include specific metrics, error codes, and system details
- Reference logs, traces, and technical indicators
- Provide actionable troubleshooting steps

REPORT FORMAT:

## 🔧 Technical Health Report

### System Status
[Overall status with technical justification]

### Performance Metrics
| Metric | Current | Threshold | Status |
|--------|---------|-----------|--------|
[Detailed metric tables with exact values]

### Error Analysis
- Error codes and counts
- Stack traces or log snippets if relevant
- Root cause indicators

### Resource Utilization
- CPU, Memory, Disk, Network stats
- Connection pool status
- Queue depths

### Technical Recommendations
1. Specific actions with commands/steps
2. Configuration changes needed
3. Monitoring adjustments

### Alert Thresholds
- Current thresholds vs recommended
- Suggested alert rules

GUIDELINES:
- Be precise with numbers (no rounding unless appropriate)
- Include timestamps and time ranges
- Reference specific tables, columns, and queries used
- Suggest specific technical actions
"""


class ITAgent(BaseAgent):
    """
    IT Persona Agent for technical reports.
    Provides detailed technical analysis for IT professionals.
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.IT,
            name="it_analyst",
            description="Technical analyst providing detailed IT reports"
        )
    
    def _get_system_prompt(self) -> str:
        return IT_AGENT_PROMPT
    
    def _get_tools(self) -> list:
        # IT agent doesn't need data tools - it transforms data from data agent
        return []
    
    async def transform_report(self, raw_data: str, context: dict = None) -> AgentResult:
        """
        Transform raw data into an IT-focused technical report.
        
        Args:
            raw_data: Raw data from the data agent
            context: Additional context (user query, time range, etc.)
            
        Returns:
            AgentResult with formatted IT report
        """
        query = context.get("user_query", "system health") if context else "system health"
        
        prompt = f"""Transform the following data into a detailed technical report for IT professionals.

USER QUERY: {query}

RAW DATA AND ANALYSIS:
{raw_data}

Create a comprehensive technical report following the format in your instructions.
Include all relevant metrics, specific values, and actionable technical recommendations."""

        return await self.process(prompt, context)


# ============================================================================
# MANAGER PERSONA AGENT
# ============================================================================

MANAGER_AGENT_PROMPT = """You are a Business Manager Analyst Agent that transforms system health data into operational reports for middle management.

YOUR AUDIENCE: Team Managers, Project Managers, Operations Managers

COMMUNICATION STYLE:
- Balance technical accuracy with business clarity
- Focus on operational impact and team implications
- Highlight trends and patterns
- Connect technical issues to business operations

REPORT FORMAT:

## 📊 Operational Health Report

### Executive Summary
[2-3 sentences summarizing status and key concerns]

### Health Score: [X/100]
🟢 Healthy (80-100) | 🟡 Attention Needed (60-79) | 🔴 Critical (<60)

### Key Performance Indicators
| KPI | Value | Trend | Impact |
|-----|-------|-------|--------|
[Focus on operationally relevant metrics]

### Operational Impact
- Service availability
- User experience impact
- Processing efficiency
- Team workload implications

### Issues Requiring Attention
| Priority | Issue | Impact | Owner |
|----------|-------|--------|-------|
[Prioritized issue list]

### Trend Analysis
- Week-over-week comparisons
- Pattern identification
- Forecasted concerns

### Recommended Actions
1. Immediate (within 24h)
2. Short-term (this week)
3. Medium-term (this month)

### Resource Requirements
- Team time needed
- Budget implications if any
- External dependencies

GUIDELINES:
- Use business-friendly language with some technical terms
- Quantify impact in business terms (users affected, transactions impacted)
- Provide clear ownership and timelines
- Include trend context (better/worse than last period)
"""


class ManagerAgent(BaseAgent):
    """
    Manager Persona Agent for operational reports.
    Provides balanced technical and business analysis.
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.MANAGER,
            name="manager_analyst",
            description="Operational analyst providing management reports"
        )
    
    def _get_system_prompt(self) -> str:
        return MANAGER_AGENT_PROMPT
    
    def _get_tools(self) -> list:
        return []
    
    async def transform_report(self, raw_data: str, context: dict = None) -> AgentResult:
        """
        Transform raw data into a manager-focused operational report.
        
        Args:
            raw_data: Raw data from the data agent
            context: Additional context
            
        Returns:
            AgentResult with formatted manager report
        """
        query = context.get("user_query", "system health") if context else "system health"
        
        prompt = f"""Transform the following data into an operational report for management.

USER QUERY: {query}

RAW DATA AND ANALYSIS:
{raw_data}

Create a clear operational report following your format guidelines.
Focus on business impact, trends, and actionable recommendations with ownership."""

        return await self.process(prompt, context)


# ============================================================================
# EXECUTIVE PERSONA AGENT
# ============================================================================

EXECUTIVE_AGENT_PROMPT = """You are an Executive Briefing Agent that transforms system health data into concise strategic summaries for C-level executives.

YOUR AUDIENCE: CTO, CIO, CEO, VP of Engineering, Senior Leadership

COMMUNICATION STYLE:
- Extremely concise and to the point
- Focus on business outcomes and risk
- No technical jargon unless absolutely necessary
- Strategic implications over operational details

REPORT FORMAT:

## 📈 Executive Health Summary

### Status: [🟢 HEALTHY / 🟡 ATTENTION / 🔴 CRITICAL]

### Bottom Line
[One sentence: Are we OK? What's the risk?]

### Key Numbers
| Metric | Value | Status |
|--------|-------|--------|
[3-5 most critical business metrics only]

### Business Impact
- Customer impact: [None/Minor/Moderate/Severe]
- Revenue risk: [None/Low/Medium/High]
- Reputation risk: [None/Low/Medium/High]

### Strategic Concerns
[Bullet points - only if there are genuine concerns]

### Decisions Needed
[Only if executive action is required]

### 30-Second Summary
[Literally what to say if asked "How are systems doing?"]

GUIDELINES:
- Maximum 1 page equivalent
- No technical details - just outcomes
- Focus on: risk, cost, customer impact
- Only escalate what truly needs executive attention
- Use traffic light colors for quick scanning
- If everything is fine, say so briefly
"""


class ExecutiveAgent(BaseAgent):
    """
    Executive Persona Agent for strategic summaries.
    Provides concise, high-level briefings for leadership.
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.EXECUTIVE,
            name="executive_analyst",
            description="Executive briefing analyst providing strategic summaries"
        )
    
    def _get_system_prompt(self) -> str:
        return EXECUTIVE_AGENT_PROMPT
    
    def _get_tools(self) -> list:
        return []
    
    async def transform_report(self, raw_data: str, context: dict = None) -> AgentResult:
        """
        Transform raw data into an executive-focused strategic summary.
        
        Args:
            raw_data: Raw data from the data agent
            context: Additional context
            
        Returns:
            AgentResult with formatted executive summary
        """
        query = context.get("user_query", "system health") if context else "system health"
        
        prompt = f"""Transform the following data into a brief executive summary.

USER QUERY: {query}

RAW DATA AND ANALYSIS:
{raw_data}

Create a concise executive briefing following your format guidelines.
Focus on business impact, risk, and only what executives need to know.
Be extremely brief - executives have 30 seconds."""

        return await self.process(prompt, context)


# ============================================================================
# FACTORY FUNCTION
# ============================================================================

def get_persona_agent(persona: PersonaType) -> BaseAgent:
    """
    Get the appropriate persona agent.
    
    Args:
        persona: The persona type
        
    Returns:
        The corresponding persona agent
    """
    agents = {
        PersonaType.IT: ITAgent,
        PersonaType.MANAGER: ManagerAgent,
        PersonaType.EXECUTIVE: ExecutiveAgent
    }
    
    agent_class = agents.get(persona)
    if not agent_class:
        raise ValueError(f"Unknown persona: {persona}")
    
    return agent_class()
