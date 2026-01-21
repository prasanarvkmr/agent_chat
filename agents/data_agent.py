"""
Data Agent Module.
Specialized agent for data retrieval and analysis from Databricks.
This agent focuses on raw data operations and provides data to persona agents.
"""

from typing import Any, Optional
from .base_agent import BaseAgent, AgentType, AgentResult
from observability import get_logger

logger = get_logger(__name__)


# Data agent tools (imported from main tools module)
def _get_data_tools():
    """Get tools specific to data operations."""
    from tools import (
        query_tool,
        list_tables_tool,
        describe_table_tool,
        get_table_sample_tool,
        get_table_stats_tool,
        analyze_time_series_tool,
        detect_anomalies_tool,
        get_error_summary_tool
    )
    return [
        query_tool,
        list_tables_tool,
        describe_table_tool,
        get_table_sample_tool,
        get_table_stats_tool,
        analyze_time_series_tool,
        detect_anomalies_tool,
        get_error_summary_tool
    ]


DATA_AGENT_PROMPT = """You are a specialized Data Analyst Agent responsible for retrieving and analyzing data from Azure Databricks.

YOUR ROLE:
- Execute SQL queries to retrieve data
- Analyze table structures and schemas
- Calculate metrics and statistics
- Provide raw data and analysis results

GUIDELINES:
1. Always verify table existence before querying
2. Use appropriate time filters for relevant data
3. Calculate key statistics: counts, averages, min/max, percentiles
4. Identify trends and anomalies in the data
5. Return structured data that can be consumed by other agents

OUTPUT FORMAT:
Return your findings as structured data with:
- **Raw Data**: The actual query results
- **Summary Statistics**: Key metrics calculated
- **Data Quality Notes**: Any issues or limitations found
- **Time Range**: The period covered by the data

IMPORTANT:
- You are a data retrieval specialist
- Do NOT format for end users - other agents will do that
- Focus on accuracy and completeness of data
- Include relevant metadata about the data
"""


class DataAgent(BaseAgent):
    """
    Data Agent for retrieving and analyzing data from Databricks.
    Provides raw data and analysis to persona agents.
    """
    
    def __init__(self):
        super().__init__(
            agent_type=AgentType.DATA,
            name="data_analyst",
            description="Specialized agent for data retrieval and analysis from Databricks"
        )
        self._metadata_summary = ""
    
    def _get_system_prompt(self) -> str:
        """Get data agent system prompt with metadata context."""
        from metadata_cache import metadata_cache
        
        try:
            metadata_cache.ensure_fresh_cache()
            self._metadata_summary = metadata_cache.get_metadata_summary_for_agent(compact=True)
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            self._metadata_summary = "*Use list_tables and describe_table to discover data.*"
        
        return f"""{DATA_AGENT_PROMPT}

---

## AVAILABLE DATA CATALOG

{self._metadata_summary}

Use describe_table for detailed column information.
Always use fully qualified names: catalog.schema.table
"""
    
    def _get_tools(self) -> list:
        """Get data-specific tools."""
        return _get_data_tools()
    
    async def fetch_health_data(self, time_range: str = "24h") -> AgentResult:
        """
        Fetch system health data for the specified time range.
        
        Args:
            time_range: Time range like "24h", "7d", "30d"
            
        Returns:
            AgentResult with health data
        """
        prompt = f"""Analyze system health data for the last {time_range}.

Steps:
1. List available tables related to system health, logs, metrics, or monitoring
2. Describe the most relevant tables
3. Query for key health indicators:
   - Error counts and rates
   - Response times / latency
   - Transaction volumes
   - Success/failure ratios
   - Any anomalies
4. Calculate summary statistics

Return the raw data and statistics in a structured format."""

        return await self.process(prompt)
    
    async def fetch_metrics(self, metric_type: str, time_range: str = "24h") -> AgentResult:
        """
        Fetch specific metrics data.
        
        Args:
            metric_type: Type of metrics (performance, errors, usage, etc.)
            time_range: Time range for the data
            
        Returns:
            AgentResult with metrics data
        """
        prompt = f"""Retrieve {metric_type} metrics for the last {time_range}.

Find and query tables containing {metric_type} data.
Calculate:
- Current values
- Trends over time
- Comparisons to thresholds
- Min/Max/Avg statistics

Return structured data with all metrics and calculations."""

        return await self.process(prompt)
    
    async def run_custom_query(self, query_description: str) -> AgentResult:
        """
        Run a custom data query based on description.
        
        Args:
            query_description: Natural language description of data needed
            
        Returns:
            AgentResult with query results
        """
        return await self.process(query_description)


# Singleton instance
data_agent = DataAgent()
