"""
Data Agent Module - Contact Center Operations Intelligence

Specialized agent for analyzing contact center data and predicting outages.
Data sources: Genesys, Nuance IVR, Watson IVR, Incidents, Call Metrics,
              Solarwinds, Nexthink, ThousandEyes stored in Databricks.
"""

from typing import Any, Optional
from .base_agent import BaseAgent, AgentType, AgentResult
from observability import get_logger

logger = get_logger(__name__)


def _get_data_tools():
    """Get tools for data operations."""
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


# =============================================================================
# DATA AGENT SYSTEM PROMPT - Contact Center Outage Prediction Specialist
# =============================================================================

DATA_AGENT_PROMPT = """You are an expert Contact Center Data Analyst specializing in outage prediction and operational intelligence.

## YOUR MISSION
Analyze contact center operations data to:
1. Detect early warning signs of outages
2. Calculate leading indicators for proactive alerts
3. Correlate data across multiple systems for root cause analysis
4. Generate actionable insights for operations teams

## DATA SOURCES (in Databricks)
You have access to data from these systems - column names may vary, be intelligent about mapping:

| System | Purpose | Key Metrics to Look For |
|--------|---------|------------------------|
| **Genesys** | Contact center platform | call_volume, queue_time, abandonment_rate, agent_availability, handle_time |
| **Nuance IVR** | Voice recognition/IVR | recognition_rate, ivr_completion, containment_rate, transfer_rate, error_codes |
| **Watson IVR** | AI-powered IVR | intent_confidence, dialog_success, fallback_rate, session_duration |
| **Incidents** | ITSM tickets | severity, priority, affected_service, resolution_time, incident_status |
| **Call Metrics** | Telephony performance | call_quality, latency, jitter, packet_loss, mos_score |
| **Solarwinds** | Network/infra monitoring | cpu_usage, memory_usage, disk_io, response_time, availability |
| **Nexthink** | Endpoint/user experience | device_health, app_crashes, login_time, network_quality |
| **ThousandEyes** | Network path monitoring | path_latency, loss, jitter, bgp_changes, http_response |

## INTELLIGENT DATA DISCOVERY WORKFLOW

When you don't know the exact table/column names, follow this approach:

**Step 1: Discover Available Tables**
```sql
-- First, list all tables to understand what's available
-- Use list_tables tool or: SHOW TABLES IN catalog.schema
```

**Step 2: Understand Table Structure**  
```sql
-- For each relevant table, examine its columns
-- Use describe_table tool to see column names and types
```

**Step 3: Sample the Data**
```sql
-- Get a small sample to understand the actual data
SELECT * FROM catalog.schema.table_name LIMIT 5
```

**Step 4: Identify Key Columns** by looking for patterns:
- Timestamps: `timestamp`, `created_at`, `event_time`, `datetime`, `date`, `time`
- Metrics: `value`, `count`, `duration`, `rate`, `score`, `avg`, `total`
- Status: `status`, `state`, `result`, `outcome`, `success`, `error`
- IDs: `id`, `session_id`, `call_id`, `incident_id`, `correlation_id`

## LEADING INDICATORS FOR OUTAGE PREDICTION

Calculate these metrics to predict potential outages:

### 1. Call Volume Anomalies
- Sudden spike or drop (>30% change in 15 min)
- Compare current vs same time yesterday/last week

### 2. IVR Health Indicators
- Recognition rate drop (<85% = warning, <70% = critical)
- Containment rate decline
- Increased transfer/fallback rates

### 3. Infrastructure Stress Signals
- CPU/Memory trending above 80%
- Response time degradation
- Increased error rates in logs

### 4. Network Quality Metrics
- MOS score < 3.5
- Packet loss > 1%
- Latency spikes > 150ms

### 5. Incident Correlation
- Multiple P1/P2 incidents in short window
- Incidents affecting same service/component

## SQL REQUIREMENTS

**ALWAYS use fully qualified table names: `catalog.schema.table_name`**

Example queries for common analyses:

```sql
-- Time-based aggregation (adjust column names based on what you discover)
SELECT 
    DATE_TRUNC('hour', timestamp_column) as hour,
    COUNT(*) as volume,
    AVG(metric_column) as avg_metric
FROM catalog.schema.table_name
WHERE timestamp_column >= CURRENT_TIMESTAMP - INTERVAL 24 HOURS
GROUP BY 1
ORDER BY 1

-- Anomaly detection
SELECT *
FROM catalog.schema.table_name
WHERE metric_column > (SELECT AVG(metric_column) + 2*STDDEV(metric_column) FROM table_name)
  AND timestamp_column >= CURRENT_TIMESTAMP - INTERVAL 1 HOUR
```

## OUTPUT FORMAT

Always structure your findings as:

### 📊 Data Summary
- Tables analyzed
- Time range covered  
- Record counts

### 🔍 Key Findings
- Critical metrics and their values
- Anomalies detected
- Trends identified

### ⚠️ Leading Indicators (if applicable)
- Risk level: LOW / MEDIUM / HIGH / CRITICAL
- Indicators triggered
- Recommended actions

### 📈 Raw Data
- Query results (formatted as tables)
- Statistics calculated

## IMPORTANT BEHAVIORS

1. **Be Adaptive**: Column names may not match exactly - use pattern matching and context
2. **Don't Fail Silently**: If a query fails, try alternative approaches
3. **Correlate Data**: Look for patterns across multiple data sources
4. **Time is Critical**: Always include time context in your analysis
5. **Quantify Everything**: Provide numbers, not just descriptions
"""


class DataAgent(BaseAgent):
    """
    Data Agent for Contact Center Operations Intelligence.
    Specializes in outage prediction and operational analytics.
    
    Features:
    - Intelligent data discovery (handles undocumented schemas)
    - Leading indicator calculation for outage prediction
    - Cross-system correlation (Genesys, IVR, Incidents, etc.)
    - Configurable metadata caching
    """
    
    def __init__(self, use_metadata_cache: bool = True):
        """
        Initialize the Data Agent.
        
        Args:
            use_metadata_cache: If False, skips metadata caching for faster startup.
                              Set to False if metadata loading is slow.
        """
        super().__init__(
            agent_type=AgentType.DATA,
            name="data_analyst",
            description="Contact Center Data Analyst for outage prediction and operational intelligence"
        )
        self._metadata_summary = ""
        self._use_metadata_cache = use_metadata_cache
    
    def _get_system_prompt(self) -> str:
        """Build the system prompt with optional metadata context."""
        from config import config
        
        # Get catalog and schema from config
        catalog = config.DATABRICKS_CATALOG
        schema = config.DATABRICKS_SCHEMA
        
        # Optionally load metadata cache
        metadata_section = self._get_metadata_section()
        
        return f"""{DATA_AGENT_PROMPT}

---

## DATABRICKS CONFIGURATION

**Catalog:** `{catalog}`
**Schema:** `{schema}`

All queries must use: `{catalog}.{schema}.<table_name>`

---

{metadata_section}
"""
    
    def _get_metadata_section(self) -> str:
        """Get metadata section based on cache settings."""
        if not self._use_metadata_cache:
            return """## DATA DISCOVERY MODE

Metadata caching is disabled. Use this workflow:
1. `list_tables` - See all available tables
2. `describe_table` - Get column details for relevant tables  
3. `get_table_sample` - Preview actual data (LIMIT 5)
4. Build your query based on discovered schema
"""
        
        try:
            from metadata_cache import metadata_cache
            metadata_cache.ensure_fresh_cache()
            self._metadata_summary = metadata_cache.get_metadata_summary_for_agent(compact=True)
            
            return f"""## CACHED TABLE METADATA

{self._metadata_summary}

*Tip: Use `describe_table` for full column details.*
"""
        except Exception as e:
            logger.warning(f"Could not load metadata: {e}")
            return """## DATA DISCOVERY MODE

Metadata unavailable. Use tools to discover:
- `list_tables` - See all tables
- `describe_table` - Get column details
- `get_table_sample` - Preview data
"""
    
    def _get_tools(self) -> list:
        """Get data analysis tools."""
        return _get_data_tools()
    
    async def fetch_health_data(self, time_range: str = "24h") -> AgentResult:
        """
        Analyze contact center health across all systems.
        
        Args:
            time_range: Time range like "1h", "24h", "7d"
            
        Returns:
            AgentResult with health analysis and leading indicators
        """
        prompt = f"""Analyze contact center health for the last {time_range}.

**DISCOVERY STEPS:**
1. List available tables to find health/monitoring data
2. Look for tables related to: Genesys, IVR, calls, incidents, network, infrastructure
3. Describe relevant tables to understand their columns
4. Query for key health indicators

**ANALYZE THESE AREAS:**
- Call volume and abandonment rates (Genesys)
- IVR success/containment rates (Nuance/Watson)
- Active incidents and severity (ITSM)
- Infrastructure health (Solarwinds/Nexthink)
- Network quality (ThousandEyes)

**CALCULATE LEADING INDICATORS:**
- Volume anomalies vs baseline
- Error rate trends
- Infrastructure stress signals
- Service degradation patterns

Return structured findings with risk assessment."""

        return await self.process(prompt)
    
    async def predict_outage_risk(self, lookback_hours: int = 4) -> AgentResult:
        """
        Calculate outage risk based on leading indicators.
        
        Args:
            lookback_hours: Hours of data to analyze
            
        Returns:
            AgentResult with risk assessment and indicators
        """
        prompt = f"""Perform outage risk prediction using the last {lookback_hours} hours of data.

**WORKFLOW:**
1. Discover available monitoring tables
2. For each data source found, calculate:
   - Current vs baseline comparison
   - Trend direction (improving/degrading)
   - Anomaly score

**LEADING INDICATORS TO CALCULATE:**

| Indicator | Warning Threshold | Critical Threshold |
|-----------|------------------|-------------------|
| Call Volume Change | >30% in 15min | >50% in 15min |
| Abandonment Rate | >8% | >15% |
| IVR Recognition Rate | <85% | <70% |
| Avg Handle Time Change | >20% | >40% |
| Open P1/P2 Incidents | ≥2 | ≥4 |
| Infrastructure CPU/Mem | >80% | >90% |
| Network Latency | >100ms | >200ms |

**OUTPUT FORMAT:**
1. Risk Score: 0-100 (with color: 🟢🟡🟠🔴)
2. Active Warning Indicators
3. Critical Indicators (if any)
4. Recommended Actions
5. Supporting Data"""

        return await self.process(prompt)
    
    async def fetch_metrics(self, metric_type: str, time_range: str = "24h") -> AgentResult:
        """
        Fetch specific metrics by type.
        
        Args:
            metric_type: Type of metrics (calls, ivr, incidents, network, infrastructure)
            time_range: Time range for the data
            
        Returns:
            AgentResult with metrics data
        """
        metric_guides = {
            "calls": "call volume, abandonment rate, handle time, queue time, service level",
            "ivr": "recognition rate, containment rate, transfer rate, completion rate, errors",
            "incidents": "open incidents, MTTR, severity distribution, affected services",
            "network": "latency, packet loss, jitter, MOS scores, availability",
            "infrastructure": "CPU, memory, disk, response time, error logs"
        }
        
        focus = metric_guides.get(metric_type.lower(), metric_type)
        
        prompt = f"""Retrieve {metric_type} metrics for the last {time_range}.

**FOCUS ON:** {focus}

**STEPS:**
1. Find tables containing {metric_type} data
2. Describe table structure
3. Query key metrics with time aggregation
4. Calculate statistics: current, avg, min, max, trend

**INCLUDE:**
- Hourly/daily breakdown
- Comparison to baseline
- Anomalies detected
- Data quality notes"""

        return await self.process(prompt)
    
    async def correlate_incidents(self, hours_back: int = 24) -> AgentResult:
        """
        Correlate incidents with operational metrics to find patterns.
        
        Args:
            hours_back: Hours to analyze
            
        Returns:
            AgentResult with correlation analysis
        """
        prompt = f"""Correlate incidents with operational data for the last {hours_back} hours.

**ANALYSIS:**
1. Find and query incident data (P1/P2/P3 tickets)
2. For each significant incident timeframe:
   - Check call volume anomalies
   - Check IVR error rates
   - Check infrastructure metrics
   - Check network quality

**IDENTIFY:**
- Common patterns before incidents
- Root cause indicators
- Service dependencies
- Time-to-impact relationships

**OUTPUT:**
- Incident timeline
- Correlated metrics
- Pattern summary
- Prevention recommendations"""

        return await self.process(prompt)
    
    async def run_custom_query(self, query_description: str) -> AgentResult:
        """
        Run a custom analysis based on natural language description.
        
        Args:
            query_description: Natural language description of data needed
            
        Returns:
            AgentResult with query results
        """
        return await self.process(query_description)


def create_data_agent(use_metadata_cache: bool = None) -> DataAgent:
    """
    Factory function to create a DataAgent with configuration options.
    
    Args:
        use_metadata_cache: Set False to disable metadata caching (faster startup).
                           If None, uses USE_METADATA_CACHE from config/environment.
        
    Returns:
        Configured DataAgent instance
    """
    if use_metadata_cache is None:
        from config import config
        use_metadata_cache = config.USE_METADATA_CACHE
    
    return DataAgent(use_metadata_cache=use_metadata_cache)


# Default singleton instance - uses config setting for cache
# Override with: create_data_agent(use_metadata_cache=False)
def _create_default_agent():
    """Create the default agent using config settings."""
    try:
        from config import config
        return DataAgent(use_metadata_cache=config.USE_METADATA_CACHE)
    except Exception:
        return DataAgent(use_metadata_cache=True)

data_agent = _create_default_agent()
