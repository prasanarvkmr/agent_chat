"""
Agent Tools Module.
Defines tools that the AI agent can use to interact with Databricks
and perform intelligent system health analysis.
"""

from google.adk.tools import FunctionTool
from databricks_client import databricks_client, DatabricksQueryError
from typing import Any
from datetime import datetime, timedelta
from observability import get_logger, metrics, get_tracer

logger = get_logger(__name__)
tracer = get_tracer()


def query_databricks(sql_query: str) -> dict[str, Any]:
    """
    Execute a SQL query against Azure Databricks.
    
    Args:
        sql_query: The SQL query to execute
        
    Returns:
        Dictionary with 'success' status, 'data' or 'error' message
    """
    logger.info("Tool: query_databricks called", extra={"extra_data": {"query_preview": sql_query[:100]}})
    metrics.tool_calls.inc(tool="query_databricks")
    
    try:
        with metrics.time_tool("query_databricks"):
            results = databricks_client.execute_query(sql_query)
            logger.info(f"Query returned {len(results)} rows")
            return {
                "success": True,
                "data": results,
                "row_count": len(results)
            }
    except DatabricksQueryError as e:
        logger.error(f"Query failed: {str(e)}")
        metrics.tool_errors.inc(tool="query_databricks")
        return {
            "success": False,
            "error": str(e)
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        metrics.tool_errors.inc(tool="query_databricks")
        return {
            "success": False,
            "error": f"Unexpected error: {str(e)}"
        }


def list_tables(catalog: str = None, schema: str = None) -> dict[str, Any]:
    """
    List all available tables in Databricks.
    
    Args:
        catalog: Optional catalog name
        schema: Optional schema name
        
    Returns:
        Dictionary with 'success' status and 'tables' list or 'error' message
    """
    logger.info("Tool: list_tables called")
    metrics.tool_calls.inc(tool="list_tables")
    
    try:
        with metrics.time_tool("list_tables"):
            tables = databricks_client.get_tables(catalog, schema)
            logger.info(f"Found {len(tables)} tables")
            return {
                "success": True,
                "tables": tables
            }
    except Exception as e:
        logger.error(f"list_tables failed: {str(e)}")
        metrics.tool_errors.inc(tool="list_tables")
        return {
            "success": False,
            "error": str(e)
        }


def describe_table(table_name: str, catalog: str = None, schema: str = None) -> dict[str, Any]:
    """
    Get the schema/structure of a specific table including column names, types, and descriptions.
    Use this to understand what data a table contains before analyzing it.
    
    Args:
        table_name: Name of the table to describe
        catalog: Optional catalog name
        schema: Optional schema name
        
    Returns:
        Dictionary with 'success' status and 'columns' or 'error' message
    """
    logger.info(f"Tool: describe_table called for {table_name}")
    metrics.tool_calls.inc(tool="describe_table")
    
    try:
        with metrics.time_tool("describe_table"):
            columns = databricks_client.get_table_schema(table_name, catalog, schema)
            return {
                "success": True,
                "table_name": table_name,
                "columns": columns
            }
    except Exception as e:
        logger.error(f"describe_table failed: {str(e)}")
        metrics.tool_errors.inc(tool="describe_table")
        return {
            "success": False,
            "error": str(e)
        }


def get_table_sample(table_name: str, limit: int = 10, catalog: str = None, schema: str = None) -> dict[str, Any]:
    """
    Get a sample of data from a table to understand its contents and data patterns.
    Useful for understanding what kind of data is stored before deeper analysis.
    
    Args:
        table_name: Name of the table to sample
        limit: Number of rows to return (default 10)
        catalog: Optional catalog name
        schema: Optional schema name
        
    Returns:
        Dictionary with sample data and column info
    """
    try:
        cat = catalog or databricks_client.catalog
        sch = schema or databricks_client.schema
        
        query = f"SELECT * FROM {cat}.{sch}.{table_name} LIMIT {limit}"
        results = databricks_client.execute_query(query)
        
        return {
            "success": True,
            "table_name": table_name,
            "sample_data": results,
            "row_count": len(results)
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_table_stats(table_name: str, catalog: str = None, schema: str = None) -> dict[str, Any]:
    """
    Get basic statistics about a table including row count and date range if applicable.
    Helps understand the volume and recency of data.
    
    Args:
        table_name: Name of the table
        catalog: Optional catalog name
        schema: Optional schema name
        
    Returns:
        Dictionary with table statistics
    """
    try:
        cat = catalog or databricks_client.catalog
        sch = schema or databricks_client.schema
        full_table = f"{cat}.{sch}.{table_name}"
        
        # Get row count
        count_query = f"SELECT COUNT(*) as total_rows FROM {full_table}"
        count_result = databricks_client.execute_query(count_query)
        
        return {
            "success": True,
            "table_name": table_name,
            "total_rows": count_result[0]["total_rows"] if count_result else 0
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def analyze_time_series_health(
    table_name: str,
    timestamp_column: str,
    hours_back: int = 24,
    metric_columns: list[str] = None,
    status_column: str = None,
    catalog: str = None,
    schema: str = None
) -> dict[str, Any]:
    """
    Analyze time-series data for health indicators over a specified time period.
    Automatically calculates trends, anomalies, and health metrics.
    
    Args:
        table_name: Name of the table to analyze
        timestamp_column: Name of the timestamp/datetime column
        hours_back: Number of hours to look back (default 24)
        metric_columns: List of numeric columns to analyze (optional)
        status_column: Column containing status/error codes (optional)
        catalog: Optional catalog name
        schema: Optional schema name
        
    Returns:
        Dictionary with health analysis results
    """
    try:
        cat = catalog or databricks_client.catalog
        sch = schema or databricks_client.schema
        full_table = f"{cat}.{sch}.{table_name}"
        
        # Calculate time boundary
        time_filter = f"{timestamp_column} >= CURRENT_TIMESTAMP - INTERVAL {hours_back} HOURS"
        
        analysis = {
            "success": True,
            "table_name": table_name,
            "analysis_period_hours": hours_back,
            "metrics": {}
        }
        
        # Get total record count in period
        count_query = f"""
            SELECT 
                COUNT(*) as total_records,
                MIN({timestamp_column}) as earliest_record,
                MAX({timestamp_column}) as latest_record
            FROM {full_table}
            WHERE {time_filter}
        """
        count_result = databricks_client.execute_query(count_query)
        if count_result:
            analysis["metrics"]["total_records"] = count_result[0]["total_records"]
            analysis["metrics"]["earliest_record"] = str(count_result[0]["earliest_record"])
            analysis["metrics"]["latest_record"] = str(count_result[0]["latest_record"])
        
        # Analyze status distribution if status column provided
        if status_column:
            status_query = f"""
                SELECT 
                    {status_column} as status,
                    COUNT(*) as count,
                    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER(), 2) as percentage
                FROM {full_table}
                WHERE {time_filter}
                GROUP BY {status_column}
                ORDER BY count DESC
            """
            status_result = databricks_client.execute_query(status_query)
            analysis["metrics"]["status_distribution"] = status_result
        
        # Analyze numeric metrics if provided
        if metric_columns:
            for col in metric_columns:
                metric_query = f"""
                    SELECT 
                        AVG({col}) as avg_value,
                        MIN({col}) as min_value,
                        MAX({col}) as max_value,
                        STDDEV({col}) as std_dev,
                        PERCENTILE({col}, 0.5) as median,
                        PERCENTILE({col}, 0.95) as p95,
                        PERCENTILE({col}, 0.99) as p99
                    FROM {full_table}
                    WHERE {time_filter}
                """
                metric_result = databricks_client.execute_query(metric_query)
                if metric_result:
                    analysis["metrics"][col] = metric_result[0]
        
        # Get hourly trend
        trend_query = f"""
            SELECT 
                DATE_TRUNC('hour', {timestamp_column}) as hour,
                COUNT(*) as record_count
            FROM {full_table}
            WHERE {time_filter}
            GROUP BY DATE_TRUNC('hour', {timestamp_column})
            ORDER BY hour
        """
        trend_result = databricks_client.execute_query(trend_query)
        analysis["metrics"]["hourly_trend"] = [
            {"hour": str(r["hour"]), "count": r["record_count"]} 
            for r in trend_result
        ]
        
        return analysis
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def detect_anomalies(
    table_name: str,
    timestamp_column: str,
    value_column: str,
    hours_back: int = 24,
    threshold_std: float = 2.0,
    catalog: str = None,
    schema: str = None
) -> dict[str, Any]:
    """
    Detect anomalies in a metric by finding values outside normal range.
    Uses standard deviation to identify outliers.
    
    Args:
        table_name: Name of the table
        timestamp_column: Timestamp column name
        value_column: Numeric column to check for anomalies
        hours_back: Hours to analyze (default 24)
        threshold_std: Number of standard deviations for anomaly threshold (default 2.0)
        catalog: Optional catalog name
        schema: Optional schema name
        
    Returns:
        Dictionary with anomaly detection results
    """
    try:
        cat = catalog or databricks_client.catalog
        sch = schema or databricks_client.schema
        full_table = f"{cat}.{sch}.{table_name}"
        
        time_filter = f"{timestamp_column} >= CURRENT_TIMESTAMP - INTERVAL {hours_back} HOURS"
        
        # Calculate baseline statistics
        stats_query = f"""
            SELECT 
                AVG({value_column}) as mean_val,
                STDDEV({value_column}) as std_val
            FROM {full_table}
            WHERE {time_filter}
        """
        stats = databricks_client.execute_query(stats_query)
        
        if not stats or stats[0]["std_val"] is None:
            return {
                "success": True,
                "anomalies_found": 0,
                "message": "Insufficient data for anomaly detection"
            }
        
        mean_val = stats[0]["mean_val"]
        std_val = stats[0]["std_val"]
        lower_bound = mean_val - (threshold_std * std_val)
        upper_bound = mean_val + (threshold_std * std_val)
        
        # Find anomalies
        anomaly_query = f"""
            SELECT 
                {timestamp_column} as timestamp,
                {value_column} as value,
                CASE 
                    WHEN {value_column} > {upper_bound} THEN 'HIGH'
                    WHEN {value_column} < {lower_bound} THEN 'LOW'
                END as anomaly_type
            FROM {full_table}
            WHERE {time_filter}
                AND ({value_column} > {upper_bound} OR {value_column} < {lower_bound})
            ORDER BY {timestamp_column} DESC
            LIMIT 50
        """
        anomalies = databricks_client.execute_query(anomaly_query)
        
        return {
            "success": True,
            "table_name": table_name,
            "column_analyzed": value_column,
            "baseline": {
                "mean": round(mean_val, 4),
                "std_dev": round(std_val, 4),
                "lower_threshold": round(lower_bound, 4),
                "upper_threshold": round(upper_bound, 4)
            },
            "anomalies_found": len(anomalies),
            "anomalies": [
                {"timestamp": str(a["timestamp"]), "value": a["value"], "type": a["anomaly_type"]}
                for a in anomalies
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_error_summary(
    table_name: str,
    timestamp_column: str,
    error_column: str,
    hours_back: int = 24,
    catalog: str = None,
    schema: str = None
) -> dict[str, Any]:
    """
    Get a summary of errors/failures from a table.
    Groups errors by type and shows frequency.
    
    Args:
        table_name: Name of the table
        timestamp_column: Timestamp column name
        error_column: Column containing error messages/codes
        hours_back: Hours to analyze (default 24)
        catalog: Optional catalog name
        schema: Optional schema name
        
    Returns:
        Dictionary with error summary
    """
    try:
        cat = catalog or databricks_client.catalog
        sch = schema or databricks_client.schema
        full_table = f"{cat}.{sch}.{table_name}"
        
        time_filter = f"{timestamp_column} >= CURRENT_TIMESTAMP - INTERVAL {hours_back} HOURS"
        
        query = f"""
            SELECT 
                {error_column} as error_type,
                COUNT(*) as occurrence_count,
                MIN({timestamp_column}) as first_seen,
                MAX({timestamp_column}) as last_seen
            FROM {full_table}
            WHERE {time_filter}
                AND {error_column} IS NOT NULL
                AND TRIM({error_column}) != ''
            GROUP BY {error_column}
            ORDER BY occurrence_count DESC
            LIMIT 20
        """
        
        results = databricks_client.execute_query(query)
        
        total_errors = sum(r["occurrence_count"] for r in results)
        
        return {
            "success": True,
            "table_name": table_name,
            "analysis_period_hours": hours_back,
            "total_errors": total_errors,
            "unique_error_types": len(results),
            "error_breakdown": [
                {
                    "error_type": r["error_type"],
                    "count": r["occurrence_count"],
                    "first_seen": str(r["first_seen"]),
                    "last_seen": str(r["last_seen"])
                }
                for r in results
            ]
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# Create Google ADK FunctionTools
# Google ADK infers tool name and description from function name and docstring
query_tool = FunctionTool(query_databricks)
list_tables_tool = FunctionTool(list_tables)
describe_table_tool = FunctionTool(describe_table)
get_table_sample_tool = FunctionTool(get_table_sample)
get_table_stats_tool = FunctionTool(get_table_stats)
analyze_time_series_tool = FunctionTool(analyze_time_series_health)
detect_anomalies_tool = FunctionTool(detect_anomalies)
get_error_summary_tool = FunctionTool(get_error_summary)

# Export all tools as a list
all_tools = [
    query_tool,
    list_tables_tool,
    describe_table_tool,
    get_table_sample_tool,
    get_table_stats_tool,
    analyze_time_series_tool,
    detect_anomalies_tool,
    get_error_summary_tool
]
