"""
Databricks Client Module.
Handles connection and queries to Azure Databricks.
"""

from databricks import sql
from config import config
from typing import Any
from observability import get_logger, metrics, get_tracer

logger = get_logger(__name__)
tracer = get_tracer()


class DatabricksClient:
    """Client for connecting to and querying Azure Databricks."""
    
    def __init__(self):
        """Initialize the Databricks client with configuration."""
        self.server_hostname = config.DATABRICKS_SERVER_HOSTNAME
        self.http_path = config.DATABRICKS_HTTP_PATH
        self.access_token = config.DATABRICKS_ACCESS_TOKEN
        self.catalog = config.DATABRICKS_CATALOG
        self.schema = config.DATABRICKS_SCHEMA
        
    def _get_connection(self):
        """Create and return a new database connection."""
        return sql.connect(
            server_hostname=self.server_hostname,
            http_path=self.http_path,
            access_token=self.access_token
        )
    
    def execute_query(self, query: str) -> list[dict[str, Any]]:
        """
        Execute a SQL query and return results as a list of dictionaries.
        
        Args:
            query: SQL query string to execute
            
        Returns:
            List of dictionaries where each dict represents a row
        """
        logger.debug("Executing Databricks query", extra={"extra_data": {"query": query[:200]}})
        
        try:
            with metrics.time_query():
                with tracer.span("databricks_query", {"query_preview": query[:100]}):
                    with self._get_connection() as connection:
                        with connection.cursor() as cursor:
                            cursor.execute(query)
                            
                            # Get column names from cursor description
                            columns = [desc[0] for desc in cursor.description]
                            
                            # Fetch all results and convert to list of dicts
                            rows = cursor.fetchall()
                            results = [dict(zip(columns, row)) for row in rows]
                            
                            logger.debug(f"Query returned {len(results)} rows")
                            return results
                            
        except Exception as e:
            logger.error(f"Query execution failed: {str(e)}", extra={"extra_data": {"query": query[:200]}})
            raise DatabricksQueryError(f"Query execution failed: {str(e)}") from e
    
    def get_tables(self, catalog: str = None, schema: str = None) -> list[str]:
        """
        Get list of available tables in the specified catalog and schema.
        
        Args:
            catalog: Catalog name (uses default if not specified)
            schema: Schema name (uses default if not specified)
            
        Returns:
            List of table names
        """
        cat = catalog or self.catalog
        sch = schema or self.schema
        
        query = f"SHOW TABLES IN {cat}.{sch}"
        results = self.execute_query(query)
        
        return [row.get('tableName', row.get('table_name', '')) for row in results]
    
    def get_table_schema(self, table_name: str, catalog: str = None, schema: str = None) -> list[dict]:
        """
        Get the schema (columns) of a specific table.
        
        Args:
            table_name: Name of the table
            catalog: Catalog name (uses default if not specified)
            schema: Schema name (uses default if not specified)
            
        Returns:
            List of column information dictionaries
        """
        cat = catalog or self.catalog
        sch = schema or self.schema
        
        query = f"DESCRIBE TABLE {cat}.{sch}.{table_name}"
        return self.execute_query(query)
    
    def test_connection(self) -> bool:
        """
        Test if the connection to Databricks is working.
        
        Returns:
            True if connection is successful, False otherwise
        """
        try:
            self.execute_query("SELECT 1")
            return True
        except Exception:
            return False


class DatabricksQueryError(Exception):
    """Custom exception for Databricks query errors."""
    pass


# Create a singleton instance
databricks_client = DatabricksClient()
