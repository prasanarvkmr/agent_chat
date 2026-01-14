"""
Metadata Cache Module.
Caches table metadata from Databricks to avoid repeated lookups.
Supports automatic daily refresh and manual refresh.
"""

import os
import json
import time
from datetime import datetime, timedelta
from typing import Optional, Any
from dataclasses import dataclass, field, asdict
from threading import Lock
from observability import get_logger, metrics

logger = get_logger(__name__)

# Cache file location
CACHE_DIR = os.path.join(os.path.dirname(__file__), "data", "cache")
os.makedirs(CACHE_DIR, exist_ok=True)
CACHE_FILE = os.path.join(CACHE_DIR, "table_metadata.json")


@dataclass
class ColumnInfo:
    """Column metadata."""
    name: str
    data_type: str
    description: Optional[str] = None
    nullable: bool = True
    is_primary_key: bool = False
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TableInfo:
    """Table metadata with columns and description."""
    name: str
    catalog: str
    schema: str
    full_name: str
    description: Optional[str] = None
    columns: list[ColumnInfo] = field(default_factory=list)
    row_count: Optional[int] = None
    last_updated: Optional[str] = None
    domain: Optional[str] = None  # Business domain categorization
    tags: list[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "catalog": self.catalog,
            "schema": self.schema,
            "full_name": self.full_name,
            "description": self.description,
            "columns": [c.to_dict() for c in self.columns],
            "row_count": self.row_count,
            "last_updated": self.last_updated,
            "domain": self.domain,
            "tags": self.tags
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "TableInfo":
        columns = [ColumnInfo(**c) for c in data.get("columns", [])]
        return cls(
            name=data["name"],
            catalog=data["catalog"],
            schema=data["schema"],
            full_name=data["full_name"],
            description=data.get("description"),
            columns=columns,
            row_count=data.get("row_count"),
            last_updated=data.get("last_updated"),
            domain=data.get("domain"),
            tags=data.get("tags", [])
        )
    
    def get_summary(self) -> str:
        """Get a human-readable summary of the table."""
        col_summary = ", ".join([f"{c.name}({c.data_type})" for c in self.columns[:5]])
        if len(self.columns) > 5:
            col_summary += f", ... +{len(self.columns) - 5} more"
        
        desc = self.description or "No description available"
        return f"**{self.full_name}**: {desc}\n  Columns: {col_summary}"


@dataclass
class SchemaInfo:
    """Schema metadata containing tables."""
    name: str
    catalog: str
    description: Optional[str] = None
    tables: dict[str, TableInfo] = field(default_factory=dict)
    domain: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "catalog": self.catalog,
            "description": self.description,
            "domain": self.domain,
            "tables": {k: v.to_dict() for k, v in self.tables.items()}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SchemaInfo":
        tables = {k: TableInfo.from_dict(v) for k, v in data.get("tables", {}).items()}
        return cls(
            name=data["name"],
            catalog=data["catalog"],
            description=data.get("description"),
            domain=data.get("domain"),
            tables=tables
        )


@dataclass
class CatalogInfo:
    """Catalog metadata containing schemas."""
    name: str
    description: Optional[str] = None
    schemas: dict[str, SchemaInfo] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "schemas": {k: v.to_dict() for k, v in self.schemas.items()}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "CatalogInfo":
        schemas = {k: SchemaInfo.from_dict(v) for k, v in data.get("schemas", {}).items()}
        return cls(
            name=data["name"],
            description=data.get("description"),
            schemas=schemas
        )


@dataclass
class MetadataCache:
    """Complete metadata cache."""
    catalogs: dict[str, CatalogInfo] = field(default_factory=dict)
    last_refresh: Optional[str] = None
    refresh_interval_hours: int = 24
    version: str = "1.0"
    
    def to_dict(self) -> dict:
        return {
            "version": self.version,
            "last_refresh": self.last_refresh,
            "refresh_interval_hours": self.refresh_interval_hours,
            "catalogs": {k: v.to_dict() for k, v in self.catalogs.items()}
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "MetadataCache":
        catalogs = {k: CatalogInfo.from_dict(v) for k, v in data.get("catalogs", {}).items()}
        return cls(
            version=data.get("version", "1.0"),
            last_refresh=data.get("last_refresh"),
            refresh_interval_hours=data.get("refresh_interval_hours", 24),
            catalogs=catalogs
        )
    
    def needs_refresh(self) -> bool:
        """Check if cache needs to be refreshed."""
        if not self.last_refresh:
            return True
        
        try:
            last = datetime.fromisoformat(self.last_refresh.replace("Z", "+00:00"))
            now = datetime.utcnow().replace(tzinfo=last.tzinfo)
            age = now - last
            return age > timedelta(hours=self.refresh_interval_hours)
        except Exception:
            return True


class MetadataCacheManager:
    """
    Manages the metadata cache for Databricks tables.
    Provides caching, refresh, and query capabilities.
    """
    
    def __init__(self):
        self._cache: MetadataCache = MetadataCache()
        self._lock = Lock()
        self._databricks_client = None
        self._load_cache()
    
    def _get_databricks_client(self):
        """Lazy load databricks client to avoid circular imports."""
        if self._databricks_client is None:
            from databricks_client import databricks_client
            self._databricks_client = databricks_client
        return self._databricks_client
    
    def _load_cache(self):
        """Load cache from disk."""
        try:
            if os.path.exists(CACHE_FILE):
                with open(CACHE_FILE, "r") as f:
                    data = json.load(f)
                    self._cache = MetadataCache.from_dict(data)
                    logger.info(f"Loaded metadata cache from disk", extra={"extra_data": {
                        "last_refresh": self._cache.last_refresh,
                        "catalog_count": len(self._cache.catalogs)
                    }})
        except Exception as e:
            logger.error(f"Failed to load metadata cache: {e}")
            self._cache = MetadataCache()
    
    def _save_cache(self):
        """Save cache to disk."""
        try:
            with open(CACHE_FILE, "w") as f:
                json.dump(self._cache.to_dict(), f, indent=2, default=str)
            logger.info("Saved metadata cache to disk")
        except Exception as e:
            logger.error(f"Failed to save metadata cache: {e}")
    
    def refresh_cache(self, force: bool = False) -> dict:
        """
        Refresh the metadata cache from Databricks.
        
        Args:
            force: Force refresh even if cache is not stale
            
        Returns:
            Summary of refresh operation
        """
        with self._lock:
            if not force and not self._cache.needs_refresh():
                logger.info("Cache is still fresh, skipping refresh")
                return {
                    "refreshed": False,
                    "reason": "Cache is still fresh",
                    "last_refresh": self._cache.last_refresh
                }
            
            logger.info("Starting metadata cache refresh")
            start_time = time.time()
            
            try:
                client = self._get_databricks_client()
                
                # Get the default catalog and schema from config
                from config import config
                default_catalog = config.DATABRICKS_CATALOG
                default_schema = config.DATABRICKS_SCHEMA
                
                new_cache = MetadataCache()
                tables_count = 0
                columns_count = 0
                
                # For now, focus on the configured catalog/schema
                # Can be extended to scan multiple catalogs
                catalogs_to_scan = [default_catalog] if default_catalog else []
                
                for cat_name in catalogs_to_scan:
                    catalog = CatalogInfo(name=cat_name)
                    
                    # Get schemas in catalog
                    try:
                        schemas_query = f"SHOW SCHEMAS IN {cat_name}"
                        schemas_result = client.execute_query(schemas_query)
                        schema_names = [row.get('databaseName', row.get('namespace', '')) 
                                       for row in schemas_result]
                    except Exception as e:
                        logger.warning(f"Failed to get schemas for catalog {cat_name}: {e}")
                        schema_names = [default_schema] if default_schema else []
                    
                    for schema_name in schema_names:
                        if not schema_name:
                            continue
                            
                        schema = SchemaInfo(name=schema_name, catalog=cat_name)
                        
                        # Infer domain from schema name
                        schema.domain = self._infer_domain(schema_name)
                        
                        try:
                            # Get tables in schema
                            tables = client.get_tables(cat_name, schema_name)
                            
                            for table_name in tables:
                                if not table_name:
                                    continue
                                    
                                full_name = f"{cat_name}.{schema_name}.{table_name}"
                                
                                # Get table schema (columns)
                                try:
                                    columns_data = client.get_table_schema(
                                        table_name, cat_name, schema_name
                                    )
                                    
                                    columns = []
                                    for col in columns_data:
                                        col_name = col.get('col_name', col.get('column_name', ''))
                                        if col_name and not col_name.startswith('#'):
                                            columns.append(ColumnInfo(
                                                name=col_name,
                                                data_type=col.get('data_type', 'unknown'),
                                                description=col.get('comment', None),
                                                nullable=col.get('nullable', True)
                                            ))
                                            columns_count += 1
                                    
                                    # Infer table description from columns
                                    table_desc = self._infer_table_description(
                                        table_name, columns
                                    )
                                    
                                    table = TableInfo(
                                        name=table_name,
                                        catalog=cat_name,
                                        schema=schema_name,
                                        full_name=full_name,
                                        description=table_desc,
                                        columns=columns,
                                        domain=schema.domain,
                                        tags=self._infer_tags(table_name, columns)
                                    )
                                    
                                    schema.tables[table_name] = table
                                    tables_count += 1
                                    
                                except Exception as e:
                                    logger.warning(f"Failed to get schema for {full_name}: {e}")
                                    
                        except Exception as e:
                            logger.warning(f"Failed to get tables for {cat_name}.{schema_name}: {e}")
                        
                        if schema.tables:
                            catalog.schemas[schema_name] = schema
                    
                    if catalog.schemas:
                        new_cache.catalogs[cat_name] = catalog
                
                # Update cache
                new_cache.last_refresh = datetime.utcnow().isoformat() + "Z"
                self._cache = new_cache
                self._save_cache()
                
                duration = time.time() - start_time
                
                logger.info("Metadata cache refresh complete", extra={"extra_data": {
                    "tables": tables_count,
                    "columns": columns_count,
                    "duration_seconds": round(duration, 2)
                }})
                
                metrics.tool_calls.inc(tool="metadata_cache_refresh")
                
                return {
                    "refreshed": True,
                    "tables_count": tables_count,
                    "columns_count": columns_count,
                    "catalogs_count": len(new_cache.catalogs),
                    "duration_seconds": round(duration, 2),
                    "last_refresh": new_cache.last_refresh
                }
                
            except Exception as e:
                logger.error(f"Metadata cache refresh failed: {e}")
                return {
                    "refreshed": False,
                    "error": str(e)
                }
    
    def _infer_domain(self, schema_name: str) -> str:
        """Infer business domain from schema name."""
        schema_lower = schema_name.lower()
        
        domain_keywords = {
            "health": ["health", "medical", "patient", "clinical"],
            "finance": ["finance", "payment", "billing", "transaction", "accounting"],
            "operations": ["ops", "operations", "monitoring", "system", "infra"],
            "analytics": ["analytics", "reporting", "metrics", "kpi"],
            "customer": ["customer", "user", "account", "profile"],
            "sales": ["sales", "order", "revenue", "product"],
            "hr": ["hr", "employee", "payroll", "workforce"],
            "security": ["security", "auth", "access", "audit"],
        }
        
        for domain, keywords in domain_keywords.items():
            if any(kw in schema_lower for kw in keywords):
                return domain
        
        return "general"
    
    def _infer_table_description(self, table_name: str, columns: list[ColumnInfo]) -> str:
        """Infer a description for a table based on its name and columns."""
        name_lower = table_name.lower()
        
        # Check for common patterns
        if "log" in name_lower or "event" in name_lower:
            return f"Event/log data tracking {table_name.replace('_', ' ')}"
        if "metric" in name_lower or "kpi" in name_lower:
            return f"Performance metrics and KPIs for {table_name.replace('_', ' ')}"
        if "user" in name_lower or "customer" in name_lower:
            return f"User/customer related data"
        if "transaction" in name_lower or "order" in name_lower:
            return f"Transaction records for {table_name.replace('_', ' ')}"
        if "config" in name_lower or "setting" in name_lower:
            return f"Configuration settings"
        if "audit" in name_lower:
            return f"Audit trail records"
        if "error" in name_lower or "failure" in name_lower:
            return f"Error and failure tracking data"
        if "health" in name_lower or "status" in name_lower:
            return f"Health/status monitoring data"
        
        # Check columns for hints
        col_names = [c.name.lower() for c in columns]
        if any("timestamp" in c or "date" in c or "time" in c for c in col_names):
            if any("error" in c or "status" in c for c in col_names):
                return "Time-series data with status tracking"
            return "Time-series data"
        
        return f"Data table: {table_name.replace('_', ' ').title()}"
    
    def _infer_tags(self, table_name: str, columns: list[ColumnInfo]) -> list[str]:
        """Infer tags for a table based on its characteristics."""
        tags = []
        name_lower = table_name.lower()
        col_names = [c.name.lower() for c in columns]
        
        if any("timestamp" in c or "created_at" in c or "event_time" in c for c in col_names):
            tags.append("time-series")
        if any("error" in c or "exception" in c for c in col_names):
            tags.append("errors")
        if any("status" in c or "state" in c for c in col_names):
            tags.append("status-tracking")
        if any("metric" in c or "value" in c or "count" in c for c in col_names):
            tags.append("metrics")
        if any("user" in c or "customer" in c for c in col_names):
            tags.append("user-data")
        if "log" in name_lower:
            tags.append("logs")
        if "audit" in name_lower:
            tags.append("audit")
        
        return tags
    
    def get_cache_info(self) -> dict:
        """Get information about the current cache state."""
        with self._lock:
            total_tables = sum(
                len(schema.tables)
                for catalog in self._cache.catalogs.values()
                for schema in catalog.schemas.values()
            )
            
            total_columns = sum(
                len(table.columns)
                for catalog in self._cache.catalogs.values()
                for schema in catalog.schemas.values()
                for table in schema.tables.values()
            )
            
            return {
                "last_refresh": self._cache.last_refresh,
                "needs_refresh": self._cache.needs_refresh(),
                "refresh_interval_hours": self._cache.refresh_interval_hours,
                "catalogs_count": len(self._cache.catalogs),
                "total_tables": total_tables,
                "total_columns": total_columns,
                "version": self._cache.version
            }
    
    def get_all_tables(self) -> list[dict]:
        """Get all cached tables with their metadata."""
        with self._lock:
            tables = []
            for catalog in self._cache.catalogs.values():
                for schema in catalog.schemas.values():
                    for table in schema.tables.values():
                        tables.append(table.to_dict())
            return tables
    
    def get_tables_by_domain(self, domain: str) -> list[dict]:
        """Get tables filtered by domain."""
        with self._lock:
            tables = []
            for catalog in self._cache.catalogs.values():
                for schema in catalog.schemas.values():
                    for table in schema.tables.values():
                        if table.domain and table.domain.lower() == domain.lower():
                            tables.append(table.to_dict())
            return tables
    
    def get_tables_by_schema(self, schema_name: str, catalog_name: str = None) -> list[dict]:
        """Get tables in a specific schema."""
        with self._lock:
            tables = []
            for catalog in self._cache.catalogs.values():
                if catalog_name and catalog.name != catalog_name:
                    continue
                if schema_name in catalog.schemas:
                    schema = catalog.schemas[schema_name]
                    for table in schema.tables.values():
                        tables.append(table.to_dict())
            return tables
    
    def get_table(self, table_name: str, schema: str = None, catalog: str = None) -> Optional[dict]:
        """Get a specific table's metadata."""
        with self._lock:
            for cat in self._cache.catalogs.values():
                if catalog and cat.name != catalog:
                    continue
                for sch in cat.schemas.values():
                    if schema and sch.name != schema:
                        continue
                    if table_name in sch.tables:
                        return sch.tables[table_name].to_dict()
            return None
    
    def search_tables(self, query: str) -> list[dict]:
        """Search tables by name, description, or column names."""
        with self._lock:
            query_lower = query.lower()
            matches = []
            
            for catalog in self._cache.catalogs.values():
                for schema in catalog.schemas.values():
                    for table in schema.tables.values():
                        score = 0
                        
                        # Check table name
                        if query_lower in table.name.lower():
                            score += 10
                        
                        # Check description
                        if table.description and query_lower in table.description.lower():
                            score += 5
                        
                        # Check column names
                        for col in table.columns:
                            if query_lower in col.name.lower():
                                score += 3
                            if col.description and query_lower in col.description.lower():
                                score += 2
                        
                        # Check tags
                        if any(query_lower in tag for tag in table.tags):
                            score += 4
                        
                        # Check domain
                        if table.domain and query_lower in table.domain.lower():
                            score += 3
                        
                        if score > 0:
                            matches.append({
                                "table": table.to_dict(),
                                "relevance_score": score
                            })
            
            # Sort by relevance
            matches.sort(key=lambda x: x["relevance_score"], reverse=True)
            return matches
    
    def get_metadata_summary_for_agent(self) -> str:
        """
        Generate a concise metadata summary for the agent's context.
        This helps the agent understand available data without repeated lookups.
        """
        with self._lock:
            lines = ["## Available Data Tables\n"]
            
            for catalog in self._cache.catalogs.values():
                for schema in catalog.schemas.values():
                    domain_label = f" ({schema.domain})" if schema.domain else ""
                    lines.append(f"\n### Schema: {catalog.name}.{schema.name}{domain_label}\n")
                    
                    for table in schema.tables.values():
                        # Create compact table summary
                        col_summary = ", ".join([c.name for c in table.columns[:6]])
                        if len(table.columns) > 6:
                            col_summary += f", +{len(table.columns) - 6} more"
                        
                        desc = table.description or "No description"
                        tags_str = f" [Tags: {', '.join(table.tags)}]" if table.tags else ""
                        
                        lines.append(f"- **{table.name}**: {desc}{tags_str}")
                        lines.append(f"  Columns: {col_summary}")
            
            if not any(self._cache.catalogs.values()):
                lines.append("\n*No tables cached. Use refresh to load metadata.*")
            
            lines.append(f"\n*Last refreshed: {self._cache.last_refresh or 'Never'}*")
            
            return "\n".join(lines)
    
    def ensure_fresh_cache(self) -> bool:
        """Ensure cache is fresh, refreshing if needed. Returns True if cache is valid."""
        if self._cache.needs_refresh():
            result = self.refresh_cache()
            return result.get("refreshed", False) or not result.get("error")
        return True


# Create singleton instance
metadata_cache = MetadataCacheManager()
