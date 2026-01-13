"""
Metrics module for the System Health Analyst agent.
Provides metrics collection and reporting for monitoring.
"""

import os
import json
import time
import threading
from datetime import datetime
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict
from contextlib import contextmanager


# Create metrics directory
METRICS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)


@dataclass
class MetricPoint:
    """A single metric data point."""
    name: str
    value: float
    timestamp: float
    labels: Dict[str, str] = field(default_factory=dict)
    metric_type: str = "gauge"  # gauge, counter, histogram
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "value": self.value,
            "timestamp": datetime.fromtimestamp(self.timestamp).isoformat() + "Z",
            "labels": self.labels,
            "type": self.metric_type
        }


class Counter:
    """A counter metric that can only increase."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        
    def inc(self, value: float = 1, **labels):
        """Increment the counter."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] += value
            
    def get(self, **labels) -> float:
        """Get the current counter value."""
        label_key = self._label_key(labels)
        return self._values[label_key]
    
    def _label_key(self, labels: dict) -> str:
        return json.dumps(labels, sort_keys=True) if labels else ""
    
    def collect(self) -> list[MetricPoint]:
        """Collect all metric points."""
        points = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = json.loads(label_key) if label_key else {}
                points.append(MetricPoint(
                    name=self.name,
                    value=value,
                    timestamp=time.time(),
                    labels=labels,
                    metric_type="counter"
                ))
        return points


class Gauge:
    """A gauge metric that can increase and decrease."""
    
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description
        self._values: Dict[str, float] = defaultdict(float)
        self._lock = threading.Lock()
        
    def set(self, value: float, **labels):
        """Set the gauge value."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] = value
            
    def inc(self, value: float = 1, **labels):
        """Increment the gauge."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] += value
            
    def dec(self, value: float = 1, **labels):
        """Decrement the gauge."""
        label_key = self._label_key(labels)
        with self._lock:
            self._values[label_key] -= value
            
    def get(self, **labels) -> float:
        """Get the current gauge value."""
        label_key = self._label_key(labels)
        return self._values[label_key]
    
    def _label_key(self, labels: dict) -> str:
        return json.dumps(labels, sort_keys=True) if labels else ""
    
    def collect(self) -> list[MetricPoint]:
        """Collect all metric points."""
        points = []
        with self._lock:
            for label_key, value in self._values.items():
                labels = json.loads(label_key) if label_key else {}
                points.append(MetricPoint(
                    name=self.name,
                    value=value,
                    timestamp=time.time(),
                    labels=labels,
                    metric_type="gauge"
                ))
        return points


class Histogram:
    """A histogram metric for measuring distributions."""
    
    DEFAULT_BUCKETS = [0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10]
    
    def __init__(self, name: str, description: str = "", buckets: list = None):
        self.name = name
        self.description = description
        self.buckets = sorted(buckets or self.DEFAULT_BUCKETS)
        self._counts: Dict[str, Dict[float, int]] = defaultdict(lambda: defaultdict(int))
        self._sums: Dict[str, float] = defaultdict(float)
        self._totals: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()
        
    def observe(self, value: float, **labels):
        """Record an observation."""
        label_key = self._label_key(labels)
        with self._lock:
            self._sums[label_key] += value
            self._totals[label_key] += 1
            for bucket in self.buckets:
                if value <= bucket:
                    self._counts[label_key][bucket] += 1
                    
    def get_stats(self, **labels) -> dict:
        """Get histogram statistics."""
        label_key = self._label_key(labels)
        total = self._totals[label_key]
        if total == 0:
            return {"count": 0, "sum": 0, "avg": 0}
        return {
            "count": total,
            "sum": self._sums[label_key],
            "avg": self._sums[label_key] / total
        }
    
    def _label_key(self, labels: dict) -> str:
        return json.dumps(labels, sort_keys=True) if labels else ""
    
    def collect(self) -> list[MetricPoint]:
        """Collect all metric points."""
        points = []
        timestamp = time.time()
        with self._lock:
            for label_key in self._totals.keys():
                labels = json.loads(label_key) if label_key else {}
                # Sum metric
                points.append(MetricPoint(
                    name=f"{self.name}_sum",
                    value=self._sums[label_key],
                    timestamp=timestamp,
                    labels=labels,
                    metric_type="histogram"
                ))
                # Count metric
                points.append(MetricPoint(
                    name=f"{self.name}_count",
                    value=self._totals[label_key],
                    timestamp=timestamp,
                    labels=labels,
                    metric_type="histogram"
                ))
        return points


class MetricsCollector:
    """Central metrics collector for the application."""
    
    def __init__(self):
        self._metrics: Dict[str, Any] = {}
        self._lock = threading.Lock()
        
        # Pre-defined metrics
        self.request_count = self.counter(
            "agent_requests_total",
            "Total number of agent requests"
        )
        self.request_errors = self.counter(
            "agent_request_errors_total",
            "Total number of agent request errors"
        )
        self.request_duration = self.histogram(
            "agent_request_duration_seconds",
            "Agent request duration in seconds"
        )
        self.tool_calls = self.counter(
            "agent_tool_calls_total",
            "Total number of tool calls"
        )
        self.tool_errors = self.counter(
            "agent_tool_errors_total",
            "Total number of tool call errors"
        )
        self.tool_duration = self.histogram(
            "agent_tool_duration_seconds",
            "Tool call duration in seconds"
        )
        self.llm_tokens = self.counter(
            "agent_llm_tokens_total",
            "Total LLM tokens used"
        )
        self.active_sessions = self.gauge(
            "agent_active_sessions",
            "Number of active sessions"
        )
        self.databricks_queries = self.counter(
            "databricks_queries_total",
            "Total Databricks queries executed"
        )
        self.databricks_query_duration = self.histogram(
            "databricks_query_duration_seconds",
            "Databricks query duration in seconds"
        )
        
    def counter(self, name: str, description: str = "") -> Counter:
        """Create or get a counter metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Counter(name, description)
            return self._metrics[name]
    
    def gauge(self, name: str, description: str = "") -> Gauge:
        """Create or get a gauge metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Gauge(name, description)
            return self._metrics[name]
    
    def histogram(self, name: str, description: str = "", buckets: list = None) -> Histogram:
        """Create or get a histogram metric."""
        with self._lock:
            if name not in self._metrics:
                self._metrics[name] = Histogram(name, description, buckets)
            return self._metrics[name]
    
    @contextmanager
    def time_request(self, endpoint: str = "chat"):
        """Context manager to time a request."""
        start = time.time()
        self.request_count.inc(endpoint=endpoint)
        try:
            yield
        except Exception:
            self.request_errors.inc(endpoint=endpoint)
            raise
        finally:
            duration = time.time() - start
            self.request_duration.observe(duration, endpoint=endpoint)
    
    @contextmanager
    def time_tool(self, tool_name: str):
        """Context manager to time a tool call."""
        start = time.time()
        self.tool_calls.inc(tool=tool_name)
        try:
            yield
        except Exception:
            self.tool_errors.inc(tool=tool_name)
            raise
        finally:
            duration = time.time() - start
            self.tool_duration.observe(duration, tool=tool_name)
    
    @contextmanager
    def time_query(self):
        """Context manager to time a Databricks query."""
        start = time.time()
        self.databricks_queries.inc()
        try:
            yield
        finally:
            duration = time.time() - start
            self.databricks_query_duration.observe(duration)
    
    def collect_all(self) -> list[MetricPoint]:
        """Collect all metrics."""
        points = []
        with self._lock:
            for metric in self._metrics.values():
                points.extend(metric.collect())
        return points
    
    def export_metrics(self) -> dict:
        """Export all metrics as a dictionary."""
        points = self.collect_all()
        return {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "metrics": [p.to_dict() for p in points]
        }
    
    def save_metrics(self):
        """Save current metrics to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = os.path.join(METRICS_DIR, f"metrics_{timestamp}.json")
        
        with open(metrics_file, "w") as f:
            json.dump(self.export_metrics(), f, indent=2)
            
    def get_summary(self) -> dict:
        """Get a summary of key metrics."""
        return {
            "requests": {
                "total": self.request_count.get(endpoint="chat"),
                "errors": self.request_errors.get(endpoint="chat"),
                "avg_duration_ms": round(
                    self.request_duration.get_stats(endpoint="chat").get("avg", 0) * 1000, 2
                )
            },
            "tools": {
                "total_calls": sum(
                    self.tool_calls.get(tool=t) 
                    for t in ["query_databricks", "list_tables", "describe_table", 
                              "get_table_sample", "get_table_stats", 
                              "analyze_time_series_health", "detect_anomalies", 
                              "get_error_summary"]
                ),
                "errors": sum(
                    self.tool_errors.get(tool=t)
                    for t in ["query_databricks", "list_tables", "describe_table"]
                )
            },
            "databricks": {
                "total_queries": self.databricks_queries.get(),
                "avg_query_ms": round(
                    self.databricks_query_duration.get_stats().get("avg", 0) * 1000, 2
                )
            },
            "sessions": {
                "active": self.active_sessions.get()
            }
        }


# Global metrics instance
metrics = MetricsCollector()
