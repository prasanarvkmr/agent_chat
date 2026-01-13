"""
Logging module for the System Health Analyst agent.
Provides structured logging with file and console output.
"""

import logging
import os
import json
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any


# Create logs directory
LOGS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
os.makedirs(LOGS_DIR, exist_ok=True)


class JSONFormatter(logging.Formatter):
    """Custom formatter that outputs JSON for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        log_entry = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add extra fields if present
        if hasattr(record, "extra_data"):
            log_entry["data"] = record.extra_data
            
        # Add exception info if present
        if record.exc_info:
            log_entry["exception"] = self.formatException(record.exc_info)
            
        return json.dumps(log_entry)


class PrettyFormatter(logging.Formatter):
    """Human-readable formatter for console output."""
    
    COLORS = {
        "DEBUG": "\033[36m",     # Cyan
        "INFO": "\033[32m",      # Green
        "WARNING": "\033[33m",   # Yellow
        "ERROR": "\033[31m",     # Red
        "CRITICAL": "\033[35m",  # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record: logging.LogRecord) -> str:
        color = self.COLORS.get(record.levelname, self.RESET)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        message = f"{color}[{timestamp}] [{record.levelname:8}]{self.RESET} {record.name}: {record.getMessage()}"
        
        # Add extra data if present
        if hasattr(record, "extra_data") and record.extra_data:
            data_str = json.dumps(record.extra_data, indent=2, default=str)
            message += f"\n  Data: {data_str}"
            
        return message


class ContextLogger(logging.LoggerAdapter):
    """Logger adapter that adds context data to log messages."""
    
    def process(self, msg: str, kwargs: dict) -> tuple[str, dict]:
        extra = kwargs.get("extra", {})
        if self.extra:
            extra["extra_data"] = {**self.extra, **extra.get("extra_data", {})}
        else:
            extra["extra_data"] = extra.get("extra_data", {})
        kwargs["extra"] = extra
        return msg, kwargs
    
    def with_context(self, **context) -> "ContextLogger":
        """Create a new logger with additional context."""
        new_extra = {**self.extra, **context}
        return ContextLogger(self.logger, new_extra)


def setup_logging(
    level: str = "INFO",
    enable_console: bool = True,
    enable_file: bool = True,
    log_file: str = "agent.log",
    max_bytes: int = 10 * 1024 * 1024,  # 10 MB
    backup_count: int = 5
) -> None:
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        enable_console: Whether to log to console
        enable_file: Whether to log to file
        log_file: Name of the log file
        max_bytes: Maximum size of each log file
        backup_count: Number of backup files to keep
    """
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, level.upper()))
    
    # Clear existing handlers
    root_logger.handlers = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(PrettyFormatter())
        root_logger.addHandler(console_handler)
    
    # File handler (JSON format)
    if enable_file:
        log_path = os.path.join(LOGS_DIR, log_file)
        file_handler = RotatingFileHandler(
            log_path,
            maxBytes=max_bytes,
            backupCount=backup_count
        )
        file_handler.setFormatter(JSONFormatter())
        root_logger.addHandler(file_handler)
        
    # Also create separate error log
    error_log_path = os.path.join(LOGS_DIR, "errors.log")
    error_handler = RotatingFileHandler(
        error_log_path,
        maxBytes=max_bytes,
        backupCount=backup_count
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(JSONFormatter())
    root_logger.addHandler(error_handler)


def get_logger(name: str, **context) -> ContextLogger:
    """
    Get a logger instance with optional context.
    
    Args:
        name: Logger name (usually __name__)
        **context: Additional context to include in all log messages
        
    Returns:
        ContextLogger instance
    """
    logger = logging.getLogger(name)
    return ContextLogger(logger, context)
