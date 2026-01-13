"""
Configuration module for the Agentic AI application.
Loads settings from environment variables.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Config:
    """Application configuration loaded from environment variables."""
    
    # Kong AI Gateway Settings
    KONG_AI_GATEWAY_BASE_URL: str = os.getenv("KONG_AI_GATEWAY_BASE_URL", "")
    KONG_API_KEY: str = os.getenv("KONG_API_KEY", "")
    
    # LLM Model Selection
    # Available: gpt-4, gpt-5, claude-sonnet, claude-opus, gemini-flash, gemini-pro
    LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4")
    
    # Model to Kong route mapping
    MODEL_ROUTES = {
        # OpenAI models -> /openai route
        "gpt-4": {"route": "openai", "model": "gpt-4"},
        "gpt-5": {"route": "openai", "model": "gpt-5"},
        "gpt-4-turbo": {"route": "openai", "model": "gpt-4-turbo"},
        # Anthropic models -> /claude route  
        "claude-sonnet": {"route": "claude", "model": "claude-sonnet-4-20250514"},
        "claude-opus": {"route": "claude", "model": "claude-opus-4-20250514"},
        # Google models -> /gemini route
        "gemini-flash": {"route": "gemini", "model": "gemini-2.0-flash"},
        "gemini-pro": {"route": "gemini", "model": "gemini-2.5-pro"},
    }
    
    @classmethod
    def get_llm_config(cls) -> dict:
        """Get the Kong route and model name for the selected LLM."""
        model_key = cls.LLM_MODEL.lower()
        
        if model_key in cls.MODEL_ROUTES:
            route_config = cls.MODEL_ROUTES[model_key]
            return {
                "route": route_config["route"],
                "model": route_config["model"],
                "api_base": f"{cls.KONG_AI_GATEWAY_BASE_URL.rstrip('/')}/{route_config['route']}",
                "api_key": cls.KONG_API_KEY
            }
        else:
            # Default: assume it's a direct model name with openai route
            return {
                "route": "openai",
                "model": model_key,
                "api_base": f"{cls.KONG_AI_GATEWAY_BASE_URL.rstrip('/')}/openai",
                "api_key": cls.KONG_API_KEY
            }
    
    # Azure Databricks Settings
    DATABRICKS_SERVER_HOSTNAME: str = os.getenv("DATABRICKS_SERVER_HOSTNAME", "")
    DATABRICKS_HTTP_PATH: str = os.getenv("DATABRICKS_HTTP_PATH", "")
    DATABRICKS_ACCESS_TOKEN: str = os.getenv("DATABRICKS_ACCESS_TOKEN", "")
    DATABRICKS_CATALOG: str = os.getenv("DATABRICKS_CATALOG", "")
    DATABRICKS_SCHEMA: str = os.getenv("DATABRICKS_SCHEMA", "")
    
    @classmethod
    def validate(cls) -> list[str]:
        """Validate that required configuration is present."""
        errors = []
        
        if not cls.KONG_AI_GATEWAY_BASE_URL:
            errors.append("KONG_AI_GATEWAY_BASE_URL is required")
        if not cls.KONG_API_KEY:
            errors.append("KONG_API_KEY is required")
        if not cls.DATABRICKS_SERVER_HOSTNAME:
            errors.append("DATABRICKS_SERVER_HOSTNAME is required")
        if not cls.DATABRICKS_HTTP_PATH:
            errors.append("DATABRICKS_HTTP_PATH is required")
        if not cls.DATABRICKS_ACCESS_TOKEN:
            errors.append("DATABRICKS_ACCESS_TOKEN is required")
            
        return errors


# Create a singleton config instance
config = Config()
