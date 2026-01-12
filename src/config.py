"""
Configuration management for Code Assistant.
Loads settings from environment variables.
"""
from pydantic_settings import BaseSettings
from functools import lru_cache
from typing import Optional


class Settings(BaseSettings):
    """Application settings loaded from environment."""

    # LLM Configuration
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-20250514"

    # Tavily Search
    tavily_api_key: str = ""

    # Oracle Database
    oracle_host: str = "localhost"
    oracle_port: str = "1521"
    oracle_service: str = "FREEPDB1"
    oracle_user: str = "codeassist"
    oracle_password: str = "CodeAssist123"

    # OpenTelemetry
    otel_exporter_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "code-assistant"

    # Application
    debug: bool = False

    @property
    def oracle_dsn(self) -> str:
        """Build Oracle DSN string."""
        return f"{self.oracle_host}:{self.oracle_port}/{self.oracle_service}"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Global settings instance
settings = get_settings()
