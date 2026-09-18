from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    # API Keys
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    
    # Models
    DEFAULT_MODEL: str = "openai/gpt-4o-mini"
    EFFORT_MICRO_MODEL: str = "openai/gpt-4o-mini"
    
    # Redis configuration
    REDIS_URL: Optional[str] = None
    
    # Quota Limits
    MAX_RPM: int = 100       # Requests Per Minute
    MAX_RPD: int = 5000      # Requests Per Day
    MAX_TPM: int = 1000000   # Tokens Per Minute
    
    # Classifier Weights
    WEIGHT_WORD_COUNT: float = 0.25
    WEIGHT_KEYWORDS: float = 0.35
    WEIGHT_FILE_PATHS: float = 0.20
    WEIGHT_HISTORY: float = 0.20
    
    # Defaults
    DEFAULT_HISTORICAL_RATE: float = 0.50
    
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

settings = Settings()
