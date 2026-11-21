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

    # Tier-based Model Routing
    AUTO_ROUTING: bool = True
    MODEL_FAST: str = "openai/gpt-4o-mini"
    MODEL_NORMAL: str = "openai/gpt-4o"
    MODEL_AGENT: str = "anthropic/claude-3-5-sonnet"
    
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

    def get_model_for_tier(self, tier: str, requested_model: Optional[str] = None) -> str:
        """
        Returns the appropriate model based on tier and configuration.
        If AUTO_ROUTING is False, returns requested_model or DEFAULT_MODEL.
        """
        if not self.AUTO_ROUTING:
            return requested_model or self.DEFAULT_MODEL
            
        tier_upper = (tier or "").upper()
        if tier_upper == "FAST":
            return self.MODEL_FAST
        elif tier_upper == "AGENT":
            return self.MODEL_AGENT
        else:  # NORMAL
            return requested_model or self.MODEL_NORMAL

settings = Settings()
