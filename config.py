from pydantic_settings import BaseSettings
from functools import lru_cache
import os

class Settings(BaseSettings):
    # App
    APP_NAME: str = "Locaith Data Engine"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    
    # JWT
    SECRET_KEY: str = "locaith-data-engine-secret-key-2026"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 1440  # 24 hours for development
    REFRESH_TOKEN_EXPIRE_DAYS: int = 7
    
    # Database
    DATABASE_PATH: str = "data/lakehouse.duckdb"
    DATA_DIR: str = "data"
    
    # Rate Limiting (requests per minute)
    RATE_LIMIT_STARTER: int = 10
    RATE_LIMIT_PRO: int = 100
    RATE_LIMIT_BUSINESS: int = 500
    RATE_LIMIT_ENTERPRISE: int = 10000
    
    # Storage limits (MB)
    STORAGE_LIMIT_STARTER: int = 100
    STORAGE_LIMIT_PRO: int = 5120
    STORAGE_LIMIT_BUSINESS: int = 51200
    
    # Query limits per month
    QUERY_LIMIT_STARTER: int = 1000
    QUERY_LIMIT_PRO: int = 50000
    QUERY_LIMIT_BUSINESS: int = 500000
    
    # AI/Gemini
    GEMINI_API_KEY: str = ""
    
    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra env vars not defined

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings()

# Ensure data directory exists
os.makedirs(settings.DATA_DIR, exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "raw"), exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "processed"), exist_ok=True)
os.makedirs(os.path.join(settings.DATA_DIR, "parquet"), exist_ok=True)
