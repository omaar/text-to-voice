from pydantic_settings import BaseSettings
from functools import lru_cache

class Settings(BaseSettings):
    PROJECT_NAME: str = "Text to Voice API"
    VERSION: str = "1.0.0"
    DESCRIPTION: str = "API para convertir texto a voz"
    API_V1_STR: str = "/api/v1"
    
    class Config:
        case_sensitive = True

@lru_cache()
def get_settings():
    return Settings()

settings = get_settings() 