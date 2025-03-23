from pydantic import BaseModel
from typing import Optional

class TextToVoiceRequest(BaseModel):
    text: str
    language: Optional[str] = "es"
    voice_id: Optional[str] = None

class TextToVoiceResponse(BaseModel):
    audio_url: str
    duration: float
    text_length: int 