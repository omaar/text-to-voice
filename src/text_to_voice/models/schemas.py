from pydantic import BaseModel
from typing import Optional, List
from fastapi import UploadFile, File

class TextToVoiceRequest(BaseModel):
    pdf_file: UploadFile = File(...)
    reference_audio: UploadFile = File(...)

class TextToVoiceResponse(BaseModel):
    message: str
    num_pages: int
    output_files: list[str]

class ExtractTextFromPdfResponse(BaseModel):
    message: str
    texts: List[str]
    num_paragraphs: int
