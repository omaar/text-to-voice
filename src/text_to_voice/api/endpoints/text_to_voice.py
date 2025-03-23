from fastapi import APIRouter, UploadFile, File, HTTPException
from text_to_voice.models.schemas import TextToVoiceRequest, TextToVoiceResponse
from text_to_voice.services.text_to_voice_service import TextToVoiceService
import tempfile
import os

router = APIRouter()
service = TextToVoiceService()

@router.post("/text-to-voice", response_model=TextToVoiceResponse)
async def convert_text_to_voice(request: TextToVoiceRequest):
    try:
        return await service.convert_text_to_voice(request)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf-to-voice", response_model=TextToVoiceResponse)
async def convert_pdf_to_voice(file: UploadFile = File(...)):
    try:
        # Guardar el archivo PDF temporalmente
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file_path = temp_file.name

        # Procesar el PDF
        result = await service.convert_pdf_to_voice(temp_file_path)

        # Limpiar el archivo temporal
        os.unlink(temp_file_path)

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) 