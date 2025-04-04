from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from text_to_voice.models.schemas import TextToVoiceRequest, TextToVoiceResponse, ExtractTextFromPdfResponse
from text_to_voice.services.text_to_voice_service import TextToVoiceService
from text_to_voice.services.pdf_extraction_service import PdfExtractionService
from text_to_voice.core.logging_config import setup_logger

# Configurar logger
logger = setup_logger("text_to_voice_endpoints")

router = APIRouter()
model_path = "SWivid/f5-tts"
clone_service = TextToVoiceService(model_path)
pdf_service = PdfExtractionService()

# @router.post("/text-to-voice", response_model=TextToVoiceResponse)
# async def convert_text_to_voice(request: TextToVoiceRequest):
#     logger.info(f"Iniciando conversión de texto a voz para archivo: {request.pdf_file.filename}")
#     try:
#         result = await service.convert_text_to_voice(request)
#         logger.info(f"Conversión exitosa para archivo: {request.pdf_file.filename}")
#         return result
#     except Exception as e:
#         logger.error(f"Error en conversión de texto a voz: {str(e)}", exc_info=True)
#         raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/text-from-pdf", response_model=ExtractTextFromPdfResponse)
async def extract_text_from_pdf(pdf_file: UploadFile = File(...)):
    """
    Extrae el texto de un archivo PDF, ignorando títulos y pies de página
    """
    logger.info(f"Iniciando extracción de texto del PDF: {pdf_file.filename}")
    
    try:
        result = await pdf_service.extract_text_from_pdf(pdf_file)
        return result
    except ValueError as e:
        logger.warning(f"Error de validación: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en extracción de texto del PDF: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/pdf-to-voice", response_model=TextToVoiceResponse)
async def convert_pdf_to_voice(
        pdf_file: UploadFile = File(...),
        reference_audio: UploadFile = File(...)
    ):
    """
    Convierte el texto de un PDF a voz clonada usando un audio de referencia
    """
    logger.info(f"Iniciando conversión de PDF a voz: {pdf_file.filename}")
    
    try:
        # Primero extraemos el texto del PDF usando el servicio
        pdf_extraction_result = await pdf_service.extract_text_from_pdf(pdf_file)
        
        # Ahora procesamos el texto extraído con el servicio de voz
        # Aquí podríamos implementar la lógica de clonación de voz
        result = await clone_service.convert_text_to_voice(
            pdf_extraction_result.texts[0],
            reference_audio
        )
        
        # Por ahora, devolvemos un resultado simplificado
        return TextToVoiceResponse(
            message="Conversión exitosa",
            num_pages=pdf_extraction_result.num_paragraphs,
            output_files=["audio_output.wav"]  # Placeholder
        )
        
    except ValueError as e:
        logger.warning(f"Error de validación: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error en conversión de PDF a voz: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e)) 