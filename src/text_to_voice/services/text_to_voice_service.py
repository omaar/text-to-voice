from fastapi import UploadFile
from text_to_voice.models.schemas import TextToVoiceRequest, TextToVoiceResponse
from text_to_voice.services.pdf_extraction_service import PdfExtractionService
from text_to_voice.utils.voice_clone_manager import VoiceCloneManager
from text_to_voice.utils.huggingface_voice_clone_manager import HuggingFaceVoiceCloneManager
from text_to_voice.core.logging_config import setup_logger
from text_to_voice.services.temp_file_manager import TempFileManager
import tempfile
from datetime import datetime
from pathlib import Path
# Configurar logger
logger = setup_logger("text_to_voice_service")

class TextToVoiceService:
    def __init__(self, model_path: str = None):
        self.output_dir = Path("audio_files")
        self.output_dir.mkdir(exist_ok=True)
        self.pdf_service = PdfExtractionService()
        self.voice_manager = HuggingFaceVoiceCloneManager(model_name=model_path, device="cpu")
        logger.info("TextToVoiceService inicializado")
        self.temp_file_manager = TempFileManager()

    async def convert_text_to_voice(self, text: str, reference_audio: UploadFile) -> TextToVoiceResponse:
        """
        Convierte texto a voz usando clonación de voz
        
        Args:
            request (TextToVoiceRequest): Solicitud con texto y audio de referencia
            
        Returns:
            TextToVoiceResponse: Respuesta con la URL del audio generado
        """
        logger.info("Iniciando conversión de texto a voz")
        
        try:
            
            ref_audio_path = await self.temp_file_manager.save_temp_file(reference_audio)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"audio_{timestamp}.wav"
            output_path = self.output_dir / output_filename
                
            # Clonar la voz
            cloned_audio = self.voice_manager.clone_voice(
                reference_audio=ref_audio_path,
                text=text,
                output_path=str(output_path)
            )
            
            logger.info(f"Audio generado exitosamente: {output_filename}")
            
            return TextToVoiceResponse(
                message="Conversión exitosa",
                num_pages=1,  # Para texto simple, siempre es 1
                output_files=[f"/audio/{output_filename}"]
            )
                
        except Exception as e:
            logger.error(f"Error en conversión de texto a voz: {str(e)}", exc_info=True)
            raise

    async def convert_pdf_to_voice(self, request: TextToVoiceRequest) -> TextToVoiceResponse:
        """
        Convierte el contenido de un PDF a voz usando clonación de voz
        
        Args:
            request (TextToVoiceRequest): Solicitud con PDF y audio de referencia
            
        Returns:
            TextToVoiceResponse: Respuesta con las URLs de los audios generados
        """
        logger.info("Iniciando conversión de PDF a voz")
        
        try:
            # Extraer texto del PDF
            pdf_result = await self.pdf_service.extract_text_from_pdf(request.pdf_file)
            
            # Crear directorio temporal para el audio de referencia
            with tempfile.TemporaryDirectory() as temp_dir:
                # Guardar el audio de referencia
                ref_audio_path = Path(temp_dir) / request.reference_audio.filename
                content = await request.reference_audio.read()
                with open(ref_audio_path, "wb") as buffer:
                    buffer.write(content)
                
                # Generar audios para cada párrafo
                output_files = []
                for i, text in enumerate(pdf_result.texts):
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_filename = f"audio_{timestamp}_{i}.wav"
                    output_path = self.output_dir / output_filename
                    
                    # Clonar la voz
                    cloned_audio = self.voice_manager.clone_voice(
                        reference_audio=str(ref_audio_path),
                        text=text,
                        output_path=str(output_path)
                    )
                    
                    output_files.append(f"/audio/{output_filename}")
                
                logger.info(f"Generados {len(output_files)} archivos de audio")
                
                return TextToVoiceResponse(
                    message="Conversión exitosa",
                    num_pages=pdf_result.num_paragraphs,
                    output_files=output_files
                )
                
        except Exception as e:
            logger.error(f"Error en conversión de PDF a voz: {str(e)}", exc_info=True)
            raise 