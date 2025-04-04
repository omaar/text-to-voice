from fastapi import UploadFile
from text_to_voice.utils.read_file_manager import ReadFileManager
from text_to_voice.models.schemas import ExtractTextFromPdfResponse
from text_to_voice.core.logging_config import setup_logger
from text_to_voice.services.temp_file_manager import TempFileManager

logger = setup_logger("pdf_extraction_service")

class PdfExtractionService:
    def __init__(self):
        self.temp_file_manager = TempFileManager()
        
    async def extract_text_from_pdf(self, pdf_file: UploadFile) -> ExtractTextFromPdfResponse:
        """
        Extrae el texto de un archivo PDF, ignorando títulos y pies de página
        
        Args:
            pdf_file (UploadFile): Archivo PDF a procesar
            
        Returns:
            ExtractTextFromPdfResponse: Objeto con la respuesta de la extracción
            
        Raises:
            Exception: Si hay errores durante el procesamiento
        """
        logger.info(f"Iniciando extracción de texto del PDF: {pdf_file.filename}")
        
        if not pdf_file.filename.endswith('.pdf'):
            logger.warning(f"Archivo no válido: {pdf_file.filename}")
            raise ValueError("El archivo debe ser un PDF")
        
        try:
            pdf_path = await self.temp_file_manager.save_temp_file(pdf_file)

            logger.debug("Iniciando procesamiento del PDF")
            pdf_manager = ReadFileManager(pdf_path)
            pdf_manager.open_pdf()
            
            logger.info("Extrayendo párrafos del PDF")
            paragraphs = pdf_manager.extract_text_paragraphs()
            
            pdf_manager.close_pdf()
            logger.info("Extracción de texto completada exitosamente")

            return ExtractTextFromPdfResponse(
                message="Texto extraído exitosamente",
                texts=paragraphs,
                num_paragraphs=len(paragraphs)
            )
        finally:
            self.temp_file_manager.cleanup() 