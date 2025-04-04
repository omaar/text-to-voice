from markitdown import MarkItDown
from text_to_voice.core.logging_config import setup_logger

logger = setup_logger("markdown_manager")

class MarkdownManager:
    def __init__(self, enable_plugins=False):
        """
        Inicializa el MarkdownManager con la opción de habilitar plugins.
        
        Args:
            enable_plugins (bool): Si se deben habilitar plugins. Por defecto es False.
        """
        self.md = MarkItDown(enable_plugins=enable_plugins)
        logger.info("MarkdownManager inicializado")
    
    def convert_file(self, file_path):
        """
        Convierte un archivo a formato markdown.
        
        Args:
            file_path (str): Ruta al archivo a convertir.
            
        Returns:
            str: Contenido del archivo convertido a markdown.
        """
        try:
            logger.info(f"Iniciando conversión del archivo: {file_path}")
            result = self.md.convert(file_path)
            logger.info("Conversión completada exitosamente")
            return result.text_content
        except Exception as e:
            logger.error(f"Error al convertir el archivo: {str(e)}", exc_info=True)
            return ""
    
    def convert_stream(self, file_stream):
        """
        Convierte un stream de archivo a formato markdown.
        
        Args:
            file_stream: Stream del archivo a convertir (debe ser un objeto binario).
            
        Returns:
            str: Contenido del archivo convertido a markdown.
        """
        try:
            logger.info("Iniciando conversión del stream")
            result = self.md.convert_stream(file_stream)
            logger.info("Conversión completada exitosamente")
            return result.text_content
        except Exception as e:
            logger.error(f"Error al convertir el stream: {str(e)}", exc_info=True)
            return ""