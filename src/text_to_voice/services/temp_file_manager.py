import tempfile
import os
from fastapi import UploadFile
from text_to_voice.core.logging_config import setup_logger

logger = setup_logger("temp_file_manager")

class TempFileManager:
    def __init__(self):
        self.temp_dir = None
        self.file_path = None

    async def save_temp_file(self, file: UploadFile) -> str:
        """
        Guarda un archivo temporalmente y retorna su ubicación
        
        Args:
            file (UploadFile): Archivo a guardar temporalmente
            
        Returns:
            str: Ruta del archivo temporal guardado
        """
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = os.path.join(self.temp_dir.name, file.filename)
        
        logger.debug(f"Guardando archivo temporalmente en: {self.file_path}")
        
        content = await file.read()
        with open(self.file_path, "wb") as buffer:
            buffer.write(content)
            
        return self.file_path

    def cleanup(self):
        """
        Limpia los archivos temporales
        """
        if self.temp_dir:
            self.temp_dir.cleanup()
            logger.debug("Archivos temporales eliminados") 