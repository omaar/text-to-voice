import fitz  # PyMuPDF
from text_to_voice.utils.spellchecker_manager import SpellCheckerManager
from text_to_voice.core.logging_config import setup_logger

logger = setup_logger("read_file_manager")

class ReadFileManager:
    def __init__(self, file_path):
        self.file_path = file_path
        self.file_document = None
        self.spellchecker_manager = SpellCheckerManager()
        
    def open_pdf(self):
        try:
            self.file_document = fitz.open(self.file_path)
            logger.info(f"PDF abierto exitosamente: {self.file_path}")
            return True
        except Exception as e:
            logger.error(f"Error al abrir el PDF: {str(e)}", exc_info=True)
            return False
        
    def get_num_pages(self):
        if self.file_document:
            return len(self.file_document)
        return 0
    
    def get_page_text(self, page_num):
        if self.file_document:
            page = self.file_document[page_num]
            return page.get_text()
        return ""
    
    def extract_text_paragraphs(self):
        """
        Extrae solo párrafos de texto, ignorando títulos y pies de página.
        Utiliza heurísticas para identificar párrafos basados en:
        - Tamaño de fuente
        - Posición en la página
        - Espaciado
        """
        if not self.file_document:
            return ""

        logger.info("Iniciando extracción de párrafos")
        paragraphs = []

        for page_num in range(len(self.file_document)):
            page = self.file_document[page_num]
            logger.debug(f"Procesando página {page_num + 1}")

            # Obtener bloques de texto con información de formato
            blocks = page.get_text("dict")["blocks"]
            
            # Calcular altura promedio de texto para la página
            text_heights = []
            for block in blocks:
                if "lines" in block:
                    for line in block["lines"]:
                        if "spans" in line:
                            for span in line["spans"]:
                                if "size" in span:
                                    text_heights.append(span["size"])
            
            if text_heights:
                avg_height = sum(text_heights) / len(text_heights)
                logger.debug(f"Altura promedio de texto en página {page_num + 1}: {avg_height}")
            else:
                avg_height = 0
                logger.warning(f"No se encontraron alturas de texto en la página {page_num + 1}")

            # Procesar cada bloque
            for block in blocks:
                if "lines" not in block:
                    continue

                block_text = ""
                is_paragraph = True
                
                # Obtener la posición Y del bloque
                block_y = block.get("bbox", [0, 0, 0, 0])[1]
                
                # Verificar si es un título o pie de página
                if block_y < 50 or block_y > page.rect.height - 50:
                    logger.debug(f"Bloque ignorado por posición (posible título/pie de página): Y={block_y}")
                    continue

                for line in block["lines"]:
                    if "spans" not in line:
                        continue

                    line_text = ""
                    for span in line["spans"]:
                        # Verificar si el texto es demasiado grande (posible título)
                        if "size" in span and span["size"] > avg_height * 1.5:
                            logger.debug(f"Texto ignorado por tamaño (posible título): {span['text'][:30]}...")
                            is_paragraph = False
                            break
                        
                        line_text += span.get("text", "")
                    
                    if is_paragraph:
                        block_text += line_text + " "

                # Limpiar y validar el texto del bloque
                block_text = block_text.strip()
                if block_text and len(block_text.split()) > 5:  # Ignorar bloques muy cortos
                    # Corregir ortografía del bloque
                    corrected_text = self.spellchecker_manager.correct_text(block_text)
                    paragraphs.append(corrected_text)
                    logger.debug(f"Párrafo corregido: {corrected_text[:50]}...")

        logger.info(f"Extracción completada. {len(paragraphs)} párrafos encontrados")
        return paragraphs
    
    def close_pdf(self):
        if self.file_document:
            self.file_document.close()
            logger.debug("PDF cerrado")