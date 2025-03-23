import PyPDF2

def read_pdf(file_path: str) -> str:
    """
    Lee un archivo PDF y extrae su texto.
    
    Args:
        file_path (str): Ruta al archivo PDF
        
    Returns:
        str: Texto extraído del PDF
    """
    try:
        with open(file_path, 'rb') as file:
            # Crear un objeto PDFReader
            pdf_reader = PyPDF2.PdfReader(file)
            
            # Extraer texto de todas las páginas
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"
            
            return text.strip()
    except Exception as e:
        raise Exception(f"Error al leer el PDF: {str(e)}") 