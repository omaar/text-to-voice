import fitz  # pymupdf se importa como fitz


class PdfManager:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path
        self.pdf_document = None
        
    def open_pdf(self):
        try:
            self.pdf_document = fitz.open(self.pdf_path)
            return True
        except Exception as e:
            print(f"Error al abrir el PDF: {str(e)}")
            return False
        
    def get_num_pages(self):
        if self.pdf_document:
            return len(self.pdf_document)
        return 0
        
    def close_pdf(self):
        if self.pdf_document:
            self.pdf_document.close()