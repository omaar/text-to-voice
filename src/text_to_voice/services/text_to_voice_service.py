from text_to_voice.models.schemas import TextToVoiceRequest, TextToVoiceResponse
from text_to_voice.utils.pdf_utils import read_pdf
import pyttsx3
import os
from datetime import datetime

class TextToVoiceService:
    def __init__(self):
        self.engine = pyttsx3.init()
        self.output_dir = "audio_files"
        os.makedirs(self.output_dir, exist_ok=True)

    async def convert_text_to_voice(self, request: TextToVoiceRequest) -> TextToVoiceResponse:
        # Configurar el idioma y la voz
        if request.language == "es":
            self.engine.setProperty('rate', 150)
        
        # Generar nombre único para el archivo
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"audio_{timestamp}.mp3"
        filepath = os.path.join(self.output_dir, filename)

        # Convertir texto a voz
        self.engine.save_to_file(request.text, filepath)
        self.engine.runAndWait()

        # Obtener duración del archivo (implementar lógica para obtener duración real)
        duration = 0.0  # Placeholder

        return TextToVoiceResponse(
            audio_url=f"/audio/{filename}",
            duration=duration,
            text_length=len(request.text)
        )

    async def convert_pdf_to_voice(self, pdf_path: str) -> TextToVoiceResponse:
        text = read_pdf(pdf_path)
        return await self.convert_text_to_voice(TextToVoiceRequest(text=text)) 