from utils.read_file_manager import PdfManager
from utils.voice_clone_manager import VoiceCloneManager

# Example usage
pdf_manager = PdfManager('example.pdf')
pdf_manager.open_pdf()
print(f"Número de páginas: {pdf_manager.get_num_pages()}")
pdf_manager.close_pdf()
# Inicializar el manager
manager = VoiceCloneManager("ruta/al/modelo/f5-tts.pt")

# Clonar voz para un texto
audio = manager.clone_voice(
    reference_audio="ruta/al/audio/referencia.wav",
    text="Hola, este es un texto de prueba",
    output_path="salida.wav"
)

# Clonar voz para múltiples textos
texts = ["Primer texto", "Segundo texto", "Tercer texto"]
audios = manager.batch_clone_voice(
    reference_audio="ruta/al/audio/referencia.wav",
    texts=texts,
    output_dir="directorio/salida"
)