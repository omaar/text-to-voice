import torch
import torchaudio
from pathlib import Path
import numpy as np
from typing import Optional, Union, List
import logging
import os
from transformers import AutoProcessor, SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
import soundfile as sf

class HuggingFaceVoiceCloneManager:
    def __init__(self, model_name: str = "microsoft/speecht5_tts", device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Inicializa el manager de clonación de voz usando Hugging Face.
        
        Args:
            model_name (str): Nombre del modelo de Hugging Face a usar
            device (str): Dispositivo a usar (cuda/cpu)
        """
        self.device = device
        self.model_name = model_name
        self.model = None
        self.processor = None
        self.vocoder = None
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> None:
        """Carga el modelo y el procesador de Hugging Face."""
        try:
            self.processor = SpeechT5Processor.from_pretrained(self.model_name)
            self.model = SpeechT5ForTextToSpeech.from_pretrained(self.model_name).to(self.device)
            self.vocoder = SpeechT5HifiGan.from_pretrained(self.model_name).to(self.device)
            self.model.eval()
            self.vocoder.eval()
            self.logger.info(f"Modelo {self.model_name} cargado exitosamente")
        except Exception as e:
            self.logger.error(f"Error al cargar el modelo: {str(e)}")
            raise
            
    def preprocess_audio(self, audio_path: Union[str, Path], sample_rate: int = 16000) -> torch.Tensor:
        """
        Preprocesa el audio de entrada.
        
        Args:
            audio_path (Union[str, Path]): Ruta al archivo de audio
            sample_rate (int): Tasa de muestreo deseada
            
        Returns:
            torch.Tensor: Tensor de audio preprocesado
        """
        waveform, sr = torchaudio.load(audio_path)
        if sr != sample_rate:
            resampler = torchaudio.transforms.Resample(sr, sample_rate)
            waveform = resampler(waveform)
        return waveform.to(self.device)
        
    def get_speaker_embeddings(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Genera embeddings del hablante a partir del audio de referencia.
        
        Args:
            waveform (torch.Tensor): Audio de referencia
            
        Returns:
            torch.Tensor: Embeddings del hablante con la forma correcta
        """
        # Tomar el primer segundo del audio
        first_second = waveform[:, :16000]
        
        # Calcular la media y asegurar la forma correcta (1, 1, 512)
        embeddings = torch.mean(first_second, dim=1, keepdim=True)
        embeddings = embeddings.unsqueeze(1)  # Añadir dimensión de batch
        
        # Asegurar que los embeddings tengan la forma correcta
        if embeddings.shape[-1] != 512:
            embeddings = torch.nn.functional.pad(embeddings, (0, 512 - embeddings.shape[-1]))
            
        return embeddings
        
    def clone_voice(
            self, 
            reference_audio: Union[str, Path],
            text: str,
            output_path: Optional[Union[str, Path]] = None,
            speaker_embeddings: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
        """
        Clona la voz del audio de referencia para el texto especificado.
        
        Args:
            reference_audio (Union[str, Path]): Ruta al audio de referencia
            text (str): Texto a sintetizar
            output_path (Optional[Union[str, Path]]): Ruta para guardar el audio resultante
            speaker_embeddings (Optional[torch.Tensor]): Embeddings del hablante
            
        Returns:
            torch.Tensor: Tensor del audio sintetizado
        """
        if self.model is None:
            self.load_model()
            
        # Preprocesar audio de referencia
        ref_waveform = self.preprocess_audio(reference_audio)
        
        # Generar embeddings del hablante si no se proporcionan
        if speaker_embeddings is None:
            with torch.no_grad():
                speaker_embeddings = self.get_speaker_embeddings(ref_waveform)
        
        # Procesar el texto
        inputs = self.processor(
            text=text,
            return_tensors="pt"
        ).to(self.device)
        
        # Generar audio
        with torch.no_grad():
            speech = self.model.generate_speech(
                inputs["input_ids"],
                speaker_embeddings,
                vocoder=self.vocoder
            )
            
        # Guardar si se especifica una ruta de salida
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(output_path, speech.cpu().numpy(), self.processor.sampling_rate)
            self.logger.info(f"Audio guardado en: {output_path}")
            
        return speech
        
    def batch_clone_voice(
            self, 
            reference_audio: Union[str, Path], 
            texts: List[str], 
            output_dir: Union[str, Path]
        ) -> List[torch.Tensor]:
        """
        Clona la voz para múltiples textos.
        
        Args:
            reference_audio (Union[str, Path]): Ruta al audio de referencia
            texts (List[str]): Lista de textos a sintetizar
            output_dir (Union[str, Path]): Directorio para guardar los audios
            
        Returns:
            List[torch.Tensor]: Lista de tensores de audio sintetizados
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Obtener embeddings del hablante una sola vez
        ref_waveform = self.preprocess_audio(reference_audio)
        with torch.no_grad():
            speaker_embeddings = self.get_speaker_embeddings(ref_waveform)
        
        results = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"cloned_{i}.wav"
            cloned_audio = self.clone_voice(
                reference_audio=reference_audio,
                text=text,
                output_path=output_path,
                speaker_embeddings=speaker_embeddings
            )
            results.append(cloned_audio)
            
        return results 