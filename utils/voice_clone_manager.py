import torch
import torchaudio
from pathlib import Path
import numpy as np
from typing import Optional, Union, List
import logging

class VoiceCloneManager:
    def __init__(self, model_path: str, device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Inicializa el manager de clonación de voz.
        
        Args:
            model_path (str): Ruta al modelo f5-tts
            device (str): Dispositivo a usar (cuda/cpu)
        """
        self.device = device
        self.model_path = Path(model_path)
        self.model = None
        self.logger = logging.getLogger(__name__)
        
    def load_model(self) -> None:
        """Carga el modelo f5-tts."""
        try:
            self.model = torch.jit.load(self.model_path, map_location=self.device)
            self.model.eval()
            self.logger.info("Modelo f5-tts cargado exitosamente")
        except Exception as e:
            self.logger.error(f"Error al cargar el modelo: {str(e)}")
            raise
            
    def preprocess_audio(self, audio_path: Union[str, Path], sample_rate: int = 22050) -> torch.Tensor:
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
        
    def clone_voice(
            self, 
            reference_audio: Union[str, Path],
            text: str,
            output_path: Optional[Union[str, Path]] = None
        ) -> torch.Tensor:
        """
        Clona la voz del audio de referencia para el texto especificado.
        
        Args:
            reference_audio (Union[str, Path]): Ruta al audio de referencia
            text (str): Texto a sintetizar
            output_path (Optional[Union[str, Path]]): Ruta para guardar el audio resultante
            
        Returns:
            torch.Tensor: Tensor del audio sintetizado
        """
        if self.model is None:
            self.load_model()
            
        # Preprocesar audio de referencia
        ref_waveform = self.preprocess_audio(reference_audio)
        
        # Generar audio clonado
        with torch.no_grad():
            cloned_audio = self.model(ref_waveform, text)
            
        # Guardar si se especifica una ruta de salida
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torchaudio.save(output_path, cloned_audio.cpu(), 22050)
            self.logger.info(f"Audio guardado en: {output_path}")
            
        return cloned_audio
        
    def batch_clone_voice(self, reference_audio: Union[str, Path], texts: List[str], output_dir: Union[str, Path]) -> List[torch.Tensor]:
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
        
        results = []
        for i, text in enumerate(texts):
            output_path = output_dir / f"cloned_{i}.wav"
            cloned_audio = self.clone_voice(reference_audio, text, output_path)
            results.append(cloned_audio)
            
        return results 