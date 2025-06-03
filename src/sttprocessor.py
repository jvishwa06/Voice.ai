"""
Speech-to-Text Transcription Service

Uses Faster Whisper to transcribe speech audio.
"""

import numpy as np
import logging
import io
from typing import Dict, Any, Tuple
from faster_whisper import WhisperModel
import time
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class STT:
    """
    Speech-to-Text service using Faster Whisper.
    
    This class handles transcription of speech audio segments.
    """
    
    def __init__(
        self,
        model_size: str = "base",
        device: str = None,
        compute_type: str = None,
        beam_size: int = 2,
        sample_rate: int = 44100
    ):
        """
        Initialize the transcription service.
        
        Args:
            model_size: Whisper model size (tiny.en, base.en, small.en, medium.en, large)
            device: Device to run model on ('cpu' or 'cuda'), if None will auto-detect
            compute_type: Model computation type (int8, int16, float16, float32), if None will select based on device
            beam_size: Beam size for decoding
            sample_rate: Audio sample rate in Hz
        """
        self.model_size = model_size
        
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            else:
                self.device = "cpu"
                if torch.backends.mps.is_available():
                    logger.info("MPS is available but not supported by faster-whisper. Using CPU with optimizations.")
        else:
            self.device = device
            
        if compute_type is None:
            if self.device == "cuda":
                self.compute_type = "float16"
            else:
                self.compute_type = "int8"
        else:
            self.compute_type = compute_type
            
        self.beam_size = beam_size
        self.sample_rate = sample_rate
        
        self._initialize_model()
        
        self.is_processing = False
        
        logger.info(f"Initialized Whisper Transcriber with model={model_size}, "f"device={self.device}, compute_type={self.compute_type}")
    
    def _initialize_model(self):
        """Initialize Whisper model."""
        try:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type
            )
            logger.info(f"Successfully loaded Whisper model: {self.model_size}")
        except Exception as e:
            logger.error(f"Failed to load Whisper model: {e}")
            raise
    
    def transcribe(self, audio: np.ndarray) -> Tuple[str, Dict[str, Any]]:
        """
        Transcribe audio data to text.
        
        Args:
            audio: Audio data as numpy array
            
        Returns:
            Tuple[str, Dict[str, Any]]: 
                - Transcribed text
                - Dictionary with additional information (confidence, language, etc.)
        """
        start_time = time.time()
        self.is_processing = True
        
        try:
            if audio.dtype == np.uint8:
                header = bytes(audio[:44])
                if header[:4] == b'RIFF' and header[8:12] == b'WAVE':
                    audio_file = io.BytesIO(bytes(audio))
                    audio = audio_file
                else:
                    logger.warning("Received audio data with incorrect WAV header")
                    audio = audio.astype(np.float32) / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            else:
                audio = audio.astype(np.float32) / np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else audio
            
            # Transcribe
            segments, info = self.model.transcribe(
                audio, 
                beam_size=self.beam_size,
                language="en",
                vad_filter=False
            )
            
            text_segments = [segment.text for segment in segments]
            full_text = " ".join(text_segments).strip()
            
            processing_time = time.time() - start_time
            logger.info(f"Transcription completed in {processing_time:.2f}s: {full_text[:50]}...")
            
            metadata = {
                "confidence": getattr(info, "avg_logprob", 0),
                "language": getattr(info, "language", "en"),
                "processing_time": processing_time,
                "segments_count": len(text_segments)
            }
            
            return full_text, metadata
            
        except Exception as e:
            logger.error(f"Transcription error: {e}")
            return "", {"error": str(e)}
        finally:
            self.is_processing = False
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the current configuration.
        
        Returns:
            Dict containing the current configuration
        """
        return {
            "model_size": self.model_size,
            "device": self.device,
            "compute_type": self.compute_type,
            "beam_size": self.beam_size,
            "sample_rate": self.sample_rate,
            "is_processing": self.is_processing
        }
