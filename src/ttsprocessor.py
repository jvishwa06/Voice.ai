"""
Text-to-Speech Service

Handles text-to-speech using Kokoro model loaded directly in memory.
"""

import logging
import time
import soundfile as sf
import tempfile
import os
from kokoro import KPipeline
from typing import Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TTS:
    """
    Text-to-Speech processor using Kokoro model loaded directly in memory.
    
    This class loads the Kokoro model into memory for fast, local text-to-speech
    generation without requiring a separate API server.
    """
    
    def __init__(
        self,
        lang_code: str = 'a',
        voice: str = "af_sky",
        sample_rate: int = 24000,
        speed: float = 1.0
    ):
        """
        Initialize the TTS processor with Kokoro model.
        
        Args:
            lang_code: Language code for Kokoro pipeline
            voice: Voice to use for synthesis
            sample_rate: Audio sample rate
            speed: Speech speed multiplier (0.25 to 4.0)
        """
        self.lang_code = lang_code
        self.voice = voice
        self.sample_rate = sample_rate
        self.speed = speed
        
        self.is_processing = False
        self.last_processing_time = 0
        
        # Load Kokoro pipeline into memory
        logger.info(f"Loading Kokoro pipeline with lang_code={lang_code}, voice={voice}")
        self.pipeline = KPipeline(lang_code=lang_code)
        logger.info("Kokoro pipeline loaded successfully")
    
    def text_to_speech(self, text: str) -> bytes:
        """
        Convert text to speech using Kokoro model.
        
        Args:
            text: Text to convert to speech
            
        Returns:
            Audio data as bytes
        """
        self.is_processing = True
        start_time = time.time()
        
        try:
            logger.info(f"Generating speech for {len(text)} characters of text")
            
            # Generate audio using Kokoro pipeline
            generator = self.pipeline(text, voice=self.voice)
            
            # Get the first (and typically only) result from the generator
            for i, (gs, ps, audio) in enumerate(generator):
                # Create a temporary file to save the audio
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                    temp_path = temp_file.name
                    sf.write(temp_path, audio, self.sample_rate)
                
                # Read the audio data back as bytes
                with open(temp_path, 'rb') as f:
                    audio_data = f.read()
                
                # Clean up the temporary file
                os.unlink(temp_path)
                
                self.last_processing_time = time.time() - start_time
                
                logger.info(f"Generated speech after {self.last_processing_time:.2f}s, "
                           f"size: {len(audio_data)} bytes")
                
                return audio_data
                
        except Exception as e:
            logger.error(f"TTS generation failed: {e}")
            raise e
        
        finally:
            self.is_processing = False
    
    def get_status(self) -> dict:
        """
        Get current status of the TTS processor.
        
        Returns:
            Dict containing processing status and performance metrics
        """
        return {
            "is_processing": self.is_processing,
            "last_processing_time": self.last_processing_time,
            "voice": self.voice,
            "lang_code": self.lang_code,
            "sample_rate": self.sample_rate
        }
