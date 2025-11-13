# src/tts.py
from __future__ import annotations
from typing import Tuple, Union
import os
import torch
import numpy as np
import tempfile

# Global TTS model instance to avoid reloading
_tts_model = None

def get_device():
    """Detect best available device for Mac M1."""
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def get_tts_model():
    """Load XTTS v2 model (singleton pattern)."""
    global _tts_model
    
    if _tts_model is not None:
        return _tts_model
    
    from TTS.api import TTS
    
    device = get_device()
    print(f"[XTTS] Loading model for device: {device}")
    
    # Load with gpu=False to avoid CUDA check, then move to MPS
    _tts_model = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
        gpu=False
    )
    
    # Move to MPS if available
    if device == "mps":
        _tts_model.to(device)
        print(f"[XTTS] ✓ Model loaded and moved to MPS")
    else:
        print(f"[XTTS] ✓ Model loaded on CPU")
    
    return _tts_model

def run_tts_clone(
    ref_audio_path: str,
    text_to_speak: str,
    model_id: str = "coqui/XTTS-v2",
    language: str = "en",
) -> Union[Tuple[int, np.ndarray], Exception]:
    """
    Synthesize speech using XTTS v2 voice cloning.
    
    Args:
        ref_audio_path: Path to reference audio for voice cloning
        text_to_speak: Text to synthesize
        model_id: Model ID (unused, kept for compatibility)
        language: Language code (en, fr, es, de, it, pt, zh, ja, ko, ar)
    
    Returns:
        (sampling_rate, waveform) on success, or Exception on failure.
    """
    try:
        if not ref_audio_path or not os.path.exists(ref_audio_path):
            return Exception("Reference audio file not found")
        
        if not text_to_speak or len(text_to_speak.strip()) == 0:
            return Exception("No text provided")
        
        print(f"[XTTS] Cloning voice (language: {language})")
        
        # Load model (cached after first call)
        tts = get_tts_model()
        
        # Generate output to temporary file
        output_path = os.path.join(tempfile.gettempdir(), f"xtts_{os.getpid()}.wav")
        
        # Run TTS with voice cloning
        tts.tts_to_file(
            text=text_to_speak,
            speaker_wav=ref_audio_path,
            language=language,
            file_path=output_path
        )
        
        # Load generated audio
        import scipy.io.wavfile as wavfile
        sr, wav = wavfile.read(output_path)
        
        # Convert to float32 and normalize
        if wav.dtype == np.int16:
            wav = wav.astype(np.float32) / 32768.0
        elif wav.dtype == np.int32:
            wav = wav.astype(np.float32) / 2147483648.0
        
        # Clean up
        try:
            os.remove(output_path)
        except:
            pass
        
        print(f"[XTTS] ✓ Generated {len(wav)} samples at {sr}Hz")
        return sr, wav
        
    except Exception as e:
        print(f"[XTTS] ✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return e
