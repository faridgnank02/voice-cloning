# src/tts.py
from __future__ import annotations
from typing import Tuple, Union

import numpy as np
from transformers import pipeline

# We use the text-to-speech pipeline with XTTS v2 (zero-shot cloning)
# Example forward params: {"speaker_wav": "/path/to/ref.wav", "language": "en"}

def get_tts_pipeline(model_id: str):
    """
    Create a TTS pipeline for the given model.
    XTTS v2 works well for zero-shot cloning and is available on the Hub.
    """
    # NOTE: Add device selection similar to ASR if needed
    return pipeline("text-to-speech", model=model_id)

def run_tts_clone(
    ref_audio_path: str,
    text_to_speak: str,
    model_id: str = "coqui/XTTS-v2",
    language: str = "en",
) -> Union[Tuple[int, np.ndarray], Exception]:
    """
    Synthesize 'text_to_speak' in the cloned voice from 'ref_audio_path'.

    Returns:
        (sampling_rate, waveform) on success, or Exception on failure.
    """
    try:
        tts = get_tts_pipeline(model_id)
        result = tts(
            text_to_speak,
            forward_params={"speaker_wav": ref_audio_path, "language": language},
        )
        # transformers TTS returns dict like: {"audio": {"array": np.ndarray, "sampling_rate": 24000}}
        audio = result["audio"]
        sr = int(audio["sampling_rate"])
        wav = audio["array"].astype(np.float32)
        return sr, wav
    except Exception as e:
        return e
