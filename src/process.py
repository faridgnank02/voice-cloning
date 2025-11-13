import difflib
import re
from functools import lru_cache

#import spaces
import gradio.components.audio as gr_audio
import torch
from transformers import pipeline


# ------------------- Utilities -------------------
def normalize_text(t: str, lower: bool = True) -> str:
    """For normalizing LLM-generated and human-generated strings.
    For LLMs, this removes extraneous quote marks and spaces.
    Now supports multiple languages with accented characters."""
    if lower:
        t = t.lower()
    # Multilingual normalization: keep letters (including accented), digits, and basic punctuation
    # Support for accented characters (é, è, ñ, ü, etc.)
    t = re.sub(r"[^\w\s'\-.,]+", " ", t, flags=re.UNICODE)
    t = re.sub(r"\s+", " ", t).strip()
    return t


#@spaces.GPU
@lru_cache(maxsize=2)
def get_asr_pipeline(model_id: str, device_preference: str) -> pipeline:
    """Cache an ASR pipeline.
    Parameters:
        model_id: String of desired ASR model.
        device_preference: String of desired device for ASR processing, "cuda", "cpu", or "auto".
    Returns:
        transformers.pipeline ASR component.
    """
    if device_preference == "cuda" and torch.cuda.is_available():
        device = 0
    elif device_preference == "auto":
        device = 0 if torch.cuda.is_available() else -1
    else:
        device = -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,           # use English-only Whisper models (.en)
        device=device,
        chunk_length_s=30,
        return_timestamps=False,
    )

def run_asr(audio_path: gr_audio, model_id: str, device_pref: str, language: str = "en") -> str | Exception:
    """Returns the recognized user utterance from the input audio stream.
    Parameters:
        audio_path: gradio.Audio component.
        model_id: String of desired ASR model.
        device_pref: String of desired device for ASR processing, "cuda", "cpu", or "auto".
        language: Language code for transcription (e.g., "en", "fr", "es").
    Returns:
        hyp_raw: Recognized user utterance.
    """
    asr = get_asr_pipeline(model_id, device_pref)
    try:
        # For multilingual models, specify language and task
        # For English-only models (.en), don't pass language/task args
        if ".en" in model_id:
            result = asr(audio_path)
        else:
            result = asr(audio_path, generate_kwargs={"language": language, "task": "transcribe"})
        hyp_raw = result["text"].strip()
    except Exception as e:
        return e
    return hyp_raw

def similarity_and_diff(ref_tokens: list, hyp_tokens: list) -> (float, list[str, int, int, int]):
    """
    Returns:
        ratio: Similarity ratio (0..1).
        opcodes: List of differences between target and recognized user utterance.
    """
    sm = difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    ratio = sm.ratio()
    opcodes = sm.get_opcodes()
    return ratio, opcodes

class SentenceMatcher:
    """Class for keeping track of (target sentence, user utterance) match features."""
    def __init__(self, target_sentence, user_transcript, pass_threshold):
        self.target_sentence: str = target_sentence
        self.user_transcript: str = user_transcript
        self.pass_threshold: float = pass_threshold
        self.target_tokens: list = normalize_text(target_sentence).split()
        self.user_tokens: list = normalize_text(user_transcript).split()
        self.ratio: float
        self.alignments: list
        self.ratio, self.alignments = similarity_and_diff(self.target_tokens,
                                                          self.user_tokens)
        self.passed: bool = self.ratio >= self.pass_threshold

