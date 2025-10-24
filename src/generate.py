# src/generate.py
"""
Module: generate
----------------
Handles the generation of "consent sentences" for the Voice Consent Gate demo.

This module connects to an external language model (in this case, the public
Hugging Face Space for Llama 3.2 3B Instruct) to generate natural-sounding
sentences that users can read aloud to give informed consent for voice cloning.

If the model call fails (e.g., due to rate limits or network issues),
a fallback sentence is chosen from a small built-in sentence bank.

Functions:
    - _extract_llama_text(): Normalize the API output from the Llama demo.
    - gen_sentence_llm(): Generate a consent sentence from the Llama model Space.
    - gen_sentence_set(): Select a random prewritten sentence (for fallback/testing).
"""

import os
import random
from typing import Any
from gradio_client import Client

import src.process as process
from src.prompts import get_consent_generation_prompt


# ------------------- Sentence Bank (unchanged) -------------------
SENTENCE_BANK = [
    "The quick brown fox jumps over the lazy dog.",
    "I promise to speak clearly and at a steady pace.",
    "Open source makes AI more transparent and inclusive.",
    "Hugging Face Spaces make demos easy to share.",
    "Today the weather in Berlin is pleasantly cool.",
    "Privacy and transparency should go hand in hand.",
    "Please generate a new sentence for me to read.",
    "Machine learning can amplify or reduce inequality.",
    "Responsible AI requires participation from everyone.",
    "This microphone test checks my pronunciation accuracy.",
]


# ------------------- Model / Space Configuration -------------------
# The demo connects to the Llama 3.2 3B Instruct Space on Hugging Face.
# You can override these defaults by setting environment variables in your Space.
LLAMA_SPACE_ID = os.getenv(
    "LLAMA_SPACE_ID", "huggingface-projects/llama-3.2-3B-Instruct"
)
LLAMA_API_NAME = "/chat"  # The Space exposes a single /chat endpoint.
HF_TOKEN = os.getenv("HF_TOKEN")  # Optional; not required for public Spaces.


def _extract_llama_text(result: Any) -> str:
    """
    Normalize the API response from the Llama 3.2 3B demo Space into plain text.

    The Space’s `/chat` endpoint may return different shapes depending on how
    the Gradio app is structured — sometimes a string, other times a dictionary
    or list. This function recursively traverses and extracts the first
    meaningful text string it finds.

    Parameters
    ----------
    result : Any
        The raw output returned by `client.predict()`.

    Returns
    -------
    str
        Cleaned text output (may be empty string if extraction fails).
    """
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, (int, float, bool)):
        return str(result)
    if isinstance(result, list):
        # If multiple segments are returned (e.g., multiple sentences),
        # join them into one string.
        parts = []
        for x in result:
            s = _extract_llama_text(x)
            if s:
                parts.append(s)
        return " ".join(parts).strip()
    if isinstance(result, dict):
        # Common key names used in Gradio JSON responses
        for key in ("text", "response", "content", "generated_text", "message"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def gen_sentence(sentence_method="Pre-written", audio_model_name="Chatterbox"):
    # chatterbox model name, detailed prompt (short_prompt=False)
    if sentence_method == "Pre-written":
        return gen_sentence_set()
    else:
        try:
            return gen_sentence_llm(sentence_method,
                audio_model_name,
                fallback_on_error=False  # ← show errors during testing
            )
        except Exception as e:
            # Show a helpful message directly in the Target sentence box
            return f"[ERROR calling LLM] {type(e).__name__}: {e}"

# TODO: Support more than just Llama 3.2 3B Instruct
def gen_sentence_llm(sentence_method="Llama 3.2 3B Instruct", audio_model_name: str = "Chatterbox", *, fallback_on_error: bool = False  # Set True for production to avoid crashes
) -> str:
    """
    Generate a consent sentence using the Llama 3.2 3B Instruct demo Space.

    This function constructs a prompt describing the linguistic and ethical
    requirements for a consent sentence (via `get_consent_generation_prompt`)
    and sends it to the Llama demo hosted on Hugging Face Spaces.

    The response is normalized into a single English sentence suitable
    for reading aloud.

    Parameters
    ----------
    audio_model_name : str, optional
        The name of the voice-cloning model to mention in the sentence.
        Defaults to "Chatterbox".
    fallback_on_error : bool, optional
        If True, return a random fallback sentence instead of raising
        an error when the Space call fails. Default is False for debugging.

    Returns
    -------
    str
        A clean, human-readable consent sentence.

    Raises
    ------
    Exception
        Re-raises the underlying error if `fallback_on_error` is False.
    """
    # Generate the full natural-language prompt that the LLM will receive
    prompt = get_consent_generation_prompt(audio_model_name)

    try:
        # Initialize Gradio client for the Llama demo Space
        client = Client(LLAMA_SPACE_ID, hf_token=HF_TOKEN)

        # The Llama demo exposes a simple /chat endpoint with standard decoding params
        result = client.predict(
            message=prompt,
            max_new_tokens=128,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            api_name=LLAMA_API_NAME,
        )

        # Normalize and clean up model output
        text = _extract_llama_text(result)
        text = process.normalize_text(text, lower=False)

        # Handle empty or malformed outputs
        if not text:
            raise ValueError("Empty response from Llama Space")

        # In case the model produces multiple lines or options, pick the first full sentence
        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return first_line or text

    except Exception as e:
        print(f"[gen_sentence_llm] Llama Space call failed: {type(e).__name__}: {e}")
        if fallback_on_error:
            # If fallback is enabled, use a predefined sentence instead
            return random.choice(SENTENCE_BANK)
        # Otherwise propagate the exception so the UI displays it
        raise


def gen_sentence_set() -> str:
    """
    Return a sentence from a predefined static list.

    This is used as a simple fallback generator when model-based
    generation is unavailable or for testing the ASR pipeline
    without network access.

    Returns
    -------
    str
        A single English sentence from the fallback bank.
    """
    return random.choice(SENTENCE_BANK)
