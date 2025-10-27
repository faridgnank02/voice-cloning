# src/generate.py
"""
Module: generate
----------------
Handles the generation of "consent sentences" for the Voice Consent Gate demo.

This module connects to an external language model (in this case, the public
Hugging Face Space for Llama 3.2 3B Instruct) to generate natural-sounding
sentences that users can read aloud to give informed consent for voice cloning.

If the model call fails (e.g., due to rate limits or network issues),
an error is surfaced to the UI.
"""

import os
from typing import Any
from gradio_client import Client

import src.process as process
from src.prompts import get_consent_generation_prompt


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
        parts = []
        for x in result:
            s = _extract_llama_text(x)
            if s:
                parts.append(s)
        return " ".join(parts).strip()
    if isinstance(result, dict):
        for key in ("text", "response", "content", "generated_text", "message"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def gen_sentence(audio_model_name="Chatterbox"):
    """
    Always generate a sentence via the LLM. UI may still pass a 'method' arg,
    but it's ignored to keep the callback signature stable.
    """
    try:
        return gen_sentence_llm(audio_model_name=audio_model_name)
    except Exception as e:
        # Show a helpful message directly in the Target sentence box
        return f"[ERROR calling LLM] {type(e).__name__}: {e}"

# TODO: Support more than just Llama 3.2 3B Instruct
def gen_sentence_llm(
    sentence_method: str = "Llama 3.2 3B Instruct",
    audio_model_name: str = "Chatterbox",
    *
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

    Returns
    -------
    str
        A clean, human-readable consent sentence.
    """
    # Generate the full natural-language prompt that the LLM will receive
    prompt = get_consent_generation_prompt(audio_model_name)

    try:
        client = Client(LLAMA_SPACE_ID, hf_token=HF_TOKEN)
        result = client.predict(
            message=prompt,
            max_new_tokens=128,
            temperature=0.6,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.2,
            api_name=LLAMA_API_NAME,
        )

        text = _extract_llama_text(result)
        text = process.normalize_text(text, lower=False)

        if not text:
            raise ValueError("Empty response from Llama Space")

        first_line = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
        return first_line or text

    except Exception as e:
        print(f"[gen_sentence_llm] Llama Space call failed: {type(e).__name__}: {e}")
        raise