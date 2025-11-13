# src/generate.py
"""
Multi-backend LLM generation for consent sentences.
Supports: Ollama, LMStudio, HuggingFace, Template fallback
"""

import os
import random
from typing import Any

import src.process as process
from src.prompts import get_consent_generation_prompt

# Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")  # Default to llama3:8b

LMSTUDIO_BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")

LLAMA_SPACE_ID = os.getenv("LLAMA_SPACE_ID", "huggingface-projects/llama-3.2-3B-Instruct")
LLAMA_API_NAME = "/chat"
HF_TOKEN = os.getenv("HF_TOKEN")

BACKEND_ORDER = os.getenv("LLM_BACKEND_ORDER", "ollama,template,lmstudio,huggingface").split(",")


def clean_llm_output(text: str) -> str:
    """Clean LLM output to extract only the consent sentences."""
    import re
    # First, try to extract content between explicit markers we require in the prompt.
    # This is the most reliable method to avoid any leading intro text added by models.
    marker_start = '<<<SENTENCES>>>'
    marker_end = '<<<END>>>'

    if marker_start in text and marker_end in text:
        start = text.index(marker_start) + len(marker_start)
        end = text.index(marker_end, start)
        inner = text[start:end].strip()
        # Remove any surrounding quotes and normalize spaces
        inner = re.sub(r'^\s*["\']+|["\']+\s*$', '', inner).strip()
        inner = re.sub(r'\s+', ' ', inner)
        return inner

    # Fallback: remove common introductory phrases in multiple languages
    intro_patterns = [
        r'^(here are|here is|voici|aquí están|ecco)\s+(two|2|deux|dos|due)\s+sentences?[:\s]*',
        r'^(sentence\s*1|sentence\s*2|phrase\s*1|phrase\s*2)[:\s]*',
        r'^\d+[\.\)]\s*',  # Remove numbering like "1. " or "1) "
        r'^[-•]\s*',  # Remove bullets
    ]

    lines = text.strip().splitlines()
    cleaned_lines = []

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Apply cleaning patterns
        for pattern in intro_patterns:
            line = re.sub(pattern, '', line, flags=re.IGNORECASE)

        # Remove leading/trailing quotes
        line = line.strip('"\'')

        if line:
            cleaned_lines.append(line)

    # Join all cleaned lines
    result = ' '.join(cleaned_lines)

    # Try a last-resort: extract first two sentences using punctuation as delimiter
    # Split on sentence-ending punctuation, but keep common abbreviations naive fallback
    if result:
        # Simple sentence split by periods/exclamation/question followed by space or line end
        parts = re.split(r'(?<=[\.\!\?])\s+', result)
        # Keep non-empty parts
        parts = [p.strip() for p in parts if p.strip()]
        if len(parts) >= 2:
            first_two = parts[0] + ' ' + parts[1]
            first_two = re.sub(r'\s+', ' ', first_two).strip()
            return first_two

    # Normalize spaces and return whatever cleaned result we have
    result = re.sub(r'\s+', ' ', result).strip()
    return result


def gen_sentence_template(voice_clone_model: str, language: str) -> str:
    """Static templates fallback."""
    templates = {
        "en": [
            f"I give my consent to use my voice for generating audio with {voice_clone_model}. The weather is calm and pleasant today.",
            f"I agree to allow {voice_clone_model} to clone my voice for audio generation. This recording is made clearly and freely."
        ],
        "fr": [
            f"Je donne mon consentement pour utiliser ma voix avec {voice_clone_model}. La météo est agréable aujourd'hui.",
            f"J'accepte que {voice_clone_model} clone ma voix pour générer de l'audio. Cet enregistrement est fait librement."
        ],
        "es": [
            f"Doy mi consentimiento para usar mi voz con {voice_clone_model}. El clima es agradable hoy.",
            f"Acepto que {voice_clone_model} clone mi voz para generar audio. Esta grabación se hace libremente."
        ]
    }
    choice = random.choice(templates.get(language, templates["en"]))
    # Wrap template output with the strict markers so cleaning is deterministic
    return f"<<<SENTENCES>>>\n{choice}\n<<<END>>>"


def gen_sentence_ollama(voice_clone_model: str, language: str) -> str:
    """Ollama local LLM."""
    import requests
    
    # Always generate in English first
    prompt = get_consent_generation_prompt(voice_clone_model, "en")
    
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.6, "top_p": 0.9}
        },
        timeout=30
    )
    
    if response.status_code != 200:
        raise Exception(f"Ollama error: {response.status_code}")
    
    text = response.json().get("response", "").strip()
    text = clean_llm_output(text)
    text = process.normalize_text(text, lower=False)
    
    if not text:
        raise Exception("Empty Ollama response")
    
    # If language is not English, translate
    if language != "en":
        language_names = {
            "fr": "French",
            "es": "Spanish",
            "de": "German",
            "it": "Italian",
            "pt": "Portuguese",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "ar": "Arabic"
        }
        target_lang = language_names.get(language, language)
        
        translation_prompt = f"""Translate the following English text to {target_lang}. Output ONLY the translation, nothing else.

English text: {text}

{target_lang} translation:"""
        
        translation_response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": translation_prompt,
                "stream": False,
                "options": {"temperature": 0.3, "top_p": 0.9}
            },
            timeout=30
        )
        
        if translation_response.status_code == 200:
            translated = translation_response.json().get("response", "").strip()
            translated = process.normalize_text(translated, lower=False)
            if translated:
                return translated
    
    return text


def gen_sentence_lmstudio(voice_clone_model: str, language: str) -> str:
    """LMStudio OpenAI-compatible API."""
    import requests
    
    prompt = get_consent_generation_prompt(voice_clone_model, language)
    
    response = requests.post(
        f"{LMSTUDIO_BASE_URL}/chat/completions",
        json={
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.6,
            "max_tokens": 128
        },
        timeout=30
    )
    
    if response.status_code == 200:
        text = response.json()["choices"][0]["message"]["content"].strip()
        text = clean_llm_output(text)  # Clean output
        text = process.normalize_text(text, lower=False)
        if text:
            return text
    
    raise Exception(f"LMStudio error: {response.status_code}")


def _extract_llama_text(result: Any) -> str:
    """Extract text from HuggingFace API response."""
    if isinstance(result, str):
        return result.strip()
    if isinstance(result, list):
        return " ".join([_extract_llama_text(x) for x in result if _extract_llama_text(x)]).strip()
    if isinstance(result, dict):
        for key in ("text", "response", "content", "generated_text", "message"):
            v = result.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
    return ""


def gen_sentence_huggingface(voice_clone_model: str, language: str) -> str:
    """HuggingFace Spaces API."""
    from gradio_client import Client
    
    prompt = get_consent_generation_prompt(voice_clone_model, language)
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
    text = clean_llm_output(text)  # Clean output
    text = process.normalize_text(text, lower=False)
    
    if not text:
        raise ValueError("Empty HuggingFace response")
    
    return text


def gen_sentence(consent_method="Llama 3.2 3B Instruct", voice_clone_model="Chatterbox", language="en"):
    """
    Multi-backend sentence generation with automatic fallback.
    Priority: ollama → template → lmstudio → huggingface
    """
    backends = {
        "ollama": gen_sentence_ollama,
        "lmstudio": gen_sentence_lmstudio,
        "huggingface": gen_sentence_huggingface,
        "template": gen_sentence_template
    }
    
    last_error = None
    
    for backend_name in BACKEND_ORDER:
        backend_name = backend_name.strip().lower()
        backend_func = backends.get(backend_name)
        
        if not backend_func:
            continue
        
        try:
            print(f"[LLM] Trying: {backend_name}")
            result = backend_func(voice_clone_model, language)
            print(f"[LLM] ✓ Success with: {backend_name}")
            return result
        except Exception as e:
            last_error = e
            print(f"[LLM] ✗ {backend_name} failed: {e}")
    
    return f"[ERROR] All LLM backends failed. Last: {last_error}"


# Legacy compatibility
def gen_sentence_llm(consent_method="Llama 3.2 3B Instruct", voice_clone_model="Chatterbox", language="en") -> str:
    return gen_sentence(consent_method, voice_clone_model, language)
