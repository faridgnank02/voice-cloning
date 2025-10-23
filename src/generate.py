import random
import re
from typing import Literal
import os

from transformers import pipeline, AutoTokenizer

import src.process as process
from src.prompts import get_consent_generation_prompt

HF_TOKEN = os.getenv("HF_TOKEN")

# ------------------- Sentence Bank (customize freely) -------------------
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

# ------------------- Model IDs -------------------
CHAT_MODEL_ID = "Qwen/Qwen2.5-1.5B-Instruct"     # changed from chat format to instruct format
INSTRUCT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"          # plain prompt

# ------------------- Helpers -------------------
def _clean_output(text: str) -> str:
    """Trim prompt echoes / role tags / quotes and keep it neat."""
    # Remove typical chat role prefixes if present
    text = re.sub(r"^\s*(assistant|system|user)\s*[:：]\s*", "", text, flags=re.I)
    # Drop surrounding quotes/backticks
    text = text.strip().strip('`"\' ')
    # Collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ------------------- Generators -------------------
def _clean(text: str) -> str:
    # Remove prompt echo if present and tidy whitespace/quotes
    text = text.strip().strip('`"\' ')
    text = re.sub(r"\s+", " ", text)
    return text

def gen_sentence_llm_chat():
    model_id = "Qwen/Qwen2.5-1.5B-Instruct"
    prompt = get_consent_generation_prompt("chatterbox")

    tok = AutoTokenizer.from_pretrained(model_id)
    gen = pipeline(
        "text-generation",
        model=model_id,
        tokenizer=tok,
        device_map="auto",
        torch_dtype="auto",
    )

    out = gen(
        prompt,
        max_new_tokens=60,
        temperature=0.6,
        repetition_penalty=1.08,
        pad_token_id=tok.eos_token_id,
    )[0]["generated_text"]

    # strip prompt echo if model returns prompt+completion
    if out.startswith(prompt):
        out = out[len(prompt):]

    return process.normalize_text(_clean(out), lower=False)


def gen_sentence_llm_instruct() -> str:
    """
    Llama Instruct (plain prompt): pass the instruction directly.
    Returns a cleaned sentence.
    """
    prompt_text = get_consent_generation_prompt("chatterbox")

    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_MODEL_ID, token=HF_TOKEN)
    generator = pipeline(
        "text-generation",
        model=INSTRUCT_MODEL_ID,
        tokenizer=tokenizer,
        device_map="auto",
        torch_dtype="auto",
        token=HF_TOKEN
    )

    out = generator(
        prompt_text,
        max_new_tokens=80,
        temperature=0.7,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
    )[0]["generated_text"]

    # Some instruct models return prompt+completion
    if out.startswith(prompt_text):
        out = out[len(prompt_text):]

    return _clean_output(out)


def gen_sentence(model_choice: Literal["qwen-instruct", "llama-instruct"]) -> str:
    """
    Switcher: call the appropriate generator by a simple string key.
    """
    if model_choice == "qwen-instruct":
        return gen_sentence_llm_chat()
    elif model_choice == "llama-instruct":
        return gen_sentence_llm_instruct()
    else:
        # Fallback to instruct to avoid prompt-echoing on unknown keys
        return gen_sentence_llm_instruct()


def gen_sentence_set() -> str:
    """Returns a sentence for the user to say using a prespecified set of options."""
    return random.choice(SENTENCE_BANK)