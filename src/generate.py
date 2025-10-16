import random

from transformers import pipeline, AutoTokenizer

import src.process as process

# You can choose to use either:
# (1) a list of pre-specified sentences, in SENTENCE_BANK
# (2) an LLM-generated sentence.
# SENTENCE_BANK is used in the `gen_sentence_set` function.
# LLM generation is used in the `gen_sentence_llm` function.

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


def gen_sentence_llm():
    """Generates a sentence using an LLM.
    Returns:
        Normalized text string to display in the UI.
    """
    prompt = ""
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    generator = pipeline('text-generation', model='gpt2')
    result = generator(prompt, stop_strings=[".", ], num_return_sequences=1,
                       tokenizer=tokenizer, pad_token_id=tokenizer.eos_token_id)
    display_text = process.normalize_text(result[0]["generated_text"],
                                          lower=False)
    return display_text


def gen_sentence_set():
    """Returns a sentence for the user to say using a prespecified set of options."""
    return random.choice(SENTENCE_BANK)
