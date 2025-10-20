# src/utils/prompts.py

def get_consent_generation_prompt(audio_model_name: str, short_prompt: bool = False) -> str:
    """
    Returns a text prompt instructing the model to generate a natural-sounding
    consent sentence for voice cloning with the specified model.

    Args:
        audio_model_name (str): Name of the audio model to mention in the prompt.
        short_prompt (bool): If True, returns a concise one-line prompt suitable
            for direct model input. If False (default), returns the full detailed prompt.

    Returns:
        str: The prompt text.
    """

    if short_prompt:
        return (
            f"Generate one natural, spoken-style English sentence (10–20 words) in which a person "
            f"clearly gives informed consent to use their voice for generating synthetic audio "
            f"with the model {audio_model_name}. The sentence should sound conversational, include "
            f"a clear consent phrase like 'I give my consent' or 'I agree', mention {audio_model_name} "
            f"by name, and be phonetically varied but neutral in tone. Output only the final sentence."
        )

    return f"""
        Generate a short, natural-sounding English sentence (10–20 words) that a person could say aloud 
        to clearly state their informed consent to use their voice for generating synthetic audio with 
        an AI model called {audio_model_name}.

        The sentence should:
        - Sound natural and conversational, not like legal text.
        - Explicitly include a consent phrase, such as “I give my consent,” “I agree,” or “I allow.”
        - Mention the model name ({audio_model_name}) clearly in the sentence.
        - Include a neutral descriptive clause before or after the consent phrase to add phonetic variety 
        (e.g., “The weather today is bright and calm” or “This recording is made clearly and freely.”)
        - Have a neutral or polite tone (no emotional extremes).
        - Be comfortable to read aloud and phonetically rich, covering diverse vowels and consonants naturally.
        - Be self-contained, so the full sentence can serve as an independent audio clip.

        Examples of structure to follow:
        - “The weather is clear and warm today. I give my consent to use my voice for generating audio with the model {audio_model_name}.”
        - “I give my consent to use my voice for generating audio with the model {audio_model_name}. This statement is made freely and clearly.”
        - “Good afternoon. I agree to the use of my recorded voice for audio generation with the model {audio_model_name}.”

        The output should be a single, natural sentence ready to be spoken aloud for recording purposes.
        """