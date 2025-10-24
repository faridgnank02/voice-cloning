import random

def get_consent_generation_prompt(audio_model_name: str) -> str:
    """
    Returns a text prompt instructing the model to generate a natural-sounding
    consent sentence for voice cloning with the specified model.

    Args:
        audio_model_name (str): Name of the audio model to mention in the prompt.

    Returns:
        str: The prompt text, with a randomized topic for the second sentence.
    """

    # Possible neutral or everyday topics to diversify phonetic variety
    topics = [
        "the weather",
        "daily routines",
        "travel or commuting",
        "food or cooking",
        "music",
        "nature or seasons",
        "time of day",
        "a calm place like a park or café",
        "light exercise or relaxation",
        "reading or learning something new",
        "a pleasant conversation with a friend",
        "observing surroundings like streets or sky",
        "working or focusing quietly"
    ]

    # Randomly choose one for this prompt instance
    topic = random.choice(topics)

    return f"""
        Generate exactly two short, natural-sounding English sentences (10-15 words each) that a person could say aloud, using everyday language.

        Sentence 1 (Consent sentence):
        * Clearly states informed consent to use their voice for generating synthetic audio with an AI model called {audio_model_name}.
        * Must explicitly include a consent phrase such as “I give my consent,” “I agree,” or “I allow.”
        * Must clearly mention the model name {audio_model_name} in the sentence.
        * Should sound fluent, polite, and natural to read aloud.
        * Should have a neutral or positive tone and be self-contained.

        Sentence 2 (Phonetic variety sentence):
        * Should not repeat the consent content.
        * Adds phonetic variety with a neutral descriptive clause, for example about {topic}.
        * Should be fluent, natural, and comfortable to read aloud.
        * Should sound polite and neutral, without emotional extremes.
        * Should include diverse vowels and consonants naturally for clear pronunciation.

        FORMAT:
        * Output EXACTLY two sentences.
        * No numbering, no quotes, no bullet points, and no introductory text.
        * Use standard punctuation.

        Example format (don’t copy text, just the format):
        I give my consent to use my voice for generating audio with the model {audio_model_name}. The weather is clear and calm this afternoon, and I’m speaking at an even pace.
        """