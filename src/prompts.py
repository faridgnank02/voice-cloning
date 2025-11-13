import random

def get_consent_generation_prompt(audio_model_name: str, language: str = "en") -> str:
    """
    Returns a text prompt instructing the model to generate a natural-sounding
    consent sentence for voice cloning with the specified model.
    Args:
        audio_model_name (str): Name of the audio model to mention in the prompt.
        language (str): Language code (e.g., "en", "fr", "es", "de", "it", "pt", "zh", "ja", "ko", "ar").
    Returns:
        str: The prompt text, with a randomized topic for the second sentence.
    """

    # Language-specific configurations
    language_config = {
        "en": {
            "name": "English",
            "consent_phrases": '"I give my consent," "I agree," or "I allow"',
            "topics": [
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
        },
        "fr": {
            "name": "French",
            "consent_phrases": '"Je donne mon consentement," "J\'accepte," ou "J\'autorise"',
            "topics": [
                "la météo",
                "les routines quotidiennes",
                "les voyages",
                "la nourriture",
                "la musique",
                "les saisons",
                "un parc ou un café",
                "la lecture",
                "une conversation agréable",
                "le travail au calme"
            ]
        },
        "es": {
            "name": "Spanish",
            "consent_phrases": '"Doy mi consentimiento," "Acepto," o "Autorizo"',
            "topics": [
                "el clima",
                "las rutinas diarias",
                "los viajes",
                "la comida",
                "la música",
                "las estaciones",
                "un parque o una cafetería",
                "la lectura",
                "una conversación agradable",
                "el trabajo tranquilo"
            ]
        },
        "de": {
            "name": "German",
            "consent_phrases": '"Ich gebe meine Zustimmung," "Ich stimme zu," oder "Ich erlaube"',
            "topics": [
                "das Wetter",
                "tägliche Routinen",
                "Reisen",
                "Essen",
                "Musik",
                "die Jahreszeiten",
                "einen Park oder ein Café",
                "das Lesen",
                "ein angenehmes Gespräch"
            ]
        },
        "it": {
            "name": "Italian",
            "consent_phrases": '"Do il mio consenso," "Accetto," o "Autorizzo"',
            "topics": [
                "il tempo",
                "le routine quotidiane",
                "i viaggi",
                "il cibo",
                "la musica",
                "le stagioni",
                "un parco o un caffè",
                "la lettura",
                "una conversazione piacevole"
            ]
        },
        "pt": {
            "name": "Portuguese",
            "consent_phrases": '"Dou meu consentimento," "Aceito," ou "Autorizo"',
            "topics": [
                "o tempo",
                "as rotinas diárias",
                "as viagens",
                "a comida",
                "a música",
                "as estações",
                "um parque ou um café",
                "a leitura",
                "uma conversa agradável"
            ]
        },
        "zh": {
            "name": "Chinese",
            "consent_phrases": '"我同意," "我接受," 或 "我授权"',
            "topics": [
                "天气",
                "日常生活",
                "旅行",
                "食物",
                "音乐",
                "季节",
                "公园或咖啡馆",
                "阅读",
                "愉快的谈话"
            ]
        },
        "ja": {
            "name": "Japanese",
            "consent_phrases": '"同意します," "承諾します," または "許可します"',
            "topics": [
                "天気",
                "日常生活",
                "旅行",
                "食べ物",
                "音楽",
                "季節",
                "公園やカフェ",
                "読書",
                "楽しい会話"
            ]
        },
        "ko": {
            "name": "Korean",
            "consent_phrases": '"동의합니다," "수락합니다," 또는 "허가합니다"',
            "topics": [
                "날씨",
                "일상",
                "여행",
                "음식",
                "음악",
                "계절",
                "공원이나 카페",
                "독서",
                "즐거운 대화"
            ]
        },
        "ar": {
            "name": "Arabic",
            "consent_phrases": '"أوافق," "أقبل," أو "أسمح"',
            "topics": [
                "الطقس",
                "الروتين اليومي",
                "السفر",
                "الطعام",
                "الموسيقى",
                "الفصول",
                "حديقة أو مقهى",
                "القراءة",
                "محادثة ممتعة"
            ]
        }
    }

    config = language_config.get(language, language_config["en"])
    topic = random.choice(config["topics"])

    return f"""
        Generate exactly two short, natural-sounding {config['name']} sentences (10-15 words each) that a person could say aloud, using everyday language.
        
        Sentence 1 (Consent sentence):
        * Clearly states informed consent to use their voice for generating synthetic audio with an AI model called {audio_model_name}.
        * Must explicitly include a consent phrase such as {config['consent_phrases']}.
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
        * Output EXACTLY two sentences in {config['name']}.
        * No numbering, no quotes, no bullet points, and no introductory text.
        * Use standard punctuation.
        
        Example format (don't copy text, just the format):
        I give my consent to use my voice for generating audio with the model {audio_model_name}. The weather is clear and calm this afternoon, and I'm speaking at an even pace.
        """
