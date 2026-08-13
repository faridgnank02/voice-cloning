import gradio as gr
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import src.generate as generate
import src.process as process
import src.tts as tts
from voice_agent.profiles import ProfileStore, VoiceProfile
from voice_agent.storage import SQLiteConsentVerifier, SQLiteProfileRepository

GATE_IMAGE_PATH = "./assets/voice_consent_gate_50.png"


_profile_repository = SQLiteProfileRepository(str(Path("data") / "voice_profiles.sqlite3"))
_consent_verifier = SQLiteConsentVerifier(_profile_repository)
_profile_store = ProfileStore(_consent_verifier, _profile_repository)


def register_verified_profile(
    reference_audio_path: str, consent_id: str, language: str
) -> VoiceProfile:
    """Register a profile after the consent result has been independently recorded."""
    profile = VoiceProfile(
        profile_id=str(uuid4()),
        consent_id=consent_id,
        consented_at=datetime.now(timezone.utc),
        language=language,
        reference_audio_path=reference_audio_path,
    )
    _profile_store.register(profile)
    return profile

# Language-specific example texts
EXAMPLE_TEXTS = {
    "en": "Now let's make my mum's favourite. So three mars bars into the pan.",
    "fr": "Aujourd'hui il fait beau et je suis content de pouvoir tester ce système.",
    "es": "Hoy hace buen tiempo y estoy feliz de poder probar este sistema.",
    "de": "Heute ist schönes Wetter und ich freue mich.",
    "it": "Oggi è bel tempo e sono felice di poter testare questo sistema.",
    "pt": "Hoje está bom tempo e estou feliz de poder testar este sistema.",
    "zh": "今天天气很好，我很高兴能测试这个语音克隆系统。",
    "ja": "今日は良い天気で、この音声クローンシステムをテストできて嬉しいです。",
    "ko": "오늘은 날씨가 좋고 이 음성 복제 시스템을 테스트할 수 있어서 기쁩니다.",
    "ar": "اليوم الطقس جميل وأنا سعيد لأتمكن من اختبار نظام استنساخ الصوت هذا."
}

def clear_all():
    return "", "", "", "", "", gr.Row(update(visible=False)), "", ""

def make_result_html(pass_threshold, passed, ratio):
    summary = (
        f"✅ Correct (≥ {int(pass_threshold * 100)}%)" if passed
        else f"❌ Not a match (need ≥ {int(pass_threshold * 100)}%)"
    )
    score = f"Similarity: {ratio * 100:.1f}%"
    return summary, score

def make_alignment_html(ref_tokens, hyp_tokens, alignments):
    out = []
    no_match_html = ' <span style="background:#ffe0e0;text-decoration:line-through;">'
    match_html = ' <span style="background:#e0ffe0;">'
    for span in alignments:
        op, i1, i2, j1, j2 = span
        ref_string = " ".join(ref_tokens[i1:i2])
        hyp_string = " ".join(hyp_tokens[j1:j2])
        if op == "equal":
            out.append(" " + ref_string)
        elif op == "delete":
            out.append(no_match_html + ref_string + "</span>")
        elif op == "insert":
            out.append(match_html + hyp_string + "</span>")
        elif op == "replace":
            out.append(no_match_html + ref_string + "</span>")
            out.append(match_html + hyp_string + "</span>")
    html = '<div style="line-height:1.6;font-size:1rem;">' + "".join(out).strip() + "</div>"
    return html

def make_html(sentence_match):
    diff_html = make_alignment_html(
        sentence_match.target_tokens,
        sentence_match.user_tokens,
        sentence_match.alignments
    )
    result_html, score_html = make_result_html(
        sentence_match.pass_threshold,
        sentence_match.passed,
        sentence_match.ratio
    )
    return score_html, result_html, diff_html

def get_user_transcript(audio_path, target_sentence, asr_model_id, device_pref, language):
    if not target_sentence:
        return "Please generate a sentence first.", ""
    if audio_path is None:
        return "Please start, record, then stop the audio recording before trying to transcribe.", ""
    
    user_transcript = process.run_asr(audio_path, asr_model_id, device_pref, language)
    
    if isinstance(user_transcript, Exception):
        return f"Transcription failed: {user_transcript}", ""
    return "", user_transcript

def transcribe_check(audio_path, target_sentence, asr_model_id, device_pref, pass_threshold, language):
    clone_audio = False
    profile_id = ""
    profile_status = ""
    error_msg, user_transcript = get_user_transcript(
        audio_path, target_sentence, asr_model_id, device_pref, language
    )
    
    if error_msg:
        score_html = ""
        diff_html = ""
        result_html = error_msg
    else:
        sentence_match = process.SentenceMatcher(
            target_sentence, user_transcript, pass_threshold
        )
        if sentence_match.passed:
            clone_audio = True
            consent_id = str(uuid4())
            _profile_repository.record_consent(consent_id, language, datetime.now(timezone.utc))
            try:
                profile = register_verified_profile(audio_path, consent_id, language)
                profile_id = profile.profile_id
                profile_status = "Voice profile registered and ready to use."
            except Exception:
                clone_audio = False
                profile_status = "Consent matched, but voice profile registration failed."
        score_html, result_html, diff_html = make_html(sentence_match)
    
    return (
        user_transcript,
        score_html,
        result_html,
        diff_html,
        gr.Row(visible=clone_audio),
        EXAMPLE_TEXTS.get(language, EXAMPLE_TEXTS["en"]),
        profile_id,
        profile_status,
    )

def clone_voice_wrapper(profile_id, text, language):
    """Call XTTS v2 for voice cloning using transformers pipeline"""
    if not profile_id:
        return None

    try:
        profile = _profile_store.assert_usable(profile_id, language).profile
    except Exception:
        return None
    
    result = tts.run_tts_clone(
        ref_audio_path=profile.reference_audio_path,
        text_to_speak=text,
        model_id="coqui/XTTS-v2",
        language=language
    )
    
    # Check if result is an Exception
    if isinstance(result, Exception):
        print(f"Cloning error: {result}")
        return None
    
    sr, wav = result
    return (sr, wav)

# ------------------- UI -------------------
with gr.Blocks(title="Voice Consent Gate") as demo:
    gr.Markdown("# Voice Consent Gate: Demo")
    
    with gr.Row():
        with gr.Column():
            gr.Image(GATE_IMAGE_PATH, interactive=False, show_download_button=False)
        with gr.Column():
            with gr.Accordion(label="Click for further information on this demo", open=False):
                gr.Markdown("""
                To create a basic voice cloning system with a voice consent gate, you need three parts:
                1. A way of generating novel consent sentences for the person whose voice will be cloned.
                2. An _automatic speech recognition (ASR) system_ that recognizes the sentence conveying consent.
                3. A _voice-cloning text-to-speech (TTS) system_ that takes as input text and the speaker's speech.
                """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("# 🎤 Say the Sentence (Multilingual)")
            gr.Markdown("""
            ## 1) Select your language.
            ## 2) Generate a sentence.  
            ## 3) Record yourself reading it.  
            ## 4) Transcribe & check your accuracy.  
            ## 5) If matched, clone your voice.
            """)
        with gr.Column():
            consent_method = gr.Dropdown(
                label="Sentence generation method",
                choices=["Llama 3.2 3B Instruct"],
                value="Llama 3.2 3B Instruct"
            )
            language = gr.Dropdown(
                label="Language / Langue",
                choices=[
                    ("English", "en"), ("Français", "fr"), ("Español", "es"),
                    ("Deutsch", "de"), ("Italiano", "it"), ("Português", "pt"),
                    ("中文", "zh"), ("日本語", "ja"), ("한국어", "ko"), ("العربية", "ar"),
                ],
                value="en",
                info="Select language for speech recognition and generation"
            )
            asr_model = gr.Dropdown(
                label="Speech recognition model",
                choices=[
                    "openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small",
                    "openai/whisper-tiny.en", "openai/whisper-base.en"
                ],
                value="openai/whisper-base"
            )
            voice_clone_model = gr.Dropdown(
                label="Voice cloning model",
                choices=["XTTS v2"], value="XTTS v2"
            )
    
    with gr.Row():
        target = gr.Textbox(
            label="Target sentence",
            interactive=False,
            placeholder="Click 'Generate sentence'"
        )
    
    with gr.Row():
        btn_gen = gr.Button("🎲 Generate sentence", variant="primary")
        btn_clear = gr.Button("🧹 Clear")
    
    with gr.Row():
        consent_audio = gr.Audio(
            sources=["microphone"],
            type="filepath",
            label="Record your voice"
        )
    
    with gr.Accordion("Advanced ASR settings", open=False):
        device_pref = gr.Radio(
            choices=["auto", "cpu", "cuda"],
            value="auto",
            label="Device preference"
        )
        pass_threshold = gr.Slider(
            0.50, 1.00, value=0.85, step=0.01,
            label="Match threshold"
        )
    
    with gr.Row():
        btn_check = gr.Button("✅ Transcribe & Check", variant="primary")
    
    with gr.Row():
        user_transcript = gr.Textbox(label="Transcription", interactive=False)
    
    with gr.Row():
        score_html = gr.Label(label="Score")
        result_html = gr.Label(label="Result")
    
    diff_html = gr.HTML(
        label="Word-level diff (red = expected but missing / green = extra or replacement)"
    )

    profile_id = gr.Textbox(label="Voice profile ID", interactive=False)
    profile_status = gr.Textbox(label="Profile status", interactive=False)
    
    gr.Markdown("## 🔁 Voice Consent Gate (opens upon consent)")
    gr.Markdown("⚠️ **Note:** Sentences are generated in English then translated using llama3:8b. Voice cloning uses XTTS v2 (local, Mac M1 optimized).")
    
    with gr.Row(visible=False) as tts_ui:
        with gr.Column():
            gr.Markdown("## 🎤 Verified Reference Voice")
            gr.Markdown("Your reference audio remains on the server.")
        
        with gr.Column():
            gr.Markdown("## 📝 Text to Clone")
            tts_text = gr.Textbox(
                value=EXAMPLE_TEXTS["en"],
                interactive=True,
                label="Enter text to synthesize",
                lines=3
            )
        
        with gr.Column():
            gr.Markdown("## 🔊 Cloned Output")
            clone_btn = gr.Button("🎙️ Clone Voice (XTTS v2)", variant="primary")
            cloned_audio = gr.Audio(
                show_download_button=True,
                label="Cloned voice output"
            )
    
    # -------- Events --------
    btn_gen.click(
        fn=generate.gen_sentence,
        inputs=[consent_method, voice_clone_model, language],
        outputs=target
    )
    
    btn_clear.click(
        fn=clear_all,
        outputs=[
            target,
            user_transcript,
            score_html,
            result_html,
            diff_html,
            tts_ui,
            profile_id,
            profile_status,
        ]
    )
    
    btn_check.click(
        fn=transcribe_check,
        inputs=[consent_audio, target, asr_model, device_pref, pass_threshold, language],
        outputs=[
            user_transcript,
            score_html,
            result_html,
            diff_html,
            tts_ui,
            tts_text,
            profile_id,
            profile_status,
        ]
    ).then(
        fn=lambda audio: audio,
        inputs=[consent_audio],
        outputs=[tts_audio]
    )
    
    clone_btn.click(
        fn=clone_voice_wrapper,
        inputs=[profile_id, tts_text, language],
        outputs=[cloned_audio]
    )

if __name__ == "__main__":
    demo.launch(show_error=True)
