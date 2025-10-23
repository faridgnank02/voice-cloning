import gradio as gr
from gradio_client import Client, handle_file

import src.generate as generate
import src.process as process

# ------------------- Globals -------------------
global client


# ------------------- UI helper functions -------------------
def clear_all():
    """Reset all displayed fields."""
    return "", "", "", "", "", "", "", None,


def make_result_html(pass_threshold, passed, ratio):
    """Returns HTML summarizing results."""
    summary = (
        f"✅ Correct (≥ {int(pass_threshold * 100)}%)"
        if passed
        else f"❌ Not a match (need ≥ {int(pass_threshold * 100)}%)"
    )
    score = f"Similarity: {ratio * 100:.1f}%"
    return summary, score


def make_alignment_html(ref_tokens, hyp_tokens, alignments):
    """Returns HTML showing alignment between target and recognized user audio."""
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
    """Creates the HTML for UI based on the sentence match."""
    diff_html = make_alignment_html(
        sentence_match.target_tokens,
        sentence_match.user_tokens,
        sentence_match.alignments,
    )
    result_html, score_html = make_result_html(
        sentence_match.pass_threshold, sentence_match.passed, sentence_match.ratio
    )
    return score_html, result_html, diff_html


# ------------------- Core Check (English-only) -------------------
def get_user_transcript(audio_path: gr.Audio, target_sentence: str,
                        model_id: str, device_pref: str):
    """Runs ASR for the input audio."""
    if not target_sentence:
        return "Please generate a sentence first.", ""
    if audio_path is None:
        return (
            "Please start, record, then stop the audio recording before trying to transcribe.",
            "",
        )

    user_transcript = process.run_asr(audio_path, model_id, device_pref)

    if isinstance(user_transcript, Exception):
        return f"Transcription failed: {user_transcript}", ""
    return "", user_transcript


def transcribe_check(audio_path, target_sentence, model_id, device_pref, pass_threshold):
    """Transcribe user audio, compare with target, and generate HTML."""
    clone_audio = False

    error_msg, user_transcript = get_user_transcript(
        audio_path, target_sentence, model_id, device_pref
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

        score_html, result_html, diff_html = make_html(sentence_match)

    return user_transcript, score_html, result_html, diff_html, gr.Row(visible=clone_audio)


def clone_voice(audio_input, text_input):
    """Calls Chatterbox Space to clone the voice."""
    global client
    return client.predict(
        text_input=text_input, audio_prompt_path_input=handle_file(audio_input)
    )


# ------------------- Gradio UI -------------------
with gr.Blocks(title="Say the Sentence (English)") as demo:
    gr.Markdown(
        """
        # 🎤 Say the Sentence (English)
        1) Generate a sentence.  
        2) Record yourself reading it.  
        3) Transcribe & check your accuracy.  
        4) If matched, clone your voice to speak any sentence you enter.
        """
    )

    # --- Sentence generation section ---
    with gr.Row():
        target = gr.Textbox(
            label="Target sentence",
            interactive=False,
            placeholder="Click 'Generate sentence'"
        )

    with gr.Row():
        # 🔽 New: sentence generator model selector
        sentence_gen_model = gr.Dropdown(
            choices=["qwen-instruct", "llama-instruct"],
            value="llama-instruct",
            label="Sentence generator model",
        )

    with gr.Row():
        btn_gen = gr.Button("🎲 Generate sentence", variant="primary")
        btn_clear = gr.Button("🧹 Clear")

    # --- Recording section ---
    with gr.Row():
        consent_audio = gr.Audio(
            sources=["microphone"], type="filepath", label="Record your voice", key='consent_audio'
        )

    # --- Advanced settings ---
    with gr.Accordion("Advanced settings", open=False):
        model_id = gr.Dropdown(
            choices=[
                "openai/whisper-tiny.en",  # fastest (CPU-friendly)
                "openai/whisper-base.en",  # better accuracy
                "distil-whisper/distil-small.en",
            ],
            value="openai/whisper-tiny.en",
            label="ASR model (English only)",
        )
        device_pref = gr.Radio(
            choices=["auto", "cpu", "cuda"],
            value="auto",
            label="Device preference",
        )
        pass_threshold = gr.Slider(
            0.50, 1.00, value=0.85, step=0.01, label="Match threshold"
        )

    # --- Transcription + comparison section ---
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

    # --- Voice cloning UI (appears only on match) ---
    with gr.Row(visible=False) as tts_ui:
        @gr.render(inputs=consent_audio)
        def show_tts(audio_input):
            global client
            if audio_input:
                client = Client("ResembleAI/Chatterbox")
                with gr.Row():
                    gr.Markdown("# 🔁 Voice cloning")

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Audio input")
                        tts_audio = gr.Audio(
                            audio_input, interactive=True, type="filepath"
                        )

                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Text input")
                        tts_text = gr.Textbox(
                            "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.",
                            interactive=True,
                        )

                with gr.Row():
                    clone_btn = gr.Button("Clone!")
                    cloned_audio = gr.Audio()
                    clone_btn.click(
                        fn=clone_voice, inputs=[tts_audio, tts_text], outputs=[cloned_audio]
                    )

    # ------------------- Event wiring -------------------
    # 🎲 Generate sentence using selected LLM
    btn_gen.click(
        fn=generate.gen_sentence,
        inputs=[sentence_gen_model],
        outputs=target,
    )

    # 🧹 Clear button
    btn_clear.click(
        fn=clear_all,
        outputs=[target, user_transcript, score_html, result_html, diff_html],
    )

    # ✅ Transcribe & Check
    btn_check.click(
        fn=transcribe_check,
        inputs=[consent_audio, target, model_id, device_pref, pass_threshold],
        outputs=[user_transcript, score_html, result_html, diff_html, tts_ui],
    )

# ------------------- Run the app -------------------
if __name__ == "__main__":
    demo.launch(show_error=True)