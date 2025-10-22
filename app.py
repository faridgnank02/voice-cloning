import gradio as gr
# import spaces

import src.generate as generate
import src.process as process

chatterbox_space = gr.load("spaces/ResembleAI/Chatterbox")

# ------------------- UI printing functions -------------------
def clear_all():
    # target, user_transcript, score_html, diff_html, result_html,
    # tts_text, clone_status, tts_audio
    return "", "", "", "", "", "", "", None


def make_result_html(pass_threshold, passed, ratio):
    """Returns HTML summarizing results.
    Parameters:
        pass_threshold: Minimum percentage of match between target and recognized user utterance that counts as passing.
        passed: Whether the recognized user utterance is >= `pass_threshold`.
        ratio: Sequence match ratio.
    """
    summary = (
        f"✅ Correct (≥ {int(pass_threshold * 100)}%)"
        if passed else
        f"❌ Not a match (need ≥ {int(pass_threshold * 100)}%)"
    )
    score = f"Similarity: {ratio * 100:.1f}%"
    return summary, score


def make_alignment_html(ref_tokens, hyp_tokens, alignments):
    """Returns HTML showing alignment between the target and recognized user audio.
    Parameters:
        ref_tokens: Target sentence for the user to say, tokenized.
        hyp_tokens: Recognized utterance from the user, tokenized.
        alignments: Tuples of alignment pattern (equal, delete, insert) and corresponding indices in `hyp_tokens`.
    """
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
    html = '<div style="line-height:1.6;font-size:1rem;">' + "".join(
        out).strip() + "</div>"
    return html


def make_html(sentence_match):
    """Creates the HTML written out to the UI based on the results.
    Parameters:
        sentence_match: Class that stores the features of the target - user utterance alignment
    Returns:
        diff_html: An HTML string showing how the target sentence and recognized user utterance matches.
        result_html: An HTML string summarizing the results of the match between target and user utterance.
    """
    diff_html = make_alignment_html(sentence_match.target_tokens,
                                    sentence_match.user_tokens,
                                    sentence_match.alignments)
    result_html, score_html = make_result_html(sentence_match.pass_threshold,
                                               sentence_match.passed,
                                               sentence_match.ratio)

    return score_html, result_html, diff_html


# ------------------- Core Check (English-only) -------------------
# @spaces.GPU
def get_user_transcript(audio_path: gr.Audio, target_sentence: str,
                        model_id: str, device_pref: str) -> (str, str):
    """ASR for the input audio and basic validation.
    Uses the selected ASR model `model_id` to recognize words in the input `audio_path`.
    Parameters:
        audio_path: Processed audio file returned from gradio Audio component.
        target_sentence: Sentence the user needs to say.
        model_id: Desired ASR model.
        device_pref: Preferred ASR processing device. Can be "auto", "cpu", "cuda".
    Returns:
        error_msg: If there's an error, a string describing what happened.
        user_transcript: The recognized user utterance.
    """
    # Handles user interaction errors.
    if not target_sentence:
        return "Please generate a sentence first.", ""
    # TODO: Automatically stop the recording if someone presses the Transcribe & Check button.
    if audio_path is None:
        return "Please start, record, then stop the audio recording before trying to transcribe.", ""

    # Runs the automatic speech recognition
    user_transcript = process.run_asr(audio_path, model_id, device_pref)

    # Handles processing errors.
    if isinstance(user_transcript, Exception):
        return f"Transcription failed: {user_transcript}", ""
    return "", user_transcript


def transcribe_check(audio_path, target_sentence, model_id, device_pref,
                     pass_threshold):
    """Transcribe user, calculate match to target sentence, create results HTML.
    Parameters:
        audio_path: Local path to recorded audio.
        target_sentence: Sentence the user needs to say.
        model_id: Desired ASR model.
        device_pref: Preferred ASR processing device. Can be "auto", "cpu", "cuda".
    Returns:
        user_transcript: The recognized user utterance
        score_html: HTML string to display the score
        diff_html: HTML string for displaying the differences between target and user utterance
        result_html: HTML string describing the results, or an error message
        clone_audio: Bool for whether to allow audio cloning: This makes the audio cloning component visible
    """
    clone_audio = False
    # Transcribe user input
    error_msg, user_transcript = get_user_transcript(audio_path,
                                                     target_sentence, model_id,
                                                     device_pref)
    if error_msg:
        score_html = ""
        diff_html = ""
        result_html = error_msg
    else:
        # Calculate match details between the target and recognized user input
        sentence_match = process.SentenceMatcher(target_sentence,
                                                 user_transcript,
                                                 pass_threshold)
        if sentence_match.passed:
            clone_audio = True
        # Create the output to print out
        score_html, result_html, diff_html = make_html(sentence_match)

    return user_transcript, score_html, result_html, diff_html, gr.Row(visible=clone_audio)


# ------------------- UI -------------------
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

    with gr.Row():
        target = gr.Textbox(label="Target sentence", interactive=False,
                            placeholder="Click 'Generate sentence'")

    with gr.Row():
        btn_gen = gr.Button("🎲 Generate sentence", variant="primary")
        btn_clear = gr.Button("🧹 Clear")

    with gr.Row():
        audio = gr.Audio(sources=["microphone"], type="filepath",
                         label="Record your voice")

    with gr.Accordion("Advanced settings", open=False):
        model_id = gr.Dropdown(
            choices=[
                "openai/whisper-tiny.en",  # fastest (CPU-friendly)
                "openai/whisper-base.en",  # better accuracy, a bit slower
                "distil-whisper/distil-small.en"  # optional distil English model
                "distil-whisper/distil-small.en",
            ],
            value="openai/whisper-tiny.en",
            label="ASR model (English only)",
        )
        device_pref = gr.Radio(
            choices=["auto", "cpu", "cuda"],
            value="auto",
            label="Device preference"
        )
        pass_threshold = gr.Slider(0.50, 1.00, value=0.85, step=0.01,
                                   label="Match threshold")

    with gr.Row():
        btn_check = gr.Button("✅ Transcribe & Check", variant="primary")
    with gr.Row():
        user_transcript = gr.Textbox(label="Transcription", interactive=False)
    with gr.Row():
        score_html = gr.Label(label="Score")
        result_html = gr.Label(label="Result")
    diff_html = gr.HTML(
        label="Word-level diff (red = expected but missing / green = extra or replacement)")

    with gr.Row(visible=False) as tts_ui:
        with gr.Row():
            gr.Markdown("## 🔁 Voice cloning (gated)")
        chatterbox_space.render()

    # -------- Events --------
    # Use pre-specified sentence bank by default
    btn_gen.click(fn=generate.gen_sentence_set, outputs=target)
    # Or use LLM generation:
    # btn_gen.click(fn=generate.gen_sentence_llm, outputs=target)

    btn_clear.click(
        fn=clear_all,
        outputs=[target, user_transcript, score_html, result_html, diff_html,]
#                 tts_text, clone_status, tts_audio]
    )

    btn_check.click(
        fn=transcribe_check,
        inputs=[audio, target, model_id, device_pref, pass_threshold],
        outputs=[user_transcript, score_html, result_html, diff_html, tts_ui]
    )


if __name__ == "__main__":
    demo.launch()
