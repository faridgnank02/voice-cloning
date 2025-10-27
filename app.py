import gradio as gr

from gradio_client import Client, handle_file

import src.generate as generate
import src.process as process

global client

# TODO: Ideally, instead of the Client method we're using for an external voice cloning app, we use the .load() function and pass in arguments to it directly while displaying the developer's desired UI.
#chatterbox_space = gr.load("spaces/ResembleAI/Chatterbox")
# ------------------- UI printing functions -------------------
def clear_all():
    # target, user_transcript, score_html, result_html, diff_html, tts_ui
    return "", "", "", "", "", gr.Row.update(visible=False)


def make_result_html(pass_threshold, passed, ratio):
    """Returns HTML summarizing results.
    Parameters:
        pass_threshold: Minimum percentage of match between target and
        recognized user utterance that counts as passing.
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


# ------------------- Core Check (Currently English-only) -------------------
# @spaces.GPU
def get_user_transcript(audio_path: gr.Audio, target_sentence: str,
                        asr_model_id: str, device_pref: str) -> (str, str):
    """ASR for the input audio and basic validation.
    Uses the selected ASR model `asr_model_id` to recognize words in the input `audio_path`.
    Parameters:
        audio_path: Processed audio file returned from gradio Audio component.
        target_sentence: Sentence the user needs to say.
        asr_model_id: Desired ASR model.
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
    user_transcript = process.run_asr(audio_path, asr_model_id, device_pref)

    # Handles processing errors.
    if isinstance(user_transcript, Exception):
        return f"Transcription failed: {user_transcript}", ""
    return "", user_transcript


def transcribe_check(audio_path, target_sentence, asr_model_id, device_pref,
                     pass_threshold):
    """Transcribe user, calculate match to target sentence, create results HTML.
    Parameters:
        audio_path: Local path to recorded audio.
        target_sentence: Sentence the user needs to say.
        asr_model_id: Desired ASR model.
        device_pref: Preferred ASR processing device. Can be "auto", "cpu", "cuda".
    Returns:
        user_transcript: The recognized user utterance
        score_html: HTML string to display the score
        diff_html: HTML string for displaying the differences between target and user utterance
        result_html: HTML string describing the results, or an error message
        clone_audio: Bool for whether to allow audio cloning: This makes the audio cloning components visible
    """
    clone_audio = False
    # Transcribe user input
    error_msg, user_transcript = get_user_transcript(audio_path,
                                                     target_sentence,
                                                     asr_model_id,
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

    return (user_transcript, score_html, result_html, diff_html,
            gr.Row(visible=clone_audio))

def clone_voice(audio_input, text_input, exaggeration_input, cfgw_input,
                seed_num_input, temperature_input):
    global client
    # Additional specifications for Chatterbox include:
    # exaggeration_input=0.5,
    # temperature_input=0.8,
    # seed_num_input=0,z
    # cfgw_input=0.5,
    # api_name="/generate_tts_audio"
    return client.predict(text_input=text_input,
                          audio_prompt_path_input=handle_file(audio_input),
                          exaggeration_input=exaggeration_input,
                          cfgw_input=cfgw_input,
                          seed_num_input=seed_num_input,
                          temperature_input=temperature_input)


# ------------------- UI -------------------
with gr.Blocks(title="Voice Consent Gate") as demo:
    gr.Markdown("# Voice Consent Gate: Demo")
    with gr.Row():
        with gr.Column():
            with gr.Accordion(
                    label="Click for further information on this demo",
                    open=False):
                gr.Markdown("""
                    To create a basic voice cloning system with a voice consent gate, you need three parts:
                    1. A way of generating novel consent sentences for the person whose voice will be cloned – the “speaker” – to say, uniquely referencing the current consent context.
                    2. An _automatic speech recognition (ASR) system_ that recognizes the sentence conveying consent.
                    3. A _voice-cloning text-to-speech (TTS) system_ that takes as input text and the speaker's speech snippets to generate speech.
                    
                    Since some voice-cloning TTS systems can now generate speech similar to a speaker’s voice using _just one sentence_, a sentence used for consent can **also** be used for voice cloning. 
                    """)
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown(
                """# 🎤 Say the Sentence (English)"""
            )
            gr.Markdown(
            """
            ## 1) Generate a sentence.  
            ## 2) Record yourself reading it.  
            ## 3) Transcribe & check your accuracy.  
            ## 4) If matched, clone your voice to speak any sentence you enter.
            """
            )
        with gr.Column():
            consent_method = gr.Dropdown(
                label="Sentence generation method (currently limited to Llama 3.2 3B Instruct)",
                choices=["Llama 3.2 3B Instruct"],
                value="Llama 3.2 3B Instruct"
            )
            asr_model = gr.Dropdown(label="Speech recognition model (currently limited to Whisper)",
                                    choices=["openai/whisper-tiny.en",  # fastest (CPU-friendly)
                                            "openai/whisper-base.en",  # better accuracy, a bit slower
                                            "distil-whisper/distil-small.en"
                                            # optional distil English model
                                            ],
                                    value="openai/whisper-tiny.en",
                                    )
            voice_clone_model = gr.Dropdown(
                label="Voice cloning model (currently limited to Chatterbox)",
                choices=["Chatterbox", ], value="Chatterbox")
    with gr.Row():
        target = gr.Textbox(label="Target sentence", interactive=False,
                            placeholder="Click 'Generate sentence'")
    with gr.Row():
        btn_gen = gr.Button("🎲 Generate sentence", variant="primary")
        btn_clear = gr.Button("🧹 Clear")
    with gr.Row():
        consent_audio = gr.Audio(sources=["microphone"], type="filepath",
                                 label="Record your voice", key='consent_audio')
    with gr.Accordion("Advanced ASR settings", open=False):
        device_pref = gr.Radio(
            choices=["auto", "cpu", "cuda"],
            value="auto",
            label="Device preference"
        )
        # In your own code, do not provide users with the option to change this: Set it yourself.
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

    gr.Markdown("## 🔁 Voice Consent Gate (opens upon consent)")
    # TODO: Ideally this is gr.Blocks, but that seems to have a visibility-change bug.
    with gr.Row(visible=False) as tts_ui:
        # Using the render decorator so that we can access consent audio after it's recorded.
        @gr.render(inputs=consent_audio)
        def show_tts(audio_input):
            global client
            if audio_input:
                client = Client("ResembleAI/Chatterbox")
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Audio input")
                        # Prepopulating with the consent audio.
                        # Setting interactive=False keeps it from being possible to upload something else.
                        tts_audio = gr.Audio(audio_input, type="filepath", interactive=False)
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("## Text input")
                        tts_text = gr.Textbox(
                            "Now let's make my mum's favourite. So three mars bars into the pan. Then we add the tuna and just stir for a bit, just let the chocolate and fish infuse. A sprinkle of olive oil and some tomato ketchup. Now smell that. Oh boy this is going to be incredible.", interactive=True)
                with gr.Row():
                    # TODO: Ideally, these options aren't hardcoded -- e.g., using .load(), where they're imported, allowing for different options depending on the client.
                    with gr.Accordion("More options", open=False):
                        exaggeration = gr.Slider(
                            0.25, 2, step=.05,
                            label="Exaggeration (Neutral = 0.5, extreme values can be unstable)",
                            value=.5
                        )
                        cfg_weight = gr.Slider(
                            0.2, 1, step=.05, label="CFG/Pace", value=0.5
                        )
                        seed_num = gr.Number(value=0,
                                             label="Random seed (0 for random)")
                        temp = gr.Slider(0.05, 5, step=.05,
                                         label="Temperature", value=.8)
                with gr.Row():
                    clone_btn = gr.Button("Clone!")
                    cloned_audio = gr.Audio(show_download_button=True)
                    clone_btn.click(fn=clone_voice,
                                    inputs=[tts_audio, tts_text, exaggeration,
                                            cfg_weight, seed_num, temp],
                                    outputs=[cloned_audio])

    # -------- Events --------
    # Generate sentence: including model name + detailed prompt
    btn_gen.click(
        fn=generate.gen_sentence,
        inputs=[consent_method, voice_clone_model],
        outputs=target
    )

    btn_clear.click(
        fn=clear_all,
        outputs=[target, user_transcript, score_html, result_html, diff_html,
                 tts_ui]
    )

    btn_check.click(
        fn=transcribe_check,
        inputs=[consent_audio, target, asr_model, device_pref, pass_threshold],
        outputs=[user_transcript, score_html, result_html, diff_html, tts_ui]
    )


if __name__ == "__main__":
    demo.launch(show_error=True)
