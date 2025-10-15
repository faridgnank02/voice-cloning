import gradio as gr
import random
import re
import difflib
import torch
from functools import lru_cache
from transformers import pipeline

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

# ------------------- Utilities -------------------
def normalize_text(t: str) -> str:
    # English-only normalization: lowercase, keep letters/digits/' and -
    t = t.lower()
    t = re.sub(r"[^a-z0-9'\-]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def similarity_and_diff(ref: str, hyp: str):
    """Return similarity ratio (0..1) and HTML diff highlighting changes."""
    ref_tokens = ref.split()
    hyp_tokens = hyp.split()
    sm = difflib.SequenceMatcher(a=ref_tokens, b=hyp_tokens)
    ratio = sm.ratio()

    out = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            out.append(" " + " ".join(ref_tokens[i1:i2]))
        elif op == "delete":
            out.append(
                ' <span style="background:#ffe0e0;text-decoration:line-through;">'
                + " ".join(ref_tokens[i1:i2]) + "</span>"
            )
        elif op == "insert":
            out.append(
                ' <span style="background:#e0ffe0;">'
                + " ".join(hyp_tokens[j1:j2]) + "</span>"
            )
        elif op == "replace":
            out.append(
                ' <span style="background:#ffe0e0;text-decoration:line-through;">'
                + " ".join(ref_tokens[i1:i2]) + "</span>"
            )
            out.append(
                ' <span style="background:#e0ffe0;">'
                + " ".join(hyp_tokens[j1:j2]) + "</span>"
            )
    html = '<div style="line-height:1.6;font-size:1rem;">' + "".join(out).strip() + "</div>"
    return ratio, html

@lru_cache(maxsize=2)
def get_asr(model_id: str, device_preference: str):
    """Cache an ASR pipeline. device_preference: 'auto'|'cpu'|'cuda'."""
    if device_preference == "cuda" and torch.cuda.is_available():
        device = 0
    elif device_preference == "auto":
        device = 0 if torch.cuda.is_available() else -1
    else:
        device = -1
    return pipeline(
        "automatic-speech-recognition",
        model=model_id,           # use English-only Whisper models (.en)
        device=device,
        chunk_length_s=30,
        return_timestamps=False,
    )

def gen_sentence():
    return random.choice(SENTENCE_BANK)

def clear_all():
    # target, hyp_out, score_out, diff_out, summary_out
    return "", "", "", "", ""

# ------------------- Core Check (English-only) -------------------
def check_pronunciation(audio_path, target_sentence, model_id, device_pref, pass_threshold):
    if not target_sentence:
        return "", "", "", "Please generate a sentence first."

    asr = get_asr(model_id, device_pref)

    try:
        # IMPORTANT: For English-only Whisper (.en), do NOT pass language/task args.
        result = asr(audio_path)
        hyp_raw = result["text"].strip()
    except Exception as e:
        return "", "", "", f"Transcription failed: {e}"

    ref_norm = normalize_text(target_sentence)
    hyp_norm = normalize_text(hyp_raw)

    ratio, diff_html = similarity_and_diff(ref_norm, hyp_norm)
    passed = ratio >= pass_threshold

    summary = (
        f"✅ Correct (≥ {int(pass_threshold*100)}%)"
        if passed else
        f"❌ Not a match (need ≥ {int(pass_threshold*100)}%)"
    )
    score = f"Similarity: {ratio*100:.1f}%"

    return hyp_raw, score, diff_html, summary

# ------------------- UI -------------------
with gr.Blocks(title="Say the Sentence (English)") as demo:
    gr.Markdown(
        """
        # 🎤 Say the Sentence (English)
        1) Generate a sentence.  
        2) Record yourself reading it.  
        3) Transcribe & check your accuracy.  
        """
    )

    with gr.Row():
        target = gr.Textbox(label="Target sentence", interactive=False, placeholder="Click 'Generate sentence'")

    with gr.Row():
        btn_gen = gr.Button("🎲 Generate sentence", variant="primary")
        btn_clear = gr.Button("🧹 Clear")

    with gr.Row():
        audio = gr.Audio(sources=["microphone"], type="filepath", label="Record your voice")

    with gr.Accordion("Advanced settings", open=False):
        model_id = gr.Dropdown(
            choices=[
                "openai/whisper-tiny.en",        # fastest (CPU-friendly)
                "openai/whisper-base.en",        # better accuracy, a bit slower
                "distil-whisper/distil-small.en" # optional distil English model
            ],
            value="openai/whisper-tiny.en",
            label="ASR model (English only)",
        )
        device_pref = gr.Radio(
            choices=["auto", "cpu", "cuda"],
            value="auto",
            label="Device preference"
        )
        pass_threshold = gr.Slider(0.50, 1.00, value=0.85, step=0.01, label="Match threshold")

    with gr.Row():
        btn_check = gr.Button("✅ Transcribe & Check", variant="primary")

    with gr.Row():
        hyp_out = gr.Textbox(label="Transcription", interactive=False)
    with gr.Row():
        score_out = gr.Label(label="Score")
        summary_out = gr.Label(label="Result")
    diff_out = gr.HTML(label="Word-level diff (red = expected but missing / green = extra or replacement)")

    # Events
    btn_gen.click(fn=gen_sentence, outputs=target)
    btn_clear.click(fn=clear_all, outputs=[target, hyp_out, score_out, diff_out, summary_out])
    btn_check.click(
        fn=check_pronunciation,
        inputs=[audio, target, model_id, device_pref, pass_threshold],
        outputs=[hyp_out, score_out, diff_out, summary_out]
    )

if __name__ == "__main__":
    demo.launch()