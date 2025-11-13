# Contributions to Voice Cloning Project

## 📋 Overview

This document details the major contributions made to the Voice Consent Gate project, including the addition of complete multilingual support and the integration of a local voice cloning system optimized for Mac M1.

---

## 🌍 1. Complete Multilingual Support

### 1.1 Multilingual Sentence Generation

**Initial Problem:** The system only generated sentences in English.

**Implemented Solution:**
- **Two-step translation strategy:**
  1. Generation in English (optimal language for LLMs)
  2. Automatic translation to target language via LLM

**Technical Architecture:**

```python
# src/generate.py - gen_sentence_ollama() function

def gen_sentence_ollama(voice_clone_model, language):
    # Step 1: Generation in English
    english_prompt = get_consent_generation_prompt(voice_clone_model, "en")
    english_result = ollama.generate(model=OLLAMA_MODEL, prompt=english_prompt)
    
    # Step 2: Translation if necessary
    if language != "en":
        translation_prompt = f"Translate the following English text to {language}..."
        result = ollama.generate(model=OLLAMA_MODEL, prompt=translation_prompt)
```

**Supported Languages:** 10 languages
- English (en), French (fr), Spanish (es), German (de), Italian (it)
- Portuguese (pt), Chinese (zh), Japanese (ja), Korean (ko), Arabic (ar)

**Configuration:** `OLLAMA_MODEL = "llama3:8b"` (in `src/generate.py`)

### 1.2 Multilingual Speech Recognition

**Implementation:**
- Using **Whisper** (OpenAI) via `transformers`
- Native multilingual support without additional configuration
- Available models: `whisper-tiny`, `whisper-base`, `whisper-small`

```python
# src/process.py - run_asr() function

def run_asr(audio_path, model_id, device_pref, language):
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=model_id,
        device=device
    )
    result = asr_pipeline(
        audio_path,
        generate_kwargs={"language": language, "task": "transcribe"}
    )
```

**Optimization:** The `language` parameter improves transcription accuracy.

---

## 🎙️ 2. Local Voice Cloning with XTTS v2

### 2.1 Migration from Chatterbox to XTTS v2

**Initial Problem:**
- Dependency on external API (Chatterbox) with limited quota
- No control over latency and availability

**Solution:** Local integration of **XTTS v2** (Coqui TTS)

### 2.2 Technical Architecture

#### File: `src/tts.py`

**Singleton Pattern for the Model:**
```python
_tts_model = None  # Global cache

def get_tts_model():
    global _tts_model
    if _tts_model is not None:
        return _tts_model
    
    from TTS.api import TTS
    _tts_model = TTS(
        model_name="tts_models/multilingual/multi-dataset/xtts_v2",
        progress_bar=False,
        gpu=False  # Avoids CUDA check
    )
    
    # Manual move to MPS (Mac M1)
    if device == "mps":
        _tts_model.to(device)
    
    return _tts_model
```

**Cloning Function:**
```python
def run_tts_clone(ref_audio_path, text_to_speak, model_id, language):
    tts = get_tts_model()
    
    # Generation with voice cloning
    tts.tts_to_file(
        text=text_to_speak,
        speaker_wav=ref_audio_path,  # Reference audio
        language=language,             # Target language
        file_path=output_path
    )
    
    # Loading and normalization
    sr, wav = wavfile.read(output_path)
    wav = wav.astype(np.float32) / 32768.0  # Normalization [-1, 1]
    
    return sr, wav
```

### 2.3 Mac M1 Optimizations (Apple Silicon)

**Automatic Device Detection:**
```python
def get_device():
    if torch.backends.mps.is_available():
        return "mps"  # Metal Performance Shaders
    return "cpu"
```

**MPS Advantages:**
- GPU acceleration via Metal
- Reduced inference time (~2-3x faster than CPU)
- Optimized memory management (16GB RAM)

### 2.4 Technical Problem Resolution

#### Problem 1: CUDA Check on Mac
**Error:** `AssertionError: CUDA is not available on this machine`

**Solution:**
```python
# Loading with gpu=False then manual move
_tts_model = TTS(..., gpu=False)
if device == "mps":
    _tts_model.to(device)
```

#### Problem 2: Missing TorchCodec
**Error:** `ImportError: TorchCodec is required for load_with_torchcodec`

**Solution:** Patch XTTS to use `soundfile`
```python
# patch_xtts_audio.py
# Replacement in venv/lib/python3.10/site-packages/TTS/tts/models/xtts.py

old_code = "audio, lsr = torchaudio.load(audiopath)"
new_code = """
    import soundfile as sf
    audio_data, lsr = sf.read(audiopath)
    audio = torch.FloatTensor(audio_data)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.T  # (samples, channels) → (channels, samples)
"""
```

**Running the patch:**
```bash
python patch_xtts_audio.py
```

#### Problem 3: Version Incompatibilities
**Solution:** Downgrade specific libraries
```bash
pip install 'transformers<4.50.0' 'numpy<2.0'
```

### 2.5 XTTS v2 Multilingual Support

**Supported Languages:** 17 native languages
- en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

**Features:**
- ✅ **Voice timbre** cloning independent of language
- ✅ **Native pronunciation** in target language
- ✅ **Prosody adapted** to each language
- ✅ Ability to clone a French voice and generate in English/Spanish/etc.

---

## 🔧 3. User Interface Modifications

### 3.1 Gradio Architecture Simplification

**Problem:** `KeyError: 5` with `@gr.render` (indexing incompatibility)

**Solution:** Rewrite without `@gr.render`
```python
# Before (app.py - original version)
with gr.Row(visible=False) as tts_ui:
    @gr.render(inputs=[consent_audio, language])
    def show_tts(audio_input, selected_language):
        # Problematic dynamic rendering

# After (app.py - corrected version)
with gr.Row(visible=False) as tts_ui:
    tts_audio = gr.Audio(value=None, ...)
    tts_text = gr.Textbox(...)
    clone_btn = gr.Button(...)
    cloned_audio = gr.Audio(...)

# Event-based management
btn_check.click(...).then(
    fn=lambda audio: audio,
    inputs=[consent_audio],
    outputs=[tts_audio]
)
```

### 3.2 UI Update

**Modifications:**
- Replacement "Chatterbox" → "XTTS v2"
- Removal of Chatterbox parameters (exaggeration, cfg, seed, temperature)
- Addition of multilingual text examples
- Explanatory note on translation strategy

```python
EXAMPLE_TEXTS = {
    "en": "Now let's make my mum's favourite...",
    "fr": "Aujourd'hui il fait beau et je suis content...",
    "es": "Hoy hace buen tiempo y estoy feliz...",
    # ... 7 other languages
}
```

---

## 📦 4. Dependency Management

### 4.1 New Dependencies

**Additions to `requirements.txt`:**
```txt
TTS>=0.22.0              # Coqui TTS with XTTS v2
scipy>=1.11.0            # Audio processing
soundfile>=0.12.0        # Alternative audio backend
```

**Version constraints:**
```txt
numpy>=1.24.0,<2.0       # TTS compatibility
transformers<4.50.0      # Avoid breaking changes
torch>=2.0.0             # MPS support
```

### 4.2 Python Environment

**Compatible Versions:** Python 3.9, 3.10, 3.11

**Important Note:** Python 3.12 not supported by TTS (<3.12 required)

---

## 🚀 5. Performance and Optimization

### 5.1 Performance Metrics (Mac M1, 16GB RAM)

**XTTS v2 on MPS:**
- Initial loading time: ~5-8 seconds
- Audio generation (10s): ~2-4 seconds
- Real-time factor: ~1.1-2.3x
- Model size: ~1.87 GB

**Ollama llama3:8b:**
- Sentence generation (English): ~1-2 seconds
- Translation: ~1-2 seconds
- Model size: 4.7 GB

### 5.2 Applied Optimizations

1. **TTS model cache** (singleton pattern)
2. **Generation on MPS** instead of CPU
3. **Reduced translation temperature** (0.3) for consistency
4. **Automatic cleanup** of temporary files

---

## 📚 6. Patch Documentation

### 6.1 TTS PyTorch Patch
**File:** `patch_tts.py`
```python
# Force weights_only=False for PyTorch 2.6+
tts_io_path = "venv/.../TTS/utils/io.py"
old = 'return torch.load(f, map_location=map_location, **kwargs)'
new = 'return torch.load(f, map_location=map_location, weights_only=False, **kwargs)'
```

### 6.2 XTTS Audio Backend Patch
**File:** `patch_xtts_audio.py`
```python
# Replace torchaudio with soundfile in XTTS
xtts_file = "venv/.../TTS/tts/models/xtts.py"
# Modifies the load_audio() function to use soundfile
```

---

## 🛠️ Installation and Configuration

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3) or Intel
- **Python 3.10** (recommended) or 3.9/3.11
- **16 GB RAM** minimum recommended
- **~10 GB disk space** (for models)

### Step 1: Clone the Project

```bash
git clone <repository-url>
cd voice_cloning
```

### Step 2: Create Virtual Environment

```bash
# Use Python 3.10
python3.10 -m venv venv

# Activate environment
source venv/bin/activate

# Update pip
pip install --upgrade pip
```

### Step 3: Install Dependencies

```bash
# Install main dependencies
pip install -r requirements.txt

# Install soundfile (for XTTS)
pip install soundfile

# Verify versions
pip list | grep -E "TTS|torch|transformers|gradio"
```

**Expected versions:**
- TTS: 0.22.0
- torch: 2.8.0+ (with MPS)
- transformers: <4.50.0
- gradio: 4.44.1+

### Step 4: Install and Configure Ollama

#### 4.1 Ollama Installation

```bash
# macOS (via Homebrew)
brew install ollama

# Or download from https://ollama.ai/download
```

#### 4.2 Start Ollama Service

```bash
# Start in background
ollama serve &

# Verify service is running
curl http://localhost:11434/api/tags
```

#### 4.3 Download llama3:8b Model

```bash
# Download (approximately 4.7 GB)
ollama pull llama3:8b

# Verify installation
ollama list
```

**Expected output:**
```
NAME           ID              SIZE      MODIFIED
llama3:8b      a6990ed6be41    4.7 GB    X days ago
```

#### 4.4 Test Ollama

```bash
# Simple test
ollama run llama3:8b "Say hello in French"

# Test via API
curl http://localhost:11434/api/generate -d '{
  "model": "llama3:8b",
  "prompt": "Hello world",
  "stream": false
}'
```

### Step 5: Apply Patches

#### 5.1 TTS Patch for PyTorch 2.6+

```bash
# Apply the patch
python patch_tts.py
```

**Expected output:**
```
📝 Patching venv/lib/python3.10/site-packages/TTS/utils/io.py
✅ Patch applied successfully!
```

#### 5.2 XTTS Patch for Audio Backend

```bash
# Apply the patch
python patch_xtts_audio.py
```

**Expected output:**
```
📝 Patching venv/lib/python3.10/site-packages/TTS/tts/models/xtts.py
✅ Patch applied successfully!
```

### Step 6: Verify Installation

#### 6.1 PyTorch MPS Test

```bash
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Expected output (Mac M1):** `MPS available: True`

#### 6.2 XTTS Import Test

```bash
python -c "from TTS.api import TTS; print('✓ TTS import successful')"
```

#### 6.3 Ollama Connection Test

```bash
python -c "import ollama; print(ollama.list())"
```

### Step 7: Launch the Application

```bash
# Launch Gradio application
python app.py
```

**Expected output:**
```
* Running on local URL:  http://127.0.0.1:7860
* To create a public link, set `share=True` in `launch()`.
```

**Open in browser:** http://127.0.0.1:7860

### Step 8: First Complete Test

1. **Select a language** (e.g., French)
2. **Generate a sentence** (via llama3:8b)
3. **Record your voice** reading the sentence
4. **Verify transcription** (Whisper)
5. **Clone your voice** with new text (XTTS v2)

---

## 🐛 Troubleshooting

### Problem: "CUDA is not available"

**Cause:** TTS tries to use CUDA on Mac

**Solution:** Patches are automatically applied (`gpu=False` in `get_tts_model()`)

### Problem: "TorchCodec is required"

**Cause:** Missing default audio backend

**Solution:**
```bash
pip install soundfile
python patch_xtts_audio.py
```

### Problem: "Ollama connection refused"

**Cause:** Ollama service not started

**Solution:**
```bash
ollama serve &
ollama list  # Verify llama3:8b is present
```

### Problem: Python 3.12 incompatible

**Cause:** TTS requires Python <3.12

**Solution:**
```bash
# Install Python 3.10
brew install python@3.10

# Recreate venv
rm -rf venv
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Problem: MPS not available

**Cause:** CPU only (Intel Mac or incorrect config)

**Impact:** Reduced performance but functional

**Verification:**
```python
import torch
print(f"MPS: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")
```

---

## 📊 Modified File Structure

```
voice_cloning/
├── app.py                      # ✨ Gradio UI (XTTS integrated)
├── src/
│   ├── generate.py             # ✨ Multilingual generation
│   ├── process.py              # ✨ Multilingual ASR
│   ├── tts.py                  # ✨ XTTS v2 cloning (new)
│   ├── prompts.py              # Prompts (unchanged)
│   └── utils/
│       └── prompts.py
├── requirements.txt            # ✨ Updated dependencies
├── patch_tts.py               # ✨ PyTorch patch (new)
├── patch_xtts_audio.py        # ✨ Audio backend patch (new)
├── test_xtts.py               # ✨ XTTS tests (new)
└── CONTRIBUTIONS.md           # ✨ This document
```

**Legend:** ✨ = Modified or created file

---

## 🎯 Final Features

### ✅ Multilingual Generation
- 10 supported languages
- English generation + automatic translation
- LLM: llama3:8b (Ollama)

### ✅ Multilingual Speech Recognition
- Whisper (OpenAI)
- Native support for 100+ languages
- Optimized for 10 main languages

### ✅ Local Voice Cloning
- XTTS v2 (Coqui TTS)
- 17 native languages
- Mac M1 optimized (MPS)
- 100% local, no external API

### ✅ User Interface
- Gradio 4.44.1
- Complete integrated workflow
- Multilingual examples

---

## 📈 Project Metrics

- **Lines of code added:** ~800
- **Files created:** 4 (tts.py, patches, tests, doc)
- **Files modified:** 5 (app.py, generate.py, process.py, requirements.txt, README)
- **Languages added:** 9 (from 1 to 10)
- **External dependencies removed:** 1 (Chatterbox API)
- **Performance improved:** ~2-3x with MPS vs CPU

---

## 🙏 Acknowledgments

- **Coqui TTS** for XTTS v2
- **OpenAI** for Whisper
- **Meta** for llama3
- **Ollama** for local LLM infrastructure
- **Gradio** for user interface

---

## 📝 Version Notes

**Version:** 2.0.0 (Multilingual + Local)

**Date:** November 2025

**Compatibility:**
- macOS (M1/M2/M3 recommended)
- Python 3.9, 3.10, 3.11
- 16 GB RAM minimum

---

## 📞 Support

For any questions or problems:
1. Check the "Troubleshooting" section
2. Review logs in the terminal
3. Verify dependency versions
4. Ensure Ollama and patches are correctly applied

---

**End of contributions document**
