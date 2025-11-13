# Voice Consent Gate with Multilingual Voice Cloning

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.1-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A multilingual voice consent verification system with local voice cloning capabilities, powered by XTTS v2 and Ollama.

> **Original Project**: This project is based on the [Voice Consent Gate](https://huggingface.co/blog/voice-consent-gate) by Hugging Face. This version adds multilingual support (10 languages), local XTTS v2 voice cloning (17 languages), Mac M1 MPS optimization, and removes dependency on external APIs.

## 🌟 Features

### 🗣️ Voice Consent Gate
- **Multilingual Sentence Generation**: Generate consent sentences in 10 languages using llama3:8b
- **High-Accuracy Speech Recognition**: Whisper-based ASR with multilingual support
- **Similarity Verification**: 85% threshold matching between generated and spoken text
- **Privacy-First**: All processing happens locally, no data leaves your machine

### 🎙️ Voice Cloning
- **Local XTTS v2 Integration**: Clone any voice with high fidelity
- **17 Language Support**: Native pronunciation in multiple languages
- **Mac M1 Optimized**: Hardware acceleration via Metal Performance Shaders (MPS)
- **Cross-Lingual Cloning**: Clone a voice in one language, generate speech in another

### 🌍 Supported Languages

**UI Languages (10):**
- English, French, Spanish, German, Italian
- Portuguese, Chinese, Japanese, Korean, Arabic

**XTTS v2 Native Languages (17):**
- en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

## � Quick Start

### Prerequisites

- macOS with Apple Silicon (M1/M2/M3) or Intel
- Python 3.10 (recommended) or 3.9/3.11
- 16 GB RAM minimum
- ~10 GB disk space for models

### Installation

1. **Clone the repository**
```bash
git clone <repository-url>
cd voice_cloning
```

2. **Create virtual environment**
```bash
python3.10 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
pip install soundfile
```

4. **Install and configure Ollama**
```bash
# Install Ollama
brew install ollama

# Start Ollama service
ollama serve &

# Download llama3:8b model (~4.7 GB)
ollama pull llama3:8b
```

5. **Apply patches**
```bash
# Patch TTS for PyTorch 2.6+ compatibility
python patch_tts.py

# Patch XTTS for audio backend
python patch_xtts_audio.py
```

6. **Launch the application**
```bash
python app.py
```

7. **Open in browser**
```
http://127.0.0.1:7860
```

## 📖 How It Works

### Consent Gate Workflow

1. **Select Language**: Choose from 10 supported languages
2. **Generate Sentence**: System generates a random consent sentence via llama3:8b
3. **Record Voice**: Speak the generated sentence
4. **Verify**: Whisper transcribes and checks similarity (≥85% required)
5. **Clone Voice**: On successful verification, unlock voice cloning features

### Voice Cloning Process

1. **Reference Audio**: Use the recorded consent audio as voice reference
2. **Input Text**: Enter any text in the selected language
3. **Generate**: XTTS v2 synthesizes speech with cloned voice characteristics
4. **Download**: Export the generated audio

## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────┐
│           Gradio Web Interface                  │
└───────────────┬─────────────────────────────────┘
                │
    ┌───────────┴───────────┐
    │                       │
┌───▼────────┐      ┌──────▼──────┐
│  Ollama    │      │   Whisper   │
│ (llama3:8b)│      │  (ASR)      │
│            │      │             │
│ - Generate │      │ - Transcribe│
│ - Translate│      │ - Verify    │
└────────────┘      └─────────────┘
                            │
                    ┌───────▼────────┐
                    │   XTTS v2      │
                    │  (Coqui TTS)   │
                    │                │
                    │ - Voice Clone  │
                    │ - MPS Accel.   │
                    └────────────────┘
```

### Key Technologies

- **Gradio 4.44.1**: Web interface
- **Ollama + llama3:8b**: LLM for generation and translation
- **Whisper**: Automatic speech recognition
- **XTTS v2 (Coqui TTS 0.22.0)**: Voice cloning engine
- **PyTorch 2.9.0**: ML framework with MPS support
- **Transformers <4.50.0**: Model hub access

## 🔧 Configuration

### Environment Variables

No environment variables required - all models run locally.

### Model Configuration

**Ollama Model** (in `src/generate.py`):
```python
OLLAMA_MODEL = "llama3:8b"
```

**Whisper Model** (in UI):
- Options: `whisper-tiny`, `whisper-base`, `whisper-small`
- Default: `whisper-base`

**XTTS v2**:
- Model: `tts_models/multilingual/multi-dataset/xtts_v2`
- Auto-downloads on first use (~1.87 GB)

### Device Selection

System automatically detects and uses:
1. **MPS** (Mac M1/M2/M3) - GPU acceleration
2. **CPU** (fallback) - Slower but functional

## 📊 Performance

### Benchmarks (Mac M1, 16GB RAM)

| Operation | Time | Notes |
|-----------|------|-------|
| Sentence Generation | 1-2s | English via llama3:8b |
| Translation | 1-2s | Target language |
| ASR Transcription | 2-4s | Depends on audio length |
| XTTS v2 Loading | 5-8s | First time only (cached) |
| Voice Cloning (10s) | 2-4s | Real-time factor: 1.1-2.3x |

### Resource Usage

- **RAM**: ~4-6 GB during operation
- **Disk**: ~10 GB (models cached)
- **GPU**: MPS utilization ~60-80% during synthesis

## 🛠️ Development

### Project Structure

```
voice_cloning/
├── app.py                    # Main Gradio application
├── src/
│   ├── generate.py           # Multilingual generation
│   ├── process.py            # ASR processing
│   ├── tts.py                # XTTS v2 cloning
│   ├── prompts.py            # Prompt templates
│   └── utils/
│       └── prompts.py
├── requirements.txt          # Python dependencies
├── patch_tts.py             # PyTorch 2.6+ patch
├── patch_xtts_audio.py      # Audio backend patch
├── test_xtts.py             # XTTS tests
└── docs/
    ├── README.md            # This file
    ├── CONTRIBUTIONS.md     # Technical contributions
    └── IDEAS.md             # Future ideas
```

### Key Modules

**`src/generate.py`**
- `gen_sentence_ollama()`: Generates and translates sentences
- Two-step process: English generation → Translation

**`src/process.py`**
- `run_asr()`: Whisper-based transcription
- Similarity checking with threshold

**`src/tts.py`**
- `get_tts_model()`: Singleton XTTS v2 loader
- `run_tts_clone()`: Voice cloning with language support

**`app.py`**
- Gradio interface with consent gate workflow
- Event handling for UI interactions

## 🐛 Troubleshooting

### Common Issues

#### "CUDA is not available"
**Cause**: TTS tries to use CUDA on Mac  
**Solution**: Patches automatically handle this (`gpu=False` + manual MPS transfer)

#### "TorchCodec is required"
**Cause**: Missing audio backend  
**Solution**: 
```bash
pip install soundfile
python patch_xtts_audio.py
```

#### "Ollama connection refused"
**Cause**: Ollama service not running  
**Solution**:
```bash
ollama serve &
ollama list  # Verify llama3:8b is present
```

#### Python 3.12 Incompatible
**Cause**: TTS requires Python <3.12  
**Solution**:
```bash
brew install python@3.10
rm -rf venv
/opt/homebrew/bin/python3.10 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

#### MPS Not Available
**Cause**: CPU-only mode (Intel Mac)  
**Impact**: Slower but functional  
**Verify**:
```python
import torch
print(f"MPS available: {torch.backends.mps.is_available()}")
```

### Debug Mode

Enable verbose logging:
```python
# In app.py or src/tts.py
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🤝 Contributing

Contributions are welcome! Please see [CONTRIBUTIONS.md](CONTRIBUTIONS.md) for detailed information about the project architecture and recent improvements.

### Areas for Contribution

- Additional language support
- Voice quality improvements
- Performance optimizations
- UI/UX enhancements
- Documentation improvements

## 📄 License

MIT License - See [LICENSE](LICENSE) file for details

## 🙏 Acknowledgments

- **[Hugging Face](https://huggingface.co/blog/voice-consent-gate)** - Original Voice Consent Gate project
- **Coqui TTS** for XTTS v2 voice cloning engine
- **OpenAI** for Whisper ASR model
- **Meta** for llama3 language model
- **Ollama** for local LLM infrastructure
- **Gradio** for the web interface framework

## 🔗 Links

- [Original Project](https://huggingface.co/blog/voice-consent-gate) - Hugging Face Voice Consent Gate
- [Ollama](https://ollama.ai/)
- [Coqui TTS](https://github.com/coqui-ai/TTS)
- [Gradio](https://gradio.app/)
- [Whisper](https://github.com/openai/whisper)
- [llama3](https://ollama.ai/library/llama3)

---
