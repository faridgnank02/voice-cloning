# Voice Cloning with Multilingual Consent Gate

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![Gradio](https://img.shields.io/badge/Gradio-4.44.1-orange.svg)](https://gradio.app/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A privacy-focused voice consent verification system with local voice cloning capabilities. Generate consent sentences in 10 languages, verify voice recordings, and clone voices using XTTS v2 - all running locally on your machine.

## ✨ Key Features

- 🗣️ **Multilingual Consent Gate**: Generate and verify consent in 10 languages
- 🎙️ **Local Voice Cloning**: XTTS v2 powered voice synthesis (17 languages)
- 🔒 **Privacy First**: All processing happens locally, no external APIs
- ⚡ **Mac M1 Optimized**: Hardware acceleration via Metal Performance Shaders
- 🌍 **Cross-Lingual**: Clone a voice in one language, speak in another

## 🚀 Quick Start

### Prerequisites

- macOS (M1/M2/M3 recommended) or Linux
- Python 3.10 (3.9 or 3.11 also supported)
- 16 GB RAM minimum
- ~10 GB disk space for models

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/faridgnank02/voice-cloning.git
cd voice-cloning
```

2. **Set up Python environment**
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
# Install Ollama (macOS)
brew install ollama

# Start Ollama service
ollama serve &

# Download llama3:8b model (~4.7 GB)
ollama pull llama3:8b
```

5. **Apply system patches**
```bash
# Patch TTS for PyTorch 2.6+ compatibility
python scripts/patch_tts.py

# Patch XTTS for audio backend
python scripts/patch_xtts_audio.py
```

6. **Test installation** (optional)
```bash
python scripts/test_xtts.py
```

7. **Launch the application**
```bash
python app.py
```

8. **Open in browser**: http://127.0.0.1:7860

## 📖 Documentation

- **[Full Documentation](docs/README.md)**: Complete guide with architecture details
- **[Contributions Guide](docs/CONTRIBUTIONS.md)**: Technical deep-dive and implementation details
- **[Scripts README](scripts/README.md)**: Utility scripts documentation

## 🏗️ Project Structure

```
voice-cloning/
├── app.py                  # Main Gradio application
├── src/
│   ├── generate.py         # Multilingual sentence generation
│   ├── process.py          # Speech recognition (Whisper)
│   ├── tts.py              # Voice cloning (XTTS v2)
│   ├── prompts.py          # Prompt templates
│   └── utils/
│       └── prompts.py      # Prompt utilities
├── scripts/
│   ├── patch_tts.py        # PyTorch compatibility patch
│   ├── patch_xtts_audio.py # Audio backend patch
│   ├── test_xtts.py        # Installation test
│   └── README.md           # Scripts documentation
├── docs/
│   ├── README.md           # Full documentation
│   ├── CONTRIBUTIONS.md    # Technical contributions
│   └── IDEAS.md            # Future ideas
├── assets/                 # Static assets
├── requirements.txt        # Python dependencies
└── .gitignore             # Git ignore rules
```

## 🌍 Supported Languages

**UI Languages (10):**
English, French, Spanish, German, Italian, Portuguese, Chinese, Japanese, Korean, Arabic

**XTTS v2 Native (17):**
en, es, fr, de, it, pt, pl, tr, ru, nl, cs, ar, zh-cn, ja, hu, ko, hi

## 💡 How It Works

1. **Select Language**: Choose from 10 supported languages
2. **Generate Sentence**: System creates a random consent sentence via llama3:8b
3. **Record Voice**: Speak the generated sentence
4. **Verify**: Whisper transcribes and checks similarity (≥85%)
5. **Clone Voice**: On success, use your voice to synthesize any text

## 🔧 Configuration

### Models Used

- **LLM**: llama3:8b (Ollama) - Sentence generation and translation
- **ASR**: Whisper (OpenAI) - Speech recognition
- **TTS**: XTTS v2 (Coqui) - Voice cloning

### Device Selection

System automatically detects and uses:
- **MPS** (Mac M1/M2/M3) - GPU acceleration
- **CUDA** (NVIDIA) - GPU acceleration
- **CPU** - Fallback (slower but functional)

## 🐛 Troubleshooting

### Common Issues

**"CUDA is not available"**
- Already handled by patches (`gpu=False` + manual MPS)

**"TorchCodec is required"**
```bash
pip install soundfile
python scripts/patch_xtts_audio.py
```

**"Ollama connection refused"**
```bash
ollama serve &
```

**Python 3.12 incompatible**
```bash
brew install python@3.10
# Recreate venv with Python 3.10
```

See [Full Documentation](docs/README.md) for more troubleshooting.

## 📊 Performance

**Mac M1, 16GB RAM:**
- Sentence Generation: 1-2s
- ASR Transcription: 2-4s
- Voice Cloning (10s audio): 2-4s
- Real-time factor: 1.1-2.3x

## 🤝 Contributing

Contributions welcome! See [CONTRIBUTIONS.md](docs/CONTRIBUTIONS.md) for technical details.

## 📄 License

MIT License - See LICENSE file for details

## 🙏 Acknowledgments

- [Coqui TTS](https://github.com/coqui-ai/TTS) - XTTS v2 engine
- [OpenAI Whisper](https://github.com/openai/whisper) - Speech recognition
- [Ollama](https://ollama.ai/) - Local LLM infrastructure
- [Meta](https://ai.meta.com/) - llama3 model
- [Gradio](https://gradio.app/) - Web interface

## 🔗 Links

- [GitHub Repository](https://github.com/faridgnank02/voice-cloning)
- [Full Documentation](docs/README.md)
- [Technical Guide](docs/CONTRIBUTIONS.md)

---

**Built with ❤️ for privacy-conscious voice applications**
