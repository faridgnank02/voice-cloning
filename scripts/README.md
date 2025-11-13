# Scripts Directory

This directory contains utility scripts for setting up and testing the voice cloning system.

## 📜 Scripts

### `patch_tts.py`
Patches TTS library for PyTorch 2.6+ compatibility.

**Usage:**
```bash
python scripts/patch_tts.py
```

**What it does:**
- Adds `weights_only=False` parameter to `torch.load()` calls in TTS
- Fixes compatibility issues with PyTorch 2.6+
- Must be run after installing TTS

### `patch_xtts_audio.py`
Patches XTTS to use soundfile instead of torchaudio.

**Usage:**
```bash
python scripts/patch_xtts_audio.py
```

**What it does:**
- Replaces `torchaudio.load()` with `soundfile.read()` in XTTS
- Resolves "TorchCodec required" error
- Enables audio loading without torchcodec dependency
- Must be run after installing TTS

### `test_xtts.py`
Tests XTTS v2 installation and configuration.

**Usage:**
```bash
python scripts/test_xtts.py
```

**What it tests:**
- Package imports (PyTorch, TTS, soundfile)
- Device availability (MPS/CUDA/CPU)
- XTTS v2 model loading
- Basic synthesis functionality

## 🔄 Installation Workflow

After setting up your virtual environment and installing dependencies:

```bash
# 1. Install dependencies
pip install -r requirements.txt
pip install soundfile

# 2. Apply patches
python scripts/patch_tts.py
python scripts/patch_xtts_audio.py

# 3. Test installation
python scripts/test_xtts.py
```

## ⚠️ Important Notes

- These scripts must be run **after** installing TTS
- Patches modify installed packages in your virtual environment
- If you reinstall TTS, you must reapply the patches
- Keep these scripts in version control for reproducibility
