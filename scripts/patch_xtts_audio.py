"""
Patch XTTS to use soundfile backend instead of torchcodec
"""
import os

venv_path = "venv/lib/python3.10/site-packages"
xtts_file = os.path.join(venv_path, "TTS/tts/models/xtts.py")

print(f"📝 Patching {xtts_file}")

with open(xtts_file, 'r') as f:
    content = f.read()

# Replace torchaudio.load with soundfile loading
old_code = '''    audio, lsr = torchaudio.load(audiopath)'''

new_code = '''    # Use soundfile instead of torchaudio to avoid torchcodec requirement
    import soundfile as sf
    audio_data, lsr = sf.read(audiopath)
    audio = torch.FloatTensor(audio_data)
    if len(audio.shape) == 1:
        audio = audio.unsqueeze(0)
    else:
        audio = audio.T  # soundfile returns (samples, channels), need (channels, samples)'''

if old_code in content:
    content = content.replace(old_code, new_code)
    with open(xtts_file, 'w') as f:
        f.write(content)
    print("✅ Patch applied successfully!")
else:
    if "import soundfile as sf" in content:
        print("ℹ️  Already patched!")
    else:
        print("⚠️  Code pattern not found")
