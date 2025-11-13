"""
Patch TTS to work with PyTorch 2.6+ weights_only changes
"""
import os
import sys

# Find TTS installation
venv_path = os.path.join(os.path.dirname(__file__), "venv")
tts_io_path = os.path.join(venv_path, "lib", "python3.10", "site-packages", "TTS", "utils", "io.py")

if not os.path.exists(tts_io_path):
    print(f"❌ TTS not found at {tts_io_path}")
    sys.exit(1)

print(f"📝 Patching {tts_io_path}")

# Read current content
with open(tts_io_path, 'r') as f:
    content = f.read()

# Replace torch.load calls to include weights_only=False
original = 'return torch.load(f, map_location=map_location, **kwargs)'
patched = 'return torch.load(f, map_location=map_location, weights_only=False, **kwargs)'

if original in content:
    content = content.replace(original, patched)
    with open(tts_io_path, 'w') as f:
        f.write(content)
    print("✅ Patch applied successfully!")
else:
    if patched in content:
        print("ℹ️  Already patched!")
    else:
        print("⚠️  Could not find expected torch.load pattern")
