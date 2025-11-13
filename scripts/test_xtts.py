#!/usr/bin/env python3
"""
Test XTTS v2 avec détection MPS (Mac M1)
"""

import torch
print(f"PyTorch version: {torch.__version__}")
print(f"MPS available: {torch.backends.mps.is_available()}")
print(f"MPS built: {torch.backends.mps.is_built()}")

if torch.backends.mps.is_available():
    device = "mps"
    print("✓ Using MPS (Metal Performance Shaders)")
else:
    device = "cpu"
    print("⚠ Using CPU")

print(f"\nDevice selected: {device}")

# Test simple import TTS
try:
    from TTS.api import TTS
    print("✓ TTS imported successfully")
    
    # Liste des modèles disponibles
    print("\nLoading XTTS v2...")
    tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", progress_bar=True)
    print(f"✓ XTTS v2 loaded")
    
    # Move to device
    if device == "mps":
        try:
            tts.to(device)
            print(f"✓ Model moved to {device}")
        except Exception as e:
            print(f"⚠ Could not move to MPS: {e}")
            print("  Falling back to CPU")
            device = "cpu"
            tts.to(device)
    
    print(f"\n✓ XTTS v2 ready on {device}!")
    print("\nNote: Le modèle a été téléchargé (~2GB). Les prochains lancements seront instantanés.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()
