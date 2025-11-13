# XTTS v2 - Configuration Mac M1

## Installation

L'application utilise maintenant **XTTS v2** pour le clonage vocal local avec support multilingue.

### Étape 1 : Installer les dépendances

```bash
# Activer votre environnement virtuel
source venv/bin/activate

# Installer TTS (Coqui) et scipy
pip install TTS>=0.22.0 scipy>=1.11.0
```

### Étape 2 : Vérifier le support MPS (Metal Performance Shaders)

```bash
python3 -c "import torch; print(f'MPS disponible: {torch.backends.mps.is_available()}')"
```

Devrait afficher : `MPS disponible: True`

### Étape 3 : Premier lancement

Au premier lancement, XTTS v2 téléchargera automatiquement le modèle (~2GB). Cela peut prendre quelques minutes.

```bash
python3 app.py
```

## Optimisations Mac M1

- ✅ **Accélération MPS** : Utilise Metal Performance Shaders pour accélérer l'inférence
- ✅ **Support multilingue** : 10+ langues supportées (en, fr, es, de, it, pt, zh, ja, ko, ar)
- ✅ **Mémoire optimisée** : Fonctionne avec 16 Go de RAM

## Performance attendue

- **Premier run** : ~10-15 secondes (chargement modèle)
- **Génération audio** : ~3-5 secondes pour 1 phrase (MPS)
- **Mémoire utilisée** : ~4-6 Go

## Langues supportées

| Code | Langue | Exemple |
|------|--------|---------|
| `en` | English | "Hello, this is a test." |
| `fr` | Français | "Bonjour, ceci est un test." |
| `es` | Español | "Hola, esto es una prueba." |
| `de` | Deutsch | "Hallo, das ist ein Test." |
| `it` | Italiano | "Ciao, questo è un test." |
| `pt` | Português | "Olá, isto é um teste." |
| `zh` | 中文 | "你好，这是一个测试。" |
| `ja` | 日本語 | "こんにちは、これはテストです。" |
| `ko` | 한국어 | "안녕하세요, 이것은 테스트입니다." |
| `ar` | العربية | "مرحبا، هذا اختبار." |

## Dépannage

### Erreur "MPS not available"

Si MPS n'est pas disponible, le modèle utilisera automatiquement le CPU (plus lent).

### Mémoire insuffisante

Si vous manquez de RAM :
1. Fermez les autres applications
2. Le modèle s'adaptera automatiquement

### Modèle lent à charger

Le premier chargement peut prendre 10-15 secondes. Les chargements suivants sont instantanés (cache).

## Migration depuis Chatterbox

L'ancienne intégration Chatterbox a été remplacée par XTTS v2 :

- ✅ Plus de quota API HuggingFace
- ✅ Exécution 100% locale
- ✅ Support multilingue natif
- ✅ Meilleure qualité audio
- ✅ Pas de dépendance réseau
