# backend/text_emotion.py
# High-accuracy text emotion detection using j-hartmann/distilroberta emotion model if available.
# Provides predict_text_emotion(text) -> returns label string like 'joy','sadness','anger', etc.

from typing import Optional
import logging

# Try to import transformers; provide a fallback function if missing
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import torch
    import torch.nn.functional as F
    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("transformers not available in backend.text_emotion: %s", e)

# Model config
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

# Emotion label ordering (model-specific)
LABELS = [
    "anger",
    "disgust",
    "fear",
    "joy",
    "neutral",
    "sadness",
    "surprise"
]

# Negative-word override helps fix false-joy on "sad" text
NEGATIVE_WORDS = [
    "sad", "depress", "depressed", "unhappy", "worthless", "hopeless", "lonely",
    "cry", "crying", "suicide", "suicidal", "hurt", "pain", "tired", "anxious",
    "panic", "scared", "fear", "panic attack"
]

# Lazy-load model/tokenizer
_tokenizer = None
_model = None
_device = None

def _load_model_if_needed():
    global _tokenizer, _model, _device
    if not TRANSFORMERS_AVAILABLE:
        return False
    if _model is None or _tokenizer is None:
        # prefer CPU unless GPU is available
        _device = 0 if torch.cuda.is_available() else -1
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        # move model to CPU/GPU as required
        if torch.cuda.is_available():
            _model.to("cuda")
    return True

def _fallback_rule_based(text: str) -> str:
    """Simple lexical fallback if transformers isn't available."""
    t = (text or "").lower()
    if any(w in t for w in NEGATIVE_WORDS):
        return "sadness"
    if any(w in t for w in ["happy","joy","glad","excited","great","wonderful"]):
        return "joy"
    if any(w in t for w in ["angry","furious","mad","hate"]):
        return "anger"
    if any(w in t for w in ["scared","fear","panic"]):
        return "fear"
    return "neutral"

def predict_text_emotion(text: str) -> str:
    """
    Return one of LABELS as the predicted emotion.
    Uses the transformer model if available, otherwise a rule-based fallback.
    Includes negative-word override and a confidence threshold.
    """
    if not text or not text.strip():
        return "neutral"

    # quick lexical override for clear negative signals (improves accuracy)
    lowered = text.lower()
    if any(w in lowered for w in NEGATIVE_WORDS):
        return "sadness"

    if not TRANSFORMERS_AVAILABLE:
        return _fallback_rule_based(text)

    # ensure model loaded
    ok = _load_model_if_needed()
    if not ok:
        return _fallback_rule_based(text)

    try:
        inputs = _tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
        # move tensors to GPU if available
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}
            _model.to("cuda")
        with torch.no_grad():
            outputs = _model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]  # numpy array
        # get top label and probability
        top_idx = int(probs.argmax())
        top_prob = float(probs[top_idx])

        # if confidence is low, prefer neutral to avoid random "joy"
        if top_prob < 0.45:
            return "neutral"

        label = LABELS[top_idx]
        # extra safety: if label is joy but text contains obvious negative words, override
        if label == "joy" and any(w in lowered for w in NEGATIVE_WORDS):
            return "sadness"
        return label
    except Exception:
        # fallback safe return
        return _fallback_rule_based(text)
