#!/usr/bin/env python3
"""Test script to verify voice analysis functions."""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from backend.voice_emotion import (
    record_audio, speech_to_text, detect_voice_emotion, 
    analyze_text_sentiment, run_voice_analysis
)
import numpy as np

print("=" * 60)
print("Testing Voice Analysis Components")
print("=" * 60)

# Test 1: Create synthetic audio (silence)
print("\n[TEST 1] Synthetic silence audio (1 second)")
print("-" * 60)
silence = np.zeros(16000, dtype=np.float32)
text = speech_to_text(silence, 16000)
emotion = detect_voice_emotion(silence, 16000)
print(f"Speech detected: '{text}'")
print(f"Emotion: {emotion}")

# Test 2: Create synthetic audio (sound - white noise)
print("\n[TEST 2] Synthetic audio with noise (1 second)")
print("-" * 60)
noise = np.random.normal(0, 0.1, 16000).astype(np.float32)
text = speech_to_text(noise, 16000)
emotion = detect_voice_emotion(noise, 16000)
print(f"Speech detected: '{text}'")
print(f"Emotion: {emotion}")

# Test 3: Create louder synthetic audio
print("\n[TEST 3] Synthetic louder audio (1 second)")
print("-" * 60)
loud_noise = np.random.normal(0, 0.3, 16000).astype(np.float32)
text = speech_to_text(loud_noise, 16000)
emotion = detect_voice_emotion(loud_noise, 16000)
print(f"Speech detected: '{text}'")
print(f"Emotion: {emotion}")

# Test 4: Text sentiment analysis
print("\n[TEST 4] Text sentiment analysis")
print("-" * 60)
test_texts = [
    "I am very happy today!",
    "I feel sad and depressed",
    "This is neutral",
    "",
]
for txt in test_texts:
    sentiment = analyze_text_sentiment(txt)
    print(f"Text: '{txt}' -> Sentiment: {sentiment}")

print("\n" + "=" * 60)
print("Tests completed!")
print("=" * 60)

