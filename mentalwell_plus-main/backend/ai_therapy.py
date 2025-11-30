"""
backend/ai_therapy.py

Therapist-style AI therapy backend for MentalWell+.

Features:
- session-based short-term conversation memory (in-memory + optional disk persistence)
- emotion-aware, therapist-style prompting (adaptive tone)
- generate_ai_response(session_id, user_message, emotion_context) -> string
- speak_text(text) -> plays local TTS (pyttsx3)
- get_conversation / clear_conversation utilities

No Streamlit code here — this module provides pure functions to be used by your app.py.
"""

import os
import json
import time
import requests
import pyttsx3
from typing import List, Dict, Optional

# Try to import openai; use fallback if unavailable
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# ----------------------------
# Configuration
# ----------------------------
OLLAMA_API_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.2:1b"

# OpenAI config (read from environment; falls back to None if not set)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", None)
OPENAI_MODEL = "gpt-3.5-turbo"  # or gpt-4 if you prefer

# File to persist short-term conversations (line-delimited JSON per session)
CONVERSATION_STORE = os.path.join(os.path.dirname(__file__), "..", "data", "conversations.json")

# In-memory short-term memory (session_id -> list of exchanges)
# Each exchange: {"role": "user"|"assistant", "text": "...", "emotion": "Neutral", "ts": 1234567890}
_conversations: Dict[str, List[Dict]] = {}


# ----------------------------
# Persistence helpers
# ----------------------------
def _ensure_store_exists():
    folder = os.path.dirname(CONVERSATION_STORE)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    if not os.path.exists(CONVERSATION_STORE):
        open(CONVERSATION_STORE, "a").close()


def _load_persisted():
    """
    Load persisted conversations into memory on demand.
    We keep persisted storage as append-only lines (one JSON per saved session).
    """
    _ensure_store_exists()
    try:
        with open(CONVERSATION_STORE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        sid = item.get("session_id")
                        conv = item.get("conversation", [])
                        if sid and sid not in _conversations:
                            _conversations[sid] = conv
                    except Exception:
                        # ignore malformed lines
                        continue
    except FileNotFoundError:
        pass


def _persist_session(session_id: str):
    """
    Persist a session's conversation as one JSON line (safe append).
    Useful for later review/debugging; not required for runtime.
    """
    _ensure_store_exists()
    conv = _conversations.get(session_id, [])
    to_write = {"session_id": session_id, "ts": int(time.time()), "conversation": conv}
    with open(CONVERSATION_STORE, "a", encoding="utf-8") as f:
        f.write(json.dumps(to_write, ensure_ascii=False) + "\n")


# ----------------------------
# Session management
# ----------------------------
def start_session(session_id: str):
    """Create a fresh in-memory conversation for session_id."""
    _load_persisted()
    if session_id not in _conversations:
        _conversations[session_id] = []


def add_user_message(session_id: str, message: str, emotion: Optional[str] = "Neutral"):
    """Add a user message to session memory."""
    if session_id not in _conversations:
        start_session(session_id)
    _conversations[session_id].append({
        "role": "user",
        "text": message,
        "emotion": emotion or "Neutral",
        "ts": int(time.time())
    })


def add_assistant_message(session_id: str, message: str, emotion: Optional[str] = None):
    """Add an assistant message to session memory."""
    if session_id not in _conversations:
        start_session(session_id)
    _conversations[session_id].append({
        "role": "assistant",
        "text": message,
        "emotion": emotion,
        "ts": int(time.time())
    })


def get_conversation(session_id: str) -> List[Dict]:
    """Return the list of exchanges for a session (most recent last)."""
    return _conversations.get(session_id, [])


def clear_conversation(session_id: str, persist: bool = False):
    """Clear the in-memory conversation. Optionally persist before clearing."""
    if persist and session_id in _conversations:
        _persist_session(session_id)
    _conversations.pop(session_id, None)


# ----------------------------
# Low-level query functions (Ollama + OpenAI fallback)
# ----------------------------
def query_ollama(prompt: str, model: str = MODEL_NAME, timeout: int = 30) -> str:
    """Try to query Ollama; return error string if unavailable."""
    try:
        r = requests.post(OLLAMA_API_URL, json={"model": model, "prompt": prompt, "stream": False}, timeout=timeout)
        if r.status_code == 200:
            data = r.json()
            return data.get("response", "").strip()
        else:
            return f"ollama_error: {r.status_code}"
    except requests.exceptions.ConnectionError:
        return "ollama_offline"
    except Exception as e:
        return f"ollama_error: {e}"


def query_openai(prompt: str) -> str:
    """Query OpenAI as fallback; returns error string if unavailable."""
    if not OPENAI_AVAILABLE or not OPENAI_API_KEY:
        return "openai_unavailable"
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=300
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"openai_error: {e}"


def query_llm(prompt: str) -> str:
    """Query LLM: try Ollama first, fall back to OpenAI if Ollama fails."""
    result = query_ollama(prompt)
    if result.startswith("ollama"):
        # Ollama unavailable or error; try OpenAI
        result = query_openai(prompt)
    return result


# ----------------------------
# Therapist-style prompt builder
# ----------------------------
def _compose_therapist_prompt(session_id: str, user_message: str, emotion_context: str = "Neutral",
                              recall_limit: int = 6) -> str:
    """
    Build a therapist-style prompt combining recent conversation and the current user input.
    recall_limit: how many recent exchanges to include (keeps prompt compact).
    """
    history = get_conversation(session_id)[-recall_limit * 2:]  # take last few exchanges
    history_text_lines = []
    for ex in history:
        role = "User" if ex["role"] == "user" else "Therapist"
        emo = ex.get("emotion", "Neutral")
        history_text_lines.append(f"{role} ({emo}): {ex['text']}")

    history_text = "\n".join(history_text_lines) if history_text_lines else "No prior conversation."

    # Therapist-style instruction with adaptive tone based on emotion_context
    tone_map = {
        "Calm": "steady, warm, and validating",
        "Neutral": "neutral and supportive",
        "Stressed": "calm, grounding, and reassuring",
        "Stressed / Excited": "calm, grounding, and reassuring",
        "Sad": "gentle, validating, and patient",
        "Negative / Sad": "gentle, validating, and patient",
        "Positive / Happy": "warm, encouraging, and affirming",
        "Angry": "measured, de-escalating, and reflective",
        "Unknown": "empathetic and neutral"
    }
    tone = tone_map.get(emotion_context, "empathetic and supportive")

    prompt = f"""
You are a Senior Clinical Psychologist. Your goal is to provide a comprehensive, actionable, and empathetic assessment.
Do not provide medical diagnoses. Keep responses professional, structured, and insightful.

Recent conversation:
{history_text}

Current user message (emotion context: {emotion_context}):
{user_message}

Task:
1) Synthesize the user's input and emotional state.
2) Offer a specific, evidence-based psychological insight (e.g., CBT, Mindfulness).
3) Provide a concrete, actionable step the user can take right now.

Respond in a {tone} tone. Be direct yet compassionate.
"""
    return prompt.strip()


def generate_multimodal_response(session_id: str, face_emotion: str, voice_emotion: str, text_input: str) -> str:
    """
    Generate a comprehensive analysis based on Face, Voice, and Text inputs.
    """
    start_session(session_id)
    
    # Construct a rich user message
    combined_message = (
        f"[Multi-modal Check-in]\n"
        f"Face Expression: {face_emotion}\n"
        f"Voice Tone: {voice_emotion}\n"
        f"User Journal: {text_input}"
    )
    
    add_user_message(session_id, combined_message, emotion=f"{face_emotion}/{voice_emotion}")
    
    prompt = f"""
You are a Senior Clinical Psychologist conducting a multi-modal assessment.
You have data from the user's facial expressions, voice tone, and written text.

Data:
- Face: {face_emotion}
- Voice: {voice_emotion}
- Text: "{text_input}"

Task:
1. Analyze the consistency between their non-verbal cues (Face/Voice) and their words.
   - Example: "You say you are fine, but your voice sounds stressed."
2. Provide a holistic assessment of their current state.
3. Recommend 2 specific therapeutic exercises based on this triad of data.

Keep the response structured and professional.
"""
    response = query_llm(prompt)
    
    if not response or response.startswith("ollama") or response.startswith("openai"):
        fallback = "I'm having trouble analyzing all the data points right now. Please try again."
        add_assistant_message(session_id, fallback, None)
        return fallback
        
    add_assistant_message(session_id, response, None)
    return response.strip()



# ----------------------------
# High-level generate function
# ----------------------------
def generate_ai_response(session_id: str, user_message: str, emotion_context: Optional[str] = "Neutral",
                         persist_history: bool = False) -> str:
    """
    Generate a therapist-style response for a given session.
    - session_id: unique id representing the user/session (string)
    - user_message: the latest message from the user
    - emotion_context: short label like "Sad", "Calm", "Stressed", etc.
    - persist_history: if True, the conversation will be appended to the on-disk store after response
    """
    # Ensure session memory exists
    start_session(session_id)

    # Add user message to memory
    add_user_message(session_id, user_message, emotion_context or "Neutral")

    # Build prompt
    prompt = _compose_therapist_prompt(session_id, user_message, emotion_context or "Neutral")

    # Query model
    response = query_llm(prompt)

    # If the model returned an error-like string, provide a safe fallback
    if not response or response.startswith("ollama") or response.startswith("openai") or response.startswith("Error:") or response.startswith("❌"):
        fallback = ("I'm having trouble generating a thoughtful response right now. "
                    "If you'd like, try rephrasing briefly or come back in a moment.")
        # Add assistant fallback to memory and return
        add_assistant_message(session_id, fallback, None)
        if persist_history:
            _persist_session(session_id)
        return fallback

    # Add assistant reply to memory
    add_assistant_message(session_id, response, None)

    # Optionally persist conversation for later analysis
    if persist_history:
        _persist_session(session_id)

    return response.strip()


# ----------------------------
# Local TTS
# ----------------------------
def save_speech_audio(text: str, file_path: str = "temp_speech.mp3"):
    """
    Save text as audio using local pyttsx3.
    """
    try:
        engine = pyttsx3.init()
        engine.setProperty("rate", 165)
        voices = engine.getProperty("voices")
        if len(voices) > 1:
            engine.setProperty("voice", voices[1].id)
        
        # On some systems, save_to_file might need a full path or specific extension handling
        # We'll try to save to the requested path.
        engine.save_to_file(text, file_path)
        engine.runAndWait()
        return True
    except Exception as e:
        print(f"[TTS] error: {e}")
        return False


# ----------------------------
# Quick utility: summary helper (optional)
# ----------------------------
def summarize_recent(session_id: str, summary_length: int = 80) -> str:
    """
    Ask the LLM to summarize the recent conversation into a compact insight.
    Useful for the Insights panel on the dashboard.
    """
    history = get_conversation(session_id)[-10:]
    if not history:
        return "No recent conversation to summarize."

    combined = "\n".join([f"{ex['role']}: {ex['text']}" for ex in history])
    prompt = f"""
You are a concise therapist summarizer. Read the recent exchanges and produce a two-sentence summary
that captures the user's main concern, emotional trend, and one gentle recommendation.

Recent exchanges:
{combined}

Produce the summary in plain text, 1-2 sentences, suitable for a 'Daily Insight' card.
"""
    res = query_llm(prompt)
    if not res or res.startswith("ollama") or res.startswith("openai"):
        return "Could not generate a summary right now."
    return res.strip()
