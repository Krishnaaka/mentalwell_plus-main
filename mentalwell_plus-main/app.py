# app.py — MindWave (Lenix Edition)
import streamlit as st
import json
import datetime
import uuid
import threading
import subprocess
from pathlib import Path
import pandas as pd
import plotly.express as px
import random
import time
import hashlib
import base64
import os

# -------------------------
# Paths & data directory
# -------------------------
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FALLBACK_TO_BASE = False
try:
    DATA_DIR.mkdir(exist_ok=True)
except PermissionError:
    FALLBACK_TO_BASE = True
    print("[storage] Permission denied creating data directory; will fallback to project root for storage.")

USERS_FILE = (DATA_DIR if not FALLBACK_TO_BASE else BASE_DIR) / "users.json"
MOOD_FILE = (DATA_DIR if not FALLBACK_TO_BASE else BASE_DIR) / "mood.json"
SLEEP_FILE = (DATA_DIR if not FALLBACK_TO_BASE else BASE_DIR) / "sleep.json"

for f in (USERS_FILE, MOOD_FILE, SLEEP_FILE):
    try:
        if not f.exists():
            f.write_text("")
    except PermissionError:
        print(f"[storage] Permission denied creating file {f}; continuing with fallback behavior.")

# -------------------------
# Backend imports (safe fallbacks)
# -------------------------
try:
    from backend.voice_emotion import run_voice_analysis
except Exception:
    def run_voice_analysis(duration=10, audio_bytes=None):
        return {"error": "backend.voice_emotion not found"}

try:
    from backend.ai_therapy import generate_ai_response, save_speech_audio, start_session, summarize_recent, generate_multimodal_response
except Exception:
    def generate_ai_response(*a, **kw): return "AI backend not available."
    def generate_multimodal_response(*a, **kw): return "AI backend not available."
    def save_speech_audio(*a, **kw): return False
    def start_session(*a, **kw): pass
    def summarize_recent(*a, **kw): return "AI backend unavailable."


try:
    from backend.text_emotion import predict_text_emotion
except Exception:
    def predict_text_emotion(text): return "neutral"

# -------------------------
# Streamlit config & Lenix CSS
# -------------------------
st.set_page_config(page_title="MindWave", page_icon="🌊", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&family=Instrument+Serif:ital@0;1&display=swap');

:root {
    --bg-dark: #050505;
    --card-bg: rgba(255, 255, 255, 0.03);
    --primary: #c4f934; /* Lenix Green */
    --secondary: #e0e0e0;
    --accent: #3b82f6;
    --text: #ffffff;
    --text-muted: #888888;
    --border: rgba(255, 255, 255, 0.08);
}

* { box-sizing: border-box; }

html, body, .stApp {
    background-color: var(--bg-dark);
    color: var(--text);
    font-family: 'Plus Jakarta Sans', sans-serif;
    scroll-behavior: smooth;
}

/* Video Intro Background for Login */
.video-bg {
    position: fixed;
    top: 0; left: 0; width: 100%; height: 100%;
    z-index: -1;
    background: linear-gradient(45deg, #000000, #1a1a1a);
    overflow: hidden;
}

.video-blob {
    position: absolute;
    filter: blur(80px);
    opacity: 0.6;
    animation: blob-float 20s infinite alternate;
}

.blob-1 { top: -10%; left: -10%; width: 50vw; height: 50vw; background: #3b82f6; animation-delay: 0s; }
.blob-2 { bottom: -10%; right: -10%; width: 60vw; height: 60vw; background: #c4f934; animation-delay: -5s; }
.blob-3 { top: 40%; left: 40%; width: 40vw; height: 40vw; background: #ff0055; animation-delay: -10s; opacity: 0.4; }

@keyframes blob-float {
    0% { transform: translate(0, 0) scale(1); }
    100% { transform: translate(50px, 50px) scale(1.1); }
}

/* Fade In Animation */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(10px); }
    to { opacity: 1; transform: translateY(0); }
}

.animate-enter {
    animation: fadeIn 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards;
}

.stButton button {
    transition: all 0.3s ease !important;
}
.stButton button:hover {
    transform: scale(1.02);
}


/* Hide default elements */
header, footer { display: none !important; }
.stDeployButton { display: none !important; }

/* Typography */
h1 {
    font-family: 'Instrument Serif', serif;
    font-weight: 400;
    font-size: 4rem;
    letter-spacing: -0.02em;
    line-height: 1.1;
    margin-bottom: 0.5em;
}

h2, h3 {
    font-weight: 600;
    letter-spacing: -0.01em;
    color: var(--text);
}

/* Cards */
.lenix-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 32px;
    padding: 32px;
    transition: transform 0.4s cubic-bezier(0.16, 1, 0.3, 1), box-shadow 0.4s ease;

    position: relative;
    overflow: hidden;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}

.lenix-card:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(255, 255, 255, 0.2);
    transform: translateY(-4px);
}

.card-icon {
    font-size: 32px;
    margin-bottom: 24px;
    background: rgba(255,255,255,0.1);
    width: 64px; height: 64px;
    display: flex; align-items: center; justify-content: center;
    border-radius: 50%;
}

.card-title {
    font-size: 24px;
    font-weight: 600;
    margin-bottom: 8px;
}

.card-desc {
    font-size: 16px;
    color: var(--text-muted);
    line-height: 1.6;
    margin-bottom: 24px;
}

/* Buttons */
.stButton > button {
    background: var(--primary);
    border: none;
    border-radius: 100px;
    color: #000;
    font-weight: 600;
    padding: 12px 32px;
    transition: all 0.3s ease;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    font-size: 14px;
    width: 100%;
}

.stButton > button:hover {
    background: #b0e620;
    transform: scale(1.02);
    box-shadow: 0 0 20px rgba(196, 249, 52, 0.3);
}

/* Inputs */
.stTextInput > div > div > input, .stTextArea > div > div > textarea, .stNumberInput > div > div > input {
    background: transparent;
    border: 1px solid var(--border);
    color: white;
    border-radius: 16px;
    padding: 12px;
}

.stTextInput > div > div > input:focus, .stTextArea > div > div > textarea:focus {
    border-color: var(--primary);
    box-shadow: 0 0 0 1px var(--primary);
}

/* Metrics */
[data-testid="stMetricValue"] {
    font-family: 'Instrument Serif', serif;
    color: var(--primary);
    font-size: 3rem;
}

/* Cursor Effect */
.cursor-trail {
    position: fixed;
    width: 20px;
    height: 20px;
    border-radius: 50%;
    background: rgba(196, 249, 52, 0.5);
    pointer-events: none;
    z-index: 9999;
    transition: transform 0.1s ease;
    mix-blend-mode: difference;
}

/* Larger Login Input */
.login-input input {
    font-size: 1.2rem !important;
    padding: 15px !important;
}
</style>
<script>
document.addEventListener('mousemove', function(e) {
    let trail = document.createElement('div');
    trail.className = 'cursor-trail';
    trail.style.left = e.pageX + 'px';
    trail.style.top = e.pageY + 'px';
    document.body.appendChild(trail);
    setTimeout(() => {
        trail.style.transform = 'scale(0)';
        setTimeout(() => { trail.remove(); }, 300);
    }, 50);
});
</script>
""", unsafe_allow_html=True)

# -------------------------
# Session State Management
# -------------------------
if "user" not in st.session_state: st.session_state.user = None
if "active_card" not in st.session_state: st.session_state.active_card = None
if "camera_on" not in st.session_state: st.session_state.camera_on = False

def navigate_to(card_key):
    st.session_state.active_card = card_key

def back_home():
    st.session_state.active_card = None
    st.session_state.camera_on = False
    # Cleanup camera if exists
    if 'face_cam' in st.session_state:
        try:
            st.session_state.face_cam.stop()
            del st.session_state.face_cam
        except: pass

def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

def set_bg_image(png_file):
    try:
        bin_str = get_base64_of_bin_file(png_file)
        page_bg_img = f'''
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{bin_str}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}
        /* Overlay to darken it slightly for readability */
        .stApp::before {{
            content: "";
            position: absolute;
            top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0, 0, 0, 0.4);
            z-index: -1;
        }}
        </style>
        '''
        st.markdown(page_bg_img, unsafe_allow_html=True)
    except Exception as e:
        print(f"Error loading background: {e}")
# -------------------------
# Helpers
# -------------------------
def _ensure_file(path: Path):
    if not path.exists():
        try: path.parent.mkdir(parents=True, exist_ok=True)
        except: pass
        path.write_text("")

def append_json_line(filepath: Path, data: dict):
    _ensure_file(filepath)
    with open(filepath, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")

def read_json_lines_file(filepath: Path):
    _ensure_file(filepath)
    items = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try: items.append(json.loads(line))
                except: pass
    return items

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()

def create_user(username, password):
    users = {u["username"]: u for u in read_json_lines_file(USERS_FILE) if "username" in u}
    if username in users: return False, "Username exists"
    salt = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("utf-8").rstrip("=")
    pw_hash = _hash_password(password, salt)
    append_json_line(USERS_FILE, {"username": username, "salt": salt, "pw_hash": pw_hash, "created": str(datetime.date.today())})
    return True, "Created"

def verify_user(username, password):
    users = {u["username"]: u for u in read_json_lines_file(USERS_FILE) if "username" in u}
    if username not in users: return False, "User not found"
    u = users[username]
    if _hash_password(password, u.get("salt","")) == u.get("pw_hash",""): return True, "Login success"
    return False, "Invalid password"

# -------------------------
# Components
# -------------------------
def lenix_card(title, desc, icon, key):
    st.markdown(f"""
    <div class="lenix-card">
        <div>
            <div class="card-icon">{icon}</div>
            <div class="card-title">{title}</div>
            <div class="card-desc">{desc}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.button("Open", key=f"btn_{key}", on_click=navigate_to, args=(key,))

# -------------------------
# Pages
# -------------------------
def show_login():
    # Set Background
    set_bg_image("assets/login_bg.png")
    
    st.markdown("""
    <style>
    @keyframes float {
        0% { transform: translateY(0px); }
        50% { transform: translateY(-10px); }
        100% { transform: translateY(0px); }
    }
    .logo-img {
        animation: float 3s ease-in-out infinite;
        display: block;
        margin-left: auto;
        margin-right: auto;
        width: 150px;
        border-radius: 50%;
        box-shadow: 0 0 20px rgba(196, 249, 52, 0.5);
    }
    </style>
    <div class="animate-enter">
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 1.5, 1])
    with c2:
        st.markdown("<div style='height: 10vh'></div>", unsafe_allow_html=True)
        
        # Logo
        try:
            logo_b64 = get_base64_of_bin_file("assets/logo.png")
            st.markdown(f'<img src="data:image/png;base64,{logo_b64}" class="logo-img">', unsafe_allow_html=True)
        except:
            st.markdown("<h1>🌊</h1>", unsafe_allow_html=True)

        st.markdown("<h1 style='text-align: center; color: white; margin-top: 20px;'>MindWave</h1>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: center; color: #ccc; margin-bottom: 40px; font-size: 1.2rem;'>Your journey to mental clarity starts here.</p>", unsafe_allow_html=True)
        
        tab1, tab2 = st.tabs(["Login", "Sign Up"])
        with tab1:
            st.markdown('<div class="login-input">', unsafe_allow_html=True)
            u = st.text_input("Username", key="login_u")
            p = st.text_input("Password", type="password", key="login_p")
            st.markdown('</div>', unsafe_allow_html=True)
            if st.button("Enter MindWave", use_container_width=True):
                ok, msg = verify_user(u, p)
                if ok:
                    st.session_state.user = u
                    st.rerun()
                else:
                    st.error(msg)
        with tab2:
            nu = st.text_input("New Username", key="signup_u")
            np = st.text_input("New Password", type="password", key="signup_p")
            if st.button("Join MindWave", use_container_width=True):
                ok, msg = create_user(nu, np)
                if ok: st.success(msg)
                else: st.error(msg)

def show_dashboard():
    # Set Background
    set_bg_image("assets/login_bg.png")

    st.markdown("""
    <div class="animate-enter">
    """, unsafe_allow_html=True)

    st.title("MindWave")
    st.markdown(f"Welcome back, **{st.session_state.user}**.")

    st.markdown("Here is your daily overview.")
    
    # Grid Layout
    c1, c2, c3, c4 = st.columns(4)
    with c1: lenix_card("Face Emotion", "Real-time analysis", "📷", "face")
    with c2: lenix_card("Voice Analysis", "Tone detection", "🎙️", "voice")
    with c3: lenix_card("Multi-modal", "Face + Voice + Text", "🧠", "multimodal")
    with c4: lenix_card("AI Therapy", "Chat session", "💬", "therapy")

    
    st.markdown("### Journal & Wellness")
    c5, c6, c7, c8 = st.columns(4)
    with c5: lenix_card("Mood Diary", "Track feelings", "📊", "mood")
    with c6: lenix_card("Sleep Journal", "Log rest", "😴", "sleep")
    with c7: lenix_card("Games", "De-stress", "🎮", "games")
    with c8: lenix_card("Meditation", "Breathe", "🧘", "meditation")

    st.markdown("---")
    if st.button("Sign Out", key="logout"):
        st.session_state.user = None
        st.rerun()

def show_face_page():
    # Placeholders for dynamic updates
    bg_placeholder = st.empty()
    title_placeholder = st.empty()
    
    # Initial State
    emotion = "Neutral"
    if 'face_cam' in st.session_state:
        with st.session_state.face_cam.lock:
            emotion = st.session_state.face_cam.current_emotion

    # Function to update UI based on emotion
    def update_ui(current_emotion):
        # Map emotion to background color/gradient overlay
        color_map = {
            "happy": "rgba(255, 223, 0, 0.3)", # Yellow
            "sad": "rgba(0, 0, 255, 0.3)", # Blue
            "angry": "rgba(255, 0, 0, 0.3)", # Red
            "surprise": "rgba(255, 105, 180, 0.3)", # Pink
            "neutral": "rgba(0, 0, 0, 0.4)", # Dark
            "fear": "rgba(128, 0, 128, 0.3)" # Purple
        }
        overlay_color = color_map.get(current_emotion.lower(), "rgba(0, 0, 0, 0.4)")
        
        bg_placeholder.markdown(f"""
        <style>
        .stApp::before {{
            background: {overlay_color} !important;
            transition: background 0.5s ease;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        title_placeholder.title(f"Face Emotion: {current_emotion}")
        print(f"DEBUG: UI Update - Emotion: {current_emotion}")

    # Initial UI Update
    update_ui(emotion)

    # Set Background Image (Base) - Static
    set_bg_image("assets/login_bg.png")
    
    st.button("← Back", on_click=back_home)
    
    c1, c2 = st.columns([3, 1])
    with c1:
        placeholder = st.empty()
        
        col_start, col_stop = st.columns(2)
        if col_start.button("▶ Start Camera"):
            st.session_state.camera_on = True
        if col_stop.button("⏹ Stop Camera"):
            st.session_state.camera_on = False
            
        if st.session_state.camera_on:
            try:
                from backend.face_emotion import FaceCamera
                if 'face_cam' not in st.session_state:
                    st.session_state.face_cam = FaceCamera()
                cam = st.session_state.face_cam
                
                while st.session_state.camera_on:
                    # 1. Get Frame
                    frame = cam.get_frame_bytes()
                    
                    # 2. Get Emotion
                    current_emo = "Neutral"
                    with cam.lock:
                        current_emo = cam.current_emotion
                    
                    # 3. Update UI
                    update_ui(current_emo)
                    
                    # 4. Show Frame
                    if frame is not None:
                        placeholder.image(frame, use_container_width=True)
                    else:
                        time.sleep(0.1)
                    time.sleep(0.03)
            except Exception as e:
                st.error(f"Camera error: {e}")
        else:
            placeholder.info("Camera is inactive.")

def show_voice_page():
    st.button("← Back", on_click=back_home)
    st.title("Voice Analysis")
    audio_val = st.audio_input("Record")
    if audio_val:
        st.audio(audio_val)
        if st.button("Analyze"):
            with st.spinner("Listening..."):
                res = run_voice_analysis(audio_bytes=audio_val)
                if "error" in res: st.error(res["error"])
                else:
                    c1, c2 = st.columns(2)
                    c1.metric("Tone", res.get("voice", "N/A"))
                    c2.metric("Sentiment", res.get("text_emotion", "N/A"))
                    st.info(f"Transcript: {res.get('text', '')}")

def show_multimodal_page():
    st.button("← Back", on_click=back_home)
    st.title("Multi-modal Analysis")
    st.markdown("We will analyze your **Face**, **Voice**, and **Text** together for a holistic assessment.")
    
    # Step 1: Face
    st.subheader("1. Face Snapshot")
    if st.button("Capture Face Emotion"):
        # Mock capture for demo if camera not active, or use real logic
        # For simplicity in this flow, we'll assume the user wants to start the cam briefly
        st.info("Camera starting for snapshot...")
        try:
            from backend.face_emotion import FaceCamera
            cam = FaceCamera()
            time.sleep(2) # Warmup
            frame = cam.get_frame_bytes() # In a real app we'd analyze this frame
            cam.stop()
            st.image(frame, width=300)
            st.session_state.mm_face = "Neutral" # Placeholder for actual detection
            st.success("Face captured: Neutral (Simulated)")
        except:
            st.session_state.mm_face = "Neutral"
            st.warning("Camera unavailable, assuming Neutral.")

    # Step 2: Voice
    st.subheader("2. Voice Record")
    audio_val = st.audio_input("Record a short sentence")
    if audio_val:
        st.session_state.mm_voice = "Calm" # Placeholder
        st.success("Voice captured: Calm (Simulated)")

    # Step 3: Text
    st.subheader("3. Journal")
    txt = st.text_area("How do you feel right now?")
    
    if st.button("Generate Holistic Report"):
        face = st.session_state.get("mm_face", "Unknown")
        voice = st.session_state.get("mm_voice", "Unknown")
        
        if txt:
            with st.spinner("Synthesizing data..."):
                if "therapy_sid" not in st.session_state:
                    st.session_state.therapy_sid = f"{st.session_state.user}_{uuid.uuid4()}"
                
                report = generate_multimodal_response(st.session_state.therapy_sid, face, voice, txt)
                st.markdown("### 🧠 Holistic Assessment")
                st.write(report)
        else:
            st.warning("Please enter some text.")


def show_therapy_page():
    st.button("← Back", on_click=back_home)
    st.title("AI Therapy")
    if "therapy_sid" not in st.session_state:
        st.session_state.therapy_sid = f"{st.session_state.user}_{uuid.uuid4()}"
        start_session(st.session_state.therapy_sid)
    sid = st.session_state.therapy_sid
    user_input = st.text_input("Message...", key="chat_in")
    if st.button("Send") and user_input:
        with st.spinner("Thinking..."):
            resp = generate_ai_response(sid, user_input, "neutral", persist_history=True)
            st.session_state.last_response = resp
    if "last_response" in st.session_state:
        st.markdown(f"**Therapist:** {st.session_state.last_response}")
        if st.button("🔊 Speak"):
            # Generate audio file
            audio_file = "response_audio.mp3"
            if save_speech_audio(st.session_state.last_response, audio_file):
                st.audio(audio_file, format="audio/mp3")
            else:
                st.error("Could not generate audio.")

def show_mood_page():
    st.button("← Back", on_click=back_home)
    st.title("Mood Diary")
    
    with st.form("mood_form"):
        mood_score = st.slider("How are you feeling? (1-10)", 1, 10, 5)
        note = st.text_area("Notes")
        submitted = st.form_submit_button("Log Mood")
        if submitted:
            entry = {"user": st.session_state.user, "date": str(datetime.datetime.now()), "score": mood_score, "note": note}
            append_json_line(MOOD_FILE, entry)
            st.success("Mood logged!")
    
    st.subheader("History")
    data = [x for x in read_json_lines_file(MOOD_FILE) if x.get("user") == st.session_state.user]
    if data:
        df = pd.DataFrame(data)
        # Convert date to datetime for plotting
        df["date"] = pd.to_datetime(df["date"])
        
        # Plot
        fig = px.line(df, x="date", y="score", title="Mood Trend", markers=True, template="plotly_dark")
        fig.update_layout(yaxis_range=[0, 11])
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(df[["date", "score", "note"]], use_container_width=True)
    else:
        st.info("No entries yet.")

def show_sleep_page():
    st.button("← Back", on_click=back_home)
    st.title("Sleep Journal")
    
    with st.form("sleep_form"):
        hours = st.number_input("Hours slept", 0.0, 24.0, 7.0, 0.5)
        quality = st.select_slider("Quality", options=["Poor", "Fair", "Good", "Excellent"])
        submitted = st.form_submit_button("Log Sleep")
        if submitted:
            entry = {"user": st.session_state.user, "date": str(datetime.datetime.now()), "hours": hours, "quality": quality}
            append_json_line(SLEEP_FILE, entry)
            st.success("Sleep logged!")

    st.subheader("History")
    data = [x for x in read_json_lines_file(SLEEP_FILE) if x.get("user") == st.session_state.user]
    if data:
        df = pd.DataFrame(data)
        st.dataframe(df[["date", "hours", "quality"]], use_container_width=True)
    else:
        st.info("No entries yet.")

def show_games_page():
    st.button("← Back", on_click=back_home)
    st.title("Games")
    
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Bubble Pop", "Memory Focus", "Reaction Time", "Pattern Match", "Sentence Recall"])
    
    with tab1:
        if "bubble_count" not in st.session_state: st.session_state.bubble_count = 0
        st.metric("Bubbles Popped", st.session_state.bubble_count)
        
        if st.button("Pop Bubbles! 🎈"):
            st.session_state.bubble_count += 1
            st.balloons()
            st.success(f"Popped! Total: {st.session_state.bubble_count}")
            
    with tab2:
        st.write("Memorize the number. The length increases as you get it right!")
        
        # Initialize state
        if "memory_level" not in st.session_state: st.session_state.memory_level = 4
        if "memory_num" not in st.session_state: 
            st.session_state.memory_num = random.randint(10**(st.session_state.memory_level-1), 10**st.session_state.memory_level - 1)
        if "memory_high_score" not in st.session_state: st.session_state.memory_high_score = 0

        st.metric("Current Level (Digits)", st.session_state.memory_level)
        st.metric("High Score (Level)", st.session_state.memory_high_score)
        
        if st.button("Show Number"):
            placeholder = st.empty()
            placeholder.header(st.session_state.memory_num)
            time.sleep(3)
            placeholder.empty()
            st.info("Number hidden! Type it below.")
        
        guess = st.number_input("Enter number", 0, 999999999999)
        if st.button("Check"):
            if guess == st.session_state.memory_num:
                st.success("Correct! 🎉 Level Up!")
                if st.session_state.memory_level > st.session_state.memory_high_score:
                    st.session_state.memory_high_score = st.session_state.memory_level
                st.session_state.memory_level += 1
                st.session_state.memory_num = random.randint(10**(st.session_state.memory_level-1), 10**st.session_state.memory_level - 1)
            else:
                st.error(f"Incorrect. The number was {st.session_state.memory_num}. Resetting to Level 4.")
                st.session_state.memory_level = 4
                st.session_state.memory_num = random.randint(1000, 9999)

    with tab3:
        st.write("Click 'Stop' when the color turns GREEN!")
        
        # State initialization
        if "reaction_state" not in st.session_state: st.session_state.reaction_state = "idle" # idle, waiting, ready, finished
        if "reaction_start_time" not in st.session_state: st.session_state.reaction_start_time = 0
        if "reaction_best_time" not in st.session_state: st.session_state.reaction_best_time = 999.0
        
        if st.session_state.reaction_best_time < 999:
            st.metric("Best Time", f"{st.session_state.reaction_best_time:.3f}s")

        # Logic
        if st.session_state.reaction_state == "idle":
            st.markdown("""
            <div style="background-color: #333; height: 200px; border-radius: 20px; display: flex; align-items: center; justify-content: center;">
                <h2 style="color: white; margin: 0;">Click 'Start'</h2>
            </div>
            """, unsafe_allow_html=True)
            if st.button("Start Timer"):
                st.session_state.reaction_state = "waiting"
                st.session_state.reaction_delay = random.uniform(2, 5)
                st.session_state.reaction_init_time = time.time()
                st.rerun()
                
        elif st.session_state.reaction_state == "waiting":
            st.markdown("""
            <div style="background-color: #ff4b4b; height: 200px; border-radius: 20px; display: flex; align-items: center; justify-content: center;">
                <h2 style="color: white; margin: 0;">WAIT...</h2>
            </div>
            """, unsafe_allow_html=True)
            
            elapsed = time.time() - st.session_state.reaction_init_time
            if elapsed >= st.session_state.reaction_delay:
                st.session_state.reaction_state = "ready"
                st.session_state.reaction_start_time = time.time()
                st.rerun()
            else:
                time.sleep(0.1)
                st.rerun()
                
        elif st.session_state.reaction_state == "ready":
            st.markdown("""
            <div style="background-color: #4caf50; height: 200px; border-radius: 20px; display: flex; align-items: center; justify-content: center; cursor: pointer;">
                <h2 style="color: white; margin: 0;">CLICK STOP!</h2>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("STOP!", type="primary", use_container_width=True):
                diff = time.time() - st.session_state.reaction_start_time
                st.balloons()
                st.success(f"Reaction Time: {diff:.3f} seconds")
                if diff < st.session_state.reaction_best_time:
                    st.session_state.reaction_best_time = diff
                    st.success("New Best Time!")
                st.session_state.reaction_state = "idle"
        
        if st.button("Reset"):
            st.session_state.reaction_state = "idle"


    with tab4:
        st.write("Pattern Match (Simon Says)")
        colors = ["🔴", "🟢", "🔵", "🟡"]
        
        if "pattern_level" not in st.session_state: st.session_state.pattern_level = 3
        if "pattern" not in st.session_state: st.session_state.pattern = [random.choice(colors) for _ in range(st.session_state.pattern_level)]
        if "pattern_high_score" not in st.session_state: st.session_state.pattern_high_score = 0
        
        st.metric("Current Level (Length)", st.session_state.pattern_level)
        st.metric("High Score", st.session_state.pattern_high_score)

        if "user_pattern" not in st.session_state: st.session_state.user_pattern = []

        if st.button("Show Pattern"):
            st.info("Memorize this:")
            placeholder = st.empty()
            placeholder.header(" ".join(st.session_state.pattern))
            time.sleep(3)
            placeholder.empty()
            st.info("Pattern hidden! Recreate it below.")
            st.session_state.user_pattern = [] # Reset user input on new show
            
        st.write("Your Input:")
        st.header(" ".join(st.session_state.user_pattern) if st.session_state.user_pattern else "...")
        
        c1, c2, c3, c4 = st.columns(4)
        if c1.button("🔴"): st.session_state.user_pattern.append("🔴")
        if c2.button("🟢"): st.session_state.user_pattern.append("🟢")
        if c3.button("🔵"): st.session_state.user_pattern.append("🔵")
        if c4.button("�"): st.session_state.user_pattern.append("�")
        
        if st.button("Clear Input"):
            st.session_state.user_pattern = []

        if st.button("Check Pattern"):
            if st.session_state.user_pattern == st.session_state.pattern:
                st.success("Perfect Match! 🧠 Level Up!")
                if st.session_state.pattern_level > st.session_state.pattern_high_score:
                    st.session_state.pattern_high_score = st.session_state.pattern_level
                st.session_state.pattern_level += 1
                st.session_state.pattern = [random.choice(colors) for _ in range(st.session_state.pattern_level)]
                st.session_state.user_pattern = []
            else:
                st.error(f"Incorrect. The pattern was: {' '.join(st.session_state.pattern)}. Resetting to Level 3.")
                st.session_state.pattern_level = 3
                st.session_state.pattern = [random.choice(colors) for _ in range(3)]
                st.session_state.user_pattern = []

    with tab5:
        st.write("Sentence Recall")
        sentences = [
            "The sun shines bright.",
            "Blue birds fly over the rainbow.",
            "A quick brown fox jumps over the lazy dog.",
            "To be or not to be, that is the question.",
            "All that glitters is not gold, but it shines."
        ]
        
        if "sentence_level" not in st.session_state: st.session_state.sentence_level = 0
        if "current_sentence" not in st.session_state: st.session_state.current_sentence = sentences[0]
        
        st.metric("Level", st.session_state.sentence_level + 1)
        
        if st.button("Show Sentence"):
            placeholder = st.empty()
            placeholder.markdown(f"### {st.session_state.current_sentence}")
            time.sleep(5)
            placeholder.empty()
            st.info("Sentence hidden! Write it below.")
            
        user_sentence = st.text_input("Type the sentence exactly:")
        
        if st.button("Check Sentence"):
            if user_sentence.strip() == st.session_state.current_sentence:
                st.success("Correct! Moving to next level.")
                if st.session_state.sentence_level < len(sentences) - 1:
                    st.session_state.sentence_level += 1
                    st.session_state.current_sentence = sentences[st.session_state.sentence_level]
                else:
                    st.balloons()
                    st.success("You completed all levels! 🎉")
                    st.session_state.sentence_level = 0
                    st.session_state.current_sentence = sentences[0]
            else:
                st.error("Incorrect. Try again.")



def show_meditation_page():
    st.button("← Back", on_click=back_home)
    st.title("Meditation")
    
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### Breathing Circle")
        # CSS Animation for breathing
        st.markdown("""
        <div style="
            width: 200px; height: 200px; 
            background: radial-gradient(circle, #c4f934 0%, transparent 70%);
            border-radius: 50%;
            margin: 0 auto;
            animation: breathe 8s infinite ease-in-out;
        "></div>
        <style>
        @keyframes breathe {
            0%, 100% { transform: scale(0.5); opacity: 0.5; }
            50% { transform: scale(1.2); opacity: 1; }
        }
        </style>
        <p style="text-align:center; margin-top:20px">Breathe with the glow</p>
        """, unsafe_allow_html=True)

    with c2:
        if st.button("Start 1 Min Timer"):
            ph = st.empty()
            for i in range(60, 0, -1):
                ph.metric("Time Remaining", f"{i}s")
                time.sleep(1)
            ph.success("Session Complete.")

# -------------------------
# Main Router
# -------------------------
if not st.session_state.user:
    show_login()
else:
    card = st.session_state.active_card
    if card is None:
        show_dashboard()
    elif card == "face": show_face_page()
    elif card == "voice": show_voice_page()
    elif card == "multimodal": show_multimodal_page()
    elif card == "text": show_text_page()

    elif card == "therapy": show_therapy_page()
    elif card == "mood": show_mood_page()
    elif card == "sleep": show_sleep_page()
    elif card == "games": show_games_page()
    elif card == "meditation": show_meditation_page()
    else:
        st.info("Coming soon.")
        st.button("Back", on_click=back_home)
