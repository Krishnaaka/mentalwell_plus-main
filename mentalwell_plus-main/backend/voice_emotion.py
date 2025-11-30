import json
import os
import wave
import numpy as np
import traceback
import io

# Lazy imports for heavy / optional backends so module can import even when
# system-level dependencies (PortAudio, C extensions) are missing.
sd = None
librosa = None
Model = None
KaldiRecognizer = None
pipeline = None
recognizer = None


def _lazy_import_sounddevice():
    global sd
    if sd is None:
        try:
            import sounddevice as _sd
            sd = _sd
        except Exception:
            sd = None


def _lazy_import_librosa():
    global librosa
    if librosa is None:
        try:
            import librosa as _librosa
            librosa = _librosa
        except Exception:
            librosa = None


def _lazy_import_vosk():
    global Model, KaldiRecognizer
    if Model is None or KaldiRecognizer is None:
        try:
            from vosk import Model as _Model, KaldiRecognizer as _KaldiRecognizer
            Model = _Model
            KaldiRecognizer = _KaldiRecognizer
        except Exception:
            Model = None
            KaldiRecognizer = None


def _lazy_import_transformers():
    global pipeline
    if pipeline is None:
        try:
            from transformers import pipeline as _pipeline
            pipeline = _pipeline
        except Exception:
            pipeline = None


def _lazy_import_speechrecognition():
    global recognizer
    if recognizer is None:
        try:
            from speech_recognition import Recognizer as _Recognizer
            recognizer = _Recognizer()
        except Exception:
            recognizer = None

# ==============================
# 1. Initialize Vosk Model
# ==============================
vosk_model_path = "models/vosk-model-small-en-us-0.15"


_vosk_model = None


def _get_vosk_model():
    global _vosk_model
    if _vosk_model is not None:
        return _vosk_model
    _lazy_import_vosk()
    if Model is None:
        return None
    if not os.path.exists(vosk_model_path):
        return None
    try:
        _vosk_model = Model(vosk_model_path)
        return _vosk_model
    except Exception as e:
        print(f"[Vosk] Failed to load model: {e}")
        _vosk_model = None
        return None


_sentiment_analyzer = None


def _get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is not None:
        return _sentiment_analyzer
    _lazy_import_transformers()
    if pipeline is None:
        return None
    try:
        _sentiment_analyzer = pipeline("sentiment-analysis")
        return _sentiment_analyzer
    except Exception as e:
        print(f"[Transformers] Failed to load sentiment model: {e}")
        _sentiment_analyzer = None
        return None

# ==============================
# 3. Record Audio from Microphone (or accept pre-recorded bytes)
# ==============================
def record_audio(duration=15, sr=16000, audio_bytes=None):
    """Record audio from microphone or process pre-recorded audio bytes.
    
    Args:
        duration: Recording duration in seconds (ignored if audio_bytes provided)
        sr: Sample rate (default 16000)
        audio_bytes: Optional WAV file bytes from Streamlit's st.audio_input() (can be bytes or file-like object)
    
    Returns:
        Tuple of (audio_array, sample_rate)
    """
    if audio_bytes is not None:
        # Process pre-recorded audio from Streamlit
        try:
            print("📊 Processing uploaded audio...")
            
            # Handle Streamlit's UploadedFile or bytes
            if hasattr(audio_bytes, 'read'):
                # It's a file-like object (UploadedFile)
                audio_data = audio_bytes.read()
            else:
                # It's already bytes
                audio_data = audio_bytes
            
            wav_file = io.BytesIO(audio_data)
            with wave.open(wav_file, 'rb') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                
                frame_data = wf.readframes(n_frames)
                audio_array = np.frombuffer(frame_data, dtype=np.int16)
                
                # Convert to float32 and normalize
                audio_array = audio_array.astype(np.float32) / 32767.0
                
                # Convert to mono if stereo
                if n_channels > 1:
                    audio_array = audio_array.reshape(-1, n_channels).mean(axis=1)
                
                print(f"✓ Audio processed: {len(audio_array)} samples at {framerate} Hz")
                return audio_array, framerate
        except Exception as e:
            print(f"❌ Error processing audio bytes: {e}")
            traceback.print_exc()
            raise RuntimeError(f"Failed to process audio: {e}")
    
    # Original sounddevice fallback for direct recording
    _lazy_import_sounddevice()
    if sd is None:
        raise RuntimeError("sounddevice is not available. Please use the Streamlit audio recorder instead.")
    try:
        print(f"🎤 Recording for {duration} seconds... please speak now...")
        audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
        sd.wait()
        print("✓ Recording finished.")
        return np.squeeze(audio), sr
    except Exception as e:
        print(f"❌ Recording failed: {e}")
        traceback.print_exc()
        raise RuntimeError(f"Audio recording failed: {e}")

# ==============================
# 4. Speech to Text (Multiple Backends)
# ==============================
def speech_to_text(y, sr):
    """Transcribe audio using multiple backends: SpeechRecognition > Vosk > Voice Activity Detection."""
    
    # Try SpeechRecognition first (supports Google, IBM, etc.)
    print("[speech_to_text] Trying SpeechRecognition...")
    try:
        _lazy_import_speechrecognition()
        if recognizer is not None:
            # Convert numpy array to wav bytes
            temp_wav = "temp_sr.wav"
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes((y * 32767).astype(np.int16).tobytes())
            
            # Load and recognize
            from speech_recognition import AudioFile
            with AudioFile(temp_wav) as source:
                audio_data = recognizer.record(source)
            
            try:
                # Try Google Speech Recognition (online)
                text = recognizer.recognize_google(audio_data)
                print(f"[speech_to_text] Google recognized: {text}")
                # Clean up
                try:
                    os.remove(temp_wav)
                except:
                    pass
                return text
            except Exception as google_err:
                print(f"[speech_to_text] Google recognition failed: {google_err}")
                # Try offline Sphinx if available
                try:
                    text = recognizer.recognize_sphinx(audio_data)
                    print(f"[speech_to_text] Sphinx recognized: {text}")
                    try:
                        os.remove(temp_wav)
                    except:
                        pass
                    return text
                except Exception as sphinx_err:
                    print(f"[speech_to_text] Sphinx not available: {sphinx_err}")
            finally:
                try:
                    os.remove(temp_wav)
                except:
                    pass
    except Exception as e:
        print(f"[speech_to_text] SpeechRecognition error: {e}")
    
    # Try Vosk as second option
    print("[speech_to_text] Trying Vosk model...")
    model = _get_vosk_model()
    if model is not None and KaldiRecognizer is not None:
        try:
            temp_wav = "temp.wav"
            with wave.open(temp_wav, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(sr)
                wf.writeframes((y * 32767).astype(np.int16).tobytes())

            rec = KaldiRecognizer(model, sr)
            wf = wave.open(temp_wav, "rb")

            text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    result = json.loads(rec.Result())
                    text += result.get("text", "") + " "

            final_result = json.loads(rec.FinalResult())
            text += final_result.get("text", "")
            wf.close()
            
            # Clean up temp file
            try:
                os.remove(temp_wav)
            except:
                pass
            
            if text.strip():
                print(f"[speech_to_text] Vosk detected: {text.strip()}")
                return text.strip()
        except Exception as e:
            print(f"[speech_to_text] Vosk error: {e}")
    else:
        print("[speech_to_text] Vosk model not available")
    
    # Fallback: simple voice activity detection
    print("[speech_to_text] Using voice activity fallback detection...")
    try:
        _lazy_import_librosa()
        if librosa is not None:
            # Detect speech using energy and spectral properties
            S = np.abs(librosa.stft(y))
            energy = np.sqrt(np.sum(S**2, axis=0))
            
            # Simple threshold: if average energy is high enough, assume speech
            avg_energy = np.mean(energy)
            if avg_energy > 0.01:
                print(f"[speech_to_text] Voice activity detected (energy: {avg_energy:.4f})")
                return "[Voice detected - transcription unavailable]"
            else:
                print(f"[speech_to_text] Low energy, likely silence (energy: {avg_energy:.4f})")
                return ""
        else:
            # Fallback to numpy-only energy detection
            energy = np.mean(np.square(y))
            if energy > 0.001:
                print(f"[speech_to_text] Voice activity detected (energy: {energy:.6f})")
                return "[Voice detected - transcription unavailable]"
            else:
                print(f"[speech_to_text] No voice activity (energy: {energy:.6f})")
                return ""
    except Exception as e:
        print(f"[speech_to_text] Fallback error: {e}")
        traceback.print_exc()
        return ""


# ==============================
# 5. Simple Voice Emotion Analysis
# ==============================
def detect_voice_emotion(y, sr):
    """Analyze voice emotion from audio waveform using energy and pitch analysis."""
    try:
        _lazy_import_librosa()
        
        # Calculate energy
        if librosa is not None:
            try:
                energy = np.mean(librosa.feature.rms(y=y))
                print(f"[emotion] Energy (librosa): {energy:.6f}")
            except:
                energy = float(np.mean(np.square(y)))
                print(f"[emotion] Energy (numpy): {energy:.6f}")
        else:
            energy = float(np.mean(np.square(y)))
            print(f"[emotion] Energy (numpy fallback): {energy:.6f}")
        
        # Detect pitch/frequency characteristics
        try:
            if librosa is not None:
                pitch, _ = librosa.piptrack(y=y, sr=sr)
                mean_pitch = np.mean(pitch[pitch > 0]) if np.any(pitch > 0) else 0
                print(f"[emotion] Mean pitch: {mean_pitch:.2f} Hz")
        except:
            mean_pitch = 0
        
        # Emotion classification based on energy and pitch
        # Adjusted thresholds for better detection
        if energy < 0.0001:
            emotion = "Silence"
        elif energy < 0.001:
            emotion = "Calm / Whisper"
        elif energy < 0.005:
            emotion = "Calm"
        elif energy < 0.015:
            emotion = "Neutral"
        elif energy < 0.03:
            emotion = "Energetic"
        else:
            emotion = "Stressed / Excited"
        
        print(f"[emotion] Detected: {emotion}")
        return emotion
        
    except Exception as e:
        print(f"[detect_voice_emotion] Error: {e}")
        traceback.print_exc()
        # Ultimate fallback
        try:
            energy = float(np.mean(np.square(y)))
            if energy < 0.0005:
                return "Silence"
            elif energy < 0.01:
                return "Calm"
            else:
                return "Energetic"
        except:
            return "Neutral"
    except Exception as e:
        print(f"[energy analysis fallback] Error: {e}")
        return "Neutral"

# ==============================
# 6. Text Sentiment Analysis
# ==============================
def analyze_text_sentiment(text):
    """Analyze sentiment from text using transformers model."""
    if not text or not text.strip():
        return "No text detected"
    analyzer = _get_sentiment_analyzer()
    if analyzer is None:
        # Simple keyword-based fallback
        t = text.lower()
        if any(word in t for word in ["sad", "depress", "unhappy", "cry", "hate", "angry"]):
            return "Negative / Sad"
        if any(word in t for word in ["happy", "joy", "glad", "love", "awesome", "great"]):
            return "Positive / Happy"
        return "Neutral"
    try:
        result = analyzer(text[:512])[0]  # limit text to 512 chars
        label = result.get('label', 'NEUTRAL')
        score = result.get('score', 0.0)
        if label == "POSITIVE":
            emotion = "Positive / Happy"
        elif label == "NEGATIVE":
            emotion = "Negative / Sad"
        else:
            emotion = "Neutral"
        return f"{emotion} (confidence: {score:.2f})"
    except Exception as e:
        print(f"[sentiment analysis] Error: {e}")
        return "Sentiment analysis failed"

# ==============================
# 7. Unified Voice Analysis Function
# ==============================
def run_voice_analysis(duration=10, audio_bytes=None):
    """Record voice, transcribe, and analyze emotion.
    
    Args:
        duration: Recording duration in seconds (used only if audio_bytes is None)
        audio_bytes: Optional WAV bytes from Streamlit st.audio_input() widget
    
    This function returns a dict with keys `text`, `voice`, and `text_emotion`.
    If recording fails, returns a dict with an `error` key instead.
    """
    try:
        y, sr = record_audio(duration, audio_bytes=audio_bytes)
    except Exception as e:
        return {"error": str(e)}

    try:
        text = speech_to_text(y, sr)
    except Exception as e:
        print(f"[speech_to_text error] {e}")
        text = ""

    try:
        voice_emotion = detect_voice_emotion(y, sr)
    except Exception as e:
        print(f"[voice emotion error] {e}")
        voice_emotion = "Neutral"

    try:
        text_emotion = analyze_text_sentiment(text) if text else "No text detected"
    except Exception as e:
        print(f"[text emotion error] {e}")
        text_emotion = "Analysis unavailable"

    return {
        "text": text,
        "voice": voice_emotion,
        "text_emotion": text_emotion
    }
