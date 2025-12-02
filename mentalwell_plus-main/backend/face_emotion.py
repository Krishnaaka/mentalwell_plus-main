"""
Standalone YOLO face emotion script used by the Streamlit app.

This version uses YOLO for face detection and Hugging Face Transformers for emotion recognition.
Model: dima806/facial_emotions_image_detection
"""

import cv2
import threading
import time
import numpy as np
import os
import requests
from ultralytics import YOLO
from PIL import Image

# Try to import transformers
try:
    from transformers import pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Transformers not installed.")

MODEL_URL = "https://github.com/lindevs/yolov8-face/releases/download/v1.0.0/yolov8n-face.pt"
MODEL_PATH = "yolov8n-face.pt"

def ensure_model():
    if not os.path.exists(MODEL_PATH):
        print(f"Downloading {MODEL_PATH} from {MODEL_URL}...")
        try:
            response = requests.get(MODEL_URL, stream=True)
            response.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            print("Download complete.")
        except Exception as e:
            print(f"Failed to download model: {e}")
            return False
    return True

# Global pipeline cache
_emotion_pipeline = None

def _get_pipeline():
    global _emotion_pipeline
    if _emotion_pipeline is None and TRANSFORMERS_AVAILABLE:
        try:
            print("Loading emotion pipeline...")
            _emotion_pipeline = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
            print("Emotion pipeline loaded.")
        except Exception as e:
            print(f"Failed to load pipeline: {e}")
            _emotion_pipeline = None
    return _emotion_pipeline

def simple_fallback(face_roi):
    """Basic heuristic when transformers is not available."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    if mean > 170:
        print("DEBUG: Using Fallback -> surprised")
        return "surprised"
    if mean < 80:
        print("DEBUG: Using Fallback -> sad")
        return "sad"
    print("DEBUG: Using Fallback -> neutral")
    return "neutral"


# Global cache for emotion results
# Key: (x, y) approx coordinates, Value: (emotion, timestamp)
_emotion_cache = {}

def detect_emotions(frame, detector, process_emotion=True):
    """Run YOLO, then infer emotion with Transformers -> heuristic."""
    results = detector(frame)[0]
    
    current_time = time.time()
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        emotion_text = "Unknown"
        
        # Cache key based on rough position
        cache_key = (round(x1 / 50), round(y1 / 50))
        cached = _emotion_cache.get(cache_key)
        
        should_analyze = False
        if not cached:
            should_analyze = True
        elif (current_time - cached[1]) > 0.1:
            should_analyze = True
        elif process_emotion:
            should_analyze = True
            
        if should_analyze:
            pipe = _get_pipeline()
            if pipe:
                try:
                    # Convert BGR to RGB for PIL
                    rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    pil_img = Image.fromarray(rgb_face)
                    
                    # Run inference
                    results = pipe(pil_img)
                    # results is a list of dicts like [{'label': 'happy', 'score': 0.9}, ...]
                    if results:
                        top_result = results[0]
                        emotion_text = top_result['label']
                        print(f"DEBUG: Using Transformers -> {emotion_text} ({top_result['score']:.2f})")
                except Exception as e:
                    print(f"Inference error: {e}")
                    emotion_text = "Unknown"

            if emotion_text == "Unknown":
                emotion_text = simple_fallback(face)
            
            # Update cache
            _emotion_cache[cache_key] = (emotion_text, current_time)
        else:
            emotion_text = cached[0]

        # Draw
        color_map = {
            "happy": (0, 255, 255),    # Yellow
            "sad": (255, 0, 0),        # Blue
            "angry": (0, 0, 255),      # Red
            "surprise": (180, 105, 255), # Pinkish
            "neutral": (0, 255, 0),    # Green
            "fear": (128, 0, 128),      # Purple
            "disgust": (0, 128, 0)      # Dark Green
        }
        color = color_map.get(emotion_text.lower(), (0, 255, 0))
        
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    return frame, emotion_text


class FaceCamera:
    """
    Threaded camera capture to prevent blocking Streamlit.
    """
    def __init__(self, source=0, resize_width=640):
        self.source = source
        self.resize_width = resize_width
        
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.lock = threading.Lock()
        self.running = True
        
        self.raw_frame = None
        self.processed_frame = None
        self.current_emotion = "Neutral"
        self.latest_detections = [] # List of (box, emotion)
        
        ensure_model()
        try:
            self._model = YOLO(MODEL_PATH)
        except Exception:
            print("Failed to load YOLO face model, falling back to standard YOLOv8n")
            self._model = YOLO("yolov8n.pt")
            
        # Pre-load pipeline in background
        threading.Thread(target=_get_pipeline, daemon=True).start()

        self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.process_thread = threading.Thread(target=self._process_loop, daemon=True)
        
        self.capture_thread.start()
        self.process_thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if self.resize_width:
                    h, w = frame.shape[:2]
                    scale = self.resize_width / w
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                
                with self.lock:
                    self.raw_frame = frame
            else:
                time.sleep(0.1)
            time.sleep(0.01)

    def _process_loop(self):
        while self.running:
            frame_to_process = None
            with self.lock:
                if self.raw_frame is not None:
                    frame_to_process = self.raw_frame.copy()
            
            if frame_to_process is not None:
                try:
                    detections = self._detect_data(frame_to_process)
                    
                    with self.lock:
                        self.latest_detections = detections
                        if detections:
                            self.current_emotion = detections[0][1]
                except Exception as e:
                    print(f"Processing error: {e}")
            
            time.sleep(0.05)

    def _detect_data(self, frame):
        results = self._model(frame, verbose=False)[0]
        detections = []
        current_time = time.time()
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue
            
            cache_key = (round(x1 / 50), round(y1 / 50))
            cached = _emotion_cache.get(cache_key)
            
            emotion_text = "Unknown"
            should_analyze = False
            
            if not cached: should_analyze = True
            elif (current_time - cached[1]) > 0.1: should_analyze = True
            
            if should_analyze:
                pipe = _get_pipeline()
                if pipe:
                    try:
                        rgb_face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(rgb_face)
                        results = pipe(pil_img)
                        if results:
                            emotion_text = results[0]['label']
                            print(f"DEBUG: Using Transformers (Thread) -> {emotion_text}")
                    except: pass
                
                if emotion_text == "Unknown":
                    emotion_text = simple_fallback(face)
                
                _emotion_cache[cache_key] = (emotion_text, current_time)
            else:
                emotion_text = cached[0]
            
            detections.append(((x1, y1, x2, y2), emotion_text))
            
        return detections

    def get_frame_bytes(self):
        with self.lock:
            if self.raw_frame is None: return None
            display_frame = self.raw_frame.copy()
            detections = self.latest_detections
        
            color_map = {
                "happy": (0, 255, 255),
                "sad": (255, 0, 0),
                "angry": (0, 0, 255),
                "surprise": (180, 105, 255),
                "neutral": (0, 255, 0),
                "fear": (128, 0, 128),
                "disgust": (0, 128, 0)
            }
            
            for (x1, y1, x2, y2), emo in detections:
                color = color_map.get(emo.lower(), (0, 255, 0))
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(display_frame, emo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    def stop(self):
        self.running = False
        self.capture_thread.join()
        self.process_thread.join()
        self.cap.release()


def main():
    yolo = YOLO("yolov8n-face.pt")
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        detect_emotions(frame, yolo)
        cv2.imshow("YOLO + Emotion", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
