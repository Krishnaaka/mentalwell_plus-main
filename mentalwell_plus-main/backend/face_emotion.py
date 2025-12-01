"""
Standalone YOLO face emotion script used by the Streamlit app.

This version keeps YOLO for face detection, then tries two emotion heads:
1. DeepFace (if installed)
2. FER (PyTorch CNN on FER2013) as a lightweight fallback

Install FER via: pip install fer
"""

import cv2
import threading
import time
import numpy as np
import os
import requests
from ultralytics import YOLO

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


try:
    from deepface import DeepFace
except Exception:
    DeepFace = None

try:
    from fer import FER
except Exception:
    FER = None


_fer_detector = None


def _get_fer():
    """Lazy-load FER detector to avoid startup cost if unused."""
    global _fer_detector
    if _fer_detector is None and FER is not None:
        _fer_detector = FER(mtcnn=False)
    return _fer_detector


def simple_fallback(face_roi):
    """Basic heuristic when neither DeepFace nor FER is available."""
    gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    mean = gray.mean()
    if mean > 170:
        return "surprised"
    if mean < 80:
        return "sad"
    return "neutral"


# Global cache for emotion results to avoid re-running DeepFace every frame
# Key: (x, y) approx coordinates, Value: (emotion, timestamp)
_emotion_cache = {}

def detect_emotions(frame, detector, process_emotion=True):
    """Run YOLO, then infer emotion with DeepFace -> FER -> heuristic."""
    results = detector(frame)[0]
    
    current_time = time.time()
    
    for box in results.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        face = frame[y1:y2, x1:x2]
        if face.size == 0:
            continue

        emotion_text = "Unknown"
        
        # Cache key based on rough position (to track same face)
        # Round to nearest 50px to handle slight movement
        cache_key = (round(x1 / 50), round(y1 / 50))
        
        cached = _emotion_cache.get(cache_key)
        
        # Re-analyze if:
        # 1. No cache
        # 2. Cache expired (> 1.0s old)
        # 3. Explicitly requested (process_emotion=True) AND cache is somewhat old (> 0.2s)
        should_analyze = False
        if not cached:
            should_analyze = True
        elif (current_time - cached[1]) > 1.0:
            should_analyze = True
        elif process_emotion:
            should_analyze = True
            
        if should_analyze:
            if DeepFace is not None:
                try:
                    # Run DeepFace
                    analysis = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
                    dominant = analysis[0]["dominant_emotion"]
                    emotion_text = dominant
                except Exception:
                    emotion_text = "Unknown"

            if emotion_text == "Unknown":
                fer = _get_fer()
                if fer is not None:
                    try:
                        fer_result = fer.top_emotion(face)
                        if fer_result:
                            emotion_text = fer_result[0]
                    except Exception:
                        emotion_text = "Unknown"

            if emotion_text == "Unknown":
                emotion_text = simple_fallback(face)
            
            # Update cache
            _emotion_cache[cache_key] = (emotion_text, current_time)
        else:
            # Use cached value
            emotion_text = cached[0]

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, emotion_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    return frame, emotion_text


class FaceCamera:
    """
    Threaded camera capture to prevent blocking Streamlit.
    """
    def __init__(self, source=0, resize_width=640):
        self.source = source
        self.resize_width = resize_width
        
        # Use DirectShow on Windows to avoid MSMF errors
        self.cap = cv2.VideoCapture(self.source, cv2.CAP_DSHOW)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        
        self.lock = threading.Lock()
        self.running = True
        
        self.raw_frame = None
        self.processed_frame = None
        self.current_emotion = "Neutral"
        self.latest_detections = [] # List of (box, emotion)
        
        # Ensure model exists
        ensure_model()
        try:
            self._model = YOLO(MODEL_PATH)
        except Exception:
            print("Failed to load YOLO face model, falling back to standard YOLOv8n")
            self._model = YOLO("yolov8n.pt")

        # Start threads
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
                    # Run detection (this might be slow)
                    # We modify detect_emotions to return just data, not draw on frame
                    detections = self._detect_data(frame_to_process)
                    
                    with self.lock:
                        self.latest_detections = detections
                        # Update dominant emotion
                        if detections:
                            self.current_emotion = detections[0][1] # First face
                except Exception as e:
                    print(f"Processing error: {e}")
            
            time.sleep(0.05) # Process at ~20 FPS max to save CPU

    def _detect_data(self, frame):
        """
        Internal method to get boxes and emotions without drawing.
        """
        results = self._model(frame, verbose=False)[0]
        detections = []
        current_time = time.time()
        
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            face = frame[y1:y2, x1:x2]
            if face.size == 0: continue
            
            # Cache logic
            cache_key = (round(x1 / 50), round(y1 / 50))
            cached = _emotion_cache.get(cache_key)
            
            emotion_text = "Unknown"
            should_analyze = False
            
            if not cached: should_analyze = True
            elif (current_time - cached[1]) > 0.5: should_analyze = True # Re-check every 0.5s
            
            if should_analyze:
                if DeepFace is not None:
                    try:
                        analysis = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False, silent=True)
                        emotion_text = analysis[0]["dominant_emotion"]
                    except: pass
                
                if emotion_text == "Unknown" and _get_fer():
                    try:
                        res = _get_fer().top_emotion(face)
                        if res: emotion_text = res[0]
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
            # Draw latest detections on the current raw frame
            display_frame = self.raw_frame.copy()
            detections = self.latest_detections
        
        for (x1, y1, x2, y2), emo in detections:
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, emo, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        return cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB)

    def stop(self):
        self.running = False
        self.capture_thread.join()
        self.process_thread.join()
        self.cap.release()


def attempt_download_model(url):
    # Placeholder for model download logic if needed
    return True, "Model download not implemented yet."

def ensure_ultralytics_fallback():
    # Placeholder for fallback logic
    return True, "Fallback ensured."

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
