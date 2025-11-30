import sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

import traceback

print("=" * 60)
print("Checking YOLOv8 & Face Emotion Backend")
print("=" * 60)

# Test 1: Check ultralytics (YOLOv8)
print("\n[1] Testing ultralytics (YOLOv8)...")
try:
    from ultralytics import YOLO
    print("    ✓ ultralytics imported successfully")
    print("    Model file expected at: models/yolov8n-face.pt")
    
    model_path = Path("models/yolov8n-face.pt")
    if model_path.exists():
        print(f"    ✓ Model file exists: {model_path}")
    else:
        print(f"    ✗ Model file NOT found: {model_path}")
        print(f"      Size needed: ~6-7 MB")
        print(f"      Download from: https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8n-face.pt")
except Exception as e:
    print(f"    ✗ ultralytics import failed: {e}")
    traceback.print_exc()

# Test 2: Check deepface
print("\n[2] Testing deepface...")
try:
    from deepface import DeepFace
    print("    ✓ deepface imported successfully")
except Exception as e:
    print(f"    ✗ deepface import failed: {e}")
    traceback.print_exc()

# Test 3: Check backend.face_emotion
print("\n[3] Testing backend.face_emotion module...")
try:
    from backend.face_emotion import analyze_face_emotion
    print("    ✓ backend.face_emotion imported successfully")
    print("    ✓ analyze_face_emotion function available")
except Exception as e:
    print(f"    ✗ backend.face_emotion import failed: {e}")
    traceback.print_exc()

print("\n" + "=" * 60)
print("Summary:")
print("  If all three tests pass, face detection will work in the app.")
print("  If the model file is missing, download it from the URL above.")
print("=" * 60)
