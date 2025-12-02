try:
    from transformers import pipeline
    from PIL import Image
    import numpy as np
    
    print("Transformers imported.")
    
    # Create a dummy image
    img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    
    # Load pipeline (using a small, fast model)
    # dima806/facial_emotions_image_detection is a good candidate
    pipe = pipeline("image-classification", model="dima806/facial_emotions_image_detection")
    
    print("Pipeline loaded.")
    
    result = pipe(img)
    print(f"Result: {result}")
    
except Exception as e:
    print(f"Error: {e}")
