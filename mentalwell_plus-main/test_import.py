try:
    import ultralytics
    print(f"Ultralytics imported successfully: {ultralytics.__file__}")
except ImportError as e:
    print(f"Failed to import ultralytics: {e}")
except Exception as e:
    print(f"An error occurred: {e}")
