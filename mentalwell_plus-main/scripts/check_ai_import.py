import importlib,traceback,sys
from pathlib import Path

# Ensure project root is on sys.path
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    m = importlib.import_module('backend.ai_therapy')
    print('Imported module:', m)
    print('module file:', getattr(m, '__file__', None))
    print('has generate_ai_response:', hasattr(m, 'generate_ai_response'))
    print('has speak_text:', hasattr(m, 'speak_text'))
except Exception:
    traceback.print_exc()
    sys.exit(1)
