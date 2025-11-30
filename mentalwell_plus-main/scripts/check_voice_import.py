import importlib,traceback,sys
from pathlib import Path

# Ensure project root is on sys.path so 'backend' package can be found
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

try:
    m = importlib.import_module('backend.voice_emotion')
    print('Imported module:', m)
    print('module file:', getattr(m, '__file__', None))
    print('has run_voice_analysis:', hasattr(m, 'run_voice_analysis'))
except Exception:
    traceback.print_exc()
    sys.exit(1)
