from pathlib import Path
import json
from app import MOOD_FILE, append_json_line
print('MOOD_FILE ->', MOOD_FILE)
try:
    append_json_line(MOOD_FILE, {'test':'ok','time':__import__('datetime').datetime.now().isoformat()})
    print('append_json_line succeeded')
    p = MOOD_FILE
    if not p.exists():
        # look for fallback
        home = Path.home()/'.mentalwell_plus'/p.name
        if home.exists():
            p = home
            print('Found home fallback file:', p)
    print('Reading last 5 lines from', p)
    with open(p,'r',encoding='utf-8') as f:
        lines = f.read().splitlines()
        for l in lines[-5:]:
            print(l)
except Exception as e:
    print('append_json_line failed:', e)
