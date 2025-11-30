# backend/login_manager.py
import hashlib
import uuid
import base64
import json
from pathlib import Path
import datetime

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
USERS_FILE = DATA_DIR / "users.json"
DATA_DIR.mkdir(exist_ok=True)
if not USERS_FILE.exists():
    USERS_FILE.write_text("")

def _hash_password(password: str, salt: str) -> str:
    h = hashlib.sha256()
    h.update((salt + password).encode("utf-8"))
    return h.hexdigest()

def load_users():
    users = {}
    with open(USERS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                u = json.loads(line)
                users[u["username"]] = u
            except:
                pass
    return users

def create_user(username: str, password: str):
    users = load_users()
    if username in users:
        return False, "exists"
    salt = base64.urlsafe_b64encode(uuid.uuid4().bytes).decode("utf-8").rstrip("=")
    pw_hash = _hash_password(password, salt)
    entry = {"username": username, "salt": salt, "pw_hash": pw_hash, "created": str(datetime.date.today())}
    with open(USERS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")
    return True, "created"

def verify_user(username: str, password: str):
    users = load_users()
    if username not in users:
        return False, "not found"
    u = users[username]
    pw_hash = _hash_password(password, u.get("salt",""))
    if pw_hash == u.get("pw_hash",""):
        return True, "ok"
    return False, "bad"
