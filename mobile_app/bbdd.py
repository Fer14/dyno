import hashlib
import secrets
import sqlite3
import threading
from pathlib import Path

import numpy as np

DB_PATH = Path(__file__).parent / "dyno.db"

_local = threading.local()


def get_db() -> sqlite3.Connection:
    """Get a thread-local SQLite connection."""
    if not hasattr(_local, "conn") or _local.conn is None:
        _local.conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute("PRAGMA journal_mode=WAL")
        _local.conn.execute("PRAGMA foreign_keys=ON")
    return _local.conn


def init_db():
    """Create tables if they don't exist."""
    db = get_db()
    db.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            username    TEXT UNIQUE NOT NULL,
            password    TEXT NOT NULL,
            salt        TEXT NOT NULL,
            token       TEXT UNIQUE,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS dragons (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            latent      BLOB NOT NULL,
            name        TEXT NOT NULL,
            parent1_id  INTEGER,
            parent2_id  INTEGER,
            received    INTEGER DEFAULT 0,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS eggs (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id     INTEGER NOT NULL REFERENCES users(id),
            type        TEXT NOT NULL CHECK(type IN ('random', 'bred')),
            latent      BLOB,
            parent1_id  INTEGER,
            parent2_id  INTEGER,
            created_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_dragons_user ON dragons(user_id);
        CREATE INDEX IF NOT EXISTS idx_eggs_user ON eggs(user_id);
    """)
    # Add google_id column if it doesn't exist (migration for existing DBs)
    try:
        db.execute("ALTER TABLE users ADD COLUMN google_id TEXT UNIQUE")
    except sqlite3.OperationalError:
        pass  # column already exists
    db.commit()


# --- Latent conversion ---

def latent_to_blob(latent: list[float]) -> bytes:
    return np.array(latent, dtype=np.float32).tobytes()


def blob_to_latent(blob: bytes) -> list[float]:
    return np.frombuffer(blob, dtype=np.float32).tolist()


# --- Password hashing ---

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()


# --- Users ---

def create_user(username: str, password: str) -> tuple[int, str]:
    """Create a new user. Returns (user_id, token). Raises sqlite3.IntegrityError if username taken."""
    db = get_db()
    salt = secrets.token_hex(16)
    hashed = _hash_password(password, salt)
    token = secrets.token_hex(32)
    cursor = db.execute(
        "INSERT INTO users (username, password, salt, token) VALUES (?, ?, ?, ?)",
        (username, hashed, salt, token),
    )
    db.commit()
    return cursor.lastrowid, token


def login_user(username: str, password: str) -> tuple[int, str] | None:
    """Authenticate user. Returns (user_id, token) or None if invalid."""
    db = get_db()
    row = db.execute("SELECT id, password, salt FROM users WHERE username = ?", (username,)).fetchone()
    if row is None:
        return None
    if _hash_password(password, row["salt"]) != row["password"]:
        return None
    token = secrets.token_hex(32)
    db.execute("UPDATE users SET token = ? WHERE id = ?", (token, row["id"]))
    db.commit()
    return row["id"], token


def google_auth(google_id: str, email: str, name: str) -> tuple[int, str, str, bool]:
    """Login or register via Google. Returns (user_id, token, username, is_new)."""
    db = get_db()
    row = db.execute("SELECT id, username FROM users WHERE google_id = ?", (google_id,)).fetchone()
    if row:
        # Existing Google user — refresh token
        token = secrets.token_hex(32)
        db.execute("UPDATE users SET token = ? WHERE id = ?", (token, row["id"]))
        db.commit()
        return row["id"], token, row["username"], False

    # New Google user — create account
    # Use email prefix as username, ensure uniqueness
    base_username = email.split("@")[0] if email else name or "player"
    username = base_username
    suffix = 1
    while db.execute("SELECT 1 FROM users WHERE username = ?", (username,)).fetchone():
        username = f"{base_username}{suffix}"
        suffix += 1

    salt = secrets.token_hex(16)
    # Google users don't need a password, but we store a random one to keep schema consistent
    hashed = _hash_password(secrets.token_hex(16), salt)
    token = secrets.token_hex(32)
    cursor = db.execute(
        "INSERT INTO users (username, password, salt, token, google_id) VALUES (?, ?, ?, ?, ?)",
        (username, hashed, salt, token, google_id),
    )
    db.commit()
    return cursor.lastrowid, token, username, True


def get_user_by_token(token: str) -> dict | None:
    """Look up user by session token. Returns dict with id, username or None."""
    db = get_db()
    row = db.execute("SELECT id, username FROM users WHERE token = ?", (token,)).fetchone()
    if row is None:
        return None
    return {"id": row["id"], "username": row["username"]}


# --- Dragons ---

def create_dragon(user_id: int, latent: list[float], name: str,
                  parent1_id: int | None = None, parent2_id: int | None = None,
                  received: bool = False) -> int:
    """Create a dragon. Returns dragon_id."""
    db = get_db()
    cursor = db.execute(
        "INSERT INTO dragons (user_id, latent, name, parent1_id, parent2_id, received) VALUES (?, ?, ?, ?, ?, ?)",
        (user_id, latent_to_blob(latent), name, parent1_id, parent2_id, int(received)),
    )
    db.commit()
    return cursor.lastrowid


def get_dragons(user_id: int) -> list[dict]:
    """Get all dragons for a user."""
    db = get_db()
    rows = db.execute(
        "SELECT id, latent, name, parent1_id, parent2_id, received FROM dragons WHERE user_id = ? ORDER BY id",
        (user_id,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "latent": blob_to_latent(r["latent"]),
            "name": r["name"],
            "parent1_id": r["parent1_id"],
            "parent2_id": r["parent2_id"],
            "received": bool(r["received"]),
        }
        for r in rows
    ]


def get_dragon(dragon_id: int, user_id: int) -> dict | None:
    """Get a single dragon owned by user."""
    db = get_db()
    r = db.execute(
        "SELECT id, latent, name, parent1_id, parent2_id, received FROM dragons WHERE id = ? AND user_id = ?",
        (dragon_id, user_id),
    ).fetchone()
    if r is None:
        return None
    return {
        "id": r["id"],
        "latent": blob_to_latent(r["latent"]),
        "name": r["name"],
        "parent1_id": r["parent1_id"],
        "parent2_id": r["parent2_id"],
        "received": bool(r["received"]),
    }


def update_dragon_name(dragon_id: int, user_id: int, name: str) -> bool:
    """Rename a dragon. Returns True if found and updated."""
    db = get_db()
    cursor = db.execute(
        "UPDATE dragons SET name = ? WHERE id = ? AND user_id = ?",
        (name, dragon_id, user_id),
    )
    db.commit()
    return cursor.rowcount > 0


def delete_dragon(dragon_id: int, user_id: int) -> bool:
    """Delete a dragon. Returns True if found and deleted."""
    db = get_db()
    cursor = db.execute("DELETE FROM dragons WHERE id = ? AND user_id = ?", (dragon_id, user_id))
    db.commit()
    return cursor.rowcount > 0


# --- Eggs ---

def create_egg(user_id: int, egg_type: str, latent: list[float] | None = None,
               parent1_id: int | None = None, parent2_id: int | None = None) -> int:
    """Create an egg. Returns egg_id."""
    db = get_db()
    blob = latent_to_blob(latent) if latent else None
    cursor = db.execute(
        "INSERT INTO eggs (user_id, type, latent, parent1_id, parent2_id) VALUES (?, ?, ?, ?, ?)",
        (user_id, egg_type, blob, parent1_id, parent2_id),
    )
    db.commit()
    return cursor.lastrowid


def get_eggs(user_id: int) -> list[dict]:
    """Get all eggs for a user."""
    db = get_db()
    rows = db.execute(
        "SELECT id, type, latent, parent1_id, parent2_id FROM eggs WHERE user_id = ? ORDER BY id",
        (user_id,),
    ).fetchall()
    return [
        {
            "id": r["id"],
            "type": r["type"],
            "latent": blob_to_latent(r["latent"]) if r["latent"] else None,
            "parent1_id": r["parent1_id"],
            "parent2_id": r["parent2_id"],
        }
        for r in rows
    ]


def get_egg(egg_id: int, user_id: int) -> dict | None:
    """Get a single egg owned by user."""
    db = get_db()
    r = db.execute(
        "SELECT id, type, latent, parent1_id, parent2_id FROM eggs WHERE id = ? AND user_id = ?",
        (egg_id, user_id),
    ).fetchone()
    if r is None:
        return None
    return {
        "id": r["id"],
        "type": r["type"],
        "latent": blob_to_latent(r["latent"]) if r["latent"] else None,
        "parent1_id": r["parent1_id"],
        "parent2_id": r["parent2_id"],
    }


def delete_egg(egg_id: int, user_id: int) -> bool:
    """Delete an egg. Returns True if found and deleted."""
    db = get_db()
    cursor = db.execute("DELETE FROM eggs WHERE id = ? AND user_id = ?", (egg_id, user_id))
    db.commit()
    return cursor.rowcount > 0


def create_initial_eggs(user_id: int, count: int = 5):
    """Create initial random eggs for a new user."""
    db = get_db()
    for _ in range(count):
        db.execute(
            "INSERT INTO eggs (user_id, type) VALUES (?, 'random')",
            (user_id,),
        )
    db.commit()
