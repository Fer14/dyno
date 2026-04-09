import base64
import io
import json
import math
import os
import random
import sqlite3
import tomllib
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import bbdd

load_dotenv()
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")

# Load config
_config_path = Path(__file__).parent / "config.toml"
with open(_config_path, "rb") as f:
    CONFIG = tomllib.load(f)

MODEL_PATH = "/home/fer/Escritorio/dragons/dragon/app/vae_decoder.onnx"
LATENT_DIM = 1024
EXPECTED_NORM = math.sqrt(LATENT_DIM)  # ~32.0
NORM_STD = 0.71  # empirical std for 1024-dim standard normal
MAX_DRAGONS = CONFIG["dragons"]["max_dragons"]
RANDOM_HATCH_MINUTES = CONFIG["eggs"]["random_hatch_minutes"]
BRED_HATCH_MINUTES = CONFIG["eggs"]["bred_hatch_minutes"]
GOLDEN_CHANCE = CONFIG["eggs"]["golden_chance_percent"] / 100.0

# Dragon name generation (ported from frontend)
NAME_PREFIXES = [
    "Ash", "Blaze", "Cinder", "Drake", "Ember", "Fang", "Glimmer", "Hex",
    "Ignis", "Jade", "Kael", "Luna", "Myst", "Nyx", "Onyx", "Pyro",
    "Quill", "Rune", "Storm", "Thorn", "Umbra", "Vex", "Wyvern", "Xeno",
    "Yara", "Zephyr", "Aether", "Bolt", "Crimson", "Dusk",
]
NAME_SUFFIXES = [
    "ion", "ara", "ius", "ora", "yx", "en", "is", "ax", "um", "el",
    "ir", "os", "an", "ur", "ix", "al", "on", "as", "er", "us",
]
NAME_TITLES = [
    "the Fierce", "the Wise", "the Swift", "the Mighty", "the Shadow",
    "the Bright", "the Ancient", "the Young", "the Bold", "the Silent",
    "the Brave", "the Cunning", "the Noble", "the Wild", "the Gentle",
]


def generate_name() -> str:
    prefix = random.choice(NAME_PREFIXES)
    suffix = random.choice(NAME_SUFFIXES)
    title = random.choice(NAME_TITLES)
    return f"{prefix}{suffix} {title}"


def compute_rarity(latent: list[float]) -> tuple[str, float]:
    """Return (tier, sigma) for a latent vector."""
    norm = math.sqrt(sum(x * x for x in latent))
    sigma = abs(norm - EXPECTED_NORM) / NORM_STD
    if sigma >= 3.5:
        return "mythic", sigma
    if sigma >= 2.5:
        return "legendary", sigma
    if sigma >= 2.0:
        return "epic", sigma
    if sigma >= 1.5:
        return "rare", sigma
    if sigma >= 1.0:
        return "uncommon", sigma
    return "common", sigma


app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# Load model once at startup
session: ort.InferenceSession | None = None


@app.on_event("startup")
def startup():
    global session
    print("Loading VAE decoder...")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("Model loaded!")
    bbdd.init_db()
    print("Database initialized!")


def randn_bm():
    u, v = 0.0, 0.0
    while u == 0:
        u = random.random()
    while v == 0:
        v = random.random()
    return math.sqrt(-2.0 * math.log(u)) * math.cos(2.0 * math.pi * v)


def decode_latent(latent: list[float]) -> str:
    """Run VAE decoder and return base64-encoded PNG."""
    latent_np = np.array(latent, dtype=np.float32).reshape(1, LATENT_DIM)
    result = session.run(["reconstructed"], {"latent": latent_np})
    output = result[0]  # [1, 3, 256, 256]
    img = (output[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    from PIL import Image

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


# --- Auth dependency ---

def get_current_user(request: Request) -> dict:
    """Extract and validate Bearer token. Returns user dict or raises 401."""
    auth = request.headers.get("Authorization", "")
    if not auth.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid token")
    token = auth[7:]
    user = bbdd.get_user_by_token(token)
    if user is None:
        raise HTTPException(status_code=401, detail="Invalid token")
    return user


# --- Auth endpoints ---

class AuthRequest(BaseModel):
    username: str
    password: str


@app.post("/api/register")
def register(req: AuthRequest):
    if not req.username.strip() or not req.password:
        raise HTTPException(status_code=400, detail="Username and password required")
    try:
        user_id, token = bbdd.create_user(req.username.strip(), req.password)
    except sqlite3.IntegrityError:
        raise HTTPException(status_code=409, detail="Username already taken")
    bbdd.create_initial_eggs(user_id)
    return {"token": token, "user_id": user_id, "username": req.username.strip()}


@app.post("/api/login")
def login(req: AuthRequest):
    result = bbdd.login_user(req.username.strip(), req.password)
    if result is None:
        raise HTTPException(status_code=401, detail="Invalid credentials")
    user_id, token = result
    return {"token": token, "user_id": user_id, "username": req.username.strip()}


class GoogleAuthRequest(BaseModel):
    credential: str  # Google ID token (JWT)


def _verify_google_token(id_token: str) -> dict:
    """Verify Google ID token via Google's tokeninfo endpoint. Returns payload or raises."""
    url = "https://oauth2.googleapis.com/tokeninfo?id_token=" + id_token
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            payload = json.loads(resp.read())
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid Google token")
    if payload.get("aud") != GOOGLE_CLIENT_ID:
        raise HTTPException(status_code=401, detail="Token not issued for this app")
    return payload


@app.post("/api/auth/google")
def google_login(req: GoogleAuthRequest):
    """Authenticate with Google. Creates account on first login."""
    payload = _verify_google_token(req.credential)
    google_id = payload["sub"]
    email = payload.get("email", "")
    name = payload.get("name", "")

    user_id, token, username, is_new = bbdd.google_auth(google_id, email, name)
    if is_new:
        bbdd.create_initial_eggs(user_id)
    return {"token": token, "user_id": user_id, "username": username}


# --- Collection endpoints ---

@app.get("/api/dragons")
def list_dragons(request: Request):
    user = get_current_user(request)
    return {"dragons": bbdd.get_dragons(user["id"])}


@app.get("/api/eggs")
def list_eggs(request: Request):
    user = get_current_user(request)
    return {"eggs": bbdd.get_eggs(user["id"])}


class DragonImagesRequest(BaseModel):
    dragon_ids: list[int]


@app.post("/api/dragons/images")
def get_dragon_images(req: DragonImagesRequest, request: Request):
    """Batch decode dragon latents to images."""
    user = get_current_user(request)
    images = {}
    for did in req.dragon_ids:
        dragon = bbdd.get_dragon(did, user["id"])
        if dragon:
            images[str(did)] = decode_latent(dragon["latent"])
    return {"images": images}


class RenameRequest(BaseModel):
    name: str


@app.put("/api/dragons/{dragon_id}/name")
def rename_dragon(dragon_id: int, req: RenameRequest, request: Request):
    user = get_current_user(request)
    if not bbdd.update_dragon_name(dragon_id, user["id"], req.name):
        raise HTTPException(status_code=404, detail="Dragon not found")
    return {"ok": True}


@app.delete("/api/dragons/{dragon_id}")
def release_dragon(dragon_id: int, request: Request):
    user = get_current_user(request)
    if not bbdd.delete_dragon(dragon_id, user["id"]):
        raise HTTPException(status_code=404, detail="Dragon not found")
    return {"ok": True}


# --- Utility endpoints ---

class HatchLatentRequest(BaseModel):
    latent: list[float]


@app.post("/api/hatch_latent")
def hatch_latent(req: HatchLatentRequest):
    """Decode a latent vector into an image (utility, used for QR scan preview)."""
    img_b64 = decode_latent(req.latent)
    return {"latent": req.latent, "image": img_b64}


# --- Game endpoints ---

class HatchRequest(BaseModel):
    egg_id: int
    name: str | None = None


@app.post("/api/hatch")
def hatch(req: HatchRequest, request: Request):
    """Hatch an egg: generate/use latent, create dragon, delete egg."""
    user = get_current_user(request)
    egg = bbdd.get_egg(req.egg_id, user["id"])
    if egg is None:
        raise HTTPException(status_code=404, detail="Egg not found")

    # Check incubation time (NULL = immediately hatchable for legacy/starter eggs)
    if egg.get("hatch_ready_at"):
        from datetime import datetime, timezone
        ready = datetime.fromisoformat(egg["hatch_ready_at"]).replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        if now < ready:
            raise HTTPException(status_code=400, detail="Egg not ready yet")

    # Check dragon limit
    current_dragons = bbdd.get_dragons(user["id"])
    if len(current_dragons) >= MAX_DRAGONS:
        raise HTTPException(status_code=400, detail=f"Dragon limit reached ({MAX_DRAGONS})")

    if egg["type"] in ("random", "golden"):
        latent = [randn_bm() for _ in range(LATENT_DIM)]
        # Golden eggs: force rarity to rare or above (sigma >= 1.5)
        if egg["type"] == "golden":
            norm = math.sqrt(sum(x * x for x in latent))
            if norm == 0:
                norm = 1.0
            direction = [x / norm for x in latent]
            # Weighted random: mostly rare/epic, sometimes legendary, very rarely mythic
            roll = random.random()
            if roll < 0.40:
                target_sigma = 1.5 + random.random() * 0.5   # Rare (1.5-2.0)
            elif roll < 0.75:
                target_sigma = 2.0 + random.random() * 0.5   # Epic (2.0-2.5)
            elif roll < 0.95:
                target_sigma = 2.5 + random.random() * 1.0   # Legendary (2.5-3.5)
            else:
                target_sigma = 3.5 + random.random() * 0.5   # Mythic (3.5-4.0)
            if norm >= EXPECTED_NORM:
                new_norm = EXPECTED_NORM + target_sigma * NORM_STD
            else:
                new_norm = EXPECTED_NORM - target_sigma * NORM_STD
            latent = [d * new_norm for d in direction]
    else:
        latent = egg["latent"]

    img_b64 = decode_latent(latent)
    name = req.name if req.name else generate_name()
    dragon_id = bbdd.create_dragon(
        user["id"], latent, name,
        parent1_id=egg["parent1_id"],
        parent2_id=egg["parent2_id"],
    )
    bbdd.delete_egg(req.egg_id, user["id"])

    return {
        "dragon_id": dragon_id,
        "latent": latent,
        "image": img_b64,
        "name": name,
        "parent1_id": egg["parent1_id"],
        "parent2_id": egg["parent2_id"],
    }


class BreedRequest(BaseModel):
    dragon1_id: int
    dragon2_id: int | None = None
    friend_latent: list[float] | None = None


@app.post("/api/breed")
def breed(req: BreedRequest, request: Request):
    """Breed two dragons. Returns new egg."""
    user = get_current_user(request)

    dragon1 = bbdd.get_dragon(req.dragon1_id, user["id"])
    if dragon1 is None:
        raise HTTPException(status_code=404, detail="Dragon 1 not found")

    if req.friend_latent:
        latent2 = req.friend_latent
        parent2_id = None
    elif req.dragon2_id is not None:
        dragon2 = bbdd.get_dragon(req.dragon2_id, user["id"])
        if dragon2 is None:
            raise HTTPException(status_code=404, detail="Dragon 2 not found")
        latent2 = dragon2["latent"]
        parent2_id = dragon2["id"]
    else:
        raise HTTPException(status_code=400, detail="Must provide dragon2_id or friend_latent")

    latent1 = dragon1["latent"]

    # Average latents
    mean = [(a + b) / 2 for a, b in zip(latent1, latent2)]

    # Rescale to preserve average norm
    norm1 = math.sqrt(sum(x * x for x in latent1))
    norm2 = math.sqrt(sum(x * x for x in latent2))
    target_norm = (norm1 + norm2) / 2
    mean_norm = math.sqrt(sum(x * x for x in mean))
    if mean_norm > 0:
        scale = target_norm / mean_norm
        mean = [v * scale for v in mean]

    # Add mutation
    mutated = [v + randn_bm() * 0.15 for v in mean]

    # If both parents share rarity tier, force child to match
    tier1, _ = compute_rarity(latent1)
    tier2, _ = compute_rarity(latent2)
    if tier1 == tier2:
        child_tier, _ = compute_rarity(mutated)
        if child_tier != tier1:
            tier_mid = {
                "common": 0.5,
                "uncommon": 1.25,
                "rare": 1.75,
                "epic": 2.25,
                "legendary": 3.0,
                "mythic": 4.0,
            }
            mid_sigma = tier_mid[tier1]
            if target_norm >= EXPECTED_NORM:
                forced_norm = EXPECTED_NORM + mid_sigma * NORM_STD
            else:
                forced_norm = EXPECTED_NORM - mid_sigma * NORM_STD
            child_norm = math.sqrt(sum(x * x for x in mutated))
            if child_norm > 0:
                mutated = [v * (forced_norm / child_norm) for v in mutated]

    egg_result = bbdd.create_egg(
        user["id"], "bred", mutated,
        parent1_id=dragon1["id"],
        parent2_id=parent2_id,
        random_hatch_minutes=RANDOM_HATCH_MINUTES,
        bred_hatch_minutes=BRED_HATCH_MINUTES,
    )
    return {"egg_id": egg_result["egg_id"], "latent": mutated, "hatch_ready_at": egg_result["hatch_ready_at"]}


@app.post("/api/generate_opponent")
def generate_opponent():
    """Generate a random opponent dragon for battles."""
    latent = [randn_bm() for _ in range(LATENT_DIM)]
    img_b64 = decode_latent(latent)
    return {"latent": latent, "image": img_b64}


class QRAddRequest(BaseModel):
    latent: list[float]
    name: str


@app.post("/api/dragons/add_from_qr")
def add_from_qr(req: QRAddRequest, request: Request):
    """Add a dragon received via QR scan."""
    user = get_current_user(request)
    current_dragons = bbdd.get_dragons(user["id"])
    if len(current_dragons) >= MAX_DRAGONS:
        raise HTTPException(status_code=400, detail=f"Dragon limit reached ({MAX_DRAGONS})")
    img_b64 = decode_latent(req.latent)
    dragon_id = bbdd.create_dragon(
        user["id"], req.latent, req.name, received=True,
    )
    return {"dragon_id": dragon_id, "image": img_b64}


class BattleResultRequest(BaseModel):
    dragon_id: int
    won: bool
    xp_bonus: int = 0


@app.post("/api/battle/result")
def battle_result(req: BattleResultRequest, request: Request):
    """Record battle result. Win: earn XP + random egg. Lose: dragon dies."""
    user = get_current_user(request)

    dragon = bbdd.get_dragon(req.dragon_id, user["id"])
    if dragon is None:
        raise HTTPException(status_code=404, detail="Dragon not found")

    if req.won:
        xp_amount = 50 + min(max(req.xp_bonus, 0), 50)
        xp_result = bbdd.award_xp(req.dragon_id, user["id"], xp_amount)
        egg_type = "golden" if random.random() < GOLDEN_CHANCE else "random"
        egg_result = bbdd.create_egg(user["id"], egg_type, random_hatch_minutes=RANDOM_HATCH_MINUTES, bred_hatch_minutes=BRED_HATCH_MINUTES)
        return {
            "ok": True,
            "egg_id": egg_result["egg_id"],
            "egg_type": egg_type,
            "hatch_ready_at": egg_result["hatch_ready_at"],
            "xp_gained": xp_amount,
            "new_xp": xp_result["xp"] if xp_result else 0,
            "new_level": xp_result["level"] if xp_result else 1,
            "leveled_up": xp_result["leveled_up"] if xp_result else False,
        }
    else:
        bbdd.delete_dragon(req.dragon_id, user["id"])
        return {"ok": True}


# --- Egg warming ---

@app.post("/api/eggs/{egg_id}/warm")
def warm_egg(egg_id: int, request: Request):
    """Warm an egg to reduce incubation time."""
    user = get_current_user(request)
    result = bbdd.warm_egg(egg_id, user["id"], seconds=10)
    if result is None:
        raise HTTPException(status_code=400, detail="Too soon or egg not found")
    return {"ok": True, "hatch_ready_at": result["hatch_ready_at"]}


# --- Dragon evolution ---

@app.post("/api/dragons/{dragon_id}/evolve")
def evolve_dragon(dragon_id: int, request: Request):
    """Evolve a dragon: scale latent norm to next rarity tier."""
    user = get_current_user(request)
    dragon = bbdd.get_dragon(dragon_id, user["id"])
    if dragon is None:
        raise HTTPException(status_code=404, detail="Dragon not found")

    level = dragon.get("level", 1)
    evolution = dragon.get("evolution", 0)

    required_level = 10 * (evolution + 1)  # 10, 20
    if level < required_level:
        raise HTTPException(status_code=400, detail=f"Requires level {required_level}")
    if evolution >= 2:
        raise HTTPException(status_code=400, detail="Max evolution reached")

    latent = dragon["latent"]
    norm = math.sqrt(sum(x * x for x in latent))
    if norm == 0:
        raise HTTPException(status_code=400, detail="Invalid latent vector")

    direction = [x / norm for x in latent]
    current_sigma = abs(norm - EXPECTED_NORM) / NORM_STD

    # Find next rarity tier boundary
    tier_boundaries = [1.0, 1.5, 2.0, 2.5, 3.5, 4.0]
    target_sigma = None
    for boundary in tier_boundaries:
        if boundary > current_sigma:
            target_sigma = boundary
            break
    if target_sigma is None:
        target_sigma = current_sigma + 0.5  # already legendary, push further

    # Preserve direction (above/below EXPECTED_NORM)
    if norm >= EXPECTED_NORM:
        new_norm = EXPECTED_NORM + target_sigma * NORM_STD
    else:
        new_norm = EXPECTED_NORM - target_sigma * NORM_STD

    evolved_latent = [d * new_norm for d in direction]

    bbdd.update_dragon_latent(dragon_id, user["id"], evolved_latent, evolution + 1)
    img_b64 = decode_latent(evolved_latent)
    new_rarity, new_sigma = compute_rarity(evolved_latent)

    return {
        "ok": True,
        "latent": evolved_latent,
        "image": img_b64,
        "evolution": evolution + 1,
        "rarity": new_rarity,
        "sigma": new_sigma,
    }


# Serve frontend
@app.get("/api/config")
def get_config():
    """Return public config to the frontend."""
    return {"google_client_id": GOOGLE_CLIENT_ID, "max_dragons": MAX_DRAGONS}


app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
