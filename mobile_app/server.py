import base64
import io
import math
import random

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

MODEL_PATH = "/home/fer/Escritorio/dragons/dragon/app/vae_decoder.onnx"
LATENT_DIM = 1024
EXPECTED_NORM = math.sqrt(LATENT_DIM)  # ~32.0
NORM_STD = 0.71  # empirical std for 1024-dim standard normal


def compute_rarity(latent: list[float]) -> tuple[str, float]:
    """Return (tier, sigma) for a latent vector."""
    norm = math.sqrt(sum(x * x for x in latent))
    sigma = abs(norm - EXPECTED_NORM) / NORM_STD
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
def load_model():
    global session
    print("Loading VAE decoder...")
    session = ort.InferenceSession(MODEL_PATH, providers=["CPUExecutionProvider"])
    print("Model loaded!")


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
    # Convert CHW float [0,1] -> HWC uint8
    img = (output[0].transpose(1, 2, 0) * 255).clip(0, 255).astype(np.uint8)

    from PIL import Image

    pil_img = Image.fromarray(img)
    pil_img = pil_img.resize((512, 512), Image.LANCZOS)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


@app.post("/api/hatch")
def hatch():
    """Generate a dragon from a random latent vector."""
    latent = [randn_bm() for _ in range(LATENT_DIM)]
    img_b64 = decode_latent(latent)
    return {"latent": latent, "image": img_b64}


class BreedRequest(BaseModel):
    latent1: list[float]
    latent2: list[float]


@app.post("/api/breed")
def breed(req: BreedRequest):
    """Breed two dragons: average latents + small mutation, return new egg latent."""
    mean = [(a + b) / 2 for a, b in zip(req.latent1, req.latent2)]

    # Rescale the mean vector to preserve the average norm of the parents.
    # In high dimensions, averaging nearly-orthogonal vectors shrinks the norm
    # by ~1/sqrt(2), which would make all bred dragons artificially rare.
    norm1 = math.sqrt(sum(x * x for x in req.latent1))
    norm2 = math.sqrt(sum(x * x for x in req.latent2))
    target_norm = (norm1 + norm2) / 2
    mean_norm = math.sqrt(sum(x * x for x in mean))
    if mean_norm > 0:
        scale = target_norm / mean_norm
        mean = [v * scale for v in mean]

    mutated = [v + randn_bm() * 0.15 for v in mean]

    # If both parents share the same rarity tier, force the child to match
    # by rescaling its norm into the tier's sigma range.
    tier1, _ = compute_rarity(req.latent1)
    tier2, _ = compute_rarity(req.latent2)
    if tier1 == tier2:
        child_tier, _ = compute_rarity(mutated)
        if child_tier != tier1:
            # Sigma ranges per tier (min_sigma, mid_sigma)
            tier_mid = {
                "common": 0.5,
                "uncommon": 1.25,
                "rare": 1.75,
                "epic": 2.25,
                "legendary": 3.0,
            }
            # Pick the same side (above/below EXPECTED_NORM) as the parent average
            mid_sigma = tier_mid[tier1]
            if target_norm >= EXPECTED_NORM:
                forced_norm = EXPECTED_NORM + mid_sigma * NORM_STD
            else:
                forced_norm = EXPECTED_NORM - mid_sigma * NORM_STD
            child_norm = math.sqrt(sum(x * x for x in mutated))
            if child_norm > 0:
                mutated = [v * (forced_norm / child_norm) for v in mutated]

    return {"latent": mutated}


class HatchLatentRequest(BaseModel):
    latent: list[float]


@app.post("/api/hatch_latent")
def hatch_latent(req: HatchLatentRequest):
    """Hatch a bred egg with a known latent vector."""
    img_b64 = decode_latent(req.latent)
    return {"latent": req.latent, "image": img_b64}


@app.post("/api/generate_opponent")
def generate_opponent():
    """Generate a random opponent dragon for battles."""
    latent = [randn_bm() for _ in range(LATENT_DIM)]
    img_b64 = decode_latent(latent)
    return {"latent": latent, "image": img_b64}


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
