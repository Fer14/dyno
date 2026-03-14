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
    mutated = [v + randn_bm() * 0.15 for v in mean]
    return {"latent": mutated}


class HatchLatentRequest(BaseModel):
    latent: list[float]


@app.post("/api/hatch_latent")
def hatch_latent(req: HatchLatentRequest):
    """Hatch a bred egg with a known latent vector."""
    img_b64 = decode_latent(req.latent)
    return {"latent": req.latent, "image": img_b64}


# Serve frontend
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def index():
    return FileResponse("static/index.html")
