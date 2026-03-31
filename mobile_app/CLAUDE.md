# DYNO - Dragon Mobile App

## Overview
A mobile-first browser-based dragon game where dragons are generated from a VAE (Variational Autoencoder). Players hatch eggs, breed dragons, battle them, and share them with friends via QR codes.

## Architecture

### Backend (`server.py`)
- **FastAPI** server on port 8000 (HTTPS with self-signed certs for mobile camera access)
- Loads a pre-trained VAE decoder (`../app/vae_decoder.onnx`, ~550MB)
- ONNX Runtime for inference: 1024-dim latent vector -> 256x256 -> 512x512 PNG
- No database — all state lives in the browser

### Frontend (`static/index.html`)
- **Single file**: all HTML, CSS, and JS in one ~1700-line file
- Vanilla JavaScript, no frameworks
- Mobile-first dark green theme, font: "Sono" (Google Fonts)
- External CDN libs: `qrcode-generator` (QR generation), `html5-qrcode` (camera scanning)

### API Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/hatch` | POST | Generate random dragon (random latent + decode) |
| `/api/hatch_latent` | POST | Decode a known latent vector into an image |
| `/api/breed` | POST | Breed two latent vectors into offspring |
| `/api/generate_opponent` | POST | Generate random opponent for battle |

## Core Data Model

### Dragon
```js
{ id, latent: float[1024], imgSrc: base64, name, parent1Id, parent2Id, received: bool }
```
- `latent` is the dragon's DNA — everything derives from it
- `received` flag indicates dragon was imported via QR scan

### Egg
```js
{ type: 'random' | 'bred', latent?, parent1Id?, parent2Id? }
```

## Key Systems

### Rarity (from L2 norm of latent vector)
- Expected norm: sqrt(1024) ~ 32.0, std: 0.71
- Sigma = |norm - expected| / std
- Tiers: Common (<1σ), Uncommon (1-1.5σ), Rare (1.5-2σ), Epic (2-2.5σ), Legendary (>2.5σ)

### Battle Stats (from latent chunks)
- Latent split into 4 chunks of 256 dims -> HP, ATK, DEF, SPD
- Each chunk's L2 norm mapped to [0,1], then scaled to base stats
- Rarity multiplier: Common 1.0x ... Legendary 2.1x

### Breeding
- Server-side: average parents' latents, preserve norm, add gaussian mutation (0.15 strength)
- If both parents share rarity tier, offspring is forced to same tier

### QR Dragon Sharing
- **4-bit quantization**: each latent float quantized to 4 bits, packed 2 per byte (512 bytes)
- Stores original L2 norm (4 extra bytes) to preserve exact rarity after decode
- Format: `DYNO:v2:<name>:<base64 payload>` (~730 chars total)
- QR version 22 (105 modules), M error correction
- Encode: `encodeDragonToQR(dragon)` -> string; Decode: `decodeDragonFromQR(text)` -> {latent, name}
- Reusable scanner: `scanDragonQR(callback)` — used for add-to-collection, breed, and battle
- Stats drift ~1-3 points due to quantization; rarity is 100% preserved via norm rescaling

### QR Scanning Notes
- Requires HTTPS for camera access on non-localhost (self-signed cert: `key.pem`, `cert.pem`)
- QR must be large on screen to scan reliably — container is nearly full viewport width
- Scanner qrbox is 85% of camera feed for maximum coverage
- Only processes QR codes starting with `DYNO:v2:` — ignores all others

## Running
```bash
cd dragon/mobile_app
bash run_server.sh  # starts HTTPS server on 0.0.0.0:8000
```
Access from phone via `https://<local-ip>:8000/` (accept self-signed cert warning).

## UI Structure
- **4 tabs**: Eggs (nursery) | Dragons (collection) | Breed | Battle
- **Top bar**: logo, egg count, dragon count, QR scan button
- **Overlays**: Dragon detail (stats + share QR + release), Hatch animation, Battle arena, QR display, QR scan
- Dragon detail shows: image, editable name, rarity badge + sigma, full stats (HP/ATK/DEF/SPD/Power), Share QR button, lineage (parents/grandparents/offspring), Release button
- Breed and Battle tabs have "Scan Friend's Dragon" buttons for multiplayer via QR
- Battle with friend uses real stats (no power scaling); random opponents are scaled to match player power

## Dependencies (pyproject.toml)
fastapi, uvicorn, numpy, pillow, onnxruntime (CPU), python-multipart
