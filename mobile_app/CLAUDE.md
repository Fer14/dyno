# DYNO - Dragon Mobile App

## Overview
A mobile-first browser-based dragon game where dragons are generated from a VAE (Variational Autoencoder). Players hatch eggs, breed dragons, battle them, and share them with friends via QR codes. Now with user accounts, persistent SQLite storage, and Google Sign-In.

## Architecture

### Backend (`server.py`)
- **FastAPI** server on port 8000 (HTTPS with self-signed certs for mobile camera access)
- Loads a pre-trained VAE decoder (`../app/vae_decoder.onnx`, ~550MB)
- ONNX Runtime for inference: 1024-dim latent vector -> 256x256 -> 512x512 PNG
- **SQLite database** (`dyno.db`) for persistent storage via `bbdd.py`
- Google OAuth via Google Identity Services
- Config loaded from `.env` file (GOOGLE_CLIENT_ID)

### Database (`bbdd.py`)
- **SQLite** with WAL mode, thread-local connections
- Three tables: `users`, `dragons`, `eggs`
- Latent vectors stored as 4096-byte BLOBs (`np.float32.tobytes()`)
- Password hashing: `hashlib.sha256(salt + password)`
- Session tokens: `secrets.token_hex(32)` stored in `users.token`
- Google OAuth: `google_id` column on users table, auto-creates account on first Google login

### Frontend (`static/index.html`)
- **Single file**: all HTML, CSS, and JS in one file
- Vanilla JavaScript, no frameworks
- Mobile-first dark green theme, font: "Sono" (Google Fonts)
- External CDN libs: `qrcode-generator` (QR generation), `html5-qrcode` (camera scanning), Google Identity Services
- Auth token stored in `localStorage` (`dyno_token`)
- Dragon images cached in-memory `imageCache` Map, loaded via batch endpoint on login

### API Endpoints
| Endpoint | Method | Auth | Purpose |
|----------|--------|------|---------|
| `/api/register` | POST | No | Create user + 5 initial eggs |
| `/api/login` | POST | No | Authenticate, return token |
| `/api/auth/google` | POST | No | Google OAuth login/register |
| `/api/config` | GET | No | Public config (Google Client ID) |
| `/api/dragons` | GET | Yes | List user's dragons (with latents, no images) |
| `/api/eggs` | GET | Yes | List user's eggs |
| `/api/dragons/images` | POST | Yes | Batch decode latents -> base64 images |
| `/api/dragons/{id}/name` | PUT | Yes | Rename dragon |
| `/api/dragons/{id}` | DELETE | Yes | Release dragon |
| `/api/hatch` | POST | Yes | Hatch egg by egg_id, create dragon in DB |
| `/api/hatch_latent` | POST | No | Utility: decode latent to image (QR scan preview) |
| `/api/breed` | POST | Yes | Breed two dragons, create egg in DB |
| `/api/generate_opponent` | POST | No | Generate random opponent for battle |
| `/api/dragons/add_from_qr` | POST | Yes | Add scanned QR dragon (received=true) |
| `/api/battle/result` | POST | Yes | Win -> create egg; Lose -> delete dragon |

## Core Data Model

### Dragon (DB: `dragons` table)
```
id INTEGER PK, user_id FK, latent BLOB (4096 bytes), name TEXT,
parent1_id INTEGER, parent2_id INTEGER, received INTEGER, created_at TEXT
```
- Frontend also caches `imageCache.get(id)` for base64 image

### Egg (DB: `eggs` table)
```
id INTEGER PK, user_id FK, type TEXT ('random'|'bred'),
latent BLOB (null for random), parent1_id INTEGER, parent2_id INTEGER, created_at TEXT
```

### User (DB: `users` table)
```
id INTEGER PK, username TEXT UNIQUE, password TEXT, salt TEXT,
token TEXT UNIQUE, google_id TEXT UNIQUE, created_at TEXT
```

## Key Systems

### Auth
- Bearer token via `Authorization` header on all authenticated endpoints
- `authFetch()` helper in frontend injects token, handles 401 -> redirect to login
- Google Sign-In: frontend gets credential from Google Identity Services, sends to `/api/auth/google`, backend verifies via Google tokeninfo endpoint
- New users get 5 random eggs on registration

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
- Supports breeding with friend's QR-scanned dragon via `friend_latent` field

### QR Dragon Sharing
- **4-bit quantization**: each latent float quantized to 4 bits, packed 2 per byte (512 bytes)
- Stores original L2 norm (4 extra bytes) to preserve exact rarity after decode
- Format: `DYNO:v2:<name>:<base64 payload>` (~730 chars total)
- QR version 22 (105 modules), M error correction
- Top bar scan: adds dragon to DB via `/api/dragons/add_from_qr`
- Breed/battle scan: temporary reconstruction via `/api/hatch_latent` (not saved to DB)

### Visual Effects
- **Hatching**: egg shake -> crack -> rarity-colored particle burst (12 CSS particles) + screen flash + dragon entrance with golden glow
- **Breeding**: parent slots pulse toward center -> red center flash -> shimmer during API call -> egg materializes -> "Egg Created!" toast
- **Battle**: VS splash, attack pulse (green/red glow), slash impact marks, sprite knockback on hit, critical hit (gold text + hard shake when variance > 1.10), HP bar color transitions (green/yellow/red), low HP danger pulse (< 25%), victory burst + winner celebration pulse, defeat grayscale fade, arena divider line with glow pulse

## Running
```bash
cd dragon/mobile_app
bash run_server.sh  # starts HTTPS server on 0.0.0.0:8000
```
Access from phone via `https://<local-ip>:8000/` (accept self-signed cert warning).

## Environment
- `.env` file in `dragon/mobile_app/` with `GOOGLE_CLIENT_ID=...`
- Self-signed certs: `key.pem`, `cert.pem` (required for HTTPS/camera access)

## UI Structure
- **Login screen**: username/password + Google Sign-In button
- **4 tabs**: Eggs (nursery) | Dragons (collection) | Breed | Battle
- **Top bar**: logo, egg count, dragon count, QR scan button, username/logout
- **Overlays**: Dragon detail (stats + share QR + release), Hatch animation, Battle arena, QR display, QR scan
- Dragon detail shows: image, editable name, rarity badge + sigma, full stats (HP/ATK/DEF/SPD/Power), lineage, Share QR, Release
- Battle arena: 220px player sprite (bottom-left), 200px enemy sprite (top-right, mirrored), HP bars, stat cards, divider line, battle log

## Dependencies
- **Python** (pyproject.toml): fastapi, uvicorn, numpy, pillow, onnxruntime (CPU), python-multipart, python-dotenv
- **Frontend CDN**: qrcode-generator, html5-qrcode, Google Identity Services
- **Stdlib**: sqlite3, hashlib, secrets, json, urllib.request (Google token verification)
