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
| `/api/battle/result` | POST | Yes | Win -> XP + egg; Lose -> delete dragon |
| `/api/eggs/{id}/warm` | POST | Yes | Warm egg to reduce incubation time |
| `/api/dragons/{id}/evolve` | POST | Yes | Evolve dragon (scale latent to next rarity tier) |

## Core Data Model

### Dragon (DB: `dragons` table)
```
id INTEGER PK, user_id FK, latent BLOB (4096 bytes), name TEXT,
parent1_id INTEGER, parent2_id INTEGER, received INTEGER, created_at TEXT,
xp INTEGER DEFAULT 0, level INTEGER DEFAULT 1, evolution INTEGER DEFAULT 0
```
- Frontend also caches `imageCache.get(id)` for base64 image

### Egg (DB: `eggs` table)
```
id INTEGER PK, user_id FK, type TEXT ('random'|'bred'),
latent BLOB (null for random), parent1_id INTEGER, parent2_id INTEGER, created_at TEXT,
hatch_ready_at TEXT, last_warm_at TEXT
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
- **6 tiers:**
  | Tier | Sigma | Stat Bonus | Probability | Color | Badge CSS |
  |------|-------|------------|-------------|-------|-----------|
  | Common | < 1.0σ | 1.0x | 68.27% (1 in 1) | Gray (#9ca3af) | `rarity-common` |
  | Uncommon | 1.0 – 1.5σ | 1.20x | 18.37% (1 in 5) | Green (#34d399) | `rarity-uncommon` |
  | Rare | 1.5 – 2.0σ | 1.45x | 8.81% (1 in 11) | Blue (#60a5fa) | `rarity-rare` |
  | Epic | 2.0 – 2.5σ | 1.75x | 3.31% (1 in 30) | Purple (#a78bfa) | `rarity-epic` |
  | Legendary | 2.5 – 3.5σ | 2.10x | 1.20% (1 in 84) | Gold gradient (#f59e0b → #ef4444) | `rarity-legendary` |
  | Mythic | ≥ 3.5σ | 2.60x | 0.05% (1 in 2,149) | Red-purple animated shimmer (#dc2626 ↔ #7c3aed) | `rarity-mythic` |
- Rarity particle colors (hatching): common=#9ca3af, uncommon=#34d399, rare=#60a5fa, epic=#a78bfa, legendary=#f59e0b, mythic=#ef4444

### Battle Stats (from latent chunks)
- Latent split into 4 chunks of 256 dims -> HP, ATK, DEF, SPD
- Each chunk's L2 norm mapped to [0,1], then scaled to base stats
- Rarity multiplier: Common 1.0x, Uncommon 1.20x, Rare 1.45x, Epic 1.75x, Legendary 2.10x
- Level multiplier: `1 + (level - 1) * 0.02` (max level 20 = 1.38x)

### XP / Leveling
- Dragons earn XP from battle victories: base 50 XP + up to 50 bonus from QTE performance
- Level threshold: `level * 100` XP to advance (level 1 = 100 XP, level 2 = 200 XP, etc.)
- Max level: 20. Stats scale with level via multiplier
- XP only awarded on wins (lost dragons are deleted, so XP would be wasted)

### Battle QTEs (Quick Time Events)
- Each turn, player performs a "Shrinking Ring" QTE: a circle shrinks toward a target ring
- Player taps when rings overlap. Timing determines grade:
  - **Perfect** (60-85% progress): 1.5x attack damage / 0.5x defense damage
  - **Good** (40-95%): 1.0x attack / 0.75x defense
  - **Miss** (outside or timeout): 0.5x attack / 1.0x defense (full damage taken)
- Offensive QTE: player's turn to attack. Defensive QTE: enemy attacks, player blocks
- QTE timing window scales with opponent rarity (harder opponents = faster ring)
- QTE performance contributes bonus XP at end of battle

### Egg Incubation
- Eggs have a real-time countdown: random eggs = 5 minutes, bred eggs = 15 minutes
- `hatch_ready_at` timestamp set on creation, validated server-side on hatch
- Initial eggs (new user) have NULL hatch_ready_at = immediately hatchable
- **Warming minigame**: tap egg to reduce time by 10 seconds per tap
  - Rhythm-based: taps 0.8-2.0s apart = warm zone (effective)
  - Taps < 0.4s apart = overheat (3s cooldown, no effect)
  - Rate-limited server-side (min 1s between /warm requests)

### Dragon Evolution
- At level 10 (1st evolution) and level 20 (2nd evolution), dragon can evolve
- Evolution scales the latent vector's L2 norm to the next rarity tier boundary
- Direction vector preserved: same visual identity but more intense colors/patterns
- Tier boundaries: 1.0σ, 1.5σ, 2.0σ, 2.5σ, 3.0σ. Evolution pushes to next boundary
- Max 2 evolutions (evolution column: 0, 1, or 2)
- New image generated from evolved latent, stats recalculated with new rarity

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
- **Battle**: VS splash, QTE shrinking ring overlay, attack pulse (green/red glow), slash impact marks, sprite knockback on hit, critical hit (gold text + hard shake when variance > 1.10), HP bar color transitions (green/yellow/red), low HP danger pulse (< 25%), victory burst + winner celebration pulse + XP/level-up display, defeat grayscale fade, arena divider line with glow pulse
- **QTE**: Shrinking ring with green target zone, result text (gold PERFECT / green GOOD / red MISS)
- **Evolution**: Golden glow pulse animation -> screen flash -> image crossfade to evolved form
- **Egg warming**: Temperature gauge (green/red), egg tap animation, overheat state with red glow
- **Level-up**: Golden pop animation with "LEVEL UP!" text after battle victory

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
- **Overlays**: Dragon detail (stats + evolve + share QR + release), Hatch animation, Warming minigame, Battle arena (with QTE), QR display, QR scan
- Dragon detail shows: image, editable name, rarity badge + sigma, level + XP bar, evolve button, full stats (HP/ATK/DEF/SPD/Power), lineage, Share QR, Release
- Battle arena: 220px player sprite (bottom-left), 200px enemy sprite (top-right, mirrored), HP bars, stat cards with level, QTE shrinking ring, divider line, battle log
- Egg cards show: incubation timer countdown, locked/ready state, type label
- Dragon cards show: level badge (top-right), evolution stars (bottom-right), rarity badge (top-left)

## Dependencies
- **Python** (pyproject.toml): fastapi, uvicorn, numpy, pillow, onnxruntime (CPU), python-multipart, python-dotenv
- **Frontend CDN**: qrcode-generator, html5-qrcode, Google Identity Services
- **Stdlib**: sqlite3, hashlib, secrets, json, urllib.request (Google token verification)
