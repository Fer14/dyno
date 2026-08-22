<p align="center">
  <img src="logo.png" alt="DYNO" width="320">
</p>

<p align="center">
  <strong>A mobile-first dragon collector where every dragon is a point in a VAE latent space.</strong>
</p>

<p align="center">
  <a href="#rarity">Rarity</a> ·
  <a href="#breeding">Breeding</a> ·
  <a href="#battles">Battles</a> ·
  <a href="#the-model">The Model</a> ·
  <a href="#running-it">Running It</a>
</p>

---

DYNO has no sprite sheet. Every dragon you will ever see is a **1024-dimensional vector** decoded on demand by a Variational Autoencoder trained on ~7,400 dragons that a diffusion model hallucinated from a single hand-drawn original.

That one design choice drives everything else in the game:

| Game concept | What it actually is |
|---|---|
| A dragon's identity | A 1024-d latent vector `z` |
| Its rarity | How far `‖z‖` sits from the expected norm `√1024` |
| Its stats | The L2 norms of four 256-d slices of `z` |
| Breeding | The midpoint of two vectors + Gaussian mutation |
| Evolution | Rescaling `‖z‖` outward while preserving direction |
| Sharing a dragon | A 4-bit quantization of `z` packed into a QR code |

---

## Rarity

A latent sampled from a standard normal has an expected L2 norm of `√1024 ≈ 32.0`, with an empirical standard deviation of `≈ 0.71`. Rarity is simply **how much of an outlier the dragon is**:

```
sigma = |‖z‖ − 32.0| / 0.71
```

Dragons near the center of the distribution look "normal". Dragons far out in either tail decode into unusual, saturated, strange-looking creatures — so *visual* weirdness and *mechanical* rarity are the same number. Nothing is rolled on a separate table.

| Tier | Sigma | Odds | Power budget |
|---|---|---|---|
| **Common** | `< 1.0σ` | 68.3% | 105 – 200 |
| **Uncommon** | `1.0 – 1.5σ` | 18.4% | 200 – 290 |
| **Rare** | `1.5 – 2.0σ` | 8.8% | 290 – 400 |
| **Epic** | `2.0 – 2.5σ` | 3.3% | 400 – 530 |
| **Legendary** | `2.5 – 3.5σ` | 1.2% | 530 – 700 |
| **Mythic** | `≥ 3.5σ` | 0.05% | 700 – 900+ |

Odds are the two-tailed normal probabilities of that sigma band — a Mythic is roughly a 1-in-2,100 hatch.

### From rarity to stats

Sigma sets a **total power budget**, with a discrete jump at each tier boundary so tiers never overlap. The latent then decides how that budget is *distributed*: `z` is split into four 256-d chunks, and each chunk's L2 norm becomes a proportion for one stat.

```
z[0:256]    → HP      target weight 48%
z[256:512]  → ATK     target weight 20%
z[512:768]  → DEF     target weight 13%
z[768:1024] → SPD     target weight 19%
```

Raw chunk proportions are blended 60/40 with the even baseline, which keeps builds distinct without producing a dragon with 4 HP. Level adds `1 + (level − 1) × 0.02` on top, so a maxed level-20 dragon carries a 1.38× multiplier.

Two dragons of the same tier therefore have the same *total* power but different shapes — and shape is what the battle AI reads (see [Battles](#battles)).

### Evolution

At **level 10** and **level 20** a dragon can evolve. Evolution normalizes `z` to a unit direction, then rescales it to the **next tier boundary**, preserving which side of `√1024` it was on:

```
z_evolved = (z / ‖z‖) × (32.0 ± σ_next × 0.71)
```

Because the direction is untouched, the dragon keeps its visual identity but its colors and features intensify — and it genuinely changes rarity tier, gaining the larger power budget. Two evolutions maximum.

---

## Breeding

<video src="https://github.com/Fer14/dyno/raw/mobile-app/video/breed.mp4" controls muted width="100%"></video>

> If the player above doesn't load, watch [`video/breed.mp4`](video/breed.mp4).

Pick two dragons. The server interpolates their latents:

1. **Average** the parents element-wise — `mean = (z₁ + z₂) / 2`.
2. **Rescale** to the mean of the parents' norms. Averaging two vectors shrinks the norm toward zero, which would quietly downgrade every child to Common. Rescaling to `(‖z₁‖ + ‖z₂‖) / 2` keeps the lineage's rarity intact.
3. **Mutate** — add Gaussian noise at `0.15` strength, so siblings are never identical.
4. **Lock the tier** — if both parents share a rarity tier, the child is forced to the midpoint sigma of that tier. Breeding two Epics always yields an Epic.

The result isn't a dragon yet: it's a **bred egg** that has to incubate. Breeding is the only way to reliably climb the rarity ladder, since random hatches are at the mercy of the normal distribution.

You can also breed with a **friend's dragon** by scanning their QR code — their latent is used as the second parent without ever being added to your collection.

### Eggs and incubation

| Egg | Source | Contents |
|---|---|---|
| **Starter** | 5 given at registration | Random latent, hatches instantly |
| **Random** | Battle victory | Fresh `randn` sample — pure distribution luck |
| **Golden** | Battle victory (`golden_chance_percent`) | Forced to Rare or above |
| **Bred** | Breeding | The mutated child latent |

Golden eggs skip the bad end of the distribution entirely: the latent is rescaled to a weighted target sigma — 40% Rare, 35% Epic, 20% Legendary, 5% Mythic.

Incubation timers live in `mobile_app/config.toml` and are validated server-side on hatch. You can shave time off with the **warming minigame**: tapping the egg removes 10 seconds, but only in rhythm — taps 0.8–2.0s apart count, taps under 0.4s apart **overheat** the egg and lock it out for 3 seconds. The server independently rate-limits warming to one call per second.

### Sharing via QR

A raw latent is 4 KB of floats — far too much for a QR code. Dragons are compressed to **524 bytes**:

- Each of the 1024 floats is quantized to **4 bits** and packed two-per-byte → 512 bytes.
- `min`, `max` and the **original L2 norm** are stored as float32 headers → 12 bytes.

On decode the latent is dequantized and then **rescaled back to the stored norm**, which matters: 4-bit quantization would otherwise perturb `‖z‖` enough to shift the dragon's rarity tier. The payload becomes `DYNO:v2:<name>:<base64>` (~730 chars) in a version-22 QR code, scannable straight from another phone's screen.

---

## Battles

<video src="https://github.com/Fer14/dyno/raw/mobile-app/video/battle.mp4" controls muted width="100%"></video>

> If the player above doesn't load, watch [`video/battle.mp4`](video/battle.mp4).

Battles are **Breath Forge**: a simultaneous-reveal mind game, not a stat comparison. Each turn you and your opponent secretly pick one of three actions and they resolve together.

| Action | Effect | Risk |
|---|---|---|
| 🔨 **FORGE** | +1 ember (max 5) | Takes **+20% damage** if struck this turn |
| ⚔️ **STRIKE** | Spend all embers, damage scaled by ember count | Fully **blocked** if they roar |
| 🔥 **ROAR** | Costs 1 ember, negates an incoming strike | Wasted if they weren't striking |

Embers are the whole game. Damage scales *super-linearly* with them:

| Embers | 0 | 1 | 2 | 3 | 4 | 5 |
|---|---|---|---|---|---|---|
| Multiplier | 0.4× | 1.0× | 2.3× | 3.9× | 5.8× | **8.0×** |

At **5 embers** your dragon glows, the screen pulses red, and STRIKE becomes **Dragon's Wrath** — an 8× cinematic hit that usually ends the fight. But charging to 5 means forging five times while visibly telegraphing it, and a single well-timed ROAR from your opponent wipes all five embers for nothing.

### Resolution

```
STRIKE  vs  ROAR    →  blocked, attacker loses all embers
STRIKE  vs  FORGE   →  "Caught forging!" — +20% damage
ROAR    vs  FORGE   →  ember wasted
FORGE   vs  FORGE   →  both escalate
```

That triangle is the tension: forging is how you win, and also when you're most vulnerable.

**Damage:**

```
dmg = ATK × (ember_mul + desperation) × (1 − DEF / (DEF + 50)) × [0.9 … 1.1]
```

Below **25% HP**, `desperation` adds a flat `+0.5` to the multiplier — a cornered dragon hits materially harder, so being nearly dead is a real comeback position rather than a formality. DEF is diminishing-returns, so stacking it never makes a dragon immune.

You have **7 seconds** per turn (`battle.turn_time_ms`): tap to select, tap again to confirm, and if the timer runs out you FORGE by default.

### Reading the opponent

Opponents aren't random button-mashers. Each derives a **personality** from its stat shape — the dominant stat, or the pair of stats if the top two are within 10% of each other:

| Dominant | Personality | Behavior |
|---|---|---|
| ATK | **Berserker** | Strikes early and often, barely blocks |
| DEF | **Fortress** | Hoards embers to 4+, blocks well |
| SPD | **Viper** | Constant 1-ember snap strikes |
| HP | **Titan** | Patient, then all-in below 25% HP |
| ATK+SPD | **Assassin** | Fast, precise 2-ember strikes |
| ATK+DEF | **Juggernaut** | Blocks first, then heavy strikes |
| ATK+HP | **Warlord** | Escalates as it gets wounded |
| DEF+SPD | **Ghost** | The best blocker in the game |
| DEF+HP | **Mountain** | Turtles to 4 embers, then commits |
| HP+SPD | **Survivor** | Aggressive early, cagey after turn 6 |

Before the first turn a hint line telegraphs it — *"This dragon radiates patience..."* means Fortress, so rushing a 2-ember strike into its block is a mistake. **Opponent rarity sets difficulty**, and the only thing difficulty changes is how well the AI predicts your strikes: block sense goes 5% → 15% → 25% → 40%. A Mythic opponent isn't just bigger, it reads you.

### Stakes

Winning gives **50 XP + up to 50 bonus** (from turns survived, reaching 5 embers, landing a Wrath kill), an egg, and Forge Mastery.

Losing is **permanent**. The dragon is deleted from the database. There is no revive.

### Forge Mastery

Forge Mastery (FM) is a per-dragon veterancy track, 0–20, earning +1 per win plus bonuses for reaching 5 embers or landing a Wrath kill. It doesn't raise stats — it changes what your dragon can *do*:

| FM | Perk | Effect |
|---|---|---|
| 1–3 | **Warm Scales** | −5% forge vulnerability per point |
| 4–6 | **Quick Ignition** | 8% chance per point of **+2 embers** from one FORGE |
| 13+ | **Ember Retention** | Keep 1 ember after a Wrath strike |
| 16+ | **Forge Adept** | Forge vulnerability removed entirely |

A veteran dragon forges faster and safer, which shifts the whole risk calculus — at FM 16+ forging costs you nothing, so the triangle bends in your favor. And FM is **partially heritable**: a bred egg inherits `⌊(parent1_FM + parent2_FM) / 3⌋`, so a lineage of champions produces children that start ahead. This is the main reason to keep an old winner alive rather than throw it into one more fight.

---

## The Model

### Gathering the data

The dataset started as **one hand-drawn dragon** ([`lambda/original.png`](lambda/original.png)).

To turn one dragon into thousands, a **Lambda Labs GPU machine** ran [`lambda/k.py`](lambda/k.py) — a **FLUX.1-Kontext-dev** image-editing pipeline that recolors the original via prompting. Each iteration draws **four independent colors** and asks the model to repaint four separate body parts:

```python
prompt = (
    f"Update the dragon's appearance with the following colors: main body skin to {random_color}, "
    f"horns to {random_color2}, wings to {random_color3}, and belly to {random_color4} (excluding the head). "
    "Preserve the background exactly as is—do not modify its color, lighting, or any visual details."
)
```

Two details make the resulting distribution useful rather than uniform noise:

- **Weighted palette.** 12 ordinary colors (red, blue, black, white…) share **70%** of the probability mass; 34 exotic ones (Fuchsia, Rose Gold, Pistachio, Platinum…) share the remaining **30%**. Common dragons genuinely look common, and unusual color combinations are *rare in the training data itself* — the scarcity is baked into the latent space before any rarity formula exists.
- **Background pinning.** The prompt explicitly forbids touching the background. Without it, the VAE spends its capacity modeling background variation instead of dragons.

Sampling four parts independently means the combinatorial space is ~46⁴, so the generated images cover a wide, structured range of the dragon's color manifold. Across **11 runs** this produced **~7,150 images**, joined by **247** hand-curated color variants — the ~7,400 total in `vae.py`'s folder list. (Smaller exploratory sets for wings, horns, heads and body size live in `dragons/images/` but are not in the training folder list — the trained model varies color, not silhouette.)

[`lambda/klein.py`](lambda/klein.py) is a newer experiment on the same idea, using **FLUX.2-klein** for structural edits (extra heads, bigger wings, spikes) rather than pure recoloring.

### Training the VAE

[`vae/vae.py`](vae/vae.py) trains a convolutional VAE on all ~7,400 images at 256×256.

```
Encoder:  256 → 128 → 64 → 32 → 16 → 8 → 4     (3 → 128 → 256 → 512 → 1024 → 2048 → 2048 ch)
          Conv2d(4,2,1) + BatchNorm + ReLU per block, then Flatten
Latent:   fc_mu / fc_var : 2048×4×4 → 1024
Decoder:  mirrored ConvTranspose stack, 4 → 256, final Conv2d(64→3) + Sigmoid
```

| Hyperparameter | Value |
|---|---|
| Latent dim | 1024 |
| Image size | 256×256 |
| Batch size | 32 |
| Optimizer | AdamW, lr `1e-5` |
| Reconstruction loss | MSE (`reduction="sum"`) |
| Total loss | `recon_weight × MSE + β × KL` |
| β schedule | Flat `1.0` (a 4-cycle cyclical schedule is implemented and selectable) |
| Epochs | 1000, resumed from a checkpoint at epoch 418 |

Design notes that mattered:

- **MSE over BCE**, because color fidelity is the entire point — these dragons are defined by their palette.
- **A 1024-d latent is deliberately oversized** for ~7,400 images. It's what makes `‖z‖` a smooth, well-behaved rarity signal and lets the four 256-d chunks carry enough independent information to derive stats from.
- **`reconstruction_weight` is the key knob.** Weighted high (100), reconstructions are sharp but the latent space becomes lumpy and interpolation breaks — bad for breeding. The final run uses `1.0`, trading some sharpness for a latent space where the midpoint of two dragons is still a dragon.
- **Checkpoint selection** tracks `recon + KL` rather than the β-weighted total, so best-model choice doesn't drift as β changes.

Training ran across successive `vae/VAE_TOTAL_*` directories, each resuming from the previous best checkpoint.

### Serving it

The decoder alone is exported to ONNX (~555 MB) and loaded once at startup by `onnxruntime` on CPU. The encoder is never needed at runtime — the game only ever goes *latent → image*, never the reverse.

**The weights are not in this repo.** At 555 MB they don't belong in git — export your own from a trained checkpoint with [`vae/convert.ipynb`](vae/convert.ipynb), then point `MODEL_PATH` in `mobile_app/server.py` at the result.

```
1024-d latent  →  ONNX decoder  →  256×256 float tensor  →  LANCZOS upscale  →  512×512 PNG (base64)
```

Latents are the source of truth in SQLite (4096-byte `float32` BLOBs); images are derived and cached in browser memory. A dragon costs 4 KB to store forever and is re-rendered on demand.

---

## Architecture

```
├── mobile_app/            The game
│   ├── server.py          FastAPI — auth, game logic, ONNX decode
│   ├── bbdd.py            SQLite layer (WAL, thread-local connections)
│   ├── config.toml        Tunables: dragon cap, hatch timers, golden odds, turn time
│   ├── static/index.html  The entire client — one file, vanilla JS, no framework
│   └── dyno.db            Users, dragons, eggs
├── vae/vae.py             VAE definition + training loop
├── lambda/k.py            Dataset generation (FLUX.1-Kontext on a Lambda GPU box)
├── dragons/images/        Original drawing + curated training images
├── app/                   Earlier in-browser prototype (expects the ONNX decoder here)
└── video/                 Demo captures
```

**Rarity, stats and battle resolution are implemented twice** — in `server.py` and again in `static/index.html`. The client copy drives the UI at 60fps without round-trips; the server copy is authoritative for anything persisted (hatching, breeding, evolution, incubation timers). Changing a rarity threshold means changing both.

| Table | Key columns |
|---|---|
| `users` | `username`, `password` (salted SHA-256), `salt`, `token`, `google_id` |
| `dragons` | `latent` BLOB, `name`, `parent1_id`, `parent2_id`, `xp`, `level`, `evolution`, `forge_mastery` |
| `eggs` | `type`, `latent`, `parent1_id`, `parent2_id`, `hatch_ready_at`, `last_warm_at` |

Auth is a bearer token in `localStorage`, or Google Sign-In verified server-side against Google's `tokeninfo` endpoint.

`app/` holds the original prototype: the same decoder running fully client-side via ONNX.js, with no server, no accounts, and no battles — just hatching and breeding. It still works, and it's the smaller thing to read first if you only care about the VAE.

---

## Running It

```bash
cd mobile_app
bash run_server.sh          # uvicorn on 0.0.0.0:8000 over HTTPS
```

Then open `https://<your-local-ip>:8000/` on your phone and accept the self-signed certificate warning.

**HTTPS is mandatory, not optional** — the QR scanner needs `getUserMedia`, and browsers only grant camera access on a secure origin. Generate the certs once:

```bash
cd mobile_app
openssl req -x509 -newkey rsa:2048 -nodes -keyout key.pem -out cert.pem -days 365 -subj "/CN=localhost"
```

You also need:

- **The ONNX decoder** — not distributed with the repo (555 MB). Export it from a trained checkpoint with [`vae/convert.ipynb`](vae/convert.ipynb) and set `MODEL_PATH` in `mobile_app/server.py` to wherever you put it. Nothing runs without this.
- **`mobile_app/.env`** — `GOOGLE_CLIENT_ID=...`, only if you want Google Sign-In. Username/password auth works without it.
- **`mobile_app/config.toml`** — check the values before playing. Timers and drop rates in the committed file are tuned for *development* (fast hatches, generous golden-egg odds), not for a real game.

Dependencies are managed with [uv](https://docs.astral.sh/uv/): `fastapi`, `uvicorn`, `numpy`, `pillow`, `onnxruntime`, `python-dotenv`.

---

## Design Notes

A few decisions worth calling out, since they're the non-obvious parts:

**Rarity is a property of the generative model, not a lookup table.** Because rarity is `‖z‖`'s distance from the norm, rare dragons *look* rare — the same number that grants the stat budget also makes the image unusual. Nothing has to be authored per tier.

**Battles are decisions, not dice.** Stats set the ceiling; ember management decides the fight. A Common dragon with 5 embers hits for 8× and can kill a Legendary that mistimed a block. That's why the AI has legible personalities and a telegraphed hint — the player is supposed to be *reading* the opponent, and difficulty scales the AI's ability to read *back*.

**Permanent death is what gives Forge Mastery weight.** An FM 16 dragon is genuinely irreplaceable — the perks aren't purchasable and its lineage passes only a third of them on. Every battle with a veteran is a real wager.

**The vulnerability window is the core loop.** Forging is both the path to power and the moment of exposure, so the interesting question every single turn is *"do they think I'm about to strike?"*
