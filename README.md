![logo](logo.png)



An interactive browser-based game where you hatch and breed **AI-generated dragons** using a **Variational Autoencoder (VAE)**.  
Click to hatch a unique dragon, collect them in your gallery, and combine any two to create entirely new hybrids.


## Features
- **Hatch Eggs** → Click a magical egg to generate a one-of-a-kind dragon sprite.  
- **Gallery** → Every dragon you hatch is saved to your collection.  
- **Breeding** → Select two dragons and combine their "DNA" (latent vectors) to create offspring with mixed traits.  
- **Offline Play** → Runs entirely in the browser with [ONNX.js](https://onnxruntime.ai/), no backend needed.  
- **Infinite Variety** → The VAE latent space allows for endless color, pattern, and shape combinations.  


## Demo Video



https://github.com/user-attachments/assets/fe76d6c2-069b-48b7-95bd-4160a0018ef5



## How It Works
1. **VAE Model**  
   - A pre-trained Variational Autoencoder encodes and decodes dragons from a 1024-dimensional latent vector space.  
   - The decoder runs in-browser via ONNX.js for instant rendering.

2. **Hatching**  
   - A new latent vector is sampled randomly.  
   - The vector is decoded into a 256×256 PNG image of a dragon.

3. **Breeding**  
   - Select two dragons from your gallery.  
   - Their latent vectors are averaged (and optionally mutated) to produce a new vector.  
   - The result is decoded into a hybrid dragon.


