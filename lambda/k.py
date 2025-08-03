import random
import time

import torch
import tqdm
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image

pipe = FluxKontextPipeline.from_pretrained(
    "black-forest-labs/FLUX.1-Kontext-dev", torch_dtype=torch.bfloat16
)
pipe.to("cuda")

input_image = load_image("original.png")


colors = [
    "Scarlet",
    "Ruby",
    "Maroon",
    "Rose",
    "Magenta",
    "Fuchsia",
    "Turquoise",
    "Cyan",
    "Lime",
    "Emerald",
    "Mint",
    "Pistachio",
    "Gold",
    "Bronze",
    "Copper",
    "Lavender",
    "Beige",
    "Sepia",
    "Steel",
    "Silver",
    "Platinum",
    "Pastel Pink",
    "Pastel Blue",
    "Pastel Green",
    "Pastel Yellow",
    "Pastel Purple",
    "Gold",
    "Silver",
    "Bronze",
    "Copper",
    "Rose Gold",
    "Ochre",
    "Dark grey",
    "Gray",
]

normal_colors = [
    "Red",
    "Blue",
    "Yellow",
    "Green",
    "Orange",
    "Purple",
    "Black",
    "Soft black",
    "White",
    "Gray",
    "Brown",
    "Pink",
]

# Combine all unique colors
all_colors = normal_colors + colors

# Set weights: 0.65 probability distributed among normal_colors,
# and 0.35 probability distributed among the rest
weights = []
for color in all_colors:
    if color in normal_colors:
        weights.append(0.7 / len(normal_colors))
    else:
        weights.append(0.3 / len(colors))

for i in tqdm.tqdm(range(10000)):
    random_color = random.choices(all_colors, weights=weights, k=1)[0]
    random_color2 = random.choices(all_colors, weights=weights, k=1)[0]
    random_color3 = random.choices(all_colors, weights=weights, k=1)[0]
    random_color4 = random.choices(all_colors, weights=weights, k=1)[0]

    print(random_color, random_color2, random_color3, random_color4)

    prompt = (
        f"Update the dragon's appearance with the following colors: main body skin to {random_color}, "
        f"horns to {random_color2}, wings to {random_color3}, and belly to {random_color4} (excluding the head). "
        "Preserve the background exactly as is—do not modify its color, lighting, or any visual details."
    )

    image = pipe(image=input_image, prompt=prompt, guidance_scale=7.5).images[0]

    # 💾 Save the image
    image.save(f"images/kontext_{time.time()}.png")
