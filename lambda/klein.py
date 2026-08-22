import torch
from diffusers import Flux2KleinPipeline
from diffusers.utils import load_image

device = "cuda"
dtype = torch.bfloat16

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-base-4B", torch_dtype=dtype
)
pipe.enable_sequential_cpu_offload()  # save some VRAM by offloading the model to CPU

input_image = load_image("/home/fer/Escritorio/dragons/dragon/lambda/original.png")

prompt = "Make the dragon have 2 heads"
image = pipe(
    image=input_image,
    prompt=prompt,
    height=512,
    width=512,
    guidance_scale=5.0,
    num_inference_steps=20,
    generator=torch.Generator(device=device).manual_seed(42),
).images[0]
image.save("flux-klein.png")


# prompt = (
#     "Change the dragon's skin to a glowy pink color, with a radiant pinky shimmer "
#     "across its entire body. Keep the same cartoon style, pose, and background exactly as is."
# )ç


# prompt = "Change the dragon to have bigger wings"


# prompt = "Change the dragon skin to have cartoon spikes all over its body"
