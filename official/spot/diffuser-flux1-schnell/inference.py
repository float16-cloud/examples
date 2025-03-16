import torch
from diffusers import FluxPipeline

pipe = FluxPipeline.from_pretrained("../../model-weight/FLUX.1-schnell", torch_dtype=torch.bfloat16).to("cuda")

prompt = "A cat holding a sign that says hello world"
image = pipe(
    prompt,
    guidance_scale=0.0,
    num_inference_steps=4,
    max_sequence_length=256
).images[0]
image.save("./flux-schnell.png")