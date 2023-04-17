import torch
from diffusers import StableDiffusionPipeline
import random 

# To enable fp16 instead of fp32
#pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained("./models/gllwjsv9/", torch_dtype=torch.float16)
pipe.to("cuda")

#prompt = input("Enter your prompt: ")
rand_prefix = random.randint(0, 1000)

prompts = [
    ('Retro comic style artwork, highly detailed gllwjs cat, comic book cover, symmetrical, vibrant', 'comic'),
    ('Die-cut sticker, Cute gllwjs cat sticker, white background, illustration minimalism, vector, pastel colors', 'sticker'),
    ('3d fluffy gllwjs cat, closeup cute and adorable, cute big circular reflective eyes, long fuzzy fur, Pixar render, unreal engine cinematic smooth, intricate detail, cinematic', 'cute'),
 
]

inference_steps = [50, 100]
guidance_scales = [4, 7.5]

for prompt, name in prompts:
    for step in inference_steps:
        for scale in guidance_scales:
            for j in range(2):
                for i, image in enumerate(pipe(prompt, num_images_per_prompt=2, num_inference_steps=step, guidance_scale=scale).images):
                    # you can save the image with
                    image.save(f"./output/logo/{name}-{step}-{scale}-{i}{j}.png")