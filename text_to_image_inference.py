import torch
from diffusers import StableDiffusionPipeline
import random 

# To enable fp16 instead of fp32
#pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", revision="fp16", torch_dtype=torch.float16)
pipe = StableDiffusionPipeline.from_pretrained("./models/gllwjsv4/", revision="fp16", torch_dtype=torch.float16)
pipe.to("cuda")

#prompt = input("Enter your prompt: ")
rand_prefix = random.randint(0, 1000)

prompts = [
    ('A portrait of gllwjs cat, fernando botero style, award winning painting, mate, trending in artstation, oil painting', 'botero2')
    # ('a photo of gllwjs cat by the window, detailed fur, cannon DSLR, 300mm lens', 'photo-cannon-300mm'),
    # ('a painting of gllwjs cat in the jungle in the style of Frida Khalo', 'frida-khalo'),
    # ('a watercolor drawing of gllwjs cat and the moon', 'watercolor'),
    # ('a 3d render of gllwjs cat, detailed fur', '3d-model'),
    # ('gllwjs cat as an anime character, detailed fur, big eyes', 'anime'),
    # ('pencil drawing of gllwjs cat and a telescope, detailed fur', 'pencil-drawing'),
    # ('gllwjs cat full body 3d cartoon', '3d-cartoon'),
    # ('street grafitti painting of gllwjs cat', 'grafitti'),
    # ('a painting of gllwjs cat by escher', 'escher'),
    #('3 d rendered character portrait of gllwjs cat, 3D, octane render, depth of field, unreal engine 5, concept art, vibrant colors, glow, trending on artstation, ultra high detail, ultra realistic, cinematic lighting, focused, 8 k', '3d-character'),
    #('gllwjs cat sitting in a red cushion, full body, detailed fur, screenshot in a typical pixar movie, disney infinity 3 star wars style, volumetric lighting, subsurface scattering, photorealistic, octane render, medium shot, studio ghibli, pixar and disney animation, sharp, rendered in unreal engine 5, anime key art by greg rutkowski and josh black, bloom, dramatic lighting', 'pixar'),
    #('oil painting of gllwjs cat by the window in the style of van gogh', 'vangogh'),
]

inference_steps = [50, 100, 150]
guidance_scales = [4, 7.5, 9]

for prompt, name in prompts:
    for step in inference_steps:
        for scale in guidance_scales:
            for i, image in enumerate(pipe(prompt, num_images_per_prompt=2, num_inference_steps=step, guidance_scale=scale).images):
                # you can save the image with
                image.save(f"./output/{name}-{step}-{scale}-{i}.png")