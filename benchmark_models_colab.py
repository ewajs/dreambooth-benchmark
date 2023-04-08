import torch
from diffusers import StableDiffusionPipeline
import os
import sys
import json


if len(sys.argv) != 3:
    print('ERROR: Wrong Number of arguments received!\nUSAGE: python benchmark_models_colab.py MODEL_FOLDER OUTPUT_FOLDER')
    exit()



models_root_path = sys.argv[1]
# Output Directory
benchmark_output_dir = sys.argv[2]

# Generate a list of dictionaries containing model paths and whether this is the base or extended model
models = [
    {
        'name': file,
        'path': os.path.join(models_root_path, file),
        'extended': True
    }
    for file in os.listdir(models_root_path)
]
# Add untrained base stable diffusion 2 (which is the base model thats retrained for each custom model) for benchmark comparison
models.append({
    'name': 'StableDiffusion2',
    'path': 'stabilityai/stable-diffusion-2',
    'extended': False
})

# Load propmts
with open('benchmark_prompts.json', 'r') as f:
    prompts = json.load(f)

# Prompt replacement data
instance_prompt_token = 'gllwjs cat'
class_prompt_token = 'a cat'

# Benchmarking parameters
num_inference_steps = [50, 100, 150]
guidance_scales = [4, 7.5, 9]

print("-------------- Stable Diffusion Dreambooth Training Benchmark --------------")
print(f"Found {len(models)} models to Benchmark")

for model in models:
    print(f"- Benchmarking model: {model['name']} - Extended: {model['extended']}")
    # Create the pipeline and set it to fp16 CUDA
    pipe = StableDiffusionPipeline.from_pretrained(model['path'])
    pipe.to("cuda")
    # Get the Token to be used in prompts for this model
    token = instance_prompt_token if model['extended'] else class_prompt_token
    print(f"- Using Token: '{token}' in prompts")
    # Now iterate over all settings and all prompts to create the images and save them
    for prompt in prompts:
        final_prompt = prompt['text'].replace('_TOKEN_', token)
        print(f"-- Generating Prompt: '{final_prompt}'")
        for step in num_inference_steps:
            for scale in guidance_scales:
                # Use seeds provided in config file, if not default to 0 and 1
                seeds = prompt['seeds'] if 'seeds' in prompt else [0, 1]
                # Generators seem to be changed when reused?
                generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
                for seed, generator in zip(seeds, generators):
                    print(f'--- Inference Steps: {step} - Guidance Scale: {scale} - Seed #: {seed}')
                    
                    output_dir = f"{benchmark_output_dir}{prompt['name']}/steps-{step}/scale-{scale}"
                    # Create directory if missing
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    
                    # Check if files for current seed already exists, if so, skip generation
                    partial_filenames = [
                        f"{output_dir}/{model['name']}-{seed}_0.png",
                        f"{output_dir}/{model['name']}-{seed}_1.png"
                    ]
                    
                    if all(map(os.path.exists, partial_filenames)):
                           print("Images already created! Skipping generation!")
                           continue
                    
                    # Diffuse!
                    result = pipe(
                        final_prompt, 
                        num_images_per_prompt=2, 
                        num_inference_steps=step, 
                        guidance_scale=scale,
                        generator=generator
                    )
                    for i, image in enumerate(result.images):
                        filename = f"{model['name']}-{seed}_{i}.png"
                        image.save(f"{output_dir}/{filename}")