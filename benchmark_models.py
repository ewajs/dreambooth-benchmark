import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import os
import json

def generate_benchmark_images(pipe, token, settings, base_image_filename):
    # Now iterate over all settings and all prompts to create the images and save them
    for prompt in settings['prompts']:
        final_prompt = prompt['text'].replace('_TOKEN_', token)
        print(f"-- Generating Prompt: '{final_prompt}'")
        for step in settings['num_inference_steps']:
            for scale in settings['guidance_scales']:
                # Use seeds provided in config file, if not default to 0 and 1
                seeds = prompt['seeds'] if 'seeds' in prompt else [0, 1]
                # Generators seem to be changed when reused?
                generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]
                for seed, generator in zip(seeds, generators):
                    print(f'--- Inference Steps: {step} - Guidance Scale: {scale} - Seed #: {seed}')
                    output_dir = f"{settings['output_dir']}/{prompt['name']}/steps-{step}/scale-{scale}"
                    # Create directory if missing
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    # Check if files for current seed already exists, if so, skip generation
                    partial_filenames = [
                        f"{output_dir}/{base_image_filename}-{seed}_0.png",
                        f"{output_dir}/{base_image_filename}-{seed}_1.png"
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
                        filename = f"{base_image_filename}-{seed}_{i}.png"
                        image.save(f"{output_dir}/{filename}")

def read_models(models_root_path, add_untrained = False):
    # Generate a list of dictionaries containing model paths and whether this is the base or extended model
    models = [
        {
            'name': file,
            'path': os.path.join(models_root_path, file),
            'extended': True
        }
        for file in os.listdir(models_root_path)
    ]
    if add_untrained:
        # Add untrained base stable diffusion 2 (which is the base model thats retrained for each custom model) for benchmark comparison
        models.append({
            'name': 'StableDiffusion2',
            'path': 'stabilityai/stable-diffusion-2',
            'extended': False
        })
    return models

if __name__ == "__main__":
    models = read_models('./models', True) # Read all models plus regular Stable Diffusion
   
    # Load propmts
    with open('benchmark_prompts.json', 'r') as f:
        prompts = json.load(f)

    # Prompt replacement data
    instance_prompt_token = 'gllwjs cat'
    class_prompt_token = 'a cat'

    benchmark_settings = {
        'num_inference_steps': [50, 100, 150],
        'guidance_scales': [4, 7.5, 9],
        'output_dir': './output/benchmark',
        'prompts': prompts
    }

    print("-------------- Stable Diffusion Dreambooth Training Benchmark --------------")
    print(f"Found {len(models)} models to Benchmark")

    for model in models:
        print(f"- Benchmarking model: {model['name']} - Extended: {model['extended']}")
        # Get the Token to be used in prompts for this model
        token = instance_prompt_token if model['extended'] else class_prompt_token
        print(f"- Using Token: '{token}' in prompts")

        # Create the pipeline and set it to fp16 CUDA
        pipe = StableDiffusionPipeline.from_pretrained(model['path'], revision="fp16", torch_dtype=torch.float16)
        pipe.to("cuda")

        generate_benchmark_images(pipe, token, benchmark_settings, model['name'])
        
        # If we're using a local model, check for existing checkpoints and benchmark those as well
        if (model['extended']):
            checkpoints = [
                {
                    'steps': checkpoint[11:], 
                    'path': os.path.join(model['path'], checkpoint)
                }
                for checkpoint in os.listdir(model['path']) if checkpoint.startswith('checkpoint-')
            ]
            # If there are checkpoints, run the benchmark for those
            if len(checkpoints) > 0:
                print(f'Found {len(checkpoints)} additional training checkpoints to run inference!')
                for checkpoint in checkpoints:
                    unet = UNet2DConditionModel.from_pretrained(f"{checkpoint['path']}/unet", torch_dtype=torch.float16)
                    text_encoder = CLIPTextModel.from_pretrained(f"{checkpoint['path']}/text_encoder", torch_dtype=torch.float16)
                    pipe = DiffusionPipeline.from_pretrained(model['path'], unet=unet, text_encoder=text_encoder, torch_dtype=torch.float16)
                    pipe.to("cuda")
                    # Attach the number of checkpoint steps to the end of the base image filename
                    base_image_filename = f"{model['name']}_{checkpoint['steps']}"
                    print(f"-- Benchmarking checkpoint at {checkpoint['steps']} steps.")
                    generate_benchmark_images(pipe, token, benchmark_settings, base_image_filename)