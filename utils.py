import torch
import os

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
            'name': 'StableDiffusion15',
            'path': 'runwayml/stable-diffusion-v1-5',
            'extended': False
        })
    return models