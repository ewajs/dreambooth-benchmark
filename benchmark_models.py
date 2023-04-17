import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import os
import json
from utils import generate_benchmark_images, read_models


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