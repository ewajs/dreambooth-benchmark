import torch
from diffusers import StableDiffusionPipeline, DiffusionPipeline, UNet2DConditionModel
from transformers import CLIPTextModel
import os
import json
from utils import generate_benchmark_images


if __name__ == "__main__":
    model = {
        'name': 'gllwjsv9',
        'path': './models/gllwjsv9',
        'extended': True,
    }
   
    # Load propmts
    with open('benchmark_scheduler_prompts.json', 'r') as f:
        prompts = json.load(f)

    # Prompt replacement data
    instance_prompt_token = 'gllwjs cat'
    class_prompt_token = 'a cat'

    benchmark_settings = {
        'num_inference_steps': [20, 50],
        'guidance_scales': [4, 10],
        'output_dir': './output/benchmark_scheduler/gllwjsv9',
        'prompts': prompts
    }

    print("-------------- Stable Diffusion Scheduler Benchmark --------------")

   
    print(f"- Benchmarking model: {model['name']} - Extended: {model['extended']}")
    # Get the Token to be used in prompts for this model
    token = instance_prompt_token if model['extended'] else class_prompt_token
    print(f"- Using Token: '{token}' in prompts")

    # Create the pipeline and set it to fp16 CUDA
    pipe = StableDiffusionPipeline.from_pretrained(model['path'], revision="fp16", torch_dtype=torch.float16)
    pipe.to("cuda")
    scheduler_config = pipe.scheduler.config
    for scheduler in pipe.scheduler.compatibles:
        # Change the current scheduler a new scheduler using the same configuration
        pipe.scheduler = scheduler.from_config(scheduler_config)
        scheduler_name = scheduler.__name__.replace("Scheduler", "")
        print(f'-- Benchmarking {scheduler_name}Scheduler...')
        # Run the benchmark for this Scheduler
        generate_benchmark_images(pipe, token, benchmark_settings, scheduler_name)
