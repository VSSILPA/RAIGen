import argparse
import torch
import os
import re
import math
import multiprocessing as mp

from diffusers import DiffusionPipeline

from diffusers.utils import logging
logging.disable_progress_bar() 


def generate(gpu_id, start_idx, end_idx, args):
    torch.cuda.set_device(gpu_id)
    device = torch.device(f"cuda:{gpu_id}")
    pipeline = DiffusionPipeline.from_pretrained(args.model_name, torch_dtype=torch.float16)
    pipeline = pipeline.to(device)
    
    pipeline.set_progress_bar_config(disable=True)
    for batch_start in range(start_idx, end_idx, args.batch_size):
        current_batch = min(args.batch_size, end_idx - batch_start)
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_start)
        images = pipeline(
            prompt=[args.prompt] * current_batch,
            negative_prompt=[args.negative_prompt] * current_batch if args.negative_prompt else None,
            generator=generator,
            guidance_scale=args.guidance_scale,
            num_images_per_prompt=1,
        ).images

        for i, image in enumerate(images):
            image_idx = batch_start + i
            image.save(os.path.join(args.im_dir, f"image_{image_idx}.jpg"))

    torch.cuda.empty_cache()


            
def create_images(args):
    
        if os.path.exists(args.im_dir):
            existing_files = [f for f in os.listdir(args.im_dir) if f.endswith('.jpg')]
            existing_indices = []
            for filename in existing_files:
                match = re.match(r"image_(\d+)\.jpg", filename)
                if match:
                    existing_indices.append(int(match.group(1)))
            start_index = max(existing_indices) + 1 if existing_indices else 0
        else:
            start_index = 0

        if start_index >= args.num_samples:
            print(f"Image generation complete. {start_index} images already exist in {args.im_dir}")
            return

        remaining = args.num_samples - start_index
        per_gpu = math.ceil(remaining / args.num_gpus)

        ctx = mp.get_context("spawn")
        procs = []
        for gpu_id in range(args.num_gpus):
            gpu_start = start_index + gpu_id * per_gpu
            gpu_end = min(args.num_samples, gpu_start + per_gpu)
            if gpu_start >= gpu_end:
                continue
            proc = ctx.Process(target=generate, args=(gpu_id, gpu_start, gpu_end, args))
            proc.start()
            procs.append(proc)

        for proc in procs:
            proc.join()
            if proc.exitcode != 0:
                raise RuntimeError("One of the GPU generation workers failed.")

        print(f"Image generation complete. {args.num_samples} images saved in {args.im_dir}")
        
   
if __name__=='__main__':
    parser = argparse.ArgumentParser(
                    prog = 'Generate Stable Diffusion images')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--task', type=str, default='coco', choices=['coco', 'prof'],
                        help="'prof' wraps prompt as 'a photo of a <prompt>'; 'coco' uses prompt as-is.")
    parser.add_argument('--model_name', type=str, default='stabilityai/stable-diffusion-xl-base-1.0')
    parser.add_argument('--negative_prompt', type=str, default='')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--num_samples', type=int, default=1000) 
    parser.add_argument('--batch_size', type=int, default=50)
    parser.add_argument('--num_gpus', type=int, default=1)
    parser.add_argument('--seed', type=int, default=0)
    
    args = parser.parse_args()
    if args.task == 'prof':
        args.prompt = f"a photo of a {args.prompt}"
    print(f"Generating images for prompt: {args.prompt!r}")
    args.im_dir = os.path.join(args.output_dir)
    os.makedirs(args.im_dir, exist_ok=True)
    create_images(args)