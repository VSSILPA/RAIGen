from dataclasses import dataclass

import torch


@dataclass
class CacheActivationsRunnerConfig:
    hook_names: list[str] | None = None
    new_cached_activations_path: str | None = None
    prompt: str = "Doctor"
    image_dir:str = "dataset"
    column: str = "caption"
    model_name: str = "CompVis/stable-diffusion-v1-4"
    dtype: torch.dtype = torch.float16
    num_inference_steps: int = 50
    seed: int = 42 # 42 for training
    batch_size: int = 50
    task: str = "prof" # prof or coco
    output_or_diff: str = "output"
    max_num_examples: int | None = None
    cache_every_n_timesteps: int = 1
    guidance_scale: float = 9.0
    num_examples_per_class:int = 50 

    def __post_init__(self):
        # if self.new_cached_activations_path is None:
        self.new_cached_activations_path = f"{self.new_cached_activations_path}/{self.task}/{self.prompt}/{self.model_name.split('/')[-1]}"
        self.image_dir = f"{self.image_dir}/{self.task}/{self.prompt}/{self.model_name.split('/')[-1]}"
        if isinstance(self.hook_names, str):
            self.hook_names = [self.hook_names]
