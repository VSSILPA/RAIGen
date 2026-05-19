import json
import os
import shutil
import sys
import random
import numpy as np
from pathlib import Path
from PIL import Image
from diffusers.utils.import_utils import is_xformers_available
from diffusers import DiffusionPipeline,StableDiffusionPipeline, StableDiffusion3Pipeline, EulerDiscreteScheduler
try:
    from diffusers import FluxPipeline
except ImportError:  # optional dependency in some diffusers installs
    FluxPipeline = None

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
from accelerate import Accelerator
from accelerate.utils import gather_object
from datasets import Array2D, Dataset, Features, Value
from datasets.fingerprint import generate_fingerprint
from tqdm import tqdm

from config import CacheActivationsRunnerConfig
from activation_tracer import run_with_tracer
torch.backends.cuda.matmul.allow_tf32 = True
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

TORCH_STRING_DTYPE_MAP = {torch.float16: "float16", torch.float32: "float32"}


class CacheActivationsRunner:
    def __init__(self, cfg: CacheActivationsRunnerConfig):
        self.cfg = cfg
        self.accelerator = Accelerator()


        print(
            f"[Rank {self.accelerator.local_process_index}] "
            f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}, "
            f"device_count={torch.cuda.device_count()}, "
            f"current_device={torch.cuda.current_device()}",
            flush=True
        )


        if self.cfg.hook_names is not None:
            if self.cfg.model_name == "stabilityai/stable-diffusion-2":
                scheduler = EulerDiscreteScheduler.from_pretrained(self.cfg.model_name, subfolder="scheduler")
                self.pipe = StableDiffusionPipeline.from_pretrained(self.cfg.model_name, scheduler=scheduler, torch_dtype=self.cfg.dtype)
            elif self.cfg.model_name ==  "CompVis/stable-diffusion-v1-4":
                self.pipe =  StableDiffusionPipeline.from_pretrained(
                    self.cfg.model_name, torch_dtype=self.cfg.dtype, safety_checker=None)
            elif self.cfg.model_name ==  "stabilityai/stable-diffusion-xl-base-1.0":
                self.pipe =  DiffusionPipeline.from_pretrained(
                    self.cfg.model_name, torch_dtype=self.cfg.dtype, use_safetensors=True, safety_checker=None)
            elif self.cfg.model_name ==  "stabilityai/stable-diffusion-3-medium-diffusers":
                self.pipe = StableDiffusion3Pipeline.from_pretrained(self.cfg.model_name, torch_dtype=self.cfg.dtype)
            elif self.cfg.model_name == "black-forest-labs/FLUX.1-schnell":
                if FluxPipeline is None:
                    raise ImportError("FluxPipeline is not available in this diffusers installation")
                self.pipe = FluxPipeline.from_pretrained(self.cfg.model_name, torch_dtype=self.cfg.dtype)



            if is_xformers_available():
                print("Enabling xFormers memory efficient attention")
                self.pipe.unet.enable_xformers_memory_efficient_attention()
            self.pipe.to(self.accelerator.device)
            # self.pipe.vae.to("cpu")
            self.pipe.set_progress_bar_config(disable=True)
            
            if hasattr(self.pipe, "scheduler"):
                self.scheduler = self.pipe.scheduler
                self.scheduler.set_timesteps(self.cfg.num_inference_steps, device="cpu")
                self.scheduler_timesteps = self.scheduler.timesteps
            else:
                self.scheduler = None
                self.scheduler_timesteps = torch.arange(self.cfg.num_inference_steps)


            self.features_dict = {hookpoint: None for hookpoint in self.cfg.hook_names}

            all_prompts = []

            if self.cfg.task == "prof":
                all_prompts = [f'A photo of a {self.cfg.prompt}.'] * self.cfg.num_examples_per_class
            elif self.cfg.task == "coco":
                all_prompts = [self.cfg.prompt] * self.cfg.num_examples_per_class

            self.dataset = Dataset.from_dict({"caption": all_prompts})
            self.dataset = self.dataset.shuffle(self.cfg.seed)
            if limit := self.cfg.max_num_examples:
                self.dataset = self.dataset.select(range(limit))

            self.num_examples = len(self.dataset)
            self.dataloader = self.get_batches(self.dataset, self.cfg.batch_size)
            self.n_buffers = len(self.dataloader)

    @staticmethod
    def get_batches(items, batch_size):
        num_batches = (len(items) + batch_size - 1) // batch_size
        batches = []

        for i in range(num_batches):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, len(items))
            batch = items[start_index:end_index]
            batches.append(batch)

        return batches

    @staticmethod
    def _consolidate_shards(
        source_dir: Path, output_dir: Path, copy_files: bool = True
    ) -> Dataset:
        """Consolidate sharded datasets into a single directory without rewriting data.

        Each of the shards must be of the same format, aka the full dataset must be able to
        be recreated like so:

        ```
        ds = concatenate_datasets(
            [Dataset.load_from_disk(str(shard_dir)) for shard_dir in sorted(source_dir.iterdir())]
        )

        ```

        Sharded dataset format:
        ```
        source_dir/
            shard_00000/
                dataset_info.json
                state.json
                data-00000-of-00002.arrow
                data-00001-of-00002.arrow
            shard_00001/
                dataset_info.json
                state.json
                data-00000-of-00001.arrow
        ```

        And flattens them into the format:

        ```
        output_dir/
            dataset_info.json
            state.json
            data-00000-of-00003.arrow
            data-00001-of-00003.arrow
            data-00002-of-00003.arrow
        ```

        allowing the dataset to be loaded like so:

        ```
        ds = datasets.load_from_disk(output_dir)
        ```

        Args:
            source_dir: Directory containing the sharded datasets
            output_dir: Directory to consolidate the shards into
            copy_files: If True, copy files; if False, move them and delete source_dir
        """
        first_shard_dir_name = "shard_00000"  # shard_{i:05d}

        assert source_dir.exists() and source_dir.is_dir()
        assert (
            output_dir.exists()
            and output_dir.is_dir()
            and not any(p for p in output_dir.iterdir() if not p.name == ".tmp_shards")
        )
        if not (source_dir / first_shard_dir_name).exists():
            raise Exception(f"No shards in {source_dir} exist!")

        transfer_fn = shutil.copy2 if copy_files else shutil.move

        # Move dataset_info.json from any shard (all the same)
        transfer_fn(
            source_dir / first_shard_dir_name / "dataset_info.json",
            output_dir / "dataset_info.json",
        )

        arrow_files = []
        file_count = 0

        for shard_dir in sorted(source_dir.iterdir()):
            if not shard_dir.name.startswith("shard_"):
                continue

            # state.json contains arrow filenames
            state = json.loads((shard_dir / "state.json").read_text())

            for data_file in state["_data_files"]:
                src = shard_dir / data_file["filename"]
                new_name = f"data-{file_count:05d}-of-{len(list(source_dir.iterdir())):05d}.arrow"
                dst = output_dir / new_name
                transfer_fn(src, dst)
                arrow_files.append({"filename": new_name})
                file_count += 1

        new_state = {
            "_data_files": arrow_files,
            "_fingerprint": None,  # temporary
            "_format_columns": None,
            "_format_kwargs": {},
            "_format_type": None,
            "_output_all_columns": False,
            "_split": None,
        }

        # fingerprint is generated from dataset.__getstate__ (not including _fingerprint)
        with open(output_dir / "state.json", "w") as f:
            json.dump(new_state, f, indent=2)

        ds = Dataset.load_from_disk(str(output_dir))
        fingerprint = generate_fingerprint(ds)
        del ds

        with open(output_dir / "state.json", "r+") as f:
            state = json.loads(f.read())
            state["_fingerprint"] = fingerprint
            f.seek(0)
            json.dump(state, f, indent=2)
            f.truncate()

        if not copy_files:  # cleanup source dir
            shutil.rmtree(source_dir)

        return Dataset.load_from_disk(output_dir)

    @torch.no_grad()
    def _create_shard(
        self,
        buffer: torch.Tensor,  # buffer shape: "bs num_inference_steps+1 d_sample_size d_in",
        hook_name: str,
    ) -> Dataset:
        batch_size, _, d_sample_size, d_in = buffer.shape

        # Filter buffer based on every N steps
        buffer = buffer[:, :: self.cfg.cache_every_n_timesteps, :, :]

        activations = buffer.reshape(-1, d_sample_size, d_in)
        timesteps = self.scheduler_timesteps[
            :: self.cfg.cache_every_n_timesteps
        ].repeat(batch_size)

        shard = Dataset.from_dict(
            {
                "activations": activations,
                "timestep": timesteps,
            },
            features=self.features_dict[hook_name],
        )
        return shard


    def create_dataset_feature(self, hook_name, d_in, d_out):
        self.features_dict[hook_name] = Features(
            {
                "activations": Array2D(
                    shape=(
                        d_in,
                        d_out,
                    ),
                    dtype=TORCH_STRING_DTYPE_MAP[self.cfg.dtype],
                ),
                "timestep": Value(dtype="uint16"),
            }
        )


    
    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # for multi-GPU
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


    @torch.no_grad()
    def run(self) -> dict[str, Dataset]:
        ### Paths setup
        assert self.cfg.new_cached_activations_path is not None


        image_dir = f"{self.cfg.image_dir}/images"
        os.makedirs(image_dir, exist_ok=True)
        global_img_index = 0

        base_seed = self.cfg.seed
        unique_seed = base_seed + self.accelerator.process_index


        self.set_seed(unique_seed)


        all_img_seeds = []

        save_hook_name= self.cfg.hook_names[0]
        
        # final_cached_activation_paths = {
        #     n: Path(os.path.join(self.cfg.new_cached_activations_path, n))
        #     for n in save_hook_name
        # }

        final_cached_activation_paths = {
            save_hook_name: Path(os.path.join(self.cfg.new_cached_activations_path, save_hook_name))
        }


        if self.accelerator.is_main_process:
            for path in final_cached_activation_paths.values():
                path.mkdir(exist_ok=True, parents=True)
                if any(path.iterdir()):
                    raise Exception(
                        f"Activations directory ({path}) is not empty. Please delete it or specify a different path. Exiting the script to prevent accidental deletion of files."
                    )

            tmp_cached_activation_paths = {
                n: path / ".tmp_shards/"
                for n, path in final_cached_activation_paths.items()
            }
            for path in tmp_cached_activation_paths.values():
                path.mkdir(exist_ok=False, parents=False)

        self.accelerator.wait_for_everyone()

        ### Create temporary sharded datasets
        if self.accelerator.is_main_process:
            print(f"Started caching {self.num_examples} activations")

        for i, batch in tqdm(
            enumerate(self.dataloader),
            desc="Caching activations",
            total=self.n_buffers,
            disable=not self.accelerator.is_main_process,
        ):
            with self.accelerator.split_between_processes(batch) as prompt:
                prompt = prompt[self.cfg.column]
                seeds = [random.randint(0, 2**32 - 1) for _ in range(len(prompt))]
                generators = [torch.Generator(device="cuda").manual_seed(seed) for seed in seeds]

                img, acts_cache = run_with_tracer(
                    self.pipe,
                    prompt=prompt,
                    positions_to_cache=self.cfg.hook_names,
                    num_inference_steps=self.cfg.num_inference_steps,
                    guidance_scale=self.cfg.guidance_scale,
                    generator=generators,
                    save_input=(self.cfg.output_or_diff == "diff"),
                    save_output=True,
                    unconditional=False,            # or True if you want the uncond branch
                )


            self.accelerator.wait_for_everyone()


            # Compute output or diff
            if self.cfg.output_or_diff == "diff":
                gathered_tensor = acts_cache["output"][save_hook_name] - acts_cache["input"][save_hook_name]
            else:
                gathered_tensor = acts_cache["output"][save_hook_name]
            gathered_buffer = gather_object([gathered_tensor]) 
            all_imgs = gather_object(img)
            batch_seeds = gather_object(seeds)

            if self.accelerator.is_main_process:
                gathered_acts = torch.cat(gathered_buffer, dim=0)  # (N, T, ..., ...)

                if self.features_dict[save_hook_name] is None:
                    self.create_dataset_feature(
                        save_hook_name,
                        gathered_acts.shape[-2],
                        gathered_acts.shape[-1],
                    )

                print(f"{save_hook_name=} {gathered_acts.shape=}")

                shard = self._create_shard(gathered_acts, save_hook_name)
                shard.save_to_disk(
                    f"{tmp_cached_activation_paths[save_hook_name]}/shard_{i:05d}",
                    num_shards=1,
                )

                del gathered_acts


                for image in all_imgs:
                    image = image.resize((256, 256), Image.Resampling.LANCZOS)
                    image.save(os.path.join(image_dir, f"image_{global_img_index}.png"))
                    global_img_index += 1

                all_img_seeds.extend(batch_seeds)
        datasets = {}

        if self.accelerator.is_main_process:
            torch.save(all_img_seeds, os.path.join(image_dir,'seeds_img.pt'))

            for hook_name, path in tmp_cached_activation_paths.items():
                datasets[hook_name] = self._consolidate_shards(
                    path, final_cached_activation_paths[hook_name], copy_files=False
                )
                print(f"Consolidated the dataset for hook {hook_name}")

        return datasets

   