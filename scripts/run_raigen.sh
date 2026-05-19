#!/bin/bash
set -euo pipefail

root_dir="."
task="prof"       # "prof" or "coco"
prof="Doctor"     # profession name (prof) or COCO caption (coco)

accelerate launch --num_processes=4 hook_scripts/collect_activations.py \
    --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
    --hook_names unet.mid_block \
    --new_cached_activations_path "$root_dir/activations_sdxl" \
    --image_dir "$root_dir/dataset_sdxl" \
    --prompt "$prof" \
    --task "$task" \
    --num_examples_per_class 5000 \
    --batch_size 20 \
    --seed 0

# To use pre-trained MSAE checkpoints instead of training from scratch:
# comment out the activation collection step above and remove --train_sae below.
python msae/train_msae.py \
    --model_name "stabilityai/stable-diffusion-xl-base-1.0" \
    --dataset_path "$root_dir/activations_sdxl" \
    --d_in 1280 \
    --expansion_factor 16 \
    --group_sizes 2048 \
    --train_sae \
    --checkpoint_root "$root_dir/checkpoints_sdxl" \
    --image_dir "$root_dir/dataset_sdxl" \
    --task "$task" \
    --prompt "$prof" \
    --num_epochs 1 \
    --seed 0

python annotate.py \
    --base_prompt "$prof" \
    --checkpoint_root_dir "$root_dir/checkpoints_sdxl" \
    --task "$task" \
    --top_n_minority_neurons 10 \
    --minority_dir "sdxl_unet.mid_block_20480_2_[2048, 18432]/final/raigen_neurons/neurons_analysis_group_0_2048"
