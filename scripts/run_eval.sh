#!/bin/bash

root_dir="."
task="prof"       # "coco" or "prof"
prompt="Doctor"     # profession name (prof) or COCO caption (coco)
top_n=10           # 5 for coco, 10 for prof

minority_dir="sdxl_unet.mid_block_20480_2_[2048, 18432]/final/raigen_neurons/neurons_analysis_group_0_2048"
checkpoint_root="$root_dir/checkpoints_sdxl"

# annotations_path points to where annotate.py (run_raigen.sh) saves its output.
# If using pre-computed annotations from raigen_annotations/, set it to:
#   annotations_path="$root_dir/raigen_annotations/$task/${prompt}_annotations.json"
annotations_path="$checkpoint_root/$task/$prompt/$minority_dir/minority_prompt_annotations.json"

image_dir="$root_dir/raigen_eval_outputs/generated_images_sdxl/$task/$prompt"
metrics_dir="$root_dir/raigen_eval_outputs/metrics"

num_samples=1000

mkdir -p "$image_dir"

python "$root_dir/generate_images.py" \
    --output_dir "$image_dir" \
    --prompt "$prompt" \
    --task "$task" \
    --num_samples "$num_samples" \
    --batch_size 20 \
    --num_gpus 1 \
    --seed 0

python "$root_dir/evaluate.py" \
    --annotations_path "$annotations_path" \
    --image_dir "$image_dir" \
    --task "$task" \
    --metrics_dir "$metrics_dir" \
    --top_n_minority_neurons "$top_n" \
    --num_samples_per_attr "$num_samples" \
    --model_name "Qwen/Qwen3-VL-8B-Instruct"

python "$root_dir/evaluate.py" \
    --annotations_path "$annotations_path" \
    --image_dir "$image_dir" \
    --task "$task" \
    --metrics_dir "$metrics_dir" \
    --top_n_minority_neurons "$top_n" \
    --num_samples_per_attr "$num_samples" \
    --model_name "meta-llama/Llama-4-Scout-17B-16E-Instruct"

