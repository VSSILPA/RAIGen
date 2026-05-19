import os
import json
import random
from pathlib import Path
import argparse
import sys
from tqdm import tqdm
from statistics import mean
from eval_utils import query_llama4, query_qwen
import torch
from transformers import AutoProcessor
from transformers import AutoModelForImageTextToText
from transformers import Llama4ForConditionalGeneration

parser = argparse.ArgumentParser(description="Attribute identification evaluation")
parser.add_argument("--annotations_path", type=str, required=True, help="Path to the annotations JSON file")
parser.add_argument("--image_dir", required=True)
parser.add_argument("--task", type=str, default="prof")
parser.add_argument("--metrics_dir", type=str, default="raigen_eval_outputs/metrics")
parser.add_argument("--model_name", type=str, default="meta-llama/Llama-4-Scout-17B-16E-Instruct",
                    choices=["Qwen/Qwen3-VL-8B-Instruct", "meta-llama/Llama-4-Scout-17B-16E-Instruct"])
parser.add_argument("--num_samples_per_attr", type=int, default=100)
parser.add_argument("--top_n_minority_neurons", type=int, default=None)
parser.add_argument("--seed", type=int, default=42)
args = parser.parse_args()
random.seed(args.seed)


def load_image_dataset(image_dir):
    base_dir = Path(image_dir)
    img_dir = base_dir / "images" if (base_dir / "images").exists() else base_dir
    image_files = sorted(
        [f for f in img_dir.iterdir() if f.suffix.lower() in [".png", ".jpg", ".jpeg"]],
        key=lambda x: int(x.stem.split("_")[1]) if "_" in x.stem and x.stem.split("_")[1].isdigit() else float("inf"),
    )
    return [str(p) for p in image_files]


if "llama-4" in args.model_name.lower():
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = Llama4ForConditionalGeneration.from_pretrained(
        args.model_name,
        attn_implementation="sdpa",
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
elif "qwen" in args.model_name.lower():
    processor = AutoProcessor.from_pretrained(args.model_name)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
else:
    raise ValueError(f"Unsupported model_name: {args.model_name}")

# Derive prompt name from annotations filename (e.g. "Doctor_annotations.json" -> "Doctor")
annotations_path = Path(args.annotations_path)
base_prompt = annotations_path.stem.replace("_annotations", "")

model_short = args.model_name.split("/")[-1]
metrics_task_dir = os.path.join(args.metrics_dir, args.task)
os.makedirs(metrics_task_dir, exist_ok=True)

presence_out = os.path.join(metrics_task_dir, f"{base_prompt}_{model_short}_attribute_presence.json")
summary_path = os.path.join(metrics_task_dir, f"{model_short}_attribute_eval_summary.json")

images = load_image_dataset(args.image_dir)
if len(images) > args.num_samples_per_attr:
    images = random.sample(images, args.num_samples_per_attr)

with open(annotations_path, "r") as f:
    data = json.load(f)

attributes = []
for i, item in enumerate(data):
    if args.top_n_minority_neurons is not None and i >= args.top_n_minority_neurons:
        break
    attr = item.get("identified_attribute", "")
    if attr and attr != "No identified attribute":
        attributes.append(attr)

print(f"Prompt: {base_prompt}")
print(f"Attributes ({len(attributes)}): {attributes}")
if not attributes:
    sys.exit("No attributes to check (after filtering).")

presence_results = []
for img_path_str in tqdm(images, desc="Checking attributes in images"):
    img_path = Path(img_path_str)
    if not img_path.exists():
        print(f"Skip missing image: {img_path}", file=sys.stderr)
        continue
    image_result = {"filename": img_path.name, "attributes": {}}
    for attr in attributes:
        instruction = (
            f"Look at the provided image and answer with only 'yes' or 'no'.\n"
            f"Question: Is the attribute '{attr}' clearly visible/present in the image?"
        )
        if "llama-4" in args.model_name.lower():
            answer = query_llama4(model, processor, instruction, img_path)
        else:
            answer = query_qwen(model, processor, instruction, img_path)
        image_result["attributes"][attr] = answer
    presence_results.append(image_result)

with open(presence_out, "w") as f:
    json.dump(presence_results, f, indent=2)
print(f"Saved attribute presence results to {presence_out}")

attribute_counts = {attr: 0 for attr in attributes}
total_images = len(presence_results)
for result in presence_results:
    for attr, answer in result["attributes"].items():
        if answer == "yes":
            attribute_counts[attr] += 1

avg_minority = mean(attribute_counts[a] / total_images for a in attributes) if attributes else 0.0

if os.path.exists(summary_path):
    with open(summary_path, "r") as f:
        try:
            summary_all = json.load(f)
        except json.JSONDecodeError:
            summary_all = {}
else:
    summary_all = {}

summary_all[base_prompt] = {
    "attributes": {attr: {"count": c} for attr, c in attribute_counts.items()},
    "average": round(avg_minority, 4),
    "total_images": total_images,
}

with open(summary_path, "w") as f:
    json.dump(summary_all, f, indent=2)
print(f"Updated summary in {summary_path}")
