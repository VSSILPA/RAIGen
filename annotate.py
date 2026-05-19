import os
import re
import json
import argparse
import base64
import time
from openai import OpenAI
from utils import annotate_neuron, annotate_neuron_coco

# ---------- CLI ----------
parser = argparse.ArgumentParser(
    description="Prompt rebalancing to amplify minority generation"
)

parser.add_argument("--base_prompt", required=True,
                    help='Profession, e.g. "doctor", "driver" …')
parser.add_argument("--model", default="gpt-5.2",
                    help='OpenAI model name (default: gpt-5.2)')
parser.add_argument("--checkpoint_root_dir", required=True,
                    help='Root directory that holds all result files to be annotated')
parser.add_argument("--task", required=False,default="prof",
                    help='Task to identify, prof or coco')
parser.add_argument("--top_n_minority_neurons", type=int, required=False,default=None,
                    help='Number of minority neurons to annotate')
parser.add_argument("--minority_dir", required=True,
                    help='Folder where the minority results are saved')
args = parser.parse_args()



client = OpenAI()  

ANNOTATION_SCHEMA = {
  "type": "object",
  "additionalProperties": False,
  "properties": {
    "neuron_id": {"type": "string"},
    "input prompt": {"type": "string"},
    "identified_attribute": {"type": "string"},
    "suggested_prompt": {"type": "string"},
    "keywords": {"type": "array", "items": {"type": "string"}},
  },
  "required": ["neuron_id", "input prompt", "identified_attribute", "suggested_prompt", "keywords"]
}

def encode_image_to_data_url(image_path: str) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:image/png;base64,{b64}"

def call_with_retry(instruction: str, image_path: str, model="gpt-5.2",
                    max_output_tokens=700, max_retries=8):
    backoff = 0.5
    data_url = encode_image_to_data_url(image_path)

    for _ in range(max_retries):
        try:
            return client.responses.create(
                model=model,
                input=[{
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": instruction},
                        {"type": "input_image", "image_url": data_url},
                    ],
                }],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "neuron_annotation",
                        "strict": True,
                        "schema": ANNOTATION_SCHEMA,
                    }
                },
                max_output_tokens=max_output_tokens,
                reasoning={"effort": "none"},
            )
        except Exception:
            time.sleep(min(60.0, backoff))
            backoff *= 2

    raise RuntimeError(f"OpenAI call failed after {max_retries} retries.")



minority_image_dir = f"{args.checkpoint_root_dir}/{args.task}/{args.base_prompt}/{args.minority_dir}/minority_neurons"
annotation_dir = f"{args.checkpoint_root_dir}/{args.task}/{args.base_prompt}/{args.minority_dir}/minority_prompt_annotations.json"

if args.task == 'prof':
    args.base_prompt = f'A photo of a {args.base_prompt}'
    

if not os.path.exists(annotation_dir):
    print(annotation_dir)
    results = []
    if args.top_n_minority_neurons:
        minority_neurons = sorted(os.listdir(minority_image_dir))[:args.top_n_minority_neurons]
    else:
        minority_neurons = sorted(os.listdir(minority_image_dir))
    for fname in minority_neurons :
        if not fname.endswith(".png"):
            continue

        match = re.search(r'neuron(\d+)', fname)
        neuron_id = match.group(1) if match else "UNKNOWN"

        try:
            image_path = os.path.join(minority_image_dir, fname)
            if args.task == 'prof':
                instruction = annotate_neuron(args.base_prompt, str(neuron_id))
            else:
                instruction = annotate_neuron_coco(args.base_prompt, str(neuron_id))
            resp = call_with_retry(instruction, image_path, model=args.model, max_output_tokens=700)
            result = json.loads(resp.output_text)

            result["neuron_id"] = str(neuron_id).zfill(4)   
            result["input prompt"] = args.base_prompt      
            result["filename"] = fname
            results.append(result)

            print(json.dumps(result, indent=2, ensure_ascii=False))

        except Exception as e:
            print(f"Error with {fname}: {e}")

    with open(annotation_dir, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Saved results to {annotation_dir}")

else:
    print("Annotations already exist. Loading...")
    with open(annotation_dir, "r") as f:
        results = json.load(f)