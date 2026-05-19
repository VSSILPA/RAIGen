from PIL import Image
import sys
import re

def query_qwen(model, processor, instruction, img_path):
    try:
        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(
            **inputs,
            max_new_tokens=10,
            do_sample=False,
        )

        response = processor.decode(
            output[0][inputs["input_ids"].shape[-1]:],
            skip_special_tokens=True,
        ).strip().lower()

        m = re.search(r"\b(yes|no)\b", response)
        return m.group(1) if m else "unknown"

    except Exception as e:
        print(f"Qwen error on {img_path}: {e}", file=sys.stderr)
        return "error"


def query_llama4(model, processor, instruction, img_path):
    try:
        image = Image.open(img_path).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": instruction},
                ],
            }
        ]

        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(model.device)

        output = model.generate(**inputs, max_new_tokens=10, do_sample=False)
        response = processor.batch_decode(output[:, inputs["input_ids"].shape[-1]:])[0].strip()

        # remove common wrappers / special tokens
        response = re.sub(r"^```.*?\n|```.*?$|<\|eot\|>", "", response, flags=re.DOTALL).strip().lower()

        # robust yes/no extraction (avoids matching "yesterday")
        m = re.search(r"\b(yes|no)\b", response)
        return m.group(1) if m else "unknown"

    except Exception as e:
        print(f"Llama4 error on {img_path}: {e}", file=sys.stderr)
        return "error"