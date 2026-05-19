import numpy as np
from PIL import Image
import torch
from transformers import CLIPModel, CLIPProcessor
from tqdm import tqdm

def load_clip_model(model_name="openai/clip-vit-large-patch14"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained(model_name).to(device)
    model.eval()
    processor = CLIPProcessor.from_pretrained(model_name)  # default resize 224x224
    return model, processor, device


def load_images(image_paths):
    images = []
    for path in image_paths:
        try:
            image = Image.open(path).convert("RGB")
            images.append(image)
        except:
            print(f"Failed to load image: {path}")
            images.append(Image.new("RGB", (224, 224)))  
    return images

def get_clip_embeddings(image_paths, batch_size=32):
    model, processor, device = load_clip_model()
    all_embeddings = []

    for i in tqdm(range(0, len(image_paths), batch_size)):
        batch_paths = image_paths[i:i+batch_size]
        images = load_images(batch_paths)

        inputs = processor(images=images, return_tensors="pt", padding=True).to(device)

        with torch.no_grad():
            features = model.get_image_features(**inputs)
            # features = features / features.norm(dim=-1, keepdim=True)

        all_embeddings.append(features.cpu().numpy())

    return np.vstack(all_embeddings)