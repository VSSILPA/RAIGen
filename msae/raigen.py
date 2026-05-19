from get_clip_embeddings import get_clip_embeddings
from sklearn.metrics.pairwise import cosine_distances
from datasets import Dataset
from gather_activations import gather_act, TimestepDataset
import os
import logging
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import json
from pathlib import Path
from PIL import Image



def load_image_dataset(image_dir):
    img_dir = Path(image_dir) / "images"
    image_files = sorted(
    [f for f in img_dir.iterdir() if f.suffix.lower() in ['.png', '.jpg', '.jpeg']],
    key=lambda x: int(x.stem.split('_')[1])
    )
    image_paths = [str(path) for path in image_files]
    return image_paths


def load_sae_activations(args, sae, sd_activation_dataset, steps, timestep):
    sd_activation_dataset.set_format("torch", columns=["activations"])
    timestep_activations_dataset = TimestepDataset(sd_activation_dataset, steps=steps, timestep=timestep)
    sae_activations = gather_act(args, sae, timestep_activations_dataset)
    return sae_activations


def prune_redundant_neurons(clip_centroids, minority_scores, distance_threshold=0.003, percentile_cutoff=90):
    N = clip_centroids.shape[0]
    all_indices = list(range(N))
    remaining = set(all_indices)
    retained = []

    dist_matrix = cosine_distances(clip_centroids)
    sorted_indices = sorted(all_indices, key=lambda i: minority_scores[i], reverse=True)
    score_cutoff = np.percentile(minority_scores, percentile_cutoff)

    while sorted_indices:
        i = sorted_indices.pop(0)
        if i not in remaining:
            continue
        group = [j for j in remaining if dist_matrix[i, j] < distance_threshold]
        best = max(group, key=lambda j: minority_scores[j]) 

        if minority_scores[best] > score_cutoff:
            retained.append(best)
            remaining -= set(group)  

    return np.array(retained)


def compute_weighted_centroid_neuron(embeddings, activations, eps=1e-8):
    activations = np.maximum(activations, 0) 
    weights = activations / (activations.sum(axis=0, keepdims=True) + eps)
    # weighted_centroid = np.sum(weights[:, None] * embeddings, axis=0)
    weighted_centroid = weights.T @ embeddings
    return weighted_centroid

def compute_dataset_centroid(embeddings):
    return np.mean(embeddings, axis=0)


def get_top_activating_images(activations, top_k=None):
    neuron_activations= torch.from_numpy(np.array(activations, copy=True))
    sorted_vals, sorted_idx = torch.sort(neuron_activations, descending=True)
    if top_k is not None:
        return sorted_idx[:top_k], sorted_vals[:top_k]
    else:
        return sorted_idx, sorted_vals


def get_activating_images_for_minority(sae_activations, neurons):
    
    minority_sample_data = {}
    for neuron in neurons:
        activations = sae_activations[:, neuron]
        top_img_idxs, top_img_act =  get_top_activating_images(activations, top_k=100)
        minority_sample_data[neuron.item()] = {
            "indices": top_img_idxs.tolist(),  
            "activations": top_img_act.tolist()
        }

    return minority_sample_data


def get_sae_heatmap(sae, sd_act_dataset, top_image_idxs, neuron, timestep, steps, image_size=256):
    """Spatial heatmap of one neuron's activation for a batch of images."""
    all_indices = [(idx.item() * steps) + timestep for idx in top_image_idxs]
    selected = sd_act_dataset.select(all_indices)
    selected.set_format(type="torch", columns=["activations"])
    activations = selected["activations"].to(sae.W_dec.device, dtype=sae.W_enc.dtype)
    spatial_dim = int(activations.shape[1] ** 0.5)

    with torch.no_grad():
        sae.eval()
        feats = sae.encode(activations)
        heatmap = feats[..., neuron].reshape(len(all_indices), spatial_dim, spatial_dim)
        heatmap = F.interpolate(
            heatmap.unsqueeze(1).float(),
            size=(image_size, image_size),
            mode='bilinear', align_corners=False,
        ).squeeze(1)
    return heatmap.cpu()


def overlay_heatmap_on_image(image, heatmap):
    heatmap = heatmap.cpu().numpy().astype(np.float32)
    rng = heatmap.max() - heatmap.min()
    heatmap = (heatmap - heatmap.min()) / (rng if rng > 0 else 1.0)
    heatmap = (heatmap * 255).astype(np.uint8)
    heatmap_colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

    img_arr = np.array(image.convert('RGB'))
    heatmap_resized = cv2.resize(
        heatmap_colored, (img_arr.shape[1], img_arr.shape[0]), interpolation=cv2.INTER_LINEAR
    )
    overlayed = cv2.addWeighted(img_arr, 0.5, heatmap_resized, 0.5, 0)
    return Image.fromarray(cv2.cvtColor(overlayed, cv2.COLOR_BGR2RGB))


def save_minority_neurons(
    latents,
    sorted_neurons_local,
    sorted_neurons_global,
    image_dataset,
    sae,
    sd_act_dataset,
    steps,
    timestep,
    save_dir,
    top_k=10,
    image_size=(256, 256),
):
    minority_png_dir = os.path.join(save_dir, "minority_neurons")
    os.makedirs(minority_png_dir, exist_ok=True)

    cell_w, cell_h = image_size
    for rank, (neuron_local, neuron_global) in enumerate(zip(sorted_neurons_local, sorted_neurons_global)):
        activations = latents[:, neuron_local]
        top_img_idxs, _ = get_top_activating_images(activations, top_k=top_k)

        # row 1: original images
        top_images = [
            Image.open(image_dataset[idx.item()]).convert("RGB").resize(image_size)
            for idx in top_img_idxs
        ]

        # row 2: heatmap overlays for the same images
        heatmaps = get_sae_heatmap(
            sae, sd_act_dataset, top_img_idxs, int(neuron_global), timestep, steps,
            image_size=cell_w,
        )
        overlays = [overlay_heatmap_on_image(img, hm) for img, hm in zip(top_images, heatmaps)]

        grid = Image.new("RGB", (top_k * cell_w, 2 * cell_h), color=(255, 255, 255))
        for col, img in enumerate(top_images):
            grid.paste(img, (col * cell_w, 0))
        for col, img in enumerate(overlays):
            grid.paste(img, (col * cell_w, cell_h))

        filename = f"{rank:04}_neuron{int(neuron_global):04}.png"
        grid.save(os.path.join(minority_png_dir, filename), "PNG")


def analyze_single_group(args, sae_activations, image_dataset, sae, sd_act_dataset, steps, timestep, dir, sample_data, group_start=None, group_end=None):
    if group_start is None:
        sae_latents = sae_activations
        group_label = "full"
    else:
        sae_latents = sae_activations[:, group_start:group_end]
        group_label = f"group_{group_start}_{group_end}"

    latents = np.array(sae_latents, copy=True).astype(np.float32)
    save_dir = os.path.join(dir, f'neurons_analysis_{group_label}')
    os.makedirs(save_dir, exist_ok=True)

    clip_embed_path = f"{args.image_dir}/clip_embed.pt"
    if os.path.exists(clip_embed_path):
        print(f"Loading existing CLIP embeddings from {clip_embed_path}")
        embeddings = torch.load(f"{args.image_dir}/clip_embed.pt", map_location="cpu")
    else:
        print(f"CLIP activations not found at {clip_embed_path}, gathering now...")
        embeddings = get_clip_embeddings(image_dataset, batch_size=32)
        torch.save(embeddings,  clip_embed_path)
        print(f"Saved gathered activations to {clip_embed_path}")

    if sample_data is not None:
        embeddings = embeddings[sample_data]
        print(f"Selected {len(sample_data)} CLIP embeddings from full set")
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    dataset_centroid = compute_dataset_centroid(embeddings_norm)
    weighted_centroid = compute_weighted_centroid_neuron(embeddings_norm, latents)
    weighted_centroid = weighted_centroid / (np.linalg.norm(weighted_centroid, axis=1, keepdims=True) + 1e-8)

    
    dataset_centroid = dataset_centroid / np.linalg.norm(dataset_centroid)
    distances = cosine_distances(weighted_centroid, dataset_centroid.reshape(1, -1)).squeeze()

   
    frequency = (latents > 0.0).mean(axis=0)
    ## Remove neurons where images do not activate at all
    active_mask = frequency > 0
    active_idx = np.nonzero(active_mask)[0]
    frequency = frequency[active_mask]
    distances = distances[active_mask]
    weighted_centroid = weighted_centroid[active_mask]

    distances = (distances - distances.min()) / (distances.max() - distances.min())
    frequency = (frequency - frequency.min()) / (frequency.max() - frequency.min())
    minority_score = (1-frequency) * distances 

     
    pd.DataFrame({"neuron_idx": active_idx, "cosine_distance" :  distances.tolist(), "frequency" : frequency.tolist(), "minority_score": minority_score.tolist()}).to_csv(f"{save_dir}/all_scores.csv", index=False)

    final_minority_neurons_active = prune_redundant_neurons(weighted_centroid, minority_score)
    final_minority_neurons = active_idx[final_minority_neurons_active]   # original group indices

    final_minority_scores = minority_score[final_minority_neurons_active]
    final_minority_distances = distances[final_minority_neurons_active]
    final_minority_frequency = frequency[final_minority_neurons_active]


    minority_sample_data = get_activating_images_for_minority(
    sae_activations, (final_minority_neurons + group_start)
)

    with open(f"{save_dir}/minority_neuron_image_activations.json", "w") as f:
        json.dump(minority_sample_data, f, indent=4)



    # ## Visualise neurons
    final_minority_neurons = np.array(final_minority_neurons)
    sorted_indices = np.argsort(-final_minority_scores)
    sorted_final_neurons_local = final_minority_neurons[sorted_indices].tolist()
    sorted_final_neurons_global = (final_minority_neurons[sorted_indices] + group_start).tolist()

    df = pd.DataFrame({
    "neuron_index": sorted_final_neurons_global,
    "cosine_distance": final_minority_distances[sorted_indices].tolist(),
    "frequency": final_minority_frequency[sorted_indices].tolist(),
    "minority_score": final_minority_scores[sorted_indices].tolist()
    })
   

    df.to_csv(f"{save_dir}/final_minority_neurons.csv", index=False)
    save_minority_neurons(
        latents=latents,
        sorted_neurons_local=sorted_final_neurons_local,
        sorted_neurons_global=sorted_final_neurons_global,
        image_dataset=image_dataset,
        sae=sae,
        sd_act_dataset=sd_act_dataset,
        steps=steps,
        timestep=timestep,
        save_dir=save_dir,
        top_k=10,
    )
    

def identify_minority(args, checkpoint_path, sae, sample_data=None):
    class_name=os.path.basename(os.path.normpath(args.image_dir))
    if class_name.endswith("_valid"):
        checkpoint_path =  f"{checkpoint_path}/valid"
        os.makedirs(checkpoint_path, exist_ok=True)
    logging.basicConfig(
            filename=f"{checkpoint_path}/logfile.txt",
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            filemode='w' 
        )

    
    print(args.image_dir)
    image_dataset = load_image_dataset(args.image_dir)
    sd_activation_dataset  = Dataset.load_from_disk(
        os.path.join(args.dataset_path, args.hook_point), keep_in_memory=False)

    steps = int((sd_activation_dataset.num_rows) / len(image_dataset))
    timestep = steps - 1
  
    print(steps)
    print(timestep)
    sae_activations = load_sae_activations(args, sae, sd_activation_dataset, steps=steps, timestep=timestep)
    output_dir = f"{checkpoint_path}/raigen_neurons"
    os.makedirs(output_dir, exist_ok=True)
    logging.info(f"Timestep : {timestep}")
    group_sizes = [0] + args.group_sizes 
    for i in range(len(group_sizes) - 2):
        group_start = group_sizes[i]
        group_end = group_sizes[i+1]
        logging.info(f"Group is >> {group_start}:{group_end}")
        analyze_single_group(args, sae_activations, image_dataset, sae, sd_activation_dataset, steps, timestep, output_dir, sample_data, group_start=group_start, group_end=group_end)
  
    