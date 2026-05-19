import torch 
import os

def get_default_cfg():
    default_cfg = {
        "seed": 0,
        "effective_batch_size": 4096,
        "lr": 3e-4,
        "l1_coeff": 0,
        "beta1": 0.9,
        "beta2": 0.99,
        "max_grad_norm": 100000,
        "dtype": torch.bfloat16,
        "train_sae": False,
        "model_name": "CompVis/stable-diffusion-v1-4",
        "device" : 'cuda:0',
        "hook_point" : "unet.mid_block",
        "task":"prof",
        "d_in": 1280,
        "expansion_factor" : 1,
        "dataset_path": "activations",
        "image_dir" : "dataset",
        "prompt":"Doctor",
        'wandb_project': 'raigen',
        "input_unit_norm": False,
        "checkpoint_root": "checkpoints",
        "checkpoint_freq": 20000,
        "wandb_log_freq" : 1000,
        "n_batches_to_dead": 20,

        # (Batch)TopKSAE specific
        "top_k": 32,
        "top_k_aux": 512,
        "aux_penalty": (1/32),
        
        # for jumprelu
        "bandwidth": 0.001,

        #Matryoshki
        # "top_k_matryoshka": [10, 10, 10, 10, 10],
        "group_sizes": [1280//4, 1280 // 4 ,1280 // 2, 1280, 1280*2, 1280*4, 1280*8],
        'num_epochs': 5

    }
    # default_cfg = post_init_cfg(default_cfg)
    return default_cfg

def post_init_cfg(cfg):
    # class_name=os.path.basename(os.path.normpath(cfg.image_dir))
    # if class_name.endswith("_valid"):
    #     model_class_name = class_name.rsplit("_", 1)[0]
    # else:
    #     model_class_name = class_name
    cfg.dataset_path = f"{cfg.dataset_path}/{cfg.task}/{cfg.prompt}/{cfg.model_name.split('/')[-1]}"
    cfg.checkpoint_root = os.path.join(cfg.checkpoint_root, cfg.task, cfg.prompt)
    cfg.image_dir = f"{cfg.image_dir}/{cfg.task}/{cfg.prompt}/{cfg.model_name.split('/')[-1]}"
    cfg.dict_size = cfg.d_in * cfg.expansion_factor
    final_size = cfg.dict_size - sum(cfg.group_sizes)
    cfg.group_sizes = cfg.group_sizes + [final_size]
    if cfg.model_name == "CompVis/stable-diffusion-v1-4":
        cfg.model_name = "sd_v14"
    if cfg.model_name == "stabilityai/stable-diffusion-xl-base-1.0":
        cfg.model_name = "sdxl"
    elif  cfg.model_name == "stabilityai/stable-diffusion-2":
        cfg.model_name = "sd_v21"
    elif  cfg.model_name == "stabilityai/stable-diffusion-3-medium-diffusers":
        cfg.model_name = "sd_v3"
    elif  cfg.model_name == "black-forest-labs/FLUX.1-schnell":
        cfg.model_name = "flux"
    cfg.name = f"{cfg.model_name}_{cfg.hook_point}_{cfg.dict_size}_{len(cfg.group_sizes)}_{cfg.group_sizes}"
    os.makedirs(cfg.checkpoint_root, exist_ok=True)
    return cfg