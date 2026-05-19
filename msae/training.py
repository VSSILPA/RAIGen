import torch
import tqdm
import wandb
import json
import os
from torch.utils.data import DataLoader


def save_checkpoint_mp(sae, cfg, step, final=None):
    save_dir = f"{cfg.checkpoint_root}/{cfg.name}/{step}"
    os.makedirs(save_dir, exist_ok=True)

    if final:
        save_dir = f"{cfg.checkpoint_root}/{cfg.name}/{final}"
        os.makedirs(save_dir, exist_ok=True)

    torch.save(sae.state_dict(), os.path.join(save_dir, "sae.pt"))

    json_safe_cfg = {}
    for key, value in vars(cfg).items():
        if isinstance(value, (int, float, str, bool, type(None))):
            json_safe_cfg[key] = value
        elif isinstance(value, (torch.dtype, type)):
            json_safe_cfg[key] = str(value)
        else:
            json_safe_cfg[key] = str(value)

    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(json_safe_cfg, f, indent=4)

    print(f"Model and config saved at step {step} in {save_dir}")

def train_sae(sae, dataset_dict, cfg):
    wandb.init(
        entity=getattr(cfg, "wandb_entity", ""),
        project=cfg.wandb_project,
        config=cfg,
        name=f"{cfg.name}_{cfg.prompt}",
    )

    num_examples = len(dataset_dict[list(dataset_dict.keys())[0]])
    sample_size = dataset_dict[list(dataset_dict.keys())[0]][0]["activations"].shape[-2]
    effective_batch_size = cfg.effective_batch_size
    batch_size = effective_batch_size // sample_size

    dataloaders = {
    hook: DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )
    for hook, ds in dataset_dict.items()
}


    num_batches = (num_examples // batch_size) * cfg.num_epochs
    print(f"Number of batches: {num_batches}")

    optimizer = torch.optim.Adam(
        sae.parameters(),
        lr=cfg.lr,
        betas=(cfg.beta1, cfg.beta2)
    )

    pbar = tqdm.trange(num_batches, desc="Training Progress", total=num_batches)

    global_step = 0

    for _ in range(cfg.num_epochs):
        for batch_dict in zip(*dataloaders.values()):
            batch = batch_dict[0]["activations"].to(cfg.device, non_blocking=True)
            sae_output = sae(batch)
            loss = sae_output["loss"]
            log_dict = {}
            for key, value in sae_output.items():
                if isinstance(value, torch.Tensor):
                    log_dict[key] = value.item() if value.dim() == 0 else value.mean().item()
                elif isinstance(value, (int, float)):
                    log_dict[key] = value
            if "explained_variance" in sae_output:
                ev = sae_output["explained_variance"]
                if isinstance(ev, torch.Tensor) and ev.dim() > 0:
                    log_dict["explained_variance"] = ev.mean().item()
                    
            if global_step % cfg.wandb_log_freq == 0:
                wandb.log(log_dict, step=global_step)

            if global_step % cfg.checkpoint_freq == 0:
                save_checkpoint_mp(sae, cfg, global_step)
                
            pbar.set_postfix({
                "Loss": f"{loss.item():.4f}",
                "L0": f"{sae_output['l0_norm']:.4f}",
                "L2": f"{sae_output['l2_loss']:.4f}",
                "L1": f"{sae_output['l1_loss']:.4f}",
            })

            loss.backward()
            torch.nn.utils.clip_grad_norm_(sae.parameters(), cfg.max_grad_norm)

            sae.make_decoder_weights_and_grad_unit_norm()

            optimizer.step()
            optimizer.zero_grad()

            pbar.update(1)
            global_step += 1

    save_checkpoint_mp(sae, cfg, global_step, final="final")

    wandb.finish()
    return global_step
