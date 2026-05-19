import torch
from torch.utils.data import DataLoader
import tqdm


class TimestepDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, steps=50, timestep=49):
        assert 0 <= timestep < steps, "Timestep must be in range [0, steps-1]"
        self.base = base_dataset
        self.indices = list(range(timestep, len(base_dataset), steps))  # e.g., 49, 99, ...

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        row = self.base[self.indices[idx]]
        return row["activations"]  # [64, 1280]




def gather_act(args, sae, dataset):

    
    dataloader = DataLoader( dataset,batch_size=8,shuffle=False,num_workers=4,pin_memory=True, drop_last=False)
    sae_latents = []
    for batch in tqdm.tqdm(dataloader, desc="Final timestep batches"):
        batch = batch.to(args.device, dtype=torch.float16)  # [B, 64, 1280]
        with torch.no_grad():
            _, top_acts = sae.compute_activations(batch)  # output: [B, dict_size]
            sae_latents.append(top_acts.mean(1).cpu())

    return torch.cat(sae_latents, dim=0)  # shape: [num_sequences, dict_size]







