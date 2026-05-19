
from typing import Dict, List, Optional, Union, Any

import torch

__all__ = [
    "ActivationTracer",
    "run_with_tracer",
]


class ActivationTracer:
    """Light‑weight activation / latent recorder.

    Parameters
    ----------
    pipe : diffusers.DiffusionPipeline (or subclass)
        The pipeline whose modules you want to observe.
    positions : list[str]
        Dotted paths pointing to sub‑modules to hook.
    unconditional : bool, default ``False``
        If *True*, keep the *unconditional* half of the batch (first split) when
        classifier‑free guidance is in use; otherwise keep the *conditional*
        half.
    save_input : bool, default ``False``
        Cache forward *inputs* at each hooked module.
    save_output : bool, default ``True``
        Cache forward *outputs* at each hooked module.
    """

    def __init__(
        self,
        pipe,
        *,
        positions: List[str],
        unconditional: bool = False,
        save_input: bool = False,
        save_output: bool = True,
    ) -> None:
        self.pipe = pipe
        self.positions = positions
        self.unconditional = unconditional
        self.save_input = save_input
        self.save_output = save_output
       

        # nested dict:  {position: {"input": [T, ...], "output": [T, ...]}}
        self.cache: Dict[str, Dict[str, List[torch.Tensor]]] = {}
        self._hooks: List[torch.utils.hooks.RemovableHandle] = []
     
    def __enter__(self):
        for pos in self.positions:
            module = self._locate(pos)
            self.cache.setdefault(pos, {})
            hook = module.register_forward_hook(self._make_hook(pos), with_kwargs=True)
            self._hooks.append(hook)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        for h in self._hooks:
            h.remove()
        return False


    def _locate(self, dotted: str):
        obj = self.pipe
        for token in dotted.split("."):
            obj = obj[int(token)] if token.isdigit() else getattr(obj, token)
        return obj

    def _make_hook(self, pos: str):
        def _hook(_module, _inp, _kw, out):
            if self.save_input:
                tensor_in = self._retrieve(_inp)
                self.cache[pos].setdefault("input", []).append(self._flatten(tensor_in))
            if self.save_output:
                tensor_out = self._retrieve(out)
                self.cache[pos].setdefault("output", []).append(self._flatten(tensor_out))

        return _hook

    def _retrieve(self, io: Union[torch.Tensor, tuple]):
        if isinstance(io, tuple):
            if len(io) >= 2 and isinstance(io[1], torch.Tensor):
                io = io[1]  # SD3: pick hidden_states (second element)
            elif len(io) == 1 and isinstance(io[0], torch.Tensor):
                io = io[0]
            else:
                raise ValueError("Unexpected tuple structure from model output")


        # Split conditional / unconditional if classifier‑free guidance doubled the batch
        if io.shape[0] % 2 == 0:
            uncond, cond = io.chunk(2)
            io = uncond if self.unconditional else cond
        return io.detach().cpu()

    def _flatten(self, tensor: torch.Tensor) -> torch.Tensor:
        """Match original run_with_cache reshape for 4‑D feature maps."""
        if tensor.ndim == 4:  # (B, C, H, W) -> (B, HW, C)
            b, c, h, w = tensor.shape
            tensor = tensor.view(b, c, -1).permute(0, 2, 1)
        return tensor

def run_with_tracer(
    pipe,
    *,
    prompt: Union[str, List[str]],
    positions_to_cache: List[str],
    num_inference_steps: int = 50,
    guidance_scale: float = 7.5,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.Tensor] = None,
    output_type: str = "pil",
    save_input: bool = False,
    save_output: bool = True,
    unconditional: bool = False,
    callback_steps: int = 1,
    **pipe_kwargs: Any,
):
    with ActivationTracer(
        pipe,
        positions=positions_to_cache,
        unconditional=unconditional,
        save_input=save_input,
        save_output=save_output,
    ) as tracer:
        output = pipe(
            prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator,
            latents=latents,
            output_type=output_type,
            **pipe_kwargs,
        )


    cache_dict: Dict[str, Any] = {}

    if save_input:
        cache_dict["input"] = {
            pos: torch.stack(block["input"], dim=1)
            for pos, block in tracer.cache.items()
            if "input" in block
        }
    if save_output:
        cache_dict["output"] = {
            pos: torch.stack(block["output"], dim=1)
            for pos, block in tracer.cache.items()
            if "output" in block
        }


    keys = list(cache_dict["output"].keys())
    if len(keys) > 1:
        avg = torch.stack([cache_dict["output"][k] for k in keys], dim=0).mean(dim=0)
        cache_dict["output"] = {keys[0]: avg}


    images = output.images if hasattr(output, "images") else output
    return images, cache_dict

