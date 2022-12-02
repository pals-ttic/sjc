import sys
from pathlib import Path

# BUG resolve this
sys.path.append(str(
    Path(__file__).resolve().parent / "sd1"
))

from adapt import ScoreAdapter
from omegaconf import OmegaConf
from math import sqrt
from sd1.ldm.util import instantiate_from_config
import torch
import numpy as np
from torch import autocast
from contextlib import contextmanager, nullcontext
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from transformers import logging
logging.set_verbosity_error()


device = torch.device("cuda")


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.to(device)
    model.eval()
    return model


def _sqrt(x):
    if isinstance(x, float):
        return sqrt(x)
    else:
        assert isinstance(x, torch.Tensor)
        return torch.sqrt(x)


class StableDiffusion(ScoreAdapter):
    def __init__(self, prompt, HW, C, f, scale, precision, embedding_path, ckpt_fname='sd-v1-3'):
        H, W = HW, HW
        cfg_fname = Path(__file__).resolve().parent \
            / "sd1" / "configs" / "v1-inference.yaml"
        config = OmegaConf.load(str(cfg_fname))

        ckpt = self.checkpoint_root() / "stable_diffusion" / f"{ckpt_fname}.ckpt"
        self.model = load_model_from_config(config, ckpt)
        if Path(embedding_path).is_file():
            self.model.embedding_manager.load(embedding_path)
        self._device = self.model._device

        self.prompt = prompt
        self.scale = scale
        self.precision = precision
        self.precision_scope = autocast if self.precision == "autocast" else nullcontext
        self._data_shape = (C, H//f, W//f)

        self.cond_func = self.model.get_learned_conditioning

        self._unet_is_cond = True

        self.M = 1000
        noise_schedule = "linear"
        self.noise_schedule = noise_schedule
        self.us = self.linear_us(self.M)

    def data_shape(self):
        return self._data_shape

    @property
    def σ_max(self):
        return self.us[0]

    @property
    def σ_min(self):
        return self.us[-1]

    @torch.no_grad()
    def denoise(self, xs, σ, **model_kwargs):
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                N = xs.shape[0]
                c = model_kwargs.pop('c')
                uc = model_kwargs.pop('uc')
                cond_t, σ = self.time_cond_vec(N, σ)
                unscaled_xs = xs
                xs = xs / _sqrt(1 + σ**2)
                if uc is None or self.scale == 1.:
                    output = self.model.apply_model(xs, cond_t, c)
                else:
                    x_in = torch.cat([xs] * 2)
                    t_in = torch.cat([cond_t] * 2)
                    c_in = torch.cat([uc, c])
                    e_t_uncond, e_t = self.model.apply_model(x_in, t_in, c_in).chunk(2)
                    output = e_t_uncond + self.scale * (e_t - e_t_uncond)
                    # print('scale',self.scale)

                Ds = unscaled_xs - σ * output
                return Ds

    def cond_info(self, batch_size):
        prompts = batch_size * [self.prompt]
        return self.prompts_emb(prompts)

    @torch.no_grad()
    def prompts_emb(self, prompts):
        assert isinstance(prompts, list)
        batch_size = len(prompts)
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                cond = {}
                c = self.cond_func(prompts)
                cond['c'] = c
                uc = None
                if self.scale != 1.0:
                    uc = self.cond_func(batch_size * [""])
                cond['uc'] = uc
                return cond

    def unet_is_cond(self):
        return self._unet_is_cond

    def use_cls_guidance(self):
        return False

    def snap_t_to_nearest_tick(self, t):
        j = np.abs(t - self.us).argmin()
        return self.us[j], j

    def time_cond_vec(self, N, σ):
        if isinstance(σ, float):
            σ, j = self.snap_t_to_nearest_tick(σ)  # σ might change due to snapping
            cond_t = (self.M - 1) - j
            cond_t = torch.tensor([cond_t] * N, device=self.device)
            return cond_t, σ
        else:
            assert isinstance(σ, torch.Tensor)
            σ = σ.reshape(-1).cpu().numpy()
            σs = []
            js = []
            for elem in σ:
                _σ, _j = self.snap_t_to_nearest_tick(elem)
                σs.append(_σ)
                js.append((self.M - 1) - _j)

            cond_t = torch.tensor(js, device=self.device)
            σs = torch.tensor(σs, device=self.device, dtype=torch.float32).reshape(-1, 1, 1, 1)
            return cond_t, σs

    @staticmethod
    def linear_us(M=1000):
        assert M == 1000
        β_start = 0.00085
        β_end = 0.0120
        βs = np.linspace(β_start**0.5, β_end**0.5, M, dtype=np.float64)**2
        αs = np.cumprod(1 - βs)
        us = np.sqrt((1 - αs) / αs)
        us = us[::-1]
        return us

    @torch.no_grad()
    def encode(self, xs):
        model = self.model
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                zs = model.get_first_stage_encoding(
                    model.encode_first_stage(xs)
                )
        return zs

    @torch.no_grad()
    def decode(self, xs):
        with self.precision_scope("cuda"):
            with self.model.ema_scope():
                xs = self.model.decode_first_stage(xs)
                return xs


def test():
    sd = StableDiffusion(
        "hah", 512, 4, 8, 10.0,
        precision="autocast", embedding_path="", ckpt_fname="sd-v1-5"
    )
    print(sd)


if __name__ == "__main__":
    test()
