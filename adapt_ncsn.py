from pathlib import Path
import argparse
import yaml

import numpy as np
import torch

from ncsn.ncsnv2 import NCSNv2, NCSNv2Deeper, NCSNv2Deepest, get_sigmas
from ncsn.ema import EMAHelper

from adapt import ScoreAdapter

device = torch.device("cuda")


def get_model(config):
    if config.data.dataset == 'CIFAR10' or config.data.dataset == 'CELEBA':
        return NCSNv2(config).to(config.device)
    elif config.data.dataset == "FFHQ":
        return NCSNv2Deepest(config).to(config.device)
    elif config.data.dataset == 'LSUN':
        return NCSNv2Deeper(config).to(config.device)


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


class NCSN(ScoreAdapter):
    def __init__(self):
        config_fname = Path(__file__).resolve().parent / "ncsn" / "bedroom.yml"
        with config_fname.open("r") as f:
            config = yaml.safe_load(f)
            config = dict2namespace(config)

        config.device = device

        states = torch.load(
            self.checkpoint_root() / "ncsn/exp/logs/bedroom/checkpoint_150000.pth"
        )

        model = get_model(config)
        model = torch.nn.DataParallel(model)
        model.load_state_dict(states[0], strict=True)

        if config.model.ema:
            ema_helper = EMAHelper(mu=config.model.ema_rate)
            ema_helper.register(model)
            ema_helper.load_state_dict(states[-1])
            # HC: update the model param with history ema.
            # if don't do this the colors of images become strangely saturated.
            # this is reported in the paper.
            ema_helper.ema(model)

        model = model.module  # remove DataParallel
        model.eval()
        self.model = model
        self._data_shape = (3, config.data.image_size, config.data.image_size)

        self.σs = model.sigmas.cpu().numpy()
        self._device = device

    def data_shape(self):
        return self._data_shape

    def samps_centered(self):
        return False

    @property
    def σ_max(self):
        return self.σs[0]

    @property
    def σ_min(self):
        return self.σs[-1]

    @torch.no_grad()
    def denoise(self, xs, σ):
        σ, j = self.snap_t_to_nearest_tick(σ)
        N = xs.shape[0]
        cond_t = torch.tensor([j] * N, dtype=torch.long, device=self.device)
        score = self.model(xs, cond_t)
        Ds = xs + score * (σ ** 2)
        return Ds

    def unet_is_cond(self):
        return False

    def use_cls_guidance(self):
        return False

    def snap_t_to_nearest_tick(self, t):
        j = np.abs(t - self.σs).argmin()
        return self.σs[j], j
