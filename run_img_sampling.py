from pathlib import Path
import numpy as np
import torch

from misc import torch_samps_to_imgs
from adapt import Karras, ScoreAdapter, power_schedule
from adapt_gddpm import GuidedDDPM
from adapt_ncsn import NCSN as _NCSN
# from adapt_vesde import VESDE  # not included to prevent import conflicts
from adapt_sd import StableDiffusion

from my.utils import tqdm, EventStorage, HeartBeat, EarlyLoopBreak
from my.config import BaseConf, dispatch
from my.utils.seed import seed_everything


class GDDPM(BaseConf):
    """Guided DDPM from OpenAI"""
    model:          str = "m_lsun_256"
    lsun_cat:       str = "bedroom"
    imgnet_cat:     int = -1

    def make(self):
        args = self.dict()
        model = GuidedDDPM(**args)
        return model


class SD(BaseConf):
    """Stable Diffusion"""
    variant:        str = "v1"
    v2_highres:     bool = False
    prompt:         str = "a photograph of an astronaut riding a horse"
    scale:          float = 3.0  # classifier free guidance scale
    precision:      str = 'autocast'

    def make(self):
        args = self.dict()
        model = StableDiffusion(**args)
        return model


class SDE(BaseConf):
    def make(self):
        args = self.dict()
        model = VESDE(**args)
        return model


class NCSN(BaseConf):
    def make(self):
        args = self.dict()
        model = _NCSN(**args)
        return model


class KarrasGen(BaseConf):
    family:         str = "gddpm"
    gddpm:          GDDPM = GDDPM()
    sd:             SD = SD()
    # sde:            SDE = SDE()
    ncsn:           NCSN = NCSN()

    batch_size:     int = 10
    num_images:     int = 1250
    num_t:          int = 40
    σ_max:          float = 80.0
    heun:           bool = True
    langevin:       bool = False
    cls_scaling:    float = 1.0  # classifier guidance scaling

    def run(self):
        args = self.dict()
        family = args.pop("family")
        model = getattr(self, family).make()
        self.karras_generate(model, **args)

    @staticmethod
    def karras_generate(
        model: ScoreAdapter,
        batch_size, num_images, σ_max, num_t, langevin, heun, cls_scaling,
        **kwargs
    ):
        del kwargs  # removed extra args
        num_batches = num_images // batch_size

        fuse = EarlyLoopBreak(5)
        with tqdm(total=num_batches) as pbar, \
            HeartBeat(pbar) as hbeat, \
                EventStorage() as metric:

            all_imgs = []

            for _ in range(num_batches):
                if fuse.on_break():
                    break

                pipeline = Karras.inference(
                    model, batch_size, num_t,
                    init_xs=None, heun=heun, σ_max=σ_max,
                    langevin=langevin, cls_scaling=cls_scaling
                )

                for imgs in tqdm(pipeline, total=num_t+1, disable=False):
                    # _std = imgs.std().item()
                    # print(_std)
                    hbeat.beat()
                    pass

                if isinstance(model, StableDiffusion):
                    imgs = model.decode(imgs)

                imgs = torch_samps_to_imgs(imgs, uncenter=model.samps_centered())
                all_imgs.append(imgs)

                pbar.update()

            all_imgs = np.concatenate(all_imgs, axis=0)
            metric.put_artifact("imgs", ".npy", lambda fn: np.save(fn, all_imgs))
            metric.step()
            hbeat.done()


class SMLDGen(BaseConf):
    family:         str = "ncsn"
    gddpm:          GDDPM = GDDPM()
    # sde:            SDE = SDE()
    ncsn:           NCSN = NCSN()

    batch_size:     int = 16
    num_images:     int = 16
    num_stages:     int = 80
    num_steps:      int = 15
    σ_max:          float = 80.0
    ε:              float = 1e-5

    def run(self):
        args = self.dict()
        family = args.pop("family")
        model = getattr(self, family).make()
        self.smld_generate(model, **args)

    @staticmethod
    def smld_generate(
        model: ScoreAdapter,
        batch_size, num_images, num_stages, num_steps, σ_max, ε,
        **kwargs
    ):
        num_batches = num_images // batch_size
        σs = power_schedule(σ_max, model.σ_min, num_stages)
        σs = [model.snap_t_to_nearest_tick(σ)[0] for σ in σs]

        fuse = EarlyLoopBreak(5)
        with tqdm(total=num_batches) as pbar, \
            HeartBeat(pbar) as hbeat, \
                EventStorage() as metric:

            all_imgs = []

            for _ in range(num_batches):
                if fuse.on_break():
                    break

                init_xs = torch.rand(batch_size, *model.data_shape(), device=model.device)
                if model.samps_centered():
                    init_xs = init_xs * 2 - 1  # [0, 1] -> [-1, 1]

                pipeline = smld_inference(
                    model, σs, num_steps, ε, init_xs
                )

                for imgs in tqdm(pipeline, total=(num_stages * num_steps)+1, disable=False):
                    pbar.set_description(f"{imgs.max().item():.3f}")
                    metric.put_scalars(
                        max=imgs.max().item(), min=imgs.min().item(), std=imgs.std().item()
                    )
                    metric.step()
                    hbeat.beat()

                pbar.update()
                imgs = torch_samps_to_imgs(imgs, uncenter=model.samps_centered())
                all_imgs.append(imgs)

            all_imgs = np.concatenate(all_imgs, axis=0)
            metric.put_artifact("imgs", ".npy", lambda fn: np.save(fn, all_imgs))
            metric.step()
            hbeat.done()


def smld_inference(model, σs, num_steps, ε, init_xs):
    from math import sqrt
    # not doing conditioning or cls guidance; for gddpm only lsun works; fine.

    xs = init_xs
    yield xs

    for i in range(len(σs)):
        α_i = ε * ((σs[i] / σs[-1]) ** 2)
        for _ in range(num_steps):
            grad = model.score(xs, σs[i])
            z = torch.randn_like(xs)
            xs = xs + α_i * grad + sqrt(2 * α_i) * z
            yield xs


def load_np_imgs(fname):
    fname = Path(fname)
    data = np.load(fname)
    if fname.suffix == ".npz":
        imgs = data['arr_0']
    else:
        imgs = data
    return imgs


def visualize(max_n_imgs=16):
    import torchvision.utils as vutils
    from imageio import imwrite
    from einops import rearrange

    all_imgs = load_np_imgs("imgs/step_0.npy")

    imgs = all_imgs[:max_n_imgs]
    imgs = rearrange(imgs, "N H W C -> N C H W", C=3)
    imgs = torch.from_numpy(imgs)
    pane = vutils.make_grid(imgs, padding=2, nrow=4)
    pane = rearrange(pane, "C H W -> H W C", C=3)
    pane = pane.numpy()
    imwrite("preview.jpg", pane)


if __name__ == "__main__":
    seed_everything(0)
    dispatch(KarrasGen)
    visualize(16)
