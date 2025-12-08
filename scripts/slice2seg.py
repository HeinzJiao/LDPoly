"""
Utility functions adapted from:

    https://github.com/lin-tianyu/Stable-Diffusion-Seg

Original script was an evaluation entry point for multiple medical
segmentation datasets (Synapse, REFUGE2, STS-3D, CVC, KSEG). Here we
only keep the generic helper functions (prepare_for_first_stage,
dice_score, iou_score, load_model_from_config, calculate_volume_dice)
for reuse in our own pipelines.
"""

from einops import rearrange
import torch
import numpy as np
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config


def prepare_for_first_stage(x, gpu=True):
    x = x.clone().detach()
    if len(x.shape) == 3:
        x = x[None, ...]
    x = rearrange(x, 'b h w c -> b c h w')
    if gpu:
        x = x.to(memory_format=torch.contiguous_format).float().cuda()
    else:
        x = x.float()
    return x


def dice_score(pred, targs):
    assert pred.shape == targs.shape, (pred.shape, targs.shape)
    pred[pred > 0] = 1
    targs[targs > 0] = 1
    # if targs is None:
    #     return None
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)
    if pred.sum() == 0 and targs.sum() == 0:
        return 1
    elif pred.sum() > 0 and targs.sum() == 0:
        return 1
    elif pred.sum() > 0 and targs.sum() > 0:
        # intersection = (pred * targs).sum()
        # union = pred.sum() + targs.sum() - intersection
        # return (2. * intersection) / (union + 10e-6)
        return (2. * (pred * targs).sum()) / (pred.sum() + targs.sum() + 1e-10)
    else:
        return 0


def iou_score(pred, targs):
    pred[pred > 0] = 1
    targs[targs > 0] = 1
    # pred = (pred > 0.5).astype(np.float32)
    # targs = (targs > 0.5).astype(np.float32)
    if pred.sum() == 0 and targs.sum() == 0:
        return 1
    intersection = (pred * targs).sum()
    union = pred.sum() + targs.sum() - intersection
    # return intersection, union
    return intersection / (union + 1e-10)


def load_model_from_config(config, ckpt):
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    # print(set(key.split(".")[0] for key in sd.keys()))
    print(f"\033[31m[Model Weights Rewrite]: Loading model from {ckpt}\033[0m")
    m, u = model.load_state_dict(sd, strict=False)
    # if len(m) > 0 and verbose:
    print("\033[31mmissing keys:\033[0m")
    print(m)
    # if len(u) > 0 and verbose:
    print("\033[31munexpected keys:\033[0m")
    print(u)
    # model.cuda()
    model.eval()
    return model, pl_sd


def calculate_volume_dice(**kwargs):
    # inter_list, union_list, pred_sum, gt_sum = kwargs
    inter = sum(kwargs["inter_list"])
    union = sum(kwargs["union_list"])
    if kwargs["pred_sum"] > 0 and kwargs["gt_sum"] > 0:
        return 2 * inter / (union + 1e-10)
    elif kwargs["pred_sum"] > 0 and kwargs["gt_sum"] == 0:
        return 1
    else:
        return 0
