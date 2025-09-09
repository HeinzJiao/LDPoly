"""
This script evaluates the performance of the LDPoly model.

Usage:
    python -u scripts/evaluate.py --dataset [shanghai/shanghai_vertex] \
    --outdir [output dir] \
    --run [experiment name] \
    --model_ckpt [checkpoint name] \
    --model \
    --save_vertices \
    --sampler \
    --save_results \
    --save_logits_npy \
    --ddim_steps \

Arguments:
    --dataset: The dataset to use for evaluation.
    --outdir: Directory to save the output results.
    --run: The name of the experiment run.
    --model_ckpt: The name of the model checkpoint to load.
    --save_results: If indicated, save the visualized results.
    --model: the model used for evaluation. "building_mask" or "vertex_heatmap" or "building_mask_vertex_heatmap"
    --save_vertices: If indicated, save the extracted vertices.
    --sampler: the sampler used for sampling. "direct" or "ddim" or "plms" or "ddpm"
    --save_logits_npy: If indicated, save the predicted building segmentation probability map
    --ddim_steps:

Example:
"""
import argparse, os, sys, glob

import torch
import numpy as np
from omegaconf import OmegaConf
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
import time
from pytorch_lightning import seed_everything
from torch import autocast
from torch.utils.data import DataLoader
from contextlib import contextmanager, nullcontext

from ldm.util import instantiate_from_config, default
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler
from ldm.models.diffusion.dpm_solver import DPMSolverSampler

from ldm.data.synapse import SynapseValidation, SynapseValidationVolume
from ldm.data.refuge2 import REFUGE2Validation, REFUGE2Test
from ldm.data.sts3d import STS3DValidation, STS3DTest
from ldm.data.cvc import CVCValidation, CVCTest
from ldm.data.kseg import KSEGValidation, KSEGTest
import torchvision
from PIL import Image
# from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from transformers import AutoFeatureExtractor


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


def main():
    parser = argparse.ArgumentParser()
    # saving settings
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/txt2img-samples")
    parser.add_argument("--name", type=str, help="name to call this inference", default="test")
    parser.add_argument("--run", type=str, nargs="?", help="the name of your experiment",
                        default="2024-07-13T17-50-40_cvc")
    parser.add_argument("--model_ckpt", type=str, nargs="?", help="the name of the checkpoint",
                        default="epoch=991-step=121999.ckpt")
    # sampler settings
    parser.add_argument("--sampler", type=str,
                        choices=["ddpm", "direct", "ddim", "plms", "dpm_solver"],
                        help="the sampler used for sampling", )
    parser.add_argument("--ddim_steps", type=int, default=200, help="number of ddim sampling steps", )
    parser.add_argument("--ddim_eta", type=float, default=1.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling", )
    # dataset settings
    parser.add_argument("--dataset", type=str,  # '-b' for binary, '-m' for multi
                        help="uses the model trained for given dataset", )
    # sampling settings
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across samples ", )
    parser.add_argument("--H", type=int, default=256, help="image height, in pixel space", )
    parser.add_argument("--W", type=int, default=256, help="image width, in pixel space", )
    parser.add_argument("--C", type=int, default=4, help="latent channels", )
    parser.add_argument("--f", type=int, default=8, help="downsampling factor", )
    parser.add_argument("--n_samples", type=int, default=1,
                        help="how many samples to produce for each given prompt. A.k.a. batch size", )
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt",
                        help="path to checkpoint of model", )
    parser.add_argument("--seed", type=int, default=0,
                        help="the seed (for reproducible sampling)", )
    parser.add_argument("--times", type=int, default=1,
                        help="times of testing for stability evaluation", )
    parser.add_argument("--save_results", action='store_true',  # will slow down inference
                        help="saving the visualized predictions for the whole test set.", )
    parser.add_argument("--model", type=str,
                        help="the type of model", )
    parser.add_argument("--save_vertices", action='store_true',  # will slow down inference
                        help="saving the extracted vertices for the whole test set.", )
    parser.add_argument("--save_logits_npy", action='store_true',  # will slow down inference
                        help="saving the predicted segmentation probability map for the whole test set.", )
    opt = parser.parse_args()

    if opt.dataset == "cvc":
        run = opt.run
        model_ckpt = opt.model_ckpt
        print("Evaluate on cvc dataset in binary segmentation manner.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/{model_ckpt}"
        opt.outdir = "outputs/slice2seg-samples-cvc"
        dataset = CVCTest()
    elif opt.dataset == "deventer_road_mask_vertex_heatmap":
        run = opt.run
        model_ckpt = opt.model_ckpt
        print("Evaluate on deventer road mask and vertex heatmap dataset.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/{model_ckpt}"
        from ldm.data.shanghai_building_mask_vertex_heatmap import DeventerRoadTest, EnschedeRoadTest, GeethornRoadTest
        dataset = DeventerRoadTest()
        # dataset = EnschedeRoadTest()
        # dataset = GeethornRoadTest()
    elif opt.dataset == "vaihingen_map_generalization":
        run = opt.run
        model_ckpt = opt.model_ckpt
        print("Evaluate on vaihingen map generalization dataset.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/{model_ckpt}"
        from ldm.data.vaihingen_map_generalization import Geb10Test
        dataset = Geb10Test()
    elif opt.dataset == "vaihingen_map_generalization_sigma2.5_geb15":
        run = opt.run
        model_ckpt = opt.model_ckpt
        print("Evaluate on vaihingen map generalization sigma2.5 geb15 dataset.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = f"logs/{run}/checkpoints/{model_ckpt}"
        from ldm.data.vaihingen_map_generalization import Geb15Test
        dataset = Geb15Test()
    else:
        raise NotImplementedError(f"Not implement for dataset {opt.dataset}")

    data = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False)  # len(data)=357

    config = OmegaConf.load(f"{opt.config}")

    if opt.model == "building_mask_vertex_heatmap":
        # regress building mask and vertex heatmap simultaneously
        config["model"]["target"] = "ldm.models.diffusion.custom_ddpm_building_mask_vertex_heatmap.ExtendedLatentDiffusion"
    elif opt.model == "vaihingen_map_generalization":
        config["model"]["target"] = "ldm.models.diffusion.custom_ddpm_vaihingen_map_generalization.ExtendedLatentDiffusion"
    else:
        raise NotImplementedError(f"The model option '{opt.model}' is not implemented yet.")

    model, pl_sd = load_model_from_config(config, f"{opt.ckpt}")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    os.makedirs(opt.outdir, exist_ok=True)
    for idx in range(opt.times):  # opt.times=1
        if opt.times > 1:  # if test only once, use specified seed.
            opt.seed = idx
        seed_everything(opt.seed)

        model_info = model_ckpt.split('.')[0]

        outpath = os.path.join(opt.outdir, str(model_info))
        os.makedirs(outpath, exist_ok=True)

        model.log_images_loop(data,
                              save_results=opt.save_results,
                              save_dir=outpath,
                              used_sampler=opt.sampler,
                              save_samples_seg_logits_npy=opt.save_logits_npy,
                              save_samples_heat_logits_npy=opt.save_logits_npy
                              )

        if opt.times > 1:
            print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
                  f" \nEnjoy.")


if __name__ == "__main__":
    main()
