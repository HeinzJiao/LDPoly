"""
Evaluate LDPoly on a test set.

This script:
    1) Loads the experiment config and checkpoint from the `logs/` directory;
    2) Builds the corresponding dataset and DataLoader;
    3) Runs inference via `model.log_images_loop(...)` over the whole test set;
    4) Optionally saves visualized predictions and logits (segmentation / vertex heatmaps).

Typical usage:
    PYTHONPATH=./:$PYTHONPATH python -u scripts/evaluate.py --dataset deventer_road_mask_vertex_heatmap \
        --outdir outputs/deventer_road_reproduction \
        --run 2024-12-24T23-55-18_deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed \
        --model_ckpt epoch=824-step=739199.ckpt \
        --model seg_vert_ldm \
        --sampler ddim \
        --save_results \
        --save_logits_npy \
        --ddim_steps 20

Command-line arguments (summary):
    --dataset           Dataset name used for evaluation.
    --outdir            Root directory where evaluation outputs will be written.
    --run               Experiment folder name under `logs/`, e.g., logs/<run>.
    --model_ckpt        Checkpoint filename inside `logs/<run>/checkpoints/`.
    --model             Model type used for evaluation.
    --sampler           Sampler name used inside `log_images_loop`, one of:
                        ["ddpm", "direct", "ddim"].
    --ddim_steps        Number of DDIM sampling steps (if DDIM is used).
    --ddim_eta          DDIM eta; eta=0.0 corresponds to deterministic sampling.
    --save_results      If set, save visualized predictions (PNG, etc.) to disk.
    --save_logits_npy   If set, save predicted segmentation / vertex heatmap
                        probability maps as .npy files for the whole test set.
    --seed              Random seed for reproducible sampling.
    --times             Number of repeated evaluation runs with different seeds
                        (if > 1, seed = 0,1,2,... times-1).
    --n_samples         Batch size for evaluation (samples processed per step).
"""

import argparse
import glob
import os

import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything

from ldm.util import instantiate_from_config
from ldm.data.cvc import CVCTest

# Deventer road dataset (mask + vertex heatmap)
from ldm.data.dataset_seg_vertex import (
    DeventerRoadTest,
    # EnschedeRoadTest,
    # GiethoornRoadTest,
    KSA_SpaceGeoAI_ITC_Project
)


def load_model_from_config(config, ckpt_path):
    """
    Load a model from a given OmegaConf config and checkpoint path.
    """
    print(f"\033[31m[Model Weights Rewrite]: Loading model from {ckpt_path}\033[0m")
    pl_sd = torch.load(ckpt_path, map_location="cpu")

    # Optional logging of training step information
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")

    # checkpoint is expected to store weights under "state_dict"
    state_dict = pl_sd["state_dict"]

    # Instantiate model from config
    model = instantiate_from_config(config.model)

    # Load weights (non-strict to be robust to minor key mismatches)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    print("\033[31mmissing keys:\033[0m")
    print(missing_keys)
    print("\033[31munexpected keys:\033[0m")
    print(unexpected_keys)

    model.eval()
    return model, pl_sd


def build_dataset_and_paths(opt):
    """
    Build the dataset object and resolve config / checkpoint paths based on CLI options.

    This function:
        - Uses `opt.run` and `opt.model_ckpt` to locate the config and checkpoint;
        - Creates the appropriate dataset instance for `opt.dataset`;
        - Sets sensible defaults for `opt.outdir` if not given.

    Args:
        opt (argparse.Namespace): Parsed command-line options (will be modified in-place).

    Returns:
        dataset (torch.utils.data.Dataset): Dataset object for evaluation.
        model_ckpt_str (str): Basename of the checkpoint (without file extension),
                              used to name the output folder.
    """
    run = opt.run
    model_ckpt = opt.model_ckpt

    if opt.dataset == "cvc":
        print("Evaluate on CVC dataset in binary segmentation manner.")
        # Config and checkpoint are taken from the corresponding run directory
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = os.path.join("logs", run, "checkpoints", model_ckpt)
        opt.outdir = opt.outdir or "outputs/slice2seg-samples-cvc"
        dataset = CVCTest()

    elif opt.dataset == "deventer_road_mask_vertex_heatmap":
        print("Evaluate on Deventer road mask and vertex heatmap dataset.")
        opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
        opt.ckpt = os.path.join("logs", run, "checkpoints", model_ckpt)

        # Default: use Deventer test set
        dataset = DeventerRoadTest()

        # If you want to test cross-domain generalization to other areas,
        # comment the line above and uncomment one of the following:
        # dataset = EnschedeRoadTest()
        # dataset = GiethoornRoadTest()
        # dataset = KSA_SpaceGeoAI_ITC_Project()

    else:
        raise NotImplementedError(f"[build_dataset_and_paths] Dataset '{opt.dataset}' is not supported yet.")

    # Strip extension for nicer folder names: "epoch=xxx-step=yyy"
    model_ckpt_str = model_ckpt.split(".")[0]

    return dataset, model_ckpt_str


def main():
    parser = argparse.ArgumentParser(description="Evaluate LDPoly / latent-diffusion models on test datasets.")

    # -------------------------------------------------------------------------
    # General saving / bookkeeping settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        default=None,
        help="Root directory to write results to (will create a subfolder per checkpoint).",
    )
    parser.add_argument(
        "--run",
        type=str,
        nargs="?",
        default="2024-07-13T17-50-40_cvc",
        help="Name of the experiment run under `logs/`.",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        nargs="?",
        default="epoch=991-step=121999.ckpt",
        help="Checkpoint filename under `logs/<run>/checkpoints/`.",
    )

    # -------------------------------------------------------------------------
    # Sampler settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["ddpm", "direct", "ddim"],
        help="Sampler type to be used inside `log_images_loop`.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=200,
        help="Number of DDIM sampling steps (if DDIM sampler is used).",
    )
    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=1.0,
        help="DDIM eta; eta=0.0 corresponds to deterministic sampling.",  # “ddim_eta” is not used.
    )

    # -------------------------------------------------------------------------
    # Dataset settings
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Dataset name used for evaluation, e.g. 'cvc' or 'deventer_road_mask_vertex_heatmap'.",
    )

    # -------------------------------------------------------------------------
    # Sampling & reproducibility
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--n_samples",
        type=int,
        default=1,
        help="Batch size during evaluation (#samples processed per step).",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/stable-diffusion/v1-inference.yaml",
        help="Fallback path to a config yaml (will be overridden by run-specific config).",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/ldm/stable-diffusion-v1/model.ckpt",
        help="Fallback path to a checkpoint (will be overridden by run-specific checkpoint).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Base seed for reproducible sampling.",
    )
    parser.add_argument(
        "--times",
        type=int,
        default=1,
        help=(
            "Number of repeated evaluation runs. "
            "If > 1, we will evaluate multiple times with different seeds (0..times-1)."
        ),
    )

    # -------------------------------------------------------------------------
    # I/O flags related to evaluation artifacts
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--save_results",
        action="store_true",
        help="If set, save visualized predictions for the whole test set (slows down inference).",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model type used for evaluation, e.g. 'seg_vert_ldm'.",
    )
    parser.add_argument(
        "--save_logits_npy",
        action="store_true",
        help="If set, save predicted segmentation probability maps / vertex heatmaps as .npy for the whole test set.",
    )

    opt = parser.parse_args()

    # -------------------------------------------------------------------------
    # Build dataset and resolve paths
    # -------------------------------------------------------------------------
    dataset, model_ckpt_str = build_dataset_and_paths(opt)

    # DataLoader over the test set, no shuffling
    data = DataLoader(dataset, batch_size=opt.n_samples, shuffle=False)

    # Load run-specific config
    config = OmegaConf.load(opt.config)

    # -------------------------------------------------------------------------
    # Select model target based on `--model`
    # -------------------------------------------------------------------------
    if opt.model == "seg_vert_ldm":
        # Regress building/road mask and vertex heatmap simultaneously
        config["model"]["target"] = (
            "ldm.models.diffusion.ddpm_seg_vertex_inference.ExtendedLatentDiffusion"
        )
    else:
        raise NotImplementedError(f"The model option '{opt.model}' is not implemented yet in this script.")

    # -------------------------------------------------------------------------
    # Instantiate model and move to device
    # -------------------------------------------------------------------------
    model, _ = load_model_from_config(config, opt.ckpt)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    # Make sure output directory exists
    os.makedirs(opt.outdir, exist_ok=True)

    # -------------------------------------------------------------------------
    # Main evaluation loop
    # -------------------------------------------------------------------------
    for idx in range(opt.times):
        # If times > 1, use different seeds for each repetition
        if opt.times > 1:
            current_seed = idx
        else:
            current_seed = opt.seed

        seed_everything(current_seed)
        print(f"[Eval] Run {idx + 1}/{opt.times} with seed = {current_seed}")

        # For each checkpoint, we create a dedicated subfolder under `outdir`
        outpath = os.path.join(opt.outdir, model_ckpt_str)
        os.makedirs(outpath, exist_ok=True)

        # NOTE: The behavior of `log_images_loop` is defined in the model class.
        #       Here we only pass high-level options and let the model handle:
        #           - sampling
        #           - visualization
        #           - saving logits / vertices (if implemented)
        model.log_images_loop(
            data,
            save_results=opt.save_results,
            save_dir=outpath,
            used_sampler=opt.sampler,
            ddim_steps=opt.ddim_steps,
            save_samples_seg_logits_npy=opt.save_logits_npy,
            save_samples_heat_logits_npy=opt.save_logits_npy,
            # TODO (optional): once `ExtendedLatentDiffusion.log_images_loop` supports it,
            # you can add:
            # save_vertices=opt.save_vertices,
        )

        if opt.times > 1:
            print(f"Your samples for run {idx + 1} are ready at:\n  {outpath}\n")


if __name__ == "__main__":
    main()
