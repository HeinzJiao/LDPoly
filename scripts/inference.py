"""
Single-image / folder inference for LDPoly (segmentation + vertex heatmap + polygonization).

This script:
    1) Loads a trained LDPoly model from a given run + checkpoint;
    2) Runs inference on a single image or all images in a folder;
    3) Saves:
        - Predicted segmentation probability maps (.npy + .png);
        - Predicted vertex heatmaps (.npy + .png);
        - Polygonized instance masks overlaid on the original image (.png).

Example:
    PYTHONPATH=./:$PYTHONPATH python scripts/inference.py \
        --input path/to/any/image_folder \
        --outdir path/to/any/image_folder \
        --run 2024-12-24T23-55-18_deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed \
        --model_ckpt epoch=824-step=739199.ckpt \
        --sampler ddim \
        --ddim_steps 20 \
        --d_th 5
"""

import os
import glob
import argparse
import json
import cv2
import numpy as np
import torch
from PIL import Image
from einops import rearrange
from omegaconf import OmegaConf
from tqdm import tqdm
from skimage.measure import label, regionprops
from ldm.util import instantiate_from_config
from scripts.extract_vertices_from_heatmap import extract_vertices_from_heatmap
from scripts.polygonization import get_poly


# -------------------------------------------------------------------------
#  Image loading / preprocessing
# -------------------------------------------------------------------------
def load_image(image_path, image_size=(256, 256)):
    """
    Load an RGB image and map it to the model's expected range [-1, 1].

    Args:
        image_path (str): Path to the input image.
        image_size (tuple): Target (width, height) for resizing.

    Returns:
        Tensor: Image tensor of shape [1, H, W, 3], in [-1, 1].
    """
    image = Image.open(image_path).convert("RGB")
    image = image.resize(image_size, Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image * 2.0) - 1.0  # [-1, 1]
    image = torch.from_numpy(image).unsqueeze(0)  # [1, H, W, 3]
    return image


# -------------------------------------------------------------------------
#  Model loading
# -------------------------------------------------------------------------
def load_model(config_path, checkpoint_path, device):
    """
    Load ExtendedLatentDiffusion model for joint seg+vertex inference.

    Args:
        config_path (str): Path to the project yaml.
        checkpoint_path (str): Path to the model checkpoint (.ckpt).
        device (torch.device): Target device.

    Returns:
        nn.Module: Loaded model in eval mode.
    """
    config = OmegaConf.load(config_path)
    # Use the inference-oriented variant of the model
    config["model"]["target"] = "ldm.models.diffusion.ddpm_seg_vertex_inference.ExtendedLatentDiffusion"

    model = instantiate_from_config(config.model)
    pl_sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(pl_sd["state_dict"], strict=False)

    model.to(device)
    model.eval()
    return model


# -------------------------------------------------------------------------
#  Single-image inference: run model + save logits
# -------------------------------------------------------------------------
@torch.no_grad()
def run_inference_on_image(
    model,
    image,
    save_dir,
    image_name,
    sampler="ddim",
    device="cuda",
    ddim_steps=20,
):
    """
    Run the model on a single preprocessed image and save intermediate outputs.

    Saves:
        - <save_dir>/samples_seg_<sampler>_logits_npy/<image_name>.npy
        - <save_dir>/samples_seg_<sampler>/<image_name>.png
        - <save_dir>/samples_heat_<sampler>_npy/<image_name>.npy
        - <save_dir>/samples_heat_<sampler>/<image_name>.png

    Args:
        model (nn.Module): LDPoly model.
        image (Tensor): [1, H, W, 3] image in [-1, 1].
        save_dir (str): Output root directory.
        image_name (str): Base name (without extension) of the image.
        sampler (str): "direct", "ddim", or "ddpm".
        device (str or torch.device): Target device.
        ddim_steps (int): Number of DDIM steps (if applicable).

    Returns:
        seg_logits (ndarray): 2D array (H, W) in [0, 1].
        heat_logits (ndarray): 2D array (H, W) in [0, 1].
    """
    image = image.to(device)

    # Build a minimal batch dict expected by ExtendedLatentDiffusion.log_images
    batch = {
        "image": image,                              # [1, H, W, 3], [-1, 1]
        "segmentation": torch.zeros_like(image),     # dummy, not used
        "heatmap": torch.zeros_like(image),          # dummy, not used
        "file_path_": [image_name],
        "class_id": torch.tensor([[-1]], device=device),
    }

    outputs = model.log_images(
        batch,
        sampler=sampler,
        ddim_steps=ddim_steps,
        plot_denoise_rows=False,
        plot_diffusion_rows=False,
        return_first_stage_outputs=False,
        plot_conditioning_latent=False,
    )

    os.makedirs(save_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Save segmentation logits (probability map)
    # ------------------------------------------------------------------
    seg_output = outputs[f"samples_seg_{sampler}"]  # [1, C, H, W], in [-1, 1]
    seg_output = torch.clamp(seg_output, -1.0, 1.0)
    seg_output = (seg_output + 1.0) / 2.0          # [0, 1]
    seg_output = seg_output.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    seg_npy = rearrange(seg_output.squeeze(0).cpu().numpy(), "c h w -> h w c")  # (H, W, 3)

    seg_npy_dir = os.path.join(save_dir, f"samples_seg_{sampler}_logits_npy")
    seg_png_dir = os.path.join(save_dir, f"samples_seg_{sampler}")
    os.makedirs(seg_npy_dir, exist_ok=True)
    os.makedirs(seg_png_dir, exist_ok=True)

    np.save(
        os.path.join(seg_npy_dir, f"{image_name}.npy"),
        seg_npy[:, :, 0].astype(np.float32),
    )
    cv2.imwrite(
        os.path.join(seg_png_dir, f"{image_name}.png"),
        (seg_npy * 255.0).astype(np.uint8),
    )

    # ------------------------------------------------------------------
    # Save vertex heatmap logits
    # ------------------------------------------------------------------
    heat_output = outputs[f"samples_heat_{sampler}"]  # [1, C, H, W], in [-1, 1]
    heat_output = torch.clamp(heat_output, -1.0, 1.0)
    heat_output = (heat_output + 1.0) / 2.0
    heat_output = heat_output.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
    heat_npy = rearrange(heat_output.squeeze(0).cpu().numpy(), "c h w -> h w c")

    heat_npy_dir = os.path.join(save_dir, f"samples_heat_{sampler}_npy")
    heat_png_dir = os.path.join(save_dir, f"samples_heat_{sampler}")
    os.makedirs(heat_npy_dir, exist_ok=True)
    os.makedirs(heat_png_dir, exist_ok=True)

    np.save(
        os.path.join(heat_npy_dir, f"{image_name}.npy"),
        heat_npy[:, :, 0].astype(np.float32),
    )
    cv2.imwrite(
        os.path.join(heat_png_dir, f"{image_name}.png"),
        (heat_npy * 255.0).astype(np.uint8),
    )

    print(f"[Inference] Processed: {image_name}")

    return seg_npy[:, :, 0], heat_npy[:, :, 0]


# -------------------------------------------------------------------------
#  Polygon visualization
# -------------------------------------------------------------------------
def visualize_predictions(image, polygons, image_name, save_dir, alpha=0.1, scale_factor=2):
    """
    Visualize polygon predictions on top of the input image.

    Args:
        image (ndarray): Input image in BGR, shape (H, W, 3), [0, 255].
        polygons (list): List of polygon lists per instance:
                         each instance: [poly_outer, poly_hole_1, ...],
                         each polygon: [x1, y1, x2, y2, ...].
        image_name (str): Base name for saving.
        save_dir (str): Output directory.
        alpha (float): Alpha for filled overlay.
        scale_factor (int): Visualization scaling factor for clarity.
    """
    os.makedirs(save_dir, exist_ok=True)

    # Upscale image
    image = cv2.resize(
        image,
        None,
        fx=scale_factor,
        fy=scale_factor,
        interpolation=cv2.INTER_LINEAR,
    )
    overlay = np.zeros_like(image, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Fill polygons on overlay
    # ------------------------------------------------------------------
    for segmentation in polygons:
        # Scale all polygons
        scaled_segments = []
        for poly in segmentation:
            pts = (np.array(poly, dtype=np.float32).reshape(-1, 2) * scale_factor).tolist()
            scaled_segments.append(pts)

        exterior = np.array(scaled_segments[0], dtype=np.int32)
        cv2.fillPoly(overlay, [exterior], (0, 255, 0))  # green outer region

        for interior in scaled_segments[1:]:
            interior = np.array(interior, dtype=np.int32)
            cv2.fillPoly(overlay, [interior], (0, 0, 0))  # cut holes

    # Blend overlay with image
    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # ------------------------------------------------------------------
    # Draw polygon outlines and vertices
    # ------------------------------------------------------------------
    for segmentation in polygons:
        scaled_segments = []
        for poly in segmentation:
            pts = (np.array(poly, dtype=np.float32).reshape(-1, 2) * scale_factor).tolist()
            scaled_segments.append(pts)

        exterior = np.array(scaled_segments[0], dtype=np.int32)
        cv2.polylines(image, [exterior], isClosed=True, color=(255, 255, 0), thickness=2)

        for v in exterior:
            cv2.circle(image, tuple(v), radius=3, color=(204, 102, 255), thickness=-1)

        for interior in scaled_segments[1:]:
            interior = np.array(interior, dtype=np.int32)
            cv2.polylines(image, [interior], isClosed=True, color=(255, 255, 0), thickness=2)
            for v in interior:
                cv2.circle(image, tuple(v), radius=3, color=(204, 102, 255), thickness=-1)

    output_path = os.path.join(save_dir, f"{image_name}.png")
    cv2.imwrite(output_path, image)
    print(f"[Visualization] Saved to: {output_path}")


# -------------------------------------------------------------------------
#  Main: single-image or folder inference
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Run LDPoly inference on a single image or a folder of images "
                    "and perform polygonization."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Path to a single image or a directory of images.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Output directory root.",
    )
    parser.add_argument(
        "--run",
        type=str,
        nargs="?",
        default="2024-07-13T17-50-40_cvc",
        help="Name of the experiment run (under logs/).",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        nargs="?",
        default="epoch=991-step=121999.ckpt",
        help="Checkpoint file name under logs/<run>/checkpoints/.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        default="ddim",
        choices=["direct", "ddim", "ddpm"],
        help="Sampling method for the diffusion model.",
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=20,
        help="Number of DDIM sampling steps (if applicable).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use: 'cuda' or 'cpu'.",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        nargs=2,
        default=[256, 256],
        help="Image size (width height) for resizing before inference.",
    )
    parser.add_argument(
        "--d_th",
        type=float,
        default=5.0,
        help="Distance threshold for snapping contour vertices to junctions.",
    )

    opt = parser.parse_args()

    # Resolve config / checkpoint paths from run name
    run = opt.run
    model_ckpt = opt.model_ckpt
    print(f"[Setup] Using run: {run}, ckpt: {model_ckpt}")

    config_path = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
    ckpt_path = os.path.join("logs", run, "checkpoints", model_ckpt)

    device = torch.device(opt.device if torch.cuda.is_available() else "cpu")
    model = load_model(config_path, ckpt_path, device)

    # ------------------------------------------------------------------
    # Determine input mode: single image or folder
    # ------------------------------------------------------------------
    if os.path.isfile(opt.input):
        # Single image
        img_path = opt.input
        image = load_image(img_path, image_size=tuple(opt.image_size))
        image_name = os.path.splitext(os.path.basename(img_path))[0]

        seg_npy, heat_npy = run_inference_on_image(
            model,
            image,
            opt.outdir,
            image_name,
            sampler=opt.sampler,
            device=device,
            ddim_steps=opt.ddim_steps,
        )

        # Vertex extraction
        junctions, _ = extract_vertices_from_heatmap(
            heat_npy,
            th=0.1,
            kernel_size=5,
            topk=300,
            upscale_factor=1,
        )

        # Polygonization
        logit = seg_npy
        mask = logit > 0.5
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)

        polygons = []
        for i, prop in enumerate(props):
            poly_list, score = get_poly(
                prop,
                logit,
                junctions,
                d_th=opt.d_th,
                vis_save_path=None,
                file_name=image_name + ".npy",
                region_idx=i,
            )
            if len(poly_list) == 0:
                continue
            polygons.append(poly_list)

        # Save polygons as JSON: one file per image
        poly_json_dir = os.path.join(opt.outdir, "polygons_json")
        os.makedirs(poly_json_dir, exist_ok=True)
        poly_json_path = os.path.join(poly_json_dir, f"{image_name}.json")
        with open(poly_json_path, "w") as f:
            json.dump(
                {
                    "image_file_name": os.path.basename(img_path),
                    "polygons": polygons,  # [[[x1,y1,...], [hole...], ...], ...]
                },
                f,
            )

        # Visualization
        img_np = image.squeeze(0).cpu().numpy()  # [H, W, 3], [-1, 1]
        img_np = ((img_np + 1.0) / 2.0 * 255.0).astype(np.uint8)  # [0, 255]
        img_np = img_np[..., ::-1]  # RGB -> BGR for OpenCV

        vis_dir = os.path.join(opt.outdir, "polygons_vis")
        visualize_predictions(img_np, polygons, image_name, vis_dir)

    elif os.path.isdir(opt.input):
        # Folder of images
        image_paths = sorted(glob.glob(os.path.join(opt.input, "*")))
        vis_dir = os.path.join(opt.outdir, "polygons_vis")
        os.makedirs(vis_dir, exist_ok=True)

        poly_json_dir = os.path.join(opt.outdir, "polygons_json")
        os.makedirs(poly_json_dir, exist_ok=True)

        for img_path in tqdm(image_paths, desc="Processing images"):
            if not os.path.isfile(img_path):
                continue

            image = load_image(img_path, image_size=tuple(opt.image_size))
            image_name = os.path.splitext(os.path.basename(img_path))[0]

            seg_npy, heat_npy = run_inference_on_image(
                model,
                image,
                opt.outdir,
                image_name,
                sampler=opt.sampler,
                device=device,
                ddim_steps=opt.ddim_steps,
            )

            junctions, _ = extract_vertices_from_heatmap(
                heat_npy,
                th=0.1,
                kernel_size=5,
                topk=300,
                upscale_factor=1,
            )

            logit = seg_npy
            mask = logit > 0.5
            labeled_mask = label(mask)
            props = regionprops(labeled_mask)

            polygons = []
            for i, prop in enumerate(props):
                poly_list, score = get_poly(
                    prop,
                    logit,
                    junctions,
                    d_th=opt.d_th,
                    vis_save_path=None,
                    file_name=image_name + ".npy",
                    region_idx=i,
                )
                if len(poly_list) == 0:
                    continue
                polygons.append(poly_list)

            # Save polygons for this image as JSON
            poly_json_path = os.path.join(poly_json_dir, f"{image_name}.json")
            with open(poly_json_path, "w") as f:
                json.dump(
                    {
                        "image_file_name": os.path.basename(img_path),
                        "polygons": polygons,
                    },
                    f,
                )

            # Visualization
            img_np = image.squeeze(0).cpu().numpy()
            img_np = ((img_np + 1.0) / 2.0 * 255.0).astype(np.uint8)
            img_np = img_np[..., ::-1]

            visualize_predictions(img_np, polygons, image_name, vis_dir)
    else:
        print("[Error] Invalid input path. Please provide a valid image file or directory.")


if __name__ == "__main__":
    main()
