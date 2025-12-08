"""
Extract vertices from predicted vertex heatmaps and save them as JSON.

This script:
    1) Loads per-image vertex heatmaps (as .npy files) from a user-specified folder;
    2) Optionally upscales the heatmaps;
    3) Applies non-maximum suppression (NMS) and top-k selection;
    4) Filters vertices by a confidence threshold;
    5) Saves the extracted vertices to a JSON file.

Expected heatmap layout:
    - Heatmaps are stored as single-channel 2D arrays (H x W) in .npy format.
    - All .npy files under the given heatmap directory will be processed.

Usage:
    python scripts/extract_vertices_from_heatmap.py \
        --heatmaps_dir path/to/any/heatmap_folder \
        --outputs_dir path/to/save/json \
        --th 0.1 \
        --sampler ddim \
        --upscale_factor 4 \
        --kernel_size 3

Example:
    python scripts/extract_vertices_from_heatmap.py \
        --heatmaps_dir "./outputs/deventer_road_reproduction/epoch=824-step=739199/samples_heat_ddim_npy" \
        --outputs_dir "./outputs/deventer_road_reproduction/epoch=824-step=739199" \
        --th 0.1 \
        --sampler ddim \
        --upscale_factor 4 \
        --kernel_size 3
"""

import os
import json
import argparse
import numpy as np
import torch
import torch.nn.functional as F


def non_maximum_suppression(a, kernel_size=3):
    """
    Apply local non-maximum suppression (NMS) to a heatmap tensor.

    Args:
        a (Tensor): Input heatmap of shape [N, C, H, W].
        kernel_size (int): Window size for max pooling (must be odd).

    Returns:
        Tensor: Tensor with the same shape as `a`, where only local maxima
        are preserved and all other locations are set to zero.
    """
    assert kernel_size % 2 == 1, "kernel_size must be an odd integer."
    kernel_size = int(kernel_size)
    pad = kernel_size // 2

    # Local max pooling
    ap = F.max_pool2d(a, kernel_size, stride=1, padding=pad)

    # Keep only positions equal to the local maximum
    mask = (a == ap).float()
    return a * mask


def extract_vertices_from_heatmap(heatmap, th, kernel_size, topk=300, upscale_factor=1):
    """
    Extract vertex candidates from a single heatmap.

    Steps:
        1) Convert to torch tensor and optionally upscale via bilinear interpolation.
        2) Apply NMS with the given kernel size.
        3) Select top-k highest responses.
        4) Filter by confidence threshold and rescale coordinates back to
           the original resolution.

    Args:
        heatmap (ndarray): 2D array of shape (H, W) with values in [0, 1].
        th (float): Confidence threshold for valid vertices.
        kernel_size (int): NMS kernel size (odd integer).
        topk (int): Maximum number of vertices to keep.
        upscale_factor (int): Factor to upscale the heatmap before NMS.

    Returns:
        vertices (ndarray): Array of shape (N, 2) with (x, y) coordinates.
        scores (ndarray): Array of shape (N,) with corresponding confidences.
    """
    # [1, 1, H, W]
    heatmap = torch.tensor(heatmap, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Optional upscaling
    if upscale_factor != 1:
        heatmap = F.interpolate(
            heatmap,
            scale_factor=upscale_factor,
            mode="bilinear",
            align_corners=False,
        )

    # NMS on [1, 1, H', W']
    heatmap_nms = non_maximum_suppression(heatmap, kernel_size)  # [1, 1, H', W']
    heatmap_nms = heatmap_nms.squeeze(0).squeeze(0)  # [H', W']

    height, width = heatmap_nms.shape
    heatmap_flat = heatmap_nms.reshape(-1)

    # Top-K
    scores, index = torch.topk(heatmap_flat, k=topk)
    y = (index // width).float()
    x = (index % width).float()

    # Project back to original resolution
    if upscale_factor != 1:
        x /= upscale_factor
        y /= upscale_factor

    vertices = torch.stack((x, y), dim=1)  # [topk, 2]
    mask = scores > th

    return vertices[mask].cpu().numpy(), scores[mask].cpu().numpy()


def extract_vertices_from_heatmap_folder(
    heatmaps_folder,
    image_dir,
    th,
    kernel_size,
    output_json_file,
    upscale_factor,
    visualize_vertices_dir=None,
):
    """
    Batch process all heatmaps in a folder and save extracted vertices to JSON.

    Args:
        heatmaps_folder (str): Directory containing heatmap .npy files.
        image_dir (str or None): Directory of original test images
            (reserved for optional visualization).
        th (float): Confidence threshold.
        kernel_size (int): NMS kernel size.
        output_json_file (str): Path to the output JSON file.
        upscale_factor (int): Heatmap upscale factor.
        visualize_vertices_dir (str or None): If provided, save visualizations there.
    """
    results = []

    os.makedirs(os.path.dirname(output_json_file), exist_ok=True)
    if visualize_vertices_dir is not None:
        os.makedirs(visualize_vertices_dir, exist_ok=True)

    for file_name in os.listdir(heatmaps_folder):
        if not file_name.endswith(".npy"):
            continue

        file_path = os.path.join(heatmaps_folder, file_name)

        # Load heatmap: 2D array (H, W)
        h_prediction = np.load(file_path)

        # Extract vertices
        vertices, scores = extract_vertices_from_heatmap(
            h_prediction,
            th=th,
            kernel_size=kernel_size,
            topk=300,
            upscale_factor=upscale_factor,
        )

        # Optional visualization (currently kept commented)
        # if visualize_vertices_dir is not None and image_dir is not None:
        #     image_name = file_name.split('.')[0] + '.png'
        #     image_path = os.path.join(image_dir, image_name)
        #     image = cv2.imread(image_path)
        #     for point in vertices:
        #         cv2.circle(image, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
        #     cv2.imwrite(os.path.join(visualize_vertices_dir, image_name), image)

        results.append(
            {
                "image_file_name": file_name,
                "extracted_vertices": vertices.tolist(),
            }
        )

    with open(output_json_file, "w") as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract vertices from heatmaps with a given threshold and save results."
    )

    parser.add_argument(
        "--heatmaps_dir",
        type=str,
        required=True,
        help="Directory containing per-image vertex heatmaps saved as .npy files.",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="Directory where JSON results will be saved.",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        required=False,
        default=None,
        help="Directory of test images (required only if visualization is enabled).",
    )
    parser.add_argument(
        "--th",
        type=float,
        required=True,
        help="Confidence threshold for filtering vertices.",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        required=True,
        help="NMS kernel size (odd integer).",
    )
    parser.add_argument(
        "--upscale_factor",
        type=int,
        required=True,
        help="Upscale factor applied to the heatmap before NMS.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["direct", "ddim", "ddpm"],
        required=True,
        help="Sampling method used during testing (direct, ddim, or ddpm).",
    )

    args = parser.parse_args()

    heatmaps_folder = args.heatmaps_dir
    outputs_dir = args.outputs_dir
    image_dir = args.image_dir
    th = args.th
    kernel_size = args.kernel_size
    upscale_factor = args.upscale_factor
    sampler = args.sampler

    output_json_file = os.path.join(
        outputs_dir,
        f"output_vertices_from_heatmap_x{upscale_factor}_{sampler}_th-{th}_k-{kernel_size}.json",
    )

    # If you want to enable visualization, uncomment and configure:
    # visualize_vertices_dir = os.path.join(outputs_dir, f"visualize_vertices_{sampler}_th-{th}")
    # if visualize_vertices_dir is not None and image_dir is None:
    #     raise ValueError("--image_dir must be provided when visualization is enabled.")
    # extract_vertices_from_heatmap_folder(
    #     heatmaps_folder,
    #     image_dir,
    #     th,
    #     kernel_size,
    #     output_json_file,
    #     upscale_factor=upscale_factor,
    #     visualize_vertices_dir=visualize_vertices_dir,
    # )

    extract_vertices_from_heatmap_folder(
        heatmaps_folder,
        image_dir,
        th,
        kernel_size,
        output_json_file,
        upscale_factor=upscale_factor,
        visualize_vertices_dir=None,
    )
