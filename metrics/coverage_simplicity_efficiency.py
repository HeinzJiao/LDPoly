"""
Compute pixel-level coverage, polygon simplicity, and vertex efficiency metrics between
COCO-format polygon predictions and ground truth:

    - IoU
    - Boundary IoU (B-IoU)
    - Complexity-aware IoU (C-IoU)
    - Simplicity-aware IoU (S-IoU)
    - N-ratio (vertex efficiency)

Usage (standalone):
    python coverage_simplicity_efficiency.py --gt-file path/to/gt.json --dt-file path/to/pred.json

Typically this function is called from a higher-level evaluation.py.
"""
from pycocotools.coco import COCO
from pycocotools import mask as cocomask
import numpy as np
import json
import argparse
from tqdm import tqdm
import os
import cv2


def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1
    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]
    # G_d intersects G in the paper.
    return mask - mask_erode


def calc_iou(a, b):
    """Compute standard IoU between two binary masks."""
    i = np.logical_and(a, b)  # intersection
    u = np.logical_or(a, b)  # union
    I = np.sum(i)
    U = np.sum(u)

    iou = I/(U + 1e-9)

    is_void = U == 0
    if is_void:
        return 1.0
    else:
        return iou


def compute_n_ratio(pred_vertices, gt_vertices):
    """Compute vertex efficiency ratio N_pred / N_gt with edge cases handled."""
    # Handle cases where the ground truth has no vertices
    if gt_vertices == 0:
        if pred_vertices == 0:
            return 1  # Both are zero, return a ratio of 1
        else:
            return float('NaN')  # If no ground truth vertices but there are predicted, return NaN
    else:
        return pred_vertices / gt_vertices  # Normal case, compute ratio


def _decode_coco_mask(annotation, height, width):
    """Decode COCO polygon segmentation to a single binary mask and vertex count."""
    rle = cocomask.frPyObjects(annotation["segmentation"], height, width)
    m = cocomask.decode(rle)

    # Handle exterior + interior holes
    if m.ndim > 2:
        final_mask = m[:, :, 0].copy()
        for i in range(1, m.shape[-1]):
            final_mask = np.logical_and(final_mask, np.logical_not(m[:, :, i]))
        final_mask = final_mask.astype(np.uint8)
    else:
        final_mask = m.astype(np.uint8)

    num_vertices = len(annotation["segmentation"][0]) // 2
    return final_mask.reshape(height, width), num_vertices


def compute_cse(input_json, gti_annotations, output_dir=None):
    """
    Main evaluation routine.

    Args:
        input_json: Path to COCO-format prediction file.
        gti_annotations: Path to COCO-format ground truth annotation file.
        output_dir: (Optional) directory to save per-image results.
    """
    coco_gt = COCO(gti_annotations)

    with open(input_json, "r") as f:
        submission_file = json.load(f)
    coco_dt = COCO(gti_annotations)
    coco_dt = coco_dt.loadRes(submission_file)

    all_image_ids = coco_gt.getImgIds()
    pred_image_ids = set(coco_dt.getImgIds(catIds=coco_dt.getCatIds()))

    # Sanity check for ID consistency
    if not pred_image_ids.issubset(all_image_ids):
        missing_ids = pred_image_ids - set(all_image_ids)
        raise ValueError(
            f"Predictions contain image IDs not found in ground truth: {missing_ids}"
        )

    bar = tqdm(all_image_ids)

    list_iou = []
    list_ciou = []
    list_siou_sigma = []
    list_siou_2sigma = []
    list_siou_3sigma = []
    list_siou = []
    list_n_ratio = []
    list_boundary_iou = []
    pss = []
    sps = []
    sps_sigma = []
    sps_2sigma = []
    sps_3sigma = []
    results = []

    for image_id in bar:
        img_info = coco_gt.loadImgs(image_id)[0]
        image_name = os.path.basename(img_info["file_name"])

        # ---------------- GT mask & vertex count ----------------
        ann_ids_gt = coco_gt.getAnnIds(imgIds=img_info["id"])
        if not ann_ids_gt:
            # Skip images without GT polygons
            continue

        anns_gt = coco_gt.loadAnns(ann_ids_gt)

        mask_gt = None
        N_gt = 0
        for idx, ann in enumerate(anns_gt):
            m, n_v = _decode_coco_mask(
                ann, img_info["height"], img_info["width"]
            )
            if idx == 0:
                mask_gt = m
            else:
                mask_gt = mask_gt + m
            N_gt += n_v

        mask_gt = mask_gt != 0

        # ---------------- Pred mask & vertex count ----------------
        if image_id in pred_image_ids:
            ann_ids_dt = coco_dt.getAnnIds(imgIds=img_info["id"])
            anns_dt = coco_dt.loadAnns(ann_ids_dt)

            mask_pred = None
            N_pred = 0
            for idx, ann in enumerate(anns_dt):
                m, n_v = _decode_coco_mask(
                    ann, img_info["height"], img_info["width"]
                )
                if idx == 0:
                    mask_pred = m
                else:
                    mask_pred = mask_pred + m
                N_pred += n_v

            mask_pred = mask_pred != 0
        else:
            mask_pred = np.zeros(
                (img_info["height"], img_info["width"]), dtype=np.uint8
            )
            N_pred = 0

        # ---------------- IoU & Boundary IoU ----------------
        iou = calc_iou(mask_pred, mask_gt)

        boundary_pred = mask_to_boundary(mask_pred.astype(np.uint8), dilation_ratio=0.02)
        boundary_gt = mask_to_boundary(mask_gt.astype(np.uint8), dilation_ratio=0.02)
        boundary_iou = calc_iou(boundary_pred, boundary_gt)

        # ---------------- C-IoU (vertex efficiency) ----------------
        ps = 1.0 - np.abs(N_pred - N_gt) / (N_pred + N_gt + 1e-9)

        # ---------------- S-IoU (polygon simplicity) ----------------
        N0_sigma, N0_2sigma, N0_3sigma = 50, 90, 500

        sp_sigma = (1 + np.exp(0.1 * (3 - N0_sigma))) / (
            1 + np.exp(0.1 * (N_pred - N0_sigma))
        )
        sp_2sigma = (1 + np.exp(0.1 * (3 - N0_2sigma))) / (
            1 + np.exp(0.1 * (N_pred - N0_2sigma))
        )
        sp_3sigma = (1 + np.exp(0.1 * (3 - N0_3sigma))) / (
            1 + np.exp(0.1 * (N_pred - N0_3sigma))
        )

        siou_sigma = iou * sp_sigma
        siou_2sigma = iou * sp_2sigma
        siou_3sigma = iou * sp_3sigma
        siou = (siou_sigma + siou_2sigma + siou_3sigma) / 3.0

        # ---------------- N-ratio ----------------
        n_ratio = compute_n_ratio(N_pred, N_gt)

        # ---------------- Accumulate ----------------
        list_iou.append(iou)
        list_ciou.append(iou * ps)
        list_siou_sigma.append(siou_sigma)
        list_siou_2sigma.append(siou_2sigma)
        list_siou_3sigma.append(siou_3sigma)
        list_siou.append(siou)
        list_boundary_iou.append(boundary_iou)
        pss.append(ps)
        sps.append((sp_sigma + sp_2sigma + sp_3sigma) / 3.0)
        sps_sigma.append(sp_sigma)
        sps_2sigma.append(sp_2sigma)
        sps_3sigma.append(sp_3sigma)
        list_n_ratio.append(n_ratio)

        results.append(
            {
                "image_name": image_name,
                "n_ratio": round(float(n_ratio), 2) if not np.isnan(n_ratio) else None,
                "iou": round(float(iou), 2),
                "c_iou": round(float(iou * ps), 2),
                "boundary_iou": round(float(boundary_iou), 2),
                "s_iou": round(float(siou), 2),
            }
        )

        bar.set_description(
            "iou: %2.4f, c-iou: %2.4f, boundary-iou: %2.4f, "
            "s-iou: %2.4f, s-iou-sigma: %2.4f, s-iou-2sigma: %2.4f, "
            "s-iou-3sigma: %2.4f, ps: %2.4f, n_ratio: %2.4f"
            % (
                np.mean(list_iou),
                np.mean(list_ciou),
                np.mean(list_boundary_iou),
                np.mean(list_siou),
                np.mean(list_siou_sigma),
                np.mean(list_siou_2sigma),
                np.mean(list_siou_3sigma),
                np.mean(pss),
                np.nanmean(list_n_ratio),
            )
        )

    # ---------------- Final summary ----------------
    print("Done!")
    print("Mean IoU: ", np.mean(list_iou))
    print("Mean Boundary IoU: ", np.mean(list_boundary_iou))
    print("Mean C-IoU: ", np.mean(list_ciou))
    print("Mean S-IoU: ", np.mean(list_siou))
    print("Mean N-Ratio: ", np.nanmean(list_n_ratio))

    # Optional: save per-image results if output_dir is given
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        annotation_name = os.path.splitext(os.path.basename(input_json))[0]
        out_path = os.path.join(output_dir, f"{annotation_name}_coverage_simplicity_metrics.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)
        print("Per-image metrics saved to:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute IoU, Boundary IoU, C-IoU, S-IoU and N-ratio."
    )
    parser.add_argument("--gt-file", required=True, help="COCO-format ground truth JSON.")
    parser.add_argument("--dt-file", required=True, help="COCO-format prediction JSON.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save per-image metrics JSON.",
    )
    args = parser.parse_args()

    compute_cse(
        input_json=args.dt_file,
        gti_annotations=args.gt_file,
        output_dir=args.output_dir,
    )
