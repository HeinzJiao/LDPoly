"""
Polygon regularity metrics.

Included metrics:
    1) PoLiS:
       - Shape fidelity between matched GT and prediction polygons.
    2) SCR (Smoothness Consistency Ratio):
       - Ratio of distorted (high-curvature) vertices in prediction vs GT.

Usage (standalone):
    python regularity.py --gt-file path/to/gt.json --dt-file path/to/pred.json

Typically this module is called from a higher-level evaluation.py, where
"eval-type = r" triggers both PoLiS and SCR evaluation.
"""
import os
import json
from collections import defaultdict

import numpy as np
from tqdm import tqdm
from shapely import geometry
from shapely.geometry import Polygon
from pycocotools import mask as maskUtils
from pycocotools.coco import COCO


# ----------------------------------------------------------------------
# 1. PoLiS-related helpers
# ----------------------------------------------------------------------
def bounding_box(points):
    """
    Compute [x_min, y_min, width, height] for a set of 2D points.

    Args:
        points (array-like): Shape (N, 2), list or ndarray of (x, y).

    Returns:
        list: [x_min, y_min, width, height]
    """
    bot_left_x, bot_left_y = float("inf"), float("inf")
    top_right_x, top_right_y = float("-inf"), float("-inf")

    for x, y in points:
        bot_left_x = min(bot_left_x, x)
        bot_left_y = min(bot_left_y, y)
        top_right_x = max(top_right_x, x)
        top_right_y = max(top_right_y, y)

    return [
        bot_left_x,
        bot_left_y,
        top_right_x - bot_left_x,
        top_right_y - bot_left_y,
    ]


def polis_one_side(coords, boundary):
    """
    One-sided PoLiS term: average distance from vertices 'coords'
    to the boundary 'boundary'.

    Args:
        coords: Coordinate sequence (e.g., polygon.exterior.coords).
        boundary: Shapely LineString representing another polygon boundary.

    Returns:
        float: One-sided PoLiS term.
    """
    s = 0.0
    # Skip last point because it is equal to the first for closed polygons
    for pt in (geometry.Point(c) for c in coords[:-1]):
        s += boundary.distance(pt)
    return s / float(2 * len(coords))


def compare_polys(poly_a, poly_b):
    """
    Symmetric PoLiS distance between two polygons.

    Args:
        poly_a (Polygon): Shapely polygon.
        poly_b (Polygon): Shapely polygon.

    Returns:
        float: PoLiS distance.
    """
    bndry_a, bndry_b = poly_a.exterior, poly_b.exterior
    d = polis_one_side(bndry_a.coords, bndry_b)
    d += polis_one_side(bndry_b.coords, bndry_a)
    return d


class PolisEval:
    """
    PoLiS evaluation on COCO-style GT and prediction files.

    Matching strategy:
        - For each GT instance, find the prediction with the highest bbox IoU.
        - If IoU >= iou_thresh, compute PoLiS between this pair.
        - Unmatched GTs and predictions are ignored (no penalty).
    """

    def __init__(self, cocoGt, cocoDt, iou_thresh=0.5):
        """
        Args:
            cocoGt (COCO): COCO object for ground truth.
            cocoDt (COCO): COCO object for detections (loadRes result).
            iou_thresh (float): IoU threshold for GT–DT matching.
        """
        self.cocoGt = cocoGt
        self.cocoDt = cocoDt
        self.iou_thresh = iou_thresh

        self.evalImgs = defaultdict(list)
        self.eval = {}
        self._gts = defaultdict(list)
        self._dts = defaultdict(list)
        self.stats = []

        self.imgIds = list(sorted(self.cocoGt.imgs.keys()))

    def _prepare(self):
        """Group GT and DT annotations by image_id."""
        gts = self.cocoGt.loadAnns(self.cocoGt.getAnnIds(imgIds=self.imgIds))
        dts = self.cocoDt.loadAnns(self.cocoDt.getAnnIds(imgIds=self.imgIds))

        self._gts = defaultdict(list)
        self._dts = defaultdict(list)

        for gt in gts:
            self._gts[gt["image_id"]].append(gt)
        for dt in dts:
            self._dts[dt["image_id"]].append(dt)

        self.evalImgs = defaultdict(list)
        self.eval = {}

    def evaluateImg(self, imgId):
        """
        Compute average PoLiS for a single image over matched GT–DT pairs.

        Args:
            imgId (int): Image id.

        Returns:
            float: Average PoLiS for this image (0.0 if no valid match).
        """
        gts = self._gts[imgId]
        dts = self._dts[imgId]

        if len(gts) == 0 or len(dts) == 0:
            return 0.0

        # Extract polygon and corresponding bbox for each GT/DT
        gt_polygons = [
            np.array(gt["segmentation"][0]).reshape(-1, 2) for gt in gts
        ]
        dt_polygons = [
            np.array(dt["segmentation"][0]).reshape(-1, 2) for dt in dts
        ]

        gt_bboxs = [bounding_box(p) for p in gt_polygons]
        dt_bboxs = [bounding_box(p) for p in dt_polygons]

        # bbox IoU matrix: rows = dt, cols = gt
        iscrowd = [0] * len(gt_bboxs)
        ious = maskUtils.iou(dt_bboxs, gt_bboxs, iscrowd)

        img_polis_sum = 0.0
        num_matched = 0

        # For each GT, pick the DT with maximum IoU
        for j, gt_poly in enumerate(gt_polygons):
            matched_idx = np.argmax(ious[:, j])
            iou = ious[matched_idx, j]

            if iou >= self.iou_thresh:
                polis_val = compare_polys(
                    Polygon(gt_poly), Polygon(dt_polygons[matched_idx])
                )
                img_polis_sum += polis_val
                num_matched += 1

        if num_matched == 0:
            return 0.0
        return img_polis_sum / float(num_matched)

    def evaluate(self, verbose=True):
        """
        Compute dataset-level average PoLiS.

        Returns:
            float: Average PoLiS over all images with at least one matched pair.
        """
        self._prepare()
        polis_tot = 0.0
        num_valid_imgs = 0

        for imgId in tqdm(self.imgIds, desc="Evaluating PoLiS"):
            img_polis_avg = self.evaluateImg(imgId)
            if img_polis_avg == 0.0:
                continue
            polis_tot += img_polis_avg
            num_valid_imgs += 1

        polis_avg = polis_tot / float(num_valid_imgs) if num_valid_imgs > 0 else 0.0

        if verbose:
            print("Average PoLiS: %.2f" % polis_avg)

        return polis_avg


# ----------------------------------------------------------------------
# 2. SCR (Smoothness Consistency Ratio)
# ----------------------------------------------------------------------
def analyze_polygon_smoothness(polygon, angle_threshold_rad):
    """
    Count smooth vs distorted vertices for a polygon.

    A vertex is considered:
        - smooth    if |angle_diff| <= threshold
        - distorted if |angle_diff|  > threshold

    Args:
        polygon (ndarray): (N, 2) array of (x, y) vertices.
        angle_threshold_rad (float): Threshold in radians.

    Returns:
        (int, int): (smooth_points, distorted_points)
    """
    smooth_points = 0
    distorted_points = 0
    num_points = len(polygon)

    for i in range(num_points):
        prev_point = polygon[i - 1]
        current_point = polygon[i]
        next_point = polygon[(i + 1) % num_points]

        vec1 = prev_point - current_point
        vec2 = next_point - current_point

        angle_diff = np.arctan2(vec2[1], vec2[0]) - np.arctan2(vec1[1], vec1[0])
        # Normalize to [-π, π]
        angle_diff = np.arctan2(np.sin(angle_diff), np.cos(angle_diff))

        if abs(angle_diff) <= angle_threshold_rad:
            smooth_points += 1
        else:
            distorted_points += 1

    return smooth_points, distorted_points


def compute_scr(input_json, gti_annotations, output_dir, angle_threshold_deg=15):
    """
    Compute Smoothness Consistency Ratio (SCR).

    Definition:
        For each image i:
            SCR_i = (# distorted vertices in prediction) / (# distorted vertices in GT)
        Final metric:
            Mean SCR = average of SCR_i over all valid images.

    Args:
        input_json (str): Path to prediction JSON (COCO detections format).
        gti_annotations (str): Path to GT JSON (COCO annotations format).
        output_dir (str): Directory to save per-image results (optional, can be unused).
        angle_threshold_deg (float): Threshold (in degrees) for defining "distorted".

    Returns:
        float: Mean SCR over all images with GT distorted points > 0.
    """
    angle_threshold_rad = np.radians(angle_threshold_deg)

    coco_gti = COCO(gti_annotations)
    with open(input_json, "r") as f:
        submission_file = json.load(f)
    coco_pred = coco_gti.loadRes(submission_file)

    list_scr = []
    scr_results = []

    img_ids = coco_gti.getImgIds()
    bar = tqdm(img_ids, desc="Evaluating SCR")

    for image_id in bar:
        img = coco_gti.loadImgs(image_id)[0]
        image_name = os.path.basename(img["file_name"])

        gt_annotations = coco_gti.loadAnns(coco_gti.getAnnIds(imgIds=image_id))
        pred_annotations = coco_pred.loadAnns(coco_pred.getAnnIds(imgIds=image_id))

        # Skip images without GT
        if not gt_annotations:
            continue

        # Count distorted vertices in GT
        gt_distorted_points = 0
        for ann in gt_annotations:
            if "segmentation" in ann and len(ann["segmentation"]) > 0:
                for poly in ann["segmentation"]:
                    poly = np.array(poly).reshape(-1, 2)
                    _, gt_distorted = analyze_polygon_smoothness(
                        poly, angle_threshold_rad
                    )
                    gt_distorted_points += gt_distorted

        # Count distorted vertices in predictions
        pred_distorted_points = 0
        for ann in pred_annotations:
            if "segmentation" in ann and len(ann["segmentation"]) > 0:
                for poly in ann["segmentation"]:
                    poly = np.array(poly).reshape(-1, 2)
                    _, pred_distorted = analyze_polygon_smoothness(
                        poly, angle_threshold_rad
                    )
                    pred_distorted_points += pred_distorted

        if gt_distorted_points > 0:
            scr = pred_distorted_points / float(gt_distorted_points)
        else:
            scr = 0.0

        list_scr.append(scr)
        bar.set_description("Mean SCR: %.6f" % (np.mean(list_scr)))
        bar.refresh()

        scr_results.append(
            {
                "image_id": image_id,
                "file_name": image_name,
                "SCR": scr,
                "pred_distorted_points": int(pred_distorted_points),
                "gt_distorted_points": int(gt_distorted_points),
            }
        )

    # Optionally save per-image results
    # os.makedirs(output_dir, exist_ok=True)
    # out_path = os.path.join(output_dir, "scr_metrics.json")
    # with open(out_path, "w") as f:
    #     json.dump(scr_results, f, indent=4)

    mean_scr = float(np.mean(list_scr)) if len(list_scr) > 0 else 0.0
    print("Average SCR: %.2f" % mean_scr)

    return mean_scr

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute polygon regularity metrics: PoLiS and SCR."
    )
    parser.add_argument("--gt-file", required=True, help="COCO-format ground truth JSON.")
    parser.add_argument("--dt-file", required=True, help="COCO-format prediction JSON.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to save per-image metrics JSON.",
    )
    parser.add_argument(
        "--angle-threshold",
        type=float,
        default=30.0,
        help="Angle threshold in degrees for defining distorted vertices in SCR.",
    )
    args = parser.parse_args()

    # --- PoLiS ---
    coco_gt = COCO(args.gt_file)
    coco_dt = coco_gt.loadRes(args.dt_file)
    polis_eval = PolisEval(coco_gt, coco_dt)
    avg_polis = polis_eval.evaluate()

    # --- SCR ---
    avg_scr = compute_scr(
        input_json=args.dt_file,
        gti_annotations=args.gt_file,
        output_dir=args.output_dir,
        angle_threshold_deg=args.angle_threshold,
    )

    print("\n=== Regularity metrics summary ===")
    print(f"Average PoLiS: {avg_polis:.6f}")
    print(f"Average SCR  : {avg_scr:.6f}")