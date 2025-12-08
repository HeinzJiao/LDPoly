"""
This script evaluates predicted polygons using a set of metrics:

    - pixel-level Coverage / polygon Simplicity / vertex Efficiency (CSE):
        * IoU
        * Boundary IoU (B-IoU)
        * Complexity-aware IoU (C-IoU)
        * N-ratio (vertex efficiency)
        * Simplicity-aware IoU (S-IoU)  [proposed in our LDPoly work]

    - Polygon regularity:
        * PoLiS
        * Smoothness Consistency Ratio (SCR)  [also introduced in LDPoly]

Usage:
    python evaluation.py --gt-file <path_to_ground_truth_annotation> \
                         --dt-file <path_to_prediction_file> \
                         --output <output_folder> \
                         --eval-type <evaluation_family>

Arguments:
    --gt-file : Path to the COCO-format ground truth annotation file.
    --dt-file : Path to the COCO-format detection result (predictions) file.
    --output  : Directory to save per-image evaluation results (used by CSE- and
                SCR-related outputs).
    --eval-type : Evaluation family to run. Choose from:
                  - "cse"        : Coverage + simplicity + efficiency
                                   (IoU, B-IoU, C-IoU, S-IoU, N-ratio).
                  - "r"          : Polygon regularity (PoLiS, SCR)

Example:
    python evaluation.py --gt-file ./data/deventer_road/annotations/test.json   \
        --dt-file ./outputs/deventer_road_reproduction/epoch=824-step=739199/polygons_seg_ddim_vertices_from_heat_th-0.1_k-3_dp_eps2.json \
        --eval-type cse
"""
import argparse
import sys
sys.path.append("..")
from pycocotools.coco import COCO
from metrics.coverage_simplicity_efficiency import compute_cse
from metrics.regularity import compute_scr, PolisEval


def polis_eval(annFile, resFile):
    gt_coco = COCO(annFile)
    dt_coco = gt_coco.loadRes(resFile)
    polisEval = PolisEval(gt_coco, dt_coco)
    polisEval.evaluate()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt-file", default="")
    parser.add_argument("--dt-file", default="")
    parser.add_argument("--output", default="./output", metavar="DIR", help="Directory to save evaluation results")
    parser.add_argument(
        "--eval-type",
        default="cse",
        choices=["cse", "r"],
        help=(
            'Evaluation type: '
            '"cse" (coverage + simplicity + efficiency: IoU, B-IoU, C-IoU, S-IoU, N-ratio), '
            '"r" (polygon regularity: PoLiS, SCR)'
        ),
    )
    args = parser.parse_args()

    eval_type = args.eval_type
    gt_file = args.gt_file
    dt_file = args.dt_file
    if eval_type == 'cse':
        compute_cse(dt_file, gt_file, args.output)
    elif eval_type == 'r':
        polis_eval(gt_file, dt_file)
        compute_scr(dt_file, gt_file, args.output, angle_threshold_deg=30)
    else:
        raise RuntimeError(
            'please choose a correct type from '
            '["cse", "r"]'
        )
