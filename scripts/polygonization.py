"""
Polygonization of instance masks using predicted vertices (junctions).

This script takes:
    1) A COCO-style annotation file for the test set;
    2) A JSON file with per-image predicted vertices (from the vertex heatmap);
    3) Per-image segmentation probability maps (saved as .npy files);

and produces:
    - A COCO-style JSON file with polygonized instance predictions
      (segmentation polygons + bounding boxes + scores).
    - Optionally, per-instance visualization of junctions, contours, and polygons.

Typical pipeline:
    1) Run the diffusion model to generate segmentation logits and vertex heatmaps;
    2) Run `extract_vertices_from_heatmap.py` to obtain vertices JSON;
    3) Run this script to perform polygonization and save polygon predictions.

Example:
    PYTHONPATH=./:$PYTHONPATH python -u scripts/polygonization.py \
        --annotation_path ./data/deventer_road/annotations/test.json \
        --outputs_dir ./outputs/deventer_road_reproduction/epoch=824-step=739199 \
        --sampler ddim \
        --output_vertices_file "output_vertices_from_heatmap_x4_ddim_th-0.1_k-3.json" \
        --samples_seg_logits_file "samples_seg_ddim_logits_npy" \
        --save_file "polygons_seg_ddim_vertices_from_heat_th-0.1_k-3_dp_eps2.json" \
        --d_th 5 \
        --polygonization_vis_path ./outputs/deventer_road_reproduction/epoch=824-step=739199/polygonization_vis
"""

import os
import json
import argparse

import cv2
import numpy as np
from scipy.spatial.distance import cdist
from skimage.measure import label, regionprops
# -------------------------------------------------------------------------
#  Low-level helpers: raster ↔ polygon conversion and contour cleanup
# -------------------------------------------------------------------------
def ext_c_to_poly_coco(ext_c, im_h, im_w):
    """
    Convert an outer contour to a pixel-aligned polygon (COCO style), with a slight dilation to avoid thin gaps.

    Args:
        ext_c (ndarray): Outer contour points of shape (N, 1, 2).
        im_h (int): Image height.
        im_w (int): Image width.

    Returns:
        ndarray: Closed polygon of shape (M, 2), with the first and last
                 vertex duplicated.
    """
    mask = np.zeros([im_h + 1, im_w + 1], dtype=np.uint8)
    polygon = np.int0(ext_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)

    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)

    # Slightly dilate the contour by one pixel to avoid thin gaps
    trans_prop_mask[f_y + 1, f_x] = 1
    trans_prop_mask[f_y, f_x + 1] = 1
    trans_prop_mask[f_y + 1, f_x + 1] = 1

    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    new_poly = diagonal_to_square(poly)
    return new_poly


def diagonal_to_square(poly):
    """
    Convert diagonal pixel steps into axis-aligned steps to obtain
    4-connected (Manhattan) polygons on the pixel grid.
    """
    new_c = []
    for idx, p in enumerate(poly[:-1]):
        q = poly[idx + 1]
        # Axis-aligned neighbours
        if ((p[0] + 1 == q[0] and p[1] == q[1]) or
                (p[0] == q[0] and p[1] + 1 == q[1]) or
                (p[0] - 1 == q[0] and p[1] == q[1]) or
                (p[0] == q[0] and p[1] - 1 == q[1])):
            new_c.append(p)
        # Diagonal steps: insert an intermediate pixel to keep 4-connectivity
        elif (p[0] + 1 == q[0] and p[1] + 1 == q[1]):
            new_c.append(p)
            new_c.append([p[0] + 1, p[1]])
        elif (p[0] - 1 == q[0] and p[1] - 1 == q[1]):
            new_c.append(p)
            new_c.append([p[0] - 1, p[1]])
        elif (p[0] + 1 == q[0] and p[1] - 1 == q[1]):
            new_c.append(p)
            new_c.append([p[0], p[1] - 1])
        else:
            new_c.append(p)
            new_c.append([p[0], p[1] + 1])

    new_poly = np.asarray(new_c)
    new_poly = np.concatenate((new_poly, new_poly[0].reshape(-1, 2)))
    return new_poly


def inn_c_to_poly_coco(inn_c, im_h, im_w):
    """
    Convert an inner contour (hole) to a pixel-aligned polygon, with a slight
    inward shrink to avoid touching the outer boundary.

    Args:
        inn_c (ndarray): Inner contour points of shape (N, 1, 2).
        im_h (int): Image height.
        im_w (int): Image width.

    Returns:
        ndarray: Closed polygon representing the hole.
    """
    mask = np.zeros([im_h + 1, im_w + 1], dtype=np.uint8)
    polygon = np.int0(inn_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)

    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)

    # Shrink inward: remove top-most and left-most boundary pixels
    trans_prop_mask[f_y[np.where(f_y == min(f_y))], f_x[np.where(f_y == min(f_y))]] = 0
    trans_prop_mask[f_y[np.where(f_x == min(f_x))], f_x[np.where(f_x == min(f_x))]] = 0

    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)[::-1]
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    new_poly = diagonal_to_square(poly)
    return new_poly


def simple_polygon(poly, thres=10):
    """
    Merge nearly collinear edges by removing small angle changes.

    Args:
        poly (ndarray): Closed polygon of shape (N, 2).
        thres (float): Angle threshold in degrees.

    Returns:
        ndarray: Simplified closed polygon.
    """
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]

    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    vec0 = lines[:, 2:] - lines[:, :2]
    vec1 = np.roll(vec0, -1, axis=0)

    vec0_ang = np.arctan2(vec0[:, 1], vec0[:, 0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:, 1], vec1[:, 0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)

    flag1 = np.roll((lines_ang > thres), 1, axis=0)
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)
    simple_poly = poly[np.bitwise_and(flag1, flag2)]
    simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1, 2)))
    return simple_poly


# -------------------------------------------------------------------------
#  Core polygonization: contour + junctions → polygon
# -------------------------------------------------------------------------
def douglas_peucker_opencv(points, epsilon):
    """
    Simplify a polygonal chain using OpenCV's Douglas–Peucker implementation.

    Args:
        points (ndarray): Input points of shape (N, 2).
        epsilon (float): Simplification tolerance.

    Returns:
        ndarray: Simplified points of shape (M, 2).
    """
    contour = points.reshape((-1, 1, 2)).astype(np.float32)
    simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
    return simplified_contour.reshape((-1, 2))


def douglas_peucker_with_indices(c, epsilon):
    """
    Simplify a contour and also return the indices of the kept points
    in the original contour.

    Args:
        c (ndarray): Original contour points of shape (N, 2).
        epsilon (float): Simplification tolerance.

    Returns:
        simplified_c (ndarray): Simplified contour points.
        simplified_indices (list[int]): Indices in `c` corresponding to
            `simplified_c`.
    """
    simplified_c = cv2.approxPolyDP(c.astype(np.float32), epsilon, closed=False).reshape(-1, 2)

    simplified_indices = []
    for point in simplified_c:
        distances = np.linalg.norm(c - point, axis=1)
        nearest_idx = np.argmin(distances)
        simplified_indices.append(nearest_idx)

    return simplified_c, simplified_indices


def is_critical_point(prev_point, current_point, next_point, angle_threshold=30):
    """
    Determine whether a point on a polyline is a "critical" corner point
    based on its local turning angle.

    Args:
        prev_point (tuple): Previous point (x, y).
        current_point (tuple): Current point (x, y).
        next_point (tuple): Next point (x, y).
        angle_threshold (float): Threshold around 90 degrees (in degrees).

    Returns:
        bool: True if the point is considered critical (e.g., near right angle).
    """
    vec1 = np.array(prev_point) - np.array(current_point)
    vec2 = np.array(next_point) - np.array(current_point)

    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    angle_deg = np.degrees(angle)

    return 90 - angle_threshold < angle_deg < 90 + angle_threshold


def contour_to_polygon_with_junctions(
    c,
    junctions,
    distance_threshold=5,
    dp_eps=2,
    angle_threshold=30,
):
    """
    Convert a dense instance contour into a polygon by combining:
        (i) points supported by predicted junctions, and
        (ii) critical points from Douglas–Peucker simplification.

    High-level steps:
        1) If there are no junctions, fall back to Douglas–Peucker only.
        2) Compute distances between dense contour points and all junctions,
           keep contour points within `distance_threshold` of some junction.
        3) Simplify the contour via Douglas–Peucker to obtain a coarse polyline.
        4) On the simplified polyline, detect "critical" corner points
           (near right angles) and map them back to original contour indices.
        5) Merge junction-supported points and critical points, remove duplicates,
           and preserve the original ordering along the contour.
        6) If the resulting set is too small, fall back again to Douglas–Peucker.

    Args:
        c (ndarray): Dense contour points of shape (N, 2).
        junctions (ndarray): Predicted vertex locations of shape (M, 2).
        distance_threshold (float): Max distance from junctions to keep a contour point.
        dp_eps (float): Douglas–Peucker epsilon.
        angle_threshold (float): Angle threshold around 90° for detecting corners.

    Returns:
        ndarray or None: Polygon vertices of shape (K, 2), or None if
        no valid polygon can be formed.
    """
    # Case 0: no junctions → use pure Douglas–Peucker
    if len(junctions) == 0:
        simplified_poly = douglas_peucker_opencv(c, epsilon=dp_eps)
        if len(simplified_poly) > 2:
            return simplified_poly
        return None

    # 1) Compute distances between dense contour points and junctions
    distances = cdist(c, junctions)  # (N, M)
    nearest_indices = np.argmin(distances, axis=0)   # index in c for each junction
    nearest_distances = np.min(distances, axis=0)    # distance for each junction

    # 2) Keep contour points that are close to at least one junction
    valid_indices = nearest_indices[nearest_distances < distance_threshold]

    # 3) Simplify contour for corner detection
    simplified_c, simplified_indices = douglas_peucker_with_indices(c, epsilon=dp_eps)

    # 4) Detect critical (corner) points on the simplified contour
    critical_indices = []
    for i in range(1, len(simplified_c) - 1):
        if is_critical_point(
            simplified_c[i - 1], simplified_c[i], simplified_c[i + 1], angle_threshold
        ):
            critical_indices.append(simplified_indices[i])

    # 5) Merge junction-supported indices and critical indices
    final_indices = np.unique(np.concatenate((valid_indices, critical_indices))).astype(np.int64)

    # 6) Fallback: if nothing remains, use Douglas–Peucker only
    if len(final_indices) == 0:
        simplified_poly = douglas_peucker_opencv(c, epsilon=dp_eps)
        if len(simplified_poly) > 2:
            return simplified_poly
        return None

    poly = c[np.sort(final_indices)]
    if len(poly) > 2:
        return poly
    return None


def get_poly(
    prop,
    mask_pred,
    junctions,
    d_th,
    vis_save_path=None,
    file_name=None,
    region_idx=None,
):
    """
    Polygonize a single connected component (instance) given:
        - its binary mask region (prop),
        - the global segmentation probability map (mask_pred),
        - and all predicted junctions in the image.

    High-level steps per instance:
        1) Build a binary mask for this instance and compute its mean score
           from the global probability map.
        2) Extract outer and inner contours with OpenCV.
        3) Convert each contour to a pixel-aligned polygon (COCO style):
           outer → ext_c_to_poly_coco, inner → inn_c_to_poly_coco.
        4) For each contour, run junction-aware polygonization
           (contour_to_polygon_with_junctions) to select a sparse set of
           vertices aligned with the predicted junctions.
        5) Optionally visualize:
           - junctions,
           - raw contours,
           - final polygons.

    Args:
        prop (RegionProperties): One connected component from `regionprops`.
        mask_pred (ndarray): Full-resolution probability map (H, W).
        junctions (ndarray): Predicted vertices (M, 2) in image coordinates.
        d_th (float): Distance threshold passed to the junction-aware polygonization.
        vis_save_path (str or None): Directory to save visualization images.
        file_name (str or None): Original heatmap / image file name (for naming).
        region_idx (int or None): Index of the region within the image.

    Returns:
        poly (list[list[float]]): List of polygons, each as flattened [x1,y1,...,xN,yN].
        score (float): Mean probability score of this instance.
    """
    # ---------------------------------------------------------------------
    # 1) Build instance mask and mean score
    # ---------------------------------------------------------------------
    prop_mask = np.zeros_like(mask_pred, dtype=np.uint8)
    prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

    masked_instance = np.ma.masked_array(mask_pred, mask=(prop_mask != 1))
    score = masked_instance.mean()

    im_h, im_w = mask_pred.shape

    # ---------------------------------------------------------------------
    # 2) Extract outer and inner contours for this instance
    # ---------------------------------------------------------------------
    contours, hierarchy = cv2.findContours(prop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    poly = []
    contours_to_poly_coco = []

    # ---------------------------------------------------------------------
    # 3) For each contour: outer vs. inner, convert to pixel-aligned polygon
    # ---------------------------------------------------------------------
    for contour, h in zip(contours, hierarchy[0]):
        # h[3] == -1 → outer contour, otherwise inner contour (hole)
        if h[3] == -1:
            c = ext_c_to_poly_coco(contour, im_h, im_w)
        else:
            c = inn_c_to_poly_coco(contour, im_h, im_w)

        contours_to_poly_coco.append(c[:, None, :])

        # -----------------------------------------------------------------
        # 4) Junction-aware polygonization on this contour
        # -----------------------------------------------------------------
        init_poly = contour_to_polygon_with_junctions(
            c,
            junctions,
            distance_threshold=d_th,
            dp_eps=2,
            angle_threshold=30,
        )
        # Alternative: pure Douglas–Peucker
        # init_poly = douglas_peucker_opencv(c, epsilon=1)

        if init_poly is not None and len(init_poly) > 2:
            poly.append(init_poly.flatten().astype(np.float32).tolist())

    # ---------------------------------------------------------------------
    # 5) Optional visualization of junctions / contours / polygons
    # ---------------------------------------------------------------------
    if vis_save_path is not None and file_name is not None and region_idx is not None:
        os.makedirs(vis_save_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_name))[0]

        scale = 2  # for visualization only
        vis_h, vis_w = im_h * scale, im_w * scale

        def clip_point(x, y):
            x = int(np.clip(x, 0, vis_w - 1))
            y = int(np.clip(y, 0, vis_h - 1))
            return x, y

        canvas_junc = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255
        canvas_contour = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255
        canvas_composite = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255
        canvas_poly = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255

        radius = 3
        line_thickness = 1

        # Draw all junctions (green)
        for junc in junctions:
            x, y = clip_point(junc[0] * scale, junc[1] * scale)
            cv2.circle(canvas_junc, (x, y), radius=radius, color=(0, 255, 0), thickness=-1)
            cv2.circle(canvas_composite, (x, y), radius=radius, color=(0, 255, 0), thickness=-1)

        # Draw all contours (gray)
        for contour in contours_to_poly_coco:
            contour_scaled = contour * scale
            contour_scaled[:, :, 0] = np.clip(contour_scaled[:, :, 0], 0, vis_w - 1)
            contour_scaled[:, :, 1] = np.clip(contour_scaled[:, :, 1], 0, vis_h - 1)
            contour_int = contour_scaled.astype(np.int32)
            cv2.polylines(
                canvas_contour,
                [contour_int],
                isClosed=True,
                color=(128, 128, 128),
                thickness=line_thickness,
            )
            cv2.polylines(
                canvas_composite,
                [contour_int],
                isClosed=True,
                color=(128, 128, 128),
                thickness=line_thickness,
            )

        # Draw final polygons: cyan edges + magenta vertices
        for polygon_flat in poly:
            pts = np.array(polygon_flat, dtype=np.float32).reshape(-1, 2) * scale
            pts[:, 0] = np.clip(pts[:, 0], 0, vis_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, vis_h - 1)
            pts_int = pts.astype(np.int32)

            for x, y in pts_int:
                cv2.circle(canvas_composite, (x, y), radius=radius, color=(255, 0, 255), thickness=-1)

            cv2.polylines(
                canvas_poly,
                [pts_int],
                isClosed=True,
                color=(255, 255, 0),
                thickness=line_thickness,
            )
            for x, y in pts_int:
                cv2.circle(canvas_poly, (x, y), radius=radius, color=(255, 0, 255), thickness=-1)

        cv2.imwrite(os.path.join(vis_save_path, f"{base_name}_region_{region_idx}_junctions.png"), canvas_junc)
        cv2.imwrite(os.path.join(vis_save_path, f"{base_name}_region_{region_idx}_contours.png"), canvas_contour)
        cv2.imwrite(os.path.join(vis_save_path, f"{base_name}_region_{region_idx}_composite.png"), canvas_composite)
        cv2.imwrite(os.path.join(vis_save_path, f"{base_name}_region_{region_idx}_polygon.png"), canvas_poly)

    return poly, score


# -------------------------------------------------------------------------
#  Main: load vertices + logits, run polygonization over all images
# -------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Polygonization of instance masks using predicted vertices.")

    parser.add_argument(
        "--annotation_path",
        type=str,
        required=True,
        help="Path to COCO format annotation JSON (ground truth).",
    )
    parser.add_argument(
        "--outputs_dir",
        type=str,
        required=True,
        help="Directory where model outputs and polygon JSON will be saved.",
    )
    parser.add_argument(
        "--output_vertices_file",
        type=str,
        required=True,
        help="File name of the JSON containing extracted vertices (relative to outputs_dir).",
    )
    parser.add_argument(
        "--samples_seg_logits_file",
        type=str,
        required=True,
        help="Subdirectory name under outputs_dir containing segmentation logits (.npy).",
    )
    parser.add_argument(
        "--save_file",
        type=str,
        required=True,
        help="File name of the polygon prediction JSON to save under outputs_dir.",
    )
    parser.add_argument(
        "--polygonization_vis_path",
        type=str,
        required=False,
        default=None,
        help="Optional directory to save per-instance visualization images.",
    )
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["direct", "ddim", "ddpm"],
        required=True,
        help="Sampling method used during testing (for bookkeeping only).",
    )
    parser.add_argument(
        "--d_th",
        type=float,
        required=True,
        help="Distance threshold for snapping contours to junctions.",
    )

    args = parser.parse_args()

    annotation_path = args.annotation_path
    outputs_dir = args.outputs_dir

    # ---------------------------------------------------------------------
    # Load COCO annotations and build lookup maps
    # ---------------------------------------------------------------------
    with open(annotation_path, "r") as f:
        coco_annotations = json.load(f)

    file_name_to_image_id = {img["file_name"]: img["id"] for img in coco_annotations["images"]}
    image_id_to_category_id = {ann["image_id"]: ann["category_id"] for ann in coco_annotations["annotations"]}

    # Paths for vertices JSON and segmentation logits
    output_json_file = os.path.join(outputs_dir, args.output_vertices_file)
    logits_dir = os.path.join(outputs_dir, args.samples_seg_logits_file)

    # Load predicted vertices (per image)
    with open(output_json_file, "r") as f:
        results = json.load(f)

    poly_predictions = []

    for vertices_per_img in results:
        # -------------------------------------------------------------
        # Per-image processing
        # -------------------------------------------------------------
        file_name = vertices_per_img["image_file_name"]
        print("[Polygonization] Processing:", file_name)

        junctions = np.array(vertices_per_img["extracted_vertices"])

        # Map heatmap file name (.npy) to original image file name (.png/.jpg)
        file_name_png = file_name.replace(".npy", ".png")
        image_id = file_name_to_image_id.get(file_name_png)

        if image_id is None:
            file_name_jpg = file_name.replace(".npy", ".jpg")
            image_id = file_name_to_image_id.get(file_name_jpg)

        if image_id is None:
            print(f"[Warning] Image ID not found for file: {file_name}")
            continue

        category_id = image_id_to_category_id.get(image_id, 100)  # default if not found

        # Load predicted segmentation logits for this image
        logit_name = file_name.split(".")[0] + ".npy"
        logit_path = os.path.join(logits_dir, logit_name)
        logit = np.load(logit_path)
        mask = logit > 0.5

        # Label connected components
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)

        for i, prop in enumerate(props):
            poly_list, score = get_poly(
                prop,
                logit,
                junctions,
                args.d_th,
                vis_save_path=args.polygonization_vis_path,
                file_name=file_name,
                region_idx=i,
            )
            if len(poly_list) == 0:
                continue

            # Use the first polygon to derive the bounding box
            coords = np.array(poly_list[0]).reshape(-1, 2)
            x_min, y_min = np.min(coords, axis=0)
            x_max, y_max = np.max(coords, axis=0)
            bbox = [float(x_min), float(y_min), float(x_max - x_min), float(y_max - y_min)]

            poly_predictions.append(
                {
                    "image_id": image_id,
                    "category_id": category_id,
                    "segmentation": poly_list,
                    "bbox": bbox,
                    "score": float(score),
                }
            )

    # ---------------------------------------------------------------------
    # Save polygon predictions in COCO-style JSON
    # ---------------------------------------------------------------------
    poly_predictions_path = os.path.join(outputs_dir, args.save_file)
    with open(poly_predictions_path, "w") as f:
        json.dump(poly_predictions, f)
    print(f"[Done] Polygon predictions saved to: {poly_predictions_path}")


if __name__ == "__main__":
    main()
