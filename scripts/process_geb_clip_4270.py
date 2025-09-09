import os
import cv2
import numpy as np
import argparse
from skimage.measure import label, regionprops

def douglas_peucker_opencv(contour, epsilon=1.0):
    contour = contour.astype(np.int32)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx[:, 0, :] if approx is not None else None

def gaussian_2d(x, y, x0, y0, sigma):
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

def generate_heatmap(vertex_locations, heatmap_shape, sigma=2.5):
    heatmap = np.zeros(heatmap_shape, dtype=np.float32)
    if len(vertex_locations) == 0:
        return heatmap
    x = np.arange(heatmap_shape[1])
    y = np.arange(heatmap_shape[0])
    x, y = np.meshgrid(x, y)
    for loc in vertex_locations:
        heatmap = np.maximum(heatmap, gaussian_2d(x, y, loc[0], loc[1], sigma))
    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value
    heatmap[heatmap < 1e-8] = 0
    return heatmap

def visualize_polygons_on_mask(mask, ext_polygons, inn_polygons, radius=2):
    vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for poly in ext_polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 255, 0), thickness=1)
        for (x, y) in pts:
            cv2.circle(vis, (x, y), radius, (255, 255, 0), -1)
    for poly in inn_polygons:
        pts = np.array(poly, dtype=np.int32).reshape(-1, 2)
        cv2.polylines(vis, [pts], isClosed=True, color=(0, 0, 255), thickness=1)
        for (x, y) in pts:
            cv2.circle(vis, (x, y), radius, (0, 255, 255), -1)
    return vis

def process_mask(mask_path, vis_path, heatmap_path, epsilon=1.0, sigma=2.5):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask: {mask_path}")
        return

    H, W = mask.shape
    labeled = label(mask > 127)
    props = regionprops(labeled)

    ext_polygons = []
    inn_polygons = []
    vertex_locations = []

    # extract polygons from mask using Douglas-Peucker algorithm
    for prop in props:
        prop_mask = np.zeros_like(mask, dtype=np.uint8)
        prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

        contours, hierarchy = cv2.findContours(prop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        hierarchy = hierarchy[0] if hierarchy is not None else []

        for contour, h in zip(contours, hierarchy):
            if cv2.contourArea(contour) < 10:
                continue
            approx = douglas_peucker_opencv(contour, epsilon)
            if approx is not None and len(approx) >= 3:
                flat = approx.reshape(-1, 2)
                vertex_locations.extend(flat)
                if h[3] == -1:
                    ext_polygons.append(flat)
                else:
                    inn_polygons.append(flat)

    # visualize polygons on mask
    vis = visualize_polygons_on_mask(mask, ext_polygons, inn_polygons)
    cv2.imwrite(vis_path, vis)

    # generate vertex heatmap
    heatmap = generate_heatmap(vertex_locations, (H, W), sigma=sigma)
    np.save(heatmap_path, heatmap)
    print(f"Saved visualization to {vis_path} and heatmap to {heatmap_path}")

def main():
    parser = argparse.ArgumentParser(description='Extract polygons using DP and visualize + save vertex heatmaps')
    parser.add_argument('-i', '--input_dir', required=True, help='Input mask directory')
    parser.add_argument('-op', '--output_polygon_dir', required=True, help='Output visualization directory')
    parser.add_argument('-oh', '--output_heatmap_dir', required=True, help='Output visualization directory')
    parser.add_argument('-e', '--epsilon', type=float, default=1.0, help='DP algorithm epsilon (default: 1.0)')
    parser.add_argument('-s', '--sigma', type=float, default=2.5, help='Gaussian sigma for vertex heatmap')
    args = parser.parse_args()

    os.makedirs(args.output_polygon_dir, exist_ok=True)
    os.makedirs(args.output_heatmap_dir, exist_ok=True)

    for fname in sorted(os.listdir(args.input_dir)):
        if not fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
            continue
        base = os.path.splitext(fname)[0]
        in_path = os.path.join(args.input_dir, fname)
        vis_path = os.path.join(args.output_polygon_dir, f'{base}.png')
        heatmap_path = os.path.join(args.output_heatmap_dir, f'{base}.npy')
        process_mask(in_path, vis_path, heatmap_path, epsilon=args.epsilon, sigma=args.sigma)

if __name__ == '__main__':
    main()

    """python scripts/process_geb_clip_4270.py \
    -i ./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_input \
    -op ./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_input_dp1 \
    -oh ./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_input_sigma2.5 \
    -e 1.0 \
    -s 2.5
    """

