import os
import numpy as np
import re
import cv2
import json

def stitch_patches(input_folder, output_folder, patch_size=(256, 256), type='npy'):
    """
    Stitch patch files back into full maps based on row and column indices.

    Args:
        input_folder (str): Path to the folder containing patch files.
        output_folder (str): Path to save the stitched maps.
        patch_size (tuple): Size of each patch (width, height).
        type (str): 'npy' or 'png'.
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Regular expression to identify patch files
    if type == 'npy':
        pattern = re.compile(r"(.+)_patch_(\d+)_(\d+)\.npy")
    elif type == 'png':
        pattern = re.compile(r"(.+)_patch_(\d+)_(\d+)\.png")
    else:
        raise NotImplementedError

    # Group patches by their base name
    patch_groups = {}
    for filename in os.listdir(input_folder):
        match = pattern.match(filename)
        if match:
            base_name = match.group(1)
            row = int(match.group(2))
            col = int(match.group(3))
            if base_name not in patch_groups:
                patch_groups[base_name] = {}
            patch_groups[base_name][(row, col)] = filename

    for base_name, patches in patch_groups.items():
        # Determine max rows and cols
        max_row = max(row for row, _ in patches.keys())
        max_col = max(col for _, col in patches.keys())

        full_height = (max_row + 1) * patch_size[1]
        full_width = (max_col + 1) * patch_size[0]

        # Create a blank canvas for the full map
        if type == "npy":
            full_map = np.zeros((full_height, full_width), dtype=np.float32)
        elif type == "png":
            full_map = np.zeros((full_height, full_width, 3), dtype=np.uint8)

        # Place each patch in the correct position
        for (row, col), patch_filename in patches.items():
            patch_path = os.path.join(input_folder, patch_filename)
            if type == "npy":
                patch_data = np.load(patch_path)
            elif type == "png":
                patch_data = cv2.imread(patch_path)

            full_map[row * patch_size[1]:(row + 1) * patch_size[1],
                     col * patch_size[0]:(col + 1) * patch_size[0]] = patch_data

        # Save the stitched full map
        if type == "npy":
            output_path = os.path.join(output_folder, f"{base_name}.npy")
            np.save(output_path, full_map)
        elif type == "png":
            output_path = os.path.join(output_folder, f"{base_name}.png")
            cv2.imwrite(output_path, full_map)
        print(f"Saved stitched map: {output_path}")


def merge_patch_vertices(json_path, output_json_path, patch_size=(128, 128)):
    """
    将patch级别的顶点坐标，转换为大图级别坐标，并合并保存。

    Args:
        json_path (str): 输入patch级别顶点的json文件路径。
        output_json_path (str): 输出合并后的大图顶点json文件路径。
        patch_size (tuple): patch的尺寸 (width, height)。
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    all_vertices = []

    # 通过正则提取 patch 的行列索引
    pattern = re.compile(r"(.+)_patch_(\d+)_(\d+)\.npy")

    for item in data:
        filename = item["image_file_name"]
        vertices = item["extracted_vertices"]

        match = pattern.match(filename)
        if not match:
            print(f"Warning: filename {filename} does not match expected pattern, skipping.")
            continue

        base_name, row_idx, col_idx = match.group(1), int(match.group(2)), int(match.group(3))

        # 计算这个patch在大图中的左上角坐标
        offset_x = col_idx * patch_size[0]
        offset_y = row_idx * patch_size[1]

        # 将patch内的顶点坐标平移到大图上
        for vert in vertices:
            global_x = vert[0] + offset_x
            global_y = vert[1] + offset_y
            all_vertices.append([global_x, global_y])

    # 生成新的大图json
    merged_data = [{
        "image_file_name": f"{base_name}.png",  # 统一指向大图
        "extracted_vertices": all_vertices
    }]

    with open(output_json_path, 'w') as f:
        json.dump(merged_data, f, indent=2)

    print(f"Merged JSON saved to {output_json_path}")


# Example usage
# input_folder = ("./outputs/evaluate-samples-deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed/epoch=824-step=739199/"
#                 "samples_seg_ddim")  # Replace with the folder containing npy patch files
# output_folder = ("./outputs/evaluate-samples-deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed/epoch=824-step=739199/"
#                 "samples_seg_ddim_stitched")  # Replace with the folder to save stitched npy files
# stitch_patches(input_folder, output_folder, type="png")

# input_folder = ("./outputs/evaluate-samples-deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed/epoch=824-step=739199/"
#                 "samples_seg_ddim_logits_npy")  # Replace with the folder containing npy patch files
# output_folder = ("./outputs/evaluate-samples-deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed/epoch=824-step=739199/"
#                 "samples_seg_ddim_logits_npy_stitched")  # Replace with the folder to save stitched npy files
# stitch_patches(input_folder, output_folder, type="npy")

# input_folder = ("./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/"
#                 "samples_seg_ddim_logits_npy")  # Replace with the folder containing npy patch files
# output_folder = ("./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/"
#                 "samples_seg_ddim_logits_npy_stitched")  # Replace with the folder to save stitched npy files
# stitch_patches(input_folder, output_folder, patch_size=(128, 128), type="npy")

# input_json = ("./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/"
#               "output_vertices_from_scaled1_heatmap_ddim_th-0.5_k-5.0.json")
# output_json = ("./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/"
#                "output_vertices_from_scaled1_heatmap_ddim_th-0.5_k-5.0_stitched.json")
# merge_patch_vertices(input_json, output_json, patch_size=(128, 128))

# ----------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
import os
import json
from skimage.measure import label, regionprops
from process_geb_clip_4270 import douglas_peucker_opencv, visualize_polygons_on_mask

def process_mask(mask_path, vis_path, coco_json_path, heatmap_path=None, epsilon=1.0, sigma=2.5):
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

    coco_annotations = []
    ann_id = 1

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

                # Save polygons
                polygon = flat.flatten().tolist()

                coco_annotations.append({
                    "id": ann_id,
                    "image_id": 1,
                    "category_id": 1,  # 可以根据实际设置类别
                    "segmentation": [polygon],
                    "area": float(cv2.contourArea(flat)),
                    "bbox": cv2.boundingRect(flat),
                    "iscrowd": 0
                })
                ann_id += 1

                if h[3] == -1:
                    ext_polygons.append(flat)
                else:
                    inn_polygons.append(flat)

    # visualize polygons on mask
    vis = visualize_polygons_on_mask(mask, ext_polygons, inn_polygons)
    cv2.imwrite(vis_path, vis)

    # # generate vertex heatmap
    # heatmap = generate_heatmap(vertex_locations, (H, W), sigma=sigma)
    # np.save(heatmap_path, heatmap)

    # save COCO format annotation
    coco_output = {
        "images": [{
            "id": 100,
            "file_name": os.path.basename(mask_path),
            "width": W,
            "height": H
        }],
        "annotations": coco_annotations,
        "categories": [{
            "id": 100,
            "name": "building",  # 这里默认是building，你可以按需要改
            "supercategory": "object"
        }]
    }

    with open(coco_json_path, 'w') as f:
        json.dump(coco_output, f, indent=2)

    print(f"Saved visualization to {vis_path}, and COCO annotation to {coco_json_path}")

# mask_path = "./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_15k_output.png"
# vis_path = "./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_15k_output_polygon_viz.png"
# coco_json_path = "./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_15k_output.json"
# process_mask(mask_path, vis_path, coco_json_path, heatmap_path=None, epsilon=1.0, sigma=2.5)

# ----------------------------------------------------------------------------------------------------------------------
import cv2
import numpy as np
import os
import json
from skimage.measure import label, regionprops

def process_mask(mask_path, vis_path, coco_pred_json_path, epsilon=1.0, sigma=2.5, score=1.0):
    """
    从mask提取多边形并保存为COCO预测格式（prediction json）。
    """
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

    coco_predictions = []
    ann_id = 1

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

                # Save prediction
                polygon = flat.flatten().tolist()
                bbox = cv2.boundingRect(flat)
                x, y, w, h = bbox

                coco_predictions.append({
                    "id": ann_id,
                    "image_id": 100,  # 这里固定写死或者根据你的需要动态赋值
                    "category_id": 100,  # 类别ID
                    "segmentation": [polygon],
                    "area": float(cv2.contourArea(flat)),
                    "bbox": [x, y, w, h],
                    "score": score  # 预测置信度，默认统一设定
                })
                ann_id += 1

                # 加判断，确保h合法
                if isinstance(h, (list, np.ndarray)) and len(h) >= 4:
                    if h[3] == -1:
                        ext_polygons.append(flat)
                    else:
                        inn_polygons.append(flat)
                else:
                    # 如果h不正常，就默认当作外轮廓
                    ext_polygons.append(flat)

    # visualize polygons on mask
    vis = visualize_polygons_on_mask(mask, ext_polygons, inn_polygons)
    cv2.imwrite(vis_path, vis)

    # save prediction JSON
    with open(coco_pred_json_path, 'w') as f:
        json.dump(coco_predictions, f, indent=2)

    print(f"Saved visualization to {vis_path}, and COCO prediction to {coco_pred_json_path}")

mask_path = "outputs/fengyu/Test_1_for_15k/FTest1_input_inv_15_runet.png"
vis_path = "outputs/fengyu/Test_1_for_15k/FTest1_input_inv_15_runet_polygon_viz.png"
coco_pred_json_path = "outputs/fengyu/Test_1_for_15k/FTest1_input_inv_15_runet.json"
process_mask(mask_path, vis_path, coco_pred_json_path, epsilon=1.0, sigma=2.5, score=1.0)

