"""
Script for performing polygonization of building masks and vertices.

This script processes model-predicted building masks and vertices, generates
polygonal representations, and saves the results along with visualizations.
If the dataset is 'deventer', the coordinates of the segmentation and bounding boxes are scaled
by a factor of 512/500, since the input images were resized from 500x500 to 512x512 during training.

Usage:
    python polygonization.py --annotation_path <path_to_annotation_json> \
                             --outputs_dir <output_directory> \
                             --sampler <sampling_method> \
                             --dataset <dataset_name>

Arguments:
    --annotation_path  Path to the COCO format annotation JSON file for the test set.
    --outputs_dir      Directory where results will be saved (e.g., masks, polygons).
    --sampler          Sampling method used during testing: 'direct', 'ddim', or 'ddpm'.
    --dataset          Name of the dataset (e.g., 'deventer' or 'shanghai'). For 'deventer', scaling will be applied.

Example:
    PYTHONPATH=./:$PYTHONPATH python -u scripts/polygonization.py --annotation_path ./data/giethoorn_road/annotations/test.json \
                             --outputs_dir ./outputs/giethoorn_road/epoch=824-step=739199 \
                             --sampler ddim \
                             --output_vertices_file "output_vertices_from_scaled4_heatmap_ddim_th-0.1_k-3.0.json" \
                             --save_file "polygons_seg_ddim_vertices_from_scaled4_heat_th-0.1_k-3.0_hisup.json" \
                             --samples_seg_logits_file samples_seg_ddim_logits_npy
"""
import numpy as np
import cv2
import torch
import numpy as np
import torch.nn.functional as F
import json
from scipy.spatial.distance import cdist
import os
from skimage.io import imread
from skimage.measure import label, regionprops
import matplotlib.pyplot as plt
import argparse


def ext_c_to_poly_coco(ext_c, im_h, im_w):
    mask = np.zeros([im_h+1, im_w+1], dtype=np.uint8)
    polygon = np.int0(ext_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)

    # 在原始轮廓周围增加一个像素的厚度
    trans_prop_mask[f_y + 1, f_x] = 1
    trans_prop_mask[f_y, f_x + 1] = 1
    trans_prop_mask[f_y + 1, f_x + 1] = 1

    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    new_poly = diagonal_to_square(poly)
    return new_poly


def diagonal_to_square(poly):
    new_c = []
    for id, p in enumerate(poly[:-1]):
        if (p[0] + 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]) \
                or (p[0] - 1 == poly[id + 1][0] and p[1] == poly[id + 1][1]) \
                or (p[0] == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
        elif (p[0] + 1 == poly[id + 1][0] and p[1] + 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] + 1, p[1]])
        elif (p[0] - 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0] - 1, p[1]])
        elif (p[0] + 1 == poly[id + 1][0] and p[1] - 1 == poly[id + 1][1]):
            new_c.append(p)
            new_c.append([p[0], p[1] - 1])
        else:
            new_c.append(p)
            new_c.append([p[0], p[1] + 1])
    new_poly = np.asarray(new_c)
    new_poly = np.concatenate((new_poly, new_poly[0].reshape(-1, 2)))
    return new_poly

def inn_c_to_poly_coco(inn_c, im_h, im_w):
    mask = np.zeros([im_h + 1, im_w + 1], dtype=np.uint8)
    polygon = np.int0(inn_c)
    cv2.drawContours(mask, [polygon.reshape(-1, 1, 2)], -1, color=1, thickness=-1)
    trans_prop_mask = mask.copy()
    f_y, f_x = np.where(mask == 1)


    # 将轮廓最顶部的所有像素（轮廓中所有纵坐标等于轮廓最小y的像素）设为0
    trans_prop_mask[f_y[np.where(f_y == min(f_y))], f_x[np.where(f_y == min(f_y))]] = 0
    # 将轮廓最左边的所有像素（轮廓中所有横坐标等于轮廓最小x的像素）设为0
    trans_prop_mask[f_y[np.where(f_x == min(f_x))], f_x[np.where(f_x == min(f_x))]] = 0
    #trans_prop_mask[max(f_y), max(f_x)] = 1
    # 简而言之，将轮廓向右和向下收缩，潜在目的如下（chatgpt）：
    # 1. 避免内部轮廓与外部轮廓接触：在一些图像处理任务中，内部轮廓可能与外部轮廓非常接近或接触。通过让内部轮廓向内收缩，可以确保内部区域与外部区域之间保持一定的距离。
    # 2. 去除细小的内部结构：如果内部轮廓中包含很多小的凹陷或细小的结构，这些结构可能不希望保留。向内收缩可以帮助平滑这些细小的细节，减少噪声。
    # 3. 确保内部轮廓的独立性：在一些情况下，需要确保内部轮廓和其他轮廓之间的独立性。收缩内部轮廓可以确保内部轮廓不会与外部轮廓或其他内部结构产生交集。

    contours, _ = cv2.findContours(trans_prop_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = contours[0].squeeze(1)[::-1]
    poly = np.concatenate((contour, contour[0].reshape(-1, 2)))
    #return poly
    new_poly = diagonal_to_square(poly)
    return new_poly


def simple_polygon(poly, thres=10):
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    vec0 = lines[:, 2:] - lines[:, :2]
    vec1 = np.roll(vec0, -1, axis=0)
    vec0_ang = np.arctan2(vec0[:,1], vec0[:,0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:,1], vec1[:,0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)

    flag1 = np.roll((lines_ang > thres), 1, axis=0)
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)
    simple_poly = poly[np.bitwise_and(flag1, flag2)]
    simple_poly = np.concatenate((simple_poly, simple_poly[0].reshape(-1,2)))
    return simple_poly


def get_poly(prop, mask_pred, junctions, file_name):
    # process per building
    # mask_pred: 300x300, predicted building segmentation probability map
    # junctions: 600x2
    # process per building
    # prop: <class 'skimage.measure._regionprops.RegionProperties'>
    #       props = regionprops(label(mask_pred_per_im > 0.5))
    #       for prop in props:
    prop_mask = np.zeros_like(mask_pred).astype(np.uint8)

    # the mask of the building (prop_coords: the coordinates of all the points within the mask)
    prop_mask[prop.coords[:, 0], prop.coords[:, 1]] = 1

    # the segmentation probability map of the building
    masked_instance = np.ma.masked_array(mask_pred, mask=(prop_mask != 1))
    # certain entries (prop_mask != 1) will be ingnored in computations
    score = masked_instance.mean()

    im_h, im_w = mask_pred.shape
    contours, hierarchy = cv2.findContours(prop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    poly = []
    for contour, h in zip(contours, hierarchy[0]):
        c = []
        # h[3]: 当前轮廓的父轮廓的索引，如果没有父轮廓，则值为-1
        if h[3] == -1:
            # 当前轮廓为最外围轮廓
            c = ext_c_to_poly_coco(contour, im_h, im_w)  # Nx2
        if h[3] != -1:
            # 当前轮廓为内部轮廓
            if cv2.contourArea(contour) >= 50:
                c = inn_c_to_poly_coco(contour, im_h, im_w)  # Nx2
            #c = inn_c_to_poly_coco(contour, im_h, im_w)
        # 有多于3个顶点才视作有效多边形
        if len(c) > 3:
            init_poly = c.copy()
            if len(junctions) > 0:
                # 每个元素是junctions中与c中顶点最接近的顶点的索引
                cj_match_ = np.argmin(cdist(c, junctions), axis=1)  # N
                # 提取c中每个顶点与其最接近的junctions中顶点之间的距离
                cj_dis = cdist(c, junctions)[np.arange(len(cj_match_)), cj_match_]  # N
                # cj_dis < 5 筛选出距离小于5的匹配对
                # 提取出这些匹配对中junctions中的唯一索引u，以及这些索引在cj_match_中首次出现的位置ind。
                # 这一步确保了在选择多个轮廓顶点匹配到同一个预测顶点时，只保留距离最小的那个。
                u, ind = np.unique(cj_match_[cj_dis < 5], return_index=True)
                if len(u) > 2:
                    # u[np.argsort(ind)] 通过ind的顺序对u进行排序，保证顶点顺序。
                    ppoly = junctions[u[np.argsort(ind)]]
                    # 通过将第一个顶点附加到最后一个顶点，使多边形闭合
                    # ppoly = np.concatenate((ppoly, ppoly[0].reshape(-1, 2)))
                    init_poly = ppoly
            # init_poly = simple_polygon(init_poly, thres=10)
            poly.append(init_poly.flatten().astype(np.float32).tolist())
    return poly, score


def main():
    parser = argparse.ArgumentParser(description='Process polygonization and save results.')

    # Argument for COCO format annotation file (ground truth)
    parser.add_argument('--annotation_path', type=str, required=True, help='Path to COCO format annotation JSON.')

    # Argument for output directory path
    parser.add_argument('--outputs_dir', type=str, required=True, help='Directory path for saving outputs.')

    # Argument for output vertices file
    parser.add_argument('--output_vertices_file', type=str, required=True, help='File name of output vertices.')

    # Argument for output vertices file
    parser.add_argument('--samples_seg_logits_file', type=str, required=True, help='File name of output seg logits.')

    # Argument for save file
    parser.add_argument('--save_file', type=str, required=True, help='File name of saving outputs.')

    # Argument for sampling method (direct, ddim, or ddpm)
    parser.add_argument('--sampler', type=str, choices=['direct', 'ddim', 'ddpm'], required=True,
                        help='Sampling method during testing.')

    args = parser.parse_args()

    annotation_path = args.annotation_path
    outputs_dir = args.outputs_dir
    sampler = args.sampler

    # Load COCO annotations
    with open(annotation_path, 'r') as f:
        coco_annotations = json.load(f)

    # Create lookup for image_id and category_id from annotations
    file_name_to_image_id = {img['file_name']: img['id'] for img in coco_annotations['images']}
    image_id_to_category_id = {ann['image_id']: ann['category_id'] for ann in coco_annotations['annotations']}

    # Define paths for the output directories and files
    output_json_file = os.path.join(outputs_dir, args.output_vertices_file)
    logits_dir = os.path.join(outputs_dir, args.samples_seg_logits_file)  # reconstructed building segmentation probability map
    # output_polygons_dir = os.path.join(outputs_dir, f"polygonization_{sampler}")
    # os.makedirs(output_polygons_dir, exist_ok=True)

    # Load predicted vertices
    with open(output_json_file, 'r') as f:
        results = json.load(f)

    poly_predictions = []
    for vertices_per_img in results:
        # process per image
        file_name = vertices_per_img["image_file_name"]
        junctions = np.array(vertices_per_img["extracted_vertices"])

        # Get image_id from file_name
        file_name_png = file_name.replace(".npy", ".png")  # Assuming annotation uses PNG format file names
        image_id = file_name_to_image_id.get(file_name_png)

        # If .png not found, try .jpg
        if image_id is None:
            file_name_jpg = file_name.replace(".npy", ".jpg")
            image_id = file_name_to_image_id.get(file_name_jpg)

        if image_id is None:
            print(f"Image ID not found for file: {file_name}")
            continue

        # Get category_id based on the image_id from annotations
        category_id = image_id_to_category_id.get(image_id, 100)  # Default to 100 if not found

        # Load predicted mask
        logit_name = file_name.split(".")[0] + ".npy"
        logit_path = os.path.join(logits_dir, logit_name)
        logit = np.load(logit_path)
        mask = logit > 0.5

        # Label connected regions in the mask
        labeled_mask = label(mask)

        # Get region properties for each connected component
        props = regionprops(labeled_mask)

        polygons = []
        scores = []
        for prop in props:
            # Extract polygon for each building region
            poly_list, score = get_poly(prop, logit, junctions, file_name)

            x_min, y_min = np.min(np.array(poly_list[0]).reshape(-1, 2), axis=0)
            x_max, y_max = np.max(np.array(poly_list[0]).reshape(-1, 2), axis=0)
            bbox = [x_min, y_min, x_max - x_min, y_max - y_min]

            # Append polygon prediction with category_id
            poly_predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "segmentation": poly_list,
                "bbox": bbox,
                "score": score
            })

            polygons.append(poly_list)
            scores.append(score)

    # Save all polygon predictions to a JSON file
    poly_predictions_path = os.path.join(outputs_dir, args.save_file)
    with open(poly_predictions_path, 'w') as f:
        json.dump(poly_predictions, f)


if __name__ == "__main__":
    main()
