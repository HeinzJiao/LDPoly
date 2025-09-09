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
                             --image_dir <path_to_test_images> \
                             --dataset <dataset_name>

Arguments:
    --annotation_path  Path to the COCO format annotation JSON file for the test set.
    --outputs_dir      Directory where results will be saved (e.g., masks, polygons).
    --sampler          Sampling method used during testing: 'direct', 'ddim', or 'ddpm'.
    --image_dir        Path to the folder containing test images.
    --dataset          Name of the dataset (e.g., 'deventer' or 'shanghai'). For 'deventer', scaling will be applied.

Example:
    PYTHONPATH=./:$PYTHONPATH python -u scripts/polygonization_refined.py --annotation_path ./data/vaihingen_map_generalization/geb15/annotations/geb15_val_annotations.json \
                             --outputs_dir ./outputs/vaihingen_map_generalization_sigma2.5_geb15/epoch=epoch=59 \
                             --sampler ddim \
                             --output_vertices_file "output_vertices_from_scaled1_heatmap_ddim_th-0.1_k-5.0.json" \
                             --save_file "polygons_seg_ddim_vertices_from_scaled1_heat_th-0.1_k-5.0_3.2_dp_eps2.json" \
                             --samples_seg_logits_file samples_seg_ddim_logits_npy

    PYTHONPATH=./:$PYTHONPATH python -u scripts/polygonization_refined.py --annotation_path ./data/vaihingen_map_generalization/geb15/annotations/geb15_val_annotations.json \
                             --outputs_dir ./outputs/vaihingen_map_generalization_sigma2.5_geb15/epoch=epoch=59 \
                             --sampler ddim \
                             --output_vertices_file "output_vertices_from_scaled1_heatmap_ddim_th-0.1_k-5.0.json" \
                             --save_file "polygons_seg_ddim_vertices_from_scaled1_heat_th-0.1_k-5.0_3.2_dp_eps2.json" \
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


def get_poly(prop, mask_pred, junctions, d_th, vis_save_path=None, file_name=None, region_idx=None):
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

    k = 0

    contours_to_poly_coco = []

    for contour, h in zip(contours, hierarchy[0]):
        # h[3]: 当前轮廓的父轮廓的索引，如果没有父轮廓，则值为-1
        if h[3] == -1:
            # 当前轮廓为最外围轮廓
            c = ext_c_to_poly_coco(contour, im_h, im_w)  # Nx2    # ************************************************************
        else:
            # 当前轮廓为内部轮廓
            # if cv2.contourArea(contour) >= 50:  # 对面积小于 50 的孔洞直接忽略，避免噪声。
            #     c = inn_c_to_poly_coco(contour, im_h, im_w)  # Nx2
            c = inn_c_to_poly_coco(contour, im_h, im_w)  # Nx2   # ************************************************************

        # c = contour.squeeze(1)  # (N, 2)

        contours_to_poly_coco.append(c[:, None, :])

        ################### polygonization ###################
        # init_poly = filter_contour_by_junctions31(c, junctions, distance_threshold=d_th)
        init_poly = filter_contour_by_junctions32(c, junctions, distance_threshold=d_th, dp_eps=2, angle_threshold=30)
        # use douglas peucker algorithm instead of polygonization method:
        # init_poly = douglas_peucker_opencv(c, epsilon=1)  # ************************************************************

        if init_poly is not None:
            # print(init_poly.shape)
            # merge parallel edges with a threshold up to thres
            # init_poly = simple_polygon(init_poly, thres=1)  # ************************************************************
            # print("after merge 10 once: ", init_poly.shape)
            if len(init_poly) > 2:
                poly.append(init_poly.flatten().astype(np.float32).tolist())

    if vis_save_path is not None and file_name is not None and region_idx is not None:
        os.makedirs(vis_save_path, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(file_name))[0]

        scale = 2  # 放大倍数
        vis_h, vis_w = im_h * scale, im_w * scale

        def clip_point(x, y):
            x = int(np.clip(x, 0, vis_w - 1))
            y = int(np.clip(y, 0, vis_h - 1))
            return x, y

        # 初始化放大的白板图
        canvas_junc = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255
        canvas_contour = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255
        canvas_composite = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255
        canvas_poly = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255

        # 画所有 junctions（绿色）
        radius = 3
        line_thickness = 1
        for junc in junctions:
            x, y = clip_point(junc[0] * scale, junc[1] * scale)
            cv2.circle(canvas_junc, (x, y), radius=radius, color=(0, 255, 0), thickness=-1)
            cv2.circle(canvas_composite, (x, y), radius=radius, color=(0, 255, 0), thickness=-1)

        # 画所有 contours（灰色）
        for contour in contours_to_poly_coco:
            contour_scaled = (contour * scale)
            contour_scaled[:, :, 0] = np.clip(contour_scaled[:, :, 0], 0, vis_w - 1)
            contour_scaled[:, :, 1] = np.clip(contour_scaled[:, :, 1], 0, vis_h - 1)
            contour_int = contour_scaled.astype(np.int32)
            cv2.polylines(canvas_contour, [contour_int], isClosed=True, color=(128, 128, 128), thickness=line_thickness)
            cv2.polylines(canvas_composite, [contour_int], isClosed=True, color=(128, 128, 128),
                          thickness=line_thickness)

        # 画 polygons（magenta点在composite中, cyan线+magenta点在canvas_poly中）
        for polygon_flat in poly:
            pts = np.array(polygon_flat, dtype=np.float32).reshape(-1, 2) * scale
            pts[:, 0] = np.clip(pts[:, 0], 0, vis_w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, vis_h - 1)
            pts_int = pts.astype(np.int32)

            # 画 magenta 点在 composite 中（最上层）
            for x, y in pts_int:
                cv2.circle(canvas_composite, (x, y), radius=radius, color=(255, 0, 255), thickness=-1)

            # 在 polygon-only canvas 上画 polygon（cyan）和顶点（magenta）
            cv2.polylines(canvas_poly, [pts_int], isClosed=True, color=(255, 255, 0), thickness=line_thickness)
            for x, y in pts_int:
                cv2.circle(canvas_poly, (x, y), radius=radius, color=(255, 0, 255), thickness=-1)

        # 保存四张图
        cv2.imwrite(os.path.join(vis_save_path, f"{base_name}_region_{region_idx}_junctions.png"), canvas_junc)
        cv2.imwrite(os.path.join(vis_save_path, f"{base_name}_region_{region_idx}_contours.png"), canvas_contour)
        cv2.imwrite(os.path.join(vis_save_path, f"{base_name}_region_{region_idx}_composite.png"), canvas_composite)
        cv2.imwrite(os.path.join(vis_save_path, f"{base_name}_region_{region_idx}_polygon.png"), canvas_poly)

        print(f"[VIS SAVED] {base_name} region {region_idx}")

    return poly, score


def filter_contour_by_junctions31(c, junctions, distance_threshold=5):
    """
    从建筑物mask的轮廓顶点中筛选出与预测的建筑物顶点（junctions）距离小于给定阈值的顶点。

    参数：
        init_poly (numpy.ndarray): 初始多边形，更新后返回。
        c (numpy.ndarray): 建筑物mask的密集轮廓顶点集合，形状为(N, 2)。
        junctions (numpy.ndarray): 所有预测出的建筑物顶点集合，形状为(M, 2)。
        distance_threshold (float): 距离阈值，单位为像素。

    返回：
        numpy.ndarray: 筛选后的建筑物轮廓多边形顶点。
    """
    if len(junctions) == 0:
        simplified_poly = douglas_peucker_opencv(c, epsilon=2)
        if len(simplified_poly) > 2:
            return simplified_poly  # 简化后的多边形点集合
        else:
            return None

    # 计算 c 和 junctions 之间的欧几里得距离矩阵
    distances = cdist(c, junctions)  # 形状为 (N, M)

    # 对于每个 junctions 中的点，找到距离最近的 c 中的点及其距离
    nearest_indices = np.argmin(distances, axis=0)  # 每列取最小值，得到最近的 c 的索引
    nearest_distances = np.min(distances, axis=0)  # 每列的最小距离

    # 筛选出距离小于阈值的索引
    # 对于某个 junctions 中的点，如果距离它最近的 c 中的点，与它的距离小于 distance threshold，则保留该 c 点
    valid_indices = nearest_indices[nearest_distances < distance_threshold]  # ********************

    # 去重，避免重复选择，并确保多边形顶点顺序
    # """版本3"""
    # unique_indices = np.unique(nearest_indices)
    # poly = c[np.sort(unique_indices)]
    # if len(poly) > 2:
    #     return poly  # (N_poly, 2)
    #
    # return init_poly

    """版本3.1"""
    unique_indices = np.unique(valid_indices)
    # 如果筛选出的点为空，使用 Douglas-Peucker 算法生成简化轮廓
    if len(unique_indices) == 0:
        simplified_poly = douglas_peucker_opencv(c, epsilon=2)
        if len(simplified_poly) > 2:
            return simplified_poly  # 简化后的多边形点集合
        else:
            return None
    else:
        poly = c[np.sort(unique_indices)]
        if len(poly) > 2:
            return poly  # (N_poly, 2)
        else:
            return None


def is_critical_point(prev_point, current_point, next_point, angle_threshold=30):
    """
    判断一个点是否是关键点（如直角点）。

    Args:
        prev_point (tuple): 当前点的前一个点 (x, y)。
        current_point (tuple): 当前点 (x, y)。
        next_point (tuple): 当前点的后一个点 (x, y)。
        angle_threshold (float): 判断是否为关键点的角度阈值，单位为度。

    Returns:
        bool: 如果是关键点返回 True，否则返回 False。
    """
    vec1 = np.array(prev_point) - np.array(current_point)
    vec2 = np.array(next_point) - np.array(current_point)

    # 计算两个向量的夹角
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    angle_deg = np.degrees(angle)  # 转换为角度

    # 判断是否为直角点
    return 90 - angle_threshold < angle_deg < 90 + angle_threshold


def filter_contour_by_junctions32(c, junctions, distance_threshold=5, dp_eps=2, angle_threshold=30):
    """
    从建筑物mask的轮廓顶点中筛选出与预测的建筑物顶点（junctions）距离小于给定阈值的顶点。

    参数：
        init_poly (numpy.ndarray): 初始多边形，更新后返回。
        c (numpy.ndarray): 建筑物mask的密集轮廓顶点集合，形状为(N, 2)。
        junctions (numpy.ndarray): 所有预测出的建筑物顶点集合，形状为(M, 2)。
        distance_threshold (float): 距离阈值，单位为像素。

    返回：
        numpy.ndarray: 筛选后的建筑物轮廓多边形顶点。
    """
    if len(junctions) == 0:
        simplified_poly = douglas_peucker_opencv(c, epsilon=2)
        if len(simplified_poly) > 2:
            return simplified_poly  # 简化后的多边形点集合
        else:
            return None

    # 计算 c 和 junctions 之间的欧几里得距离矩阵
    distances = cdist(c, junctions)  # 形状为 (N, M)
    # 对于每个 junctions 中的点，找到距离最近的 c 中的点及其距离
    nearest_indices = np.argmin(distances, axis=0)  # 每列取最小值，得到最近的 c 的索引
    nearest_distances = np.min(distances, axis=0)  # 每列的最小距离

    # 筛选出距离小于阈值的索引
    # 对于某个 junctions 中的点，如果距离它最近的 c 中的点，与它的距离小于 distance threshold，则保留该 c 点
    valid_indices = nearest_indices[nearest_distances < distance_threshold]  # ********************
    print("len valid_indices: ", len(valid_indices))

    # 对轮廓 c 应用 Douglas-Peucker 算法，简化轮廓，用于关键点检测
    simplified_c, simplified_indices = douglas_peucker_with_indices(c, epsilon=dp_eps)    # ************************************************************
    print("simplified_c: ", simplified_c.shape)

    # 保留原多边形上的关键点
    critical_indices = []
    for i in range(1, len(simplified_c) - 1):
        if is_critical_point(simplified_c[i - 1], simplified_c[i], simplified_c[i + 1], angle_threshold):
            critical_indices.append(simplified_indices[i])  # 映射回原始 c 中的索引
    print("len critical_indices: ", len(critical_indices))

    # 合并关键点和筛选点，去重并保持顺序
    final_indices = np.unique(np.concatenate((valid_indices, critical_indices))).astype(np.int64)
    print("final_indices: ", final_indices.shape)

    # 如果筛选出的点为空，使用 Douglas-Peucker 算法生成简化轮廓
    if len(final_indices) == 0:
        simplified_poly = douglas_peucker_opencv(c, epsilon=2)    # ************************************************************
        if len(simplified_poly) > 2:
            return simplified_poly  # 简化后的多边形点集合
        else:
            return None
    else:
        poly = c[np.sort(final_indices)]
        if len(poly) > 2:
            return poly  # (N_poly, 2)
        else:
            return None


def douglas_peucker_with_indices(c, epsilon):
    """
    使用 OpenCV 的 Douglas-Peucker 算法简化轮廓，并返回简化后的点及其在原始轮廓中的索引。

    参数：
        c (numpy.ndarray): 原始轮廓点集合，形状为(N, 2)。
        epsilon (float): Douglas-Peucker 算法的简化系数。

    返回：
        simplified_c (numpy.ndarray): 简化后的轮廓点集合。
        simplified_indices (list): 简化点在原始轮廓 c 中的索引。
    """
    # 使用 OpenCV 的 approxPolyDP 进行多边形简化
    simplified_c = cv2.approxPolyDP(c.astype(np.float32), epsilon, closed=False).reshape(-1, 2)

    # 匹配简化点与原始点以获得索引
    simplified_indices = []
    for point in simplified_c:
        # 计算每个简化点到原始点的欧几里得距离，找到最近点的索引
        distances = np.linalg.norm(c - point, axis=1)
        nearest_idx = np.argmin(distances)
        simplified_indices.append(nearest_idx)

    return simplified_c, simplified_indices


def douglas_peucker_opencv(points, epsilon):
    """
    使用 OpenCV 的 Douglas-Peucker 算法简化多边形轮廓。

    参数：
        points (numpy.ndarray): 输入的点集合，形状为 (N, 2)。
        epsilon (float): 简化的阈值，控制轮廓简化程度。

    返回：
        numpy.ndarray: 简化后的点集合。
    """
    # OpenCV 的 Douglas-Peucker 需要输入形状为 (N, 1, 2)
    contour = points.reshape((-1, 1, 2)).astype(np.float32)
    simplified_contour = cv2.approxPolyDP(contour, epsilon, closed=True)
    return simplified_contour.reshape((-1, 2))


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

    parser.add_argument('--polygonization_vis_path', type=str, required=False, help='File name of saving outputs.')

    # Argument for sampling method (direct, ddim, or ddpm)
    parser.add_argument('--sampler', type=str, choices=['direct', 'ddim', 'ddpm'], required=True,
                        help='Sampling method during testing.')

    parser.add_argument('--d_th', type=float, required=True, help='Distance threshold.')

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

    # Define paths for the output directories and files **************************************************
    output_json_file = os.path.join(outputs_dir, args.output_vertices_file)
    logits_dir = os.path.join(outputs_dir, args.samples_seg_logits_file)  # reconstructed building segmentation probability map

    # Load predicted vertices
    with open(output_json_file, 'r') as f:
        results = json.load(f)

    poly_predictions = []
    for vertices_per_img in results:
        # process per image
        file_name = vertices_per_img["image_file_name"]
        print("file_name: ", file_name)
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
        for i, prop in enumerate(props):
            # Extract polygon for each building region
            poly_list, score = get_poly(prop, logit, junctions, args.d_th,
                                        vis_save_path=args.polygonization_vis_path,
                                        file_name=file_name,
                                        region_idx=i
                                        )
            if len(poly_list) == 0:
                continue

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

    poly_predictions_path = os.path.join(outputs_dir, args.save_file)
    with open(poly_predictions_path, 'w') as f:
        json.dump(poly_predictions, f)


if __name__ == "__main__":
    main()
