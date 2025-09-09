import cv2
import json
import os
import numpy as np
from shapely.geometry import Polygon, box
import math


def remove_redundant_points(polygon, threshold):
    """
    移除相邻点之间距离小于阈值的冗余顶点。
    输入 polygon 为二维坐标格式 [[x1, y1], [x2, y2], ...]。
    """
    if len(polygon) < 3:
        return polygon  # 如果多边形点数过少，直接返回
    simplified = [polygon[0]]

    # 处理多边形的内部点
    for i in range(1, len(polygon)):
        if math.dist(simplified[-1], polygon[i]) > threshold:
            simplified.append(polygon[i])

    # 检查第一个点和最后一个点是否冗余（如果多边形是闭合的）
    if len(simplified) > 1 and math.dist(simplified[0], simplified[-1]) <= threshold:
        # 去掉第一个顶点，保留最后一个顶点
        simplified = simplified[1:]

    return simplified


def simple_polygon(poly, thres):
    """
    :param poly: np.ndarray, (N, 2), N is the number of reference corners of the i-th predicted polygon after filtering
    简化多边形，去除夹角小于 angle_threshold 的点，或者去除窄缝（仅针对凹角）的点。
    """
    if (poly[0] == poly[-1]).all():
        poly = poly[:-1]
    # e.g. poly = np.array([[0, 0], [2, 0], [2, 1], [1, 2], [0, 1]])
    lines = np.concatenate((poly, np.roll(poly, -1, axis=0)), axis=1)
    # np.roll(poly, -1, axis=0): [[2, 0], [2, 1], [1, 2], [0, 1], [0, 0]] (All elements move forward one position.)
    # lines: [[0 0 2 0]  no.0 and no.1
    #         [2 0 2 1]  no.1 and no.2
    #         [2 1 1 2]  no.2 and no.3
    #         [1 2 0 1]  no.3 and no.4
    #         [0 1 0 0]] no.4 and no.0
    vec0 = lines[:, 2:] - lines[:, :2]  # edge vectors of the polygon
    vec1 = np.roll(vec0, -1, axis=0)
    vec0_ang = np.arctan2(vec0[:,1], vec0[:,0]) * 180 / np.pi
    vec1_ang = np.arctan2(vec1[:,1], vec1[:,0]) * 180 / np.pi
    lines_ang = np.abs(vec0_ang - vec1_ang)

    # 使用叉积计算每个点的角的凹凸性，判断是凹角还是凸角
    cross_products = vec0[:, 0] * vec1[:, 1] - vec0[:, 1] * vec1[:, 0]

    flag1 = np.roll((lines_ang > thres), 1, axis=0)
    flag2 = np.roll((lines_ang < 360 - thres), 1, axis=0)
    # flag_gap_angle: 只去除凹角且夹角接近 180° 的窄缝点
    # flag_gap_angle = np.roll(~((180 - thres <= lines_ang) & (lines_ang <= 180 + thres) & (cross_products < 0)), 1, axis=0)  # 凹角筛选
    # final_flag = np.bitwise_and(np.bitwise_and(flag1, flag2), flag_gap_angle)
    final_flag = np.bitwise_and(flag1, flag2)
    simple_poly = poly[final_flag]

    return simple_poly


def simple_polygon_loop(poly, angle_threshold=5):
    """
    循环合并夹角小于 angle_threshold 的点，直到没有可合并的点为止。
    同时移除窄缝点，夹角大于 gap_threshold 的点。
    """
    #print("simple_polygon_loop")
    poly1 = poly
    while True:
        #print("%%%%%%%%%%")
        #print("poly1: ", poly1)
        poly2 = simple_polygon(poly1, angle_threshold)
        #print("poly2: ", poly2)
        if poly2.shape[0] < 3:
            break
        if poly2.shape[0] == poly1.shape[0]:
            break
        poly1 = poly2
    return poly2


def simplify_polygon(polygon, distance_threshold=1, angle_threshold=5):
    """
    对原始多边形进行简化操作，输入输出均为一维格式 [x1, y1, x2, y2, ...]:
    1. 去除冗余顶点（距离小于 distance_threshold 的相邻点）
    2. 合并夹角小于 angle_threshold 的边

    :param polygon: 原始多边形点坐标 (list of [x1, y1, x2, y2, ...])
    :param distance_threshold: 合并顶点的距离阈值
    :param angle_threshold: 合并顶点的角度阈值
    :param gap_threshold: 去除窄缝的角度阈值
    :return: 简化后的多边形 (list of [x1, y1, x2, y2, ...])
    """
    polygon_2d = np.array(polygon).reshape(-1, 2)

    # 第一步：去除冗余顶点
    simplified_polygon_2d = remove_redundant_points(polygon_2d, distance_threshold)
    simplified_polygon_2d = np.array(simplified_polygon_2d)

    # 第二步：简化多边形，去除夹角小于 angle_threshold 和大于 gap_threshold 的点
    if angle_threshold > 0:
        simplified_polygon_2d = simple_polygon_loop(simplified_polygon_2d, angle_threshold)

    # 将简化后的二维数组转换回一维数组
    simplified_polygon = simplified_polygon_2d.flatten().tolist()

    return simplified_polygon


def filter_building_and_holes(simplified_segmentation, area_threshold=100):
    """
    过滤建筑物的外围轮廓和内部轮廓：
    1. 如果外围轮廓的顶点数小于3或面积小于给定阈值，则去掉整个建筑物。
    2. 如果外围轮廓满足条件，检查内部轮廓，去掉顶点数小于3或面积小于阈值的内部轮廓。

    参数:
    - simplified_segmentation: 简化后的建筑物多边形列表，格式为 [[外部轮廓], [内部轮廓1], ...]
    - area_threshold: 面积阈值，默认 100 个像素。

    返回:
    - 过滤后的多边形列表。如果外围轮廓不满足条件，返回空列表。
    """

    # 处理外围轮廓（第一个多边形）
    exterior_polygon = simplified_segmentation[0]
    polygon_2d = np.array(exterior_polygon).reshape(-1, 2)  # 转换为二维数组

    # 检查外围轮廓的顶点数和面积
    if len(polygon_2d) < 3 or Polygon(polygon_2d).area < area_threshold:
        return []  # 如果顶点数小于3或面积小于阈值，去掉整个建筑物

    # 保留外围轮廓
    new_polygon = [exterior_polygon]

    # 处理内部轮廓（从第二个多边形开始）
    for interior_polygon in simplified_segmentation[1:]:
        polygon_2d = np.array(interior_polygon).reshape(-1, 2)
        # 检查内部轮廓的顶点数和面积
        if len(polygon_2d) >= 3 and Polygon(polygon_2d).area >= area_threshold:
            new_polygon.append(interior_polygon)  # 保留符合条件的内部轮廓

    return new_polygon


def resize_annotations(annotations, scale_factor):
    """
    按比例缩放 annotations 中的多边形坐标和 bounding box。

    参数:
    - annotations: 原始的建筑物标注数据（COCO 格式的 annotations）。[dict1, dict2, ...]
    - scale_factor: 缩放因子。

    返回:
    - 缩放后的 annotations。
    """
    new_annotations = []
    for ann in annotations:
        new_ann = ann.copy()
        # 缩放 segmentation 多边形坐标
        for i in range(len(new_ann['segmentation'])):
            new_ann['segmentation'][i] = [coord * scale_factor for coord in new_ann['segmentation'][i]]

        # 缩放 bounding box
        new_ann['bbox'] = [coord * scale_factor for coord in new_ann['bbox']]

        new_annotations.append(new_ann)

    return new_annotations


def clip_polygon_to_patch(polygon, patch_box):
    """
    将多边形裁剪到patch范围内，并返回新的多边形列表（可能会被分割成多个多边形）。

    参数:
    - polygon: 原始多边形，格式为 [[外部轮廓], [内部轮廓1], [内部轮廓2], ...]。
    - patch_box: patch 的边界，格式为 shapely.geometry.box。

    返回:
    - 裁剪后的多边形列表，每个多边形格式为 [[外部轮廓], [内部轮廓1], [内部轮廓2], ...]。
    """
    if len(polygon) == 0:
        return []

    # 第一个元素是外部轮廓，后面的可能是内部轮廓（洞）
    exterior = polygon[0]
    interiors = polygon[1:] if len(polygon) > 1 else []

    # 创建一个包含外部轮廓和内部轮廓的 shapely Polygon 对象
    original_polygon = Polygon(shell=np.array(exterior).reshape(-1, 2),
                               holes=[np.array(hole).reshape(-1, 2) for hole in interiors])

    # 使用 patch_box 将原始多边形裁剪
    clipped_polygons = original_polygon.intersection(patch_box)

    if clipped_polygons.is_empty:
        return []

    # 如果裁剪结果是多个多边形部分（MultiPolygon），处理每个部分
    clipped_results = []
    if clipped_polygons.geom_type == 'MultiPolygon':
        for poly in clipped_polygons:
            # 提取外部轮廓和内部轮廓
            exterior_coords = list(poly.exterior.coords)
            interior_coords = [list(interior.coords) for interior in poly.interiors]
            clipped_results.append([exterior_coords] + interior_coords)

    elif clipped_polygons.geom_type == 'Polygon':
        # 单个多边形的情况
        exterior_coords = list(clipped_polygons.exterior.coords)
        interior_coords = [list(interior.coords) for interior in clipped_polygons.interiors]
        clipped_results.append([exterior_coords] + interior_coords)

    return clipped_results
