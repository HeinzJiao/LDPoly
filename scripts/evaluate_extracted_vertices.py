import json
import os
import numpy as np
from scipy.spatial import cKDTree
from tqdm import tqdm

def load_coco_annotations(annotation_path):
    """
    读取 COCO 格式的 annotations，并提取所有 road polygons 的顶点。
    返回 {image_file_name: [[x1, y1], [x2, y2], ...]} 格式的字典。
    """
    with open(annotation_path, "r") as f:
        data = json.load(f)

    image_id_to_filename = {img["id"]: img["file_name"] for img in data["images"]}
    image_vertices = {}

    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in image_id_to_filename:
            continue

        image_filename = image_id_to_filename[image_id]
        polygons = ann["segmentation"]

        # 提取所有的顶点
        vertices = []
        for poly in polygons:
            coords = np.array(poly).reshape(-1, 2)  # 转换为 [[x1, y1], [x2, y2], ...] 形式
            vertices.extend(coords)

        if image_filename in image_vertices:
            image_vertices[image_filename].extend(vertices)
        else:
            image_vertices[image_filename] = vertices

    # 确保所有的点都是 numpy 数组
    for img in image_vertices:
        image_vertices[img] = np.array(image_vertices[img])

    return image_vertices


def load_predicted_vertices(prediction_path):
    """
    读取预测的顶点 JSON 文件，并解析为 {image_file_name: [[x1, y1], [x2, y2], ...]} 格式的字典。
    """
    with open(prediction_path, "r") as f:
        data = json.load(f)

    image_pred_vertices = {}
    for item in data:
        image_filename = item["image_file_name"]
        pred_vertices = np.array(item["extracted_vertices"])

        if image_filename in image_pred_vertices:
            image_pred_vertices[image_filename].extend(pred_vertices)
        else:
            image_pred_vertices[image_filename] = pred_vertices

    return image_pred_vertices

from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

def calculate_precision_recall_strict(pred_coords, gt_coords, threshold=10):
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 1.0, 1.0

    if len(pred_coords) == 0 or len(gt_coords) == 0:
        return 0.0, 0.0

    # 距离矩阵
    dist_matrix = cdist(pred_coords, gt_coords)  # shape: [N_pred, N_gt]
    row_ind, col_ind = linear_sum_assignment(dist_matrix)

    # 只保留距离小于阈值的匹配
    matches = dist_matrix[row_ind, col_ind] <= threshold
    TP = np.sum(matches)
    FP = len(pred_coords) - TP
    FN = len(gt_coords) - TP
    print("TP, FP, FN: ", TP, FP, FN)

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


# def calculate_precision_recall(pred_coords, gt_coords, threshold=10):
#     """
#     计算 Precision 和 Recall。
#     :param pred_coords: 预测的顶点列表 (numpy array, shape=(N,2))
#     :param gt_coords: 真实的顶点列表 (numpy array, shape=(M,2))
#     :param threshold: 计算匹配点的距离阈值
#     :return: (precision, recall)
#     """
#     if len(pred_coords) == 0 and len(gt_coords) == 0:
#         return 1.0, 1.0  # 如果都没有点，Precision 和 Recall 设为 1.0
#
#     if len(pred_coords) == 0:
#         return 0.0, 0.0  # 没有预测点，Precision = 0，Recall = 0
#
#     if len(gt_coords) == 0:
#         return 0.0, 0.0  # 没有真值点，Precision = 0，Recall = 0
#
#     # 使用 cKDTree 计算最近邻距离
#     gt_tree = cKDTree(gt_coords)
#     pred_tree = cKDTree(pred_coords)
#
#     # 计算预测点到真值点的最小距离
#     pred_distances, _ = gt_tree.query(pred_coords, distance_upper_bound=threshold)
#     # 计算真值点到预测点的最小距离
#     gt_distances, _ = pred_tree.query(gt_coords, distance_upper_bound=threshold)
#
#     # 计算 TP, FP, FN
#     TP = np.sum(pred_distances <= threshold)  # 预测点匹配上的真值点数量
#     FP = np.sum(pred_distances > threshold)  # 没有匹配上的预测点数量
#     FN = np.sum(gt_distances > threshold)  # 没有匹配上的真值点数量
#     print("TP, FP, FN: ", TP, FP, FN)
#
#     # 计算 Precision 和 Recall
#     precision = TP / (TP + FP) if (TP + FP) > 0 else 0
#     recall = TP / (TP + FN) if (TP + FN) > 0 else 0
#
#     return precision, recall


def evaluate_predictions(annotation_path, prediction_path, threshold=10):
    """
    计算整个数据集的平均 Precision 和 Recall。
    如果某张图片没有 ground truth 多边形，则跳过该图片。

    :param annotation_path: 真实道路的 COCO annotation JSON 文件
    :param prediction_path: 预测的顶点 JSON 文件
    :param threshold: 计算匹配点的距离阈值
    """
    gt_data = load_coco_annotations(annotation_path)
    pred_data = load_predicted_vertices(prediction_path)

    precision_list = []
    recall_list = []
    num_valid_images = 0  # 记录有效的图片数量

    for image_file in tqdm(gt_data.keys(), desc="Evaluating"):
        gt_coords = gt_data.get(image_file, np.array([]))

        # 如果 GT 为空，则跳过该图像
        if len(gt_coords) == 0:
            continue

        print("image_file: ", image_file)

        pred_coords = pred_data.get(image_file.replace(".png", ".npy"), np.array([]))
        print("pred_coords: ", pred_coords.shape)
        print("gt_coords: ", gt_coords.shape)
        precision, recall = calculate_precision_recall_strict(pred_coords, gt_coords, threshold)

        precision_list.append(precision)
        recall_list.append(recall)
        num_valid_images += 1

    # 计算平均 Precision 和 Recall
    avg_precision = np.mean(precision_list) if num_valid_images > 0 else 0.0
    avg_recall = np.mean(recall_list) if num_valid_images > 0 else 0.0

    print(f"Valid Images Evaluated: {num_valid_images}")
    print(f"Average Precision: {avg_precision:.4f}")
    print(f"Average Recall: {avg_recall:.4f}")

    return avg_precision, avg_recall


# **运行评估**
if __name__ == "__main__":
    annotation_file = "./data/vaihingen_map_generalization/geb15/annotations/geb15_val_annotations.json"  # 替换为 COCO annotations 文件路径
    prediction_file = ("./outputs/vaihingen_map_generalization_sigma2.5_geb15/epoch=epoch=59/"
                       "output_vertices_from_scaled1_heatmap_ddim_th-0.1_k-3.0.json")  # 替换为预测的顶点 JSON 文件路径

    evaluate_predictions(annotation_file, prediction_file, threshold=10)
