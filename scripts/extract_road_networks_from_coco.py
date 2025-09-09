import os
import cv2
import numpy as np
import networkx as nx
from skimage.morphology import skeletonize
from skimage.measure import label
import pickle
import time
import matplotlib.pyplot as plt
import scipy
from tqdm import tqdm
import json
import argparse

from torch.cuda import graph_pool_handle


def extract_skeleton(road_mask):
    """对道路分割掩码进行骨架提取"""
    road_mask = road_mask.astype(np.uint8)
    skeleton = skeletonize(road_mask)
    return skeleton.astype(np.uint8)


def skeleton_to_graph(skeleton):
    """将 skeleton 转换为 graph"""
    G = nx.Graph()
    labeled = label(skeleton)
    coords = np.argwhere(labeled > 0)

    for y, x in coords:
        G.add_node((y, x))
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if (dy, dx) != (0, 0):
                    neighbor = (y + dy, x + dx)
                    if neighbor in G:
                        G.add_edge((y, x), neighbor)
    return G


def load_image_id_mapping(gt_annotation_file):
    """从 COCO 格式的真值 annotation 文件中获取 image_id 到 file_name 的映射"""
    with open(gt_annotation_file, 'r') as f:
        gt_data = json.load(f)
    return {img['id']: img['file_name'] for img in gt_data['images']}


def extract_road_mask_from_predictions(prediction_file, image_id, image_shape):
    """从 COCO prediction 文件中提取道路 mask"""
    with open(prediction_file, 'r') as f:
        pred_data = json.load(f)

    road_mask = np.zeros(image_shape[:2], dtype=np.uint8)

    for ann in pred_data:
        if ann['image_id'] == image_id:
            segmentation = ann.get('segmentation', [])
            for i, poly in enumerate(segmentation):
                poly_coords = np.array(poly, dtype=np.int32).reshape((-1, 2))
                if i == 0:
                    cv2.fillPoly(road_mask, [poly_coords], 1)
                else:
                    cv2.fillPoly(road_mask, [poly_coords], 0)

    return road_mask


def visualize_graph_on_image(image, graph):
    """在原始图像上绘制 graph"""
    vis_image = image.copy()

    # 画出节点
    for node in graph.nodes():
        cv2.circle(vis_image, (node[1], node[0]), 1, (0, 0, 255), -1)  # 红色小圆点

    # 画出边
    for edge in graph.edges():
        pt1, pt2 = edge
        cv2.line(vis_image, (pt1[1], pt1[0]), (pt2[1], pt2[0]), (255, 0, 0), 1)  # 蓝色细线

    return vis_image


def process_gt_mask_folder(gt_annotation_file, image_folder, output_folder, save_mask_folder=None):
    """
    从 COCO 格式 ground truth annotation 中提取伪路网。

    :param gt_annotation_file: COCO 格式 ground truth annotation 文件路径
    :param image_folder: 存放图像的文件夹
    :param output_folder: 存放输出的 graph (.gpickle) 和可视化图像
    """
    os.makedirs(output_folder, exist_ok=True)
    vis_output_folder = os.path.join(output_folder, "visualizations")
    os.makedirs(vis_output_folder, exist_ok=True)

    if save_mask_folder is not None:
        os.makedirs(save_mask_folder, exist_ok=True)

    # 读取 GT annotations
    with open(gt_annotation_file, 'r') as f:
        gt = json.load(f)

    # 建立 image_id 到 file_name 的映射
    image_id_to_name = {img['id']: img['file_name'] for img in gt['images']}

    # 建立 image_id 到 polygons 的映射
    from collections import defaultdict
    polygons_by_image = defaultdict(list)
    for ann in gt['annotations']:
        polygons_by_image[ann['image_id']].append(ann['segmentation'])

    for image_id, file_name in tqdm(image_id_to_name.items(), desc="Processing GT Images"):
        image_path = os.path.join(image_folder, file_name)
        output_graph_path = os.path.join(output_folder,
                                         file_name.replace('.jpg', '.gpickle').replace('.png', '.gpickle'))
        vis_path = os.path.join(vis_output_folder, file_name)

        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read {image_path}, skipping...")
            continue

        H, W = image.shape[:2]
        mask = np.zeros((H, W), dtype=np.uint8)

        for segmentation in polygons_by_image[image_id]:
            for i, poly in enumerate(segmentation):
                poly_coords = np.array(poly, dtype=np.int32).reshape((-1, 2))
                if i == 0:
                    cv2.fillPoly(mask, [poly_coords], 1)  # 外轮廓填1
                else:
                    cv2.fillPoly(mask, [poly_coords], 0)  # 内部孔洞抠掉

        # 可选：保存 mask
        if save_mask_folder is not None:
            mask_path = os.path.join(save_mask_folder, file_name)
            cv2.imwrite(mask_path, mask * 255)  # 保存为黑白图（255表示前景）

        # 以下是你的骨架化和图转化代码（需要你已有的实现）
        skeleton = extract_skeleton(mask)
        graph = skeleton_to_graph(skeleton)

        # 保存 graph
        with open(output_graph_path, "wb") as f:
            pickle.dump(graph, f)

        # 可视化
        vis_image = visualize_graph_on_image(image, graph)
        cv2.imwrite(vis_path, vis_image)

        print(f"Processed {file_name} → {output_graph_path}, visualization saved to {vis_path}")


def process_mask_folder(prediction_file, gt_annotation_file, image_folder, output_folder):
    """
    读取 COCO 格式的 prediction 文件，将道路多边形转换为 mask 并提取 road network。

    :param prediction_file: COCO prediction 文件路径
    :param gt_annotation_file: COCO ground truth annotation 文件路径
    :param image_folder: 存放原始图片的文件夹（文件名需匹配 annotation）
    :param output_folder: 存放生成的 road network graph 和可视化图片的文件夹
    """
    os.makedirs(output_folder, exist_ok=True)
    vis_output_folder = os.path.join(output_folder, "visualizations")
    os.makedirs(vis_output_folder, exist_ok=True)

    image_id_mapping = load_image_id_mapping(gt_annotation_file)

    for image_id, file_name in tqdm(image_id_mapping.items(), desc="Processing Images"):
        image_path = os.path.join(image_folder, file_name)
        output_graph_path = os.path.join(output_folder,
                                         file_name.replace('.jpg', '.gpickle').replace('.png', '.gpickle'))
        vis_path = os.path.join(vis_output_folder, file_name)

        # 读取原始图像
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read {image_path}, skipping...")
            continue

        # 生成道路 mask
        road_mask = extract_road_mask_from_predictions(prediction_file, image_id, image.shape)

        # 生成 skeleton
        skeleton = extract_skeleton(road_mask)

        # 生成 graph
        graph = skeleton_to_graph(skeleton)

        # 保存 graph
        with open(output_graph_path, "wb") as f:
            pickle.dump(graph, f)

        # 生成可视化图像
        vis_image = visualize_graph_on_image(image, graph)
        cv2.imwrite(vis_path, vis_image)

        print(f"Processed: {file_name} → {output_graph_path}, visualization saved to {vis_path}")

# ----------------------------------------------------------------------------------------------------------------------
# 以下为计算APLS相关代码，以上为与提取密集顶点序列的伪路网并可视化后保存至本地相关的代码。

# def compute_shortest_paths(graph):
#     """ 计算 graph 的所有节点对的最短路径长度 """
#     all_pairs_lengths = {}
#     for node in graph.nodes():
#         lengths = nx.single_source_dijkstra_path_length(graph, node)
#         all_pairs_lengths[node] = lengths
#     return all_pairs_lengths


def compute_shortest_paths(graph, control_points):
    """
    计算给定 graph 中所有 control_points 之间的最短路径
    返回字典 { (i, j): path_length }
    """
    shortest_paths = {}
    for i in range(len(control_points)):
        for j in range(i + 1, len(control_points)):  # 两两组合
            try:
                path_length = nx.shortest_path_length(graph, source=control_points[i], target=control_points[j], weight='weight')
                shortest_paths[(control_points[i], control_points[j])] = path_length
            except nx.NetworkXNoPath:  # 无法连通的情况
                shortest_paths[(control_points[i], control_points[j])] = float('inf')
    return shortest_paths


def get_endpoints_and_intersections(graph):
    """
    选取伪路网中的端点（度为1的节点）和交叉路口（度大于2的节点）作为控制点
    """
    control_points = [node for node, degree in graph.degree() if degree == 1 or degree > 2]
    # return [(node, 0) for node in control_points]  # 转换为 APLS 所需格式
    return control_points


def compute_apls(graph_folder_1, graph_folder_2):
    """计算两个文件夹中的 graph 之间的 APLS (Average Path Length Similarity)"""
    total_apls = 0.0
    count = 0
    graph_files = [f for f in os.listdir(graph_folder_1) if f.endswith(".gpickle")]

    for graph_file in tqdm(graph_files, desc="Computing APLS", unit="file"):
        graph_path_1 = os.path.join(graph_folder_1, graph_file)
        graph_path_2 = os.path.join(graph_folder_2, graph_file)

        if not os.path.exists(graph_path_2):
            print(f"Warning: {graph_file} not found in {graph_folder_2}, skipping...")
            continue

        # 读取 graph
        with open(graph_path_1, "rb") as f:
            graph_1 = pickle.load(f)
        with open(graph_path_2, "rb") as f:
            graph_2 = pickle.load(f)

        # 如果 ground truth 的 graph 是空的，则跳过
        if len(graph_2.nodes) == 0:
            print(f"Skipping {graph_file} because ground truth graph is empty.")
            continue

        # 选取端点和交叉路口作为控制点
        control_points_1 = get_endpoints_and_intersections(graph_1)
        control_points_2 = get_endpoints_and_intersections(graph_2)

        print("control_points_1: ", control_points_1)
        print("control_points_2: ", control_points_2)

        # 计算最短路径
        all_pairs_lengths_1 = compute_shortest_paths(graph_1, control_points_1)
        all_pairs_lengths_2 = compute_shortest_paths(graph_2, control_points_2)
        total_error = 0.0
        count = 0
        for key in all_pairs_lengths_2:
            L_2 = all_pairs_lengths_2[key]
            L_1 = all_pairs_lengths_1.get(key, float('inf'))  # 如果 Pred 没有该路径，则设为无穷大
            if L_2 > 0:  # 避免除零
                error = abs(L_2 - L_1) / L_2
                total_error += error
                count += 1
        apls_score = 1 - (total_error / count if count > 0 else 1)


        total_apls += apls_score
        count += 1
        print(f"APLS for {graph_file}: {apls_score:.4f}")

    avg_apls = total_apls / count if count > 0 else 0.0
    print(f"Average APLS: {avg_apls:.4f}")
    return avg_apls


def custom_visualize(image, graph, control_points, color=(255, 0, 0), keypoint_color=(0, 255, 0)):
    """
    注意：图像坐标系（OpenCV） 和 graph 节点坐标（NetworkX） 之间的坐标系统不同。
    在图像上绘制 graph 和控制点：
    - `color` 用于绘制道路（Graph）
    - `keypoint_color` 用于绘制控制点（端点/交叉口）
    """
    vis_image = image.copy()

    # 画边（道路）
    for edge in graph.edges():
        pt1, pt2 = edge
        pt1 = (int(pt1[1]), int(pt1[0]))  # 交换 x, y
        pt2 = (int(pt2[1]), int(pt2[0]))
        cv2.line(vis_image, pt1, pt2, color, thickness=2)

    # 画控制点（端点和交叉口）
    for node in control_points:
        node = (int(node[1]), int(node[0]))  # 交换 x, y
        cv2.circle(vis_image, node, radius=4, color=keypoint_color, thickness=-1)

    return vis_image


def visualize_road_networks_and_control_points(graph_folder_1, graph_folder_2, image_folder, output_folder):
    """可视化 GT 和 Prediction 的路网及其控制点"""
    os.makedirs(output_folder + '/'+ 'road_network_and_control_points_visual', exist_ok=True)

    graph_files = [f for f in os.listdir(graph_folder_1) if f.endswith(".gpickle")]

    for graph_file in tqdm(graph_files, desc="Visualizing Road Networks", unit="file"):
        graph_path_1 = os.path.join(graph_folder_1, graph_file)
        graph_path_2 = os.path.join(graph_folder_2, graph_file)
        image_path = os.path.join(image_folder, graph_file.replace('.gpickle', '.png'))

        if not os.path.exists(graph_path_2):
            print(f"Warning: {graph_file} not found in {graph_folder_2}, skipping...")
            continue

        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping visualization...")
            continue

        # 读取 graph
        with open(graph_path_1, "rb") as f:
            graph_1 = pickle.load(f)
        with open(graph_path_2, "rb") as f:
            graph_2 = pickle.load(f)

        if len(graph_1.nodes) == 0 or len(graph_2.nodes) == 0:
            print(f"Skipping {graph_file} because one of the graphs is empty.")
            continue

        # 选取端点和交叉路口作为控制点
        control_points_1 = get_endpoints_and_intersections(graph_1)
        control_points_2 = get_endpoints_and_intersections(graph_2)

        # 读取原始图片
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image {image_path}, skipping...")
            continue

        # 生成带有 Graph 可视化的图片
        vis_pred = custom_visualize(image, graph_2, control_points_2, color=(0, 255, 0),
                                  keypoint_color=(0, 0, 255))  # GT (红线 + 绿点)
        vis_gt = custom_visualize(image, graph_1, control_points_1, color=(255, 255, 0),
                                            keypoint_color=(255, 0, 0))  # Prediction (蓝线 + 青色点)

        # 拼接 GT 和 Prediction 视图
        combined_vis = np.hstack([vis_gt, vis_pred])

        # 保存可视化结果
        output_vis_path = os.path.join(output_folder, 'road_network_and_control_points_visual', graph_file.replace('.gpickle', '_comparison.png'))
        cv2.imwrite(output_vis_path, combined_vis)

        print(f"Saved visualization to {output_vis_path}")

# ----------------------------------------------------------------------------------------------------------------------
# 以下为通过理解透彻apls的计算逻辑后自行写的apls计算代码
from scipy.spatial import KDTree

def find_nearest_control_point(control_points, target_point):
    """在 control_points (而不是整个 graph) 中找到最接近 target_point 的点"""
    if len(control_points) == 0:
        return target_point  # 避免 None，直接返回原点
    nodes = np.array(control_points)
    tree = KDTree(nodes)
    _, idx = tree.query(target_point)
    return tuple(nodes[idx])


def compute_shortest_path_length(graph, point1, point2):
    """计算 point1 和 point2 之间的最短路径长度"""
    if point1 in graph.nodes and point2 in graph.nodes:
        try:
            length = nx.shortest_path_length(graph, source=point1, target=point2, weight='weight')
            return length
        except nx.NetworkXNoPath:
            return float('inf')  # 无连通路径，返回无穷大
    return float('inf')


def visualize_paths(image, graph, path, points, color=(0, 255, 0), point_color=(0, 0, 255)):
    """在图像上绘制路径"""
    vis_image = image.copy()

    # 绘制路径
    for i in range(len(path) - 1):
        pt1, pt2 = path[i], path[i + 1]
        pt1, pt2 = (int(pt1[1]), int(pt1[0])), (int(pt2[1]), int(pt2[0]))  # 交换 x, y
        cv2.line(vis_image, pt1, pt2, color, thickness=2)

    # 绘制点
    for point in points:
        pt = (int(point[1]), int(point[0]))  # 交换 x, y
        cv2.circle(vis_image, pt, radius=4, color=point_color, thickness=-1)

    return vis_image


def compute_apls_and_visualize(graph_gt_path, graph_pred_path, image_path, output_folder, file_name, visualize=False):
    """计算 APLS，并可视化 GT & Prediction 路网、路径、控制点"""

    # 读取 GT & Prediction Graph
    with open(graph_gt_path, "rb") as f:
        graph_gt = pickle.load(f)
    with open(graph_pred_path, "rb") as f:
        graph_pred = pickle.load(f)

    if len(graph_gt.nodes) == 0:
        print(f"Skipping because GT graph is empty.")
        return

    if len(graph_pred.nodes) == 0:
        print(f"Setting APLS = 0 because Pred graph is empty.")
        return 0.0  # 设定 APLS = 0

    # 读取原始图片
    if visualize:
        os.makedirs(output_folder + '/' + 'apls_calc_visual_' + file_name, exist_ok=True)
        if not os.path.exists(image_path):
            print(f"Warning: Image {image_path} not found, skipping visualization...")
            return None
        image = cv2.imread(image_path)
        if image is None:
            print(f"Warning: Unable to read image {image_path}, skipping...")
            return

    # 选取端点和交叉口作为控制点
    control_points_gt = get_endpoints_and_intersections(graph_gt)
    control_points_pred = get_endpoints_and_intersections(graph_pred)

    if len(control_points_pred) == 0:
        print(f"Skipping {graph_pred_path} because no control points found in prediction.")
        return

    total_error = 0.0
    count = 0

    for i in range(len(control_points_gt)):
        for j in range(i + 1, len(control_points_gt)):
            A, B = control_points_gt[i], control_points_gt[j]

            # 在 Prediction 中找到对应的 A' 和 B'
            A_prime = find_nearest_control_point(control_points_pred, A)
            B_prime = find_nearest_control_point(control_points_pred, B)

            # 计算 GT 中的最短路径
            if nx.has_path(graph_gt, A, B):
                path_GT = nx.shortest_path(graph_gt, source=A, target=B, weight='weight')
                L_GT = len(path_GT) - 1  # 路径长度
            else:
                path_GT = []
                L_GT = float('inf')

            # 计算 Pred 中的最短路径
            if nx.has_path(graph_pred, A_prime, B_prime):
                path_Pred = nx.shortest_path(graph_pred, source=A_prime, target=B_prime, weight='weight')
                L_Pred = len(path_Pred) - 1  # 路径长度
            else:
                path_Pred = []
                L_Pred = float('inf')

            # 计算误差
            if L_GT == float('inf') and L_Pred == float('inf'):
                error = 0  # GT 和 Pred 都不可达，误差应为 0
            elif L_GT == float('inf') and L_Pred < float('inf'):
                error = 1  # GT 不可达但 Pred 可达，误差设为 1（因为 Pred 预测了一个错误的连接）
            elif L_Pred == float('inf') and L_GT < float('inf'):
                error = 1  # GT 可达但 Pred 不可达，误差设为 1（因为 Pred 断开了一个应该连接的路径）
            elif L_GT == 0:
                error = 1 if L_Pred > 0 else 0  # GT 路径长度为 0，预测路径不为 0 时误差最大
            else:
                error = min(1.0, abs(L_GT - L_Pred) / L_GT)  # 正常计算误差，确保误差不超过 1
            total_error += error
            count += 1

            # print(f"Pair {A} - {B} → {A_prime} - {B_prime}, L_GT: {L_GT}, L_Pred:, {L_Pred}, Error: {error:.4f}")

            # 生成可视化图像
            if visualize:
                vis_gt = visualize_paths(image, graph_gt, path_GT, [A, B], color=(0, 255, 0),
                                         point_color=(0, 0, 255))  # 绿线，红点
                vis_pred = visualize_paths(image, graph_pred, path_Pred, [A_prime, B_prime], color=(255, 255, 0),
                                           point_color=(255, 0, 0))  # 黄线，蓝点
                combined_vis = np.hstack([vis_gt, vis_pred])

                # 保存可视化结果
                output_vis_path = os.path.join(output_folder + '/' + 'apls_calc_visual_' + file_name, f"comparison_{i}_{j}.png")
                cv2.imwrite(output_vis_path, combined_vis)
                print(f"Saved visualization to {output_vis_path}")

    # 计算 APLS
    avg_apls = 1 - (total_error / count if count > 0 else 1)
    print(f"Final APLS for {file_name}: {avg_apls:.4f}")
    return avg_apls


def process_graphs_in_folder(gt_folder, pred_folder, image_folder, output_folder, visualize=False):
    """遍历文件夹，计算所有 graph 的 APLS，并计算平均 APLS"""
    os.makedirs(output_folder, exist_ok=True)

    graph_files = [f for f in os.listdir(gt_folder) if f.endswith(".gpickle")]

    total_apls = 0.0
    valid_count = 0
    image_apls = {}  # 存储每张图片的 APLS

    for graph_file in tqdm(graph_files, desc="Processing graphs", unit="file"):
        graph_gt_path = os.path.join(gt_folder, graph_file)
        graph_pred_path = os.path.join(pred_folder, graph_file)
        image_path = os.path.join(image_folder, graph_file.replace(".gpickle", ".png"))
        file_name = graph_file.split('.')[0]

        # 计算单张图片的 APLS
        apls_score = compute_apls_and_visualize(graph_gt_path, graph_pred_path, image_path, output_folder, file_name, visualize=visualize)

        if apls_score is not None:  # 只有 APLS 计算成功才计入平均值
            image_apls[graph_file] = apls_score
            total_apls += apls_score
            valid_count += 1

    # 计算平均 APLS
    avg_apls = total_apls / valid_count if valid_count > 0 else 0.0
    print(f"\n=== Final Results ===")
    print(f"Valid APLS computations: {valid_count}/{len(graph_files)}")
    print(f"Average APLS: {avg_apls:.4f}")

    # 保存 APLS 结果到文件
    output_results_path = os.path.join(output_folder, "apls_results.json")
    with open(output_results_path, "w") as f:
        json.dump(image_apls, f, indent=4)

    print(f"APLS results saved to {output_results_path}")


if __name__ == "__main__":
    # # 提取enschede road数据的真值路网
    # gt_annotation_file = "data/geethorn_road/annotations/test.json"
    # image_folder = "data/geethorn_road/test_images"
    # output_folder = "data/geethorn_road/test_road_network_graphs"
    # save_mask_folder = "data/geethorn_road/test_masks"
    # process_gt_mask_folder(gt_annotation_file, image_folder, output_folder, save_mask_folder)

    # ------------------------------------------------------------------------------------------------------------------

    parser = argparse.ArgumentParser(description="Process road network predictions and evaluate them.")
    parser.add_argument('--prediction_file', type=str, required=True, help='Path to the predicted polygon JSON file.')
    parser.add_argument('--gt_annotation_file', type=str, required=True, help='Path to the ground truth annotation JSON file.')
    parser.add_argument('--image_folder', type=str, required=True, help='Path to the folder containing test images.')
    parser.add_argument('--gt_folder', type=str, required=True, help='Path to the folder containing ground truth road network graphs.')
    parser.add_argument('--output_folder', type=str, required=True, help='Output folder to store processed graphs and visualizations.')
    parser.add_argument('--visualize', action='store_true', help='Whether to generate visualizations of road networks.')
    args = parser.parse_args()

    process_mask_folder(
        prediction_file=args.prediction_file,
        gt_annotation_file=args.gt_annotation_file,
        image_folder=args.image_folder,
        output_folder=args.output_folder
    )

    # 可选：可视化
    if args.visualize:
        visualize_road_networks_and_control_points(
            graph_folder_1=args.gt_folder,
            graph_folder_2=args.output_folder,
            image_folder=args.image_folder,
            output_folder=args.output_folder
        )

    # 运行批处理
    process_graphs_in_folder(
        gt_folder=args.gt_folder,
        pred_folder=args.output_folder,
        image_folder=args.image_folder,
        output_folder=args.output_folder,
        visualize=False
    )

    # 运行单张图片
    # file_name = "True_Ortho_2064_4734_patch_3"
    # graph_pred_path = os.path.join(pred_folder, f'{file_name}.gpickle')
    # graph_gt_path = os.path.join(gt_folder, f'{file_name}.gpickle')
    # image_path = os.path.join(image_folder, f'{file_name}.png')
    # # compute_apls_and_visualize(graph_gt_path, graph_pred_path, image_path, output_folder, file_name, visualize=True)

