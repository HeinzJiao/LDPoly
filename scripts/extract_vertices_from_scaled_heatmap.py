"""
Script for extracting vertices from heatmaps and saving the results.

This script processes predicted heatmaps from a model, applies Non-Maximum Suppression (NMS) to extract building vertices,
and then filters the vertices based on a confidence threshold.

Usage:
    python extract_vertices.py --outputs_dir <output_directory> \
                               --image_dir <path_to_test_images> \
                               --th <confidence_threshold> \
                               --sampler <sampling_method>

Arguments:
    --outputs_dir   Directory where the results (vertices and visualizations) will be saved.
    --image_dir     Directory where the test images are located.
    --th            Confidence threshold for filtering valid vertices from heatmaps.
    --sampler       Sampling method used during testing: 'direct', 'ddim', or 'ddpm'.

Example:
    PYTHONPATH=./:$PYTHONPATH python -u scripts/extract_vertices_from_scaled_heatmap.py \
        --outputs_dir ./outputs/vaihingen_map_generalization_sigma2.5_geb15/epoch=epoch=59 \
        --image_dir ./data/vaihingen_map_generalization/geb15/val/geb_masks \
        --th 0.1 \
        --upscale_factor 1 \
        --kernel_size 3

    PYTHONPATH=./:$PYTHONPATH python -u scripts/extract_vertices_from_scaled_heatmap.py \
        --outputs_dir ./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79 \
        --image_dir ./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_input.png \
        --th 0.5 \
        --sampler ddim \
        --upscale_factor 1 \
        --kernel_size 5
"""
import os
import json
import numpy as np
import argparse
import torch
import torch.nn.functional as F


def extract_vertices_from_heatmap(heatmap, th, kernel_size, topk=300, upscale_factor=1):
    # 放大 heatmap
    heatmap = torch.tensor(heatmap).unsqueeze(0).unsqueeze(0).to(dtype=torch.float32)  # 增加 batch 和 channel 维度
    heatmap = F.interpolate(heatmap, scale_factor=upscale_factor, mode='bilinear', align_corners=False)
    heatmap = heatmap.squeeze(0)  # 恢复到单通道

    # NMS
    heatmap_nms = non_maximum_suppression(heatmap, kernel_size)  # torch.Size([H * upscale_factor, W * upscale_factor])
    height, width = heatmap_nms.size(0), heatmap_nms.size(1)
    heatmap_nms = heatmap_nms.reshape(-1)

    # Top-K
    scores, index = torch.topk(heatmap_nms, k=topk)  # scores: torch.Size([topk])
    y = (index // width).float()
    x = (index % width).float()

    # 恢复到原始分辨率的坐标
    x /= upscale_factor
    y /= upscale_factor

    # 提取顶点
    extracted_vertices = torch.stack((x, y)).t()  # torch.Size([topk, 2])
    return np.array(extracted_vertices[scores > th]), np.array(scores[scores > th])


def non_maximum_suppression(a, kernel_size=3):
    """
    对输入张量 a 做局部最大值抑制（NMS）。
    kernel_size 必须是奇数，以保证 padding 对称。

    Args:
        a (Tensor): 形状 [N, C, H, W] 的输入热图。
        kernel_size (int): 最大值池化的窗口大小，建议为奇数。

    Returns:
        Tensor: 同 a 形状，相当于只保留局部最大值的位置，其它位置设为 0。
    """
    assert kernel_size % 2 == 1, "kernel_size 必须为奇数"
    kernel_size = int(kernel_size)
    pad = int(kernel_size // 2)

    # 先做最大值池化，得到每个位置上的局部最大值
    ap = F.max_pool2d(a, kernel_size, stride=1, padding=pad)

    # 取出那些等于局部最大值的位置，生成 mask
    mask = (a == ap).float()

    # 只保留局部最大值处的分数，其它位置置 0
    return a * mask


def extract_vertices_from_heatmap_folder(heatmaps_folder, image_dir, th, kernel_size, output_json_file, upscale_factor,
                                         visualize_vertices_dir=None):
    results = []

    # 遍历文件夹中的所有文件
    for file_name in os.listdir(heatmaps_folder):
        file_path = os.path.join(heatmaps_folder, file_name)

        # 加载heatmap张量
        h_prediction = np.load(file_path)

        # 提取顶点
        vertices, scores = extract_vertices_from_heatmap(h_prediction, th, kernel_size, topk=300,
                                                         upscale_factor=upscale_factor)

        # 可视化顶点
        # image_name = file_name.split('.')[0] + '.png'
        # image_path = os.path.join(image_dir, image_name)
        # image = cv2.imread(image_path)
        # for point in vertices:
        #     cv2.circle(image, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
        # cv2.imwrite(os.path.join(visualize_vertices_dir, image_name), image)

        # 将数据保存到字典中
        result = {
            "image_file_name": file_name,
            "extracted_vertices": vertices.tolist()
        }
        results.append(result)

    with open(output_json_file, 'w') as json_file:
        json.dump(results, json_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Extract vertices from heatmap with a given threshold and save results.')

    # Add argument for output directory path
    parser.add_argument('--outputs_dir', type=str, required=True, help='Directory path to save output results.')

    # Add argument for image directory path (test images location)
    parser.add_argument('--image_dir', type=str, required=True, help='Directory path of test images.')

    # Add argument for confidence threshold used to filter valid vertices
    parser.add_argument('--th', type=float, required=True, help='Confidence threshold for filtering vertices.')

    # Add argument for confidence threshold used to filter valid vertices
    parser.add_argument('--kernel_size', type=float, required=True, help='NMS kernel size.')

    # Add argument for upscale factor used to scale up the vertex heatmap
    parser.add_argument('--upscale_factor', type=float, required=True, help='Upscale factor to scale up the vertex heatmap.')

    # Add argument for the sampling method used during testing (can be direct, ddim, or ddpm)
    parser.add_argument('--sampler', type=str, choices=['direct', 'ddim', 'ddpm'], required=True,
                        help='Sampling method used during testing (direct, ddim, or ddpm).')

    args = parser.parse_args()

    outputs_dir = args.outputs_dir
    image_dir = args.image_dir
    th = args.th
    sampler = args.sampler
    kernel_size = args.kernel_size

    heatmaps_folder = os.path.join(outputs_dir, f'samples_heat_{sampler}_npy')
    output_json_file = os.path.join(outputs_dir, f"output_vertices_from_scaled{int(args.upscale_factor)}_"
                                                 f"heatmap_{sampler}_th-{th}_k-{kernel_size}.json")
    # visualize_vertices_dir = os.path.join(outputs_dir, f'visualize_output_vertices_{sampler}_th-{th}')

    # Uncomment the following line if you want to create the directory for visualized output
    # os.makedirs(visualize_vertices_dir, exist_ok=True)

    extract_vertices_from_heatmap_folder(heatmaps_folder, image_dir, th, kernel_size, output_json_file,
                                         upscale_factor=args.upscale_factor)

