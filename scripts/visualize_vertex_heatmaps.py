import os
import numpy as np
import cv2
from tqdm import tqdm

def visualize_heatmaps(input_dir, output_dir1, cmap='gray'):
    """
    遍历输入文件夹中的所有顶点 heatmap（npy），将其可视化为灰度图，并保存到两个目标文件夹中。

    Args:
        input_dir (str): 存放 heatmap 的文件夹路径，格式为 .npy，每个 shape = (H, W)，值域为 [0, 1]
        output_dir1 (str): 第一个输出文件夹路径（将保存可视化灰度图）
        output_dir2 (str): 第二个输出文件夹路径（同上）
        cmap (str): 用于可视化的 colormap，默认灰度
    """

    os.makedirs(output_dir1, exist_ok=True)

    heatmap_files = [f for f in os.listdir(input_dir) if f.endswith('.npy')]

    for fname in tqdm(heatmap_files, desc="Visualizing heatmaps"):
        path = os.path.join(input_dir, fname)
        heatmap = np.load(path)

        # 将 0~1 的 heatmap 映射到 0~255 的灰度图
        heatmap_uint8 = (heatmap * 255).clip(0, 255).astype(np.uint8)

        # 保存为灰度图像
        out_path1 = os.path.join(output_dir1, fname.replace('.npy', '.png'))
        cv2.imwrite(out_path1, heatmap_uint8)

    print(f"共处理 {len(heatmap_files)} 个 heatmap 文件，已保存到：\n{output_dir1}")

if __name__ == "__main__":
    input_dir = "./data/vaihingen_map_generalization/unsplit/geb15/geb_heatmaps_sigma2.5_geb15"
    output_dir1 = "./data/vaihingen_map_generalization/unsplit/geb15/geb_heatmaps_sigma2.5_geb15_png"

    visualize_heatmaps(input_dir, output_dir1)
