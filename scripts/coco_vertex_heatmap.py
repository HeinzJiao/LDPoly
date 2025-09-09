import os
import json
import numpy as np
import cv2


# 计算 2D 高斯分布值
def gaussian_2d(x, y, x0, y0, sigma):
    return np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))


# 生成顶点热力图
def generate_heatmap(vertex_locations, heatmap_shape, sigma=2.5):
    heatmap = np.zeros(heatmap_shape, dtype=np.float32)

    if len(vertex_locations) == 0:
        return heatmap

    x = np.arange(heatmap_shape[1])  # width
    y = np.arange(heatmap_shape[0])  # height
    x, y = np.meshgrid(x, y)

    for loc in vertex_locations:
        heatmap = np.maximum(heatmap, gaussian_2d(x, y, loc[0], loc[1], sigma))

    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value

    heatmap[heatmap < 1e-8] = 0
    return heatmap


# 处理 COCO 格式的 JSON 文件并生成 heatmaps
def process_coco_annotations(coco_json_path, output_dir, image_size):
    os.makedirs(output_dir, exist_ok=True)

    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)

    # 获取图片信息
    image_id_to_info = {img['id']: img for img in coco_data['images']}

    # 读取所有 road polygon 并按图片 ID 分类
    image_polygons = {img['id']: [] for img in coco_data['images']}

    for ann in coco_data['annotations']:
        image_id = ann['image_id']
        for polygon in ann['segmentation']:  # 多个多边形
            rounded_vertices = np.array(polygon).reshape(-1, 2)
            rounded_vertices = np.rint(rounded_vertices).astype(int)  # 四舍五入并转为整数
            image_polygons[image_id].extend(rounded_vertices.tolist())

    # 生成 heatmap 并保存
    for img_id, vertices in image_polygons.items():
        img_info = image_id_to_info[img_id]
        width, height = img_info['width'], img_info['height']

        # 确保所有顶点都在图片范围内
        valid_vertices = [[max(0, min(x, width - 1)), max(0, min(y, height - 1))] for x, y in vertices]

        # 生成 heatmap
        heatmap = generate_heatmap(valid_vertices, (height, width))
        print("heatmap.shape: ", heatmap.shape, "max(): ", heatmap.max(), "min(): ", heatmap.min())

        # 确保数值范围在 0-1
        heatmap = np.clip(heatmap, 0, 1)

        # 保存为 numpy 数组
        heatmap_path = os.path.join(output_dir,
                                    f"{img_info['file_name'].replace('.jpg', '.npy').replace('.png', '.npy')}")
        np.save(heatmap_path, heatmap)
        print(f"Saved heatmap: {heatmap_path}")

    print("All heatmaps generated successfully.")



def visualize_and_save_gray_heatmap(heatmap_dir, output_dir):
    """
    读取 .npy 格式的 heatmap，转换为灰度图，并保存为 .png 图片。

    :param heatmap_dir: 存放 heatmap .npy 文件的目录
    :param output_dir: 生成的可视化图片的存放目录
    """
    os.makedirs(output_dir, exist_ok=True)

    heatmap_files = [f for f in os.listdir(heatmap_dir) if f.endswith(".npy")]

    for heatmap_file in heatmap_files:
        heatmap_path = os.path.join(heatmap_dir, heatmap_file)
        save_path = os.path.join(output_dir, heatmap_file.replace(".npy", ".png"))

        # 读取 heatmap
        heatmap = np.load(heatmap_path)

        # 确保数值范围在 0-1
        # heatmap = np.clip(heatmap, 0, 1)

        # 归一化到 0-255 (uint8) 以便保存为灰度图
        heatmap_gray = (heatmap * 255).astype(np.uint8)

        # 保存灰度图
        cv2.imwrite(save_path, heatmap_gray)

        print(f"Saved grayscale heatmap: {save_path}")

    print("All grayscale heatmaps saved successfully.")

from extract_vertices_from_scaled_heatmap import extract_vertices_from_heatmap

def visualize_heatmap_with_vertices(heatmap_path, output_path, th=0.1, upscale_factor=4):
    heatmap = np.load(heatmap_path)

    # 提取顶点
    vertices, scores = extract_vertices_from_heatmap(heatmap, th=th, upscale_factor=upscale_factor)

    # 转成 0-255 灰度图用于显示
    heatmap_gray = (heatmap * 255).astype(np.uint8)
    heatmap_rgb = cv2.cvtColor(heatmap_gray, cv2.COLOR_GRAY2BGR)

    # 画顶点
    for (x, y) in vertices:
        cv2.circle(heatmap_rgb, (int(round(x)), int(round(y))), radius=2, color=(0, 0, 255), thickness=-1)

    # 保存图像
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, heatmap_rgb)
    print(f"Saved heatmap with vertices: {output_path}")


# 运行脚本示例
if __name__ == "__main__":
    coco_json_path = "data/vaihingen_map_generalization/unsplit/geb15/annotations_geb15_merged_clean_filtered.json"  # COCO 数据集路径
    output_dir = "data/vaihingen_map_generalization/unsplit/geb15/geb15_heatmaps_sigma2.5"  # Heatmap 保存路径
    process_coco_annotations(coco_json_path, output_dir, image_size=(128, 128))  # image_size 只是占位，不使用

    # # heatmap_dir = "data/vaihingen_map_generalization/geb_heatmaps_sigma2.5"  # Heatmap 文件夹路径
    # # output_dir = "data/vaihingen_map_generalization/geb_heatmaps_sigma2.5_visual"  # 保存可视化的路径
    # # visualize_and_save_gray_heatmap(heatmap_dir, output_dir)
    #
    # heatmap_path = "data/vaihingen_map_generalization/geb_heatmaps_sigma5/0_75.npy"
    # output_path = "data/vaihingen_map_generalization/geb_sigma5_th0.9_0_75_vertices.png"
    # visualize_heatmap_with_vertices(heatmap_path, output_path, th=0.9, upscale_factor=4)
