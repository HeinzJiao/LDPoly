import os
import json
import cv2
import numpy as np

import cv2
import numpy as np
from matplotlib.patches import draw_bbox

import cv2
import numpy as np


def draw_polygons_and_bboxes_on_image(image, annotations, line_color=(0, 255, 0), vertex_color=(0, 0, 255),
                                      vertex_thickness=3, bbox_color=(255, 0, 0), draw_polygon=True, draw_bbox=False,
                                      draw_masked_polygon=False, polygon_fill_color=(0, 255, 0), alpha=0.1):
    """
    在图像上绘制建筑物的多边形、顶点和 bounding box (bbox)，并支持绘制半透明的 polygon 填充。

    参数:
    - image: 读取的图像。
    - annotations: 包含建筑物多边形和 bbox 的列表。
    - line_color: 多边形边缘的颜色，默认为绿色。
    - vertex_color: 顶点的颜色，默认为红色。
    - vertex_thickness: 顶点的粗细，默认为5。
    - bbox_color: bbox 的颜色，默认为蓝色。
    - draw_polygon: 是否绘制多边形边框，默认为 True。
    - draw_bbox: 是否绘制 bbox，默认为 False。
    - draw_masked_polygon: 是否绘制半透明填充的 polygon，默认为 False。
    - polygon_fill_color: 填充 polygon 的颜色（RGBA 格式，A 为透明度），默认为半透明绿色 (0, 255, 0, 128)。
    """
    overlay = np.zeros_like(image, dtype=np.uint8)

    for annotation in annotations:
        if draw_polygon or draw_masked_polygon:
            # 提取 segmentation 中的外部轮廓和内部孔洞
            polygons = [np.array(seg, np.int32).reshape((-1, 2)) for seg in annotation['segmentation']]
            if len(polygons) == 0:
                continue

            external_polygon = polygons[0]  # 第一个是外部轮廓
            hole_polygons = polygons[1:]  # 后续是内部孔洞

            if draw_masked_polygon:
                # 在 overlay 上绘制外部轮廓填充颜色
                cv2.fillPoly(overlay, [external_polygon], polygon_fill_color)
                # 在孔洞区域绘制黑色以清空填充
                for hole in hole_polygons:
                    cv2.fillPoly(overlay, [hole], (0, 0, 0))

            if draw_polygon:
                # 绘制多边形的轮廓
                cv2.polylines(image, [external_polygon], isClosed=True, color=line_color, thickness=1)
                for hole in hole_polygons:
                    cv2.polylines(image, [hole], isClosed=True, color=line_color, thickness=1)

                # 绘制顶点
                for vertex in external_polygon:
                    cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)
                for hole in hole_polygons:
                    for vertex in hole:
                        cv2.circle(image, tuple(vertex), vertex_thickness, vertex_color, -1)

        if draw_bbox:
            # 绘制 bbox，annotation['bbox'] 格式是 [x, y, width, height]
            x, y, w, h = annotation['bbox']
            top_left = (int(x), int(y))
            bottom_right = (int(x + w), int(y + h))
            cv2.rectangle(image, top_left, bottom_right, bbox_color, 2)

    # 仅在多边形区域叠加半透明效果
    if draw_masked_polygon:
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    return image


def process_annotations(coco_json_file, png_folder, output_folder, mask_output_folder):
    """
    读取 COCO 格式的 annotations.json 文件，将所有建筑物多边形和 bbox 绘制到对应图片上。

    参数:
    - coco_json_file: COCO 格式的 annotations.json 文件路径。
    - png_folder: 存储原始 PNG 图片的文件夹路径。
    - output_folder: 保存带有建筑物轮廓和 bbox 的图片的输出文件夹路径。
    """
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    # if not os.path.exists(mask_output_folder):
    #     os.makedirs(mask_output_folder)

    # 读取 COCO annotations.json 文件
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    # 遍历所有图像
    for image_info in coco_data['images']:
        image_id = image_info['id']
        file_name = image_info['file_name']
        # if file_name != "True_Ortho_2082_4767.png":
        #     continue
        # 拼接 PNG 图片的完整路径
        image_path = os.path.join(png_folder, file_name)

        # 检查图像文件是否存在
        if not os.path.exists(image_path):
            print(f"图像文件不存在: {image_path}")
            continue

        # 读取图像
        image = cv2.imread(image_path)
        height, width, _ = image.shape

        # 创建一个与原始图像相同大小的黑色掩码图像
        mask = np.zeros((height, width), dtype=np.uint8)

        # 收集所有与该图像相关的多边形和 bbox
        annotations = []
        for annotation in coco_data['annotations']:
            if annotation['image_id'] == image_id:
                # 获取 segmentation 信息和 bbox 信息
                annotations.append({
                    'segmentation': annotation['segmentation'],  # [[x1, y1, x2, y2, ...], [...], [...]]
                    'bbox': annotation['bbox']  # bbox 是 [x, y, width, height]
                })

        # 在图像上绘制所有多边形和 bbox
        image_with_polygons_and_bboxes = draw_polygons_and_bboxes_on_image(image, annotations, draw_polygon=True,
                                                                           draw_bbox=False, draw_masked_polygon=True)

        # # 绘制建筑物的多边形到掩码图像上
        # for annotation in annotations:
        #     segmentation = annotation['segmentation']
        #     # 外部轮廓（填充为白色）
        #     exterior = np.array(segmentation[0]).reshape((-1, 2)).astype(np.int32)
        #     cv2.fillPoly(mask, [exterior], 255)  # 填充外围轮廓为白色
        #     if len(segmentation) > 1:
        #         # 内部轮廓（填充为黑色）
        #         for interior in segmentation[1:]:
        #             interior_points = np.array(interior).reshape((-1, 2)).astype(np.int32)
        #             cv2.fillPoly(mask, [interior_points], 0)  # 填充内部轮廓为黑色

        # 保存结果到输出文件夹
        output_image_path = os.path.join(output_folder, file_name)
        # image_with_polygons_and_bboxes = cv2.resize(image_with_polygons_and_bboxes, (1000, 1000))
        cv2.imwrite(output_image_path, image_with_polygons_and_bboxes)
        print(f"保存带有多边形轮廓的图像: {output_image_path}")

        # # 保存掩码图像到掩码文件夹
        # mask_output_path = os.path.join(mask_output_folder, file_name)
        # cv2.imwrite(mask_output_path, mask)
        # print(f"保存建筑物掩码图像: {mask_output_path}")


if __name__ == "__main__":
    coco_json_file = './data/vaihingen_map_generalization/geb15/geb15_val_annotations.json'  # COCO 格式的 annotations 文件路径
    png_folder = './data/vaihingen_map_generalization/geb15/val/geb15_masks'  # 存储 PNG 图像的文件夹
    output_folder = './data/vaihingen_map_generalization/geb15/geb15_val_viz'  # 输出带有多边形轮廓的图像的文件夹
    mask_output_folder = './'

    process_annotations(coco_json_file, png_folder, output_folder, mask_output_folder)
