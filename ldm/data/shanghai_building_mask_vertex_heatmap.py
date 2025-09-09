import os
import numpy as np
import PIL
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2
from pycocotools.coco import COCO
import torchvision.transforms.functional as F
import random

def gaussian_2d(x, y, x0, y0, sigma):
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


# 通过这种方法，生成的热力图将包含多个高斯分布的“山峰”，每个顶点的位置具有相同的峰值，并且避免了矩形边框的出现。
def generate_heatmap(vertex_locations, heatmap_shape, sigma=5):
    # 生成空的热力图
    heatmap = np.zeros(heatmap_shape)

    if len(vertex_locations) == 0:
        return heatmap

    # 设置高斯核的标准差
    # sigma 控制高斯分布的宽度，即中心的“发亮”区域的扩展程度。
    # 较小的 sigma: 高斯分布的峰值尖锐，周边区域的亮度快速下降，形成较小的“发亮圆圈”。
    # 较大的 sigma: 高斯分布的峰值较为平坦，周边区域的亮度缓慢下降，形成较大的“发亮圆圈”。

    # 生成x和y的网格
    x = np.arange(heatmap_shape[1])  # image_info['width']
    y = np.arange(heatmap_shape[0])  # image_info['height']
    x, y = np.meshgrid(x, y)

    # 对于每个顶点，计算其在整个图像上的高斯分布，并在相应位置取最大值而不是相加，以确保所有顶点的位置在模糊后都能保持峰值为1。
    for loc in vertex_locations:
        heatmap = np.maximum(heatmap, gaussian_2d(x, y, loc[0], loc[1], sigma))

    # 归一化处理，将热力图的值缩放到 [0, 1] 范围
    # 归一化处理
    max_value = np.max(heatmap)
    if max_value > 0:
        heatmap = heatmap / max_value

    # 将接近零的值设为零
    heatmap[heatmap < 1e-8] = 0

    return heatmap


class ShanghaiBase(Dataset):
    def __init__(self, coco_annotation_file, image_dir, mask_dir, mode, transforms=None, size=256, num_classes=2):
        self.coco = COCO(coco_annotation_file)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms  # ['hflip', 'vflip']
        self.mode = mode
        self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_info = self.coco.loadImgs(image_id)[0]
        self.filename = image_info['file_name']
        image_path = os.path.join(self.image_dir, image_info['file_name'])
        mask_path = os.path.join(self.mask_dir, image_info['file_name'])

        # load image
        image = Image.fromarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

        # load mask
        segmentation = Image.fromarray(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB))

        ann_ids = self.coco.getAnnIds(imgIds=image_id)  # if no object, ann_ids=[]
        anns = self.coco.loadAnns(ann_ids)  # if no object, anns=[]

        # load vertices
        vertex_locations = []
        if len(anns) > 0:
            for ann in anns:
                if 'segmentation' in ann:
                    for seg in ann['segmentation']:
                        seg = np.array(seg).reshape(-1, 2)
                        vertex_locations.extend(seg)

        # data augmentation
        if self.mode == 'train' and self.transforms:
            # 数据增强；四舍五入并确保坐标在有效范围内
            # 如果图片不包含建筑物，即vertex_locations为空，则直接返回值为空列表的vertex_locations
            segmentation, image, vertex_locations = self.apply_transform(segmentation, image, vertex_locations)
        else:
            width, height = image.size
            # 四舍五入并确保坐标在有效范围内
            if len(vertex_locations) > 0:
                vertex_locations = [
                    (min(max(round(x), 0), width - 1), min(max(round(y), 0), height - 1))
                    for (x, y) in vertex_locations
                ]

        segmentation = (np.array(segmentation) > 128).astype(np.float32)

        if len(vertex_locations) > 0:
            heatmap = generate_heatmap(vertex_locations, (image_info['height'], image_info['width']), sigma=5)
        else:  # 如果图片不包含建筑物，则heatmap为全黑。
            heatmap = np.zeros((image_info['height'], image_info['width']))

        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.  # range from -1 to 1, np.float32

        if self.mode != 'test':
            heatmap = (heatmap * 2) - 1  # range from -1 to 1, np.float32
            segmentation = (segmentation * 2) - 1  # range from -1 to 1, np.float32
        heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)  # 变成三通道
        heatmap = heatmap.astype(np.float32)

        if self.mode != 'test':
            example = {
                "file_path_": image_path,
                "image": image,  # range from -1 to 1
                "heatmap": heatmap,  # range from -1 to 1
                "class_id": np.array([-1]),  # doesn't matter for binary seg
                "segmentation": segmentation  # range: binary -1 and 1
            }
        else:
            example = {
                "file_path_": image_path,
                "image": image,  # range from -1 to 1
                "heatmap": heatmap,  # range from 0 to 1
                "class_id": np.array([-1]),  # doesn't matter for binary seg
                "segmentation": segmentation,  # binary 0 or 1
                "vertex_locations": np.array(vertex_locations, dtype=np.int32)  # Nx2 or (0,)
            }

        assert np.max(heatmap) <= 1. and np.min(heatmap) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        return example

    def apply_transform(self, segmentation, image, vertex_locations):
        # vertex_locations: (N, 2)
        width, height = image.size

        for transform in self.transforms:
            if transform == 'hflip' and np.random.rand() > 0.5:
                segmentation = F.hflip(segmentation)
                image = F.hflip(image)
                if len(vertex_locations) > 0:
                    vertex_locations = [(width - x, y) for (x, y) in vertex_locations]
            elif transform == 'vflip' and np.random.rand() > 0.5:
                segmentation = F.vflip(segmentation)
                image = F.vflip(image)
                if len(vertex_locations) > 0:
                    vertex_locations = [(x, height - y) for (x, y) in vertex_locations]
            elif transform == 'rotate' and np.random.rand() > 0.5:
                angle = random.choice([90, 180, 270])
                segmentation = F.rotate(segmentation, angle)
                image = F.rotate(image, angle)

                if len(vertex_locations) > 0:
                    # 处理逆时针旋转的顶点更新
                    if angle == 90:
                        vertex_locations = [(y, width - x) for (x, y) in vertex_locations]
                        width, height = height, width
                    elif angle == 180:
                        vertex_locations = [(width - x, height - y) for (x, y) in vertex_locations]
                    elif angle == 270:
                        vertex_locations = [(height - y, x) for (x, y) in vertex_locations]
                        width, height = height, width

        # 四舍五入并确保坐标在有效范围内
        if len(vertex_locations) > 0:
            vertex_locations = [
                (min(max(round(x), 0), width - 1), min(max(round(y), 0), height - 1))
                for (x, y) in vertex_locations
            ]

        # self.visualize(segmentation, image, vertex_locations, filename=self.filename)

        return segmentation, image, vertex_locations


    # def visualize(self, segmentation, image, vertex_locations, filename, save_path="visualize_augmented"):
    #     """
    #         可视化segmentation, image和vertex locations以检查数据增强后的效果，并保存结果到本地
    #         """
    #     segmentation = np.array(segmentation)
    #     image = np.array(image)
    #
    #     for (x, y) in vertex_locations:
    #         cv2.circle(image, (int(x), int(y)), radius=2, color=(0, 0, 255), thickness=-1)
    #
    #     combined = np.concatenate((image, segmentation), axis=1)  # 水平拼接
    #     combined_image = Image.fromarray(combined)
    #
    #     if save_path is not None:
    #         os.makedirs(save_path, exist_ok=True)
    #         combined_filename = os.path.join(save_path, filename)
    #         combined_image.save(combined_filename)
    #         print(f"Saved augmented image to {combined_filename}")

# deventer road
class DeventerRoadTrain(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(coco_annotation_file="data/deventer_road/annotations/train.json",
                         image_dir="data/deventer_road/train_images",
                         mask_dir="data/deventer_road/train_masks",
                         mode="train",
                         transforms=['hflip', 'vflip', 'rotate'], **kwargs)

class DeventerRoadValidation(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(coco_annotation_file="data/deventer_road/annotations/test.json",
                         image_dir="data/deventer_road/test_images",
                         mask_dir="data/deventer_road/test_masks",
                         transforms=[], mode="val", **kwargs)

class DeventerRoadTest(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(coco_annotation_file="data/deventer_road/annotations/test.json",
                         image_dir="data/deventer_road/test_images",
                         mask_dir="data/deventer_road/test_masks",
                         transforms=[], mode="test", **kwargs)

# test on enschede
class EnschedeRoadTest(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(coco_annotation_file="data/enschede_road/annotations/test.json",
                         image_dir="data/enschede_road/test_images",
                         mask_dir="data/enschede_road/test_masks",
                         transforms=[], mode="test", **kwargs)

# test on geethorn
class GeethornRoadTest(ShanghaiBase):
    def __init__(self, **kwargs):
        super().__init__(coco_annotation_file="data/geethorn_road/annotations/test.json",
                         image_dir="data/geethorn_road/test_images",
                         mask_dir="data/geethorn_road/test_masks",
                         transforms=[], mode="test", **kwargs)