import os
import numpy as np
import PIL
import torch
from PIL import Image, ImageEnhance
from torch.utils.data import Dataset
from torchvision import transforms
import glob
import cv2
import torchvision.transforms.functional as TF
import random
from pycocotools.coco import COCO

class VaihingenBase(Dataset):
    def __init__(self,
                 raw_mask_dir,  # 原始（未简化）mask，condition
                 raw_heatmap_dir,  # 原始（未简化）heatmap，condition
                 mask_dir,  # 简化后的mask，target
                 heatmap_dir,  # 简化后的heatmap，target
                 mode,
                 # coco_annotation_file=None,
                 augment=[],
                 size=256, num_classes=2):

        self.raw_mask_dir = raw_mask_dir
        self.raw_heatmap_dir = raw_heatmap_dir
        self.mask_dir = mask_dir
        self.heatmap_dir = heatmap_dir
        self.mode = mode

        self.image_filenames = os.listdir(raw_mask_dir)  # 基于原始 mask 的文件名
        self.augment = augment

        # self.coco_annotation_file = coco_annotation_file
        # if coco_annotation_file:
        #     self.coco = COCO(coco_annotation_file)
        #     self.image_ids = list(self.coco.imgs.keys())

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_filename = self.image_filenames[idx]
        raw_mask_path = os.path.join(self.raw_mask_dir, img_filename)
        raw_heatmap_path = os.path.join(self.raw_heatmap_dir, img_filename.replace(".jpg", ".npy").replace(".png", ".npy"))
        mask_path = os.path.join(self.mask_dir, img_filename)
        heatmap_path = os.path.join(self.heatmap_dir, img_filename.replace(".jpg", ".npy").replace(".png", ".npy"))

        # 加载为 numpy 数组
        raw_mask = np.array(Image.open(raw_mask_path), dtype=np.float32)  # [H, W], uint8 -> float32
        raw_heatmap = np.load(raw_heatmap_path).astype(np.float32)  # [H, W], float32
        width, height = raw_heatmap.shape

        mask = np.array(Image.open(mask_path), dtype=np.float32)  # [H, W], uint8 -> float32
        heatmap = np.load(heatmap_path).astype(np.float32)  # [H, W], float32

        # 数据增强（对 numpy array）
        if 'hflip' in self.augment and random.random() > 0.5:
            raw_mask = np.fliplr(raw_mask)
            raw_heatmap = np.fliplr(raw_heatmap)
            mask = np.fliplr(mask)
            heatmap = np.fliplr(heatmap)

        if 'vflip' in self.augment and random.random() > 0.5:
            raw_mask = np.flipud(raw_mask)
            raw_heatmap = np.flipud(raw_heatmap)
            mask = np.flipud(mask)
            heatmap = np.flipud(heatmap)

        if 'rotate' in self.augment and random.random() > 0.5:
            k = random.choice([0, 1, 2, 3])  # 0, 90, 180, 270 度
            raw_mask = np.rot90(raw_mask, k)
            raw_heatmap = np.rot90(raw_heatmap, k)
            mask = np.rot90(mask, k)
            heatmap = np.rot90(heatmap, k)

        # 归一化处理
        raw_mask = raw_mask / 255.0  # [0, 1]
        raw_mask = raw_mask * 2.0 - 1.0  # [-1, 1]

        raw_heatmap = raw_heatmap * 2.0 - 1.0  # [-1, 1]

        mask = mask / 255.0  # [0, 1]
        if self.mode != 'test':
            segmentation = mask * 2.0 - 1.0  # [-1, 1]，用于训练
            heatmap = heatmap * 2.0 - 1.0  # [-1, 1]
        else:
            segmentation = mask  # [0, 1]
            # heatmap 保持 [0, 1]

        # shape: H x W -> H x W x 3
        segmentation = np.repeat(segmentation[:, :, np.newaxis], 3, axis=2)
        heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)
        raw_mask = np.repeat(raw_mask[:, :, np.newaxis], 3, axis=2)
        raw_heatmap = np.repeat(raw_heatmap[:, :, np.newaxis], 3, axis=2)

        # # Load polygon vertices from COCO annotations during testing
        # vertex_locations = []
        # if self.coco_annotation_file:
        #     image_id = self.image_ids[idx]
        #     ann_ids = self.coco.getAnnIds(imgIds=image_id)  # if no object, ann_ids=[]
        #     anns = self.coco.loadAnns(ann_ids)  # if no object, anns=[]
        #     # load vertices
        #     if len(anns) > 0:
        #         for ann in anns:
        #             if 'segmentation' in ann:
        #                 for seg in ann['segmentation']:
        #                     seg = np.array(seg).reshape(-1, 2)
        #                     vertex_locations.extend(seg)
        #     vertex_locations = [
        #             (min(max(x, 0), width - 1), min(max(y, 0), height - 1))
        #             for (x, y) in vertex_locations
        #         ]

        if self.mode != "test":
            example = {
                "file_path_": img_filename,
                "raw_mask": raw_mask,  # [-1, 1]
                "raw_heatmap": raw_heatmap,  # [-1, 1]
                "class_id": np.array([-1], dtype=np.int32),
                "segmentation": segmentation,  # [-1, 1]
                "heatmap": heatmap  # [-1, 1]
            }
        else:
            example = {
                "file_path_": img_filename,
                "raw_mask": raw_mask,  # [-1, 1]
                "raw_heatmap": raw_heatmap,  # [-1, 1]
                "class_id": np.array([-1], dtype=np.int32),
                "segmentation": segmentation,  # [0, 1]
                "heatmap": heatmap,  # [0, 1]
                # "vertex_locations": np.array(vertex_locations, dtype=np.float32)
            }

        # 检查范围
        assert np.max(example["raw_mask"]) <= 1. and np.min(example["raw_mask"]) >= -1.
        assert np.max(example["raw_heatmap"]) <= 1. and np.min(example["raw_heatmap"]) >= -1.

        return example

class Geb10Train(VaihingenBase):
    def __init__(self, **kwargs):
        super().__init__(raw_mask_dir="data/vaihingen_map_generalization/geb10/train/geb_masks",  # condition
                         raw_heatmap_dir="data/vaihingen_map_generalization/geb10/train/geb_heatmaps",  # condition
                         mask_dir="data/vaihingen_map_generalization/geb10/train/geb10_masks",  # gt
                         heatmap_dir="data/vaihingen_map_generalization/geb10/train/geb10_heatmaps",  # gt
                         mode="train",
                         augment=['hflip', 'vflip', 'rotate'], **kwargs)


class Geb10Validation(VaihingenBase):
    def __init__(self, **kwargs):
        super().__init__(raw_mask_dir="data/vaihingen_map_generalization/geb10/val/geb_masks",
                         raw_heatmap_dir="data/vaihingen_map_generalization/geb10/val/geb_heatmaps",
                         mask_dir="data/vaihingen_map_generalization/geb10/val/geb10_masks",
                         heatmap_dir="data/vaihingen_map_generalization/geb10/val/geb10_heatmaps",
                         mode="val",
                         augment=[],
                         **kwargs)


class Geb10Test(VaihingenBase):
    def __init__(self, **kwargs):
        super().__init__(raw_mask_dir="data/vaihingen_map_generalization/geb10/val/geb_masks",
                         raw_heatmap_dir="data/vaihingen_map_generalization/geb10/val/geb_heatmaps",
                         mask_dir="data/vaihingen_map_generalization/geb10/val/geb10_masks",
                         heatmap_dir="data/vaihingen_map_generalization/geb10/val/geb10_heatmaps",
                         mode="test",
                         augment=[], **kwargs)

# --------------------

class Geb15Train(VaihingenBase):
    def __init__(self, **kwargs):
        super().__init__(raw_mask_dir="data/vaihingen_map_generalization/geb15/train/geb_masks",  # condition
                         raw_heatmap_dir="data/vaihingen_map_generalization/geb15/train/geb_heatmaps",  # condition
                         mask_dir="data/vaihingen_map_generalization/geb15/train/geb15_masks",  # gt
                         heatmap_dir="data/vaihingen_map_generalization/geb15/train/geb15_heatmaps",  # gt
                         mode="train",
                         augment=['hflip', 'vflip', 'rotate'], **kwargs)


class Geb15Validation(VaihingenBase):
    def __init__(self, **kwargs):
        super().__init__(raw_mask_dir="data/vaihingen_map_generalization/geb15/val/geb_masks",
                         raw_heatmap_dir="data/vaihingen_map_generalization/geb15/val/geb_heatmaps",
                         mask_dir="data/vaihingen_map_generalization/geb15/val/geb15_masks",
                         heatmap_dir="data/vaihingen_map_generalization/geb15/val/geb15_heatmaps",
                         mode="val",
                         augment=[],
                         **kwargs)


class Geb15Test(VaihingenBase):
    def __init__(self, **kwargs):
        super().__init__(raw_mask_dir="data/vaihingen_map_generalization/geb15/val/geb_masks",
                         raw_heatmap_dir="data/vaihingen_map_generalization/geb15/val/geb_heatmaps",
                         mask_dir="data/vaihingen_map_generalization/geb15/val/geb15_masks",
                         heatmap_dir="data/vaihingen_map_generalization/geb15/val/geb15_heatmaps",
                         mode="test",
                         augment=[], **kwargs)