"""
Dataset definitions for joint semantic segmentation mask and vertex heatmap prediction.

This module implements:
    - A generic SegVertexBase class that loads images, masks, and polygon vertices
      from COCO-style annotations.
    - Automatic generation of Gaussian vertex heatmaps.
    - Optional geometric augmentations (flip, rotation).
    - Unified interfaces for train/val/test splits across different regions.

All pixel values are normalized to [-1, 1] for training, and kept in [0, 1] during testing.
"""

import os
import cv2
import numpy as np
from PIL import Image
import random
from torch.utils.data import Dataset
import torchvision.transforms.functional as F
from pycocotools.coco import COCO


# -------------------------------------------------------------
# Gaussian vertex heatmap generation
# -------------------------------------------------------------

def gaussian_2d(x, y, x0, y0, sigma):
    """Compute a 2D Gaussian value at coordinate (x, y) with center (x0, y0)."""
    return np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * sigma**2))


def generate_heatmap(vertex_locations, heatmap_shape, sigma=5):
    """
    Generate a vertex heatmap by placing a Gaussian peak at each vertex location.

    Args:
        vertex_locations (list[(x, y)]): List of vertex coordinates.
        heatmap_shape (tuple): (H, W) of output heatmap.
        sigma (int): Standard deviation of Gaussian kernels.

    Returns:
        heatmap (H, W): Normalized to [0, 1].
    """
    heatmap = np.zeros(heatmap_shape, dtype=np.float32)
    if len(vertex_locations) == 0:
        return heatmap

    # 生成x和y的网格
    x = np.arange(heatmap_shape[1])
    y = np.arange(heatmap_shape[0])
    xv, yv = np.meshgrid(x, y)

    # 对于每个顶点，计算其在整个图像上的高斯分布，并在相应位置取最大值而不是相加，以确保所有顶点的位置在模糊后都能保持峰值为1。
    for (x0, y0) in vertex_locations:
        heatmap = np.maximum(heatmap, gaussian_2d(xv, yv, x0, y0, sigma))

    # 归一化处理，将热力图的值缩放到 [0, 1] 范围
    max_val = heatmap.max()
    if max_val > 0:
        heatmap /= max_val

    # 将接近零的值设为零
    heatmap[heatmap < 1e-8] = 0

    return heatmap


# -------------------------------------------------------------
# Base dataset for mask + vertex heatmap learning
# -------------------------------------------------------------

class SegVertexBase(Dataset):
    """
    Generic dataset for joint segmentation-mask and vertex-heatmap learning.

    Supports three usage scenarios:
    1) Training / validation:
       - COCO annotations available;
       - Segmentation masks available;
       - Both mask and vertex heatmap supervised.

    2) Standard testing:
       - COCO annotations available;
       - Masks optional;
       - Ground-truth vertices (and optionally masks) used for evaluation.

    3) Generalization / image-only testing:
       - No annotations (coco_annotation_file=None);
       - No masks (mask_dir=None);
       - Only images loaded; heatmap and mask are zeros, and no GT vertices returned.

    If coco_annotation_file is None → no vertices are loaded.
    If mask_dir is None → segmentation mask is omitted and replaced with zeros.
    """
    def __init__(self, coco_annotation_file,
                 image_dir, mask_dir=None,
                 mode="train", transforms=None, size=256, num_classes=2):

        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms or []
        self.mode = mode
        self.size = size
        self.num_classes=num_classes

        # COCO annotation is optional
        if coco_annotation_file is not None and os.path.exists(coco_annotation_file):
            self.coco = COCO(coco_annotation_file)
            self.image_ids = list(self.coco.imgs.keys())
            self.use_coco = True
        else:
            # image-only test: take all filenames
            self.use_coco = False
            self.image_ids = sorted([
                fname for fname in os.listdir(image_dir)
                if fname.lower().endswith((".png", ".jpg", ".jpeg"))
            ])

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # -------------------------
        # Load image (always exists)
        # -------------------------
        if self.use_coco:
            image_id = self.image_ids[idx]
            img_info = self.coco.loadImgs(image_id)[0]
            fname = img_info["file_name"]
            width, height = img_info["width"], img_info["height"]
        else:
            # pure test mode: use filename directly
            fname = self.image_ids[idx]
            img = cv2.imread(os.path.join(self.image_dir, fname))
            height, width = img.shape[:2]

        image_path = os.path.join(self.image_dir, fname)
        image = Image.fromarray(cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB))

        # -------------------------
        # Load mask (optional)
        # -------------------------
        if self.mask_dir is not None:
            mask_path = os.path.join(self.mask_dir, fname)
            if os.path.exists(mask_path):
                mask = Image.fromarray(cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB))
            else:
                mask = None
        else:
            mask = None

        # -------------------------
        # Load polygon vertices (optional)
        # -------------------------
        vertex_locations = []
        if self.use_coco:
            ann_ids = self.coco.getAnnIds(imgIds=image_id)
            anns = self.coco.loadAnns(ann_ids)
            for ann in anns:
                if "segmentation" in ann:
                    for seg in ann["segmentation"]:
                        pts = np.array(seg).reshape(-1, 2)
                        vertex_locations.extend([(x, y) for x, y in pts])

        # -------------------------
        # Apply augmentation
        # -------------------------
        if self.mode == "train" and self.transforms:
            mask, image, vertex_locations = self.apply_transform(mask, image, vertex_locations)
        else:
            # clamp
            vertex_locations = [
                (min(max(int(round(x)), 0), width - 1),
                 min(max(int(round(y)), 0), height - 1))
                for (x, y) in vertex_locations
            ]

        # -------------------------
        # Convert mask → np array
        # -------------------------
        if mask is not None:
            mask = (np.array(mask) > 128).astype(np.float32)
        else:
            # For annotation-free test: keep mask as None or all zeros
            mask = np.zeros((height, width, 3), dtype=np.float32)

        # -------------------------
        # Generate heatmap
        # -------------------------
        if len(vertex_locations) > 0:
            heatmap = generate_heatmap(vertex_locations, (height, width), sigma=5)
        else:
            heatmap = np.zeros((height, width), dtype=np.float32)

        # -------------------------
        # Normalize image [-1,1]
        # -------------------------
        image = (np.array(image).astype(np.float32) / 255.) * 2 - 1

        # heatmap & mask normalization
        if self.mode != "test":
            heatmap = heatmap * 2 - 1
            mask = mask * 2 - 1

        heatmap = np.repeat(heatmap[..., None], 3, axis=2).astype(np.float32)

        # -------------------------
        # Output dict
        # -------------------------
        example = {
            "file_path_": image_path,
            "image": image,
            "heatmap": heatmap,
            "class_id": np.array([-1]),
            "segmentation": mask
        }

        if self.mode == "test":
            # only include ground-truth vertices if they exist
            example["vertex_locations"] = np.array(vertex_locations, dtype=np.int32)

        return example

    def apply_transform(self, segmentation, image, vertex_locations):
        """
        Apply geometric augmentations (flip / rotate) to image, segmentation mask,
        and vertex locations in a consistent way.

        Args:
            segmentation (PIL.Image or None): Segmentation mask.
            image (PIL.Image): Input image.
            vertex_locations (list[(x, y)]): List of vertex coordinates.

        Returns:
            segmentation, image, vertex_locations: Transformed outputs.
        """
        width, height = image.size

        for transform in self.transforms:
            # Horizontal flip
            if transform == "hflip" and np.random.rand() > 0.5:
                if segmentation is not None:
                    segmentation = F.hflip(segmentation)
                image = F.hflip(image)

                if len(vertex_locations) > 0:
                    # x' = (W - 1) - x
                    vertex_locations = [
                        (width - 1 - x, y) for (x, y) in vertex_locations
                    ]

            # Vertical flip
            elif transform == "vflip" and np.random.rand() > 0.5:
                if segmentation is not None:
                    segmentation = F.vflip(segmentation)
                image = F.vflip(image)

                if len(vertex_locations) > 0:
                    # y' = (H - 1) - y
                    vertex_locations = [
                        (x, height - 1 - y) for (x, y) in vertex_locations
                    ]

            # Rotation by 90 / 180 / 270 degrees (CCW)
            elif transform == "rotate" and np.random.rand() > 0.5:
                angle = random.choice([90, 180, 270])

                if segmentation is not None:
                    segmentation = F.rotate(segmentation, angle)
                image = F.rotate(image, angle)

                if len(vertex_locations) > 0:
                    if angle == 90:
                        # (x, y) -> (y, W - 1 - x), then swap width/height
                        vertex_locations = [
                            (y, width - 1 - x) for (x, y) in vertex_locations
                        ]
                        width, height = height, width

                    elif angle == 180:
                        # (x, y) -> (W - 1 - x, H - 1 - y)
                        vertex_locations = [
                            (width - 1 - x, height - 1 - y)
                            for (x, y) in vertex_locations
                        ]

                    elif angle == 270:
                        # (x, y) -> (H - 1 - y, x), then swap width/height
                        vertex_locations = [
                            (height - 1 - y, x) for (x, y) in vertex_locations
                        ]
                        width, height = height, width

        # Final rounding + clamping to valid image coordinates
        if len(vertex_locations) > 0:
            vertex_locations = [
                (
                    min(max(int(round(x)), 0), width - 1),
                    min(max(int(round(y)), 0), height - 1),
                )
                for (x, y) in vertex_locations
            ]

        return segmentation, image, vertex_locations

# -------------------------------------------------------------
# Region-specific dataset definitions
# -------------------------------------------------------------

class DeventerRoadTrain(SegVertexBase):
    def __init__(self, size=256, num_classes=2, **kwargs):
        """Deventer road training split."""
        super().__init__(
            coco_annotation_file="data/deventer_road/annotations/train.json",
            image_dir="data/deventer_road/train_images",
            mask_dir="data/deventer_road/train_masks",
            mode="train",
            transforms=["hflip", "vflip", "rotate"],
            size=size,
            num_classes=num_classes,
        )


class DeventerRoadValidation(SegVertexBase):
    def __init__(self, size=256, num_classes=2, **kwargs):
        """Deventer road validation split."""
        super().__init__(
            coco_annotation_file="data/deventer_road/annotations/test.json",
            image_dir="data/deventer_road/test_images",
            mask_dir="data/deventer_road/test_masks",
            mode="val",
            transforms=[],
            size=size,
            num_classes=num_classes,
        )


class DeventerRoadTest(SegVertexBase):
    def __init__(self, size=256, num_classes=2, **kwargs):
        """Deventer road test split."""
        super().__init__(
            coco_annotation_file="data/deventer_road/annotations/test.json",
            image_dir="data/deventer_road/test_images",
            mask_dir="data/deventer_road/test_masks",
            mode="test",
            transforms=[],
            size=size,
            num_classes=num_classes,
        )


class EnschedeRoadTest(SegVertexBase):
    def __init__(self, size=256, num_classes=2, **kwargs):
        super().__init__(
            coco_annotation_file="data/enschede_road/annotations/test.json",
            image_dir="data/enschede_road/test_images",
            mask_dir="data/enschede_road/test_masks",
            mode="test",
            transforms=[],
            size=size,
            num_classes=num_classes,
        )


class GiethoornRoadTest(SegVertexBase):
    def __init__(self, size=256, num_classes=2, **kwargs):
        super().__init__(
            coco_annotation_file="data/giethoorn_road/annotations/test.json",
            image_dir="data/giethoorn_road/test_images",
            mask_dir="data/giethoorn_road/test_masks",
            mode="test",
            transforms=[],
            size=size,
            num_classes=num_classes,
        )

