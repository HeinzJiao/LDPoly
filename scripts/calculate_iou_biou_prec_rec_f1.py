import os
import numpy as np
import cv2
from tqdm import tqdm

def mask_to_boundary(mask, dilation_ratio=0.02):
    """
    Convert binary mask to boundary mask.
    :param mask (numpy array, uint8): binary mask
    :param dilation_ratio (float): ratio to calculate dilation = dilation_ratio * image_diagonal
    :return: boundary mask (numpy array)
    """
    h, w = mask.shape
    img_diag = np.sqrt(h ** 2 + w ** 2)
    dilation = int(round(dilation_ratio * img_diag))
    if dilation < 1:
        dilation = 1

    # Pad image so mask truncated by the image border is also considered as boundary.
    new_mask = cv2.copyMakeBorder(mask, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    kernel = np.ones((3, 3), dtype=np.uint8)
    new_mask_erode = cv2.erode(new_mask, kernel, iterations=dilation)
    mask_erode = new_mask_erode[1 : h + 1, 1 : w + 1]

    # G_d intersects G in the paper.
    return mask - mask_erode

def calculate_iou(pred_mask, gt_mask):
    """计算 IoU（Intersection over Union）"""
    intersection = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum()
    return intersection / (union + 1e-6)  # 避免除 0

def calculate_boundary_iou(pred_mask, gt_mask):
    """计算 Boundary IoU"""
    pred_boundary = mask_to_boundary(pred_mask)
    gt_boundary = mask_to_boundary(gt_mask)

    return calculate_iou(pred_boundary, gt_boundary)

def calculate_prf(pred_mask, gt_mask):
    """
    计算 Precision, Recall, F1
    :param pred_mask: 预测二值 mask (0/1)
    :param gt_mask: 真值二值 mask (0/1)
    :return: precision, recall, f1
    """
    # True Positive, False Positive, False Negative
    tp = np.logical_and(pred_mask == 1, gt_mask == 1).sum()
    fp = np.logical_and(pred_mask == 1, gt_mask == 0).sum()
    fn = np.logical_and(pred_mask == 0, gt_mask == 1).sum()
    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    return precision, recall, f1

def evaluate_masks(pred_dir, gt_dir):
    """
    计算所有预测 mask 与真值 mask 的 IoU 和 Boundary IoU。

    :param pred_dir: 预测 mask 存放目录
    :param gt_dir: 真值 mask 存放目录
    :return: 平均 IoU, 平均 Boundary IoU
    """
    iou_list = []
    boundary_iou_list = []
    precision_list = []
    recall_list = []
    f1_list = []
    num_valid_masks = 0

    pred_files = sorted(os.listdir(pred_dir))
    gt_files = sorted(os.listdir(gt_dir))

    for pred_file, gt_file in tqdm(zip(pred_files, gt_files), total=len(pred_files), desc="Evaluating"):
        pred_path = os.path.join(pred_dir, pred_file)
        gt_path = os.path.join(gt_dir, gt_file)

        # **读取 mask 并转换为二值化 numpy 数组**
        pred_mask = cv2.imread(pred_path, cv2.IMREAD_COLOR)  # **读取 RGB**
        gt_mask = cv2.imread(gt_path, cv2.IMREAD_COLOR)  # **读取 RGB**

        # **计算三通道均值**
        pred_mask = np.mean(pred_mask, axis=2).astype(np.uint8)
        gt_mask = np.mean(gt_mask, axis=2).astype(np.uint8)

        # **归一化并二值化**
        pred_mask = (pred_mask > 127).astype(np.uint8)
        gt_mask = (gt_mask > 127).astype(np.uint8)

        # **跳过全黑真值 mask**
        if gt_mask.sum() == 0:
            continue

        num_valid_masks += 1

        # 计算 IoU 和 Boundary IoU
        iou = calculate_iou(pred_mask, gt_mask)
        boundary_iou = calculate_boundary_iou(pred_mask, gt_mask)
        precision, recall, f1 = calculate_prf(pred_mask, gt_mask)

        iou_list.append(iou)
        boundary_iou_list.append(boundary_iou)
        precision_list.append(precision)
        recall_list.append(recall)
        f1_list.append(f1)

    # 计算平均 IoU 和 Boundary IoU
    mean_iou = np.mean(iou_list) if iou_list else 0.0
    mean_boundary_iou = np.mean(boundary_iou_list) if boundary_iou_list else 0.0
    mean_precision = np.mean(precision_list) if precision_list else 0.0
    mean_recall = np.mean(recall_list) if recall_list else 0.0
    mean_f1 = np.mean(f1_list) if f1_list else 0.0

    print(f"Total evaluated masks: {num_valid_masks}")
    print(f"Mean IoU: {mean_iou:.4f}")
    print(f"Mean Boundary IoU: {mean_boundary_iou:.4f}")
    print(f"Mean Precision: {mean_precision:.4f}")
    print(f"Mean Recall: {mean_recall:.4f}")
    print(f"Mean F1 Score: {mean_f1:.4f}")

    return mean_iou, mean_boundary_iou, mean_precision, mean_recall, mean_f1


def evaluate_single_pair(pred_path, gt_path):
    """
    读取预测图和真值图，计算 IoU、Boundary IoU、Precision、Recall 和 F1-score

    :param pred_path: 预测 mask 图片路径
    :param gt_path:   真值 mask 图片路径
    :return: dict 包含指标
    """
    # 读取为灰度图
    pred_mask = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    gt_mask   = cv2.imread(gt_path,   cv2.IMREAD_GRAYSCALE)

    if pred_mask is None or gt_mask is None:
        raise ValueError(f"无法读取图像: {pred_path} 或 {gt_path}")

    # 若尺寸不同，进行 padding
    if pred_mask.shape != gt_mask.shape:
        pred_h, pred_w = pred_mask.shape
        gt_h, gt_w = gt_mask.shape

        pad_bottom = max(gt_h - pred_h, 0)
        pad_right = max(gt_w - pred_w, 0)

        pred_mask = np.pad(pred_mask,
                           ((0, pad_bottom), (0, pad_right)),
                           mode='constant', constant_values=0)

        # 如果预测图比真值图大，裁剪
        pred_mask = pred_mask[:gt_h, :gt_w]

    # 转为二值
    pred_bin = (pred_mask > 127).astype(np.uint8)
    gt_bin   = (gt_mask   > 127).astype(np.uint8)

    # 计算指标
    iou = calculate_iou(pred_bin, gt_bin)
    biou = calculate_boundary_iou(pred_bin, gt_bin)
    precision, recall, f1 = calculate_prf(pred_bin, gt_bin)

    print(f"IoU: {iou:.4f}")
    print(f"Boundary IoU: {biou:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    return {
        'IoU': iou,
        'Boundary_IoU': biou,
        'Precision': precision,
        'Recall': recall,
        'F1': f1
    }


# **运行评估**
if __name__ == "__main__":
    # pred_mask_dir = "./outputs/vaihingen_map_generalization_sigma2.5_geb10/epoch=epoch=74/samples_seg_ddim"  # 修改为你的预测 mask 文件夹路径
    # gt_mask_dir = "data/vaihingen_map_generalization/geb10/val/geb10_masks"  # 修改为你的真值 mask 文件夹路径
    #
    # evaluate_masks(pred_mask_dir, gt_mask_dir)

    # pred_mask_path = "./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/geb15_FTest1_input.png"  # 修改为你的预测 mask 文件夹路径
    pred_mask_path = "./outputs/fengyu/Test_1_for_15k/FTest1_input_inv_15_unet.png"
    gt_mask_path = "data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_15k_output.png"  # 修改为你的真值 mask 文件夹路径

    evaluate_single_pair(pred_mask_path, gt_mask_path)
