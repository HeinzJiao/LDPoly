import os
import re
import cv2
import numpy as np
import argparse
from glob import glob

def parse_patch_filename(filename):
    # 匹配像 original_patch_3_5.png 这样的命名
    match = re.match(r"(.*)_patch_(\d+)_(\d+)\.png", os.path.basename(filename))
    if match:
        base_name = match.group(1)
        row = int(match.group(2))
        col = int(match.group(3))
        return base_name, row, col
    return None, None, None

def stitch_patches(patch_dir, out_path, original_shape=(4270, 2560), patch_size=128):
    patch_paths = glob(os.path.join(patch_dir, '*.png'))
    patch_dict = {}
    base_name = None

    # 组织 patch 为二维矩阵
    for path in patch_paths:
        name, row, col = parse_patch_filename(path)
        if name is None:
            continue
        if base_name is None:
            base_name = name
        patch_dict[(row, col)] = cv2.imread(path)
        patch_dict[(row, col)] = (patch_dict[(row, col)] > 127).astype(np.uint8) * 255

    if not patch_dict:
        print("No valid patches found.")
        return

    # 推测拼图大小
    print(k[0] for k in patch_dict.keys())
    max_row = max(k[0] for k in patch_dict.keys()) + 1
    print(k[1] for k in patch_dict.keys())
    max_col = max(k[1] for k in patch_dict.keys()) + 1

    patch_h, patch_w = patch_size, patch_size
    padded_h = max_row * patch_h
    padded_w = max_col * patch_w
    full_image = np.zeros((padded_h, padded_w, 3), dtype=np.uint8)

    for (row, col), patch in patch_dict.items():
        y, x = row * patch_h, col * patch_w
        full_image[y:y+patch_h, x:x+patch_w] = patch

    # 裁掉 padding 恢复原始尺寸
    final_image = full_image[:original_shape[1], :original_shape[0]]
    cv2.imwrite(out_path, final_image)
    print(f"Reconstructed image saved to {out_path}")

def main():
    parser = argparse.ArgumentParser(description="Stitch patches back to original image")
    parser.add_argument('-i', '--input_dir', required=True, help='Directory containing patch images')
    parser.add_argument('-o', '--output_path', required=True, help='Path to save the reconstructed image')
    parser.add_argument('--patch_size', type=int, default=128, help='Size of each patch (default: 128)')
    parser.add_argument('--original_height', type=int, default=4270, help='Original image height')
    parser.add_argument('--original_width', type=int, default=2560, help='Original image width')
    args = parser.parse_args()

    stitch_patches(
        args.input_dir,
        args.output_path,
        original_shape=(args.original_height, args.original_width),
        patch_size=args.patch_size
    )

if __name__ == '__main__':
    main()
    """
    PYTHONPATH=./:$PYTHONPATH python -u scripts/stitch_patches_to_image.py \
        -i ./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/samples_seg_ddim \
        -o ./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79/geb15_FTest1_input.png \
        --patch_size 128 \
        --original_height 558 \
        --original_width 483
    """
