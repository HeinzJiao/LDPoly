import os
import cv2
import numpy as np
import argparse

def split_image_to_patches(img_path, out_dir, patch_size=128, pad_value=0):
    img = cv2.imread(img_path)
    if img is None:
        print(f"Failed to read image: {img_path}")
        return
    h, w = img.shape[:2]

    # Calculate padded dimensions
    new_w = int(np.ceil(w / patch_size) * patch_size)
    new_h = int(np.ceil(h / patch_size) * patch_size)

    # Create padded image
    padded = np.full((new_h, new_w, 3), pad_value, dtype=img.dtype)
    padded[0:h, 0:w] = img

    # Split into patches
    basename = os.path.splitext(os.path.basename(img_path))[0]
    idx = 0
    for y in range(0, new_h, patch_size):
        for x in range(0, new_w, patch_size):
            patch = padded[y:y+patch_size, x:x+patch_size]
            patch_name = f"{basename}_patch_{y//patch_size}_{x//patch_size}.png"
            out_path = os.path.join(out_dir, patch_name)
            cv2.imwrite(out_path, patch)
            idx += 1
    print(f"Saved {idx} patches for image {basename}")


def main():
    parser = argparse.ArgumentParser(
        description="Split images into non-overlapping patches with padding"
    )
    parser.add_argument(
        '--input_path', '-i', required=True,
        help='Input image path'
    )
    parser.add_argument(
        '--output_dir', '-o', required=True,
        help='Directory to save output patches'
    )
    parser.add_argument(
        '--patch_size', '-p', type=int, default=128,
        help='Size of each patch (default: 128)'
    )
    parser.add_argument(
        '--pad_value', type=int, default=0,
        help='Padding value for right/bottom borders (default: 0 - black)'
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # fpath = os.path.join(args.input_dir, fname)
    fpath = args.input_path
    if os.path.isfile(fpath):
        split_image_to_patches(
            fpath, args.output_dir,
            patch_size=args.patch_size,
            pad_value=args.pad_value
        )

if __name__ == '__main__':
    main()
    """
    python split_to_patches.py -i ./test/geb_clip_4270.png \
        -o ./test/geb_clip_4270 \
        -p 128
    """