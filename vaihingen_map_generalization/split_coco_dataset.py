import os
import json
import random
import shutil


def split_coco_dataset(
    coco_json_file,
    input_folders_dict,
    output_root,
    train_ratio=0.9,
    val_ratio=0.1
):
    """
    将 COCO 数据集划分为 train/val 并同时分割多个对应图像文件夹。

    参数:
    - coco_json_file: COCO 格式的 annotations.json 文件路径。
    - input_folders_dict: dict，键为目标子文件夹名（如 'geb_masks'），值为源图像文件夹路径。
    - output_root: 输出的根目录路径，将生成 train/ 和 val/ 文件夹。
    - train_ratio: 训练集占比，默认 0.9。
    - val_ratio: 验证集占比，默认 0.1。
    """

    # 创建输出文件夹
    for split in ['train', 'val']:
        for name in input_folders_dict.keys():
            os.makedirs(os.path.join(output_root, split, name), exist_ok=True)

    # 读取 COCO annotations.json 文件
    with open(coco_json_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # 随机打乱并划分 train/val
    random.shuffle(images)
    total = len(images)
    train_count = int(total * train_ratio)

    train_images = images[:train_count]
    val_images = images[train_count:]

    def filter_annotations(image_ids):
        return [ann for ann in annotations if ann['image_id'] in image_ids]

    train_anns = filter_annotations([img['id'] for img in train_images])
    val_anns = filter_annotations([img['id'] for img in val_images])

    # # 保存 annotations 文件
    # with open(os.path.join(output_root, 'train_annotations.json'), 'w') as f:
    #     json.dump({'images': train_images, 'annotations': train_anns, 'categories': categories}, f)
    # with open(os.path.join(output_root, 'val_annotations.json'), 'w') as f:
    #     json.dump({'images': val_images, 'annotations': val_anns, 'categories': categories}, f)

    # 将图片复制到 train/ 和 val/ 的对应文件夹中
    def copy_split_images(split_images, split_name):
        for image in split_images:
            filename = image['file_name']
            for target_subfolder, source_folder in input_folders_dict.items():
                if 'heatmap' in target_subfolder:
                    # heatmap 存储为 .npy，需替换扩展名
                    filename_npy = os.path.splitext(filename)[0] + '.npy'
                    src_path = os.path.join("./unsplit/geb15", source_folder, filename_npy)
                    dst_path = os.path.join(output_root, split_name, target_subfolder, filename_npy)
                else:
                    # mask 等为 .png
                    src_path = os.path.join("./unsplit/geb15", source_folder, filename)
                    dst_path = os.path.join(output_root, split_name, target_subfolder, filename)

                if os.path.exists(src_path):
                    shutil.copy(src_path, dst_path)
                else:
                    print(f"Warning: Missing file {src_path}")

    copy_split_images(train_images, 'train')
    copy_split_images(val_images, 'val')

    print(f"完成划分：训练集 {len(train_images)} 张，验证集 {len(val_images)} 张")


if __name__ == "__main__":
    # 示例调用
    coco_json_file = "unsplit/geb15/annotations_geb15_merged_clean_filtered.json"
    output_root = "./geb15"

    # 定义每个子文件夹对应的源路径
    input_folders_dict = {
        'geb_masks': 'geb_128_128',
        'geb15_masks': 'geb15_128_128',
        'geb_heatmaps': 'geb_heatmaps_sigma2.5',
        'geb15_heatmaps': 'geb15_heatmaps_sigma2.5'
    }

    split_coco_dataset(coco_json_file, input_folders_dict, output_root, train_ratio=0.9, val_ratio=0.1)
