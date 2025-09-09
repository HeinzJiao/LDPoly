import os
import json
from tqdm import tqdm

def split_coco_annotations(annotation_file, train_dir, val_dir, output_dir):
    """
    将一个COCO格式的annotation.json根据train/val文件夹中的图片文件名拆分为train.json和val.json

    Args:
        annotation_file (str): 完整的COCO标注文件路径
        train_dir (str): 训练图片文件夹路径
        val_dir (str): 验证图片文件夹路径
        output_dir (str): 拆分后的json保存路径
    """

    os.makedirs(output_dir, exist_ok=True)

    with open(annotation_file, 'r') as f:
        coco_data = json.load(f)

    images = coco_data['images']
    annotations = coco_data['annotations']
    categories = coco_data['categories']

    # 获取 train/val 中的图片文件名集合（不含扩展名）
    train_files = set(os.listdir(train_dir))
    val_files = set(os.listdir(val_dir))

    # 拆分 images 部分
    train_images = [img for img in images if img['file_name'] in train_files]
    val_images = [img for img in images if img['file_name'] in val_files]

    # 拆分 annotations 部分（根据 image_id）
    train_image_ids = {img['id'] for img in train_images}
    val_image_ids = {img['id'] for img in val_images}

    train_annotations = [ann for ann in annotations if ann['image_id'] in train_image_ids]
    val_annotations = [ann for ann in annotations if ann['image_id'] in val_image_ids]

    # 构建新的 COCO dict
    train_coco = {
        'images': train_images,
        'annotations': train_annotations,
        'categories': categories
    }
    val_coco = {
        'images': val_images,
        'annotations': val_annotations,
        'categories': categories
    }

    # 保存输出
    with open(os.path.join(output_dir, 'geb15_train_annotations.json'), 'w') as f:
        json.dump(train_coco, f)
    with open(os.path.join(output_dir, 'geb15_val_annotations.json'), 'w') as f:
        json.dump(val_coco, f)

    print(f"✅ 拆分完成：train.json 包含 {len(train_images)} 张图像，val.json 包含 {len(val_images)} 张图像")

if __name__ == "__main__":
    annotation_file = "./unsplit/geb15/annotations_geb15_merged_clean_filtered.json"
    train_dir = "./geb15/train/geb_masks"
    val_dir = "./geb15/val/geb_masks"
    output_dir = "./geb15"

    split_coco_annotations(annotation_file, train_dir, val_dir, output_dir)
