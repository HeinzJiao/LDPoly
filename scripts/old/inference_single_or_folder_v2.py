# inference_single_or_folder.py

import os
import argparse
from PIL import Image
import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
import glob
from einops import rearrange, repeat
import cv2
from ldm.util import instantiate_from_config
from scripts.extract_vertices_from_scaled_heatmap import extract_vertices_from_heatmap
from skimage.measure import label, regionprops
from scripts.polygonization_refined import get_poly

# 你原来保存图片的函数，可以复用
from torchvision.utils import make_grid
import torchvision

# 读取图片并预处理为模型输入格式
def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    # image = image.resize(image_size, Image.BILINEAR)
    image = np.array(image).astype(np.float32) / 255.0
    image = (image * 2.) - 1.  # range from -1 to 1, np.float32
    image = torch.from_numpy(image).unsqueeze(0)  # 1, H, W, 3
    return image

def load_heatmap(heatmap_path):
    heatmap = np.load(heatmap_path).astype(np.float32)
    heatmap = (heatmap * 2.) - 1.  # range from -1 to 1, np.float32
    heatmap = np.repeat(heatmap[:, :, np.newaxis], 3, axis=2)
    heatmap = torch.from_numpy(heatmap).unsqueeze(0)  # 1, H, W, 3
    return heatmap

# 加载模型
def load_model(config_path, checkpoint_path, device):
    config = OmegaConf.load(config_path)
    config["model"]["target"] = "ldm.models.diffusion.custom_ddpm_vaihingen_map_generalization.ExtendedLatentDiffusion"
    model = instantiate_from_config(config.model)
    pl_sd = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(pl_sd["state_dict"], strict=False)
    model.to(device)
    model.eval()
    return model

@torch.no_grad()
def inference(model, raw_mask, raw_heatmap, save_dir, image_name, sampler='ddim', device='cuda'):
    # 预处理图像 batch，模拟 dataloader 的 batch 结构
    batch = {
        "raw_mask": raw_mask.to(device),  # 1, H, W, 3, range [-1, 1]
        "raw_heatmap": raw_heatmap.to(device),  # 1, H, W, 3, range [-1, 1]
        'segmentation': torch.zeros_like(raw_mask),  # dummy segmentation
        'heatmap': torch.zeros_like(raw_mask),  # dummy heatmap
        'file_path_': [image_name],
        'class_id': torch.tensor([[-1]]).to(device)
    }

    outputs = model.log_images(batch, sampler=sampler, ddim_steps=20)

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples_seg_ddim_logits_npy'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples_seg_ddim'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples_heat_ddim_logits_npy'), exist_ok=True)
    os.makedirs(os.path.join(save_dir, 'samples_heat_ddim'), exist_ok=True)

    # 保存 segmentation mask (.npy)
    seg_output = outputs[f'samples_seg_{sampler}']  # Tensor: [1, C, H, W]
    seg_output = torch.clamp(seg_output, -1, 1)
    seg_output = (seg_output + 1.0) / 2.0  # 归一化到 [0, 1]
    seg_output = seg_output.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # 平均 -> repeat 到 3 通道
    seg_npy = rearrange(seg_output.squeeze(0).cpu().numpy(), 'c h w -> h w c')  # (256, 256, 3)
    np.save(os.path.join(save_dir, 'samples_seg_ddim_logits_npy', f'{image_name}.npy'), seg_npy[:, :, 0].astype(np.float32))  # 保存 (256, 256)
    cv2.imwrite(os.path.join(save_dir, 'samples_seg_ddim', f'{image_name}.png'), seg_npy * 255)

    # 保存 vertex heatmap (.npy)
    heat_output = outputs[f'samples_heat_{sampler}']  # torch.Size([1, 3, 256, 256])
    heat_output = torch.clamp(heat_output, -1, 1)
    heat_output = (heat_output + 1.0) / 2.0  # tensor(0.9840, device='cuda:0') tensor(-0.0173, device='cuda:0')
    heat_output = heat_output.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)  # torch.Size([1, 3, 256, 256])
    heat_npy = rearrange(heat_output.squeeze(0).cpu().numpy(), 'c h w -> h w c')  # (256, 256, 3)
    np.save(os.path.join(save_dir, 'samples_heat_ddim_logits_npy', f'{image_name}.npy'), heat_npy[:, :, 0].astype(np.float32))
    cv2.imwrite(os.path.join(save_dir, 'samples_heat_ddim', f'{image_name}.png'), heat_npy * 255)

    print(f"Processed: {image_name}")

    return seg_npy[:, :, 0], heat_npy[:, :, 0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_raw_mask', type=str, required=True, help='Path to input image or folder of images')
    parser.add_argument('--input_raw_heatmap', type=str, required=True, help='Path to input image or folder of images')
    parser.add_argument('--outdir', type=str, required=True, help='Output directory')
    parser.add_argument("--run", type=str, nargs="?", help="the name of your experiment",
                        default="2024-07-13T17-50-40_cvc")
    parser.add_argument("--model_ckpt", type=str, nargs="?", help="the name of the checkpoint",
                        default="epoch=991-step=121999.ckpt")
    parser.add_argument("--config", type=str, default="configs/stable-diffusion/v1-inference.yaml",
                        help="path to config which constructs model", )
    parser.add_argument("--ckpt", type=str, default="models/ldm/stable-diffusion-v1/model.ckpt",
                        help="path to checkpoint of model",)
    parser.add_argument('--sampler', type=str, default='ddim', choices=['direct', 'ddim', 'ddpm'], help='Sampler type')
    parser.add_argument('--device', type=str, default='cuda', help='Device: cuda or cpu')

    opt = parser.parse_args()

    run = opt.run
    model_ckpt = opt.model_ckpt
    print("Evaluate on deventer road mask and vertex heatmap dataset.")
    opt.config = glob.glob(os.path.join("logs", run, "configs", "*-project.yaml"))[0]
    opt.ckpt = f"logs/{run}/checkpoints/{model_ckpt}"

    device = torch.device(opt.device if torch.cuda.is_available() else 'cpu')
    model = load_model(opt.config, opt.ckpt, device)

    if os.path.isfile(opt.input_raw_mask) and os.path.isfile(opt.input_raw_heatmap):
        # 单张图片
        raw_mask = load_image(opt.input_raw_mask)
        raw_heatmap = load_heatmap(opt.input_raw_heatmap)
        image_name = os.path.splitext(os.path.basename(opt.input_raw_mask))[0]
        seg_npy, heat_npy = inference(model, raw_mask, raw_heatmap, opt.outdir, image_name, sampler=opt.sampler, device=device)

        # 提取顶点
        junctions, _ = extract_vertices_from_heatmap(heat_npy, th=0.1, kernel_size=5, topk=300, upscale_factor=1)
        print("junctions: ", junctions.shape)

        # 多边形化
        logit = seg_npy
        mask = logit > 0.5
        labeled_mask = label(mask)
        props = regionprops(labeled_mask)
        polygons = []
        for i, prop in enumerate(props):
            # Extract polygon for each building region
            poly_list, score = get_poly(prop, logit, junctions)  # poly_list: [[x1, y1, x2, y2, ...], [x1, y1, ...], ...]
            if len(poly_list) == 0:
                continue
            polygons.append(poly_list)

        raw_mask = raw_mask.squeeze(0).cpu().numpy()  # H, W, 3, range: [-1, 1]
        raw_mask = ((raw_mask + 1.0) / 2.0 * 255.0).astype(np.uint8)  # H, W, 3, range: [0, 255]
        raw_mask = raw_mask[..., ::-1]  # RGB to BGR for OpenCV
        visualize_predictions(raw_mask, polygons, image_name, opt.outdir)

    elif os.path.isdir(opt.input_raw_mask) and os.path.isdir(opt.input_raw_heatmap):
        # 文件夹多张图片
        image_paths = glob.glob(os.path.join(opt.input_raw_mask, '*'))
        for image_path in tqdm(image_paths, desc="Processing images"):
            try:
                raw_mask = load_image(image_path)
                image_name = os.path.splitext(os.path.basename(image_path))[0]
                heatmap_path = os.path.join(opt.input_raw_heatmap, image_name + '.npy')
                raw_heatmap = load_heatmap(heatmap_path)
                seg_npy, heat_npy = inference(model, raw_mask, raw_heatmap, opt.outdir, image_name, sampler=opt.sampler, device=device)
            except Exception as e:
                print(f"Error processing {image_path}: {e}")

    else:
        print("Invalid input path! Please provide a valid image or directory.")


def visualize_predictions(image, polygons, image_name, save_dir, alpha=0.1):
    # polygons: [poly_list, poly_list, ...]
    # poly_list: [[x1, y1, x2, y2, ...], [x1, y1, ...], ...]
    image = cv2.resize(image, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
    overlay = np.zeros_like(image, dtype=np.uint8)

    for segmentation in polygons:
        color = (0, 255, 0)  # Green for external polygons
        segmentation[0] = (np.array(segmentation[0]) * 2).tolist()  # 外部轮廓放大
        for i in range(1, len(segmentation)):
            segmentation[i] = (np.array(segmentation[i]) * 2).tolist()  # 内部轮廓放大

        # Draw external polygon
        exterior = np.array(segmentation[0], dtype=np.int32).reshape((-1, 2))
        cv2.fillPoly(overlay, [exterior], color)

        # Draw internal polygons (holes)
        for interior in segmentation[1:]:
            interior = np.array(interior, dtype=np.int32).reshape((-1, 2))
            cv2.fillPoly(overlay, [interior], (0, 0, 0))

    image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

    # Draw all polygon outlines and vertices on the image
    for segmentation in polygons:
        color = (255, 255, 0)  # Cyan for external polygons
        vertex_color = (204, 102, 255)  # Rose pink for vertices

        # Draw external polygon
        exterior = np.array(segmentation[0], dtype=np.int32).reshape((-1, 2))

        cv2.polylines(image, [exterior], isClosed=True, color=color, thickness=2)

        # Draw vertices for the external polygon
        for vertex in exterior:
            cv2.circle(image, tuple(vertex), radius=4, color=vertex_color, thickness=-1)

        # Draw internal polygons (holes)
        for interior in segmentation[1:]:
            interior = np.array(interior, dtype=np.int32).reshape((-1, 2))
            cv2.polylines(image, [interior], isClosed=True, color=color, thickness=2)

            # Draw vertices for the internal polygon
            for vertex in interior:
                cv2.circle(image, tuple(vertex), radius=4, color=vertex_color, thickness=-1)

    # Save the visualization
    output_path = os.path.join(save_dir, '', f'{image_name}.png')
    print("output_path: ", output_path)
    cv2.imwrite(output_path, image)

if __name__ == '__main__':
    main()

    """
    This code is for vaihingen map generalization.
    
    PYTHONPATH=./:$PYTHONPATH python -u scripts/inference_single_or_folder_v2.py \
        --input_raw_mask "./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_input" \
        --input_raw_heatmap "./data/vaihingen_map_generalization/test/Test_1_and_2_for_15k/FTest1_input_sigma2.5" \
        --outdir "./outputs/vaihingen_map_generalization_sigma2.5_geb15/test_geb15_FTest1_input/epoch_79" \
        --run 2025-04-24T22-18-31_vaihingen_map_generalization_sigma2.5_geb15 \
        --model_ckpt epoch=epoch=79.ckpt \
        --sampler ddim
    """
