The code, data, pretrained weights are currently being organized and will be uploaded before Chirstmas.

# 🔍 Inference & Evaluation Guide

# 1. 🧪 Testing on Deventer Road Dataset
## Step 1 — Diffusion Model Sampling
Generates segmentation logits + vertex heatmaps.
```bash
PYTHONPATH=./:$PYTHONPATH python -u scripts/evaluate.py \
    --dataset deventer_road_mask_vertex_heatmap \
    --outdir outputs/deventer_road_reproduction \
    --run 2024-12-24T23-55-18_deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed \
    --model_ckpt epoch=824-step=739199.ckpt \
    --model seg_vert_ldm \
    --sampler ddim \
    --save_results \
    --save_logits_npy \
    --ddim_steps 20
```

## Step 2 - Extract Vertices from Heatmap
Outputs per-image vertex coordinate lists in JSON format.
```bash
python scripts/extract_vertices_from_heatmap.py \
    --heatmaps_dir "./outputs/deventer_road_reproduction/epoch=824-step=739199/samples_heat_ddim_npy" \
    --outputs_dir "./outputs/deventer_road_reproduction/epoch=824-step=739199" \
    --th 0.1 \
    --sampler ddim \
    --upscale_factor 4 \
    --kernel_size 3
```

## Step 3 - Polygonization
```bash
python scripts/polygonization.py \
    --annotation_path ./data/deventer_road/annotations/test.json \
    --outputs_dir ./outputs/deventer_road_reproduction/epoch=824-step=739199 \
    --sampler ddim \
    --output_vertices_file "output_vertices_from_heatmap_x4_ddim_th-0.1_k-3.json" \
    --samples_seg_logits_file "samples_seg_ddim_logits_npy" \
    --save_file "polygons_seg_ddim_vertices_from_heat_th-0.1_k-3_dp_eps2.json" \
    --d_th 5 \
    --polygonization_vis_path ./outputs/deventer_road_reproduction/epoch=824-step=739199/polygonization_vis
```

2. 🖼️ Inference on Any Image or Folder (Generalization Testing)
```bash
PYTHONPATH=./:$PYTHONPATH python scripts/inference.py \
    --input path/to/image_or_folder \
    --outdir path/to/output_folder \
    --run 2024-12-24T23-55-18_deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed \
    --model_ckpt epoch=824-step=739199.ckpt \
    --sampler ddim \
    --ddim_steps 20 \
    --d_th 5
```
This will produce: segmentation logits, vertex heatmaps, vector polygons

# Dataset (ready)
The Dutch polygonal road outline extraction dataset can be downloaded [here] (https://drive.google.com/drive/folders/1jsjuZxFdU9a8q-m0TNCj1MfX9rixTYJl?usp=sharing)

# Demo (to be updated)
https://colab.research.google.com/drive/1IW5AGfn3w3y9wSquYgXolGhcVwIWkoNd#scrollTo=eval_run
