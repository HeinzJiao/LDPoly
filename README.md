<div align="center">
  <h1>LDPoly: Latent Diffusion for Polygonal Road Outline Extraction in Large-Scale Topographic Mapping</h1>
  <p><strong>LDPoly</strong> is the first generative framework and benchmark for polygonal road outline extraction from high-resolution aerial imagery.</p>
  <p>
    <a href="https://doi.org/10.1016/j.isprsjprs.2025.10.005"><img src="https://img.shields.io/badge/ISPRS-2025-0A66C2" alt="ISPRS 2025"></a>
    <a href="https://drive.google.com/drive/folders/1jsjuZxFdU9a8q-m0TNCj1MfX9rixTYJl?usp=sharing"><img src="https://img.shields.io/badge/Data%20%26%20Weights-Google%20Drive-34A853" alt="Data and Weights"></a>
    <a href="https://colab.research.google.com/drive/1IW5AGfn3w3y9wSquYgXolGhcVwIWkoNd#scrollTo=eval_run"><img src="https://img.shields.io/badge/Demo-Colab-F9AB00" alt="Colab Demo"></a>
  </p>
</div>

## Overview

Existing methods for road extraction from aerial imagery mainly focus on road centerlines or boundary polylines, while polygonal road outline extraction remains largely unexplored. LDPoly is a novel diffusion-model-based framework specifically designed for polygonal road outline extraction from high-resolution aerial imagery. It employs a dual-latent diffusion model that jointly diffuses and denoises road masks and keypoint heatmaps, which are subsequently vectorized into polygonal road outlines.

To the best of our knowledge, LDPoly is the first dedicated framework for polygonal road outline extraction from high-resolution aerial imagery, and the first diffusion-based approach for generating polygonal object outlines with minimal vertex redundancy in the remote sensing domain.

## Environment Setup

The project uses [uv](https://docs.astral.sh/uv/) for reproducible environments.
`uv.lock` pins the entire dependency graph, including a CUDA 12.8 PyTorch build.

```bash
uv sync
```

This creates `.venv/` and installs the project in editable mode, so `ldm`,
`scripts` and `metrics` are importable without setting `PYTHONPATH`. Prefix the
commands below with `uv run`, or activate the environment first with
`source .venv/bin/activate`.

> **GPU support.** The original Conda environment pinned `torch 1.11` /
> `cudatoolkit 11.3`, which ships no kernels for Blackwell (sm_120) GPUs such as
> the RTX 50-series. The uv environment uses `torch >= 2.7` on CUDA 12.8, which
> covers sm_75 through sm_120.

## Dataset and Pretrained Weights

The Dutch polygonal road outline extraction dataset and the pretrained LDPoly weights trained on the Deventer region are available here:

[Google Drive - Dataset and Pretrained Weights](https://drive.google.com/drive/folders/1jsjuZxFdU9a8q-m0TNCj1MfX9rixTYJl?usp=sharing)

Google Drive splits large downloads across several folders whose contents are
disjoint, so no single folder is a complete dataset. `scripts/setup_data.py`
merges them into one canonical data root and links that root into the repository:

```bash
# Print the plan without changing anything
python scripts/setup_data.py --data-root /path/to/download

# Execute it
python scripts/setup_data.py --data-root /path/to/download --apply
```

This creates `data/`, `logs/<run>/checkpoints/` and
`models/first_stage_models/kl-f8/model.ckpt` as symlinks into the data root,
then verifies that every image referenced by the COCO annotations is present.
Files inside the downloads are hardlinked rather than copied, so the originals
(retained under `_downloads/`) cost no additional disk space. Use
`--verify-only` to re-check an already-assembled layout.

## Training

> **Status.** The uv migration was validated on the inference path only.
> `main.py` still targets the PyTorch Lightning 1.x `Trainer` API (note the
> `--gpus 0,` flag below), while the environment installs Lightning 2.x, so
> the training entrypoint needs porting before the command below will run.
> Inference and evaluation are unaffected.

Train LDPoly on the Deventer region with:

```bash
uv run python -u main.py --base configs/latent-diffusion/deventer_road_mask_vertex_heatmap-ldm-kl-8.yaml -t --gpus 0, \
--name deventer_r_v-ldm-kl-8
```

To resume training from a checkpoint, replace `--name` with:

```bash
--resume logs/<your_experiment_folder>
```

Example:

```bash
--resume logs/2024-12-24T23-55-18_deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed
```

## Inference

### Testing on the Deventer Road Dataset

1. Diffusion model sampling:

```bash
uv run python -u scripts/evaluate.py \
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

2. Extract vertices from heatmaps:

```bash
uv run python scripts/extract_vertices_from_heatmap.py \
    --heatmaps_dir "./outputs/deventer_road_reproduction/epoch=824-step=739199/samples_heat_ddim_npy" \
    --outputs_dir "./outputs/deventer_road_reproduction/epoch=824-step=739199" \
    --th 0.1 \
    --sampler ddim \
    --upscale_factor 4 \
    --kernel_size 3
```

3. Polygonization:

```bash
uv run python scripts/polygonization.py \
    --annotation_path ./data/deventer_road/annotations/test.json \
    --outputs_dir ./outputs/deventer_road_reproduction/epoch=824-step=739199 \
    --sampler ddim \
    --output_vertices_file "output_vertices_from_heatmap_x4_ddim_th-0.1_k-3.json" \
    --samples_seg_logits_file "samples_seg_ddim_logits_npy" \
    --save_file "polygons_seg_ddim_vertices_from_heat_th-0.1_k-3_dp_eps2.json" \
    --d_th 5 \
    --polygonization_vis_path ./outputs/deventer_road_reproduction/epoch=824-step=739199/polygonization_vis
```

### Generalization on Custom Images or Folders

```bash
uv run python scripts/inference.py \
    --input path/to/image_or_folder \
    --outdir path/to/output_folder \
    --run 2024-12-24T23-55-18_deventer_road_mask_vertex_heatmap_split_by_image_PreConvConcat_ChannelEmbed \
    --model_ckpt epoch=824-step=739199.ckpt \
    --sampler ddim \
    --ddim_steps 20 \
    --d_th 5
```

This pipeline produces segmentation logits, vertex heatmaps, and vector polygons.

## Evaluation

Evaluate polygon quality and vertex efficiency with:

```bash
uv run python evaluation.py \
    --gt-file ./data/deventer_road/annotations/test.json \
    --dt-file ./outputs/deventer_road_reproduction/epoch=824-step=739199/polygons_seg_ddim_vertices_from_heat_th-0.1_k-3_dp_eps2.json \
    --eval-type cse
```

Supported `--eval-type` options:

- `cse`: pixel-level coverage (`IoU`, `B-IoU`), polygon simplicity (`S-IoU`), and vertex efficiency (`C-IoU`, `N-ratio`)
- `r`: polygon regularity (`PoLiS`, `SCR`)

Evaluate polygon connectivity (`APLS`) with:

```bash
uv run python eval_apls.py \
    --dt_file path/to/prediction.json \
    --gt_file path/to/ground_truth.json \
    --gt_folder path/to/gt_network_graphs \
    --output_folder path/to/output_dir
```

## Demo

[Open the Colab demo](https://colab.research.google.com/drive/1IW5AGfn3w3y9wSquYgXolGhcVwIWkoNd#scrollTo=eval_run)

## Code Status

The codebase will continue to be refined and standardized, but the current version is fully functional for both training and inference. If you encounter any issues, please open an Issue on GitHub.

## Citation

If you find LDPoly useful in your research, please consider citing:

```bibtex
@article{jiao2025ldpoly,
  title={LDPoly: Latent diffusion for polygonal road outline extraction in large-scale topographic mapping},
  author={Jiao, Weiqin and Cheng, Hao and Vosselman, George and Persello, Claudio},
  journal={ISPRS Journal of Photogrammetry and Remote Sensing},
  volume={230},
  pages={820--842},
  year={2025},
  publisher={Elsevier}
}
```

## Related Research

This repository belongs to a broader research line on polygonal vectorization from aerial imagery and large-scale topographic map generation. If LDPoly is relevant to your work, you may also want to follow the companion papers below.

| Paper | Venue | Focus | Resources |
| --- | --- | --- | --- |
| **ACPV-Net** | CVPR 2026 | All-class polygonal vectorization and topology-aware vector basemap generation from a single aerial image. | [Paper](https://arxiv.org/abs/2603.16616) · [Code](https://github.com/HeinzJiao/ACPV-Net) · [Data](https://huggingface.co/datasets/HeinzJiao/Deventer-512) · [Weights](https://huggingface.co/HeinzJiao/deventer512_vmamba-s_m_vh-ldm_kl4_b8) |
| **RoIPoly** | ISPRS 2025 | RoI query-based building polygon extraction with logit-guided vertex interaction. | [Paper](https://doi.org/10.1016/j.isprsjprs.2025.03.030) · [Code](https://github.com/HeinzJiao/RoIPoly) |
| **PolyR-CNN** | ISPRS 2024 | End-to-end polygonal building outline extraction with efficient RoI-based modeling. | [Paper](https://doi.org/10.1016/j.isprsjprs.2024.10.006) · [Code](https://github.com/HeinzJiao/PolyR-CNN) |

## Acknowledgements

This repository benefits from the excellent open-source contributions of [HiSup](https://github.com/SarahwXU/HiSup), [Stable Diffusion](https://github.com/pesser/stable-diffusion), and [SDSeg](https://github.com/lin-tianyu/Stable-Diffusion-Seg). We thank the authors for their great work.
