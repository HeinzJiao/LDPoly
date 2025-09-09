**ddpm_latent.py:
    directly use latent vector as model input, omit the process of calculating latent vector with autoencoder
**custom_ddpm.py:
    rewrite def log_dice for convenience during test; add def log_vertex_metrics to test vertex (extracted from predicted vertex mask) precision and recall

**ddpm_building_mask_vertex_heatmap.py:
    diffuse and denoise building mask and vertex heatmap at the same time
**custom_ddim_building_mask_vertex_heatmap:
    add additional input (vertex heatmap) to the ddim sampler
**custom_ddpm_building_mask_vertex_heatmap.py:
    rewrite def log_images, save predicted building mask and vertex heatmap; calculate mask IoU and Dice score, and vertex precision and recall

**ddpm_building_mask_vertex_heatmap_custom_unet_v1.py:
    diffuse building mask and vertex heatmap at the same time; use a separate prediction head at the end of the denoiser
**custom_ddim_building_mask_vertex_heatmap:
    add additional input (vertex heatmap) to the ddim sampler
**custom_building_mask_vertex_heatmap_custom_unet.py:
    from ldm.models.diffusion.ddpm_building_mask_vertex_heatmap_custom_unet_v1 import LatentDiffusion

**ddpm_building_mask_vertex_heatmap_custom_unet.py:
    diffuse building mask and vertex heatmap at the same time; use a separate branch in the last half part of the denoiser decoder
**custom_ddim_building_mask_vertex_heatmap:
    add additional input (vertex heatmap) to the ddim sampler
**custom_building_mask_vertex_heatmap_custom_unet.py:
    from ldm.models.diffusion.ddpm_building_mask_vertex_heatmap_custom_unet import LatentDiffusion

**ddpm_building_mask_vertex_edge_heatmap.py
    diffuse and denoise building mask, vertex heatmap and edge heatmap at the same time
**custom_ddim_building_mask_vertex_edge_heatmap:
    add additional inputs (vertex heatmap, edge heatmap) to the ddim sampler



