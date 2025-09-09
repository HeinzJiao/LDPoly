from ldm.models.diffusion.ddpm_building_mask_vertex_heatmap import LatentDiffusion
import torch
import tqdm
from tqdm import tqdm
from scipy.ndimage import zoom
from einops import rearrange, repeat
import numpy as np
import os
from torch import autocast
from scripts.slice2seg import prepare_for_first_stage, dice_score, iou_score
from PIL import Image
from torch.utils.data import DataLoader
from scipy.spatial import cKDTree
import torch.nn.functional as F
import cv2
import json
import torchvision
from torchvision.utils import make_grid


class ExtendedLatentDiffusion(LatentDiffusion):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def log_images_loop(self, data, save_results, save_dir, used_sampler='ddpm', save_samples_seg_logits_npy=True, save_samples_heat_logits_npy=True):
        dice_list = 0.
        iou_list = 0.
        precision_list = 0.
        recall_list = 0.
        predictions = []

        pbar = tqdm(data, desc="Validating Segmentation")  # 为可迭代对象添加一个进度条
        for batch_idx, batch in enumerate(pbar):
            image_file_name = batch["file_path_"][0].split('/')[-1]
            label = batch["segmentation"]  # 1 256 256 3  (1, H, W, D) binary 0 1
            vertex_locations = batch["vertex_locations"]  # 1 N 2, torch.int32
            images = self.log_images(batch, sampler=used_sampler,
                                     plot_denoise_rows=False,
                                     plot_diffusion_rows=False,
                                     return_first_stage_outputs=False,
                                     plot_conditioning_latent=False)
            for k in images:
                N = min(images[k].shape[0], 4)
                images[k] = images[k][:N]
                if isinstance(images[k], torch.Tensor):
                    images[k] = images[k].detach().cpu()
                    if True:  # True
                        images[k] = torch.clamp(images[k], -1., 1.)

            """retrieve reconstructed segmentation mask"""
            x_samples_ddpm = images[f"samples_seg_{used_sampler}"]  # range from -1 to 1
            x_samples_ddpm = (x_samples_ddpm + 1.0) / 2.0  # range from 0 to 1
            x_samples_ddpm = x_samples_ddpm.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            # x_out_p: reconstructed segmentation probability map
            x_out_p = rearrange(x_samples_ddpm.squeeze(0).cpu().numpy(), 'c h w -> h w c')  # 256x256x3
            # x_out_p: reconstructed segmentation mask
            x_out = (x_out_p > 0.5)
            x_prediction = x_out[:, :, 0]

            """save reconstructed segmentation probability map in .npy format"""
            if save_samples_seg_logits_npy:
                os.makedirs(os.path.join(save_dir, f"samples_seg_{used_sampler}_logits_npy"), exist_ok=True)
                path = os.path.join(save_dir, f"samples_seg_{used_sampler}_logits_npy",
                                    ".".join([image_file_name.split(".")[0], "npy"]))
                np.save(path, x_out_p[:, :, 0].astype(np.float32))

            """retrieve reconstructed vertex heatmap"""
            h_samples_ddpm = images[f"samples_heat_{used_sampler}"]  # range from -1 to 1
            h_samples_ddpm = (h_samples_ddpm + 1.0) / 2.0  # range from 0 to 1
            h_samples_ddpm = h_samples_ddpm.mean(dim=1, keepdim=True).repeat(1, 3, 1, 1)
            h_out_p = rearrange(h_samples_ddpm.squeeze(0).cpu().numpy(), 'c h w -> h w c')  # 256x256x3
            h_prediction = h_out_p[:, :, 0]

            """save reconstructed vertex heatmap in .npy format"""
            if save_samples_heat_logits_npy:
                os.makedirs(os.path.join(save_dir, f"samples_heat_{used_sampler}_npy"), exist_ok=True)
                path = os.path.join(save_dir, f"samples_heat_{used_sampler}_npy",
                                    ".".join([image_file_name.split(".")[0], "npy"]))
                np.save(path, h_out_p[:, :, 0].astype(np.float32))

            """calculate dice score and iou score"""
            metrics_list_x = [[], []]
            label = label.squeeze(0).numpy()
            label = label[:, :, 0]
            # x_prediction: 256x256; label: 256x256 (shanghai)
            # x_prediction: 512x512; label: 512x512 (shanghai)
            label = label.round().astype(int)
            for idx in range(1, self.num_classes):
                metrics_list_x[0].append(dice_score(x_prediction == idx, label == idx))
                metrics_list_x[1].append(iou_score(x_prediction == idx, label == idx))
            dice_list += np.array(metrics_list_x[0])
            iou_list += np.array(metrics_list_x[1])

            """calculate precision and recall"""
            metrics_list_h = [[], []]
            vertex_locations = vertex_locations.squeeze(0).numpy()
            # h_prediction: 256x256, numpy.ndarray; vertex_locations: Nx2, np.int32
            th = 0.1
            for idx in range(1, self.num_classes):
                extracted_vertices, scores = extract_vertices_from_heatmap(h_prediction, th, topk=300)
                precision, recall = calculate_precision_recall(extracted_vertices, vertex_locations)
                metrics_list_h[0].append(precision)
                metrics_list_h[1].append(recall)
            precision_list += np.array(metrics_list_h[0])
            recall_list += np.array(metrics_list_h[1])

            """save visualizations to save_dir"""
            if save_results:
                self.log_local(save_dir, images, image_file_name)
        pbar.close()

        """print mask mean dice and mean iou"""
        avg_dice = dice_list / len(data)  # np.array (1,)
        for idx in range(1, self.num_classes):
            print(f"\033[31m[Mean Dice][cls {idx}]: {avg_dice[idx - 1]}\033[0m")
        avg_iou = iou_list / len(data)
        for idx in range(1, self.num_classes):
            print(f"\033[31m[Mean  IoU][cls {idx}]: {avg_iou[idx - 1]}\033[0m")

        """print vertex mean precision and recall"""
        avg_precision = precision_list / len(data)
        for idx in range(1, self.num_classes):
            print(f"\033[31m[Mean Precision][cls {idx}]: {avg_precision[idx - 1]}\033[0m")
        avg_recall = recall_list / len(data)
        for idx in range(1, self.num_classes):
            print(f"\033[31m[Mean  Recall][cls {idx}]: {avg_recall[idx - 1]}\033[0m")

        # 保存为 JSON 文件
        self.save_metrics_to_json(avg_dice, avg_iou, avg_precision, avg_recall, save_dir, used_sampler)

    def save_metrics_to_json(self, avg_dice, avg_iou, avg_precision, avg_recall, save_dir, used_sampler='direct'):
        # 创建保存路径
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # 组织数据
        metrics_data = {
            "dice_scores": {f"cls_{idx + 1}": avg_dice[idx] for idx in range(len(avg_dice))},
            "iou_scores": {f"cls_{idx + 1}": avg_iou[idx] for idx in range(len(avg_iou))},
            "precision_scores": {f"cls_{idx + 1}": avg_precision[idx] for idx in range(len(avg_precision))},
            "recall_scores": {f"cls_{idx + 1}": avg_recall[idx] for idx in range(len(avg_recall))}
        }

        # 创建保存的文件名
        json_filename = f"metrics_{used_sampler}.json"
        json_path = os.path.join(save_dir, json_filename)

        # 保存为JSON文件
        with open(json_path, 'w') as json_file:
            json.dump(metrics_data, json_file, indent=4)

        print(f"\033[32mMetrics saved to: {json_path}\033[0m")

    def log_local(self, save_dir, images, image_file_name):
        root = save_dir
        for k in images:  # e.g. latent
            grid = torchvision.utils.make_grid(images[k], nrow=4)  # e.g. 3x138x138 -> 3x138x138
            if True:  # True
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1)  # e.g. 138x138x3
            grid = grid.numpy()
            grid = (grid * 255).astype(np.uint8)
            filename = "{}.png".format(image_file_name.split('.')[0])
            path = os.path.join(root, k, filename)
            os.makedirs(os.path.join(root, k), exist_ok=True)
            Image.fromarray(grid).save(path)

    @torch.no_grad()
    def log_images(self, batch, N=8, n_row=4,
                   sampler="ddpm",
                   ddim_steps=20, ddim_eta=1.,
                   plot_denoise_rows=False,  # visualize ddim intermediate results
                   plot_diffusion_rows=False,  # visualize forward diffusion process
                   return_first_stage_outputs=False,  # visualize autoencoder decoded latent {seg/heat}
                   plot_conditioning_latent=False,
                   **kwargs):
        log = dict()
        # z: latent seg, seg-map after autoencoder encode
        # zh: latent vert heatmap, vert heatmap after autoencoder encode
        # c: CT slice after autoencoder encode, range from -1 to 1
        # x: original seg-map (input of autoencoder), range from -1 to 1
        # h: original vertex heatmap, range from -1 to 1
        # xrec: autoencoder decode output of z
        # hrec: autoencoder decode output of zh
        # xc: the CT slice image
        batch["segmentation"] = batch["segmentation"] * 2 - 1  # range from -1 to 1
        batch["heatmap"] = batch["heatmap"] * 2 - 1  # range from -1 to 1
        if return_first_stage_outputs:
            z, zh, c, x, h, cls_id, xrec, hrec, xc = self.get_input(batch, self.first_stage_key,
                                                                    return_first_stage_outputs=True,
                                                                    force_c_encode=True,
                                                                    return_original_cond=False,
                                                                    bs=N)
        else:
            z, zh, c, x, h, cls_id = self.get_input(batch, self.first_stage_key,
                                                    return_first_stage_outputs=False,
                                                    force_c_encode=True,
                                                    return_original_cond=False,
                                                    bs=N)
        #c = dict(c_concat=[c], c_crossattn=[cls_id])
        if self.model.conditioning_key == 'concat':
            c = {'c_concat': [c]}
            key = 'c_concat'
        elif self.model.conditioning_key == 'crossattn':
            c = {'c_crossattn': [c]}
            key = 'c_crossattn'
        else:  # hybrid
            c = {'c_concat': [c], 'c_crossattn': [c]}
            key = 'c_crossattn'

        N = min(x.shape[0], N)
        n_row = min(x.shape[0], n_row)

        # input segmentation mask
        # log["inputs_seg"] = x
        # input vertex heatmap
        # log["inputs_heat"] = h

        # Use make_grid to merge images into a grid, displaying z_dim images per row.
        # log["latent_seg"] = self.prepare_latent_to_log(z)  # 4x4x32x32 -> 16x1x32x32 -> 3x138x138
        # log["latent_heat"] = self.prepare_latent_to_log(zh)
        log["heat"] = h

        if return_first_stage_outputs:  # False
            log["reconstruction_seg"] = xrec
            log["reconstruction_heat"] = hrec
        if plot_conditioning_latent:  # False
            if self.model.conditioning_key is not None:  # concat
                log["conditioning_latent"] = self.prepare_latent_to_log(c[key][0])

        # visualize forward diffusion process
        if plot_diffusion_rows:  # False
            # get diffusion row of building segmentation mask
            diffusion_grid_seg, diffusion_grid_latent_seg = self.plot_diffusion_rows(z, n_row, ztype="segmentation")
            # seg reconstructed from diffused latent seg of intermediate steps (0, 200, 400, 600, 800, 999)
            log["diffusion_row_seg"] = diffusion_grid_seg
            log["diffusion_row_latent_seg"] = diffusion_grid_latent_seg
            diffusion_grid_heat, diffusion_grid_latent_heat = self.plot_diffusion_rows(zh, n_row, ztype="heatmap")
            # heatmap reconstructed from diffused latent heatmap of intermediate steps (0, 200, 400, 600, 800, 999)
            log["diffusion_row_heat"] = diffusion_grid_heat
            # diffused latent heatmap of intermediate steps (0, 200, 400, 600, 800, 999)
            log["diffusion_row_latent_heat"] = diffusion_grid_latent_heat

        # perform direct (directly use the model output as the final results)
        if sampler == 'direct':
            with self.ema_scope():
                noise_x = torch.randn_like(z)  # x_T
                noise_h = torch.randn_like(z)  # h_T
                final_t = torch.tensor([self.num_timesteps - 1], device=self.device).long()
                model_output = self.apply_model(noise_x, noise_h, final_t, c)
            _, d, _, _ = model_output.shape
            split_size = d // 2
            samples = self.predict_start_from_noise(noise_x, final_t,
                                                    noise=model_output[:, :split_size, :, :])  # latent x0
            samples_h = self.predict_start_from_noise(noise_h, final_t,
                                                      noise=model_output[:, split_size:, :, :])  # latent h0
            x_samples = self.decode_first_stage(samples, ztype="segmentation")
            h_samples = self.decode_first_stage(samples_h, ztype="heatmap")
            # reconstructed {seg mask/heatmap} from final sampled latent
            log["samples_seg_direct"] = x_samples
            log["samples_heat_direct"] = h_samples
            # final sampled latent from posterior
            log["samples_latent_seg_direct"] = samples
            log["samples_latent_heat_direct"] = samples_h

        # perform ddim
        if sampler == "ddim":
            with self.ema_scope("Plotting"):
                # ddim: steps=200
                samples, samples_h, z_denoise_row = self.sample_log(cond=c, batch_size=N, ddim=True,
                                                                    ddim_steps=ddim_steps, eta=ddim_eta)
                # z_denoise_row={'x_inter': [gaussian noise(4x32x32), ... ], sampled from posterior
                #                'pred_x0': [predicted latent seg(4x32x32), ...], predict start from noise
                #                'h_iter': [gaussian noise(4x32x32), ...]
                #                'pred_h0': [predicted latent heatmap(4x32x32), ...]
                #               }  # log every 200 steps
                # samples: final predicted latent seg (sampled from posterior)
                # samples_h: final predicted latent heatmap (sampled from posterior)
            x_samples = self.decode_first_stage(samples, ztype="segmentation")
            h_samples = self.decode_first_stage(samples_h, ztype="heatmap")
            # reconstructed {seg mask/heatmap} from final sampled latent
            log["samples_seg_ddim"] = x_samples
            log["samples_heat_ddim"] = h_samples
            # final sampled latent from posterior
            log["samples_latent_seg_ddim"] = samples
            log["samples_latent_heat_ddim"] = samples_h

            # visualize ddim intermediate results
            if plot_denoise_rows:  # False
                denoise_grid = self.get_denoise_row_from_list(z_denoise_row)
                log["denoise_row_ddim"] = denoise_grid

        # perform ddpm
        if sampler == "ddpm":
            with self.ema_scope("Plotting Progressives"):  # 表示在这个上下文块内，模型将使用EMA权重进行预测或评估。
                # 使用EMA一般会提升模型的泛化能力，有助于减小过拟合的风险。
                # ddpm (timesteps=1000)
                img, img_h, progressives, progressives_h = self.progressive_denoising(c,
                                                                                      shape=(
                                                                                      self.channels, self.image_size,
                                                                                      self.image_size),
                                                                                      batch_size=N)
                # img: 1x4x32x32
                # img: the final predicted latent seg using ddpm (sampled from posterior q(x_{t-1}|x_t), the 999th step)
                # progressives=[x0_partial, ...], log every 200 steps during ddpm, len=6
                # x0_partial: predict noise from start

            prog_row_seg, prog_row_latent_seg = self.get_denoise_row_from_list(progressives,
                                                                               desc="Progressive Generation",
                                                                               ztype="segmentation")
            log["progressive_row_seg_ddpm"] = prog_row_seg
            log["progressive_row_latent_seg_ddpm"] = prog_row_latent_seg

            x_samples = self.decode_first_stage(img, ztype="segmentation")
            h_samples = self.decode_first_stage(img_h, ztype="heatmap")
            log["samples_seg_ddpm"] = x_samples
            log["samples_heat_ddpm"] = h_samples

            prog_row_heat, prog_row_latent_heat = self.get_denoise_row_from_list(progressives_h,
                                                                                 desc="Progressive Generation",
                                                                                 ztype="heatmap")
            log["progressive_row_heat_ddpm"] = prog_row_heat
            log["progressive_row_latent_heat_ddpm"] = prog_row_latent_heat

        return log

    def plot_diffusion_rows(self, z, n_row, ztype):
        diffusion_row = list()
        diffusion_row_latent = list()
        z_start = z[:n_row]
        for t in range(self.num_timesteps):
            # self.log_every_t=200, self.num_timesteps=1000
            if t % self.log_every_t == 0 or t == self.num_timesteps - 1:
                t = repeat(torch.tensor([t]), '1 -> b', b=n_row)
                t = t.to(self.device).long()
                noise = torch.randn_like(z_start)
                z_noisy = self.q_sample(x_start=z_start, t=t, noise=noise)
                diffusion_row_latent.append(z_noisy)
                diffusion_row.append(self.decode_first_stage(z_noisy, ztype=ztype))

        diffusion_row = torch.stack(diffusion_row)  # n_log_step, n_row, C, H, W
        diffusion_grid = rearrange(diffusion_row, 'n b c h w -> b n c h w')
        diffusion_grid = rearrange(diffusion_grid, 'b n c h w -> (b n) c h w')
        diffusion_grid = make_grid(diffusion_grid, nrow=diffusion_row.shape[0])

        diffusion_row_latent = torch.stack(diffusion_row_latent)  # n_log_step, n_row, C, H, W
        diffusion_grid_latent = rearrange(diffusion_row_latent, 'n b c h w -> b n c h w')
        diffusion_grid_latent = rearrange(diffusion_grid_latent, 'b n c h w -> (b n) c h w')
        diffusion_grid_latent = make_grid(diffusion_grid_latent, nrow=diffusion_row_latent.shape[0])
        return diffusion_grid, diffusion_grid_latent


# different from RoomFormer/s3d_floorplan_eval/Evaluator/Evaluator.py
def calculate_precision_recall(pred_coords, gt_coords, threshold=10):
    # 如果pred_coords和gt_coords都为空，表示没有预测和真值，返回完美结果
    if len(pred_coords) == 0 and len(gt_coords) == 0:
        return 1.0, 1.0  # 完美的Precision和Recall

    # 如果其中一个为空，返回Precision或Recall为0
    if len(pred_coords) == 0:
        return 0.0, 0.0  # 没有预测顶点
    if len(gt_coords) == 0:
        return 0.0, 0.0  # 没有真值顶点

    # 使用cKDTree来加速最近邻搜索
    gt_tree = cKDTree(gt_coords)
    pred_tree = cKDTree(pred_coords)

    # 查找每个pred坐标的最近的gt坐标
    pred_distances, _ = gt_tree.query(pred_coords, distance_upper_bound=threshold)
    # 查找每个gt坐标的最近的pred坐标
    gt_distances, _ = pred_tree.query(gt_coords, distance_upper_bound=threshold)

    # True Positives (TP) - gt点与pred点的最近距离在阈值以内 (pred点中与gt匹配上的点)
    TP = np.sum(pred_distances <= threshold)

    # False Positives (FP) - pred点中没有匹配到任何gt点的点
    FP = np.sum(pred_distances > threshold)

    # False Negatives (FN) - gt点中没有匹配到任何pred点的点
    FN = np.sum(gt_distances > threshold)

    # 计算Precision和Recall
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0

    return precision, recall


def extract_vertices_from_heatmap(heatmap, th, topk=300):
    heatmap = torch.tensor(heatmap).unsqueeze(0).to(dtype=torch.float32)
    heatmap_nms = non_maximum_suppression(heatmap)  # torch.Size([1, 256, 256])
    height, width = heatmap_nms.size(1), heatmap_nms.size(2)
    heatmap_nms = heatmap_nms.reshape(-1)
    scores, index = torch.topk(heatmap_nms, k=topk)  # scores: torch.Size([topk])
    y = (index // width).float()
    x = (index % width).float()
    extracted_vertices = torch.stack((x, y)).t()  # torch.Size([topk, 2])
    return np.array(extracted_vertices[scores > th]), np.array(scores[scores > th])


def non_maximum_suppression(a):
    ap = F.max_pool2d(a, 3, stride=1, padding=1)
    mask = (a == ap).float().clamp(min=0.0)
    return a * mask

