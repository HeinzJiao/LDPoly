import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np
from ldm.util import instantiate_from_config
import argparse, os, sys, datetime, glob, importlib, csv
from ldm.models.diffusion.ddpm import disabled_train
from einops import rearrange, repeat
#from ldm.data.cvc import CVCBase
from PIL import Image
from ldm.modules.distributions.distributions import normal_kl, DiagonalGaussianDistribution
import cv2
from torchvision.utils import make_grid


# borrowed from class LatentDiffusion(DDPM)
# instantiate pretrained Autoencoder
def instantiate_first_stage(config):
    model = instantiate_from_config(config)
    first_stage_model = model.eval()
    first_stage_model.train = disabled_train
    for param in first_stage_model.parameters():
        param.requires_grad = False
    return model


class PretrainedAutoencoder(pl.LightningModule):
    def __init__(self, first_stage_config, scale_factor=1.0, scale_by_std=True, **kwargs):
        super().__init__(**kwargs)
        self.first_stage_model = instantiate_first_stage(first_stage_config)
        self.scale_by_std = scale_by_std
        if not scale_by_std:  # False
            self.scale_factor = scale_factor
        else:  # True
            self.register_buffer('scale_factor', torch.tensor(scale_factor))

    def get_input(self, batch, k):
        x = batch[k]  # bx256x256x3
        if len(x.shape) == 3:
            x = x[..., None]
        x = rearrange(x, 'b h w c -> b c h w')
        x = x.to(memory_format=torch.contiguous_format).float()  # bx3x256x256
        return x

    def encode_first_stage(self, x):
        return self.first_stage_model.encode(x)

    def get_first_stage_encoding(self, encoder_posterior):
        if isinstance(encoder_posterior, DiagonalGaussianDistribution):
            z = encoder_posterior.sample()
        elif isinstance(encoder_posterior, torch.Tensor):
            z = encoder_posterior
        else:
            raise NotImplementedError(f"encoder_posterior of type '{type(encoder_posterior)}' not yet implemented")
        return self.scale_factor * z  # regularize

    def forward(self, batch):
        x = self.get_input(batch, k='segmentation')
        x = x.to(self.device)  # 1x3x256x256, range [-1, 1], binary mask
        file_path = batch['file_path_'][0]  # e.g. 'data/CVC/PNG/Original/437.png'
        file_path = file_path.replace('/Original/', '/Ground Truth/')  # 'data/CVC/PNG/Ground Truth/437.png'

        # autoencoder generate latent vector z (regularized by scale_factor)
        encoder_posterior = self.encode_first_stage(x)
        z = self.get_first_stage_encoding(encoder_posterior).detach()  # 1x4x32x32

        # autoencoder reconstruct original image
        xrec = self.decode_first_stage(z)

        # save latent vector z (regularized by scale_factor) as npy file
        z = z[0].squeeze(0).cpu().numpy()  # 4x32x32
        root = '/'.join(file_path.split('/')[:3])  # 'data/cvc/PNG'
        file_name = file_path.split('/')[-1][:-4]
        output_z_dir = os.path.join(root, 'z')  # 'data/cvc/PNG/z'
        if not os.path.exists(output_z_dir):
            os.makedirs(output_z_dir)
        output_z_path = os.path.join(output_z_dir, file_name)
        np.save(output_z_path, z)

        # visualize latent vector z (regularized by scale_factor)
        z = torch.tensor(z)  # 4x32x32
        z = z.unsqueeze(1)  # 4x1x32x32
        z = torch.clamp(z, -1, 1)
        z = ((z + 1) / 2) * 255
        grid = make_grid(z, nrow=z.shape[0])
        grid = rearrange(grid, 'c h w -> h w c')
        grid = grid.numpy().astype(np.uint8)
        output_zv_dir = os.path.join(root, 'zv')  # 'data/cvc/PNG/zv'
        if not os.path.exists(output_zv_dir):
            os.makedirs(output_zv_dir)
        output_zv_path = os.path.join(output_zv_dir, file_name+'.png')
        Image.fromarray(grid).save(output_zv_path)

        # save reconstructed image xrec
        xrec = xrec[0].squeeze(0).cpu().numpy()  # 3x256x256
        xrec = rearrange(xrec, 'c h w -> h w c')
        # main.py - class ImageLogger - def log_img - images[k] = torch.clamp(images[k], -1., 1.)
        xrec = np.clip(xrec, -1, 1)
        xrec = ((xrec + 1) / 2) * 255
        image = Image.fromarray(xrec.astype(np.uint8))
        output_rec_dir = os.path.join(root, 'xrec')  # 'data/cvc/PNG/xrec'
        if not os.path.exists(output_rec_dir):
            os.makedirs(output_rec_dir)
        output_rec_path = os.path.join(output_rec_dir, file_name+'.png')
        image.save(output_rec_path)

        return

    def decode_first_stage(self, z):
        z = 1. / self.scale_factor * z
        return self.first_stage_model.decode(z)

    def test_step(self, batch, batch_idx):
        # only for very first batch
        print("batch_idx: ", batch_idx)
        if batch_idx == 0:  # and not self.restarted_from_ckpt:
            print("##################################################")
            if self.scale_by_std:
                assert self.scale_factor == 1., 'rather not use custom rescaling and std-rescaling simultaneously'
                # set rescale weight to 1./std of encodings
                x = self.get_input(batch, k='segmentation')
                x = x.to(self.device)
                encoder_posterior = self.encode_first_stage(x)
                z = self.get_first_stage_encoding(encoder_posterior).detach()  # range roughly in (-18, 18)
                # print(z.shape, z.flatten().shape, z.min(), z.max(), z.flatten().std())
                del self.scale_factor
                self.register_buffer('scale_factor', 1. / z.flatten().std())  # sts3d: 1/7.8957, synapse: 1/8.7146
            print(f"setting self.scale_factor to {self.scale_factor}")
            print(f"### USING STD-RESCALING: \033[31m{self.scale_by_std}\033[0m ###")

        self.forward(batch)

        return


class CVC(Dataset):
    """CVC Dataset Base
    Notes:
        - `segmentation` is for the diffusion training stage (range binary -1 and 1)
        - `image` is for conditional signal to guided final seg-map (range -1 to 1)
    """
    def __init__(self, data_root, size=256, interpolation="nearest", transform=None, num_classes=2):
        self.data_root = data_root
        self.data_paths = self._parse_data_list()
        self._length = len(self.data_paths)
        self.labels = dict(file_path_=[path for path in self.data_paths])
        self.size = size  # 256
        self.interpolation = dict(nearest=Image.NEAREST)[interpolation]   # for segmentation slice
        self.transform = transform

    def __getitem__(self, i):
        # read segmentation and images
        example = dict((k, self.labels[k][i]) for k in self.labels)
        # segmentation = Image.open(example["file_path_"].replace("Original", "GroundTruth")).convert("RGB")
        # image = Image.open(example["file_path_"]).convert("RGB")    # same name, different postfix
        segmentation = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"].replace("Original", "Ground Truth")),cv2.COLOR_BGR2RGB))
        image = Image.fromarray(cv2.cvtColor(cv2.imread(example["file_path_"]),cv2.COLOR_BGR2RGB))

        # resize input image to 256x256
        if self.size is not None:
            segmentation = segmentation.resize((self.size, self.size), resample=Image.NEAREST)
            image = image.resize((self.size, self.size), resample=Image.BILINEAR)

        segmentation = (np.array(segmentation) > 128).astype(np.float32)

        example["segmentation"] = ((segmentation * 2) - 1)   # range: binary -1 and 1

        image = np.array(image).astype(np.float32) / 255.
        image = (image * 2.) - 1.                            # range from -1 to 1, np.float32
        example["image"] = image
        example["class_id"] = np.array([-1])  # doesn't matter for binary seg

        assert np.max(segmentation) <= 1. and np.min(segmentation) >= -1.
        assert np.max(image) <= 1. and np.min(image) >= -1.
        return example

    def __len__(self):
        return self._length

    def _parse_data_list(self): # 80% / 10% / 10%
        all_imgs = glob.glob(os.path.join(self.data_root, "*.png"))  # List['.png']
        return all_imgs


if __name__ == "__main__":
    # use to instantiate pretrained Autoencoder
    first_stage_config = {'target': 'ldm.models.autoencoder.AutoencoderKL',
                          'params': {'embed_dim': 4, 'monitor': 'val/rec_loss', 'ckpt_path': 'models/first_stage_models/kl-f8/model.ckpt',
                                     'ddconfig': {'double_z': True, 'z_channels': 4, 'resolution': 256, 'in_channels': 3,
                                                  'out_ch': 3, 'ch': 128, 'ch_mult': [1, 2, 4, 4], 'num_res_blocks': 2,
                                                  'attn_resolutions': [], 'dropout': 0.0},
                                     'lossconfig': {'target': 'torch.nn.Identity'}}}
    # instantiate model
    model = PretrainedAutoencoder(first_stage_config)

    # create dataloader
    dataset = CVC(data_root='data/CVC/PNG/Original')
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # instantiate trainer
    trainer = pl.Trainer(accelerator='gpu', gpus='0,', max_epochs=1)

    trainer.test(model, dataloaders=dataloader)






