"""
Main training script for LDPoly latent diffusion models.

Example usage:
    python -u main.py \
        --base configs/latent-diffusion/deventer_road_mask_vertex_heatmap-ldm-kl-8.yaml \
        -t --gpus 0, \
        --name deventer_r_v-ldm-kl-8
"""

import os
import sys
import glob
import time
import signal
import datetime
import argparse

import numpy as np
import torch
import torchvision
from PIL import Image

import pytorch_lightning as pl
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import Callback, ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.utilities import rank_zero_only, rank_zero_info

from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader
from functools import partial

from ldm.data.base import Txt2ImgIterableBaseDataset
from ldm.util import instantiate_from_config

# -------------------------------------------------------------------------
# Argument parsing
# -------------------------------------------------------------------------
def get_parser(**parser_kwargs):
    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir",
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="base_config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser


def nondefault_trainer_args(opt):
    """Return all Trainer args that were set explicitly via CLI."""
    parser = argparse.ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args([])
    return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))


# -------------------------------------------------------------------------
# Dataset / DataModule utilities
# -------------------------------------------------------------------------
class WrappedDataset(Dataset):
    """Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset"""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def worker_init_fn(_):
    """Custom worker init function to split iterable dataset indices per worker."""
    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)


class DataModuleFromConfig(pl.LightningDataModule):
    """
    A custom implementation of PyTorch Lightning’s `LightningDataModule`, designed to dynamically construct
    training, validation, test, and prediction DataLoaders from a `config.yaml` file.

    Main features:
        - Builds datasets and DataLoaders based on the 'train', 'validation', 'test', and 'predict' sections in the config.
        - Automatically handles batch size, number of workers, and optional `worker_init_fn` settings.

    Notes:
        Although `train_dataloader()`, `val_dataloader()`, etc. are not explicitly
        called in `main.py`, PyTorch Lightning invokes them internally. For example:

            trainer.fit(model, datamodule)

        will automatically trigger:
            datamodule.prepare_data()
            datamodule.setup()
            datamodule.train_dataloader()
            datamodule.val_dataloader()
    """
    def __init__(self, batch_size, train=None, validation=None, test=None, predict=None,
                 wrap=False, num_workers=None, shuffle_test_loader=False, use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2
        self.use_worker_init_fn = use_worker_init_fn
        if train is not None:
            self.dataset_configs["train"] = train
            self.train_dataloader = self._train_dataloader
        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)
        if test is not None:
            self.dataset_configs["test"] = test
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)
        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader
        self.wrap = wrap

    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)

    def setup(self, stage=None):
        # Instantiate datasets
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs)
        if self.wrap:  # False
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])

    # --- dataloaders ------------------------------------------------------
    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)  # False
        if is_iterable_dataset or self.use_worker_init_fn:  # False False
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn, pin_memory=True)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle, pin_memory=True)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle, pin_memory=True)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)


# -------------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------------
class SetupCallback(Callback):
    """
    Basic setup callback that creates log/ckpt/cfg dirs and saves config files.
    """

    def __init__(self, resume, now, logdir, ckptdir, cfgdir, config, lightning_config):
        super().__init__()
        self.resume = resume
        self.now = now
        self.logdir = logdir
        self.ckptdir = ckptdir
        self.cfgdir = cfgdir
        self.config = config
        self.lightning_config = lightning_config

    def on_keyboard_interrupt(self, trainer, pl_module):
        if trainer.global_rank == 0:
            print("Summoning checkpoint.")
            # ckpt_path = os.path.join(self.ckptdir, "last.ckpt")
            # trainer.save_checkpoint(ckpt_path)

    def on_pretrain_routine_start(self, trainer, pl_module):
        if trainer.global_rank == 0:
            # Create logdirs and save configs
            os.makedirs(self.logdir, exist_ok=True)
            os.makedirs(self.ckptdir, exist_ok=True)
            os.makedirs(self.cfgdir, exist_ok=True)

            if "callbacks" in self.lightning_config:
                if 'metrics_over_trainsteps_checkpoint' in self.lightning_config['callbacks']:
                    os.makedirs(os.path.join(self.ckptdir, 'trainstep_checkpoints'), exist_ok=True)
            print("Project config")
            print(OmegaConf.to_yaml(self.config))
            OmegaConf.save(self.config,
                           os.path.join(self.cfgdir, "{}-project.yaml".format(self.now)))

            print("Lightning config")
            print(OmegaConf.to_yaml(self.lightning_config))
            OmegaConf.save(OmegaConf.create({"lightning": self.lightning_config}),
                           os.path.join(self.cfgdir, "{}-lightning.yaml".format(self.now)))

        else:
            # ModelCheckpoint callback created log directory --- remove it
            if not self.resume and os.path.exists(self.logdir):
                dst, name = os.path.split(self.logdir)
                dst = os.path.join(dst, "child_runs", name)
                os.makedirs(os.path.split(dst)[0], exist_ok=True)
                try:
                    os.rename(self.logdir, dst)
                except FileNotFoundError:
                    pass


class ImageLogger(Callback):
    """
    A custom PyTorch Lightning Callback for logging and saving images during
    training and validation.

    Functionality:
        - Periodically visualizes model inputs, outputs, intermediate results,
          and diffusion steps.
        - Supports saving images to disk and/or logging them via the Lightning
          logger (e.g., TensorBoard, TestTube).
        - Optionally records segmentation-related metrics such as
          `val_avg_dice` and `val_avg_iou`.
    """

    def __init__(self, batch_frequency, max_images, clamp=False, increase_log_steps=True,
                 rescale=True, disabled=False, log_on_batch_idx=False, log_first_step=True,
                 log_images_kwargs=None, log_dice_frequency=None):
        super().__init__()
        self.batch_freq = batch_frequency
        self.max_images = max_images
        self.clamp = clamp
        self.rescale = rescale
        self.disabled = disabled
        self.log_on_batch_idx = log_on_batch_idx
        self.log_first_step = log_first_step
        self.log_images_kwargs = log_images_kwargs or {}
        self.log_dice_frequency = log_dice_frequency

        self.logger_log_images = {
            pl.loggers.TestTubeLogger: self._testtube,
        }

        # log_steps 控制训练初期的频繁可视化
        self.log_steps = [2 ** n for n in range(int(np.log2(self.batch_freq)) + 1)] if increase_log_steps else [self.batch_freq]

    @rank_zero_only
    def _testtube(self, pl_module, images, batch_idx, split):
        for k in images:
            grid = torchvision.utils.make_grid(images[k])
            grid = (grid + 1.0) / 2.0  # 将 [-1,1] 缩放到 [0,1]
            tag = f"{split}/{k}"
            pl_module.logger.experiment.add_image(tag, grid, global_step=pl_module.global_step)

    @rank_zero_only
    def log_local(self, save_dir, split, images, global_step, current_epoch, batch_idx):
        """
        Saves model-generated images to a local directory for visual inspection
        during training or validation.

        Directory structure:
            {save_dir}/images/{split}/{image_key}_gs-{step}_e-{epoch}_b-{batch}.png

        Processing steps:
            - Optionally rescales images from [-1, 1] to [0, 1]
            - Converts tensors to NumPy arrays and writes them as PNG files
        """
        root = os.path.join(save_dir, "images", split)
        for k in images:
            grid = torchvision.utils.make_grid(images[k], nrow=4)
            if self.rescale:
                grid = (grid + 1.0) / 2.0  # -1,1 -> 0,1; c,h,w
            grid = grid.transpose(0, 1).transpose(1, 2).squeeze(-1).numpy()  # e.g. 138x138x3
            grid = (grid * 255).astype(np.uint8)
            filename = f"{k}_gs-{global_step:06}_e-{current_epoch:06}_b-{batch_idx:06}.png"
            path = os.path.join(root, filename)
            os.makedirs(os.path.split(path)[0], exist_ok=True)
            Image.fromarray(grid).save(path)

    @rank_zero_only
    def log_img(self, pl_module, batch, batch_idx, split="train"):
        """
        Logs image outputs during training or validation by:

            1. Saving images to disk via `log_local()`
            2. Sending images to the Lightning logger (e.g., TensorBoard)

        Logging is triggered only when:
            - The current step satisfies the logging frequency (via `check_frequency`)
            - `pl_module` implements a `log_images(batch, ...)` method
        """
        check_idx = batch_idx if self.log_on_batch_idx else pl_module.global_step
        if (self.check_frequency(check_idx) and hasattr(pl_module, "log_images") and callable(pl_module.log_images) and self.max_images > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                images = pl_module.log_images(batch, split=split, **self.log_images_kwargs)
            for k in images:
                images[k] = images[k][:min(images[k].shape[0], self.max_images)].detach().cpu()
                if self.clamp:
                    images[k] = torch.clamp(images[k], -1., 1.)

            # save images to save_dir
            self.log_local(pl_module.logger.save_dir, split, images, pl_module.global_step, pl_module.current_epoch, batch_idx)
            self.logger_log_images.get(logger, lambda *args, **kwargs: None)(pl_module, images, pl_module.global_step, split)
            if is_train:
                pl_module.train()

        # 可选：记录 val_avg_dice / val_avg_iou
        if (self.log_dice_frequency and check_idx % self.log_dice_frequency == 0 and hasattr(pl_module, "log_dice") and callable(pl_module.log_dice) and self.max_images > 0 and check_idx > 0):
            logger = type(pl_module.logger)
            is_train = pl_module.training
            if is_train:
                pl_module.eval()
            with torch.no_grad():
                metrics_dict, seg_label_dict = pl_module.log_dice()
            dice_list = metrics_dict.get("val_avg_dice", [])
            for k in seg_label_dict:
                seg_label_dict[k] = seg_label_dict[k][:self.max_images].detach().cpu()
                if self.clamp:
                    seg_label_dict[k] = torch.clamp(seg_label_dict[k], -1., 1.)
            self.logger_log_images.get(logger, lambda *args, **kwargs: None)(pl_module, seg_label_dict, pl_module.global_step, split)
            for key, value in metrics_dict.items():
                if key in ["val_avg_dice", "val_avg_iou"]:
                    pl_module.log(key, sum(value) / len(value), prog_bar=False, logger=True, on_step=True, on_epoch=False)
                elif "val_avg_dice" in key or "val_avg_iou" in key:
                    pl_module.log(key, value, prog_bar=False, logger=True, on_step=True, on_epoch=False)
            if is_train:
                pl_module.train()

    def check_frequency(self, check_idx):
        if ((check_idx % self.batch_freq) == 0 or (check_idx in self.log_steps)) and (check_idx > 0 or self.log_first_step):
            try:
                self.log_steps.pop(0)
            except IndexError:
                pass
            return True
        return False

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and (pl_module.global_step > 0 or self.log_first_step):
            self.log_img(pl_module, batch, batch_idx, split="train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.disabled and pl_module.global_step > 0:
            self.log_img(pl_module, batch, batch_idx, split="val")
        if hasattr(pl_module, 'calibrate_grad_norm') and pl_module.calibrate_grad_norm and batch_idx % 25 == 0 and batch_idx > 0:
            self.log_gradients(trainer, pl_module, batch_idx=batch_idx)

    def on_train_epoch_end(self, trainer, pl_module):
        pass  # 可选扩展：记录 epoch 结束后的统计信息


class CUDACallback(Callback):
    # see https://github.com/SeanNaren/minGPT/blob/master/mingpt/callback.py
    def on_train_epoch_start(self, trainer, pl_module):
        # Reset the memory use counter
        torch.cuda.reset_peak_memory_stats(trainer.root_gpu)
        torch.cuda.synchronize(trainer.root_gpu)
        self.start_time = time.time()

    def on_train_epoch_end(self, trainer, pl_module, outputs):
        torch.cuda.synchronize(trainer.root_gpu)
        max_memory = torch.cuda.max_memory_allocated(trainer.root_gpu) / 2 ** 20
        epoch_time = time.time() - self.start_time

        try:
            max_memory = trainer.training_type_plugin.reduce(max_memory)
            epoch_time = trainer.training_type_plugin.reduce(epoch_time)

            rank_zero_info(f"Average Epoch time: {epoch_time:.2f} seconds")
            rank_zero_info(f"Average Peak memory {max_memory:.2f}MiB")
        except AttributeError:
            pass


# -------------------------------------------------------------------------
# Signal handlers (for quick checkpoint / debug)
# -------------------------------------------------------------------------
def melk(*args, **kwargs):
    """User signal (SIGUSR1) handler: can be extended to save a checkpoint."""
    if trainer.global_rank == 0:
        print("Summoning checkpoint.")
        # Optionally: trainer.save_checkpoint(os.path.join(ckptdir, "last.ckpt"))

def divein(*args, **kwargs):
    """User signal (SIGUSR2) handler: drop into debugger."""
    if trainer.global_rank == 0:
        import pudb; pudb.set_trace()


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------
if __name__ == "__main__":
    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())  # Ensure local modules can be found

    # --------------------------------- CLI & basic setup ---------------------------------
    parser = get_parser()
    parser = Trainer.add_argparse_args(parser)
    opt, unknown = parser.parse_known_args()

    # Handle resume mode
    if opt.name and opt.resume:
        raise ValueError("Cannot specify both --name and --resume. Use --resume_from_checkpoint with --name if needed.")

    if opt.resume:
        if not os.path.exists(opt.resume):
            raise ValueError(f"Cannot find {opt.resume}")
        ckpt = opt.resume if os.path.isfile(opt.resume) else os.path.join(opt.resume.rstrip("/"), "checkpoints", "last.ckpt")
        logdir = os.path.dirname(os.path.dirname(ckpt))
        base_configs = sorted(glob.glob(os.path.join(logdir, "configs/*.yaml")))
        opt.resume_from_checkpoint = ckpt
        opt.base = base_configs + opt.base
        nowname = os.path.basename(logdir)
    else:
        name = f"_{opt.name}" if opt.name else f"_{os.path.splitext(os.path.basename(opt.base[0]))[0]}" if opt.base else ""
        nowname = now + name + opt.postfix
        logdir = os.path.join(opt.logdir, nowname)

    # Set checkpoint and config directories
    ckptdir = os.path.join(logdir, "checkpoints")
    cfgdir = os.path.join(logdir, "configs")
    seed_everything(opt.seed)

    # --------------------------------- Load and merge configuration files ---------------------------------
    # `opt.base` is a list containing one or more config file paths provided via the command-line flag `--base`.
    configs = [OmegaConf.load(cfg) for cfg in opt.base]

    # Convert unparsed command-line arguments (those not handled by argparse) into a dictionary-style OmegaConf object.
    cli = OmegaConf.from_dotlist(unknown)

    # Merge all configuration sources:
    #   - multiple YAML config files (`configs`)
    #   - command-line overrides (`cli`)
    config = OmegaConf.merge(*configs, cli)

    # Extract lightning-specific config
    lightning_config = config.pop("lightning", OmegaConf.create())

    # Update num_classes in datasets
    for split in ["train", "validation", "test"]:
        config.data.params[split].params.update({"num_classes": config.model.params.num_classes})

    # Prepare trainer configuration
    trainer_config = lightning_config.get("trainer", OmegaConf.create())
    trainer_config["accelerator"] = "gpu"
    for k in nondefault_trainer_args(opt):
        trainer_config[k] = getattr(opt, k)
    trainer_opt = argparse.Namespace(**trainer_config)
    lightning_config.trainer = trainer_config
    cpu = "gpus" not in trainer_config

    # --------------------------------- Instantiate model ---------------------------------
    model = instantiate_from_config(config.model)

    # --------------------------------- Logger setup（日志记录器设置） ---------------------------------
    # Configure the PyTorch Lightning logger.
    # The default logger is TestTubeLogger, unless overridden in `config.yaml`.
    logger_cfg = OmegaConf.merge({
        "target": "pytorch_lightning.loggers.TestTubeLogger",
        "params": {
            "name": "testtube",
            "save_dir": logdir
        }
    }, lightning_config.get("logger", OmegaConf.create()))  # 从 config 中提取 logger 配置（可选）

    # Instantiate the logger
    logger = instantiate_from_config(logger_cfg)

    # --------------------------------- Checkpoint callback setup ---------------------------------
    # Configure the ModelCheckpoint callback.
    # This controls checkpoint saving for:
    #   - Resuming training
    #   - Keeping the best-performing model for evaluation/inference
    default_modelckpt_cfg = {
        "target": "pytorch_lightning.callbacks.ModelCheckpoint",
        "params": {
            "dirpath": ckptdir,
            "verbose": True,
            "save_last": True,
        }
    }

    # Optional checkpoint strategies depending on `model.monitor`:
    if hasattr(model, "monitor"):
        monitor_key = model.monitor
        print(f"Checkpoint monitor: {monitor_key}")

        if monitor_key == "val_avg_dice":
            # Save the best model according to validation Dice score
            default_modelckpt_cfg["params"].update(
                monitor=monitor_key,
                save_top_k=True,
                mode="max",
                filename="{epoch:02d}-{val_avg_dice:.4f}",
                auto_insert_metric_name=False,
            )
        elif monitor_key == "every_N_epochs":
            # Save a checkpoint every N epochs (no metric monitoring)
            every_n = getattr(config.model.params, "checkpoint_every_n_epochs", 10)
            default_modelckpt_cfg["params"].update(
                save_top_k=False,
                every_n_epochs=every_n,  # save every N epochs
                filename="epoch={epoch:02d}",
            )
        else:
            raise NotImplementedError(f"Unsupported monitor: {monitor_key}")

    # Merge user-defined checkpoint settings from config.yaml
    user_modelckpt_cfg = lightning_config.get("modelcheckpoint", OmegaConf.create())
    modelckpt_cfg = OmegaConf.merge(default_modelckpt_cfg, user_modelckpt_cfg)

    # --------------------------- Callback setup --------------------------------
    # Build the full set of PyTorch Lightning callbacks.
    # Callbacks automate tasks such as:
    #   - Saving checkpoints
    #   - Monitoring learning rate
    #   - Logging images
    #   - Setting up output directories
    callbacks_cfg = OmegaConf.merge({
        "checkpoint_callback": modelckpt_cfg,
        "setup_callback": {
            "target": "main.SetupCallback",
            "params": {
                "resume": opt.resume,
                "now": now,
                "logdir": logdir,
                "ckptdir": ckptdir,
                "cfgdir": cfgdir,
                "config": config,
                "lightning_config": lightning_config,
            }
        },
        "image_logger": {
            "target": "main.ImageLogger",
            "params": {"batch_frequency": 750, "max_images": 4, "clamp": True}
        },
        "learning_rate_logger": {
            "target": "main.LearningRateMonitor",
            "params": {"logging_interval": "step"}  # 每个 step 记录一次 lr
        },
        "cuda_callback": {"target": "main.CUDACallback"}  # CUDACallback：打印显存信息 / 设备状态，辅助 debug（可自定义）
    }, lightning_config.get("callbacks", OmegaConf.create()))

    # instantiate all callbacks
    callbacks = [instantiate_from_config(callbacks_cfg[k]) for k in callbacks_cfg]

    # --------------------------------- Initialize Trainer ---------------------------------
    trainer = Trainer.from_argparse_args(trainer_opt, logger=logger, callbacks=callbacks)

    trainer.logdir = logdir

    # --------------------------------- Load data ---------------------------------
    data = instantiate_from_config(config.data)
    data.prepare_data()
    data.setup()

    print("#### Data #####")
    for k in data.datasets:
        print(f"{k}, {data.datasets[k].__class__.__name__}, {len(data.datasets[k])}")

    # --------------------------------- Learning rate scaling ---------------------------------
    bs = config.data.params.batch_size
    base_lr = config.model.base_learning_rate
    ngpu = 1 if cpu else len(str(trainer_config.gpus).strip(",").split(","))
    accumulate = trainer_config.get("accumulate_grad_batches", 1)
    if opt.scale_lr:
        model.learning_rate = accumulate * ngpu * bs * base_lr
        print(f"Setting learning rate to {model.learning_rate:.2e} = {accumulate}*{ngpu}*{bs}*{base_lr:.2e}")
    else:
        model.learning_rate = base_lr
        print("++++ NOT USING LR SCALING ++++")
        print(f"Setting learning rate to {model.learning_rate:.2e}")

    # Signal handlers for debugging and checkpointing
    signal.signal(signal.SIGUSR1, melk)
    signal.signal(signal.SIGUSR2, divein)

    # --------------------------------- Run training/testing ---------------------------------
    try:
        if opt.train:
            trainer.fit(model, data)
        if not opt.no_test and not trainer.interrupted:
            trainer.test(model, data)
    except Exception:
        if opt.debug and trainer.global_rank == 0:
            try:
                import pudb as debugger
            except ImportError:
                import pdb as debugger
            debugger.post_mortem()
        raise
    finally:
        if opt.debug and not opt.resume and trainer.global_rank == 0:
            dst = os.path.join(os.path.dirname(logdir), "debug_runs", os.path.basename(logdir))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            os.rename(logdir, dst)
        if trainer.global_rank == 0:
            print(trainer.profiler.summary())
