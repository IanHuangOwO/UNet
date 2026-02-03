"""
Train a 2D/3D U-Net-style segmentation model on microscopy data using patch-based
sampling. Images and masks are loaded as volumes per subfolder, converted into
patches with optional overlap and resize, then split into train/validation sets.
During training, Dice+BCE loss and Dice score are tracked and figures are saved;
the best model checkpoint (by validation loss) is written to disk.
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import sys
import torch
import json

from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.spatial.dictionary import RandFlipd, RandZoomd, RandAffined
from monai.transforms.intensity.dictionary import (
    ScaleIntensityd, GaussianSmoothd, NormalizeIntensityd, 
    RandAdjustContrastd, RandBiasFieldd, RandShiftIntensityd, RandScaleIntensityd
)
from monai.transforms.post.dictionary import AsDiscreted
from monai.data.dataloader import DataLoader


from train.trainer import Trainer
from models.factory import get_model
from utils.datasets import MicroscopyDataset
from utils.loader import load_train_data
from utils.visualization import visualize_dataset


train_transform = Compose([
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
    GaussianSmoothd(keys=["mask"], sigma=0.2),
    AsDiscreted(keys=["mask"], threshold=0.5),
    RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    RandAdjustContrastd(keys=["image"], prob=0.3),
    RandBiasFieldd(keys=["image"], prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.3),
    RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
])

val_transform = Compose([
    ToTensord(keys=["image", "mask"], dtype=torch.float32),
    GaussianSmoothd(keys=["mask"], sigma=0.2),
    AsDiscreted(keys=["mask"], threshold=0.5),
    RandFlipd(keys=["image", "mask"], spatial_axis=1, prob=0.5),
    RandAdjustContrastd(keys=["image"], prob=0.3),
    RandBiasFieldd(keys=["image"], prob=0.2),
    RandShiftIntensityd(keys=["image"], offsets=0.2, prob=0.3),
    RandScaleIntensityd(keys=["image"], factors=0.2, prob=0.3),
])

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train 3D or 2D U-Net for Microscopy Segmentation"
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.json",
        help="Path to a JSON config file (default: configs/config.json)"
    )
    return parser.parse_args()
    
def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        full_config = json.load(f)
        
    config = full_config.get("train", {})
    model_config = full_config.get("model", {})

    img_root = config.get("img_path")
    mask_root = config.get("mask_path")
    save_root = config.get("save_path")
    model_name = config.get("model_name", "best_model")
    
    if not img_root or not mask_root or not save_root:
        logging.error("Missing mandatory paths in config (img_path, mask_path, or save_path).")
        return 1
        
    # Create the model-specific output directory immediately
    model_save_path = os.path.join(save_root, model_name)
    os.makedirs(model_save_path, exist_ok=True)

    # Add file handler to logger inside the model-specific folder
    log_path = os.path.join(model_save_path, "train.log")
    file_handler = logging.FileHandler(log_path, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(file_handler)

    logging.info("Loading image and mask data...")
    
    training_patch_size = tuple(config.get("training_patch_size", [1, 64, 64]))
    training_overlay = tuple(config.get("training_overlay", [0, 0, 0]))
    training_resize_factor = tuple(config.get("training_resize_factor", [1.0, 1.0, 1.0]))

    # Determine spatial dimensions
    if training_patch_size[0] > 1:
        spatial_dims = 3
        logging.info("Auto-selected 3D model/dataset (patch depth > 1)")
    else:
        spatial_dims = 2
        logging.info("Auto-selected 2D model/dataset (patch depth == 1)")
        
    train_patches, val_patches = load_train_data(
        img_path=img_root,
        mask_path=mask_root,
        patch_size=training_patch_size,
        overlay=training_overlay,
        resize_factor=training_resize_factor,
        balance=True,
        val_ratio=0.3,
        seed=100,
    )
    
    train_dataset = MicroscopyDataset(train_patches, transform=train_transform, spatial_dims=spatial_dims, with_mask=True)
    val_dataset = MicroscopyDataset(val_patches, transform=val_transform, spatial_dims=spatial_dims, with_mask=True)

    # Optional preview of transformed samples
    if config.get("visualize_preview", False):
        logging.info("Saving dataset preview...")
        try:
            visualize_dataset(train_dataset, title="train_samples_preview", save_path=model_save_path)
            visualize_dataset(val_dataset, title="validation_samples_preview", save_path=model_save_path)
        except Exception as e:
            logging.warning("Failed to visualize preview: %s", str(e))
    
    batch_size = config.get("training_batch_size", 8)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True)
    
    model = get_model(
        model_type=model_config.get("model_type", "monai_unet"),
        spatial_dims=spatial_dims,
        in_channels=model_config.get("in_channels", 1),
        out_channels=model_config.get("out_channels", 1),
        channels=model_config.get("channels", [32, 64, 128]),
        strides=model_config.get("strides", [2, 2]),
        num_res_units=model_config.get("num_res_units", 2),
        dropout=model_config.get("dropout", 0.1)
    )
    
    trainer = Trainer(
        model=model,
        train_loader=train_loader, 
        val_loader=val_loader, 
        save_path=save_root, # Trainer will append model_name internally
        model_name=model_name,
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
        epochs=config.get("training_epochs", 30)
    )

    logging.info("Starting training...")
    trainer.train(epochs=config.get("training_epochs", 30))
    logging.info("Training completed. Model and logs saved to %s", model_save_path)

if __name__ == "__main__":
    sys.exit(main())
