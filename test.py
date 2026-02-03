"""
Batch inference (2D/3D) over a dataset directory using patch-based tiling.

Discovers subfolders under `<input_dir>/images`, runs inference on each volume
using the same patching/stitching pipeline as inference.py, and writes
predictions under `<input_dir>/masks_{model_stem}/<subfolder>`.

Dimensionality
- Inferred from `--inference_patch_size` z: z > 1 → 3D, z == 1 → 2D.

Input layout
  <input_dir>/images/<subfolder>/*.{tif,tiff,nii.gz,...}

Outputs
  <input_dir>/masks_{model_stem}/<subfolder>/...
  If `--output_type=scroll-tiff` (or 'scroll-nii'), files are moved up from the
  intermediate `<volume_name>_scroll` folder to the subfolder root.

Examples (3D)
  python test.py \
    --input_dir ./datas/c-Fos/LI-WIN_PAPER/testing-data/V60 \
    --model_path ./datas/c-Fos/LI-WIN_PAPER/weights/fun-3.pth \
    --inference_patch_size 16 64 64 \
    --inference_overlay 2 4 4 \
    --output_type scroll-tiff

Examples (2D, Windows caret)
  python test.py ^
    --input_dir ./datas/c-Fos/LI-WIN_PAPER/testing-data/V60 ^
    --model_path ./datas/c-Fos/LI-WIN_PAPER/weights/func-3_LI-AN-32.pth ^
    --inference_patch_size 1 32 32 ^
    --inference_overlay 0 16 16 ^
    --output_type scroll-tiff
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.intensity.dictionary import ScaleIntensityRanged, NormalizeIntensityd

from IO import FileReader, FileWriter, OUTPUT_CHOICES, TYPE_MAP
from inference.inferencer import Inferencer
from utils.datasets import MicroscopyDataset
from utils.stitcher import stitch_image
from utils.loader import load_model, compute_z_plan, load_inference_data


inference_transform = Compose([
    ScaleIntensityRanged(keys=["image"], a_min=0, a_max=1000, b_min=0.0, b_max=1.0, clip=True),
    NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
    ToTensord(keys=["image"], dtype=torch.float32),
])


def list_subfolders(folder: Path) -> List[Path]:
    if not folder.exists():
        return []
    return [p for p in sorted(folder.iterdir()) if p.is_dir()]


def move_scroll_results_up(mask_dir: Path, volume_name: str) -> None:
    """If writer created '<mask_dir>/<volume_name>_scroll', move files up to '<mask_dir>' and remove the folder."""
    scroll_dir = mask_dir / f"{volume_name}_scroll"
    if not scroll_dir.exists() or not scroll_dir.is_dir():
        return
    for f in scroll_dir.iterdir():
        if f.is_file():
            target = mask_dir / f.name
            try:
                if target.exists():
                    target.unlink()
                f.replace(target)
            except Exception as e:
                logging.warning(f"Could not move {f} to {target}: {e}")
    try:
        scroll_dir.rmdir()
    except OSError:
        pass


import json

def parse_args():
    p = argparse.ArgumentParser(description="Batch inference over images/<subfolder> using config file")
    p.add_argument("--config", type=str, default="configs/config.json", help="Path to a JSON config file (default: configs/config.json)")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f).get("test", {})

    input_dir = config.get("input_dir")
    model_path = config.get("model_path")
    
    if not input_dir or not model_path:
        logging.error("Missing mandatory paths in config (input_dir or model_path).")
        return 1

    base = Path(input_dir)
    model_stem = Path(model_path).stem
    images_root = base / "images"
    masks_root = base / f"masks_{model_stem}"

    logging.info(f"Loading model: {model_path}")
    model = load_model(model_path)
    inferencer = Inferencer(model)

    inference_patch_size = tuple(config.get("inference_patch_size", [1, 64, 64]))
    inference_overlay = tuple(config.get("inference_overlay", [0, 16, 16]))
    inference_resize_factor = tuple(config.get("inference_resize_factor", [1, 1, 1]))

    # Choose spatial dims based on depth
    spatial_dims = 3 if inference_patch_size[0] > 1 else 2

    # Auto-discover subfolders under images/
    if not images_root.exists():
        logging.error(f"Images root does not exist: {images_root}")
        return 1
    subfolders = list_subfolders(images_root)
    if not subfolders:
        logging.error(f"No subfolders found in {images_root}")
        return 1
    logging.info(f"Auto-discovered subfolders: {[p.name for p in subfolders]}")

    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 8)

    for sub in subfolders:
        img_path = str(sub)
        mask_path = str(masks_root / sub.name)
        os.makedirs(mask_path, exist_ok=True)

        logging.info(f"Reading input image from: {img_path}")
        data_reader = FileReader(img_path)
        output_type_str = config.get("output_type", "scroll-tiff")
        output_type = TYPE_MAP.get(output_type_str)
        data_writer = FileWriter(
            output_path=mask_path,
            output_name=data_reader.volume_name,
            output_type=output_type,
            output_dtype=config.get("output_dtype", "uint16"),
            full_res_shape=data_reader.volume_shape,
            file_name=data_reader.volume_files,
            chunk_size=tuple(config.get("output_chunk_size", [128, 128, 128])),
            resize_factor=config.get("output_resize_factor", 2),
            resize_order=config.get("output_resize_order", 0),
            n_level=config.get("output_n_level", 5),
        )

        logging.info("Inferencing ...")
        prev_z_slices = None
        z_plan = compute_z_plan(data_reader.volume_shape[0], inference_patch_size[0], inference_overlay[0])

        for z_start, z_overlay in z_plan:
            inference_patches, data_position = load_inference_data(
                data_reader=data_reader,
                z_start=z_start,
                patch_size=inference_patch_size, 
                overlay=inference_overlay,
                resize_factor=inference_resize_factor,
            )

            inference_dataset = MicroscopyDataset(
                inference_patches,
                transform=inference_transform,
                spatial_dims=spatial_dims,
                with_mask=False,
            )
            inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

            mask_patches = inferencer.eval(inference_loader)

            stitched_volume, prev_z_slices = stitch_image(
                patches=mask_patches,
                positions=data_position,
                original_shape=(inference_patch_size[0], data_reader.volume_shape[1], data_reader.volume_shape[2]),
                patch_size=inference_patch_size,
                z_overlay=z_overlay,
                prev_z_slices=prev_z_slices,
                resize_factor=inference_resize_factor,
            )

            data_writer.write(stitched_volume, z_start=z_start, z_end=z_start + stitched_volume.shape[0])

        if output_type_str in ["scroll-tiff", 'scroll-nii']:
            move_scroll_results_up(Path(mask_path), data_reader.volume_name)

    logging.info(f"Finish results under: {masks_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
