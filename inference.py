"""
Single-volume inference with patch-based tiling and stitching.

Runs a trained model on a single 2D/3D volume by splitting the input into
overlapping patches, predicting per patch, and stitching back to the original
shape. Dimensionality is inferred from the z-size of `--inference_patch_size`
(z > 1 → 3D, z == 1 → 2D).
"""
# Setup logging
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import sys

import torch
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord
from monai.transforms.intensity.dictionary import ScaleIntensityd, NormalizeIntensityd


inference_transform = Compose([
    ToTensord(keys=["image"], dtype=torch.float32),
])


import json

from utils.loader import load_model, load_inference_data, compute_z_plan
from IO.reader import FileReader
from IO.writer import FileWriter
from IO.IO_types import TYPE_MAP
from utils.stitcher import stitch_image
from utils.datasets import MicroscopyDataset
from inference.inferencer import Inferencer

def parse_args():
    parser = argparse.ArgumentParser(
        description="3D Mask Inference: Applies a trained model to infer masks from 3D images."
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.json",
        help="Path to a JSON config file (default: configs/config.json)"
    )
    return parser.parse_args()

def main():
    args = parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f).get("inference", {})
    
    input_path = config.get("input_path")
    output_path = config.get("output_path")
    model_path = config.get("model_path")
    
    if not input_path or not output_path or not model_path:
        logging.error("Missing mandatory paths in config (input_path, output_path, or model_path).")
        return 1
        
    os.makedirs(output_path, exist_ok=True)
    
    logging.info("Loading model from: %s", model_path)
    
    model = load_model(model_path)
    inferencer = Inferencer(model)
    
    inference_patch_size = tuple(config.get("inference_patch_size", [16, 64, 64]))
    inference_overlay = tuple(config.get("inference_overlay", [2, 4, 4]))
    inference_resize_factor = tuple(config.get("inference_resize_factor", [1.0, 1.0, 1.0]))
    
    # Choose spatial dims based on depth
    spatial_dims = 3 if inference_patch_size[0] > 1 else 2
    
    prev_z_slices = None
    
    logging.info(f"Reading input image from: {input_path}")
    
    data_reader = FileReader(input_path)
    output_type_str = config.get("output_type", "Scroll-Tif")
    # Try mapping from human-friendly label, otherwise use string directly
    output_type = TYPE_MAP.get(output_type_str, output_type_str)
    
    data_writer = FileWriter(
        output_path=output_path,
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
    
    z_plan = compute_z_plan(data_reader.volume_shape[0], inference_patch_size[0], inference_overlay[0])
    
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 8)
    
    for z_start, z_overlay in z_plan:
        # Calculate actual depth to read (handles case where volume is smaller than patch size)
        z_end = min(z_start + inference_patch_size[0], data_reader.volume_shape[0])
        actual_chunk_depth = z_end - z_start

        inference_patches, data_position = load_inference_data(
            data_reader=data_reader, 
            z_start=z_start,
            patch_size=inference_patch_size,
            overlay=inference_overlay,
            resize_factor=inference_resize_factor
        )
        
        inference_dataset = MicroscopyDataset(
            inference_patches,
            transform=inference_transform,
            spatial_dims=spatial_dims,
            with_mask=False,
        )
        inference_loader = DataLoader(inference_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        logging.info(f"Inferencing Z:{z_start}-{z_end} ...")
        mask_patches = inferencer.eval(inference_loader)
        
        logging.info(f"Stitching ...")
        stitched_volume, prev_z_slices = stitch_image(
            patches=mask_patches, 
            positions=data_position,
            original_shape=(actual_chunk_depth, data_reader.volume_shape[1], data_reader.volume_shape[2]),
            patch_size=inference_patch_size,
            z_overlay=z_overlay,
            prev_z_slices=prev_z_slices,
            resize_factor=inference_resize_factor,
        )
        
        data_writer.write(stitched_volume, z_start=z_start, z_end=z_start+stitched_volume.shape[0])

    if output_type == "ome-zarr":
        data_writer.complete_ome()
        
    logging.info(f"Inference complete. Output saved to {output_path}")
        
if __name__ == "__main__":
    sys.exit(main())
