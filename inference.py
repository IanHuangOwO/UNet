"""
Batch volume inference with a three-stage asynchronous pipeline.

Stages:
1. Loader Thread: Pre-reads Z-windows and prepares shared-memory datasets.
2. Main Process: Executes model inference on the GPU.
3. Stitcher Thread: Handles CPU-intensive stitching and Disk-intensive writing.

This pipeline maximizes GPU throughput and prevents RAM duplication.
"""
import logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

import argparse
import os
import sys
import json
import threading
import queue
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
from tqdm import tqdm

import torch
from monai.data.dataloader import DataLoader
from monai.transforms.compose import Compose
from monai.transforms.utility.dictionary import ToTensord

from IO import FileReader, FileWriter, TYPE_MAP
from IO.datasets import InferenceMicroscopyDataset
from utils.cropper import compute_z_plan
from utils.stitcher import stitch_image

# Standard transform
inference_transform = Compose([
    ToTensord(keys=["image"], dtype=torch.float32),
])

def load_checkpoint(model_path: str):
    """Load a torch model checkpoint."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model = torch.load(model_path, weights_only=False)
    return model

def run_inference(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """
    Execute model inference on a dataloader.
    
    Returns:
        np.ndarray: Concatenated predictions in (N, D, H, W) format.
    """
    model.eval()
    outputs = []
    
    with torch.no_grad():
        for inputs in tqdm(loader, desc="    Inference Batch", leave=False):
            if isinstance(inputs, (list, tuple)):
                inputs = inputs[0]
                
            inputs = inputs.to(device)
            preds = model(inputs)
            
            # Normalize to (N, D, H, W)
            if preds.ndim == 5: # 3D: (N, C, D, H, W)
                preds = preds.squeeze(1) 
            elif preds.ndim == 4: # 2D: (N, C, H, W)
                preds = preds.squeeze(1)[:, np.newaxis, ...] # Force (N, 1, H, W)
            
            outputs.append(preds.detach().cpu().numpy())

    return np.concatenate(outputs, axis=0)

def loader_worker(
    data_reader: FileReader,
    z_plan: List[Tuple[int, int]],
    patch_size: Tuple[int, int, int],
    overlay: Tuple[int, int, int],
    inf_queue: queue.Queue
):
    """Stage 1: Pre-load data from disk and prepare shared-memory datasets."""
    try:
        for z_start, z_overlay in z_plan:
            z_end = min(z_start + patch_size[0], data_reader.volume_shape[0])
            
            dataset = InferenceMicroscopyDataset(
                image_reader=data_reader,
                z_range=(z_start, z_end),
                patch_size=patch_size,
                overlap=overlay,
                transform=inference_transform
            )
            inf_queue.put((dataset, z_start, z_end, z_overlay))
            
        inf_queue.put(None)
    except Exception as e:
        logging.error(f"Loader Thread failed: {e}")
        inf_queue.put(None)

def stitcher_worker(
    data_writer: FileWriter,
    volume_shape: Tuple[int, int, int],
    patch_size: Tuple[int, int, int],
    resize_factor: List[float],
    stitch_queue: queue.Queue,
    output_type: str
):
    """Stage 3: Stitch mask patches and write to disk."""
    prev_z_slices = None
    try:
        while True:
            item = stitch_queue.get()
            if item is None:
                break
                
            mask_patches, data_position, z_start, z_end, z_overlay = item
            actual_chunk_depth = z_end - z_start
            
            logging.info(f"  Stitching & Writing Z:{z_start}-{z_end}...")
            
            stitched_volume, prev_z_slices = stitch_image(
                patches=mask_patches, 
                positions=data_position,
                original_shape=(actual_chunk_depth, volume_shape[1], volume_shape[2]),
                patch_size=patch_size,
                z_overlay=z_overlay,
                prev_z_slices=prev_z_slices,
                resize_factor=resize_factor,
            )
            
            data_writer.write(stitched_volume, z_start=z_start, z_end=z_start+stitched_volume.shape[0])
            stitch_queue.task_done()
            
        if output_type == "ome-zarr":
            data_writer.complete_ome()
            
    except Exception as e:
        logging.error(f"Stitcher Thread failed: {e}")
    finally:
        stitch_queue.task_done()

def process_volume(volume_path: str, output_root: str, model: torch.nn.Module, device: torch.device, config: dict):
    """Orchestrates the asynchronous pipeline for a single volume."""
    v_path = Path(volume_path)
    data_reader = FileReader(v_path)
    volume_name = data_reader.volume_name
    
    output_path = os.path.join(output_root, volume_name)
    os.makedirs(output_path, exist_ok=True)
    
    output_type_str = config.get("output_type", "Scroll-Tif")
    output_type = TYPE_MAP.get(output_type_str, output_type_str)
    
    data_writer = FileWriter(
        output_path=output_path,
        output_name=volume_name, 
        output_type=output_type,
        output_dtype=config.get("output_dtype", "uint16"),
        full_res_shape=data_reader.volume_shape,
        file_name=data_reader.volume_files,
        chunk_size=tuple(config.get("output_chunk_size", [128, 128, 128])),
        resize_factor=config.get("output_resize_factor", 2),
    )
    
    patch_size = tuple(config.get("inference_patch_size", [16, 64, 64]))
    overlay = tuple(config.get("inference_overlay", [2, 4, 4]))
    batch_size = config.get("batch_size", 8)
    num_workers = config.get("num_workers", 4)
    resize_factor = config.get("inference_resize_factor", [1.0, 1.0, 1.0])
    
    z_plan = compute_z_plan(data_reader.volume_shape[0], patch_size[0], overlay[0])
    
    inf_queue = queue.Queue(maxsize=1)
    stitch_queue = queue.Queue(maxsize=1)
    
    loader_thread = threading.Thread(
        target=loader_worker, 
        args=(data_reader, z_plan, patch_size, overlay, inf_queue),
        daemon=True
    )
    loader_thread.start()
    
    stitcher_thread = threading.Thread(
        target=stitcher_worker,
        args=(data_writer, data_reader.volume_shape, patch_size, resize_factor, stitch_queue, output_type),
        daemon=True
    )
    stitcher_thread.start()
    
    logging.info(f"Pipeline started for volume: {volume_name} ({data_reader.volume_shape})")
    
    while True:
        inf_data = inf_queue.get()
        if inf_data is None:
            break
            
        dataset, z_start, z_end, z_overlay = inf_data
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        logging.info(f"  Inference Z:{z_start}-{z_end} | Patches: {len(dataset)}")
        mask_patches = run_inference(model, loader, device)
        
        data_position = [meta.slices.global_coords for meta in dataset.patch_indices]
        stitch_queue.put((mask_patches, data_position, z_start, z_end, z_overlay))
        
    stitch_queue.put(None)
    stitcher_thread.join()
    loader_thread.join()
    
    logging.info(f"Completed volume: {volume_name}")

def main():
    parser = argparse.ArgumentParser(description="Batch Inference: Three-stage asynchronous pipeline")
    parser.add_argument("--config", type=str, default="configs/config.json", help="Path to config file")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f).get("inference", {})
    
    input_root = config.get("input_path")
    output_root = config.get("output_path")
    model_path = config.get("model_path")
    
    if not input_root or not output_root or not model_path:
        logging.error("Missing mandatory paths in config.")
        return 1
        
    os.makedirs(output_root, exist_ok=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Loading model: {model_path} to {device}")
    model = load_checkpoint(model_path).to(device)
    
    input_path = Path(input_root)
    
    if input_path.is_dir():
        subfolders = [d for d in input_path.iterdir() if d.is_dir()]
        if not subfolders or any(s.suffix == '.zarr' for s in subfolders):
            process_volume(input_root, output_root, model, device, config)
        else:
            for v_dir in sorted(subfolders):
                process_volume(str(v_dir), output_root, model, device, config)
    else:
        process_volume(input_root, output_root, model, device, config)

    logging.info("Batch inference complete.")

if __name__ == "__main__":
    main()
