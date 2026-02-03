import numpy as np
from skimage.transform import resize

def stitch_image_xy(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1)):
    """
    Reconstructs a full 3D volume from overlapping patches using weighted averaging.

    Args:
        patches (list or np.ndarray): List or array of 3D patches (Z, Y, X).
        positions (list or np.ndarray): Corresponding positions (z, y, x) for placing each patch.
        original_shape (tuple): Shape of the full output volume (depth, height, width).
        patch_size (tuple): Size of each patch (depth, height, width).
        resize_factor (tuple): Factor used to resize patches during extraction. If not (1, 1, 1), patches will be resized back.

    Returns:
        np.ndarray: Full reconstructed volume of shape `original_shape`, merged using averaging in overlapping regions.
    """
    reconstruction = np.zeros(original_shape, dtype=np.float32)
    weight = np.zeros(original_shape, dtype=np.float32)
    pd, ph, pw = patch_size

    for patch, (d, h, w) in zip(patches, positions):
        # Resize patch back to original patch size if it was resized
        if resize_factor != (1, 1, 1):
            patch = resize(
                patch,
                (pd, ph, pw),
                order=1,
                mode='reflect',
                anti_aliasing=True,
                preserve_range=True
            ).astype(np.float32)

        # Add patch to reconstruction
        reconstruction[d:d+pd, h:h+ph, w:w+pw] += patch
        weight[d:d+pd, h:h+ph, w:w+pw] += 1

    # Avoid division by zero
    weight[weight == 0] = 1
    reconstruction /= weight

    return reconstruction

def stitch_image_z(reconstruction: np.ndarray, prev_z_slices: np.ndarray, threshold=0.5):
    """
    Performs blending across overlapping Z slices between consecutive volume chunks and returns a binary mask.
    All blending is done in logit space.
    """
    if prev_z_slices is not None:
        z_overlay = prev_z_slices.shape[0] if prev_z_slices.ndim >= 3 else 1
        
        if prev_z_slices.shape != reconstruction[:z_overlay].shape:
            # If shape mismatch happens due to single slice etc, attempt to handle or raise
            if prev_z_slices.size == reconstruction[:z_overlay].size:
                prev_z_slices = prev_z_slices.reshape(reconstruction[:z_overlay].shape)
            else:
                raise ValueError(f"Shape mismatch: {prev_z_slices.shape} vs {reconstruction[:z_overlay].shape}")
        
        reconstruction[:z_overlay] = (reconstruction[:z_overlay] + prev_z_slices) / 2

    # Apply sigmoid to logits to get probabilities before thresholding
    probs = 1 / (1 + np.exp(-reconstruction))

    return ((probs > threshold) * 255).astype(np.uint8)


def stitch_image(patches, positions, original_shape, patch_size, resize_factor=(1, 1, 1), prev_z_slices=None, z_overlay=0):
    """
    Reconstructs the full 3D volume from patches and blends overlapping Z slices across chunks.
    """
    reconstruct_xy = stitch_image_xy(patches, positions, original_shape, patch_size, resize_factor)
    
    # Save the raw logits for the next chunk's overlap BEFORE thresholding
    next_prev_z = None
    if z_overlay > 0:
        next_prev_z = reconstruct_xy[-z_overlay:].copy()

    # stitch_image_z now returns the thresholded mask
    binary_mask = stitch_image_z(reconstruct_xy, prev_z_slices)
    
    if z_overlay > 0:
        return binary_mask[:-z_overlay], next_prev_z
    else:
        return binary_mask, None
