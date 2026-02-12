import numpy as np
from skimage.transform import resize
    
def extract_patches(
    array: np.ndarray,
    patch_size=(16, 64, 64),
    overlay=(2, 4, 4),
    resize_factor=(1, 1, 1),
    return_positions=False
):
    """
    Extracts overlapping 3D patches from an input volume with optional resizing.
    """
    D, H, W = array.shape
    
    # Pad if array is smaller than patch_size
    pad_d = max(0, patch_size[0] - D)
    pad_h = max(0, patch_size[1] - H)
    pad_w = max(0, patch_size[2] - W)
    
    if pad_d > 0 or pad_h > 0 or pad_w > 0:
        array = np.pad(array, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')
        D, H, W = array.shape

    patches, positions = [], []

    step_z = max(1, patch_size[0] - overlay[0])
    step_y = max(1, patch_size[1] - overlay[1])
    step_x = max(1, patch_size[2] - overlay[2])

    for z in range(0, D, step_z):
        start_z = min(z, D - patch_size[0])
        if start_z < 0: start_z = 0
        for y in range(0, H, step_y):
            start_y = min(y, H - patch_size[1])
            if start_y < 0: start_y = 0
            for x in range(0, W, step_x):
                start_x = min(x, W - patch_size[2])
                if start_x < 0: start_x = 0
                
                patch = array[
                    start_z:start_z + patch_size[0],
                    start_y:start_y + patch_size[1],
                    start_x:start_x + patch_size[2]
                ]
                
                # Resize if needed
                if resize_factor != (1, 1, 1):
                    new_shape = (
                        int(patch.shape[0] * resize_factor[0]),
                        int(patch.shape[1] * resize_factor[1]),
                        int(patch.shape[2] * resize_factor[2])
                    )
                    patch = resize(
                        patch, new_shape,
                        order=0,
                        mode='reflect',
                        anti_aliasing=True,
                        preserve_range=True
                    ).astype(patch.dtype)
                
                patches.append(patch)
                positions.append((start_z, start_y, start_x))

    if return_positions:
        return np.array(patches), np.array(positions)
    else:
        return np.array(patches)

def balance_patches(image_patches, mask_patches, neg_keep_ratio=1.0):
    """
    Balances a dataset of image and mask patches by keeping a specific ratio of negative patches.

    Args:
        image_patches (list): List of image patch arrays.
        mask_patches (list): List of corresponding binary mask patch arrays.
        neg_keep_ratio (float): Ratio of negative patches to keep relative to positive patches (0.0 to 1.0).
            1.0 means equal numbers of positive and negative (50/50 balance).
            0.0 means only keep positive patches.

    Returns:
        tuple: (np.ndarray, np.ndarray) Balanced image and mask patches.
    """
    positive_patches = []
    negative_patches = []
    
    for img_patch, mask_patch in zip(image_patches, mask_patches):
        if np.sum(mask_patch) > 0:
            positive_patches.append((img_patch, mask_patch))
        else:
            negative_patches.append((img_patch, mask_patch))

    n_pos = len(positive_patches)
    if n_pos == 0:
        return np.array(image_patches), np.array(mask_patches)

    # Calculate how many negative patches to keep
    n_keep_neg = int(n_pos * neg_keep_ratio)
    n_keep_neg = min(n_keep_neg, len(negative_patches))

    np.random.shuffle(negative_patches)
    selected_negatives = negative_patches[:n_keep_neg]

    balanced_patches = positive_patches + selected_negatives
    np.random.shuffle(balanced_patches)

    image_out, mask_out = zip(*balanced_patches)
    return np.array(image_out), np.array(mask_out)

def extract_training_batches(
    image:np.ndarray,
    mask: np.ndarray,
    patch_size=(16, 64, 64),
    overlay=(2, 4, 4),
    resize_factor=(1, 1, 1),
    neg_keep_ratio=1.0
):
    """
    Extracts training patches with a controllable ratio of negative (empty) patches.

    Args:
        neg_keep_ratio (float): How many empty patches to keep relative to positive ones.
    """
    
    if (image.shape != mask.shape):
        raise ValueError("Image and mask must have the same shape.")
    
    image_patches = extract_patches(array=image, patch_size=patch_size, overlay=overlay, resize_factor=resize_factor)
    mask_patches = extract_patches(array=mask, patch_size=patch_size, overlay=overlay, resize_factor=resize_factor)
    
    return balance_patches(image_patches, mask_patches, neg_keep_ratio=neg_keep_ratio)