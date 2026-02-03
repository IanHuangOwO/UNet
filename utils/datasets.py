from __future__ import annotations

import numpy as np
import torch
from typing import Callable, Dict, List, Optional, Union
from monai.data.dataset import Dataset

class MicroscopyDataset(Dataset):
    """
    Unified dataset for 2D/3D microscopy segmentation for both training and inference.

    - Uses dict-based samples compatible with MONAI transforms: {"image", ["mask"]}.
    - Adds a channel dimension if missing (C=1) based on `spatial_dims`.
    - Returns (image, mask) when `with_mask=True`, else returns image only.

    Args:
        patch_dicts: List of sample dicts containing at least key "image" and
            optionally key "mask" when `with_mask=True`.
        transform: Optional transform (e.g., MONAI Compose) applied to the sample dict.
        spatial_dims: 2 or 3 indicating 2D or 3D images.
        with_mask: Whether to return the mask alongside the image.
    """

    def __init__(
        self,
        patch_dicts: List[Dict[str, Union["object"]]],
        transform: Optional[Callable] = None,
        spatial_dims: int = 3,
        with_mask: bool = True,
    ) -> None:
        if spatial_dims not in (2, 3):
            raise ValueError(f"spatial_dims must be 2 or 3, got {spatial_dims}")

        self.patch_dicts = patch_dicts
        self.transform = transform
        self.spatial_dims = spatial_dims
        self.with_mask = with_mask

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.patch_dicts)

    def __getitem__(self, idx: int):  # type: ignore[override]
        # Use a copy to avoid modifying the shared patch_dicts during multi-epoch training
        sample = dict(self.patch_dicts[idx])

        # Ensure channel dimension is present before transforms
        # Most MONAI transforms expect (C, H, W) or (C, D, H, W)
        for key in ["image", "mask"]:
            if key in sample:
                val = sample[key]
                if hasattr(val, "ndim"):
                    if self.spatial_dims == 2 and val.ndim == 2:
                        sample[key] = val[np.newaxis, ...]
                    elif self.spatial_dims == 3 and val.ndim == 3:
                        sample[key] = val[np.newaxis, ...]

        # Apply transform
        if self.transform is not None:
            sample = self.transform(sample)

        image = sample["image"]
        # Ensure it's a tensor after transform if transform didn't do it (though ToTensord usually does)
        if not hasattr(image, "unsqueeze") and hasattr(image, "ndim"):
             # It's likely still a numpy array
             import torch
             image = torch.from_numpy(image)

        if not self.with_mask:
            return image

        mask = sample.get("mask", None)
        if mask is None:
            raise KeyError("Sample is missing required key 'mask' but with_mask=True")

        if not hasattr(mask, "unsqueeze") and hasattr(mask, "ndim"):
             import torch
             mask = torch.from_numpy(mask)

        return image, mask

