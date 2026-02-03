import torch
from tqdm import tqdm
import numpy as np

class Inferencer:
    def __init__(self, model: torch.nn.Module, device=None):
        """
        Args:
            model (torch.nn.Module): The trained model for inference.
            device (torch.device): The device to run inference on (e.g., torch.device("cuda")).
        """
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def eval(self, inference_loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Run inference on the entire dataset using the provided DataLoader.

        Args:
            inference_loader (torch.utils.data.DataLoader): DataLoader with input samples.

        Returns:
            np.ndarray: Model predictions concatenated as a NumPy array.
        """
        outputs = []
        progress_bar = tqdm(inference_loader, desc=f"Inferencing ", leave=False)

        with torch.no_grad():
            for inputs in progress_bar:
                # If dataloader returns (input, target), ignore target
                if isinstance(inputs, (list, tuple)):
                    inputs = inputs[0]
                    
                inputs = inputs.to(self.device)
                preds = self.model(inputs)
                
                # Squeeze only the channel dimension (dim 1), keeping (N, D, H, W) or (N, H, W)
                # But we want to ensure we return (N, D, H, W) even if D=1 for consistent stitching
                if preds.ndim == 5: # (N, C, D, H, W)
                    preds = preds.squeeze(1) # (N, D, H, W)
                elif preds.ndim == 4: # (N, C, H, W)
                    preds = preds # Keep (N, C, H, W) which is (N, 1, H, W) if C=1
                    # Actually if it's 2D, we want (N, H, W) but the stitcher expects (N, 1, H, W) if pd=1
                    # Let's just remove the channel dim and let stitcher handle it or keep it as D.
                    preds = preds.squeeze(1)[:, np.newaxis, ...] # Force (N, 1, H, W)
                
                preds = preds.detach().cpu().numpy()
                outputs.append(preds)

        return np.concatenate(outputs, axis=0)  # [N, D, H, W]