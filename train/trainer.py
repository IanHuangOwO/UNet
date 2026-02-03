import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import logging
from typing import Dict, List, Callable, Union, Optional
from tqdm import tqdm

from utils.visualization import visualize_predictions
from utils.metrics import hard_dice_score, bce_score

# Initialize logging
logger = logging.getLogger(__name__)

MetricFn = Callable[[torch.Tensor, torch.Tensor], Union[torch.Tensor, float]]

METRICS_TO_COMPUTE: Dict[str, MetricFn] = {
    "dice_score": lambda outputs, targets: hard_dice_score(outputs, targets, from_logits=True),
    "bce_score": lambda outputs, targets: bce_score(outputs, targets, from_logits=True),
}

class Trainer:
    """Simple, readable training loop with model-defined loss.

    - Uses model.get_loss() for optimization.
    - Metrics in METRICS_TO_COMPUTE are recorded for monitoring.
    """

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        device=None,
        lr=0.001,
        weight_decay=1e-5,
        epochs=30,
        model_name="best_model",
        save_path="./",
        cache_size=20,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.model_name = model_name
        
        # All outputs go into save_path/model_name/
        self.save_path = os.path.join(save_path, model_name)
        os.makedirs(self.save_path, exist_ok=True)
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=epochs)

        # Store history of metrics (including reserved 'loss' key)
        self.metrics_history: Dict[str, Dict[str, List[float]]] = {
            name: {"train": [], "val": []} for name in list(METRICS_TO_COMPUTE.keys()) + ["loss"]
        }

        self.best_val_loss = float("inf")
        self.cache_size = cache_size
        self.best_results_cache: Optional[Dict[str, np.ndarray]] = None
        self._epoch_viz_cache: Optional[Dict[str, List[np.ndarray]]] = None

    def train(self, epochs=30):
        for epoch in range(epochs):
            print("\n")
            logger.info(f"Epoch {epoch + 1}/{epochs}")
            
            train_loss = self._run_epoch(train=True, desc=f'Training Epoch {epoch+1}')
            val_loss = self._run_epoch(train=False, desc=f'Validating Epoch {epoch+1}')
            
            # Log primary loss
            logger.info(f"Loss -> Train: {train_loss:.4f} | Val: {val_loss:.4f}")
            
            # Log additional metrics
            for m_name in METRICS_TO_COMPUTE.keys():
                t_val = self.metrics_history[m_name]["train"][-1]
                v_val = self.metrics_history[m_name]["val"][-1]
                logger.info(f"{m_name.replace('_', ' ').capitalize()} -> Train: {t_val:.4f} | Val: {v_val:.4f}")
            
            self.scheduler.step()
            
            # Check for best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                if self._epoch_viz_cache:
                    self.best_results_cache = {k: np.array(v) for k, v in self._epoch_viz_cache.items()}
                self._save_checkpoint(self.model_name)
            
            # Periodic visualization and checkpoint (Every 25 epochs)
            if (epoch + 1) % 25 == 0:
                if self._epoch_viz_cache and len(self._epoch_viz_cache["images"]) > 0:
                    visualize_predictions(
                        np.array(self._epoch_viz_cache["images"]),
                        np.array(self._epoch_viz_cache["masks"]),
                        np.array(self._epoch_viz_cache["outputs"]),
                        save_path=self.save_path,
                        title=f"Epoch_{epoch+1}_Validation"
                    )
                self._save_checkpoint(f"{self.model_name}_epoch_{epoch+1}")

            self._save_curves()
            
        if self.best_results_cache:
            visualize_predictions(
                self.best_results_cache["images"],
                self.best_results_cache["masks"],
                self.best_results_cache["outputs"],
                save_path=self.save_path,
                title=f"Best_Model_Results"
            )

    def _run_epoch(self, *, train: bool, desc: str) -> float:
        """Run a single epoch (either train or val)."""
        self.model.train(mode=train)
        loader = self.train_loader if train else self.val_loader
        
        metric_keys = list(METRICS_TO_COMPUTE.keys())
        sums: Dict[str, float] = {m: 0.0 for m in metric_keys}
        total_primary_loss = 0.0
        n_batches = max(1, len(loader))

        if not train:
            self._epoch_viz_cache = {"images": [], "masks": [], "outputs": []}

        context = torch.enable_grad if train else torch.no_grad
        with context():
            progress = tqdm(loader, desc=desc, leave=False)
            for images, masks in progress:
                images = images.to(self.device, non_blocking=True)
                masks = masks.to(self.device, non_blocking=True)

                if train:
                    self.optimizer.zero_grad(set_to_none=True)

                outputs = self.model(images)
                loss_val = self.model.get_loss(outputs, masks)
                
                if train:
                    loss_val.backward()
                    self.optimizer.step()

                # Record primary loss and metrics
                batch_loss = float(loss_val.item())
                total_primary_loss += batch_loss
                
                batch_metrics: Dict[str, float] = {"loss": batch_loss}
                for name, fn in METRICS_TO_COMPUTE.items():
                    try:
                        v = fn(outputs, masks)
                        val = float(v.item()) if isinstance(v, torch.Tensor) else float(v)
                        batch_metrics[name] = val
                        sums[name] += val
                    except Exception:
                        pass
                
                if not train:
                    self._collect_samples(images, masks, outputs)

                progress.set_postfix({k: f"{v:.4f}" for k, v in batch_metrics.items()})

        split = "train" if train else "val"
        avg_loss = total_primary_loss / n_batches
        self.metrics_history["loss"][split].append(avg_loss)
        
        for m in metric_keys:
            self.metrics_history[m][split].append(sums[m] / n_batches)

        return avg_loss

    def _save_checkpoint(self, name: str):
        """Helper to save the current model state."""
        path = os.path.join(self.save_path, f"{name}.pth")
        torch.save(self.model, path)
        logger.info(f"[OK] Model saved to {path}")

    def _collect_samples(self, images: torch.Tensor, masks: torch.Tensor, outputs: torch.Tensor):
        """Helper to collect samples into the current epoch cache."""
        if self._epoch_viz_cache is None: return
        current_count = len(self._epoch_viz_cache["images"])
        if current_count >= self.cache_size: return
        
        n_to_add = min(self.cache_size - current_count, images.size(0))
        self._epoch_viz_cache["images"].extend(images[:n_to_add].detach().cpu().numpy())
        self._epoch_viz_cache["masks"].extend(masks[:n_to_add].detach().cpu().numpy())
        self._epoch_viz_cache["outputs"].extend(outputs[:n_to_add].detach().cpu().numpy())

    def _save_curves(self):
        import matplotlib
        matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt

        metrics_to_plot = [k for k, v in self.metrics_history.items() if v["train"] and v["val"]]
        if not metrics_to_plot: return

        n = len(metrics_to_plot)
        rows = int(np.ceil(n / 2)) if n > 2 else 1
        cols = min(n, 2)
        
        fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

        for idx, m in enumerate(metrics_to_plot):
            r, c = idx // cols, idx % cols
            ax = axes[r, c]
            ax.plot(self.metrics_history[m]["train"], label=f"Train")
            ax.plot(self.metrics_history[m]["val"], label=f"Val")
            ax.set_title(m.replace('_', ' ').capitalize())
            ax.legend()
            ax.grid(True)

        # Hide unused subplots
        for j in range(n, rows * cols):
            fig.delaxes(axes[j // cols, j % cols])

        plt.tight_layout()
        plt.savefig(os.path.join(self.save_path, f"{self.model_name}-metrics_curve.png"))
        plt.close(fig)