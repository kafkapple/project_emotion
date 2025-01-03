from typing import Dict
import torch
from src.metrics.base_metrics import BaseEmotionMetrics

class MultimodalEmotionMetrics(BaseEmotionMetrics):
    def compute(self, prefix: str = "", log_wandb: bool = True) -> Dict[str, float]:
        return super().compute(prefix, log_wandb) 