from typing import Dict
import torch
from src.metrics.base_metrics import BaseEmotionMetrics

class TextEmotionMetrics(BaseEmotionMetrics):
    def compute(self, prefix: str = "", log_wandb: bool = True) -> Dict[str, float]:
        # TextEmotionMetrics만의 특별한 처리가 필요한 경우 여기에 구현
        return super().compute(prefix, log_wandb) 