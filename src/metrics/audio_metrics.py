from typing import Dict, List
from omegaconf import DictConfig
from .base_metrics import BaseEmotionMetrics
import logging
import torch
from sklearn.metrics import accuracy_score, f1_score

class AudioEmotionMetrics(BaseEmotionMetrics):
    def __init__(self, num_classes: int, class_names: List[str], config: DictConfig):
        super().__init__(num_classes, class_names, config)

    def compute(self, prefix: str = "") -> Dict[str, float]:
        """오디오 특화 메트릭 계산"""
        metrics = super().compute(prefix)
        phase = prefix.rstrip('_/')
        
        # 기본 메트릭 계산
        if self.all_labels and self.all_preds:
            y_true = torch.tensor(self.all_labels)
            y_pred = torch.tensor(self.all_preds)
            
            metrics.update({
                f"{phase}/accuracy": accuracy_score(y_true.cpu(), y_pred.cpu()),
                f"{phase}/f1": f1_score(y_true.cpu(), y_pred.cpu(), average='macro'),
                f"{phase}/weighted_f1": f1_score(y_true.cpu(), y_pred.cpu(), average='weighted')
            })
            
            # 클래스별 F1 점수
            class_f1 = f1_score(y_true.cpu(), y_pred.cpu(), average=None)
            for i, score in enumerate(class_f1):
                metrics[f"{phase}/f1_{self.class_names[i]}"] = score
        
        return metrics 