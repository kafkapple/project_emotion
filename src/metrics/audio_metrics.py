from typing import Dict, List
from omegaconf import DictConfig
from .base_metrics import BaseEmotionMetrics
import logging

class AudioEmotionMetrics(BaseEmotionMetrics):
    def __init__(self, num_classes: int, class_names: List[str], config: DictConfig):
        super().__init__(num_classes, class_names, config)

    def compute(self, prefix: str = "") -> Dict[str, float]:
        """오디오 특화 메트릭 계산"""
        metrics = super().compute(prefix)
        
        # 여기에 오디오 특화 메트릭 추가 가능
        
        # 메트릭 로깅
        self.log_metrics(prefix.replace('_', '') if prefix else 'eval')
        
        return metrics 