from typing import Dict
from .base_metrics import BaseEmotionMetrics
import logging

class ImageEmotionMetrics(BaseEmotionMetrics):
    def compute(self, prefix: str = "") -> Dict[str, float]:
        """이미지 특화 메트릭 계산"""
        metrics = super().compute(prefix)
        
        # 여기에 이미지 특화 메트릭 추가 가능
        
        # 메트릭 로깅
        self.log_metrics(prefix.replace('_', '') if prefix else 'eval')
        
        return metrics