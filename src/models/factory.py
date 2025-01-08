from typing import Dict, Any
import torch.nn as nn
import pytorch_lightning as pl
from omegaconf import OmegaConf, DictConfig
from .wav2vec import Wav2VecEmotionModel
from .image_models import PretrainedImageModel
from ..utils.model_manager import ModelManager
from ..metrics.base_metrics import BaseEmotionMetrics
from ..metrics.image_metrics import ImageEmotionMetrics
from ..metrics.audio_metrics import AudioEmotionMetrics

class ModelFactory:
    @staticmethod
    def create(model_name: str, config: DictConfig) -> pl.LightningModule:
        """모델 생성"""
        # 데이터셋의 실제 클래스 수 확인
        if hasattr(config.dataset, 'filtering') and config.dataset.filtering.enabled:
            # 필터링된 클래스 수 계산
            class_names = list(config.dataset.class_names)
            if config.dataset.filtering.exclude_emotions:
                class_names = [name for name in class_names 
                             if name not in config.dataset.filtering.exclude_emotions]
            if config.dataset.filtering.emotions:
                class_names = [name for name in class_names 
                             if name in config.dataset.filtering.emotions]
            num_classes = len(class_names)
        else:
            num_classes = config.dataset.num_classes
            class_names = list(config.dataset.class_names)
        
        # config 업데이트
        config.dataset.num_classes = num_classes
        config.dataset.class_names = class_names
        
        # 모델 생성
        if model_name == "wav2vec" or config.model.architecture == "wav2vec2":
            return Wav2VecEmotionModel(config)
        elif model_name == "resnet":
            return PretrainedImageModel(config)
        elif model_name == "efficientnet":
            return PretrainedImageModel(config)
        else:
            raise ValueError(f"Unknown model: {model_name}")

    @staticmethod
    def create_metrics(model_name: str, config: DictConfig) -> BaseEmotionMetrics:
        """메트릭스 생성"""
        num_classes = config.dataset.num_classes
        class_names = config.dataset.class_names
        
        if model_name in ["resnet", "efficientnet"]:
            return ImageEmotionMetrics(num_classes, class_names, config)
        elif model_name == "wav2vec":
            return AudioEmotionMetrics(num_classes, class_names, config)
        else:
            raise ValueError(f"Unknown model type: {model_name}")
