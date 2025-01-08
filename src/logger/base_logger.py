from omegaconf import DictConfig
from pathlib import Path
import os
from .wandb_logger import WandbLogger
from .print_logger import PrintLogger
import wandb

class Logger:
    def __init__(self, config: DictConfig):
        self.config = config
        self.print_logger = PrintLogger(config)
        self._setup_wandb_config()
    
    def _setup_wandb_config(self):
        """WandB에 로깅할 주요 설정만 선택"""
        wandb_config = {
            # 모델 정보
            "model": {
                "name": self.config.model.name,
                "architecture": self.config.model.get("architecture", None),
                "pretrained": self.config.model.get("pretrained", None),
            },
            
            # 데이터셋 정보
            "dataset": {
                "name": self.config.dataset.name,
                "num_classes": self.config.dataset.num_classes,
                "class_names": self.config.dataset.class_names,
                "filtering": self.config.dataset.filtering if hasattr(self.config.dataset, "filtering") else None,
            },
            
            # 학습 설정
            "train": {
                "max_epochs": self.config.train.max_epochs,
                "batch_size": self.config.train.batch_size,
                "learning_rate": self.config.train.learning_rate,
                "weight_decay": self.config.train.weight_decay,
            }
        }
        
        # wandb config 업데이트
        if wandb.run is not None:
            wandb.config.update(wandb_config)
