from omegaconf import DictConfig
from pathlib import Path
import os
from .wandb_logger import WandbLogger
from .print_logger import PrintLogger
import wandb
import logging

class Logger:
    def __init__(self, config: DictConfig):
        self.config = config
        self.print_logger = PrintLogger(config)
        self._setup_wandb_config()
    
    def _setup_wandb_config(self):
        """WandB 설정 초기화"""
        self.wandb_config = {
            # 기본 학습 설정
            "batch_size": self.config.train.batch_size,
            "max_epochs": self.config.train.max_epochs,
            "learning_rate": self.config.train.learning_rate,
            "num_workers": self.config.train.num_workers,
            
            # 옵티마이저 설정
            "optimizer": self.config.train.optimizer.name,
            "weight_decay": self.config.train.optimizer.weight_decay,
            
            # 스케줄러 설정
            "scheduler": self.config.train.scheduler.name,
            "warmup_epochs": self.config.train.scheduler.warmup_epochs,
            "min_lr": self.config.train.scheduler.min_lr,
            
            # 메모리 관리 설정
            "gradient_clip_val": self.config.train.memory_management.gradient_clip_val,
            "accumulate_grad_batches": self.config.train.memory_management.accumulate_grad_batches,
            "precision": self.config.train.memory_management.precision,
            
            # Early stopping 설정
            "early_stopping_patience": self.config.train.early_stopping.patience,
            "early_stopping_min_delta": self.config.train.early_stopping.min_delta,
            
            # 로깅 설정
            "logging_interval": self.config.train.logging.step_interval,
            
            # 모델 설정
            "model_name": self.config.model.name,
            "model_config": dict(self.config.model)
        }
        
        # wandb config 업데이트
        if wandb.run is not None:
            wandb.config.update(self.wandb_config)

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
