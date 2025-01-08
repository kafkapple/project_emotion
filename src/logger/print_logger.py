from pathlib import Path
from omegaconf import DictConfig, OmegaConf
import logging
from typing import Dict, List
import pytorch_lightning as pl

class PrintLogger:
    def __init__(self, config: DictConfig):
        self.config = config

    def print_training_config(self):
        """학습 설정 출력"""
        logging.info("\nTraining Configuration:")
        logging.info(f"{'='*50}")
        logging.info(f"Max epochs: {self.config.train.max_epochs}")
        logging.info(f"Batch size: {self.config.train.batch_size}")
        logging.info(f"Learning rate: {self.config.train.learning_rate}")
        logging.info(f"Weight decay: {self.config.train.weight_decay}")
        logging.info(f"Gradient clip val: {self.config.train.gradient_clip_val}")
        logging.info(f"Accumulate grad batches: {self.config.train.accumulate_grad_batches}")
        logging.info(f"{'='*50}\n")

    def print_dataloader_config(self, loaders: Dict):
        """데이터로더 설정 출력"""
        logging.info("\nDataLoader Configuration:")
        logging.info(f"{'='*50}")
        logging.info(f"Train samples: {len(loaders['train'].dataset)}")
        logging.info(f"Val samples: {len(loaders['val'].dataset)}")
        logging.info(f"Test samples: {len(loaders['test'].dataset)}")
        logging.info(f"Batch size: {loaders['train'].batch_size}")
        logging.info(f"Num workers: {loaders['train'].num_workers}")
        logging.info(f"{'='*50}\n")

    def print_trainer_config(self, trainer: pl.Trainer, callbacks: List):
        """Trainer 설정 출력"""
        logging.info("\nTrainer Configuration:")
        logging.info(f"{'='*50}")
        logging.info(f"Max epochs: {trainer.max_epochs}")
        logging.info(f"Early stopping patience: {callbacks[1].patience}")
        logging.info(f"Early stopping min delta: {callbacks[1].min_delta}")
        logging.info(f"{'='*50}\n")

    def print_test_results(self, results: Dict):
        """테스트 결과 출력"""
        logging.info(f"\nTest Results:")
        logging.info(f"{'='*50}")
        for metric, value in results.items():
            logging.info(f"{metric}: {value:.4f}")
        logging.info(f"{'='*50}\n")

    def print_info(self):
        """기본 정보 출력"""
        logging.info(f"\nProject: {self.config.project.name}")
        logging.info(f"Model: {self.config.model.name}")
        logging.info(f"Dataset: {self.config.dataset.name}")