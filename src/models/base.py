import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
import wandb
from typing import Dict, Any

class BaseModel(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.config = config
        self.save_hyperparameters()
        
        # 메트릭 누적을 위한 리스트
        self.training_step_losses = []
        self.validation_step_losses = []
        self.learning_rates = []
        self.current_epoch_steps = 0
        
    def on_train_epoch_start(self):
        self.training_step_losses = []
        self.learning_rates = []
        self.current_epoch_steps = 0
        
    def on_validation_epoch_start(self):
        self.validation_step_losses = []
        
    def training_step(self, batch, batch_idx):
        # 기존 training_step 로직
        loss = ...  # 기존 loss 계산
        
        # loss와 learning rate 누적
        self.training_step_losses.append(loss.item())
        self.learning_rates.append(self.optimizers().param_groups[0]['lr'])
        self.current_epoch_steps += 1
        
        # 매 N 스텝마다 현재까지의 평균 기록
        step_interval = self.config.train.logging.step_interval
        if self.current_epoch_steps % step_interval == 0:
            avg_loss = np.mean(self.training_step_losses[-step_interval:])
            avg_lr = np.mean(self.learning_rates[-step_interval:])
            
            # 설정된 메트릭만 로깅
            if 'loss' in self.config.train.logging.metrics:
                self.log('train/step_loss', avg_loss, prog_bar=True)
            if 'learning_rate' in self.config.train.logging.metrics:
                self.log('train/learning_rate', avg_lr)
            
        self.log("train/loss", loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        # 기존 validation_step 로직
        loss = ...  # 기존 loss 계산
        
        # loss 누적
        self.validation_step_losses.append(loss.item())
        self.log("validation/loss", loss)
        return loss
        
    def test_step(self, batch, batch_idx):
        loss = ...
        self.log("test/loss", loss)
        return loss
        
    def on_train_epoch_end(self):
        if not self.config.train.logging.save_graph:
            return
        
        # 에포크 단위 메트릭 계산 및 시각화
        epoch_loss = np.mean(self.training_step_losses)
        
        # Loss curve 그리기
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.training_step_losses, label='Train Loss')
        plt.title(f'Training Loss Curve - Epoch {self.current_epoch}')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # Learning rate curve 그리기
        fig_lr = plt.figure(figsize=(10, 5))
        plt.plot(self.learning_rates, label='Learning Rate')
        plt.title(f'Learning Rate Curve - Epoch {self.current_epoch}')
        plt.xlabel('Step')
        plt.ylabel('Learning Rate')
        plt.legend()
        plt.grid(True)
        
        # WandB에 로깅
        self.logger.experiment.log({
            "train/epoch_loss": epoch_loss,
            "train/loss_curve": wandb.Image(fig),
            "train/lr_curve": wandb.Image(fig_lr)
        })
        
        plt.close(fig)
        plt.close(fig_lr)
        
    def on_validation_epoch_end(self):
        # 검증 데이터 메트릭 계산 및 시각화
        val_loss = np.mean(self.validation_step_losses)
        
        # Loss curve 그리기
        fig = plt.figure(figsize=(10, 5))
        plt.plot(self.validation_step_losses, label='Validation Loss')
        plt.title(f'Validation Loss Curve - Epoch {self.current_epoch}')
        plt.xlabel('Step')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        
        # WandB에 로깅
        self.logger.experiment.log({
            "val/epoch_loss": val_loss,
            "val/loss_curve": wandb.Image(fig)
        })
        
        plt.close(fig) 