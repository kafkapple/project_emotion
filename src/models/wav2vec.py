from typing import Dict, Any
import torch
import torch.nn as nn
from transformers import Wav2Vec2Model
import logging
import torch.nn.functional as F
import pytorch_lightning as pl
from src.metrics.audio_metrics import AudioEmotionMetrics
from sklearn.metrics import f1_score

class Wav2VecEmotionModel(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # Tensor Cores 최적화 설정
        if hasattr(config.model, 'matmul_precision'):
            torch.set_float32_matmul_precision(config.model.matmul_precision)
        
        # Wav2Vec2 모델 초기화 (단순화)
        self.wav2vec = Wav2Vec2Model.from_pretrained(config.model.pretrained)
        
        # 분류기 레이어 분리
        self.feature_extractor = nn.Sequential(
            nn.Linear(self.wav2vec.config.hidden_size, config.model.classifier.hidden_size),
            nn.LayerNorm(config.model.classifier.hidden_size),
            nn.GELU(),
            nn.Dropout(config.model.classifier.dropout)
        )
        
        self.classifier_head = nn.Linear(config.model.classifier.hidden_size, config.dataset.num_classes)
        
        # Freezing 설정 적용
        if config.model.freeze.enabled:
            self._freeze_layers()
        
        # 메트릭스 초기화
        self.train_metrics = AudioEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        self.val_metrics = AudioEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        self.test_metrics = AudioEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        
        # Debug 모드 출력 주석 처리
        # if config.debug.enabled:
        #     logging.info("\nModel Architecture:")
        #     logging.info(f"{'='*50}")
        #     logging.info(f"{self}")
        #     logging.info(f"{'='*50}\n")
        #     
        #     # 파라미터 수 출력
        #     total_params = sum(p.numel() for p in self.parameters())
        #     trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        #     logging.info(f"Total parameters: {total_params:,}")
        #     logging.info(f"Trainable parameters: {trainable_params:,}\n")
    
    def _freeze_layers(self):
        """레이어 고정 설정"""
        # 기본적으로 모든 레이어 고정
        for param in self.wav2vec.parameters():
            param.requires_grad = False
            
        if self.config.debug.enabled:
            logging.info("Initially freezing all layers")
        
        # 상위 N개 레이어만 학습 가능하도록 설정
        num_unfrozen = self.config.model.freeze.num_unfrozen_layers
        total_layers = len(self.wav2vec.encoder.layers)
        
        if num_unfrozen > 0:
            # 상위 N개 레이어 학습 가능하도록 설정
            for i in range(total_layers - num_unfrozen, total_layers):
                for param in self.wav2vec.encoder.layers[i].parameters():
                    param.requires_grad = True
            
            if self.config.debug.enabled:
                logging.info(f"Unfreezing top {num_unfrozen} transformer layers")
        else:
            if self.config.debug.enabled:
                logging.info("Using frozen embeddings only")
        
        if self.config.debug.enabled:
            # 학습 가능한 파라미터 수 출력
            total_params = sum(p.numel() for p in self.parameters())
            trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
            logging.info(f"Total parameters: {total_params:,}")
            logging.info(f"Trainable parameters: {trainable_params:,}")
            logging.info(f"Frozen parameters: {total_params - trainable_params:,}")
    
    def forward(self, batch: Dict[str, torch.Tensor], return_features: bool = False) -> torch.Tensor:
        """Forward pass"""
        audio = batch["audio"]
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"Original input shape: {audio.shape}")
        
        # [batch_size, 1, 1, seq_len] -> [batch_size, seq_len]
        if len(audio.shape) == 4:
            audio = audio.squeeze(2)
        if len(audio.shape) == 3:
            audio = audio.squeeze(1)
        
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"Processed input shape: {audio.shape}")
        
        # wav2vec2 feature extraction (frozen)
        audio_features = self.wav2vec(audio).last_hidden_state
        embeddings = torch.mean(audio_features, dim=1)  # [batch_size, hidden_size]
        
        # 분류기 feature extraction
        features = self.feature_extractor(embeddings)  # [batch_size, classifier_hidden_size]
        
        if return_features:
            return features  # Late fusion을 위한 feature 반환
            
        # 최종 분류
        logits = self.classifier_head(features)
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"Final output shape: {logits.shape}")
        
        return logits
        
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = F.cross_entropy(outputs, batch["label"])
        self.train_metrics.update(outputs, batch["label"])
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = F.cross_entropy(outputs, batch["label"])
        self.val_metrics.update(outputs, batch["label"])
        
        # val_loss 로깅 추가
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = F.cross_entropy(outputs, batch["label"])
        self.test_metrics.update(outputs, batch["label"])
        
        # test_loss만 로깅
        self.log("test_loss", loss, prog_bar=True)
        
        return loss
    
    def on_train_epoch_end(self):
        metrics = self.train_metrics.compute(prefix="train_")
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        self.train_metrics.reset()
        
    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute(prefix="val_")
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        self.val_metrics.reset()
        
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute(prefix="test_")
        
        # 모든 메트릭 로깅
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        
        self.test_metrics.reset()
        
        # config에 설정된 메트릭 반환
        return self.test_metrics.get_test_results(metrics)
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.learning_rate
        )
        return optimizer
    
    def on_train_epoch_start(self):
        self.train_metrics.set_epoch(self.current_epoch)
        logging.info(f"\nStarting Epoch {self.current_epoch}/{self.trainer.max_epochs}")

    def on_validation_epoch_start(self):
        self.val_metrics.set_epoch(self.current_epoch)

    def on_test_epoch_start(self):
        self.test_metrics.set_epoch(self.current_epoch)
    
    def extract_features(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Late fusion을 위한 feature extraction 메서드"""
        with torch.no_grad():  # 추론 시에는 gradient 계산 불필요
            return self(batch, return_features=True)
    
# num_unfrozen_layers: 0일 때는 embedding만 사용 (모든 레이어 고정)
# num_unfrozen_layers: N일 때는 상위 N개 레이어만 학습
# 예를 들어:
# num_unfrozen_layers: 0 - embedding만 사용
# num_unfrozen_layers: 2 - 상위 2개 레이어만 학습
# num_unfrozen_layers: 12 - 모든 transformer 레이어 학습