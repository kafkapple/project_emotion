import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.models as models
from ..metrics.image_metrics import ImageEmotionMetrics
import logging
import torch
from typing import Dict, Any
from ..losses.focal_loss import FocalLoss

class PretrainedImageModel(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # dtype 설정
        self._model_dtype = torch.float32 if config.settings.precision == "32" else torch.float16
        
        # 모델 초기화 시 dtype 적용
        self.model = self._init_model().to(dtype=self._model_dtype)
        
        # 분류기 초기화
        self.classifier = self._init_classifier()
        
        # Loss function 초기화
        self.criterion = self._init_criterion().to(dtype=self._model_dtype)
        
        # Freezing 설정 적용
        if config.model.freeze.enabled:
            self._freeze_layers()
        
        # 메트릭스 초기화
        self.train_metrics = ImageEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        self.val_metrics = ImageEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        self.test_metrics = ImageEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        
        # Debug 모드일 때만 모델 정보 출력
        if config.debug.enabled:
            self._log_model_info()
    
    def _init_model(self):
        """모델 초기화"""
        if self.config.model.name == "resnet":
            model = getattr(models, self.config.model.architecture)(pretrained=self.config.model.pretrained)
            if self.config.model.grayscale:
                # Grayscale 입력을 위한 첫 번째 conv layer 수정
                model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            # FC layer 제거
            self.feature_dim = model.fc.in_features
            model.fc = nn.Identity()
            
        elif self.config.model.name == "efficientnet":
            model = getattr(models, self.config.model.architecture)(pretrained=self.config.model.pretrained)
            if self.config.model.grayscale:
                # Grayscale 입력을 위한 첫 번째 conv layer 수정
                model.features[0][0] = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
            # Classifier 제거
            self.feature_dim = model.classifier[-1].in_features
            model.classifier = nn.Identity()
            
        return model
    
    def _init_classifier(self):
        """분류기 초기화"""
        layers = []
        in_features = self.feature_dim
        
        # activation 함수 매핑
        activation_map = {
            'relu': nn.ReLU,
            'gelu': nn.GELU,
            'leakyrelu': nn.LeakyReLU,
            'tanh': nn.Tanh,
            'sigmoid': nn.Sigmoid
        }
        
        # Hidden layers
        for hidden_size in self.config.model.classifier.hidden_sizes:
            layers.extend([
                nn.Linear(in_features, hidden_size),
                nn.BatchNorm1d(hidden_size) if self.config.model.classifier.use_batch_norm else nn.Identity(),
                activation_map[self.config.model.classifier.activation.lower()](),
                nn.Dropout(self.config.model.classifier.dropout)
            ])
            in_features = hidden_size
        
        # Output layer
        layers.append(nn.Linear(in_features, self.config.dataset.num_classes))
        
        return nn.Sequential(*layers)
    
    def _freeze_layers(self):
        """레이어 고정"""
        # 먼저 모든 파라미터 고정
        for param in self.model.parameters():
            param.requires_grad = False
            
        if self.config.model.name == "resnet":
            # ResNet의 마지막 N개 레이어 학습 가능하도록 설정
            layers_to_unfreeze = self.config.model.freeze.num_unfrozen_layers
            if layers_to_unfreeze > 0:
                for param in self.model.layer4[-layers_to_unfreeze:].parameters():
                    param.requires_grad = True
                    
        elif self.config.model.name == "efficientnet":
            # EfficientNet의 마지막 N개 블록 학습 가능하도록 설정
            blocks_to_unfreeze = self.config.model.freeze.num_unfrozen_layers
            if blocks_to_unfreeze > 0:
                total_blocks = len(self.model.features)
                for i in range(total_blocks - blocks_to_unfreeze, total_blocks):
                    for param in self.model.features[i].parameters():
                        param.requires_grad = True
    
    def _log_model_info(self):
        """모델 정보 로깅"""
        if not self.config.debug.model_summary:
            return
        
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"\nModel Architecture: {self.config.model.architecture}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Frozen parameters: {total_params - trainable_params:,}\n")
    
    def forward(self, batch):
        # 배치에서 이미지 데이터 추출 및 dtype 변환
        x = batch['image'].to(dtype=self._model_dtype)
        features = self.model(x)
        
        # EfficientNet 등에서 tuple을 반환하는 경우 처리
        if isinstance(features, tuple):
            features = features[0]
            
        # 분류기 입력을 위한 형태로 변환
        features = features.view(features.size(0), -1)
        return self.classifier(features)
    
    def _init_criterion(self):
        """Loss function 초기화"""
        if self.config.train.loss.name == "focal":
            # 클래스 가중치 설정
            alpha = self._get_class_weights()
            
            return FocalLoss(
                alpha=alpha,
                gamma=self.config.train.loss.focal.gamma
            )
        return nn.CrossEntropyLoss()
    
    def _get_class_weights(self):
        """클래스 가중치 계산"""
        weight_config = self.config.dataset.class_weights
        
        if weight_config.mode == "none":
            return None
        elif weight_config.mode == "manual":
            # 수동으로 지정된 가중치 사용
            weights = torch.zeros(self.config.dataset.num_classes)
            for i, class_name in enumerate(self.config.dataset.class_names):
                weights[i] = weight_config.manual_weights[class_name]
            return weights.to(self.device)
        elif weight_config.mode == "auto":
            # 데이터셋의 클래스 분포에 기반한 자동 가중치 계산
            return self.train_dataloader().dataset.calculate_class_weights().to(self.device)
        else:
            raise ValueError(f"Unknown class weights mode: {weight_config.mode}")
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        
        # 예측값 변환 후 update (validation_step과 동일하게)
        preds = torch.argmax(outputs, dim=1)
        self.train_metrics.update(preds, batch["label"])
        
        self.log("train/loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        
        # 예측값 업데이트
        preds = torch.argmax(outputs, dim=1)
        self.val_metrics.update(preds, batch["label"])  # outputs -> preds로 변경
        
        self.log("validation/loss", loss, prog_bar=True, on_step=False, on_epoch=True, sync_dist=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = self.criterion(outputs, batch["label"])
        
        # 예측값 업데이트
        preds = torch.argmax(outputs, dim=1)
        self.test_metrics.update(preds, batch["label"])  # outputs -> preds로 변경
        
        self.log("test/loss", loss, prog_bar=True, on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        """옵티마이저와 스케줄러 설정"""
        # 옵티마이저 설정
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.train.learning_rate,
            weight_decay=self.config.train.optimizer.weight_decay
        )
        
        # 스케줄러 설정
        scheduler = {
            "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=self.config.train.scheduler.warmup_epochs,
                eta_min=self.config.train.scheduler.min_lr
            ),
            "interval": "epoch",
            "frequency": 1
        }
        
        return [optimizer], [scheduler]
    
    def on_train_epoch_end(self):
        # Classification report 출력 (리셋 전에)
        current_epoch = self.current_epoch + 1
        logging.info(f"\nTrain Epoch {current_epoch} Classification Report:")
        logging.info(self.train_metrics.get_classification_report())
        
        # 메트릭 계산 및 로깅
        metrics = self.train_metrics.compute(prefix="train")
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        
        # 메트릭 리셋
        self.train_metrics.reset()
    
    def on_validation_epoch_end(self):
        # Classification report 출력 (리셋 전에)
        current_epoch = self.current_epoch + 1
        logging.info(f"\nValidation Epoch {current_epoch} Classification Report:")
        logging.info(self.val_metrics.get_classification_report())
        
        # 메트릭 계산 및 로깅
        metrics = self.val_metrics.compute(prefix="validation")
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True, sync_dist=True)
        
        # 메트릭 리셋
        self.val_metrics.reset()
    
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute(prefix="test")
        self.test_metrics.reset()
        
        # 모든 메트릭스 로깅
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        
        # 주요 메트릭 반환
        return {
            "test/macro_f1": metrics["test/macro_f1"],
            "test/weighted_f1": metrics["test/weighted_f1"],
            "test/accuracy": metrics["test/accuracy"]
        }
    
    def on_train_epoch_start(self):
        self.train_metrics.set_epoch(self.current_epoch + 1)
    
    def on_validation_epoch_start(self):
        self.val_metrics.set_epoch(self.current_epoch + 1)
    
    def on_test_epoch_start(self):
        self.test_metrics.set_epoch(self.current_epoch + 1) 