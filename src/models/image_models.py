import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
import torchvision.models as models
from ..metrics.image_metrics import ImageEmotionMetrics
import logging
import torch
from typing import Dict, Any

class PretrainedImageModel(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        
        # 모델 초기화
        self.model = self._init_model()
        
        # 분류기 초기화
        self.classifier = self._init_classifier()
        
        # Freezing 설정 적용
        if config.model.freeze.enabled:
            self._freeze_layers()
        
        # 메트릭스 초기화
        self.train_metrics = ImageEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        self.val_metrics = ImageEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        self.test_metrics = ImageEmotionMetrics(config.dataset.num_classes, config.dataset.class_names, config)
        
        # Debug 모드 출력 주석 처리
        # if config.debug.enabled:
        #     self._log_model_info()
    
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logging.info(f"\nModel Architecture: {self.config.model.architecture}")
        logging.info(f"Total parameters: {total_params:,}")
        logging.info(f"Trainable parameters: {trainable_params:,}")
        logging.info(f"Frozen parameters: {total_params - trainable_params:,}\n")
    
    def forward(self, batch):
        # 메모리 최화를 위한 배치 처리
        with torch.cuda.amp.autocast():  # mixed precision 사용
            images = batch['image']
            features = self.model(images)
            if isinstance(features, tuple):
                features = features[0]
            features = features.view(features.size(0), -1)
            return self.classifier(features)
    
    def training_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = F.cross_entropy(outputs, batch["label"])
        self.train_metrics.update(outputs, batch["label"])
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = F.cross_entropy(outputs, batch["label"])
        self.val_metrics.update(outputs, batch["label"])
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def test_step(self, batch, batch_idx):
        outputs = self(batch)
        loss = F.cross_entropy(outputs, batch["label"])
        self.test_metrics.update(outputs, batch["label"])
        self.log("test_loss", loss, prog_bar=True)
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
        metrics = self.train_metrics.compute(prefix="train_")
        self.train_metrics.reset()
        
    def on_validation_epoch_end(self):
        metrics = self.val_metrics.compute(prefix="val_")
        self.val_metrics.reset()
        
    def on_test_epoch_end(self):
        metrics = self.test_metrics.compute(prefix="test_")
        
        # 모든 메트릭스 로깅
        for name, value in metrics.items():
            self.log(name, value, prog_bar=True)
        
        self.test_metrics.reset()
        
        # test_loss 대신 f1 score 반환
        return {
            "test_macro_f1": metrics["test_macro_f1"],
            "test_weighted_f1": metrics["test_weighted_f1"],
            "test_accuracy": metrics["test_accuracy"]
        }
    
    def on_train_epoch_start(self):
        self.train_metrics.set_epoch(self.current_epoch)
    
    def on_validation_epoch_start(self):
        self.val_metrics.set_epoch(self.current_epoch)
    
    def on_test_epoch_start(self):
        self.test_metrics.set_epoch(self.current_epoch) 