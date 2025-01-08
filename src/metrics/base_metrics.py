from typing import Dict, List
import torch
import numpy as np
import logging
import wandb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    roc_curve, 
    precision_recall_curve, 
    auc
)
from sklearn.preprocessing import label_binarize
from pathlib import Path
import pandas as pd
from omegaconf import DictConfig

class BaseEmotionMetrics:
    def __init__(self, num_classes: int, class_names: List[str], config: DictConfig):
        self.num_classes = num_classes
        self.class_names = list(class_names)
        self.config = config
        self.current_epoch = 0
        self.reset()
    
    def reset(self):
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
        
    def update(self, logits: torch.Tensor, labels: torch.Tensor):
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            self.all_preds.extend(preds.cpu().numpy())
            self.all_labels.extend(labels.cpu().numpy())
            self.all_probs.extend(probs.cpu().numpy())
    
    def _get_unique_labels(self) -> List[int]:
        """실제 데이터에 존재하는 클래스 레이블 반환"""
        return sorted([int(x) for x in set(self.all_labels)])
    
    def _get_active_class_names(self) -> List[str]:
        """실제 데이터에 존재하는 클래스의 이름 반환"""
        unique_labels = self._get_unique_labels()
        return [self.class_names[i] for i in unique_labels]
    
    def compute(self, prefix: str = "") -> Dict[str, float]:
        """메트릭 계산 및 반환"""
        # Classification Report 생성
        report_dict = classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=self._get_active_class_names(),
            zero_division=0,
            output_dict=True
        )
        
        # 주요 메트릭 추출
        metrics = {
            f"{prefix}accuracy": report_dict['accuracy'],
            f"{prefix}macro_precision": report_dict['macro avg']['precision'],
            f"{prefix}macro_recall": report_dict['macro avg']['recall'],
            f"{prefix}macro_f1": report_dict['macro avg']['f1-score'],
            f"{prefix}weighted_f1": report_dict['weighted avg']['f1-score']
        }
        
        # 메트릭 로깅 및 저장
        self.log_metrics(prefix.replace('_', '') if prefix else 'eval')
        
        return metrics

    def log_metrics(self, phase: str):
        """모든 메트릭 로깅"""
        # 실제 존재하는 클래스 이름 가져오기
        active_class_names = self._get_active_class_names()
        
        # 1. Classification Report 생성 및 저장
        report = classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=active_class_names,
            zero_division=0
        )
        
        # wandb에 테이블로 저장
        report_dict = classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=active_class_names,
            zero_division=0,
            output_dict=True
        )
        
        # F1 커브 로깅
        self._log_f1_curve(phase)
        
        # wandb에 테이블로 저장
        wandb.log({
            f"{phase}/classification_report": wandb.Table(
                dataframe=pd.DataFrame(report_dict).transpose()
            ),
            f"{phase}/epoch": self.current_epoch
        })
        
        # 2. Confusion Matrix
        self._log_confusion_matrix(phase, active_class_names)
        
        # 3. ROC & PR Curves
        self._log_combined_curves(phase, active_class_names)

    def _log_confusion_matrix(self, phase: str, class_names: List[str]):
        """Confusion Matrix 생성 및 로깅"""
        cm = confusion_matrix(self.all_labels, self.all_preds)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names)
        plt.title(f'{phase.capitalize()} Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # wandb에 저장
        wandb.log({f"{phase}/confusion_matrix": wandb.Image(plt)})
        
        # 로컬에 저장
        output_dir = Path(wandb.run.dir) / "plots"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{phase}_confusion_matrix.png")
        plt.close()

    def _log_combined_curves(self, phase: str, class_names: List[str]):
        """ROC 및 PR 커브를 클래스별로 하나의 그래프에 통합"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        unique_labels = self._get_unique_labels()
        
        # ROC curves
        for i, label in enumerate(unique_labels):
            y_true = (np.array(self.all_labels) == label).astype(int)
            y_score = np.array(self.all_probs)[:, label]
            
            fpr, tpr, _ = roc_curve(y_true, y_score)
            ax1.plot(fpr, tpr, label=f'{class_names[i]}')
        
        ax1.plot([0, 1], [0, 1], 'k--')
        ax1.set_title(f'{phase.capitalize()} ROC Curves')
        ax1.set_xlabel('False Positive Rate')
        ax1.set_ylabel('True Positive Rate')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # PR curves
        for i, label in enumerate(unique_labels):
            y_true = (np.array(self.all_labels) == label).astype(int)
            y_score = np.array(self.all_probs)[:, label]
            
            precision, recall, _ = precision_recall_curve(y_true, y_score)
            ax2.plot(recall, precision, label=f'{class_names[i]}')
        
        ax2.set_title(f'{phase.capitalize()} Precision-Recall Curves')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        
        # wandb에 저장
        wandb.log({f"{phase}/curves": wandb.Image(fig)})
        
        # 로컬에 저장
        output_dir = Path(wandb.run.dir) / "plots"
        output_dir.mkdir(exist_ok=True)
        plt.savefig(output_dir / f"{phase}_curves.png")
        plt.close()

    def _log_f1_curve(self, phase: str):
        """F1 score 변화 추"""
        plt.figure(figsize=(10, 6))
        
        # F1 score 계산 및 저장
        f1 = f1_score(self.all_labels, self.all_preds, average='macro')
        
        # wandb에 스칼라 으로 저장
        wandb.log({
            f"{phase}/f1_score": f1,
            "epoch": self.current_epoch
        })
        
        plt.close()

    def set_epoch(self, epoch: int):
        """현재 epoch 설정"""
        self.current_epoch = epoch 

    def get_metrics_for_phase(self, phase: str, metrics_dict: Dict) -> Dict:
        """설정된 메트릭만 반환"""
        if phase not in ['train', 'val', 'test']:
            return metrics_dict
            
        selected_metrics = {}
        phase_metrics = self.config.metrics[phase].metrics
        
        for metric in phase_metrics:
            key = f"{phase}_{metric}"
            if key in metrics_dict:
                selected_metrics[key] = metrics_dict[key]
                
        return selected_metrics

    def get_test_results(self, metrics_dict: Dict) -> Dict:
        """테스트 결과로 반환할 메트릭 선택"""
        results = {}
        for metric in self.config.metrics.test.monitor_metrics:
            key = f"test_{metric}"
            if key in metrics_dict:
                results[key] = metrics_dict[key]
        return results 