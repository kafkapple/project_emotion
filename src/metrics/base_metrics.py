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
        
    def update(self, preds: torch.Tensor, labels: torch.Tensor):
        """예측값과 레이블 업데이트"""
        if isinstance(preds, torch.Tensor):
            if preds.dim() > 1:  # logits인 경우
                preds = torch.argmax(preds, dim=1)
            preds = preds.cpu().numpy()
        
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        # 디버깅을 위한 로그 추가
        logging.debug(f"Updating metrics - Predictions shape: {preds.shape}, Labels shape: {labels.shape}")
        logging.debug(f"Current accumulated predictions: {len(self.all_preds)}, labels: {len(self.all_labels)}")
        
        self.all_preds.extend(preds)
        self.all_labels.extend(labels)
    
    def _get_unique_labels(self) -> List[int]:
        """실제 데이터에 존재하는 클래스 레이블 반환"""
        return sorted([int(x) for x in set(self.all_labels)])
    
    def _get_active_class_names(self) -> List[str]:
        """실제 데이터에 존재하는 클래스의 이름 반환"""
        unique_labels = self._get_unique_labels()
        return [self.class_names[i] for i in unique_labels]
    
    def _normalize_phase(self, phase: str) -> str:
        """phase 이름을 표준화"""
        phase_map = {
            'val': 'validation',
            'validation': 'validation',
            'eval': 'validation',
            'train': 'train',
            'test': 'test'
        }
        # 접두사 제거 (예: "train_", "validation/")
        clean_phase = phase.rstrip('_/').lstrip('_/')
        return phase_map.get(clean_phase, clean_phase)

    def compute_metrics(self, prefix: str = "") -> Dict[str, float]:
        """메트릭 계산만 수행 (로깅 없이)"""
        # phase 이름 정규화
        phase = self._normalize_phase(prefix.rstrip('_/'))
        
        # 표준화된 접두사 생성
        standard_prefix = f"{phase}/"
        
        # Classification Report 생성 (출력 없이 dict만 생성)
        report_dict = classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=self._get_active_class_names(),
            zero_division=0,
            output_dict=True
        )
        
        # 기본 메트릭 매핑
        metric_mapping = {
            'accuracy': report_dict['accuracy'],
            'macro_f1': report_dict['macro avg']['f1-score'],
            'weighted_f1': report_dict['weighted avg']['f1-score'],
            'macro_precision': report_dict['macro avg']['precision'],
            'macro_recall': report_dict['macro avg']['recall']
        }
        
        # config에 설정된 메트릭만 선택
        selected_metrics = {}
        try:
            if hasattr(self.config, 'metrics') and phase in self.config.metrics:
                phase_config = self.config.metrics[phase]
                if hasattr(phase_config, 'metrics'):
                    for metric in phase_config.metrics:
                        if metric in metric_mapping:
                            selected_metrics[f"{standard_prefix}{metric}"] = metric_mapping[metric]
                else:
                    selected_metrics = {f"{standard_prefix}{k}": v for k, v in metric_mapping.items()}
            else:
                selected_metrics = {f"{standard_prefix}{k}": v for k, v in metric_mapping.items()}
        except Exception as e:
            logging.warning(f"Error in metrics computation: {e}. Using all metrics.")
            selected_metrics = {f"{standard_prefix}{k}": v for k, v in metric_mapping.items()}
        
        return {f"{standard_prefix}{k}": v for k, v in selected_metrics.items()}

    def compute(self, prefix: str = "") -> Dict[str, float]:
        """이전 버전과의 호환성을 위한 메서드"""
        metrics = self.compute_metrics(prefix)
        self.log_metrics(prefix.rstrip('/').replace('_', ''))
        return metrics

    def log_metrics(self, phase: str):
        """모든 메트릭 로깅"""
        # phase 이름 정규화
        phase_map = {
            'val': 'validation',
            'eval': 'validation',
            'train': 'train',
            'test': 'test'
        }
        
        normalized_phase = phase_map.get(phase, phase)
        if normalized_phase not in ['train', 'validation', 'test']:
            logging.warning(f"Unknown phase '{phase}' for metric logging")
            return
        
        active_class_names = self._get_active_class_names()
        
        # Classification Report 생성
        report_dict = classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=active_class_names,
            zero_division=0,
            output_dict=True
        )
        
        # 텍스트 형태의 리포트 생성 및 출력 (여기서만 출력)
        report_text = classification_report(
            self.all_labels, 
            self.all_preds,
            target_names=active_class_names,
            zero_division=0
        )
        
        # 콘솔에 출력 (한 번만)
        logging.info(f"\n{normalized_phase.upper()} Epoch {self.current_epoch} Classification Report:")
        logging.info("="*50)
        logging.info(report_text)
        logging.info("="*50)
        
        # wandb에 로깅
        wandb.log({
            f"{normalized_phase}/classification_report": wandb.Table(
                dataframe=pd.DataFrame(report_dict).transpose()
            ),
            f"{normalized_phase}/epoch": self.current_epoch,
            "current_epoch": self.current_epoch
        })
        
        # Confusion Matrix & ROC Curves
        self._log_confusion_matrix(normalized_phase, active_class_names)
        self._log_combined_curves(normalized_phase, active_class_names)

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
        """F1 score 변화 추적"""
        plt.figure(figsize=(10, 6))
        
        # F1 score 계산 및 저장
        f1 = f1_score(self.all_labels, self.all_preds, average='macro')
        
        # wandb에 스칼라 값으로 저장 (phase 이름 수정)
        metric_name = f"{phase}/f1"  # validation/validation/f1 -> val/f1
        wandb.log({
            metric_name: f1,
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
            # 메트릭 키 이름을 일관되게 변경
            key = f"{phase}/{metric}"  # validation/validation/accuracy -> val/accuracy
            if key in metrics_dict:
                selected_metrics[key] = metrics_dict[key]
                
        return selected_metrics

    def get_test_results(self, metrics_dict: Dict) -> Dict:
        """테스트 결과로 반환할 메트릭 선택"""
        if not hasattr(self.config.metrics.test, 'monitor_metrics'):
            return metrics_dict
        
        results = {}
        for metric in self.config.metrics.test.monitor_metrics:
            key = f"test_{metric}"
            if key in metrics_dict:
                results[key] = metrics_dict[key]
            
        # 주 메트릭이 설정되어 있으면 추가
        if hasattr(self.config.metrics.test, 'main_metric'):
            main_key = f"test_{self.config.metrics.test.main_metric}"
            if main_key in metrics_dict:
                results['test_score'] = metrics_dict[main_key]
            
        return results 

    def get_classification_report(self) -> str:
        """현재 에포크의 classification report 반환"""
        if not self.all_labels or not self.all_preds:
            return "No predictions available"
        
        return classification_report(
            self.all_labels,  # 이미 numpy array
            self.all_preds,   # 이미 numpy array
            target_names=self.class_names,
            zero_division=0
        ) 