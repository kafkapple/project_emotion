from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple
from pathlib import Path
from torch.utils.data import Dataset
import torch

class BaseDataset(Dataset):
    """Base class for all emotion datasets"""
    
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        """Initialize dataset"""
        self.config = config
        self.split = split
        self.root_dir = Path(config.dataset.root_dir)
        
    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples in the dataset"""
        pass
        
    @abstractmethod
    def __getitem__(self, idx: int) -> Tuple[Any, int]:
        """Return a sample and its label"""
        pass 

    def calculate_class_weights(self) -> torch.Tensor:
        """클래스별 가중치 계산 - 하위 클래스에서 구현"""
        raise NotImplementedError
        
    def get_sampler(self):
        """WeightedRandomSampler 생성"""
        if not hasattr(self.config.dataset, 'weighted_sampling') or \
           not self.config.dataset.weighted_sampling.enabled or \
           self.split != 'train':
            return None
            
        try:
            # 각 샘플의 가중치 계산
            class_weights = self.calculate_class_weights()
            sample_weights = [class_weights[label] for label in self.get_labels()]
            
            # Sampler 생성
            return torch.utils.data.WeightedRandomSampler(
                weights=sample_weights,
                num_samples=len(sample_weights),
                replacement=True
            )
        except NotImplementedError:
            return None
            
    def get_labels(self):
        """데이터셋의 레이블 목록 반환 - 하위 클래스에서 구현"""
        raise NotImplementedError 