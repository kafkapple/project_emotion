from typing import List, Dict
import torch
from .base import BaseDataLoader

class ImageDataLoader(BaseDataLoader):
    """이미지 데이터 로더"""
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """이미지 배치 데이터 처리"""
        images = torch.stack([item['image'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        
        return {
            'image': images,
            'label': labels
        } 