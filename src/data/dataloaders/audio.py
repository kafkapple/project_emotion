from typing import List, Dict
import torch
from .base import BaseDataLoader

class AudioDataLoader(BaseDataLoader):
    """오디오 데이터 로더"""
    
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """오디오 배치 데이터 처리"""
        audio = torch.stack([item['audio'] for item in batch])
        labels = torch.tensor([item['label'] for item in batch])
        
        # 차원 조정: [batch_size, channels, sequence_length]
        if len(audio.shape) == 4:  # [B, 1, 1, seq_len]
            audio = audio.squeeze(2)  # [B, 1, seq_len]
            
        return {
            'audio': audio,
            'label': labels
        } 