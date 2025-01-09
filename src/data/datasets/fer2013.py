# src/data/fer2013.py
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, random_split
from PIL import Image
import torchvision.transforms as transforms
import logging
from typing import Tuple, Optional, Dict, Any, List
import kaggle
from dotenv import load_dotenv
from src.data.utils.download import DatasetDownloader
from src.data.datasets.base import BaseDataset
import random
import pandas as pd
import numpy as np

class FER2013Dataset(BaseDataset):
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        super().__init__(config, split)
        
        self.root_dir = Path(config.dataset.root_dir)
        self.csv_path = self.root_dir / "fer2013.csv"
        
        # split 매핑
        self.split_map = {
            'train': 'Training',
            'val': 'PublicTest',
            'test': 'PrivateTest'
        }
        
        # 데이터 로드
        self.samples = self._load_dataset()
        
    def _load_dataset(self) -> pd.DataFrame:
        """CSV 파일에서 데이터 로드"""
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {self.csv_path}")
            
        # CSV 파일 읽기
        df = pd.read_csv(self.csv_path)
        
        # 현재 split에 해당하는 데이터만 필터링
        split_name = self.split_map[self.split]
        df = df[df['Usage'] == split_name].copy()
        
        # pixels 문자열을 numpy array로 변환
        df['pixels'] = df['pixels'].apply(lambda x: np.array([int(p) for p in x.split()]))
        
        return df
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        """단일 샘플 반환"""
        sample = self.samples.iloc[idx]
        
        # 이미지 변환
        image = sample['pixels'].reshape(48, 48).astype(np.float32)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)  # Add channel dimension
        
        # 정규화
        if self.config.dataset.normalize:
            image = image / 255.0
            
        return {
            "image": image,
            "label": torch.tensor(sample['emotion'], dtype=torch.long)
        }