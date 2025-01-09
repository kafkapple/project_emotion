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
        
        # 클래스 분포 확인 및 로깅
        class_dist = df['emotion'].value_counts()
        logging.info(f"\n{split_name} set class distribution:")
        for emotion_idx, count in class_dist.items():
            emotion_name = self.config.dataset.class_names[emotion_idx]
            logging.info(f"{emotion_name}: {count} ({count/len(df)*100:.2f}%)")
        
        # 클래스 밸런싱 적용 (train split에만)
        if self.split == 'train' and self.config.dataset.balance.enabled:
            df = self._balance_classes(df)
        
        # pixels 문자열을 numpy array로 변환
        df['pixels'] = df['pixels'].apply(lambda x: np.array([int(p) for p in x.split()]))
        
        return df
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        """단일 샘플 반환"""
        sample = self.samples.iloc[idx]
        
        # 이미지 변환 및 augmentation
        image = sample['pixels'].reshape(48, 48).astype(np.float32)
        image = torch.from_numpy(image)
        image = image.unsqueeze(0)  # Add channel dimension
        
        # Augmentation (train split에서만)
        if self.split == 'train' and self.config.dataset.augmentation.enabled:
            if sample['emotion'] == 6:  # neutral class
                # neutral 클래스에 대해 더 강한 augmentation 적용
                num_aug = self.config.dataset.augmentation.get('neutral_aug_factor', 2)
                for _ in range(num_aug):
                    if random.random() < 0.5:
                        image = self._apply_random_rotation(image)
                    if random.random() < 0.5:
                        image = self._apply_random_noise(image)
        
        # 정규화
        if self.config.dataset.image.normalize:
            image = (image / 255.0 - self.config.dataset.image.mean[0]) / self.config.dataset.image.std[0]
        else:
            image = image / 255.0
            
        return {
            "image": image,
            "label": torch.tensor(sample['emotion'], dtype=torch.long)
        }

    def _apply_random_rotation(self, image: torch.Tensor) -> torch.Tensor:
        """랜덤 회전 적용"""
        angle = random.uniform(-10, 10)
        return TF.rotate(image, angle)

    def _apply_random_noise(self, image: torch.Tensor) -> torch.Tensor:
        """랜덤 노이즈 적용"""
        noise = torch.randn_like(image) * 0.1
        return torch.clamp(image + noise, 0, 1)

    def calculate_class_weights(self) -> torch.Tensor:
        """클래스별 가중치 계산"""
        class_counts = self.samples['emotion'].value_counts().sort_index()
        total_samples = len(self.samples)
        
        # 역수 가중치 계산
        weights = torch.FloatTensor(len(class_counts))
        for idx, count in enumerate(class_counts):
            weights[idx] = total_samples / (len(class_counts) * count)
        
        return weights

    def get_sampler(self):
        """WeightedRandomSampler 생성"""
        if not self.config.dataset.weighted_sampling.enabled:
            return None
        
        # 각 샘플의 가중치 계산
        class_weights = self.calculate_class_weights()
        sample_weights = [class_weights[label] for label in self.samples['emotion']]
        
        # Sampler 생성
        return torch.utils.data.WeightedRandomSampler(
            weights=sample_weights,
            num_samples=len(sample_weights),
            replacement=True
        )

    def _balance_classes(self, df: pd.DataFrame) -> pd.DataFrame:
        """클래스 밸런싱 적용"""
        balance_method = self.config.dataset.balance.method
        target_size = self.config.dataset.balance.target_size
        
        if balance_method == "oversample":
            # 오버샘플링
            max_size = df['emotion'].value_counts().max() if target_size == "auto" else target_size
            balanced_dfs = []
            
            for emotion in df['emotion'].unique():
                emotion_df = df[df['emotion'] == emotion]
                if len(emotion_df) < max_size:
                    # 랜덤 오버샘플링
                    resampled = emotion_df.sample(
                        n=max_size, 
                        replace=True, 
                        random_state=self.config.dataset.seed
                    )
                    balanced_dfs.append(resampled)
                else:
                    balanced_dfs.append(emotion_df)
                    
            return pd.concat(balanced_dfs, ignore_index=True)
            
        elif balance_method == "undersample":
            # 언더샘플링
            min_size = df['emotion'].value_counts().min() if target_size == "auto" else target_size
            balanced_dfs = []
            
            for emotion in df['emotion'].unique():
                emotion_df = df[df['emotion'] == emotion]
                resampled = emotion_df.sample(
                    n=min_size, 
                    replace=False, 
                    random_state=self.config.dataset.seed
                )
                balanced_dfs.append(resampled)
                
            return pd.concat(balanced_dfs, ignore_index=True)