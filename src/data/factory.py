from typing import Dict, Any, Tuple
from torch.utils.data import DataLoader
from .datasets.base import BaseDataset
from .datasets.ravdess import RavdessDataset
from .datasets.fer2013 import FER2013Dataset
from .dataloaders.base import BaseDataLoader, get_collate_fn
from .utils.download import DatasetDownloader
from omegaconf import DictConfig
class DataFactory:
    @staticmethod
    def create_dataset_and_loaders(config):
        """모든 split에 대한 데이터셋과 로더를 한 번에 생성"""
        # 데이터셋 다운로드 (한 번만 실행)
        DatasetDownloader.download_and_extract(config.dataset.name, config.dataset.root_dir)
        
        # 각 split에 대한 데이터셋과 로더 생성
        datasets = {}
        loaders = {}
        
        for split in ['train', 'val', 'test']:
            # 다운로드 없이 데이터셋 생성
            dataset = DataFactory._create_dataset(config.dataset.name, config, split)
            loader = DataFactory.create_dataloader(dataset, config, split)
            datasets[split] = dataset
            loaders[split] = loader
            
        return datasets, loaders
    
    @staticmethod
    def _create_dataset(dataset_name: str, config: DictConfig, split: str):
        """데이터셋 생성 (다운로드 없이)"""
        if dataset_name == "ravdess":
            return RavdessDataset(config, split)
        elif dataset_name == "fer2013":
            return FER2013Dataset(config, split)
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
            
    @staticmethod
    def create_dataloader(dataset, config: DictConfig, split: str):
        """데이터로더 생성"""
        is_train = split == 'train'
        
        return DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            shuffle=is_train and config.train.dataloader.shuffle,
            drop_last=config.train.dataloader.drop_last,
            pin_memory=config.train.memory_management.pin_memory,
            persistent_workers=config.train.memory_management.persistent_workers,
            prefetch_factor=config.train.memory_management.prefetch_factor
        )