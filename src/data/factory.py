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
    def create_dataset_and_loaders(config: DictConfig):
        """모든 split에 대한 데이터셋과 로더를 한 번에 생성"""
        # 데이터셋 다운로드 (한 번만 실행)
        DatasetDownloader.download_and_extract(config.dataset.name, config.dataset.root_dir)
        
        datasets = {}
        loaders = {}
        
        # Train dataset & loader (필수)
        datasets['train'] = DataFactory._create_dataset(config.dataset.name, config, 'train')
        if datasets['train'] is None:
            raise ValueError("Training dataset cannot be None")
            
        loaders['train'] = DataLoader(
            datasets['train'],
            batch_size=config.train.batch_size,
            shuffle=True,
            num_workers=config.train.num_workers,
            pin_memory=config.train.memory_management.pin_memory
        )
        
        # Validation과 Test dataset & loader (선택)
        for split in ['val', 'test']:
            dataset = DataFactory._create_dataset(config.dataset.name, config, split)
            datasets[split] = dataset
            
            if dataset is not None:
                loaders[split] = DataLoader(
                    dataset,
                    batch_size=config.train.batch_size,
                    shuffle=False,
                    num_workers=config.train.num_workers,
                    pin_memory=config.train.memory_management.pin_memory
                )
        
        return datasets, loaders
    
    @staticmethod
    def _create_dataset(dataset_name: str, config: DictConfig, split: str):
        """Create dataset for given split"""
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
        
        # Sampler 설정
        sampler = dataset.get_sampler() if is_train else None
        shuffle = is_train and not sampler and config.train.dataloader.shuffle
        
        return DataLoader(
            dataset,
            batch_size=config.train.batch_size,
            num_workers=config.train.num_workers,
            shuffle=shuffle,  # sampler가 있으면 shuffle=False
            sampler=sampler,
            drop_last=config.train.dataloader.drop_last,
            pin_memory=config.train.memory_management.pin_memory,
            persistent_workers=config.train.memory_management.persistent_workers,
            prefetch_factor=config.train.memory_management.prefetch_factor
        )