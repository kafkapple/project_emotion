import os
import shutil
import random
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig

class ExperimentManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.exp_dir = Path(config.dirs.outputs)
        self.setup_exp_dirs()
        self.set_seed(config.project.seed)
        self._backup_configs()
        
    def setup_exp_dirs(self):
        """실험 디렉토리 구조 생성"""
        for subdir in self.config.dirs.subdirs:
            os.makedirs(self.exp_dir / subdir, exist_ok=True)
            
    def get_exp_path(self, category: str) -> Path:
        """특정 카테고리의 실험 경로 반환"""
        return self.exp_dir / category 