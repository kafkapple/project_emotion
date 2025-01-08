import os
import shutil
import random
import numpy as np
import torch
from pathlib import Path
from omegaconf import DictConfig
import yaml
import wandb
from datetime import datetime
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

class ExperimentManager:
    def __init__(self, config: DictConfig):
        self.config = config
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self._setup_experiment_dir()
        self.set_seed(config.project.seed)
        
    def _setup_experiment_dir(self) -> Path:
        """실험 디렉토리 설정"""
        exp_dir = Path(self.config.dirs.outputs) / self.timestamp
        
        # 기본 디렉토리 생성
        for subdir in ["config", "metrics", "plots", "checkpoints"]:
            (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
            
        # config 폴더 복사
        self._backup_configs(exp_dir)
        
        return exp_dir
    
    def _backup_configs(self, exp_dir: Path):
        """설정 파일 백업"""
        # config 폴더 전체 복사
        src_config = Path("config")
        if src_config.exists():
            dst_config = exp_dir / "config"
            shutil.copytree(src_config, dst_config, dirs_exist_ok=True)
        
        # 현재 실행 config를 yaml로 저장
        with open(exp_dir / "config" / "current_config.yaml", "w") as f:
            yaml.dump(self.config, f)
            
    def save_experiment_results(self, test_results: Dict, model: pl.LightningModule, wandb_logger: WandbLogger):
        """실험 결과 저장"""
        # 1. 테스트 결과 저장
        test_metrics_path = self.exp_dir / "metrics" / "test_results.yaml"
        with open(test_metrics_path, "w") as f:
            yaml.dump(test_results, f)
            
        # 2. wandb 아티팩트 저장
        if wandb_logger.experiment:
            # wandb 미디어 파일 복사
            media_dir = Path(wandb_logger.experiment.dir) / "media"
            if media_dir.exists():
                for phase in ["train", "val", "test"]:
                    phase_dir = self.exp_dir / "metrics" / phase
                    phase_dir.mkdir(exist_ok=True)
                    
                    # 그래프 이미지 복사
                    for img_file in media_dir.glob(f"{phase}_*.png"):
                        shutil.copy2(img_file, phase_dir)
                    
                    # 분류 리포트 복사
                    for txt_file in media_dir.glob(f"{phase}_*.txt"):
                        shutil.copy2(txt_file, phase_dir)
                    
                    # 테이블 데이터 복사
                    for table_file in media_dir.glob(f"{phase}_*.table.json"):
                        shutil.copy2(table_file, phase_dir)
            
            # wandb 요약 정보 저장
            summary_path = self.exp_dir / "metrics" / "wandb_summary.yaml"
            with open(summary_path, "w") as f:
                yaml.dump(dict(wandb_logger.experiment.summary), f)
    
    def save_metrics(self, phase: str, metrics_dict: Dict, epoch: int):
        """메트릭 저장"""
        metrics_dir = self.exp_dir / "metrics" / phase
        metrics_dir.mkdir(parents=True, exist_ok=True)
        
        # 에포크별 메트릭 저장
        epoch_file = metrics_dir / f"epoch_{epoch:03d}_metrics.yaml"
        with open(epoch_file, "w") as f:
            yaml.dump(metrics_dict, f)
        
        # Classification Report 저장
        if 'classification_report' in metrics_dict:
            report_file = metrics_dir / f"epoch_{epoch:03d}_classification_report.txt"
            with open(report_file, "w") as f:
                f.write(metrics_dict['classification_report'])
    
    def save_plots(self, phase: str, epoch: int):
        """그래프 저장"""
        plots_dir = self.exp_dir / "plots" / f"epoch_{epoch:03d}"
        plots_dir.mkdir(exist_ok=True)
        
        # wandb에서 생성된 그래프 복사
        if wandb.run is not None:
            for plot_file in (Path(wandb.run.dir) / "media").glob(f"{phase}_*.png"):
                shutil.copy2(plot_file, plots_dir)
    
    def set_seed(self, seed: int):
        """랜덤 시드 설정"""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    def get_exp_path(self, category: str) -> Path:
        """특정 카테고리의 실험 경로 반환"""
        return self.exp_dir / category 
    
    def save_wandb_artifacts(self, wandb_logger: WandbLogger):
        """wandb 아티팩트 저장"""
        if not wandb_logger.experiment:
            return
        
        media_dir = Path(wandb_logger.experiment.dir) / "media"
        if not media_dir.exists():
            return
        
        for phase in ["train", "val", "test"]:
            phase_dir = self.exp_dir / "metrics" / phase
            phase_dir.mkdir(parents=True, exist_ok=True)
            
            # 그래프 이미지 복사
            for img_file in media_dir.glob(f"{phase}_*.png"):
                shutil.copy2(img_file, phase_dir)
            
            # 분류 리포트 복사
            for txt_file in media_dir.glob(f"{phase}_*.txt"):
                shutil.copy2(txt_file, phase_dir)
            
            # 테이블 데이터 복사
            for table_file in media_dir.glob(f"{phase}_*.table.json"):
                shutil.copy2(table_file, phase_dir) 