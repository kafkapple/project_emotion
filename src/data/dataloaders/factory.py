from typing import Dict, Any
from pathlib import Path
from omegaconf import OmegaConf  # pyyaml 대신 OmegaConf 사용
from .audio import AudioDataLoader
from .image import ImageDataLoader
from ..datasets.base import BaseDataset

class DataLoaderFactory:
    @staticmethod
    def create_dataloader(dataset: BaseDataset, config: Dict[str, Any]):
        """데이터셋 타입에 따른 적절한 DataLoader 생성"""
        # modalities.yaml 파일에서 데이터셋-모달리티 매핑 로드
        modalities_config = DataLoaderFactory._load_modalities_config()
        
        # 데이터셋의 모달리티 찾기
        dataset_name = config.dataset.name
        modality = DataLoaderFactory.get_dataset_modality(dataset_name)
        
        if modality == 'audio':
            return AudioDataLoader(dataset, config)
        elif modality == 'image':
            return ImageDataLoader(dataset, config)
        else:
            raise ValueError(f"Unknown modality for dataset: {dataset_name}")
    
    @staticmethod
    def _load_modalities_config() -> Dict:
        """모달리티 설정 파일 로드"""
        config_path = Path("config/dataset/modalities.yaml")
        if not config_path.exists():
            raise FileNotFoundError(f"Modalities config not found: {config_path}")
            
        return OmegaConf.load(config_path)
    
    @staticmethod
    def get_dataset_modality(dataset_name: str) -> str:
        """데이터셋의 모달리티 타입 반환"""
        modalities_config = DataLoaderFactory._load_modalities_config()
        
        for modality, modality_config in modalities_config['modalities'].items():
            for dataset in modality_config['datasets']:
                if dataset['name'] == dataset_name:
                    return modality
                    
        raise ValueError(f"Dataset {dataset_name} not found in modalities config") 