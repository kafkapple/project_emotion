# src/data/utils/download.py
import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import kaggle
from dotenv import load_dotenv
import requests
import shutil

class DatasetDownloader:
    DATASET_URLS = {
        "ravdess": "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip",
        "fer2013": "fer2013/fer2013/fer2013.csv"  # Kaggle dataset path
    }
    
    _instance = None
    _downloaded = set()  # 다운로드 완료된 데이터셋 추적
    
    @classmethod
    def download_and_extract(cls, dataset_name: str, root_dir: str):
        """데이터셋 다운로드 및 압축 해제"""
        root_dir = Path(root_dir)
        root_dir.mkdir(parents=True, exist_ok=True)
        
        if dataset_name == "ravdess":
            if (root_dir / "Actor_01").exists() and any((root_dir / "Actor_01").iterdir()):
                logging.info(f"RAVDESS dataset already exists at {root_dir}")
                return True
            return cls._download_ravdess(root_dir)
            
        elif dataset_name == "fer2013":
            if (root_dir / "fer2013.csv").exists():
                logging.info(f"FER2013 dataset already exists at {root_dir}")
                return True
                
            try:
                # Kaggle API 인증
                load_dotenv()
                kaggle.api.authenticate()
                
                # 데이터셋 다운로드
                logging.info("Downloading FER2013 dataset...")
                kaggle.api.dataset_download_file(
                    dataset="deadskull7/fer2013",
                    file_name="fer2013.csv",
                    path=str(root_dir)
                )
                
                # 압축 해제
                zip_path = root_dir / "fer2013.csv.zip"
                if zip_path.exists():
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall(root_dir)
                    zip_path.unlink()  # 압축 파일 삭제
                
                return True
                
            except Exception as e:
                logging.error(f"Error downloading FER2013: {e}")
                logging.info(
                    "Please ensure you have set KAGGLE_USERNAME and KAGGLE_KEY "
                    "in your .env file or manually download from: "
                    "https://www.kaggle.com/datasets/deadskull7/fer2013"
                )
                raise
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
        
    def __init__(self, config):
        self.config = config
        self.root_dir = Path(config.dataset.root_dir)
        
        # Load environment variables
        load_dotenv()
        
        # Setup Kaggle credentials
        self._setup_kaggle_credentials()
        
    def _setup_kaggle_credentials(self):
        """Setup Kaggle credentials from environment variables"""
        kaggle_username = os.getenv('KAGGLE_USERNAME')
        kaggle_key = os.getenv('KAGGLE_KEY')
        
        if not (kaggle_username and kaggle_key):
            logging.warning(
                "Kaggle credentials not found in .env file. "
                "Please set KAGGLE_USERNAME and KAGGLE_KEY"
            )
            return
            
        # Create kaggle.json if it doesn't exist
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        kaggle_cred = kaggle_dir / 'kaggle.json'
        if not kaggle_cred.exists():
            kaggle_cred.write_text(
                f'{{"username":"{kaggle_username}","key":"{kaggle_key}"}}'
            )
            # Restrict access to prevent warnings
            kaggle_cred.chmod(0o600)
            
    @staticmethod
    def _download_fer2013(root_dir: Path):
        csv_path = root_dir / "fer2013.csv"
        
        if not csv_path.exists():
            logging.info("Downloading FER2013 dataset...")
            try:
                # Download using Kaggle API
                kaggle.api.authenticate()
                kaggle.api.dataset_download_file(
                    dataset="deadskull7/fer2013",
                    file_name="fer2013.csv",
                    path=root_dir
                )
            except Exception as e:
                logging.error(f"Error downloading FER2013: {e}")
                logging.info(
                    "Please ensure you have set KAGGLE_USERNAME and KAGGLE_KEY "
                    "in your .env file or manually download from: "
                    "https://www.kaggle.com/datasets/deadskull7/fer2013"
                )
                raise
                
        # Convert CSV to more efficient format
        if not (root_dir / "processed").exists():
            logging.info("Processing FER2013 dataset...")
            DatasetDownloader._process_fer2013(csv_path, root_dir)
            
    @staticmethod
    def _process_fer2013(csv_path: Path, root_dir: Path):
        # Read CSV
        df = pd.read_csv(csv_path)
        
        # Create directories for each split
        processed_dir = root_dir / "processed"
        processed_dir.mkdir(exist_ok=True)
        
        for split in ["Training", "PublicTest", "PrivateTest"]:
            split_dir = processed_dir / split.lower()
            split_dir.mkdir(exist_ok=True)
            
            # Filter data for current split
            split_data = df[df['Usage'] == split]
            
            # Process each image and save
            for idx, row in split_data.iterrows():
                emotion = row['emotion']
                pixels = np.array([int(p) for p in row['pixels'].split()], dtype=np.uint8)
                img = pixels.reshape(48, 48)
                
                # Save as numpy array (more efficient than image files for this dataset)
                emotion_dir = split_dir / str(emotion)
                emotion_dir.mkdir(exist_ok=True)
                np.save(emotion_dir / f"{idx}.npy", img)

    def download_ravdess(self):
        """RAVDESS 데이터셋 다운로드"""
        version = self.config.dataset.version
        version_dir = self.root_dir / version  # small 또는 full 전용 디렉토리
        version_dir.mkdir(parents=True, exist_ok=True)
        
        if version == "small":
            self._download_kaggle_ravdess(version_dir)
        elif version == "full":
            self._download_full_ravdess(version_dir)
        else:
            raise ValueError(f"Unknown dataset version: {version}")
    
    def _download_kaggle_ravdess(self, version_dir: Path):
        """Kaggle의 small RAVDESS 데이터셋 다운로드"""
        if not (version_dir / "ravdess-emotional-speech-audio").exists():
            logging.info("Downloading RAVDESS small dataset from Kaggle...")
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'uwrfkaggler/ravdess-emotional-speech-audio',
                path=version_dir,
                unzip=True
            )
            
            # 메타데이터 생성
            self._generate_ravdess_metadata(version_dir, "small")

    def _download_full_ravdess(self, version_dir: Path):
        """공식 RAVDESS 전체 데이터셋 다운로드"""
        if not version_dir.exists() or not any(version_dir.iterdir()):
            logging.info("Downloading RAVDESS full dataset...")
            
            # 데이터셋 다운로드
            zip_path = version_dir / "ravdess_full.zip"
            full_dataset_url = "https://zenodo.org/record/1188976/files/Audio_Speech_Actors_01-24.zip"
            
            os.system(f"wget {full_dataset_url} -O {zip_path}")
            
            # 압축 해제
            logging.info("Extracting dataset...")
            os.system(f"unzip {zip_path} -d {version_dir}")
            
            # 압축 파일 삭제
            os.remove(zip_path)
            
            # 메타데이터 생성
            self._generate_ravdess_metadata(version_dir, "full")
            
            logging.info("Download and extraction completed!")

    def _generate_ravdess_metadata(self, version_dir: Path, version: str):
        """RAVDESS 메타데이터 생성"""
        metadata_path = version_dir / f"ravdess_{version}_metadata.csv"
        
        # 오디오 파일 찾기
        audio_files = list(version_dir.rglob("*.wav"))
        
        metadata = []
        for audio_path in audio_files:
            try:
                # 파일명 파싱 (예: 03-01-04-01-02-01-01.wav)
                filename = audio_path.stem
                parts = filename.split("-")
                
                metadata.append({
                    'file_path': str(audio_path.relative_to(version_dir)),
                    'emotion': self._get_emotion_label(parts[2]),
                    'actor_id': int(audio_path.parent.name.replace("Actor_", "")),
                    'intensity': int(parts[3]),
                    'gender': 'female' if int(parts[6]) == 2 else 'male'
                })
                
            except Exception as e:
                logging.warning(f"Error processing file {audio_path}: {e}")
                continue
        
        # 메타데이터를 DataFrame으로 변환 및 저장
        df = pd.DataFrame(metadata)
        df.to_csv(metadata_path, index=False)
        
        # 데이터셋 통계 출력
        logging.info(f"\nRAVDESS {version.upper()} Dataset Statistics:")
        logging.info(f"Total files: {len(df)}")
        logging.info("\nEmotion distribution:")
        logging.info(df['emotion'].value_counts())
        logging.info("\nGender distribution:")
        logging.info(df['gender'].value_counts())

    def _get_emotion_label(self, emotion_code: str) -> str:
        """감정 코드를 레이블로 변환"""
        emotion_map = {
            "01": "neutral",
            "02": "calm",
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        return emotion_map[emotion_code]

    def download_fer2013(self, root_dir: Path) -> bool:
        """FER2013 데이터셋 다운로드"""
        try:
            import kaggle
            
            # 데이터셋 경로 생성
            dataset_path = root_dir / "fer2013"
            dataset_path.mkdir(parents=True, exist_ok=True)
            
            # Kaggle에서 데이터셋 다운로드
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                'msambare/fer2013',
                path=str(dataset_path),
                unzip=True
            )
            
            # 메타데이터 생성
            return self._generate_fer2013_metadata(dataset_path)
            
        except Exception as e:
            logging.error(f"Error downloading FER2013 dataset: {e}")
            return False

    def _generate_fer2013_metadata(self, root_dir: Path) -> bool:
        """FER2013 메타데이터 생성"""
        try:
            metadata_path = root_dir / "fer2013_metadata.csv"
            
            # CSV 파일 읽기
            df = pd.read_csv(root_dir / "fer2013.csv")
            
            # 데이터셋 통계 출력
            logging.info("\nFER2013 Dataset Statistics:")
            logging.info(f"Total samples: {len(df)}")
            logging.info("\nClass distribution:")
            logging.info(df['emotion'].value_counts())
            
            # 메타데이터 저장
            df.to_csv(metadata_path, index=False)
            logging.info(f"\nMetadata saved to {metadata_path}")
            
            return True
            
        except Exception as e:
            logging.error(f"Error generating FER2013 metadata: {e}")
            return False

# src/data/validation.py
class DatasetValidator:
    @staticmethod
    def validate_fer2013(root_dir: Path) -> bool:
        processed_dir = root_dir / "processed"
        if not processed_dir.exists():
            return False
            
        # Check for expected splits
        expected_splits = ["training", "publictest", "privatetest"]
        for split in expected_splits:
            split_dir = processed_dir / split
            if not split_dir.exists():
                return False
                
            # Check for emotion class directories (0-6)
            for emotion in range(7):
                emotion_dir = split_dir / str(emotion)
                if not emotion_dir.exists():
                    return False
                    
                # Check if directory contains any files
                if not list(emotion_dir.glob("*.npy")):
                    return False
                    
        return True