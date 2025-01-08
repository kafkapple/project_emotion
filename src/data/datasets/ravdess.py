from typing import Dict, Any, List, Optional
from pathlib import Path
import torchaudio
import torch
from src.data.datasets.base import BaseDataset
from src.data.utils.download import DatasetDownloader
import logging
import librosa
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
from src.data.utils.dataset_filter import RavdessFilter

class RavdessDataset(BaseDataset):
    def __init__(self, config: Dict[str, Any], split: str = 'train'):
        super().__init__(config, split)
        
        self.config = config
        self.split = split
        
        # 기본 설정 초기화
        self.root_dir = Path(config.dataset.root_dir)
        self.sample_rate = config.dataset.audio.sample_rate
        self.duration = config.dataset.audio.duration
        self.feature_type = config.dataset.audio.feature_type
        self.n_mfcc = config.dataset.audio.n_mfcc
        self.augmentation = config.dataset.augmentation
        
        # 클래스 정보 초기화 (필터링 적용)
        self.original_class_names = list(config.dataset.class_names)
        self.class_names = self._get_filtered_class_names()
        self.num_classes = len(self.class_names)
        
        # 데이터셋 로드 및 필터링
        self.samples = self._load_dataset()
        
        # config 업데이트 - 실제 사용되는 클래스 정보로 갱신
        self.config.dataset.class_names = self.class_names
        self.config.dataset.num_classes = self.num_classes

    def _get_filtered_class_names(self) -> List[str]:
        """필터링이 적용된 클래스 이름 목록 반환"""
        class_names = self.original_class_names.copy()
        
        if hasattr(self.config.dataset, 'filtering') and self.config.dataset.filtering.enabled:
            if self.config.dataset.filtering.exclude_emotions:
                class_names = [name for name in class_names 
                             if name not in self.config.dataset.filtering.exclude_emotions]
            if self.config.dataset.filtering.emotions:
                class_names = [name for name in class_names 
                             if name in self.config.dataset.filtering.emotions]
        
        return class_names

    def _check_dataset(self) -> bool:
        """데이터셋이 올바르게 다운로드되었는지 확인"""
        if not self.root_dir.exists():
            return False
            
        # Actor 디렉토리 확인
        actor_dirs = list(self.root_dir.glob("Actor_*"))
        if not actor_dirs:
            logging.warning(f"No Actor directories found in {self.root_dir}")
            return False
            
        # 오디오 파일 확인
        audio_files = list(self.root_dir.glob("Actor_*/*.wav"))
        if not audio_files:
            logging.warning(f"No .wav files found in {self.root_dir}")
            return False
            
        return True
        
    def _apply_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """이터셋 분할 적용"""
        df_split = df.copy()
        
        if self.config.dataset.split_method == "random":
            return self._apply_random_split(df_split)
        elif self.config.dataset.split_method == "kfold":
            return self._apply_kfold_split(df_split)
        elif self.config.dataset.split_method == "stratified":
            return self._apply_stratified_split(df_split)
        else:
            raise ValueError(f"Unknown split method: {self.config.dataset.split_method}")

    def _apply_random_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """랜덤 분할"""
        # Train-Test 분할
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.dataset.splits.ratios.test,
            random_state=self.config.dataset.seed,
            stratify=df['emotion'] if self.config.dataset.splits.get('stratify', True) else None
        )
        
        # Train-Val 분할
        val_ratio = self.config.dataset.splits.ratios.val / (1 - self.config.dataset.splits.ratios.test)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.config.dataset.seed,
            stratify=train_val_df['emotion'] if self.config.dataset.splits.get('stratify', True) else None
        )
        
        # Split 정보 추가
        df.loc[train_df.index, 'split'] = self.config.dataset.splits.train
        df.loc[val_df.index, 'split'] = self.config.dataset.splits.val
        df.loc[test_df.index, 'split'] = self.config.dataset.splits.test
        
        return df

    def _apply_kfold_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """K-Fold 분할 적용"""
        k = self.config.dataset.splits.get('k_folds', 5)
        fold = self.config.dataset.splits.get('current_fold', 0)
        
        # Actor 기준 K-Fold 분할
        actors = np.array(sorted(df['actor'].unique()))
        kf = KFold(n_splits=k, shuffle=True, random_state=self.config.dataset.seed)
        
        # 현재 fold에 해당하는 분할 찾기
        for i, (train_idx, test_idx) in enumerate(kf.split(actors)):
            if i == fold:
                test_actors = actors[test_idx]
                
                # validation을 위해 train 데이터 추가 분할
                train_actors = actors[train_idx]
                val_size = int(len(train_actors) * self.config.dataset.splits.ratios.val)
                val_actors = train_actors[:val_size]
                train_actors = train_actors[val_size:]
                
                # 임시 데이터프레임에 split 정보 추가
                df_split = df.copy()
                df_split['split'] = self.config.dataset.splits.train
                df_split.loc[df_split['actor'].isin(val_actors), 'split'] = self.config.dataset.splits.val
                df_split.loc[df_split['actor'].isin(test_actors), 'split'] = self.config.dataset.splits.test
                
                return df_split
                
        raise ValueError(f"Invalid fold number: {fold}")

    def _apply_stratified_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """계층적 K-Fold 분할"""
        # 먼저 train+val과 test로 분할
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.dataset.splits.ratios.test,
            random_state=self.config.dataset.seed,
            stratify=df['emotion']
        )
        
        # 그 다음 train과 val로 분할
        # val_ratio를 train+val 기준으로 재계산
        val_ratio = self.config.dataset.splits.ratios.val / (1 - self.config.dataset.splits.ratios.test)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            random_state=self.config.dataset.seed,
            stratify=train_val_df['emotion']
        )
        
        # Split 정보 추가
        df_split = df.copy()
        df_split['split'] = self.config.dataset.splits.train  # 기본값을 train으로
        df_split.loc[val_df.index, 'split'] = self.config.dataset.splits.val
        df_split.loc[test_df.index, 'split'] = self.config.dataset.splits.test
        
        # 각 분할의 클��스 분포 확인을 위한 로깅
        logging.info("\nClass distribution in splits:")
        for split_name, split_df in [
            ("Train", train_df), 
            ("Val", val_df), 
            ("Test", test_df)
        ]:
            class_dist = split_df['emotion'].value_counts()
            logging.info(f"\n{split_name} set class distribution:")
            logging.info(class_dist)
        
        return df_split

    def _load_dataset(self) -> pd.DataFrame:
        """데이터셋 로드 및 필터링"""
        metadata_path = self.root_dir / "ravdess_metadata.csv"
        
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
            
        df = pd.read_csv(metadata_path)
        
        # 필터링된 클래스만 포함
        df = df[df['emotion'].isin(self.class_names)]
        
        # 레이블 인덱스 재매핑
        label_map = {name: idx for idx, name in enumerate(self.class_names)}
        df['label'] = df['emotion'].map(label_map)
        
        # 추가 필터링 적용
        if hasattr(self.config.dataset, 'filtering') and self.config.dataset.filtering.enabled:
            if self.config.dataset.filtering.speech_only:
                df = df[df['vocal_channel'] == 1]  # 1: speech
            if self.config.dataset.filtering.emotion_intensity:
                df = df[df['emotion_intensity'].isin(self.config.dataset.filtering.emotion_intensity)]
            if self.config.dataset.filtering.gender:
                df = df[df['gender'].isin(self.config.dataset.filtering.gender)]
        
        # Split 적용
        df_split = self._apply_split(df)
        target_split = self.config.dataset.splits[self.split]
        samples = df_split[df_split['split'] == target_split]
        
        return samples
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        """데이터셋에서 하나의 샘플을 가져옴"""
        sample = self.samples.iloc[index]
        
        # 오디오 로드 및 전처리
        waveform, sr = torchaudio.load(sample['file_path'])
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"1. Initial waveform shape: {waveform.shape}")
        
        # 샘플링 레이트 변환
        if sr != self.config.dataset.audio.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.config.dataset.audio.sample_rate)
            waveform = resampler(waveform)
            if self.config.debug.enabled and self.config.debug.log_shapes:
                logging.info(f"2. After resampling shape: {waveform.shape}")
        
        # 모노로 변환 (필요한 경우)
        if waveform.size(0) > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
            if self.config.debug.enabled and self.config.debug.log_shapes:
                logging.info(f"3. After mono conversion shape: {waveform.shape}")
        
        # 길이 조정
        target_length = int(self.config.dataset.audio.duration * self.config.dataset.audio.sample_rate)
        current_length = waveform.size(1)
        
        if current_length > target_length:
            start = torch.randint(0, current_length - target_length + 1, (1,))
            waveform = waveform[:, start:start + target_length]
        elif current_length < target_length:
            pad_length = target_length - current_length
            waveform = torch.nn.functional.pad(waveform, (0, pad_length))
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"4. After length adjustment shape: {waveform.shape}")
        
        # Augmentation 적용
        if self.split == 'train' and self.config.dataset.augmentation.enabled:
            waveform = self._apply_augmentation(waveform)
            if self.config.debug.enabled and self.config.debug.log_shapes:
                logging.info(f"5. After augmentation shape: {waveform.shape}")
        
        # 차원 확인 및 조정
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
        if self.config.debug.enabled and self.config.debug.log_shapes:
            logging.info(f"6. Final waveform shape: {waveform.shape}")
        
        # 레이블 준비
        label = self.class_names.index(sample['emotion'])
        
        return {
            "audio": waveform,
            "label": torch.tensor(label, dtype=torch.long)
        }

    def _generate_ravdess_metadata(root_dir: Path) -> bool:
        """RAVDESS 데이터셋의 메타데이터 생성"""
        metadata_path = root_dir / "ravdess_metadata.csv"
        
        if metadata_path.exists():
            return True
        
        # 오디오 파일 존재 여부 확인
        audio_files = list(root_dir.glob("**/*.wav"))
        if not audio_files:
            logging.error(f"No audio files found in {root_dir}")
            return False
        
        logging.info("Generating RAVDESS metadata...")
        metadata = []
        
        # 감정 매핑 (RAVDESS 코드 -> 감정 이름)
        emotion_map = {
            "01": "neutral",
            "02": "calm",      # config에 따라 포함/제외 가능
            "03": "happy",
            "04": "sad",
            "05": "angry",
            "06": "fearful",
            "07": "disgust",
            "08": "surprised"
        }
        
        for audio_path in audio_files:
            if "Actor_" not in str(audio_path):
                continue
            
            # 파일명 파싱 (Actor_01-01-01-01-01-01-01.wav 형식)
            filename = audio_path.stem
            parts = filename.split("-")
            
            # 감정 코드 확인
            emotion_code = parts[2]
            if emotion_code not in emotion_map:
                continue
            
            metadata.append({
                'file_path': str(audio_path),
                'actor': int(parts[0].replace("Actor_", "")),
                'vocal_channel': int(parts[1]),
                'emotion': emotion_map[emotion_code],  # 숫자 대신 감정 이름 저장
                'emotion_intensity': int(parts[3]),
                'statement': int(parts[4]),
                'repetition': int(parts[5]),
                'gender': 'female' if int(parts[6]) == 2 else 'male'
            })
        
        # DataFrame 생성 및 저장
        df = pd.DataFrame(metadata)
        
        # Train/Val/Test 분할 (Actor 기준)
        actors = np.array(sorted(df['actor'].unique()))
        np.random.seed(42)
        np.random.shuffle(actors)
        
        n_actors = len(actors)
        train_actors = actors[:int(0.7 * n_actors)]
        val_actors = actors[int(0.7 * n_actors):int(0.85 * n_actors)]
        test_actors = actors[int(0.85 * n_actors):]
        
        df['split'] = 'train'
        df.loc[df['actor'].isin(val_actors), 'split'] = 'validation'
        df.loc[df['actor'].isin(test_actors), 'split'] = 'test'
        
        df.to_csv(metadata_path, index=False)
        logging.info(f"Metadata saved to {metadata_path}")
        return True

    def _apply_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """오디오 파형에 augmentation 적용"""
        if not self.config.dataset.augmentation.enabled or self.split != 'train':
            return waveform
        
        aug_config = self.config.dataset.augmentation
        
        # 가우시안 노이즈 추가
        if aug_config.noise.enabled:
            noise = torch.randn_like(waveform) * aug_config.noise.noise_level
            waveform = waveform + noise
        
        # 볼륨 변경
        if aug_config.volume.enabled:
            gain = torch.FloatTensor(1).uniform_(
                aug_config.volume.min_gain,
                aug_config.volume.max_gain
            )
            waveform = waveform * gain
        
        return waveform

    @classmethod
    def create_filtered_dataset(cls, 
                              config: Dict,
                              split: str,
                              speech_only: bool = True,
                              emotions: Optional[List[str]] = None,
                              exclude_emotions: Optional[List[str]] = None,
                              emotion_intensity: Optional[List[int]] = None,
                              gender: Optional[List[str]] = None) -> 'RavdessDataset':
        """필터링된 RAVDESS 데이터셋 생성"""
        
        # 메타데이터 파일 경로
        metadata_path = Path(config.dataset.root_dir) / "ravdess_metadata.csv"
        
        # 필터 객체 생성
        filter_obj = RavdessFilter(metadata_path)
        
        # 데이터 필터링
        filtered_df = filter_obj.filter_dataset(
            speech_only=speech_only,
            emotions=emotions,
            exclude_emotions=exclude_emotions,
            emotion_intensity=emotion_intensity,
            gender=gender
        )
        
        # 통계 생성 및 저장
        stats = filter_obj.generate_statistics(filtered_df)
        filter_obj.print_statistics(stats)
        
        # 통계 저장
        stats_path = Path(config.dataset.root_dir) / f"statistics_{split}.json"
        filter_obj.save_statistics(stats_path, stats)
        
        # 필터링된 데이터셋으로 새로운 인스턴스 생성
        dataset = cls(config, split)
        dataset.samples = filtered_df
        
        return dataset

