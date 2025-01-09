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
        self.max_length = config.dataset.audio.max_length
        self.normalize = config.dataset.audio.normalize
        self.augmentation = config.dataset.augmentation
        
        # 클래스 정보 초기화 (필터링 적용)
        self.original_class_names = list(config.dataset.class_names)
        self.class_names = self._get_filtered_class_names()
        self.num_classes = len(self.class_names)
        
        # 데이터셋 로드 및 필터링
        self.samples = self._load_dataset()
        
        # config 업데이트
        self.config.dataset.class_names = self.class_names
        self.config.dataset.num_classes = self.num_classes
        
        # 클래스별 샘플 수 계산 (train split에서만)
        if self.split == 'train':
            self.class_counts = self.samples['emotion'].value_counts()
            if len(self.class_counts) > 0:  # 빈 데이터프레임이 아닌 경우에만
                self.target_samples = int(self.class_counts.max() * 
                                        config.dataset.augmentation.get('balance_ratio', 1.0))
            else:
                self.target_samples = 0
                logging.warning(f"No samples found in {split} split!")
        else:
            # train이 아닌 경우 불필요한 계산 방지
            self.class_counts = None
            self.target_samples = 0

        # 모든 오디오 파일 경로 가져오기
        all_files = self._get_all_files()
        
        # Train/Val/Test Split
        if hasattr(config.dataset, 'splits'):
            # Split ratios from config
            train_ratio = config.dataset.splits.ratios.train
            val_ratio = config.dataset.splits.ratios.val
            test_ratio = config.dataset.splits.ratios.test
            
            # Stratified split 수행
            train_files, val_files, test_files = self._stratified_split(
                all_files, 
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                test_ratio=test_ratio
            )
            
            # 현재 split에 해당하는 파일 목록 선택
            if split == 'train':
                self.files = train_files
            elif split == 'val':
                self.files = val_files
            elif split == 'test':
                self.files = test_files

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
        """데이터셋 분할 적용"""
        # 이미 split이 있는 경우 기존 split 유지
        if 'split' in df.columns:
            logging.info("Using existing splits from dataset")
            return df
            
        # Split 방법 확인
        split_method = self.config.dataset.splits.method
        logging.info(f"Applying {split_method} split...")
        
        if split_method == "actor":
            # Actor 기준으로 분할
            return self._apply_actor_split(df)
        elif split_method == "random":
            # 랜덤 분할
            return self._apply_random_split(df)
        elif split_method == "kfold":
            # K-Fold 분할
            return self._apply_kfold_split(df)
        elif split_method == "stratified":
            # Stratified 분할
            return self._apply_stratified_split(df)
        else:
            raise ValueError(f"Unknown split method: {split_method}")
            
    def _apply_actor_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Actor 기준으로 데이터셋 분할"""
        actors = np.array(sorted(df['actor'].unique()))
        np.random.seed(self.config.dataset.seed)
        np.random.shuffle(actors)
        
        n_actors = len(actors)
        n_train = int(n_actors * self.config.dataset.splits.ratios.train)
        n_val = int(n_actors * self.config.dataset.splits.ratios.val)
        
        train_actors = actors[:n_train]
        val_actors = actors[n_train:n_train + n_val]
        test_actors = actors[n_train + n_val:]
        
        # Split 정보 추가 ('validation' -> 'val')
        df['split'] = 'train'  # 기본값
        df.loc[df['actor'].isin(val_actors), 'split'] = 'val'  # 'validation' -> 'val'
        df.loc[df['actor'].isin(test_actors), 'split'] = 'test'
        
        self._log_split_statistics(df)
        return df
        
    def _apply_random_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """랜덤 분할 적용 (stratified)"""
        # Train-Test 분할
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.dataset.splits.ratios.test,
            stratify=df['emotion'],
            random_state=self.config.dataset.seed
        )
        
        # Train-Val 분할
        val_ratio = self.config.dataset.splits.ratios.val / (
            self.config.dataset.splits.ratios.train + self.config.dataset.splits.ratios.val
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df['emotion'],
            random_state=self.config.dataset.seed
        )
        
        # Split 정보 추가 ('validation' -> 'val')
        df['split'] = 'train'  # 기본값
        df.loc[val_df.index, 'split'] = 'val'  # 'validation' -> 'val'
        df.loc[test_df.index, 'split'] = 'test'
        
        self._log_split_statistics(df)
        return df
        
    def _apply_kfold_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """K-Fold 분할 적용"""
        k = self.config.dataset.splits.k_folds
        current_fold = self.config.dataset.splits.current_fold
        
        if current_fold >= k:
            raise ValueError(f"Invalid fold number: {current_fold} (total folds: {k})")
            
        # Stratified K-Fold 적용
        skf = StratifiedKFold(
            n_splits=k,
            shuffle=True,
            random_state=self.config.dataset.seed
        )
        
        # 현재 fold에 해당하는 분할 찾기
        for i, (train_val_idx, test_idx) in enumerate(skf.split(df, df['emotion'])):
            if i == current_fold:
                train_val_df = df.iloc[train_val_idx]
                test_df = df.iloc[test_idx]
                
                # Train-Val 분할
                val_ratio = self.config.dataset.splits.ratios.val / (
                    self.config.dataset.splits.ratios.train + self.config.dataset.splits.ratios.val
                )
                train_df, val_df = train_test_split(
                    train_val_df,
                    test_size=val_ratio,
                    stratify=train_val_df['emotion'],
                    random_state=self.config.dataset.seed
                )
                
                # Split 정보 추가 ('validation' -> 'val')
                df['split'] = 'train'  # 기본값
                df.loc[val_df.index, 'split'] = 'val'  # 'validation' -> 'val'
                df.loc[test_df.index, 'split'] = 'test'
                
                self._log_split_statistics(df)
                return df
                
        raise ValueError(f"Failed to find fold {current_fold}")
        
    def _log_split_statistics(self, df: pd.DataFrame) -> None:
        """분할 결과 통계 로깅"""
        logging.info("\nDataset split statistics:")
        for split in ['train', 'val', 'test']:  # 'validation' -> 'val'
            split_df = df[df['split'] == split]
            logging.info(f"\n{split} set:")
            logging.info(f"Number of samples: {len(split_df)}")
            
            # Actor 정보 로깅 (있는 경우)
            if 'actor' in df.columns:
                actors = sorted(split_df['actor'].unique())
                logging.info(f"Number of actors: {len(actors)}")
                logging.info(f"Actors: {actors}")
            
            # 클래스 분포 로깅
            class_dist = split_df['emotion'].value_counts()
            logging.info("\nClass distribution:")
            for emotion, count in class_dist.items():
                logging.info(f"{emotion}: {count} ({count/len(split_df)*100:.2f}%)")

    def _load_dataset(self) -> pd.DataFrame:
        """데이터셋 로드 및 필터링"""
        metadata_path = self.root_dir / "ravdess_metadata.csv"
        
        if not metadata_path.exists():
            # 메타데이터 파일이 없으면 생성
            self._generate_ravdess_metadata(self.root_dir)
        
        # 메타데이터 로드
        df = pd.read_csv(metadata_path)
        
        # 클래스 분포 확인 및 로깅 (분할 전)
        logging.info("\nInitial class distribution:")
        class_dist = df['emotion'].value_counts()
        for emotion, count in class_dist.items():
            logging.info(f"{emotion}: {count} ({count/len(df)*100:.2f}%)")
        
        # Split 적용 (아직 split이 없는 경우)
        if 'split' not in df.columns:
            df = self._apply_split(df)
        
        # 현재 split에 해당하는 데이터만 필터링
        df = df[df['split'] == self.split].copy()
        
        # 현재 split의 클래스 분포 로깅
        logging.info(f"\n{self.split} set class distribution:")
        class_dist = df['emotion'].value_counts()
        for emotion, count in class_dist.items():
            logging.info(f"{emotion}: {count} ({count/len(df)*100:.2f}%)")
        
        # 클래스 밸런싱 적용 (train split에만)
        if self.split == 'train' and self.config.dataset.balance.enabled:
            df = self._balance_classes(df)
            # 밸런싱 후 분포 로깅
            logging.info(f"\nAfter balancing - {self.split} set class distribution:")
            class_dist = df['emotion'].value_counts()
            for emotion, count in class_dist.items():
                logging.info(f"{emotion}: {count} ({count/len(df)*100:.2f}%)")
        
        return df
        
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
        
        # Augmentation 적용 (클래스 불균형 고려)
        if self.augmentation.enabled and self.split == 'train':
            waveform = self._apply_augmentation(waveform, sample['emotion'])
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
            "label": torch.tensor(label, dtype=torch.long),
            "emotion": sample['emotion']
        }

    def _generate_ravdess_metadata(self, root_dir: Path) -> bool:
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
        df.loc[df['actor'].isin(val_actors), 'split'] = 'val'  # 'validation' -> 'val'
        df.loc[df['actor'].isin(test_actors), 'split'] = 'test'
        
        df.to_csv(metadata_path, index=False)
        logging.info(f"Metadata saved to {metadata_path}")
        return True

    def _apply_augmentation(self, waveform: torch.Tensor, emotion: str) -> torch.Tensor:
        """오디오 augmentation 적용 (클래스 불균형 고려)"""
        if not self.config.dataset.augmentation.enabled or self.split != 'train':
            return waveform
        
        # train split이 아니거나 class_counts가 없으면 기본 augmentation만 적용
        if self.class_counts is None:
            return self._apply_basic_augmentation(waveform)
        
        # 현재 클래스의 샘플 수
        current_samples = self.class_counts.get(emotion, 0)
        
        # augmentation 횟수 계산
        if current_samples > 0 and self.target_samples > 0:
            num_aug = max(0, min(
                self.config.dataset.augmentation.get('max_augmentations', 3),
                (self.target_samples - current_samples) // current_samples
            ))
        else:
            num_aug = 0
        
        augmented = waveform
        for _ in range(num_aug):
            augmented = self._apply_basic_augmentation(augmented)
        
        return augmented

    def _apply_basic_augmentation(self, waveform: torch.Tensor) -> torch.Tensor:
        """기본 augmentation 적용"""
        if self.config.dataset.augmentation.noise.enabled:
            noise = torch.randn_like(waveform) * self.config.dataset.augmentation.noise.noise_level
            waveform = waveform + noise
        
        if self.config.dataset.augmentation.volume.enabled:
            gain = torch.FloatTensor(1).uniform_(
                self.config.dataset.augmentation.volume.min_gain,
                self.config.dataset.augmentation.volume.max_gain
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

    def calculate_class_weights(self) -> torch.Tensor:
        """클래스별 가중치 계산"""
        if not hasattr(self, '_class_weights'):
            labels = self.get_labels()
            class_counts = torch.bincount(torch.tensor(labels))
            total_samples = len(labels)
            
            # 역수 가중치 계산
            self._class_weights = torch.FloatTensor(len(class_counts))
            for idx, count in enumerate(class_counts):
                self._class_weights[idx] = total_samples / (len(class_counts) * count)
            
        return self._class_weights

    def get_labels(self):
        """데이터셋의 레이블 목록 반환"""
        return [label for _, label in self.samples]

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
        
        else:
            raise ValueError(f"Unknown balance method: {balance_method}")

    def _stratified_split(self, files: List[Path], train_ratio: float, val_ratio: float, test_ratio: float):
        """Stratified split 수행"""
        # 감정 레이블 추출
        labels = [self._get_emotion_from_filename(f.name) for f in files]
        
        # Stratified split
        train_idx, temp_idx = train_test_split(
            range(len(files)),
            train_size=train_ratio,
            stratify=labels,
            random_state=self.config.dataset.seed
        )
        
        # Remaining data를 validation과 test로 나누기
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_idx, test_idx = train_test_split(
            temp_idx,
            train_size=val_ratio_adjusted,
            stratify=[labels[i] for i in temp_idx],
            random_state=self.config.dataset.seed
        )
        
        return (
            [files[i] for i in train_idx],
            [files[i] for i in val_idx],
            [files[i] for i in test_idx]
        )

    def _get_all_files(self) -> List[Path]:
        """모든 오디오 파일 경로 가져오기"""
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {self.root_dir}")
        
        # .wav 파일만 수집
        files = list(self.root_dir.glob("Actor_*/*.wav"))
        
        if not files:
            raise FileNotFoundError(f"No .wav files found in {self.root_dir}")
        
        # 파일 정렬 (재현성을 위해)
        files.sort()
        
        if self.config.debug.enabled:
            logging.info(f"Found {len(files)} audio files")
        
        return files

    def _get_emotion_from_filename(self, filename: str) -> str:
        """파일명에서 감정 레이블 추출"""
        # 파일명 형식: "Actor_01-01-03-01-01-01-01.wav"
        # 세 번째 숫자가 감정 코드
        emotion_code = filename.split("-")[2]
        
        # 감정 코드 매핑
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
        
        return emotion_map.get(emotion_code, "unknown")

    def _apply_stratified_split(self, df: pd.DataFrame) -> pd.DataFrame:
        """Stratified 분할 적용"""
        # emotion과 actor를 결합하여 단일 레이블 생성
        combined_labels = df['emotion'] + '_' + df['actor'].astype(str)
        
        # Train-Test 분할
        train_val_df, test_df = train_test_split(
            df,
            test_size=self.config.dataset.splits.ratios.test,
            stratify=combined_labels,  # 결합된 레이블 사용
            random_state=self.config.dataset.seed
        )
        
        # validation을 위한 새로운 결합 레이블 생성
        train_val_labels = train_val_df['emotion'] + '_' + train_val_df['actor'].astype(str)
        
        # Train-Val 분할
        val_ratio = self.config.dataset.splits.ratios.val / (
            self.config.dataset.splits.ratios.train + self.config.dataset.splits.ratios.val
        )
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_labels,  # 결합된 레이블 사용
            random_state=self.config.dataset.seed
        )
        
        # Split 정보 추가
        df['split'] = 'train'  # 기본값
        df.loc[val_df.index, 'split'] = 'val'
        df.loc[test_df.index, 'split'] = 'test'
        
        self._log_split_statistics(df)
        return df

