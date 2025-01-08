from pathlib import Path
import pandas as pd
import logging
from typing import List, Dict, Optional
import json

class RavdessFilter:
    """RAVDESS 데이터셋 필터링 및 통계 분석"""
    
    def __init__(self, metadata_path: Path):
        self.metadata_path = metadata_path
        self.df = pd.read_csv(metadata_path)
        self.stats = {}
        
    def filter_dataset(self,
                      speech_only: bool = True,
                      emotions: Optional[List[str]] = None,
                      exclude_emotions: Optional[List[str]] = None,
                      emotion_intensity: Optional[List[int]] = None,
                      gender: Optional[List[str]] = None
                      ) -> pd.DataFrame:
        """데이터셋 필터링"""
        df = self.df.copy()
        
        # Speech/Song 필터링
        if speech_only:
            df = df[df['vocal_channel'] == 1]  # 1: speech, 2: song
            
        # 감정 필터링
        if emotions:
            df = df[df['emotion'].isin(emotions)]
        if exclude_emotions:
            df = df[~df['emotion'].isin(exclude_emotions)]
            
        # 감정 강도 필터링
        if emotion_intensity:
            df = df[df['emotion_intensity'].isin(emotion_intensity)]
            
        # 성별 필터링
        if gender:
            df = df[df['gender'].isin(gender)]
            
        return df
    
    def generate_statistics(self, df: Optional[pd.DataFrame] = None) -> Dict:
        """데이터셋 통계 생성"""
        if df is None:
            df = self.df
            
        stats = {
            "total_samples": len(df),
            "unique_actors": {
                "count": len(df['actor'].unique()),
                "distribution": df['actor'].value_counts().to_dict()
            },
            "vocal_channel": {
                "distribution": df['vocal_channel'].map({1: "speech", 2: "song"}).value_counts().to_dict()
            },
            "emotions": {
                "distribution": df['emotion'].value_counts().to_dict(),
                "by_gender": df.groupby(['emotion', 'gender']).size().unstack().to_dict()
            },
            "emotion_intensity": {
                "distribution": df['emotion_intensity'].value_counts().to_dict(),
                "by_emotion": df.groupby(['emotion', 'emotion_intensity']).size().unstack().to_dict()
            },
            "gender": {
                "distribution": df['gender'].value_counts().to_dict()
            }
        }
        
        self.stats = stats
        return stats
    
    def print_statistics(self, stats: Optional[Dict] = None):
        """통계 정보 출력"""
        if stats is None:
            stats = self.stats
            
        logging.info("\n" + "="*50)
        logging.info("RAVDESS Dataset Statistics")
        logging.info("="*50)
        
        logging.info(f"\nTotal Samples: {stats['total_samples']}")
        
        logging.info(f"\nActors:")
        logging.info(f"Total unique actors: {stats['unique_actors']['count']}")
        
        logging.info(f"\nVocal Channel Distribution:")
        for channel, count in stats['vocal_channel']['distribution'].items():
            logging.info(f"  {channel}: {count}")
            
        logging.info(f"\nEmotion Distribution:")
        for emotion, count in stats['emotions']['distribution'].items():
            logging.info(f"  {emotion}: {count}")
            
        logging.info(f"\nEmotion Distribution by Gender:")
        for emotion, gender_dist in stats['emotions']['by_gender'].items():
            logging.info(f"  {emotion}:")
            for gender, count in gender_dist.items():
                logging.info(f"    {gender}: {count}")
                
        logging.info(f"\nEmotion Intensity Distribution:")
        for intensity, count in stats['emotion_intensity']['distribution'].items():
            logging.info(f"  Level {intensity}: {count}")
            
        logging.info(f"\nGender Distribution:")
        for gender, count in stats['gender']['distribution'].items():
            logging.info(f"  {gender}: {count}")
            
        logging.info("\n" + "="*50 + "\n")
    
    def save_statistics(self, output_path: Path, stats: Optional[Dict] = None):
        """통계 정보 JSON 형식으로 저장"""
        if stats is None:
            stats = self.stats
            
        with open(output_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logging.info(f"Statistics saved to {output_path}") 