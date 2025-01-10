import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
import logging

from src.logger.base_logger import Logger
from src.data.factory import DataFactory
from src.models.factory import ModelFactory
from src.utils.model_manager import ModelManager
import warnings
warnings.filterwarnings("ignore")

@hydra.main(version_base="1.2", config_path="configs", config_name="config")
def train(config: DictConfig):
    logger = Logger(config)
    logger.print_logger.print_info()
 
    # Setup wandb logger
    wandb_logger = WandbLogger(
        project=config.logger.wandb.project_name,
        entity=config.logger.wandb.entity,
        tags=config.logger.wandb.tags,
        save_dir=config.dirs.outputs
    )
    
    # 모든 데이터셋과 로더를 한 번에 생성
    _, loaders = DataFactory.create_dataset_and_loaders(config)
    
    train_loader = loaders.get('train')
    val_loader = loaders.get('val')
    test_loader = loaders.get('test')

    # 데이터 로더 크기 로깅
    logging.info(f"Train loader size: {len(train_loader)}")
    if val_loader is not None:
        logging.info(f"Validation loader size: {len(val_loader)}")
    if test_loader is not None:
        logging.info(f"Test loader size: {len(test_loader)}")
    
    # Initialize model manager
    _ = ModelManager(config.model_manager.base_path)
    
    # Initialize model
    model = ModelFactory.create(config.model.name, config)
    
    # Setup callbacks
    callbacks = [
        ModelCheckpoint(
            dirpath=config.train.checkpoint.dirpath,
            filename=config.train.checkpoint.filename,
            monitor=config.train.checkpoint.monitor,
            mode=config.train.checkpoint.mode,
            save_top_k=config.train.checkpoint.save_top_k,
            save_weights_only=True
        )
    ]
    
    # Early stopping 설정
    if val_loader is not None and config.train.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor='validation/loss',
                patience=config.train.early_stopping.patience,
                mode=config.train.early_stopping.mode,
                min_delta=config.train.early_stopping.min_delta,
                check_on_train_epoch_end=False
            )
        )

    # Trainer 설정
    trainer = pl.Trainer(
        max_epochs=config.train.max_epochs,
        accelerator="auto",
        devices=1,
        callbacks=callbacks,
        logger=wandb_logger if config.logger.wandb.enabled else None,
        val_check_interval=1.0,
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        gradient_clip_val=config.train.memory_management.gradient_clip_val,
        accumulate_grad_batches=config.train.memory_management.accumulate_grad_batches,
        deterministic=True,
        precision=config.settings.precision
    )
    
    # 학습 실행 시 validation 데이터 로더 확인 및 실행
    if val_loader is not None:
        trainer.fit(model, train_loader, val_loader)
    else:
        logging.warning("No validation data loader provided! Training without validation.")
        trainer.fit(model, train_loader)
    
    # 학습 완료 후 테스트 실행
    if test_loader is not None:
        logging.info("Starting test evaluation...")
        trainer.test(model, test_loader)
    else:
        logging.warning("No test data loader provided! Skipping test evaluation.")

if __name__ == "__main__":
    train()