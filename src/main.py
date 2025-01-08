@hydra.main(version_base="1.2", config_path="config", config_name="config")
def train(config: DictConfig):
    # 폴더 통합
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
    datasets, loaders = DataFactory.create_dataset_and_loaders(config)
    
    train_loader = loaders['train']
    val_loader = loaders['val']
    test_loader = loaders['test']
    
    # Initialize model manager
    model_manager = ModelManager(config.model_manager.base_path)
    
    # Initialize model (수정된 부분)
    model = ModelFactory.create("wav2vec", config)  # 직접 모델명 지정
    
    # ... (나머지 코드는 동일) 