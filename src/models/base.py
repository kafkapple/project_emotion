import torch.nn as nn
import pytorch_lightning as pl

class BaseModel(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.save_hyperparameters() 