#----------------------
# TRAIN-01: MURAD MODIFIED PAVEMENT DATASET
# Model: Faster RCNN
# Backbone: Resnet-50
#---------------------

import os
import warnings

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import TensorBoardLogger, CometLogger

from src.lightning_classes.lightning_wheat import LitWheat
from src.utils.loggers import JsonLogger
from src.utils.utils import set_seed, save_useful_info, flatten_omegaconf

warnings.filterwarnings("ignore")

MODEL_NAME = 'train-01'

def run(cfg: DictConfig) -> None:
    """
    Run pytorch-lightning model
    Args:
        cfg: hydra config
    """
    set_seed(cfg.training.seed)
    hparams = flatten_omegaconf(cfg)

    model = LitWheat(hparams=hparams, cfg=cfg)

    early_stopping = pl.callbacks.EarlyStopping(**cfg.callbacks.early_stopping.params)
    model_checkpoint = pl.callbacks.ModelCheckpoint(**cfg.callbacks.model_checkpoint.params)
    lr_logger = pl.callbacks.LearningRateLogger()

    tb_logger = TensorBoardLogger(save_dir=cfg.general.save_dir)
    json_logger = JsonLogger()

    trainer = pl.Trainer(
        logger=[tb_logger,
                json_logger],
        early_stop_callback=early_stopping,
        checkpoint_callback=model_checkpoint,
        callbacks=[lr_logger],
        **cfg.trainer,
    )
    trainer.fit(model)

    # save as a simple torch model
    model_name = MODEL_NAME + '.pth'
    print(model_name)
    torch.save(model.model.state_dict(), model_name)


@hydra.main(config_path='config/conf_murad_frcnn.yaml')
def run_model(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # save_useful_info()
    # run(cfg)


if __name__ == '__main__':
    run_model()