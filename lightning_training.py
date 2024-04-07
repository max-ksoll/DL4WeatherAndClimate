import logging
import os
from typing import Tuple

import dotenv
import pytorch_lightning as L
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

import wandb
from src.era5_dataset import ERA5Dataset, TimeMode
from src.fuxi_ligthning import FuXi
from src.sweep_config import getSweepID

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

device = 'auto'
if torch.backends.mps.is_available():
    device = 'cpu'

def create_train_test_datasets(max_autoregression_steps) -> Tuple[DataLoader, DataLoader, torch.Tensor]:
    logger.info('Creating Dataset')
    train_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        max_autoregression_steps,
        TimeMode.BEFORE,
        end_time="2011-11-30T18:00:00",
        max_autoregression_steps=max_autoregression_steps
    )
    test_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        max_autoregression_steps,
        TimeMode.BETWEEN,
        start_time="2011-12-01T00:00:00",
        end_time="2011-12-31T18:00:00",
        max_autoregression_steps=max_autoregression_steps
    )
    loader_params = {'batch_size': None,
                     'batch_sampler': None,
                     'shuffle': False,
                     'num_workers': os.cpu_count() // 2,
                     'pin_memory': True}
    logger.info('Creating DataLoader')
    train_dl = DataLoader(train_ds, **loader_params, sampler=None)
    test_dl = DataLoader(test_ds, **loader_params, sampler=None)
    return train_dl, test_dl, train_ds.get_latitude_weights()


def get_autoregression_steps(autoregression_steps_epochs, epoch):
    smaller_values = [value for value in autoregression_steps_epochs.keys() if int(value) <= epoch]
    if not smaller_values:
        return 1

    return autoregression_steps_epochs[str(max(smaller_values))]


def train():
    with wandb.init() as run:
        config = run.config

        logger.info('Creating Model')
        model = FuXi(config)
        wandb_logger = WandbLogger(id=run.id, resume='allow')
        wandb_logger.watch(model, log_freq=100)
        checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", monitor="val_loss")
        trainer = L.Trainer(
            accelerator=device,
            logger=wandb_logger,
            callbacks=[checkpoint_callback]
        )

        for item in config.get('autoregression_steps_epochs'):
            train_dl, test_dl, lat_weights = create_train_test_datasets(item.get('steps'))

            trainer.fit_loop.max_epochs = item.get('epochs')
            trainer.fit(
                model=model,
                train_dataloaders=train_dl,
                val_dataloaders=test_dl,
            )

        wandb_logger.experiment.unwatch(model)


if __name__ == '__main__':
    wandb.agent(getSweepID(), train)
