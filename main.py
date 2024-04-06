from datetime import datetime
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
import wandb
from src.sweep_config import getSweepID
from src.fuxi import FuXi
from src.era5_dataset import ERA5Dataset, TimeMode
import logging
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'cpu'

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def create_train_test_datasets() -> Tuple[DataLoader, DataLoader]:
    logger.info('Creating Dataset')
    # train_ds = ERA5Dataset(
    #     os.environ.get('DATAFOLDER'),
    #     1,
    #     TimeMode.BEFORE,
    #     end_time="2022-12-31T18:00:00",
    #     max_autoregression_steps=1
    # )
    # test_ds = ERA5Dataset(
    #     os.environ.get('DATAFOLDER'),
    #     1,
    #     TimeMode.BETWEEN,
    #     start_time="2023-01-01T00:00:00",
    #     end_time="2023-12-31T18:00:00",
    #     max_autoregression_steps=1
    # )
    train_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        1,
        TimeMode.BEFORE,
        end_time="2010-01-31T18:00:00",
        max_autoregression_steps=1
    )
    test_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        1,
        TimeMode.BETWEEN,
        start_time="2010-02-01T00:00:00",
        end_time="2010-02-28T18:00:00",
        max_autoregression_steps=1
    )
    loader_params = {'batch_size': None,
                     'batch_sampler': None,
                     'shuffle': False,
                     'num_workers': os.cpu_count() // 2,
                     'pin_memory': True}
    logger.info('Creating DataLoader')
    train_dl = DataLoader(train_ds, **loader_params, sampler=None)
    test_dl = DataLoader(test_ds, **loader_params, sampler=None)
    return train_dl, test_dl


def train_epoch(model, optimizer, train_loader):
    whole_loss = []
    pbar = tqdm(train_loader, desc='Train Loss: ', leave=False)
    for batch in pbar:
        optimizer.zero_grad()
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = model.step(inputs, labels, autoregression_steps=1)
        loss.backward()
        optimizer.step()
        whole_loss.append(loss.detach().cpu().item())
        pbar.set_description(f'Train Loss: {loss.detach().cpu().item():.4f}')
    return np.mean(whole_loss)


def val_epoch(model, val_loader):
    whole_loss = []
    pbar = tqdm(val_loader, desc='Val Loss: ', leave=False)
    for batch in pbar:
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = model.step(inputs, labels, autoregression_steps=1)
        whole_loss.append(loss.detach().cpu().item())
        pbar.set_description(f'Val Loss: {loss.detach().cpu().item():.4f}')
    return np.mean(whole_loss)


def get_autoregression_steps(autoregression_steps_epochs, epoch):
    smaller_values = [value for value in autoregression_steps_epochs.keys() if value <= epoch]
    if not smaller_values:
        return 1

    return max(smaller_values)


def train():
    with wandb.init() as run:
        config = run.config
        os.makedirs(os.environ.get('MODEL_DIR'), exist_ok=True)
        logger.info('Using {} device'.format(device))
        logger.info('Creating Model')
        model_parameter = config.get('model_parameter')
        model = FuXi(25,
                     model_parameter['channel'],
                     model_parameter['transformer_blocks'],
                     121, 240,
                     heads=model_parameter['heads'])

        best_loss = float('inf')
        model.train()
        model = model.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.get("learning_rate"))

        train_dl, test_dl = create_train_test_datasets()

        autoregression_steps_epochs = config.get('autoregression_steps_epochs')
        for epoch in range(config.get("epochs")):

            autoregression_steps = get_autoregression_steps(autoregression_steps_epochs, epoch)

            train_loss = train_epoch(model, optimizer, train_dl)
            test_loss = val_epoch(model, test_dl)

            run.log({
                'train_loss': train_loss,
                'test_loss': test_loss
            })

            if test_loss < best_loss:
                current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
                filename = f'model_best_loss_{test_loss:.4f}_{current_time}.pth'
                save_path = os.path.join(os.environ.get('MODEL_DIR'), filename)
                torch.save(model.state_dict(), save_path)
                logger.info(f'New best model saved with loss: {best_loss:.4f}')
                best_loss = test_loss - test_loss * 0.1


if __name__ == '__main__':
    wandb.agent(getSweepID(), train)
