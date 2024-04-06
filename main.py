from datetime import datetime
import os

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


def train():
    with wandb.init() as run:
        logger.info('Using {} device'.format(device))
        logger.info('Creating Model')
        model = FuXi(25, 512, 24, 121, 240, heads=4)
        best_loss = float('inf')
        logger.info('Setting Model as Train')
        model.train()
        logger.info('Put Model on Device')
        model = model.to(device)
        logger.info('Creating Optimizer')
        optimizer = torch.optim.AdamW(model.parameters())
        logger.info('Creating Dataset')
        train_ds = ERA5Dataset(
            os.environ.get('DATAFOLDER'),
            1,
            TimeMode.BEFORE,
            end_time="2022-12-31T18:00:00",
            max_autoregression_steps=1
        )
        test_ds = ERA5Dataset(
            os.environ.get('DATAFOLDER'),
            1,
            TimeMode.BETWEEN,
            start_time="2023-01-01T00:00:00",
            end_time="2023-12-31T18:00:00",
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
        logger.info('Start training')
        pbar = tqdm(train_dl, desc='Initialisierung')
        model_save_dir = 'model'
        os.makedirs(model_save_dir, exist_ok=True)
        for batch in pbar:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = model.training_step(inputs, labels, autoregression_steps=10)
            wandb.log({
               "loss": loss.item(),
            })
            loss.backward()
            optimizer.step()
            loss = loss.detach().cpu().item()
            if loss < best_loss:
                best_loss = loss-loss*0.1
                current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
                filename = f'model_best_loss_{best_loss:.4f}_{current_time}.pth'
                save_path = os.path.join(model_save_dir, filename)
                torch.save(model.state_dict(), save_path)
                logger.info(f'New best model saved with loss: {best_loss:.4f}')
            pbar.set_description(f'Loss: {loss:.4f}')


if __name__ == '__main__':
    wandb.agent(getSweepID(), train)
