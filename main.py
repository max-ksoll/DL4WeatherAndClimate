from datetime import datetime
import os

import torch
from torch.utils.data import DataLoader
import wandb
from src.sweep_config import getSweepID
from src.fuxi import FuXi
from src.era5_dataset import ERA5Dataset
import logging
from tqdm import tqdm
import dotenv

dotenv.load_dotenv()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def train():
    entity = "philippgrill"
    project = "CaseStudiesOfAIImplementationResults"
    with wandb.init(project=project, entity=entity) as run:
        logger.info('Using {} device'.format(device))
        logger.info('Creating Model')
        model = FuXi(25, 1024, 36, 121, 240, heads=4)
        best_loss = float('inf')
        logger.info('Setting Model as Train')
        model.train()
        logger.info('Put Model on Device')
        model = model.to(device)
        logger.info('Creating Optimizer')
        optimizer = torch.optim.AdamW(model.parameters())
        logger.info('Creating Dataset')
        ds = ERA5Dataset(os.environ.get('DATAFOLDER'), 1)
        loader_params = {'batch_size': None,
                         'batch_sampler': None,
                         'shuffle': False,
                         'num_workers': os.cpu_count() // 2,
                         'pin_memory': True}
        logger.info('Creating DataLoader')
        dl = DataLoader(ds, **loader_params, sampler=None)
        logger.info('Start training')
        pbar = tqdm(dl, desc='Initialisierung')
        model_save_dir = 'model'
        os.makedirs(model_save_dir, exist_ok=True)
        for batch in pbar:
            optimizer.zero_grad()
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = model.training_step(inputs, labels)
            wandb.log({
               "loss": loss,
            })
            loss.backward()
            optimizer.step()
            if loss.item() < best_loss:
                best_loss = loss.item()
                current_time = datetime.now().strftime('%Y%m%d-%H%M%S')
                filename = f'model_best_loss_{best_loss:.4f}_{current_time}.pth'
                save_path = os.path.join(model_save_dir, filename)
                torch.save(model.state_dict(), save_path)
                logger.info(f'New best model saved with loss: {best_loss:.4f}')
            pbar.set_description(f'Loss: {loss.item():.4f}')


if __name__ == '__main__':
    wandb.agent(getSweepID(), train)
