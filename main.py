import os

import torch
from torch.utils.data import DataLoader

from src.fuxi import FuXi
from src.era5_dataset import ERA5Dataset
import logging
from tqdm import tqdm

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)


def train():
    logger.info('Creating Model')
    model = FuXi(25, 128, 2, 121, 240, heads=1)
    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    logger.info('Creating Dataset')
    ds = ERA5Dataset('/Users/ksoll/git/DL4WeatherAndClimate/data/era5_6hourly.zarr', 1)
    loader_params = {'batch_size': None,
                     'batch_sampler': None,
                     'shuffle': False,
                     'num_workers': os.cpu_count() // 2,
                     'pin_memory': True}
    logger.info('Creating DataLoader')
    dl = DataLoader(ds, **loader_params, sampler=None)
    logger.info('Start training')
    for batch in tqdm(dl):
        optimizer.zero_grad()
        batch = batch.to(device)
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        logger.info(loss)


if __name__ == '__main__':
    train()

# import wandb
#
# from src.sweep_config import getSweepID
#
# def train():
#     entity = "philippgrill"
#     project = "CaseStudiesOfAIImplementationResults"
#     with wandb.init(project=project, entity=entity) as run:
#         epochs = wandb.config.get("dataset")
#         wandb.log(
#             {
#                 "mse": 1,
#                 "epochs": epochs,
#             }
#         )
#
# if __name__ == "__main__":
#     wandb.agent(getSweepID(), train)
