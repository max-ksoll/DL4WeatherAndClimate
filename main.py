import os

import torch
from torch.utils.data import DataLoader

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
    logger.info('Using {} device'.format(device))
    logger.info('Creating Model')
    model = FuXi(25, 1024, 16, 121, 240, heads=4)
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
    for batch in tqdm(dl):
        optimizer.zero_grad()
        inputs, labels = batch
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = model.training_step(inputs, labels)
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
