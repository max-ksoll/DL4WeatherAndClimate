# import wandb
#
# from src.sweep_config import getSweepID
import torch
from torch.utils.data import DataLoader

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'mps'

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

from src.fuxi import FuXi
from src.era5_dataset import ERA5Dataset


def train():
    model = FuXi(25, 128, 2, 121, 240, heads=1)
    model.train()
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters())
    ds = ERA5Dataset('/Users/ksoll/git/DL4WeatherAndClimate/data/era5_6hourly.zarr', 1)
    loader_params = {'batch_size': None,
                     'batch_sampler': None,
                     'shuffle': False,
                     'num_workers': 4,
                     'pin_memory': True}
    dl = DataLoader(ds, **loader_params, sampler=None)
    for batch in dl:
        optimizer.zero_grad()
        batch = batch.to(device)
        loss = model.training_step(batch)
        loss.backward()
        optimizer.step()
        print(loss)


if __name__ == '__main__':
    train()
