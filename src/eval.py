import os
import xarray as xr
import torch
import wandb
from torch.utils.data import DataLoader
from tqdm import tqdm
import logging

from src.era5_dataset import ERA5Dataset, TimeMode
from src.fuxi import FuXi
import dotenv

from src.score import compute_weighted_acc, \
    compute_weighted_mae, compute_weighted_rmse

dotenv.load_dotenv()

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

if torch.cuda.is_available():
    device = 'cuda'
if torch.backends.mps.is_available():
    device = 'cpu'


def evaluate(model_path):
    # with wandb.init() as run:
    logger.info('Creating Model')
    model = FuXi(25, 64, 1, 121, 240, heads=1)
    model.load_state_dict(torch.load(model_path))
    logger.info('Load Model successfully')

    logger.info('Creating Eval Dataset')
    test_ds = ERA5Dataset(
        os.environ.get('DATAFOLDER'),
        1,
        TimeMode.AFTER,
        start_time="2010-12-31T23:59:59",
        max_autoregression_steps=1
    )
    loader_params = {'batch_size': None,
                     'batch_sampler': None,
                     'shuffle': False,
                     'num_workers': os.cpu_count() // 2,
                     'pin_memory': True}
    logger.info('Creating DataLoader')
    test_dl = DataLoader(test_ds, **loader_params, sampler=None)
    pbar = tqdm(test_dl, desc='Initialisierung')

    predicted_labels_list = []
    labels_list = []
    times = []
    i=0
    for data in pbar:
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        predicted_labels = model.forward(inputs)
        times.append(i)
        predicted_labels_list.append(predicted_labels.squeeze().detach().cpu().numpy())
        labels_list.append(labels.squeeze().detach().cpu().numpy())
        i += 1

    # Konvertieren in xarray DataArray
    predicted_labels_xr = xr.DataArray(predicted_labels_list, dims=['time', 'variables', 'lat', 'lon'],
                                       coords={'time': times})
    labels_xr = xr.DataArray(labels_list, dims=['time', 'variables', 'lat', 'lon'], coords={'time': times})
    print(compute_weighted_rmse(predicted_labels_xr, labels_xr).values)
    print(compute_weighted_acc(predicted_labels_xr, labels_xr).values)
    print(compute_weighted_mae(predicted_labels_xr, labels_xr).values)

if __name__ == '__main__':
    evaluate("/Users/xgxtphg/Documents/git/DL4WeatherAndClimate/model/model_best_loss_0.0804_20240406-103455.pth")
