"""
Functions for evaluating forecasts.
"""
import numpy as np
import torch
import xarray as xr


def compute_weighted_rmse(forecast, label, lat_weights):
    """
    Compute the RMSE with latitude weighting from two xr.DataArrays.

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        rmse: Latitude weighted root mean squared error
    """
    error = forecast - label
    lat_weights /= lat_weights.mean()
    lat_weights = lat_weights[:, None]
    rmse = torch.sqrt((error ** 2 * lat_weights).mean([0, 2, 3]))
    return rmse


def compute_weighted_acc(forecast, labels, lat_weights):
    """
    Compute the ACC with latitude weighting from two xr.DataArrays.
    WARNING: Does not work if datasets contain NaNs

    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        acc: Latitude weighted acc
    """

    mean = torch.mean(labels, dim=0)
    fa = forecast - mean
    a = labels - mean

    lat_weights /= lat_weights.mean()
    lat_weights = lat_weights[:, None]
    w = lat_weights

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            torch.sum(w * fa_prime * a_prime, dim=[0, 2, 3]) /
            torch.sqrt(
                torch.sum(w * fa_prime ** 2, dim=[0, 2, 3]) * torch.sum(w * a_prime ** 2, dim=[0, 2, 3])
            )
    )
    return acc


def compute_weighted_mae(forecast, labels, lat_weights):
    """
    Compute the MAE with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Forecast. Time coordinate must be validation time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        mae: Latitude weighted root mean absolute error
    """
    error = forecast - labels
    lat_weights /= lat_weights.mean()
    lat_weights = lat_weights[:, None]
    mae = (np.abs(error) * lat_weights).mean([0, 2, 3])
    return mae


def evaluate_iterative_forecast(da_fc, da_true, func, mean_dims=xr.ALL_DIMS):
    """
    Compute iterative score (given by func) with latitude weighting from two xr.DataArrays.
    Args:
        da_fc (xr.DataArray): Iterative Forecast. Time coordinate must be initialization time.
        da_true (xr.DataArray): Truth.
        mean_dims: dimensions over which to average score
    Returns:
        score: Latitude weighted score
    """
    rmses = []
    for f in da_fc.lead_time:
        fc = da_fc.sel(lead_time=f)
        fc['time'] = fc.time + np.timedelta64(int(f), 'h')
        rmses.append(func(fc, da_true, mean_dims))
    return xr.concat(rmses, 'lead_time')
