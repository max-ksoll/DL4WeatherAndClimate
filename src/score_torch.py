import torch


def compute_weighted_rmse(forecast, label, lat_weights):
    error = forecast - label
    rmse = torch.sqrt((error ** 2 * lat_weights).mean())
    return rmse


def compute_weighted_acc(forecast, labels, lat_weights, clim):
    fa = forecast - clim
    a = labels - clim
    w = lat_weights

    acc = torch.mean((
            torch.sum(fa * a * w, dim=(-1, -2)) /
            torch.sqrt(
                torch.sum(fa ** 2 * w, dim=(-1, -2)) * torch.sum(a ** 2 * w, dim=(-1, -2))
            )
    ))
    return acc


def compute_weighted_mae(forecast, labels, lat_weights):
    error = forecast - labels
    mae = (torch.abs(error) * lat_weights).mean()
    return mae
