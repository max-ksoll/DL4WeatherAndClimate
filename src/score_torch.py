import torch


def compute_weighted_rmse(forecast, label, lat_weights):
    error = forecast - label
    lat_weights /= lat_weights.mean()
    lat_weights = lat_weights[:, None]
    rmse = torch.sqrt((error ** 2 * lat_weights).mean([0, 2, 3]))
    return rmse


def compute_weighted_acc(forecast, labels, lat_weights):
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
    error = forecast - labels
    lat_weights /= lat_weights.mean()
    lat_weights = lat_weights[:, None]
    mae = (torch.abs(error) * lat_weights).mean([0, 2, 3])
    return mae
