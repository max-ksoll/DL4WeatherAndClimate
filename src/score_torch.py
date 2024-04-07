import torch


def compute_weighted_rmse(forecast, label, lat_weights):
    print(forecast.shape, label.shape, lat_weights.shape)
    error = forecast - label
    lat_weights /= lat_weights.mean()
    rmse = torch.sqrt((error ** 2 * lat_weights).mean([0, 1, 3, 4]))
    return rmse


def compute_weighted_acc(forecast, labels, lat_weights):
    mean = torch.mean(labels, dim=0)
    fa = forecast - mean
    a = labels - mean

    lat_weights /= lat_weights.mean()
    w = lat_weights

    fa_prime = fa - fa.mean()
    a_prime = a - a.mean()

    acc = (
            torch.sum(w * fa_prime * a_prime, dim=[0, 1, 3, 4]) /
            torch.sqrt(
                torch.sum(w * fa_prime ** 2, dim=[0, 1, 3, 4]) * torch.sum(w * a_prime ** 2, dim=[0, 1, 3, 4])
            )
    )
    return acc


def compute_weighted_mae(forecast, labels, lat_weights):
    error = forecast - labels
    lat_weights /= lat_weights.mean()
    mae = (torch.abs(error) * lat_weights).mean([0, 1, 3, 4])
    return mae
