import torch


def compute_weighted_rmse(forecast, label, lat_weights):
    error = forecast - label
    rmse = torch.sqrt((error ** 2 * lat_weights).mean([0, 1, 3, 4]))
    return rmse


def compute_weighted_acc(forecast, labels, lat_weights):
    mean = torch.mean(labels, dim=0)
    fa = forecast - mean
    a = labels - mean
    w = lat_weights

    # fa_prime = fa - fa.mean()
    # a_prime = a - a.mean()
    print(torch.sum(labels, dim=[0, 1, 3, 4]))
    print(torch.sum(a, dim=[0, 1, 3, 4]))
    acc = (
            torch.sum(w * fa * a, dim=[0, 1, 3, 4]) /
            torch.sqrt(
                torch.sum(w * fa ** 2, dim=[0, 1, 3, 4]) * torch.sum(w * a ** 2, dim=[0, 1, 3, 4])
            )
    )
    return acc


def compute_weighted_mae(forecast, labels, lat_weights):
    error = forecast - labels
    mae = (torch.abs(error) * lat_weights).mean([0, 1, 3, 4])
    return mae
