from datetime import datetime

import yaml

import wandb


def get_sweep():
    executionTime = datetime.now().strftime("%d/%m/%Y, %H:%M")
    name = str("CaseStudiesOfAIImplementationResults " + executionTime)
    sweep_config = {"method": "bayes", "name": name}
    metric = {"name": "mse", "goal": "minimize"}
    sweep_config["metric"] = metric
    return sweep_config


def get_parameters_dict():
    parameters_dict = {
        "epochs": {"values": [10, 15]},
        "batch_size": {"values": [1028, 256]},
        "learning_rate": {"values": [0.01, 0.1]},
    }
    return parameters_dict


def getSweepID():
    sweep_config = get_sweep()
    sweep_config["parameters"] = get_parameters_dict()
    sweep_config["name"] = sweep_config["name"]
    sweep_id = wandb.sweep(
        sweep=sweep_config,
        project="DL4WeatherAndClimate",
    )
    return sweep_id
