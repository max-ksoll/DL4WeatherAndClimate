from datetime import datetime

import yaml

import wandb


def get_sweep():
    executionTime = datetime.now().strftime("%d/%m/%Y, %H:%M")
    name = str("DL4WeatherAndClimate " + executionTime)
    sweep_config = {"method": "grid", "name": name}
    # metric = {"name": "mse", "goal": "minimize"}
    # sweep_config["metric"] = metric
    return sweep_config


def get_parameters_dict():
    parameters_dict = {
        "batch_size": {"value": 1},
        "init_learning_rate": {"value": 2.5e-4},
        "model_parameter": {"value": {
            "channel": 1024,
            "transformer_blocks": 8,
            "heads": 8
        }},
        "autoregression_steps_epochs": {"value": [
            {'epochs': 1, 'steps': 1},
            {'epochs': 1, 'steps': 1, 'lr': 1e-4},
            {'epochs': 1, 'steps': 1, 'lr': 7.5e-5},

            {'epochs': 1, 'steps': 2},
            {'epochs': 1, 'steps': 2, 'lr': 1e-4},
            {'epochs': 1, 'steps': 2, 'lr': 7.5e-5},

            {'epochs': 1, 'steps': 4, 'lr': 5e-5},
            {'epochs': 1, 'steps': 8, 'lr': 1e-5},
            {'epochs': 1, 'steps': 12, 'lr': 1e-5},
            {'epochs': 1, 'steps': 20, 'lr': 1e-5},
        ]}
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
