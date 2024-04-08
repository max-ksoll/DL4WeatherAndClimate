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
        "learning_rate": {"value": 2.5e-5},
        "model_parameter": {"value": {
            "channel": 32,
            "transformer_blocks": 1,
            "heads": 1
        }},
        "autoregression_steps_epochs": {"value": [
            {'epochs': 10, 'steps': 1},
            {'epochs': 10, 'steps': 2},
            {'epochs': 10, 'steps': 3},
            {'epochs': 10, 'steps': 4},
            {'epochs': 10, 'steps': 5},
            {'epochs': 10, 'steps': 6},
            {'epochs': 10, 'steps': 7},
            {'epochs': 10, 'steps': 8},
            {'epochs': 10, 'steps': 9},
            {'epochs': 10, 'steps': 10},
            {'epochs': 10, 'steps': 11},
            {'epochs': 10, 'steps': 12},
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
