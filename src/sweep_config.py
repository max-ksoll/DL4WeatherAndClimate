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
        "epochs": {"value": 50},
        "batch_size": {"value": 1},
        "learning_rate": {"value": 2.5e-5},
        "model_parameter": {"value": {
            "channel": 1024,
            "transformer_blocks": 16,
            "heads": 8
        }},
        "autoregression_steps_epochs": {"value": [
            {'epochs': 10, 'steps': 1},
            {'epochs': 10, 'steps': 2},
            {'epochs': 10, 'steps': 3},
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
