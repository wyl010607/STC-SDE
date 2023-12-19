import torch
import json
import os
import sys
def load_config(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        config = json.load(f)
        config = config[0]
        args = config['args']
        train_config = config['train_config']
        model_config = config['net_config']
    return args, train_config, model_config

def load_model(model_name, model_config_path, model_state_path, device):
    args, train_config, model_config = load_config(model_config_path)
    if model_name != args["model_name"]:
        raise ValueError("model_name is not equal to model_name in config")
    Model = getattr(sys.modules["models"], model_name, None)
    model = Model(**model_config)
    model.load_state_dict(torch.load(model_state_path))
    model.to(device)
    return model

