import json
import os
import sys
import random
import yaml
import numpy as np
import torch
import argparse
import trainers
import models

from utils import scaler
from torch.utils.data import DataLoader
from dataset.MultiLoopForecastDataset import MultiLoopForecastDataset
from utils.adj_matrix import get_Adj_matrix


def load_config(data_path):
    with open(data_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config

def main(args):
    # Load model and training configurations
    model_config = load_config(args.model_config_path)
    train_config = load_config(args.train_config_path)

    # Set random seeds for reproducibility
    torch.manual_seed(train_config["seed"])
    torch.cuda.manual_seed(train_config["seed"])
    np.random.seed(train_config["seed"])
    random.seed(train_config["seed"])

    # Training parameters
    Scaler = getattr(sys.modules["utils.scaler"], train_config["scaler"])
    device = torch.device(train_config["device"])
    train_ratio = train_config["train_ratio"]
    valid_ratio = train_config["valid_ratio"]
    batch_size = train_config["batch_size"]

    # ----------------------- Load data ------------------------
    # Extract dataset parameters from model configuration
    data_config = model_config["dataset"]
    dataset_type = data_config.get("dataset_type", "MultiLoopForecastDataset")
    data_path = data_config["data_path"]
    process_vars_list = data_config["process_vars_list"]
    control_vars_list = data_config["control_vars_list"]
    disturb_vars_list = data_config["disturb_vars_list"]
    history_len = data_config["history_len"]
    forecast_len = data_config["forecast_len"]
    with_diff = data_config.get("with_diff", False)

    # Load the dataset
    data = np.load(data_path, allow_pickle=True)
    data_array, time_stamp_array, vars_index_dict = (
        data['data_array'], data['time_stamp_array'], data['vars_index_dict'].tolist())

    # Split the data into training, validation, and test sets
    train_data = data_array[: int(len(data_array) * train_ratio)]
    valid_data = data_array[int(len(data_array) * train_ratio) : int(len(data_array) * (train_ratio + valid_ratio))]
    test_data = data_array[int(len(data_array) * (train_ratio + valid_ratio)) :]

    # Initialize and fit the data scaler
    data_scaler = Scaler(axis=0)
    data_scaler.fit(train_data)
    train_data = data_scaler.transform(train_data)
    valid_data = data_scaler.transform(valid_data)
    test_data = data_scaler.transform(test_data)

    # Create index lists for different types of variables
    PV_index_list = [vars_index_dict[var] for var in process_vars_list]
    OP_index_list = [vars_index_dict[var] for var in control_vars_list]
    DV_index_list = [vars_index_dict[var] for var in disturb_vars_list]

    # Update data configuration with additional parameters
    data_config.update({
        "PV_index_list": PV_index_list,
        "OP_index_list": OP_index_list,
        "DV_index_list": DV_index_list,
        "PV_diff_index_list": None, # unused currently
        "setpoint_const_list": None,  # unused currently
        "input_size": len(PV_index_list) * 2 + len(OP_index_list) + len(DV_index_list) if with_diff else len(PV_index_list) + len(OP_index_list) + len(DV_index_list),
        "output_size": len(PV_index_list),
    })

    # Load the dataset class
    Dataset = getattr(sys.modules["dataset"], dataset_type, MultiLoopForecastDataset)

    # Create datasets dataloaders for training, validation, and testing
    train_dataset = Dataset(train_data, PV_index_list, OP_index_list, DV_index_list, history_len, forecast_len)
    valid_dataset = Dataset(valid_data, PV_index_list, OP_index_list, DV_index_list, history_len, forecast_len)
    test_dataset = Dataset(test_data, PV_index_list, OP_index_list, DV_index_list, history_len, forecast_len)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # --------------------- Trainer setting --------------------

    # Determine the model to use based on the model name argument
    net_name = args.model_name
    net_config = model_config.get(net_name)

    if net_name == "STGODE":
        # Special processing for STGODE model
        dwt_path = data_config["dwt_path"]
        get_Adj_matrix(train_data, dwt_path, 0.1, 0.6)
    elif net_config is None:
        raise ValueError(f"Model {net_name} configuration is not defined in model_config.")

    # Update the model configuration with data configuration
    net_config.update(data_config)

    # Get the model class from the models module
    Model = getattr(sys.modules["models"], net_name, None)
    if Model is None:
        raise ValueError(f"Model {net_name} is not defined.")
    print(Model)
    # Initialize the model and move it to device
    net_pred = Model(**net_config).to(device)

    # Initialize the optimizer for the model
    Optimizer = getattr(sys.modules["torch.optim"], train_config["optimizer"])
    optimizer_pred = Optimizer(
        net_pred.parameters(),
        lr=train_config["learning_rate"],
        weight_decay=train_config["weight_decay"],
    )

    # Initialize the scheduler for the optimizer, if specified
    scheduler_config = train_config.get("lr_scheduler", None)
    if scheduler_config:
        Scheduler = getattr(sys.modules["torch.optim.lr_scheduler"], scheduler_config.pop("name"))
        scheduler_pred = Scheduler(optimizer_pred, **scheduler_config)
    else:
        scheduler_pred = None

    # Extract the early stopping configuration, if specified
    early_stop = train_config.get("early_stop", None)

    # --------------------------- Train -------------------------
    # Consolidate configurations into a single dictionary
    config = {
        "args": vars(args),  # Convert args to a dictionary
        "train_config": train_config,
        "net_config": net_config,
    }
    print("Configuration: ", config)
    # Create results directory if it doesn't exist
    if not os.path.exists(args.result_save_dir):
        os.makedirs(args.result_save_dir)
    # Save configuration to a JSON file
    with open(os.path.join(args.result_save_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=4)  # Using indent for better readability of JSON file
    # Get adjacency matrix path from network configuration, if available
    adj_mx_path = net_config.get("adj_mx_path", None)
    # Initialize the training object
    trainer_class = getattr(sys.modules["trainers"], train_config["trainer"])
    net_pred_trainer = trainer_class(
        adj_mx_path,
        net_pred,
        optimizer_pred,
        scheduler_pred,
        args.num_epoch,
        data_scaler,
        args.model_save_path,
        args.result_save_dir,
        train_config['early_stop'],
        **net_config,
    )
    epoch_results = net_pred_trainer.train(train_dataloader, valid_dataloader, **net_config)
    test_result = net_pred_trainer.test(test_dataloader)
    # Save the combined results to a JSON file
    result = {
        "config": config,
        "epoch_results": epoch_results,
        "test_result": test_result,
    }
    with open(os.path.join(args.result_save_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_config_path",
        type=str,
        default="./config/DIST_NDE.yaml",
        help="Config path of models",
    )
    parser.add_argument(
        "--train_config_path",
        type=str,
        default="./config/STCSDE_train_config.yaml",
        help="Config path of Trainer",
    )
    parser.add_argument(
        "--model_name", type=str, default="STCSDE", help="Model name"
    )
    parser.add_argument(
        "--num_epoch", type=int, default=80, help="Epoch number"
    )
    parser.add_argument(
        "--model_save_path",
        type=str,
        default="./model_states/test.pkl",
        help="Model save path",
    )
    parser.add_argument(
        "--result_save_dir",
        type=str,
        default="./result/test",
        help="Result save path",
    )
    args = parser.parse_args()
    main(args)
