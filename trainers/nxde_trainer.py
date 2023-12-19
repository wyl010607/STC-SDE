import os
import json

import numpy as np
from tqdm import tqdm
import torch
import math
from shutil import copyfile
from utils.train_tool import EarlyStop, time_decorator
from .base import Trainer
from torch.cuda.amp import autocast, GradScaler


class NXDETrainer(Trainer):
    """Neural XDE trainer"""

    def __init__(
        self,
        _,
        model,
        optimizer,
        lr_scheduler,
        max_epoch_num,
        max_iter_num,
        scaler,
        model_save_path,
        result_save_dir,
        early_stop,
        teacher_forcing=False,
        PV_index_list=None,
        OP_index_list=None,
        **kwargs,
    ):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.max_epoch_num = max_epoch_num
        self.epoch_now = 0
        self.scaler = scaler
        self.model_save_path = model_save_path
        self.model_save_dir = os.path.dirname(model_save_path)
        self.result_save_dir = result_save_dir
        if early_stop == -1:
            early_stop = None
        if early_stop is not None:
            self.early_stop = EarlyStop(5, min_is_best=True)
        else:
            self.early_stop = None
        self.teacher_forcing = teacher_forcing
        self.device = next(self.model.parameters()).device
        self.PV_index_list = PV_index_list
        self.OP_index_list = OP_index_list
        self.grad_scaler = GradScaler()
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.exists(self.result_save_dir):
            os.mkdir(self.result_save_dir)

    def model_loss_func(self, y_pred, y_true, **kargs):
        loss_func = kargs.get("loss_func", None)
        if loss_func is not None:
            if loss_func == "log_sum_exp_loss":
                loss = torch.log(torch.exp(y_pred - y_true).sum(dim=1) + 1e-8)
                return loss.mean()
            elif loss_func == "smooth_huber_loss":
                loss = torch.nn.SmoothL1Loss()(y_pred, y_true)
                return loss.mean()
            else:
                raise ValueError("loss_func {} is not supported".format(loss_func))
        else:
            return torch.nn.functional.mse_loss(y_pred, y_true)

    @time_decorator
    def train(
        self,
        train_data_loader,
        eval_data_loader,
        metrics=("mae", "rmse", "mape", "rmspe"),
        **kwargs,
    ):

        tmp_state_save_path = os.path.join(self.model_save_dir, "temp.pkl")
        epoch_result_list = []
        min_loss = torch.finfo(torch.float32).max
        print("Start training.")
        for epoch in range(1, self.max_epoch_num + 1):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            # train one epoch
            self.epoch_now = epoch
            train_loss = self.train_one_epoch(train_data_loader, **kwargs)
            # evaluate
            eval_loss, metrics_evals, *_ = self.evaluate(eval_data_loader, metrics)
            epoch_result_list.append(
                [epoch, train_loss, eval_loss, list(metrics_evals)]
            )

            # Criteria for early stopping
            if self.early_stop is not None:
                if self.early_stop.reach_stop_criteria(eval_loss):
                    break

            # save model state when meeting minimum loss
            # save to a temporary path first to avoid overwriting original state.
            if eval_loss < min_loss:
                torch.save(self.model.state_dict(), tmp_state_save_path)
                min_loss = eval_loss

            # learning rate decay
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()

            self.save_epoch_result(epoch_result_list)
        if self.early_stop is not None:
            self.early_stop.reset()
        copyfile(tmp_state_save_path, self.model_save_path)
        os.remove(tmp_state_save_path)
        epoch_result_json = self.save_epoch_result(epoch_result_list)
        return epoch_result_json

    def train_one_epoch(self, data_loader, **kwargs):
        """train one epoch"""
        self.model.train()
        total_loss = 0
        history_times = data_loader.dataset.history_times.to(self.device)
        forecast_times = data_loader.dataset.forecast_times.to(self.device)

        # 加个进度条
        tqmd_1 = tqdm(data_loader)
        for x, y in tqmd_1:
            coeffs, true_u = x
            coeffs = [coeff.clone().type(torch.float32).to(self.device) for coeff in coeffs]
            true_u = true_u.clone().type(torch.float32).to(self.device)
            # 转化为float16
            true_u = true_u.half()
            y = y.clone().type(torch.float32).to(self.device)
            self.optimizer.zero_grad()
            with autocast(enabled=True, dtype=torch.float16):
                prediction = self.model(history_times, forecast_times, coeffs, true_u)
                loss = self.model_loss_func(prediction, y, loss_func="smooth_huber_loss")
            self.grad_scaler.scale(loss).backward()
            self.grad_scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=5.0)
            self.grad_scaler.step(self.optimizer)
            self.grad_scaler.update()
            # 在tqmd中加入loss.item()，可以显示loss的值
            tqmd_1.set_description("loss is {:.4f}".format(loss.item()))
            total_loss += loss.item()
        return total_loss / len(data_loader)

    @torch.no_grad()
    def evaluate(self, data_loader, metrics, save_graph=False, **kwargs):
        history_times = data_loader.dataset.history_times.to(self.device)
        forecast_times = data_loader.dataset.forecast_times.to(self.device)
        self.model.eval()

        y_true, y_pred, tol_loss, data_num = [], [], 0, 0
        pred_step = data_loader.dataset.forecast_len
        for x, y in data_loader:
            coeffs, true_u = x
            coeffs = [coeff.clone().type(torch.float32).to(self.device) for coeff in coeffs]
            true_u = true_u.clone().type(torch.float32).to(self.device)
            y = y.clone().type(torch.float32).to(self.device)
            prediction = self.model(history_times, forecast_times, coeffs, true_u).detach()
            loss = self.model_loss_func(prediction, y).item()
            tol_loss += loss
            data_num += len(x)  # batch_size
            y_true.append(y)
            y_pred.append(prediction)
            # y_pred.shape = [len(data_loader) ,batch_size, time_step, feature_size]
            # to [batch_size * len(data_loader) * time_step, feature_size]
        y_true = self.scaler.inverse_transform(
            torch.cat(y_true, dim=0)
            .cpu()
            .numpy()
            .reshape(-1, len(self.PV_index_list)),
            index=self.PV_index_list,
        )
        y_pred = self.scaler.inverse_transform(
            torch.cat(y_pred, dim=0)
            .cpu()
            .numpy()
            .reshape(-1, len(self.PV_index_list)),
            index=self.PV_index_list,
        )
        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        # reshape y_pred to [batch_size * len(data_loader), time_step, feature_size]
        return (
            tol_loss / data_num,
            zip(metrics, eval_results),
            y_pred.reshape(-1, pred_step, len(self.PV_index_list)),
            y_true.reshape(-1, pred_step, len(self.PV_index_list)),
        )

    @torch.no_grad()
    def test(self, test_data_loader, metrics=("mae", "rmse", "mape", "rmspe"), **kwargs):
        self.model.load_state_dict(torch.load(self.model_save_path))
        eval_loss, metrics_evals, y_pred, y_true = self.evaluate(
            test_data_loader, metrics
        )

        # save y_pred, y_true to self.result_save_dir/y_pred.npy, y_true.npy
        np.save(os.path.join(self.result_save_dir, "test_y_pred.npy"), y_pred)
        np.save(os.path.join(self.result_save_dir, "test_y_true.npy"), y_true)
        test_result = {}
        for metric_name, metric_eval in metrics_evals:
            test_result[metric_name] = metric_eval
        with open(os.path.join(self.result_save_dir, "test_result.json"), "w") as f:
            json.dump(test_result, f)
        return test_result

    def save_epoch_result(self, epoch_result_list):
        # save loss and metrics for each epoch to self.result_save_dir/epoch_result.json
        epoch_result = {}
        for epoch, train_loss, eval_loss, metrics_evals in epoch_result_list:
            epoch_result[epoch] = {"train_loss": train_loss, "eval_loss": eval_loss}
            for metric_name, metric_eval in metrics_evals:
                epoch_result[epoch]["eval_" + metric_name] = metric_eval
        with open(os.path.join(self.result_save_dir, "epoch_result.json"), "w") as f:
            json.dump(epoch_result, f)
        return epoch_result

    def print_test_result(self, y_pred, y_true, metrics):
        for i in range(y_true.shape[1]):
            metric_results = self.get_eval_result(y_pred[:, i], y_true[:, i], metrics)
            print("Horizon {}".format(i + 1), end="  ")
            for j in range(len(metrics)):
                print("{}:  {:.4f}".format(metrics[j], metric_results[j]), end="  ")
            print()
