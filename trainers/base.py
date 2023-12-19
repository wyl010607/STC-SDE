import os
import sys
import json
from random import random

import numpy as np
from tqdm import tqdm
import torch
import math
from shutil import copyfile
from utils.train_tool import EarlyStop, time_decorator
from utils.metrics import get_mae, get_mape, get_rmse, get_rmspe


class Trainer:
    @staticmethod
    def get_eval_result(y_pred, y_true, metrics=("mae", "rmse", "mape", "rmspe")):
        module = sys.modules[__name__]

        eval_results = []
        for metric_name in metrics:
            eval_func = getattr(module, "get_{}".format(metric_name))
            eval_results.append(eval_func(y_pred, y_true))

        return eval_results

    def train(self, *args, **kwargs):
        raise NotImplementedError

    def train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError

    def evaluate(self, *args, **kwargs):
        raise NotImplementedError

    def test(self, *args, **kwargs):
        raise NotImplementedError

    def model_loss_func(self, y_pred, y_true, *args):
        return 0


class MSTrainer(Trainer):
    """muti-step trainer"""

    def __init__(
        self,
        model,
        optimizer,
        lr_scheduler,
        max_epoch_num,
        scaler,
        model_save_path,
        result_save_dir,
        early_stop,
        teacher_forcing=False,
        PV_index=None,
        OP_index=None,
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
        self.PV_index = PV_index
        self.OP_index = OP_index
        if not os.path.exists(self.model_save_dir):
            os.mkdir(self.model_save_dir)
        if not os.path.exists(self.result_save_dir):
            os.mkdir(self.result_save_dir)

    def model_loss_func(self, y_pred, y_true, *args):
        return torch.nn.functional.mse_loss(y_pred, y_true)

    @time_decorator
    def train(
        self, train_data_loader, eval_data_loader, metrics=("mae", "rmse", "mape", "rmspe"), **kwargs
    ):

        tmp_state_save_path = os.path.join(self.model_save_dir, "temp.pkl")
        epoch_result_list = []
        min_loss = torch.finfo(torch.float32).max
        print("Start training.")
        for epoch in range(1, self.max_epoch_num + 1):
            print(f"Epoch {epoch} / {self.max_epoch_num}")
            # train one epoch
            self.epoch_now = epoch
            self.train_one_epoch(train_data_loader, **kwargs)

            # evaluate
            eval_loss, metrics_evals, _, _ = self.evaluate(eval_data_loader, metrics)
            epoch_result_list.append([epoch, eval_loss, list(metrics_evals)])

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

    @torch.no_grad()
    def evaluate(self, data_loader, metrics, save_graph=False, **kwargs):
        self.model.eval()

        y_true, y_pred, tol_loss, data_num = [], [], 0, 0
        pred_step = data_loader.dataset.forecast_len
        for x, y in data_loader:
            x = x.type(torch.float32).to(self.device)
            y = y.type(torch.float32).to(self.device)
            muti_step_pred = torch.zeros_like(
                y if self.PV_index is None else y[:, :, self.PV_index]
            )
            sample_x = x
            for i in range(
                y.shape[1]
            ):
                prediction = self.model(
                    sample_x, **kwargs
                ).detach()  # the prediction of the next 1 step shape:[batch_size,1,feature_size]
                muti_step_pred[:, i : i + 1, :] = prediction[:, :, self.PV_index]
                sample_x = torch.cat((sample_x[:, 1:, :], prediction), dim=1)
                if self.OP_index is not None:
                    sample_x[:, -1:, self.OP_index] = y[:, i : i + 1, self.OP_index]

            if self.PV_index is not None:
                loss = self.model_loss_func(
                    muti_step_pred, y[:, :, self.PV_index]
                ).item()
            else:
                loss = self.model_loss_func(muti_step_pred, y).item()
            tol_loss += loss
            data_num += len(x)  # batch_size
            y_true.append(y[:, :, self.PV_index])
            y_pred.append(muti_step_pred)

        # y_pred.shape = [len(data_loader) ,batch_size, time_step, feature_size]
        # to [batch_size * len(data_loader) * time_step, feature_size]
        y_true = self.scaler.inverse_transform(
            torch.cat(y_true, dim=0).cpu().numpy().reshape(-1, len(self.PV_index)),
            index=self.PV_index,
        )
        y_pred = self.scaler.inverse_transform(
            torch.cat(y_pred, dim=0).cpu().numpy().reshape(-1, len(self.PV_index)),
            index=self.PV_index,
        )
        eval_results = self.get_eval_result(y_pred, y_true, metrics)
        for metric_name, eval_ret in zip(metrics, eval_results):
            print("{}:  {:.4f}".format(metric_name.upper(), eval_ret), end="  ")
        print()
        # reshape y_pred to [batch_size * len(data_loader), time_step, feature_size]
        return (
            tol_loss / data_num,
            zip(metrics, eval_results),
            y_pred.reshape(-1, pred_step, len(self.PV_index)),
            y_true.reshape(-1, pred_step, len(self.PV_index)),
        )

    @torch.no_grad()
    def test(self, test_data_loader, metrics=("mae", "rmse", "mape", "rmspe"), **kwargs):
        self.model.load_state_dict(torch.load(self.model_save_path))
        eval_loss, metrics_evals, y_pred, y_true = self.evaluate(test_data_loader, metrics)

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
        for epoch, loss, metrics_evals in epoch_result_list:
            epoch_result[epoch] = {"loss": loss}
            for metric_name, metric_eval in metrics_evals:
                epoch_result[epoch][metric_name] = metric_eval
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


class Sampling_MSTrainer(MSTrainer):
    """muti-step trainer with sampling strategy"""

    def __init__(
        self,
        _,  # 特例，不用adj——mx
        model,
        optimizer,
        lr_scheduler,
        max_epoch_num,
        num_iter,  #
        scaler,
        model_save_path,
        result_save_dir,
        early_stop,
        teacher_forcing=False,
        PV_index_list=None,
        OP_index_list=None,
        *args,
        **kwargs,
    ):
        super(Sampling_MSTrainer, self).__init__(
            model,
            optimizer,
            lr_scheduler,
            max_epoch_num,
            scaler,
            model_save_path,
            result_save_dir,
            early_stop,
            teacher_forcing,
            PV_index_list,
            OP_index_list,
        )

    def use_true_val(self, p):
        return True if random() > p else False

    def train_one_epoch(self, data_loader, **kwargs):
        """train one epoch, muti_step prediction with sampling strategy"""
        self.model.train()

        sample_true_v = self.epoch_now / (
            self.epoch_now + math.exp(self.max_epoch_num / (self.epoch_now + 1))
        )
        for x, y in tqdm(data_loader):
            x = x.type(torch.float32).to(self.device)
            y = y.type(torch.float32).to(self.device)
            sample_x = x
            muti_step_pred = torch.zeros_like(
                y if self.PV_index is None else y[:, :, self.PV_index]
            )
            for i in range(
                y.shape[1]
            ):
                prediction = self.model(
                    sample_x, **kwargs
                )  # the prediction of the next 1 step shape:[batch_size,1,feature_size]
                muti_step_pred[:, i : i + 1, :] = prediction[:, :, self.PV_index]
                if self.teacher_forcing and self.use_true_val(sample_true_v):
                    sample_x = torch.cat(
                        (sample_x[:, 1:, :], prediction) , dim=1
                    )
                    sample_x[:, -1:, self.PV_index + self.OP_index] = y[:, i : i + 1, self.PV_index + self.OP_index]
                else:
                    sample_x = torch.cat(
                        (sample_x[:, 1:, :], prediction) , dim=1
                    )
                    if self.OP_index is not None:
                        sample_x[:, -1:, self.OP_index] = y[:, i : i + 1, self.OP_index]
            if self.PV_index is not None:
                loss = self.model_loss_func(muti_step_pred, y[:, :, self.PV_index])
            else:
                loss = self.model_loss_func(muti_step_pred, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

