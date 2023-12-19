import numpy as np
import torch
from torch.utils.data import Dataset


class MultiLoopForecastDataset(Dataset):
    def __init__(
        self,
        data,
        PV_index_list,
        OP_index_list,
        DV_index_list,
        history_len,
        forecast_len,
        setpoint_index_list=None,
        setpoint_const_list=None,
        PV_diff_index_list=None,
        **kwargs
    ):
        self.PV_index_list = PV_index_list
        self.pv_data = data[:, self.PV_index_list]
        self.OP_index_list = OP_index_list
        self.op_data = data[:, self.OP_index_list]
        self.DV_index_list = DV_index_list
        self.dv_data = data[:, self.DV_index_list]
        if PV_diff_index_list:
            self.PV_diff_index_list = PV_diff_index_list
            self.pv_diff_data = data[:, self.PV_diff_index_list]
        else:
            self.pv_diff_data = None
        self.setpoint_index_list = setpoint_index_list
        if setpoint_index_list is not None:
            self.setpoint_data = data[:, self.setpoint_index_list]
        elif setpoint_const_list is not None:
            self.setpoint_data = np.ones_like(self.pv_data) * setpoint_const_list
        else:
            self.setpoint_data = None
        self.history_len = history_len
        self.forecast_len = forecast_len

    def __getitem__(self, index):
        # x is PV and SP, y is OP,return torch tensor
        x = np.concatenate(
            (
                self.pv_data[index : index + self.history_len],
                self.op_data[index : index + self.history_len],
                self.dv_data[index : index + self.history_len],
            ),
            axis=1,
        )
        if self.pv_diff_data is not None:
            x = np.concatenate(
                (x, self.pv_diff_data[index : index + self.history_len]), axis=1
            )
        if self.setpoint_data is not None:
            x = np.concatenate(
                (x, self.setpoint_data[index : index + self.history_len]), axis=1
            )
        y = np.concatenate(
            (
                self.pv_data[
                    index
                    + self.history_len : index
                    + self.history_len
                    + self.forecast_len
                ],
                self.op_data[
                    index
                    + self.history_len : index
                    + self.history_len
                    + self.forecast_len
                ],
                self.dv_data[
                    index
                    + self.history_len : index
                    + self.history_len
                    + self.forecast_len
                ],
            ),
            axis=1,
        )
        if self.pv_diff_data is not None:
            y = np.concatenate(
                (
                    y,
                    self.pv_diff_data[
                        index
                        + self.history_len : index
                        + self.history_len
                        + self.forecast_len
                    ],
                ),
                axis=1,
            )
        if self.setpoint_data is not None:
            y = np.concatenate(
                (
                    y,
                    self.setpoint_data[
                        index
                        + self.history_len : index
                        + self.history_len
                        + self.forecast_len
                    ],
                ),
                axis=1,
            )
        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )

    def __len__(self):
        return self.pv_data.shape[0] - self.history_len - self.forecast_len
