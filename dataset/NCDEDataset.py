import numpy as np
import torch
from torch.utils.data import Dataset
import utils.controldiffeq as cde


class NCDEDataset(Dataset):
    def __init__(
        self,
        data,
        PV_index_list,
        OP_index_list,
        DV_index_list,
        history_len,
        forecast_len,
        **kwargs
    ):
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=-1)
        self.PV_index_list = PV_index_list
        self.OP_index_list = OP_index_list
        self.DV_index_list = DV_index_list
        self.combined_data = np.concatenate(
            (
                data[:, PV_index_list, :],
                data[:, OP_index_list, :],
                data[:, DV_index_list, :],
            ),
            axis=1,
        )
        self.history_len = history_len
        self.history_times = torch.linspace(0, history_len - 1, history_len)
        self.forecast_len = forecast_len
        self.forecast_times = torch.linspace(
            history_len, history_len + forecast_len - 1, forecast_len
        )
        X, self.Y, self.U = self._add_window_horizon()
        self.x_coeffs = self._natural_cubic_spline_coeffs(X)

    def _add_window_horizon(self):
        length = len(self.combined_data)
        end_index = length - self.forecast_len - self.history_len + 1
        X = []  # history
        Y = []  # forecast
        U = []  # true_u
        index = 0
        while index < end_index:
            X.append(self.combined_data[index : index + self.history_len])
            Y.append(
                self.combined_data[
                    index
                    + self.history_len : index
                    + self.history_len
                    + self.forecast_len,
                    self.PV_index_list,
                    :,
                ]
            )
            U.append(
                self.combined_data[
                    index
                    + self.history_len : index
                    + self.history_len
                    + self.forecast_len,
                    self.OP_index_list,
                    :,
                ]
            )
            index = index + 1
        X = np.array(X)
        Y = np.array(Y)
        U = np.array(U)
        return X, Y, U

    def _natural_cubic_spline_coeffs(self, X):
        # for NCDE Model, need to interpolate the data to third order:
        augmented_X = []
        augmented_X.append(
            self.history_times.unsqueeze(0)
            .unsqueeze(0)
            .repeat(X.shape[0], X.shape[2], 1)
            .unsqueeze(-1)
            .transpose(1, 2)
        )
        augmented_X.append(torch.Tensor(X[..., :]))
        X_tra = torch.cat(augmented_X, dim=3)
        x_coeffs = cde.natural_cubic_spline_coeffs(
            self.history_times, X_tra.transpose(1, 2)
        )
        return x_coeffs

    def __getitem__(self, index):
        # x is PV and SP, y is OP
        x = [x_[index] for x_ in self.x_coeffs]
        u = self.U[index]
        y = self.Y[index]

        # Convert each x_ to torch.tensor
        x_tensors = []
        for x_ in x:
            if isinstance(x_, torch.Tensor):
                x_tensors.append(x_.clone().detach())
            else:
                x_tensors.append(torch.tensor(x_, dtype=torch.float32))

        u_tensor = (
            torch.tensor(u, dtype=torch.float32).clone().detach()
            if isinstance(u, torch.Tensor)
            else torch.tensor(u, dtype=torch.float32)
        )

        y_tensor = (
            torch.tensor(y, dtype=torch.float32).clone().detach()
            if isinstance(y, torch.Tensor)
            else torch.tensor(y, dtype=torch.float32)
        )

        return (x_tensors, u_tensor), y_tensor

    def __len__(self):
        return self.combined_data.shape[0] - self.history_len - self.forecast_len + 1
