import torch
from torch import nn


class F_func_hz_CDE(torch.nn.Module):
    def __init__(self, dX_dt, func_f, func_g, node_num, hidden_channels):
        """Defines a controlled vector field.

        Arguments:
            dX_dt: As cdeint.
            func_f: As cdeint.
            func_g: As cdeint.
        """
        super(F_func_hz_CDE, self).__init__()
        if not isinstance(func_f, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")
        if not isinstance(func_g, torch.nn.Module):
            raise ValueError("func must be a torch.nn.Module.")

        self.dX_dt = dX_dt
        self.step_dX_dt = None
        self.func_f = func_f
        self.func_g = func_g
        self.node_num = node_num
        self.hidden_channels = hidden_channels
        self.mode = "history"

    def set_dx_dt(self, dX_dt):
        self.dX_dt = dX_dt
        return self

    def set_step_dx_dt(self, step_dX_dt):
        self.step_dX_dt = step_dX_dt
        return self

    def forecast_mode(self):
        self.mode = "forecast"
        return self

    def history_mode(self):
        self.mode = "history"
        return self

    def forward(self, t, hz):
        if self.mode == "history":
            # control_gradient is of shape (..., input_dim)
            control_gradient = self.dX_dt(t)
        else:  # self.mode == "forecast"
            if self.step_dX_dt is None:
                raise ValueError("step_dx_dt must be given in forecast mode")
            control_gradient = self.step_dX_dt
        # 还原(batch, node，hidden，2)
        hz = hz.reshape(hz.size(0), self.node_num, self.hidden_channels, 2)
        # vector_field is of shape (..., hidden_channels, input_dim)
        h = hz[
            ..., 0
        ]  # h: torch.Size([64, 207, 32]) # hz:[dh, out] torch.Size([batch, node, hidden，2])
        z = hz[..., 1]  # z: torch.Size([64, 207, 32])
        vector_field_f = self.func_f(h)  # vector_field_f: torch.Size([64, 207, 32, 2])
        vector_field_g = self.func_g(
            z
        )  # vector_field_g: torch.Size([64, 207, 32, 2])# vector_field_g: torch.Size([64, 207, 32, 1, 2])
        # vector_field_fg = torch.mul(vector_field_g, vector_field_f) # vector_field_fg: torch.Size([64, 207, 32, 2])
        vector_field_fg = torch.matmul(
            vector_field_g, vector_field_f
        )  # batch, node, hidden, 2
        # out is of shape (..., hidden_channels)
        # (The squeezing is necessary to make the matrix-multiply properly batch in all cases)
        dh = (vector_field_f @ control_gradient.unsqueeze(-1)).squeeze(
            -1
        )  # batch, node, hidden
        out = (vector_field_fg @ control_gradient.unsqueeze(-1)).squeeze(
            -1
        )  # batch, node, hidden
        hz = torch.stack([dh, out], dim=-1).reshape(
            hz.size(0), self.node_num * self.hidden_channels * 2
        )  # batch, node* hidden* 2
        return hz


###########################################################################################
# ----------Below is the Method Implementation of the Ablation Experiment---------------- #
###########################################################################################
class F_func_hz_RNN(nn.Module):
    def __init__(self, func_f, func_g, node_num, hidden_channels):
        super(F_func_hz_RNN, self).__init__()
        # Define an RNN layer
        self.node_num = node_num
        self.hidden_channels = hidden_channels
        self.input_dim = node_num * hidden_channels * 5
        self.hidden_dim = node_num * hidden_channels * 2
        self.x_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.hidden_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self.rnn = nn.RNN(self.input_dim, self.hidden_dim, batch_first=True)
        # Store the func_f and func_g functions
        self.func_f = func_f
        self.func_g = func_g

    def forward(self, spline, history_times, init0):
        hz = init0
        for i in history_times:
            hz = hz.reshape(hz.size(0), self.node_num, self.hidden_channels, 2)
            h = hz[..., 0]
            z = hz[..., 1]
            vector_field_f = self.func_f(h)
            vector_field_g = self.func_g(z)
            vector_field_fg = torch.matmul(
                vector_field_g, vector_field_f
            )  # batch, node, hidden, 2
            hz = torch.stack([vector_field_f, vector_field_fg], dim=-1).reshape(
                hz.size(0), self.node_num * self.hidden_channels * 4
            )
            x = self.x_conv(
                spline.evaluate(i)[..., 1:].unsqueeze(1).permute(0, 1, 3, 2)
            ).reshape(hz.size(0), -1)
            input = torch.cat((x, hz), dim=1)
            hz, _ = self.rnn(input)
        return hz
