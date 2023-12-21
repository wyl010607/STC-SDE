import torch
import torchdiffeq
import torchsde
from torch import nn

from .Abstract_SDE import Abstract_SDE
from .step_awared_GLUs import step_awared_GLUs
from .base_nn import G_func_linear


class CSDE_decoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        hidden_hidden_dim,
        node_num,
        output_size,
        forecast_len,
        device,
        adjoint=True,
        solver="euler",
    ):
        super(CSDE_decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.node_num = node_num
        self.output_size = output_size
        self.forecast_len = forecast_len
        self.device = device
        self.adjoint = adjoint
        self.solver = solver
        self.decoder_F_func = step_awared_GLUs(
            num_node=node_num,
            hidden_dim=hidden_dim,
            forecast_steps=forecast_len,
            device=device,
        )
        self.decoder_G_func = G_func_linear(
            input_dim=node_num,
            hidden_channels=node_num,
            hidden_hidden_channels=hidden_hidden_dim,
            num_hidden_layers=2,
        )
        self.decoder_sde_model = Abstract_SDE(
            self.decoder_F_func,
            self.decoder_G_func,
            noise_type="diagonal",
            sde_type="ito",
        )
        self.sdeint = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

    def step_forward(self, times, last_y_pred, z):
        self.decoder_F_func.set_z(z)
        dt = times[1] - times[0]
        sde_out = self.sdeint(
            self.decoder_sde_model,
            last_y_pred.squeeze(),
            times,
            dt=dt,
            method=self.solver,
            names={"drift": "F_func", "diffusion": "G_func"},
        )  # history_len, batch, node*hidden*
        out = sde_out[-1, ...].reshape(sde_out.shape[1], self.node_num)
        return out


class CODE_decoder(nn.Module):
    def __init__(
        self,
        hidden_dim,
        hidden_hidden_dim,
        node_num,
        output_size,
        forecast_len,
        device,
        adjoint=True,
        solver="euler",
    ):
        super(CODE_decoder, self).__init__()
        self.hidden_dim = hidden_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.node_num = node_num
        self.output_size = output_size
        self.forecast_len = forecast_len
        self.device = device
        self.adjoint = adjoint
        self.solver = solver
        self.decoder_F_func = step_awared_GLUs(
            num_node=node_num,
            hidden_dim=hidden_dim,
            forecast_steps=forecast_len,
            device=device,
        )

    def step_forward(self, times, last_y_pred, z):
        self.decoder_F_func.set_z(z)
        odeint = torchdiffeq.odeint_adjoint if self.adjoint else torchdiffeq.odeint
        ode_out = odeint(
            self.decoder_F_func, last_y_pred.squeeze(), times, method="euler"
        )
        out = ode_out[-1, ...].reshape(ode_out.shape[1], self.node_num)
        return out


###########################################################################################
# ----------Below is the Method Implementation of the Ablation Experiment---------------- #
###########################################################################################
class Out_func_conv(nn.Module):
    def __init__(
        self,
        hidden_dim,
        output_dim,
        output_size,
        num_node,
        u_node,
        forecast_len,
        device,
    ):
        super(Out_func_conv, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_size = output_size
        self.num_node = num_node
        self.forecast_len = forecast_len
        self.device = device

        self.end_conv = nn.Conv2d(
            1,
            self.forecast_len * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        ).to(self.device)

        self.end_fc = nn.Linear(
            in_features=num_node + u_node, out_features=output_size
        ).to(self.device)

        self.true_u_encoder = nn.Conv2d(
            in_channels=1,
            out_channels=self.output_dim * self.hidden_dim,
            kernel_size=(1, self.forecast_len),
            stride=(1, 1),
            padding=(0, 0),
            bias=False,
        ).to(
            self.device
        )  # in b, 1, n, forecast_len, out b, output_dim * h, n, hidden_dim

    def forward(self, sde_out, true_u, **kwargs):
        sde_out = sde_out.reshape(
            sde_out.shape[0], sde_out.shape[1], self.num_node, self.hidden_dim, 2
        )[..., -1]
        # out: batch, node*hidden*2
        z = sde_out[-1:, ...].transpose(0, 1)
        # z# batch, 1, node, output_dim * hidden
        hidden_u = self.true_u_encoder(
            true_u.permute(0, 3, 2, 1)
        )  # b, output_dim * hidden_dim, u_n, 1
        hidden_u = hidden_u.permute(0, 3, 2, 1)  # b, 1, u_n, output_dim * hidden_dim
        # concat z and hidden_u in dim 2 (node), -> b, 1, n + u_n, output_dim * hidden_dim
        z = torch.cat([z, hidden_u], dim=2)
        # conv -> b, forecast_len * output_dim, n, 1
        z = self.end_conv(z)
        z = self.end_fc(z.squeeze(-1))
        # reshape -> b, forecast_len, output_dim, n
        z = z.reshape(-1, self.forecast_len, self.output_dim, self.output_size)
        output = z.permute(0, 1, 3, 2)  # B, T, N, C
        return output


class Out_func_csde(nn.Module):
    def __init__(
        self,
        sde_model,
        sdeint,
        dt,
        solver,
        decoder_mean,
        decoder_log_var,
        input_dim,
        hidden_dim,
        output_dim,
        output_size,
        num_node,
        PV_index_list,
        OP_index_list,
        forecast_len,
        device,
    ):
        super(Out_func_csde, self).__init__()
        self.sde_model = sde_model
        self.sdeint = sdeint
        self.dt = dt
        self.solver = solver
        self.input_dim = input_dim
        self.PV_index_list = torch.tensor(PV_index_list, dtype=torch.long)
        self.OP_index_list = torch.tensor(OP_index_list, dtype=torch.long)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_size = output_size
        self.num_node = num_node
        self.forecast_len = forecast_len
        self.device = device
        self.X0 = None
        self.mean_init_conv = nn.Conv2d(
            1,
            1 * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        ).to(self.device)
        self.log_var_init_conv = nn.Conv2d(
            1,
            1 * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        ).to(self.device)

        self.decoder_mean = decoder_mean
        self.decoder_log_var = decoder_log_var

    def set_X0(self, X0):
        self.X0 = X0  # batch, node, 2
        return self

    def forward(self, sde_out, true_u, history_times, forecast_times, **kwargs):
        output_mean_list = []
        output_log_var_list = []
        for num, time in enumerate(forecast_times):
            if num == 0:
                hidden = sde_out[-1, ...]
                times = torch.cat([history_times[-1:], forecast_times[:1]], dim=0)
                z = sde_out.reshape(
                    sde_out.shape[0],
                    sde_out.shape[1],
                    self.num_node,
                    self.hidden_dim,
                    2,
                )[-1:, ..., -1].transpose(
                    0, 1
                )  # batch, 1, node, hidden
                y_pred_mean = (
                    self.mean_init_conv(z).permute(0, 2, 1, 3).contiguous()
                )  # batch, 1*output_dim, node, 1 -> batch, node, 1, output_dim
                y_pred_log_var = (
                    self.log_var_init_conv(z).permute(0, 2, 1, 3).contiguous()
                )  # batch, 1*output_dim, node, 1 -> batch, node, 1, output_dim
                y_pred_mean = torch.clamp(y_pred_mean, -10, 10)
                y_pred_log_var = torch.clamp(y_pred_log_var, -10, 10)
                # set true_u
                y_pred_mean[:, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(
                    -1
                )
                dx_dt = y_pred_mean - self.X0[..., -1:].unsqueeze(
                    -1
                )  # batch, node,  output_dim，1
                dx_dt = torch.cat([torch.ones_like(dx_dt), dx_dt], dim=-2).squeeze(-1)

            # build dx_dt
            # 构建一个batch，dx_dt的tensor  dx_dt: [y_pred, u_ture, ]
            self.sde_model.F_func.set_step_dx_dt(dx_dt)
            sde_out = self.sdeint(
                self.sde_model,
                hidden,
                times,
                dt=self.dt,
                method=self.solver,
                names={"drift": "F_func", "diffusion": "G_func"},
            )  # history_len, batch, node*hidden*2

            hidden = sde_out[-1, ...]
            z = sde_out.reshape(
                sde_out.shape[0],
                sde_out.shape[1],
                self.num_node,
                self.hidden_dim,
                2,
            )[-1:, ..., -1].transpose(
                0, 1
            )  # batch, 1, node, hidden
            last_y_pred_mean = y_pred_mean
            # y_pred = self.final_conv(z).permute(
            #    0, 2, 1, 3
            # )  # batch, 1*output_dim, node, 1 -> batch, node, 1, output_dim
            y_pred_mean = (
                self.decoder_mean.step_forward(times, y_pred_mean, z.squeeze(1))
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            y_pred_log_var = (
                self.decoder_log_var.step_forward(times, y_pred_log_var, z.squeeze(1))
                .unsqueeze(-1)
                .unsqueeze(-1)
            )

            y_pred_mean[:, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(-1)
            dx_dt = y_pred_mean - last_y_pred_mean  # batch, node,  output_dim，1
            dx_dt = torch.cat([torch.ones_like(dx_dt), dx_dt], dim=-2).squeeze(-1)
            output_mean_list.append(y_pred_mean)
            output_log_var_list.append(y_pred_log_var)
            times = forecast_times[num : num + 2]
        output_mean = torch.cat(
            output_mean_list, dim=2
        )  # batch, forecast_len, node, output_dim
        output_log_var = torch.cat(
            output_log_var_list, dim=2
        )  # batch, forecast_len, node, output_dim
        output_mean = output_mean.permute(0, 2, 1, 3)[:, :, self.PV_index_list, :]
        output_log_var = output_log_var.permute(0, 2, 1, 3)[:, :, self.PV_index_list, :]

        return [output_mean, output_log_var]


class Out_func_code(nn.Module):
    def __init__(
        self,
        ode_model,
        odeint,
        solver,
        decoder_mean,
        decoder_log_var,
        input_dim,
        hidden_dim,
        output_dim,
        output_size,
        num_node,
        PV_index_list,
        OP_index_list,
        forecast_len,
        device,
    ):
        super(Out_func_code, self).__init__()
        self.ode_model = ode_model
        self.odeint = odeint
        self.solver = solver
        self.input_dim = input_dim
        self.PV_index_list = torch.tensor(PV_index_list, dtype=torch.long)
        self.OP_index_list = torch.tensor(OP_index_list, dtype=torch.long)
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_size = output_size
        self.num_node = num_node
        self.forecast_len = forecast_len
        self.device = device
        self.X0 = None
        self.mean_init_conv = nn.Conv2d(
            1,
            1 * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        ).to(self.device)
        self.log_var_init_conv = nn.Conv2d(
            1,
            1 * self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        ).to(self.device)

        self.decoder_mean = decoder_mean
        self.decoder_log_var = decoder_log_var

    def set_X0(self, X0):
        self.X0 = X0  # batch, node, 2
        return self

    def forward(self, ode_out, true_u, history_times, forecast_times, **kwargs):
        output_mean_list = []
        output_log_var_list = []
        for num, time in enumerate(forecast_times):
            if num == 0:
                hidden = ode_out[-1, ...]
                times = torch.cat([history_times[-1:], forecast_times[:1]], dim=0)
                z = ode_out.reshape(
                    ode_out.shape[0],
                    ode_out.shape[1],
                    self.num_node,
                    self.hidden_dim,
                    2,
                )[-1:, ..., -1].transpose(
                    0, 1
                )  # batch, 1, node, hidden
                y_pred_mean = (
                    self.mean_init_conv(z).permute(0, 2, 1, 3).contiguous()
                )  # batch, 1*output_dim, node, 1 -> batch, node, 1, output_dim
                y_pred_log_var = (
                    self.log_var_init_conv(z).permute(0, 2, 1, 3).contiguous()
                )  # batch, 1*output_dim, node, 1 -> batch, node, 1, output_dim
                # y_pred_mean = torch.clamp(y_pred_mean, -10, 10)
                # y_pred_log_var = torch.clamp(y_pred_log_var, -10, 10)
                # set true_u
                y_pred_mean[:, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(
                    -1
                )
                dx_dt = y_pred_mean - self.X0[..., -1:].unsqueeze(
                    -1
                )  # batch, node,  output_dim，1
                dx_dt = torch.cat([torch.ones_like(dx_dt), dx_dt], dim=-2).squeeze(-1)

            # build dx_dt
            # 构建一个batch，dx_dt的tensor  dx_dt: [y_pred, u_ture, ]
            self.ode_model.set_step_dx_dt(dx_dt)
            ode_out = self.odeint(
                func=self.ode_model,
                y0=hidden,
                t=times,
                method=self.solver,
            )  # history_len, batch, node*hidden*2

            hidden = ode_out[-1, ...]
            z = ode_out.reshape(
                ode_out.shape[0],
                ode_out.shape[1],
                self.num_node,
                self.hidden_dim,
                2,
            )[-1:, ..., -1].transpose(
                0, 1
            )  # batch, 1, node, hidden
            last_y_pred_mean = y_pred_mean
            y_pred_mean = (
                self.decoder_mean.step_forward(times, y_pred_mean, z.squeeze(1))
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            y_pred_log_var = (
                self.decoder_log_var.step_forward(times, y_pred_log_var, z.squeeze(1))
                .unsqueeze(-1)
                .unsqueeze(-1)
            )
            # set true_u
            y_pred_mean[:, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(-1)
            dx_dt = y_pred_mean - last_y_pred_mean  # batch, node,  output_dim，1
            dx_dt = torch.cat([torch.ones_like(dx_dt), dx_dt], dim=-2).squeeze(
                -1
            )  # 为了和上面的维度一致
            output_mean_list.append(y_pred_mean)
            output_log_var_list.append(y_pred_log_var)
            times = forecast_times[num : num + 2]
        output_mean = torch.cat(
            output_mean_list, dim=2
        )  # batch, forecast_len, node, output_dim
        output_log_var = torch.cat(
            output_log_var_list, dim=2
        )  # batch, forecast_len, node, output_dim
        # 只要PV
        output_mean = output_mean.permute(0, 2, 1, 3)[:, :, self.PV_index_list, :]
        output_log_var = output_log_var.permute(0, 2, 1, 3)[:, :, self.PV_index_list, :]
        return [output_mean, output_log_var]


class Out_func_rnn(nn.Module):
    def __init__(
        self,
        f_func,
        g_func,
        input_dim,
        hidden_channels,
        output_dim,
        output_size,
        num_node,
        PV_index_list,
        OP_index_list,
        forecast_len,
        device,
    ):
        super(Out_func_rnn, self).__init__()
        self.func_f = f_func
        self.func_g = g_func

        self.input_dim = input_dim
        self.PV_index_list = torch.tensor(PV_index_list, dtype=torch.long)
        self.OP_index_list = torch.tensor(OP_index_list, dtype=torch.long)
        self.hidden_channels = hidden_channels
        self.output_dim = output_dim
        self.output_size = output_size
        self.num_node = num_node
        self.forecast_len = forecast_len
        self.device = device
        self.rnn_input_dim = num_node * hidden_channels * 5
        self.rnn_hidden_dim = num_node * hidden_channels * 2
        self.final_conv = nn.Conv2d(
            1,
            1 * self.output_dim,
            kernel_size=(1, self.hidden_channels),
            bias=True,
        ).to(self.device)
        self.x_conv = nn.Conv2d(
            in_channels=1,
            out_channels=self.hidden_channels,
            kernel_size=(1, 1),
            bias=False,
        )
        self.rnn = nn.RNN(self.rnn_input_dim, self.rnn_hidden_dim, batch_first=True)

    def forward(self, rnn_out, true_u, history_times, forecast_times, **kwargs):
        rnn_out = rnn_out[0, ...]
        output_list = []
        for num, time in enumerate(forecast_times):
            if num == 0:
                hidden = rnn_out.reshape(
                    rnn_out.size(0), self.num_node, self.hidden_channels, 2
                )
                h = hidden[..., 0]
                z = hidden[..., 1]
                y_pred = self.final_conv(
                    z.unsqueeze(-1).permute(0, 3, 1, 2)
                ).contiguous()  # batch, 1*output_dim, node, 1 -> batch, output_dim， 1,  node
                # set true_u
                y_pred[:, :, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(1)

            vector_field_f = self.func_f(h)
            vector_field_g = self.func_g(z)
            vector_field_fg = torch.matmul(
                vector_field_g, vector_field_f
            )  # batch, node, hidden, 2
            hz = torch.stack([vector_field_f, vector_field_fg], dim=-1).reshape(
                hidden.size(0), self.num_node * self.hidden_channels * 4
            )
            x = self.x_conv(y_pred).reshape(hz.size(0), -1)
            input = torch.cat((x, hz), dim=1)
            hz, _ = self.rnn(input)
            hidden = hz.reshape(hz.size(0), self.num_node, self.hidden_channels, 2)
            h = hidden[..., 0]
            z = hidden[..., 1]
            y_pred = self.final_conv(
                z.unsqueeze(-1).permute(0, 3, 1, 2)
            ).contiguous()  # batch, 1*output_dim, node, 1 -> batch, output_dim， 1,  node
            # y_pred = torch.clamp(y_pred, -10, 10)
            # set true_u
            y_pred[:, :, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(1)
            output_list.append(y_pred)
        output = torch.cat(output_list, dim=1)  # batch, forecast_len, node, output_dim
        # 只要PV
        output = output[:, :, self.PV_index_list, :]
        return output
