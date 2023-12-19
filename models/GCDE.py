import torch
import torch.nn.functional as F
import torch.nn as nn
import utils.controldiffeq as controldiffeq
from models.vector_fields import *


class NeuralGCDE(nn.Module):
    def __init__(self, **kwargs):
        super(NeuralGCDE, self).__init__()
        self.g_type = kwargs.get("g_type", "agc")
        self.num_node = kwargs.get("input_size")
        self.input_dim = kwargs.get("input_dim")
        self.hidden_dim = kwargs.get("hid_dim")
        self.hid_hid_dim = kwargs.get("hid_hid_dim")
        self.output_dim = kwargs.get("output_dim")
        self.horizon = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.num_layers = kwargs.get("num_layers")
        self.embed_dim = kwargs.get("embed_dim")
        self.default_graph = kwargs.get("default_graph", None)
        self.cheb_order = kwargs.get("cheb_order", 2)
        self.node_embeddings = nn.Parameter(
            torch.randn(self.num_node, self.embed_dim), requires_grad=True
        )
        self.device = kwargs.get("device", "cpu")
        vector_field_f = FinalTanh_f(
            input_channels=self.input_dim,
            hidden_channels=self.hidden_dim,
            hidden_hidden_channels=self.hid_hid_dim,
            num_hidden_layers=self.num_layers,
        )
        vector_field_g = VectorField_g(
            input_channels=self.input_dim,
            hidden_channels=self.hidden_dim,
            hidden_hidden_channels=self.hid_hid_dim,
            num_hidden_layers=self.num_layers,
            num_nodes=self.num_node,
            cheb_k=self.cheb_order,
            embed_dim=self.embed_dim,
            g_type=self.g_type,
        )
        self.func_f = vector_field_f.to(self.device)
        self.func_g = vector_field_g.to(self.device)
        self.solver = kwargs.get("solver", "rk4")
        self.atol = kwargs.get("atol", 1e-9)
        self.rtol = kwargs.get("rtol", 1e-7)
        self.adjoint = kwargs.get("adjoint", True)

        # predictor
        self.end_conv = nn.Conv2d(
            1,
            self.output_dim,
            kernel_size=(1, self.hidden_dim),
            bias=True,
        )

        self.init_type = "fc"
        if self.init_type == "fc":
            self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim)
            self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim)
        elif self.init_type == "conv":
            self.start_conv_h = nn.Conv2d(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                kernel_size=(1, 1),
            )
            self.start_conv_z = nn.Conv2d(
                in_channels=self.input_dim,
                out_channels=self.hidden_dim,
                kernel_size=(1, 1),
            )

    def forward(self, times, _, coeffs, true_u):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        spline = controldiffeq.NaturalCubicSpline(times, coeffs)
        if self.init_type == "fc":
            h0 = self.initial_h(spline.evaluate(times[0]))
            z0 = self.initial_z(spline.evaluate(times[0]))
        elif self.init_type == "conv":
            h0 = (
                self.start_conv_h(
                    spline.evaluate(times[0]).transpose(1, 2).unsqueeze(-1)
                )
                .transpose(1, 2)
                .squeeze()
            )
            z0 = (
                self.start_conv_z(
                    spline.evaluate(times[0]).transpose(1, 2).unsqueeze(-1)
                )
                .transpose(1, 2)
                .squeeze()
            )

        z_t = controldiffeq.cdeint_gde_dev(
            dX_dt=spline.derivative,
            h0=h0,
            z0=z0,
            func_f=self.func_f,
            func_g=self.func_g,
            t=times,
            method=self.solver,
            atol=self.atol,
            rtol=self.rtol,
            adjoint=self.adjoint,
        )
        # out:[h,z]
        # init_state = self.encoder.init_hidden(source.shape[0])
        # output, _ = self.encoder(source, init_state, self.node_embeddings)      #B, T, N, hidden
        # output = output[:, -1:, :, :]                                   #B, 1, N, hidden
        z_T = z_t[-1:, ...].transpose(0, 1)  # 只取z

        # CNN based predictor
        output = self.end_conv(
            z_T
        )
        output = output.squeeze(-1).reshape(
            -1, 1, self.output_dim, self.num_node
        )
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output
