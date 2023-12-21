import torch
import torch.nn as nn
import numpy as np
import torchdiffeq
import torchsde

from Abstract_SDE import Abstract_SDE
from base_nn import f_linear, f_conv, init_hz_conv, init_hz_fc, G_func_linear
from AGC_SDE import (
    AGCSDE_Func,
    g_agcsde,
    AGCNODEFunc,
    GODEFunc,
    g_ogcn,
    g_sgcn,
    g_cheb_gcn,
)
from CSDE_decoder import (
    CSDE_decoder,
    CODE_decoder,
    Out_func_csde,
    Out_func_code,
    Out_func_conv,
    Out_func_rnn,
)
from CSDE_encoder import F_func_hz_CDE, F_func_hz_RNN
from utils import controldiffeq


class NeuralCSDE(nn.Module):
    def __init__(self, **kwargs):
        super(NeuralCSDE, self).__init__()
        self.g_type = kwargs.get("g_type", "agc")
        self.num_node = kwargs.get("input_size")
        self.ouput_size = kwargs.get("output_size")
        self.num_u_node = len(kwargs.get("OP_index_list"))
        self.input_dim = kwargs.get("input_dim")
        self.hidden_dim = kwargs.get("hid_dim")
        self.hid_hid_dim = kwargs.get("hid_hid_dim")
        self.output_dim = kwargs.get("output_dim")
        self.horizon = kwargs.get("history_len")
        self.forecast_len = kwargs.get("forecast_len")
        self.num_layers = kwargs.get("num_layers")
        self.embed_dim = kwargs.get("embed_dim")
        self.device = kwargs.get("device", "cpu")
        self.adaptive_graph = kwargs.get("adaptive_graph", False)
        self.default_graph = (
            self.load_default_graph(kwargs.get("adj_mx_path", None))
            if not self.adaptive_graph
            else None
        )
        self.cheb_order = kwargs.get("cheb_order", 2)
        self.noise_type = kwargs.get("noise_type", "diagonal")
        self.sde_type = kwargs.get("sde_type", "ito")
        self.f_type = kwargs.get("f_type", "linear")
        self.g_type = kwargs.get("g_type", "agc")
        self.out_type = kwargs.get("out_type", "conv")
        self.xde_type = kwargs.get("xde_type", "sde")
        self.init_type = kwargs.get("init_type", "fc")
        self.decoder_type = kwargs.get("decoder_type", "csde")
        self.solver = kwargs.get("solver", "euler")
        self.adjoint = kwargs.get("adjoint", True)
        self.cheb_order = kwargs.get("cheb_order", 2)
        self.g_dt = torch.tensor(
            float(kwargs.get("g_dt", 1e-3)), dtype=torch.float32
        ).to(self.device)
        self.t_dt = torch.tensor(float(kwargs.get("t_dt", 1)), dtype=torch.float32).to(
            self.device
        )
        self.attention_D = []
        self.attention_N = []
        if self.f_type == "linear":
            func_f_class = f_linear
        elif self.f_type == "conv":
            func_f_class = f_conv
        else:
            raise ValueError("Check f_type argument")
        self.func_f = func_f_class(
            input_dim=self.input_dim,
            hidden_channels=self.hidden_dim,
            hidden_hidden_channels=self.hid_hid_dim,
            num_hidden_layers=self.num_layers,
        ).to(self.device)

        if self.g_type == "agcsde":
            self.agcsde_func = AGCSDE_Func(
                hidden_dim=self.hidden_dim,
                node_num=self.num_node,
                default_graph=self.default_graph,
                device=self.device,
                cheb_K=self.cheb_order,
                noise_type=self.noise_type,
                sde_type=self.sde_type,
                save_attention_func=None,
                # save_attention_func=self.store_attention,
            )
            self.func_g = g_agcsde(
                sde_func=self.agcsde_func,
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                node_num=self.num_node,
                adjoint=self.adjoint,
                dt=self.g_dt,
            ).to(self.device)
        # ----------Below is the Method Implementation of the Ablation Experiment---------------- #
        elif self.g_type == "agc":
            func_g_class = g_cheb_gcn
            self.func_g = func_g_class(
                input_dim=self.input_dim,
                hidden_channels=self.hidden_dim,
                hidden_hidden_channels=self.hid_hid_dim,
                num_hidden_layers=self.num_layers,
                num_nodes=self.num_node,
                cheb_k=self.cheb_order,
                embed_dim=self.embed_dim,
                g_type=self.g_type,
                default_graph=self.default_graph,
            ).to(self.device)
        elif self.g_type == "agcn":
            func_g_class = g_cheb_gcn
            self.func_g = func_g_class(
                input_dim=self.input_dim,
                hidden_channels=self.hidden_dim,
                hidden_hidden_channels=self.hid_hid_dim,
                num_hidden_layers=self.num_layers,
                num_nodes=self.num_node,
                cheb_k=self.cheb_order,
                embed_dim=self.embed_dim,
                g_type=self.g_type,
                default_graph=self.default_graph,
            ).to(self.device)

        elif self.g_type == "ogcn":
            self.gode_func = GODEFunc(
                hidden_dim=self.hidden_dim,
                node_num=self.num_node,
                default_graph=self.default_graph,
                xde_type="ode",
                embed_dim=self.embed_dim,
                cheb_K=self.cheb_order,
                device=self.device,
            )
            self.func_g = g_ogcn(
                odefunc=self.gode_func,
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                adjoint=self.adjoint,
            ).to(self.device)
        elif self.g_type == "agcn_ode":
            self.gode_func = AGCNODEFunc(
                hidden_dim=self.hidden_dim,
                node_num=self.num_node,
                default_graph=self.default_graph,
                xde_type="ode",
                embed_dim=self.embed_dim,
                cheb_K=self.cheb_order,
                device=self.device,
            )
            self.func_g = g_ogcn(
                odefunc=self.gode_func,
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                adjoint=self.adjoint,
            ).to(self.device)
        elif self.g_type == "sgcn":
            self.gode_func = GODEFunc(
                hidden_dim=self.hidden_dim,
                node_num=self.num_node,
                default_graph=self.default_graph,
                xde_type="sde",
                embed_dim=self.embed_dim,
                cheb_K=self.cheb_order,
                device=self.device,
            )
            self.g_diffusion_func = G_func_linear(
                input_dim=self.input_dim,
                hidden_channels=self.hidden_dim,
                hidden_hidden_channels=self.hid_hid_dim,
                num_hidden_layers=self.num_layers,
            )
            self.func_g = g_sgcn(
                trend_func=self.gode_func,
                diffusion_func=self.g_diffusion_func,
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                node_num=self.num_node,
                adjoint=self.adjoint,
                dt=self.g_dt,
            ).to(self.device)
        elif self.g_type == "agcsde":
            self.agcsde_func = AGCSDE_Func(
                hidden_dim=self.hidden_dim,
                node_num=self.num_node,
                default_graph=self.default_graph,
                device=self.device,
                cheb_K=self.cheb_order,
                noise_type=self.noise_type,
                sde_type=self.sde_type,
                save_attention_func=None,
                # save_attention_func=self.store_attention,
            )
            self.func_g = g_agcsde(
                sde_func=self.agcsde_func,
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                node_num=self.num_node,
                adjoint=self.adjoint,
                dt=self.g_dt,
            ).to(self.device)

        else:
            raise ValueError("Check g_type argument")

        self.F_func = F_func_hz_CDE(
            dX_dt=None,
            func_f=self.func_f,
            func_g=self.func_g,
            node_num=self.num_node,
            hidden_channels=self.hidden_dim,
        ).to(self.device)

        # SDE G_func
        self.G_func = G_func_linear(
            input_dim=self.input_dim,
            hidden_channels=self.hidden_dim,
            hidden_hidden_channels=self.hid_hid_dim,
            num_hidden_layers=self.num_layers,
        ).to(self.device)
        if self.xde_type == "sde":
            self.sde_model = Abstract_SDE(
                self.F_func, self.G_func, self.noise_type, self.sde_type
            ).to(self.device)

            self.sdeint = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint

        # ----------Below is the Method Implementation of the Ablation Experiment---------------- #
        elif self.xde_type == "ode":
            self.ode_model = self.F_func
            self.odeint = (
                torchdiffeq.odeint_adjoint if self.adjoint else torchdiffeq.odeint
            )
        elif self.xde_type == "rnn":
            self.rnn_model = F_func_hz_RNN(
                func_f=self.func_f,
                func_g=self.func_g,
                node_num=self.num_node,
                hidden_channels=self.hidden_dim,
            ).to(self.device)
        else:
            raise ValueError("Check xde_type argument")
        if self.decoder_type == "csde":
            self.decoder_mean = CSDE_decoder(
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                node_num=self.num_node,
                output_size=self.ouput_size,
                forecast_len=self.forecast_len,
                device=self.device,
                adjoint=self.adjoint,
                solver=self.solver,
            )
            self.decoder_log_var = CSDE_decoder(
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                node_num=self.num_node,
                output_size=self.ouput_size,
                forecast_len=self.forecast_len,
                device=self.device,
                adjoint=self.adjoint,
                solver=self.solver,
            )
        if self.decoder_type == "code":
            self.decoder_mean = CODE_decoder(
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                node_num=self.num_node,
                output_size=self.ouput_size,
                forecast_len=self.forecast_len,
                device=self.device,
                adjoint=self.adjoint,
                solver=self.solver,
            )
            self.decoder_log_var = CODE_decoder(
                hidden_dim=self.hidden_dim,
                hidden_hidden_dim=self.hid_hid_dim,
                node_num=self.num_node,
                output_size=self.ouput_size,
                forecast_len=self.forecast_len,
                device=self.device,
                adjoint=self.adjoint,
                solver=self.solver,
            )

        if self.out_type == "csde":
            self.output_func = Out_func_csde(
                sde_model=self.sde_model,
                sdeint=self.sdeint,
                dt=self.t_dt,
                solver=self.solver,
                decoder_mean=self.decoder_mean,
                decoder_log_var=self.decoder_log_var,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                output_size=self.ouput_size,
                num_node=self.num_node,
                PV_index_list=kwargs.get("PV_index_list"),
                OP_index_list=kwargs.get("OP_index_list"),
                forecast_len=self.forecast_len,
                device=self.device,
            ).to(self.device)
        # ----------Below is the Method Implementation of the Ablation Experiment---------------- #
        elif self.out_type == "conv":
            self.output_func = Out_func_conv(
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                output_size=self.ouput_size,
                num_node=self.num_node,
                u_node=self.num_u_node,
                forecast_len=self.forecast_len,
                device=self.device,
            ).to(self.device)
        elif self.out_type == "code":
            self.output_func = Out_func_code(
                ode_model=self.ode_model,
                odeint=self.odeint,
                solver=self.solver,
                decoder_mean=self.decoder_mean,
                decoder_log_var=self.decoder_log_var,
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                output_size=self.ouput_size,
                num_node=self.num_node,
                PV_index_list=kwargs.get("PV_index_list"),
                OP_index_list=kwargs.get("OP_index_list"),
                forecast_len=self.forecast_len,
                device=self.device,
            ).to(self.device)
        elif self.out_type == "rnn":
            self.output_func = Out_func_rnn(
                f_func=self.func_f,
                g_func=self.func_g,
                input_dim=self.input_dim,
                hidden_channels=self.hidden_dim,
                output_dim=self.output_dim,
                output_size=self.ouput_size,
                num_node=self.num_node,
                PV_index_list=kwargs.get("PV_index_list"),
                OP_index_list=kwargs.get("OP_index_list"),
                forecast_len=self.forecast_len,
                device=self.device,
            ).to(self.device)
        else:
            raise ValueError("Check out_type argument")

        if self.init_type == "fc":
            self.init_func = init_hz_fc(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                device=self.device,
            ).to(self.device)
        elif self.init_type == "conv":
            self.init_func = init_hz_conv(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                device=self.device,
            ).to(self.device)
        else:
            raise ValueError("Check init_type argument")

        self.apply(self.weights_init)

    def weights_init(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.5)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
            print("initing", m)

    def load_default_graph(self, default_graph_path):
        # default_graph_path is path to adj.npy
        default_graph = np.load(default_graph_path)
        # to tensor dtype=torch.float32 to device
        default_graph = (
            torch.from_numpy(default_graph).type(torch.float32).to(self.device)
        )
        return default_graph

    def forward(self, history_times, forecast_times, coeffs, true_u):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        spline = controldiffeq.NaturalCubicSpline(history_times, coeffs)
        h0, z0 = self.init_func(spline.evaluate(history_times[0]))
        self.F_func = self.F_func.set_dx_dt(spline.derivative)
        # transform init0 to a 2d tensor (batch, node*hidden*2)
        init0 = torch.cat((h0, z0), dim=-1).reshape(h0.size(0), -1).to(self.device)
        # 还原(batch, node，hidden，2)
        # init0 = init0.reshape(h0.size(0), h0.size(1), h0.size(2), -1)
        if self.xde_type == "sde":
            self.sde_model.F_func.history_mode()
            xde_out = self.sdeint(
                self.sde_model,
                init0,
                history_times,
                dt=self.t_dt,
                method=self.solver,
                names={"drift": "F_func", "diffusion": "G_func"},
            )  # history_len, batch, node*hidden*2
            self.sde_model.F_func.forecast_mode()
        elif self.xde_type == "ode":
            self.ode_model.history_mode()
            xde_out = self.odeint(
                func=self.ode_model,
                y0=init0,
                t=history_times,
                method=self.solver,
            )
            self.ode_model.forecast_mode()
        elif self.xde_type == "rnn":
            xde_out = self.rnn_model(spline, history_times, init0)
        else:
            raise ValueError("Check xde_type argument")

        if self.out_type in ["code", "csde"]:
            self.output_func.set_X0(spline.evaluate(history_times[-1]))
        output = self.output_func(
            xde_out, true_u, history_times=history_times, forecast_times=forecast_times
        )  # batch, forecast_len, node, output_dim

        return output
