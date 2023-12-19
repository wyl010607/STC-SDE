import math
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils.controldiffeq as controldiffeq
import torchsde
import torchdiffeq
from .GCN import GCN


class Abstract_SDE(nn.Module):
    def __init__(self, F_func, G_func, noise_type, sde_type):
        super().__init__()
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.F_func = F_func
        self.G_func = G_func

    def set_x0(self, x0):
        self.F_func.x0 = x0.clone().detach()
    def F(self, t, y):
        return 0.1 * self.F_func(t, y)

    def G(self, t, y):
        return 0.1 * self.G_func(t, y)

def time_encoder(time, batch_size, num_node, hidden_dim):

    # s_t = sin(wk*step) if i=2k ; cos(wk*step) if i=2k+1. wk=1/10000^(2k/history_len)
    s_t = torch.zeros(
        batch_size, num_node, dtype=torch.float32
    )
    for i in range(num_node):
        if i % 2 == 0:
            s_t[:, i] = torch.sin(
                torch.pow(10000, torch.tensor(2 * i / hidden_dim)) * time
            )
        else:
            s_t[:, i] = torch.cos(
                torch.pow(10000, torch.tensor(2 * i / hidden_dim)) * time
            )
    return s_t


class init_hz_fc(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(init_hz_fc, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.initial_h = torch.nn.Linear(self.input_dim, self.hidden_dim).to(
            self.device
        )
        self.initial_z = torch.nn.Linear(self.input_dim, self.hidden_dim).to(
            self.device
        )

    def forward(self, x0):
        h0 = self.initial_h(x0).transpose(1, 2).squeeze()
        z0 = self.initial_z(x0).transpose(1, 2).squeeze()
        return h0, z0


class init_hz_conv(nn.Module):
    def __init__(self, input_dim, hidden_dim, device):
        super(init_hz_conv, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.start_conv_h = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=(1, 1),
        ).to(self.device)
        self.start_conv_z = nn.Conv2d(
            in_channels=self.input_dim,
            out_channels=self.hidden_dim,
            kernel_size=(1, 1),
        ).to(self.device)

    def forward(self, x0):
        h0 = (
            self.start_conv_h(x0.transpose(1, 2).unsqueeze(-1))
            .transpose(1, 2)
            .squeeze()
        )
        z0 = (
            self.start_conv_z(x0.transpose(1, 2).unsqueeze(-1))
            .transpose(1, 2)
            .squeeze()
        )
        return h0, z0


class f_linear(nn.Module):
    def __init__(
        self,
        input_dim,
        hidden_channels,
        hidden_hidden_channels,
        num_hidden_layers,
    ):
        super(f_linear, self).__init__()
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)

        self.linears = nn.ModuleList(
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        )
        self.linear_out = nn.Linear(
            hidden_hidden_channels, input_dim * hidden_channels
        )  # 32,32*4  -> # 32,32,4

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(
            *z.shape[:-1], self.hidden_channels, self.input_dim
        )
        z = z.tanh()
        return z


class f_conv(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_channels,
        hidden_hidden_channels,
        num_hidden_layers,
    ):
        super(f_conv, self).__init__()

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers

        self.start_conv = torch.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            kernel_size=(1, 1),
        )

        self.linears = torch.nn.ModuleList(
            torch.nn.Conv2d(
                in_channels=hidden_channels,
                out_channels=hidden_channels,
                kernel_size=(1, 1),
            )
            for _ in range(num_hidden_layers - 1)
        )

        self.linear_out = torch.nn.Conv2d(
            in_channels=hidden_channels,
            out_channels=input_dim * hidden_channels,
            kernel_size=(1, 1),
        )

    def forward(self, z):
        # z: torch.Size([64, 207, 32])
        z = self.start_conv(z.transpose(1, 2).unsqueeze(-1))
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()

        z = (
            self.linear_out(z)
            .squeeze()
            .transpose(1, 2)
            .view(*z.transpose(1, 2).shape[:-2], self.hidden_channels, self.input_dim)
        )
        z = z.tanh()
        return z
class g_cheb_gcn(torch.nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_channels,
        hidden_hidden_channels,
        num_hidden_layers,
        num_nodes,
        cheb_k,
        embed_dim,
        g_type,
        default_graph=None
    ):
        super(g_cheb_gcn, self).__init__()

        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        self.default_graph = default_graph

        self.linear_in = torch.nn.Linear(hidden_channels, hidden_hidden_channels)

        self.linear_out = torch.nn.Linear(
            hidden_hidden_channels, hidden_channels * hidden_channels
        )

        self.g_type = g_type
        if self.g_type == "agc" or self.g_type == "agcn":
            self.node_embeddings = nn.Parameter(
                torch.randn(num_nodes, embed_dim), requires_grad=True
            )
            self.cheb_k = 2
            self.weights_pool = nn.Parameter(
                torch.FloatTensor(
                    embed_dim, cheb_k, hidden_hidden_channels, hidden_hidden_channels
                )
            )
            self.bias_pool = nn.Parameter(
                torch.FloatTensor(embed_dim, hidden_hidden_channels)
            )
        if self.g_type == "agcn":
            self.f_spatial_attention = SpatialAttention(num_nodes, hidden_channels, 1)

    def forward(self, z):
        z = self.linear_in(z)
        z = z.relu()

        if self.g_type == "agc":
            z = self.agc(z)
        elif self.g_type == "agcn":
            z = self.agcn(z)
        else:
            raise ValueError("Check g_type argument")
        # for linear in self.linears:
        #     z = linear(x_gconv)
        #     z = z.relu()

        z = self.linear_out(z).view(
            *z.shape[:-1], self.hidden_channels, self.hidden_channels
        )
        z = z.tanh()
        return 0.1 * z  # torch.Size([64, 307, 64, 1])

    def agc(self, z):
        """
        Adaptive Graph Convolution
        - Node Adaptive Parameter Learning
        - Data Adaptive Graph Generation
        """
        node_num = self.node_embeddings.shape[0]
        if self.default_graph is not None:
            supports = self.default_graph
        else:
            supports = F.softmax(
                F.relu(
                    torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))
                ),
                dim=1,
            )
        # laplacian=False
        laplacian = False
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            )
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum(
            "nd,dkio->nkio", self.node_embeddings, self.weights_pool
        )  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, z)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum("bnki,nkio->bno", x_g, weights) + bias  # b, N, dim_out
        return z
    def agcn(self, z):
        """
        attention Graph Convolution
        """
        node_num = self.node_embeddings.shape[0]
        if self.default_graph is not None:
            supports = self.default_graph
        else:
            supports = F.softmax(
                F.relu(
                    torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))
                ),
                dim=1,
            )
        # laplacian=False
        laplacian = False
        if laplacian == True:
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            support_set = [supports, -torch.eye(node_num).to(supports.device)]
            # support_set = [torch.eye(node_num).to(supports.device), -supports]
            # support_set = [-supports]
        else:
            support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(
                torch.matmul(2 * supports, support_set[-1]) - support_set[-2]
            )
        E = self.f_spatial_attention(z.unsqueeze(1))  # B, N, N
        supports = torch.stack(support_set, dim=0)
        supports = torch.matmul(E.unsqueeze(1), supports)
        weights = torch.einsum(
            "nd,dkio->nkio", self.node_embeddings, self.weights_pool
        )  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(self.node_embeddings, self.bias_pool)  # N, dim_out
        x_g = torch.einsum("bknm,bmc->bknc", supports, z)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        z = torch.einsum("bnki,nkio->bno", x_g, weights) + bias  # b, N, dim_out
        return z

class GODEFunc(nn.Module):
    def __init__(self, hidden_dim, node_num, default_graph, embed_dim, xde_type="ode", cheb_K=2, device="cpu"):
        super(GODEFunc, self).__init__()
        self.default_graph = default_graph
        self.x0 = None
        self.node_num = node_num
        self.alpha = nn.Parameter(0.8 * torch.ones(node_num))
        self.beta = nn.Parameter(0.8 * torch.ones(node_num))
        self.w = nn.Parameter(torch.eye(hidden_dim))
        self.d = nn.Parameter(torch.zeros(hidden_dim) + 1)
        self.hidden_dim = hidden_dim
        self.xde_type = xde_type
        self.embed_dim = embed_dim
        if default_graph is None:
            self.node_embeddings = nn.Parameter(
                torch.randn(node_num, embed_dim), requires_grad=True
            )
        self.cheb_K = cheb_K
        self.channel_conv = torch.nn.Conv2d(cheb_K, 1, 1)
        self.adj_mx = GCN.build_adj_matrix(default_graph, device, adj_type="cheb", K=cheb_K)

    def forward(self, t, x):  # x: B, N, hidden
        # if xde_type = sde（b, n*f）, reshape it to 3 dim（b, n, f）
        if self.default_graph is not None:
            adj = self.adj_mx
        else:
            adj = F.softmax(
                F.relu(
                    torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))
                ),
                dim=1,
            )
        if self.xde_type == "sde":
            x = x.reshape(x.shape[0], self.node_num, -1)
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(0)  # 1, N, 1
        xa = self.channel_conv(torch.matmul(adj, x.unsqueeze(1))).squeeze(1)
        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)  # F
        w = torch.mm(self.w * d, torch.t(self.w))  # F, F
        xw = torch.einsum("bnf, fj->bnj", x, w)
        x0 = self.x0 * torch.sigmoid(self.beta).unsqueeze(-1).unsqueeze(0)  # 1, N, 1
        f = alpha / 2 * xa - x + xw - x + x0
        # if xde_type = sde, reshape it to （b, n*f）
        if self.xde_type == "sde":
            f = f.reshape(f.shape[0], -1)
        f = f.tanh()
        return f

class SpatialAttention(torch.nn.Module):
    """Compute Spatial attention scores.

    Args:
        num_nodes: Number of nodes.
        f_in: Number of features.
        c_in: Number of time steps.

    Shape:
        - Input: :math:`(batch\_size, c_{in}, num\_nodes, f_{in})`
        - Output: :math:`(batch\_size, num\_nodes, num\_nodes)`.
    """

    def __init__(self, num_nodes, f_in, c_in):
        super(SpatialAttention, self).__init__()

        self.w1 = torch.nn.Parameter(
            torch.randn(f_in, dtype=torch.float32), requires_grad=True
        )
        self.w2 = torch.nn.Parameter(
            torch.randn(f_in, dtype=torch.float32), requires_grad=True
        )
        self.vs = torch.nn.Parameter(
            torch.randn(num_nodes, num_nodes, dtype=torch.float32), requires_grad=True
        )

        self.bs = torch.nn.Parameter(
            torch.randn(num_nodes, num_nodes, dtype=torch.float32), requires_grad=True
        )
        torch.nn.init.kaiming_uniform_(self.vs, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.bs, a=math.sqrt(5))
        torch.nn.init.uniform_(self.w1)
        torch.nn.init.uniform_(self.w2)

    def forward(self, x):

        e1 = torch.matmul(x, self.w1).transpose(1, 2)
        e2 = torch.matmul(x, self.w2)

        product = torch.matmul(e1, e2)
        e = torch.matmul(self.vs, torch.sigmoid(product + self.bs))
        e = F.softmax(e, dim=-1)
        return e

class AGCNODEFunc(nn.Module):
    def __init__(self, hidden_dim, node_num, default_graph, embed_dim, xde_type="ode", cheb_K=2, device="cpu"):
        super(AGCNODEFunc, self).__init__()
        self.default_graph = default_graph
        self.x0 = None
        self.node_num = node_num
        self.alpha = nn.Parameter(0.8 * torch.ones(node_num))
        self.beta = nn.Parameter(0.8 * torch.ones(node_num))
        self.w = nn.Parameter(torch.eye(hidden_dim))
        self.d = nn.Parameter(torch.zeros(hidden_dim) + 1)
        self.f_spatial_attention = SpatialAttention(node_num, hidden_dim, 1)
        self.hidden_dim = hidden_dim
        self.xde_type = xde_type
        self.embed_dim = embed_dim
        if default_graph is None:
            self.node_embeddings = nn.Parameter(
                torch.randn(node_num, embed_dim), requires_grad=True
            )
        self.cheb_K = cheb_K
        self.channel_conv = torch.nn.Conv2d(cheb_K, 1, 1)
        self.adj_mx = GCN.build_adj_matrix(default_graph, device, adj_type="cheb", K=cheb_K)

    def forward(self, t, x):  # x: B, N, hidden
        # if xde_type = sde（b, n*f）, reshape it to 3 dim（b, n, f）
        if self.default_graph is not None:
            adj = self.adj_mx
        else:
            adj = F.softmax(
                F.relu(
                    torch.mm(self.node_embeddings, self.node_embeddings.transpose(0, 1))
                ),
                dim=1,
            )
        if self.xde_type == "sde":
            x = x.reshape(x.shape[0], self.node_num, -1)
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(0)  # 1, N, 1
        xa = self.channel_conv(torch.matmul(adj, x.unsqueeze(1))).squeeze(1)
        # E 是注意力权重矩阵
        E = self.f_spatial_attention(x.unsqueeze(1))  # B, N, N
        xe = torch.matmul(E, x)
        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)  # F
        w = torch.mm(self.w * d, torch.t(self.w))  # F, F
        xw = torch.einsum("bnf, fj->bnj", x, w)
        x0 = self.x0 * torch.sigmoid(self.beta).unsqueeze(-1).unsqueeze(0)  # 1, N, 1
        f = xe - x + alpha / 2 * xa - x + xw - x + x0
        # if xde_type = sde, reshape it to （b, n*f）
        if self.xde_type == "sde":
            f = f.reshape(f.shape[0], -1)
        f = f.tanh()
        return f


class g_ogcn(nn.Module):
    def __init__(
        self,
        odefunc,
        hidden_dim,
        hidden_hidden_dim,
        t=torch.tensor([0, 1]),
        adjoint=True,
    ):
        super(g_ogcn, self).__init__()
        self.t = t
        self.adjoint = adjoint
        self.odefunc = odefunc
        self.hidden_dim = hidden_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.linear_out = torch.nn.Linear(
            hidden_hidden_dim, hidden_dim * hidden_dim
        )  # 32,32*4  -> # 32,32,4

    def forward(self, x):
        t = self.t.type_as(x)
        self.odefunc.x0 = x.clone().detach()
        odeint = torchdiffeq.odeint_adjoint if self.adjoint else torchdiffeq.odeint
        z = odeint(self.odefunc, x, t, method="euler")[
            1
        ]
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_dim, self.hidden_dim)
        z = z.tanh()
        return z


class g_sgcn(nn.Module):
    def __init__(
        self,
        trend_func,
        diffusion_func,
        hidden_dim,
        hidden_hidden_dim,
        node_num,
        t=torch.tensor([0, 1]),
        adjoint=True,
        noise_type="diagonal",
        sde_type="ito",
        method="euler",
        dt=1e-3,
    ):
        super(g_sgcn, self).__init__()
        self.t = t
        self.adjoint = adjoint
        self.trend_func = trend_func
        self.diffusion_func = diffusion_func
        self.noise_type = noise_type
        self.sde_type = sde_type
        self.method = method
        self.dt = dt
        self.sde_func = Abstract_SDE(
            self.trend_func, self.diffusion_func, self.noise_type, self.sde_type
        )
        self.hidden_dim = hidden_dim
        self.hidden_hidden_dim = hidden_hidden_dim
        self.node_num = node_num
        self.linear_out = torch.nn.Linear(
            hidden_hidden_dim, hidden_dim * hidden_dim
        )  # 32,32*4  -> # 32,32,4

    def forward(self, x):
        t = self.t.type_as(x)
        self.sde_func.set_x0(x.clone().detach())
        # if xde_type = sde, reshape it to （b, n*f）
        x = x.reshape(x.shape[0], -1)
        sdeint = torchsde.sdeint_adjoint if self.adjoint else torchsde.sdeint
        z = sdeint(
            self.sde_func,
            x,
            t,
            dt=self.dt,
            method="euler",
            names={"drift": "F", "diffusion": "G"},
        )[
            1
        ]
        # if xde_type = sde, reshape it to （b, n, f）
        z = z.reshape(z.shape[0], self.node_num, -1)
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_dim, self.hidden_dim)
        z = z.tanh()
        return z

class SGATFunc(Abstract_SDE):
    def __init__(self, hidden_dim, node_num, default_graph,device="cpu",cheb_K=2,noise_type="diagonal",sde_type="ito",save_attention_func=None):
        super(Abstract_SDE, self).__init__()
        self.default_graph = default_graph
        self.x0 = None
        self.node_num = node_num
        # W is (hidden_dim, hidden_dim)
        self.w = nn.Parameter(torch.eye(hidden_dim))
        self.d = nn.Parameter(torch.zeros(hidden_dim) + 1)
        self.alpha = nn.Parameter(0.8 * torch.ones(node_num))
        self.hidden_dim = hidden_dim
        self.f_spatial_attention = SpatialAttention(node_num, hidden_dim, 1)
        self.g_spaital_attention = SpatialAttention(node_num, hidden_dim, 1)
        self.channel_conv = torch.nn.Conv2d(cheb_K, 1, 1)
        self.adj = default_graph
        self.cheb_K = cheb_K
        self.device = device
        self.adj_mx = GCN.build_adj_matrix(default_graph, device, adj_type="cheb", K=cheb_K)
        self.E = None
        self.E_n = None
        self.noise_type = noise_type
        self.sde_type = "ito"
        if default_graph is None:
            raise ValueError("default_graph must be given for SGATFunc")
        self.save_attention_func = save_attention_func


    def set_E(self, E):
        self.E = E#.clone().detach()
        return self

    def set_x0(self, x0):
        self.x0 = x0.clone().detach()
        return self
    def F(self, t, x):
        x = x.reshape(x.shape[0], self.node_num, -1)
        A = self.adj_mx
        A_hat = A - torch.eye(self.node_num).to(self.device)
        E = self.f_spatial_attention(x.unsqueeze(1))  # B, N, N
        E_hat = E - torch.eye(self.node_num).to(self.device)
        xe = torch.matmul(E_hat, x)
        #adj = T_hat.unsqueeze(dim=1) * A_hat
        adj = A_hat
        xa = self.channel_conv(torch.matmul(adj, x.unsqueeze(1))).squeeze(1)
        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)  # F
        W = torch.mm(self.w * d, torch.t(self.w))  # F, F
        W_hat = W - torch.eye(self.hidden_dim).to(self.device)
        xw = torch.matmul(x, W_hat)
        # self，alaph 用来控制X0的权重
        x0 = self.x0 * torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(0)  # 1, N, 1
        f = xe + xa + xw + x0
        # reshape to （b, n*f）
        f = f.reshape(f.shape[0], -1)
        f = f.tanh()
        self.set_E(E)
        return f


    def G(self, t, x):
        x = x.reshape(x.shape[0], self.node_num, -1)
        N = self.g_spaital_attention(x.unsqueeze(1))  # B, N, N
        if self.E is not None:
            E = self.E + 1e-5
        else:
            raise ValueError("T must be given for SGATFunc")
        if self.save_attention_func is not None:
            self.save_attention_func(N, E)
        N_hat = N / E
        xe = torch.matmul(N_hat, x)
        xe = xe.reshape(xe.shape[0], -1)
        xe = xe.tanh()
        return xe

class g_sgat(g_sgcn):
    def __init__(
        self,
        sde_func,
        hidden_dim,
        hidden_hidden_dim,
        node_num,
        t=torch.tensor([0, 1]),
        adjoint=True,
        noise_type="diagonal",
        sde_type="ito",
        method="euler",
        dt=1e-3,
    ):
        super(g_sgat, self).__init__(
            None,
            None,
            hidden_dim,
            hidden_hidden_dim,
            node_num,
            t,
            adjoint,
            noise_type,
            sde_type,
            method,
            dt,
        )
        self.sde_func = sde_func

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

class F_func_hz_RNN(nn.Module):
    def __init__(self,func_f, func_g, node_num, hidden_channels):
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
            )# batch, node, hidden, 2
            hz = torch.stack([vector_field_f, vector_field_fg], dim=-1).reshape(
                hz.size(0), self.node_num * self.hidden_channels * 4
            )
            x = self.x_conv(spline.evaluate(i)[...,1:].unsqueeze(1).permute(0, 1, 3, 2)).reshape(hz.size(0), -1)
            input = torch.cat((x, hz), dim=1)
            hz, _ = self.rnn(input)
        return hz


class G_func_linear(nn.Module):

    def __init__(
        self,
        input_dim,
        hidden_channels,
        hidden_hidden_channels,
        num_hidden_layers,
    ):
        super(G_func_linear, self).__init__()
        self.input_dim = input_dim
        self.hidden_channels = hidden_channels
        self.hidden_hidden_channels = hidden_hidden_channels
        self.num_hidden_layers = num_hidden_layers
        #    hidden_channels = hidden_channels * input_dim
        #    hidden_hidden_channels = hidden_hidden_channels * input_dim

        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)

        self.linears = nn.ModuleList(
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        )
        self.linear_out = nn.Linear(
            hidden_hidden_channels, hidden_channels
        )  # 32,32*4  -> # 32,32,4

    def forward(self, t, z):
        z = z.reshape(z.size(0), -1, self.hidden_channels)
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        # z: torch.Size([64, 207, 32])
        # self.linear_out(z): torch.Size([64, 207, 64])
        # transform to (batch, node*hidden*input_dim)
        z = self.linear_out(z).view(z.size(0), -1)
        z = z.tanh()
        return z

class decoder_F_func(nn.Module):
    def __init__(self, num_node, hidden_dim, forecast_steps, device="cpu"):
        super(decoder_F_func, self).__init__()
        self.num_node = num_node
        self.forecast_steps = forecast_steps
        self.device = device
        # ℎ_1=(𝑋W_1+b_1)⊙𝜎(𝑋W_2+b_2)
        self.w1 = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.w2 = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.b1 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.b2 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )

        # ℎ_2=(h_1 V_1+b_3)⊙𝜎(𝑉_2 [𝑃𝑉]V_3+b_4)
        # V1 (num_node,num_node) V2 (output_size,1) V3 (output_size,num_node)
        self.V1 = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.V2 = torch.nn.Parameter(
            torch.randn(num_node, num_node, dtype=torch.float32),
            requires_grad=True
        )
        self.V3 = torch.nn.Parameter(
            torch.randn(1, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.b3 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.b4 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )

        #ℎ_3=(h_2 U_1+b_5)⊙𝜎(𝑈_2 [p_𝑡]U_3+b_6)
        # U1 (num_node,num_node) U2 (num_node,1) U3 (output_size,num_node)
        self.U1 = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.U2 = torch.nn.Parameter(
            torch.randn(num_node, num_node, dtype=torch.float32),
            requires_grad=True
        )
        self.U3 = torch.nn.Parameter(
            torch.randn(1, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.b5 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        self.b6 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32),
            requires_grad=True
        )
        #
        self.out_func = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Conv1d(num_node, num_node, kernel_size=hidden_dim),
            )
        # 1D conv (batch, history_len, num_node) to (batch, 1, num_node)
        # Sigmoid
        self.sigmoid = nn.Sigmoid()
        self.time_encoder = time_encoder
        self.z = None
        for param_ in ["w1", "w2", "V1", "V2", "V3", "U1", "U2", "U3"]:
            torch.nn.init.kaiming_uniform_(getattr(self, param_), a=math.sqrt(5))
        for param_ in ["b1", "b2", "b3", "b4", "b5", "b6"]:
            torch.nn.init.uniform_(getattr(self, param_))

    def set_z(self, z):
        self.z = z


    def forward(self, t, last_y_pred):
        z = self.z
        batch_size, num_node, hidden_dim = z.shape
        # ℎ_1=(𝑋W_1+b_1)⊙𝜎(𝑋W_2+b_2)
        h1 = torch.mul(torch.add(torch.matmul(z, self.w1), self.b1), self.sigmoid(torch.add(torch.matmul(z, self.w2), self.b2)))
        h1 = h1 + z
        # ℎ_2=(h_1 V_1+b_3)⊙𝜎(𝑉_2 [𝑃𝑉]V_3+b_4)
        h2 = torch.mul(torch.add(torch.matmul(h1, self.V1), self.b3), self.sigmoid(torch.add(torch.matmul(torch.matmul(last_y_pred,self.V2).unsqueeze(-1), self.V3), self.b4)))
        h2 = h2 + h1
        # ℎ_3=(h_2 U_1+b_5)⊙𝜎(𝑈_2 [p_𝑡]U_3+b_6)
        step_encoding = self.time_encoder(t, batch_size, num_node, self.forecast_steps).to(self.device)
        h3 = torch.mul(torch.add(torch.matmul(h2, self.U1), self.b5), self.sigmoid(torch.add(torch.matmul(torch.matmul(step_encoding,self.U2).unsqueeze(-1) , self.U3), self.b6)))
        h3 = h3 + h2
        z = self.out_func(h3).view(batch_size, -1)
        return z


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
        self.decoder_F_func = decoder_F_func(num_node=node_num, hidden_dim=hidden_dim, forecast_steps=forecast_len, device=device)
        self.decoder_G_func = G_func_linear(
            input_dim=node_num,
            hidden_channels=node_num,
            hidden_hidden_channels=hidden_hidden_dim,
            num_hidden_layers=2,
        )
        self.decoder_sde_model = Abstract_SDE(
            self.decoder_F_func, self.decoder_G_func, noise_type="diagonal", sde_type="ito"
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
        self.decoder_F_func = decoder_F_func(num_node=node_num, hidden_dim=hidden_dim, forecast_steps=forecast_len, device=device)

    def step_forward(self, times, last_y_pred, z):
        self.decoder_F_func.set_z(z)
        odeint = torchdiffeq.odeint_adjoint if self.adjoint else torchdiffeq.odeint
        ode_out = odeint(self.decoder_F_func, last_y_pred.squeeze(), times, method="euler")
        out = ode_out[-1, ...].reshape(ode_out.shape[1], self.node_num)
        return out


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
        #self.final_conv = nn.Conv2d(
        #    1,
        #    1 * self.output_dim,
        #    kernel_size=(1, self.hidden_dim),
        #    bias=True,
        #).to(self.device)
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
                y_pred_mean[:, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(-1)
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
            #y_pred = self.final_conv(z).permute(
            #    0, 2, 1, 3
            #)  # batch, 1*output_dim, node, 1 -> batch, node, 1, output_dim
            y_pred_mean = self.decoder_mean.step_forward(times, y_pred_mean, z.squeeze(1)).unsqueeze(-1).unsqueeze(-1)
            y_pred_log_var = self.decoder_log_var.step_forward(times, y_pred_log_var, z.squeeze(1)).unsqueeze(-1).unsqueeze(-1)
            # set true_u
            #y_pred_mean = torch.clamp(y_pred_mean, -10, 10)
            #y_pred_log_var = torch.clamp(y_pred_log_var, -10, 10)
            y_pred_mean[:, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(-1)
            dx_dt = y_pred_mean - last_y_pred_mean  # batch, node,  output_dim，1
            dx_dt = torch.cat([torch.ones_like(dx_dt), dx_dt], dim=-2).squeeze(
                -1
            )
            output_mean_list.append(y_pred_mean)
            output_log_var_list.append(y_pred_log_var)
            times = forecast_times[num : num + 2]
        output_mean = torch.cat(output_mean_list, dim=2)  # batch, forecast_len, node, output_dim
        output_log_var = torch.cat(output_log_var_list, dim=2)  # batch, forecast_len, node, output_dim
        # 只要PV
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
        #self.final_conv = nn.Conv2d(
        #    1,
        #    1 * self.output_dim,
        #    kernel_size=(1, self.hidden_dim),
        #    bias=True,
        #).to(self.device)
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
                #y_pred_mean = torch.clamp(y_pred_mean, -10, 10)
                #y_pred_log_var = torch.clamp(y_pred_log_var, -10, 10)
                # set true_u
                y_pred_mean[:, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(-1)
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
            y_pred_mean = self.decoder_mean.step_forward(times, y_pred_mean, z.squeeze(1)).unsqueeze(-1).unsqueeze(-1)
            y_pred_log_var = self.decoder_log_var.step_forward(times, y_pred_log_var, z.squeeze(1)).unsqueeze(-1).unsqueeze(-1)
            # set true_u
            y_pred_mean[:, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(-1)
            dx_dt = y_pred_mean - last_y_pred_mean  # batch, node,  output_dim，1
            dx_dt = torch.cat([torch.ones_like(dx_dt), dx_dt], dim=-2).squeeze(
                -1
            )  # 为了和上面的维度一致
            output_mean_list.append(y_pred_mean)
            output_log_var_list.append(y_pred_log_var)
            times = forecast_times[num : num + 2]
        output_mean = torch.cat(output_mean_list, dim=2)  # batch, forecast_len, node, output_dim
        output_log_var = torch.cat(output_log_var_list, dim=2)  # batch, forecast_len, node, output_dim
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
        rnn_out = rnn_out[0,...]
        output_list = []
        for num, time in enumerate(forecast_times):
            if num == 0:
                hidden = rnn_out.reshape(rnn_out.size(0), self.num_node, self.hidden_channels, 2)
                h = hidden[..., 0]
                z = hidden[..., 1]
                y_pred = (
                    self.final_conv(z.unsqueeze(-1).permute(0, 3,1,2)).contiguous()
                )  # batch, 1*output_dim, node, 1 -> batch, output_dim， 1,  node
                # set true_u
                y_pred[:, : , self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(1)

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
            y_pred = (
                self.final_conv(z.unsqueeze(-1).permute(0, 3, 1, 2)).contiguous()
            )  # batch, 1*output_dim, node, 1 -> batch, output_dim， 1,  node
            #y_pred = torch.clamp(y_pred, -10, 10)
            # set true_u
            y_pred[:, :, self.OP_index_list, ...] = true_u[:, num, ...].unsqueeze(1)
            output_list.append(y_pred)
        output = torch.cat(output_list, dim=1)  # batch, forecast_len, node, output_dim
        # 只要PV
        output = output[:, :, self.PV_index_list, :]
        return output



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

        if self.g_type == "agc":
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
        elif self.g_type == "sgat":
            self.sgat_func = SGATFunc(
                hidden_dim=self.hidden_dim,
                node_num=self.num_node,
                default_graph=self.default_graph,
                device=self.device,
                cheb_K=self.cheb_order,
                noise_type=self.noise_type,
                sde_type=self.sde_type,
                save_attention_func=None,
                #save_attention_func=self.store_attention,
            )
            self.func_g = g_sgat(
                sde_func=self.sgat_func,
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
        elif self.xde_type == "ode":
            self.ode_model = self.F_func
            self.odeint = torchdiffeq.odeint_adjoint if self.adjoint else torchdiffeq.odeint
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
        # predictor
        if self.out_type == "conv":
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
        elif self.out_type == "csde":
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
    def store_attention(self, D, N):
        self.attention_D.append(D)
        self.attention_N.append(N)
    def get_attention(self):
        return self.attention_D, self.attention_N

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
        h0, z0 = self.init_func(
            spline.evaluate(history_times[0])
        )
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
            xde_out = self.rnn_model(
                spline, history_times, init0
            )
        else:
            raise ValueError("Check xde_type argument")

        if self.out_type in ["code", "csde"]:
            self.output_func.set_X0(spline.evaluate(history_times[-1]))
        output = self.output_func(
            xde_out, true_u, history_times=history_times, forecast_times=forecast_times
        )  # batch, forecast_len, node, output_dim

        return output