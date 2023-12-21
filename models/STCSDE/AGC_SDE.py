import math

import torch
import torchdiffeq
import torchsde
import torch.nn.functional as F
from torch import nn

from .Abstract_SDE import Abstract_SDE
from ..GCN import GCN


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


class AGCSDE_Func(Abstract_SDE):
    def __init__(
        self,
        hidden_dim,
        node_num,
        default_graph,
        device="cpu",
        cheb_K=2,
        noise_type="diagonal",
        sde_type="ito",
        save_attention_func=None,
    ):
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
        self.adj_mx = GCN.build_adj_matrix(
            default_graph, device, adj_type="cheb", K=cheb_K
        )
        self.E = None
        self.E_n = None
        self.noise_type = noise_type
        self.sde_type = "ito"
        if default_graph is None:
            raise ValueError("default_graph must be given for AGCSDE_Func")
        self.save_attention_func = save_attention_func

    def set_E(self, E):
        self.E = E  # .clone().detach()
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
        # adj = T_hat.unsqueeze(dim=1) * A_hat
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
            raise ValueError("T must be given for AGCSDE_Func")
        if self.save_attention_func is not None:
            self.save_attention_func(N, E)
        N_hat = N / E
        xe = torch.matmul(N_hat, x)
        xe = xe.reshape(xe.shape[0], -1)
        xe = xe.tanh()
        return xe


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
        )[1]
        # if xde_type = sde, reshape it to （b, n, f）
        z = z.reshape(z.shape[0], self.node_num, -1)
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_dim, self.hidden_dim)
        z = z.tanh()
        return z


class g_agcsde(g_sgcn):
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
        super(g_agcsde, self).__init__(
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


###########################################################################################
# ----------Below is the Method Implementation of the Ablation Experiment---------------- #
###########################################################################################


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
        default_graph=None,
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
    def __init__(
        self,
        hidden_dim,
        node_num,
        default_graph,
        embed_dim,
        xde_type="ode",
        cheb_K=2,
        device="cpu",
    ):
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
        self.adj_mx = GCN.build_adj_matrix(
            default_graph, device, adj_type="cheb", K=cheb_K
        )

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


class AGCNODEFunc(nn.Module):
    def __init__(
        self,
        hidden_dim,
        node_num,
        default_graph,
        embed_dim,
        xde_type="ode",
        cheb_K=2,
        device="cpu",
    ):
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
        self.adj_mx = GCN.build_adj_matrix(
            default_graph, device, adj_type="cheb", K=cheb_K
        )

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
        z = odeint(self.odefunc, x, t, method="euler")[1]
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_dim, self.hidden_dim)
        z = z.tanh()
        return z
