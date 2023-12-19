import math
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from .GCN import GCN
from utils.datapreprocess import get_normalized_adj

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class ODEFunc(torch.nn.Module):

    def __init__(self, feature_dim, temporal_dim, adj):
        super(ODEFunc, self).__init__()
        self.adj = adj
        self.x0 = None
        self.alpha = nn.Parameter(0.8 * torch.ones(adj.shape[1]))
        self.beta = 0.6
        self.w = nn.Parameter(torch.eye(feature_dim))
        self.d = nn.Parameter(torch.zeros(feature_dim) + 1)
        self.w2 = nn.Parameter(torch.eye(temporal_dim))
        self.d2 = nn.Parameter(torch.zeros(temporal_dim) + 1)

    def forward(self, t, x):
        alpha = torch.sigmoid(self.alpha).unsqueeze(-1).unsqueeze(-1).unsqueeze(0)
        xa = torch.einsum('ij, kjlm->kilm', self.adj, x) # torch.Size([24, 24]) torch.Size([16, 60, 24, 64])

        # ensure the eigenvalues to be less than 1
        d = torch.clamp(self.d, min=0, max=1)
        w = torch.mm(self.w * d, torch.t(self.w))
        xw = torch.einsum('ijkl, lm->ijkm', x, w)

        d2 = torch.clamp(self.d2, min=0, max=1)
        w2 = torch.mm(self.w2 * d2, torch.t(self.w2))
        xw2 = torch.einsum('ijkl, km->ijml', x, w2)

        f = alpha / 2 * xa - x + xw - x + xw2 - x + self.x0
        return f


class ODEblock(torch.nn.Module):
    def __init__(self, odefunc, t=torch.tensor([0,1])):
        super(ODEblock, self).__init__()
        self.t = t
        self.odefunc = odefunc

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def forward(self, x):
        t = self.t.type_as(x)
        z = odeint(self.odefunc, x, t, method='euler')[1]
        return z


# Define the ODEGCN model.
class ODEG(torch.nn.Module):
    def __init__(self, feature_dim, temporal_dim, adj, time):
        super(ODEG, self).__init__()
        self.odeblock = ODEblock(ODEFunc(feature_dim, temporal_dim, adj), t=torch.tensor([0, time]))

    def forward(self, x):
        self.odeblock.set_x0(x)
        z = self.odeblock(x)
        return F.relu(z)


class Chomp1d(torch.nn.Module):
    """
    extra dimension will be added by padding, remove it
    """

    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :, :-self.chomp_size].contiguous()



class TemporalConvNet(torch.nn.Module):
    """
    time dilation convolution
    """

    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        """
        Args:
            num_inputs : channel's number of input data's feature
            num_channels : numbers of data feature tranform channels, the last is the output channel
            kernel_size : using 1d convolution, so the real kernel is (1, kernel_size)
        """
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation_size
            self.conv = nn.Conv2d(in_channels, out_channels, (1, kernel_size), dilation=(1, dilation_size),
                                  padding=(0, padding))
            self.conv.weight.data.normal_(0, 0.01)
            self.chomp = Chomp1d(padding)
            self.relu = nn.ReLU()
            self.dropout = nn.Dropout(dropout)

            layers += [nn.Sequential(self.conv, self.chomp, self.relu, self.dropout)]

        self.network = nn.Sequential(*layers)
        self.downsample = nn.Conv2d(num_inputs, num_channels[-1], (1, 1)) if num_inputs != num_channels[-1] else None
        if self.downsample:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        """
        like ResNet
        Args:
            X : input data of shape (B, N, T, F)
        """
        # permute shape to (B, F, N, T)
        # if x.dim() == 3:
        #     x = x.unsqueeze(3)
        y = x.permute(0, 3, 1, 2)
        y = F.relu(self.network(y) + self.downsample(y) if self.downsample else y)
        y = y.permute(0, 2, 3, 1)
        return y


# class GCN(nn.Module):
#     def __init__(self, A_hat, in_channels, out_channels, ):
#         super(GCN, self).__init__()
#         self.A_hat = A_hat
#         self.theta = nn.Parameter(torch.FloatTensor(in_channels, out_channels))
#         self.reset()
#
#     def reset(self):
#         stdv = 1. / math.sqrt(self.theta.shape[1])
#         self.theta.data.uniform_(-stdv, stdv)
#
#     def forward(self, X):
#         y = torch.einsum('ij, kjlm-> kilm', self.A_hat, X)
#         return F.relu(torch.einsum('kjlm, mn->kjln', y, self.theta))


class STGCNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, num_nodes, A_hat):
        """
        Args:
            in_channels: Number of input features at each node in each time step.
            out_channels: a list of feature channels in timeblock, the last is output feature channel
            num_nodes: Number of nodes in the graph
            A_hat: the normalized adjacency matrix
        """
        super(STGCNBlock, self).__init__()
        self.A_hat = A_hat
        self.temporal1 = TemporalConvNet(num_inputs=in_channels,
                                         num_channels=out_channels)
        self.odeg = ODEG(out_channels[-1], 60, A_hat, time=6)
        self.temporal2 = TemporalConvNet(num_inputs=out_channels[-1],
                                         num_channels=out_channels)
        self.batch_norm = nn.BatchNorm2d(num_nodes)

    def forward(self, X):
        """
        Args:
            X: Input data of shape (batch_size, num_nodes, num_timesteps, num_features)
        Return:
            Output data of shape(batch_size, num_nodes, num_timesteps, out_channels[-1])
        """
        t = self.temporal1(X)
        t = self.odeg(t)
        t = self.temporal2(F.relu(t))
        return self.batch_norm(t)


class STGODE(torch.nn.Module):
    """ the overall network framework """

    def __init__(self, **kwargs):
        super(STGODE, self).__init__()

        c_in = kwargs.get("history_len", 12) # num_timesteps_input
        c_out = 1  # 单步预测 # num_timesteps_output
        f_in = kwargs.get("feature_size", 1) # num_feature
        input_size = kwargs.get("input_size", 1)
        output_size = kwargs.get("output_size", 1)
        stride = kwargs.get("stride", 1)
        self.first_time_conv = kwargs.get("first_time_conv", False)
        first_time_conv_kernel_size = kwargs.get("first_time_conv_kernel_size", None)
        first_time_conv_stride = kwargs.get("first_time_conv_stride", None)
        padding = kwargs.get("padding", 0)
        K = kwargs.get("K", 2)
        device_name = kwargs.get("device", None)
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        dwt_path = kwargs.get("dwt_path", None)
        adj_mx_path = kwargs.get("adj_mx_path", None)
        dtw_matrix = np.load(dwt_path)
        sp_matrix = np.load(adj_mx_path)
        A_sp_hat = get_normalized_adj(sp_matrix).to(device)
        A_se_hat = get_normalized_adj(dtw_matrix).to(device)
        num_nodes = A_sp_hat.size(-1)

        if self.first_time_conv:
            self.conv1d_t = nn.Conv1d(
                input_size,
                input_size,
                first_time_conv_kernel_size,
                first_time_conv_stride,
            )
            history_len_after_conv = (
                                             c_in - first_time_conv_kernel_size
                                     ) // first_time_conv_stride + 1
            c_in = history_len_after_conv
        if adj_mx_path is None:
            raise ValueError("Please set the path of adjacent matrix !")

        """
        Args:
            num_nodes : number of nodes in the graph
            num_features : number of features at each node in each time step
            num_timesteps_input : number of past time steps fed into the network
            num_timesteps_output : desired number of future time steps output by the network 
            A_sp_hat : nomarlized adjacency spatial matrix 
            A_se_hat : nomarlized adjacency semantic matrix 
        """

        # spatial graph
        self.sp_blocks = nn.ModuleList(
            [nn.Sequential(
                STGCNBlock(in_channels=f_in, out_channels=[32, 16, 32],
                           num_nodes=num_nodes, A_hat=A_sp_hat),
                STGCNBlock(in_channels=32, out_channels=[32, 16, 32],
                           num_nodes=num_nodes, A_hat=A_sp_hat)) for _ in range(2)
            ])
        # semantic graph
        self.se_blocks = nn.ModuleList([nn.Sequential(
            STGCNBlock(in_channels=f_in, out_channels=[32, 16, 32],
                       num_nodes=num_nodes, A_hat=A_se_hat),
            STGCNBlock(in_channels=32, out_channels=[32, 16, 32],
                       num_nodes=num_nodes, A_hat=A_se_hat)) for _ in range(2)
        ])

        self.pred = nn.Sequential(
            nn.Linear(c_in * 32, c_out * 16),
            nn.ReLU(),
            nn.Linear(c_out * 16, c_out)
        )
        # self.out_linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x, **kwargs):
        """
        Args:
            x : input data of shape (batch_size, num_nodes, num_timesteps, num_features) == (B, N, T, F)
        Returns:
            prediction for future of shape (batch_size, num_nodes, num_timesteps_output)
        """
        x = x.permute(0, 2, 1)
        if x.dim() == 3:
            x = x.unsqueeze(3)
        outs = []
        # spatial graph
        for blk in self.sp_blocks:
            outs.append(blk(x))
        # semantic graph
        for blk in self.se_blocks:
            outs.append(blk(x))
        outs = torch.stack(outs)
        x = torch.max(outs, dim=0)[0]
        x = x.reshape((x.shape[0], x.shape[1], -1))
        x = self.pred(x).squeeze()
        x = x.unsqueeze(1)
        # x = self.out_linear(x).unsqueeze(1)

        return x