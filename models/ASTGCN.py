import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .GCN import GCN


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

        self.w1 = torch.nn.Conv2d(c_in, 1, 1, bias=False)
        self.w2 = torch.nn.Linear(f_in, c_in, bias=False)
        self.w3 = torch.nn.Parameter(
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
        torch.nn.init.uniform_(self.w3)

    def forward(self, x):
        y1 = self.w2(self.w1(x).squeeze(dim=1))
        y2 = torch.matmul(x, self.w3)

        product = torch.matmul(y1, y2)
        y = torch.matmul(self.vs, torch.sigmoid(product + self.bs))
        y = F.softmax(y, dim=-1)
        return y


class ChebConv(torch.nn.Module):
    """
    Graph Convolution with Chebyshev polynominals.

    Args:
        - input_feature: Dimension of input features.
        - out_feature: Dimension of output features.
        - adj_mx: Adjacent matrix with shape :math:`(K, num\_nodes, num\_nodes)` followed by
          Kth Chebyshev polynominals, where :math:`T_k(x) = 2xT_{k-1}(x) - T_{k-2}(x-2)` with
          :math:`T_0(x)=1, T_1(x) = x`.

    Shape:
        - Input:
            x: :math:`(batch\_size, c_in, num\_nodes, f_in)`.
            spatial_att: :math:`(batch\_size, num\_nodes, num\_nodes)`
        - Output:
            :math:`(batch_size, c_in, num_\nodes, f_out).
    """

    def __init__(self, input_feature, out_feature, adj_mx):
        super(ChebConv, self).__init__()

        self.adj_mx = adj_mx
        self.w = torch.nn.Linear(
            adj_mx.size(0) * input_feature, out_feature, bias=False
        )

    def forward(self, x, spatial_att):
        b, c_in, num_nodes, _ = x.size()

        outputs = []
        adj = spatial_att.unsqueeze(dim=1) * self.adj_mx
        for i in range(c_in):
            x1 = x[:, i].unsqueeze(dim=1)
            y = torch.matmul(adj, x1).transpose(1, 2).reshape(b, num_nodes, -1)
            y = torch.relu(self.w(y))
            outputs.append(y)
        return torch.stack(outputs, dim=1)


class TemporalAttention(torch.nn.Module):
    """Compute Temporal attention scores.

    Args:
        num_nodes: Number of vertices.
        f_in: Number of features.
        c_in: Number of time steps.

    Shape:
        - Input: :math:`(batch\_size, c_{in}, num\_nodes, f_{in})`
        - Output: :math:`(batch\_size, c_in, c_in)`.
    """

    def __init__(self, num_nodes, f_in, c_in):
        super(TemporalAttention, self).__init__()
        self.E_att_stored = None
        self.w1 = torch.nn.Parameter(
            torch.randn(num_nodes, dtype=torch.float32), requires_grad=True
        )
        self.w2 = torch.nn.Linear(f_in, num_nodes, bias=False)
        self.w3 = torch.nn.Parameter(
            torch.randn(f_in, dtype=torch.float32), requires_grad=True
        )
        self.be = torch.nn.Parameter(
            torch.randn(1, c_in, c_in, dtype=torch.float32), requires_grad=True
        )
        self.ve = torch.nn.Parameter(
            torch.zeros(c_in, c_in, dtype=torch.float32), requires_grad=True
        )

        torch.nn.init.kaiming_uniform_(self.ve, a=math.sqrt(5))
        torch.nn.init.kaiming_uniform_(self.be, a=math.sqrt(5))
        torch.nn.init.uniform_(self.w1)
        torch.nn.init.uniform_(self.w3)

    def forward(self, x):
        y1 = self.w2(torch.matmul(x.transpose(2, 3), self.w1))
        y2 = torch.matmul(x, self.w3).transpose(1, 2)

        product = torch.matmul(y1, y2)
        E = torch.matmul(self.ve, torch.sigmoid(product + self.be))
        E = F.softmax(E, dim=-1)
        # save E to self.E_att_stored, mean
        # if self.E_att_stored is None:
        #    self.E_att_stored = E.mean(dim=0)
        # else:
        #    self.E_att_stored = 0.5 * self.E_att_stored + 0.5 * E.mean(dim=0)
        return E


class ASTGCNBlock(torch.nn.Module):
    """One ASTGCN block with spatio-temporal attention, graph Convolution with Chebyshev
        polyniminals and temporal convolution.

    Args:
        c_in: Nunber of time_steps.
        f_in: Number of input features.
        num_che_filter: hidden size of chebyshev graph convolution.
        num_time_filter: Number of output channel for time convolution.
        kernel_size: Kernel size for time convolution.
        adj_mx: Adjacent matrix (Tensor) with shape :math:`(K, num\_nodes, num\_nodes)`.
        stride: Stride size for time convolution (Default 1).
        padding: Padding for time convolution (Default 0).

    Shape:
        - Input: :math:`(batch\_size, c_in, num\_nodes, f_in)`
        - Output: :math:`(batch\_size, c_out, num\_nodes, num_time_filter)`.
        :math:`c_out = (c_in + paddding * 2 -kernel_size) / stride + 1`,
    """

    def __init__(
        self,
        c_in,
        f_in,
        num_cheb_filter,
        num_time_filter,
        kernel_size,
        adj_mx,
        stride=1,
        padding=0,
    ):
        super(ASTGCNBlock, self).__init__()

        self.adj_mx = adj_mx
        self._c_out_conv = (c_in + 2 * padding - kernel_size) // stride + 1
        _c_out_res = (c_in - 1) // stride + 1
        self._res_start = (_c_out_res - self._c_out_conv) // 2

        num_nodes = adj_mx.size(-1)
        self.spatial_att = SpatialAttention(num_nodes, f_in, c_in)
        self.cheb_conv = ChebConv(f_in, num_cheb_filter, self.adj_mx)
        self.temporal_att = TemporalAttention(num_nodes, f_in, c_in)

        self.time_conv = torch.nn.Conv2d(
            in_channels=num_cheb_filter,
            out_channels=num_time_filter,
            kernel_size=(1, kernel_size),
            stride=(1, stride),
            padding=(0, padding),
        )

        self.residual_conv = torch.nn.Conv2d(
            in_channels=f_in,
            out_channels=num_time_filter,
            kernel_size=(1, 1),
            stride=(1, stride),
        )

        self.ln = torch.nn.LayerNorm(num_time_filter)

    def forward(self, x):

        b, c, n_d, f = x.size()

        temporal_att = self.temporal_att(x)
        x_tat = torch.matmul(temporal_att, x.reshape(b, c, -1)).reshape(b, c, n_d, f)

        spatial_att = self.spatial_att(x_tat)
        spatial_gcn = self.cheb_conv(x, spatial_att)

        time_conv_output = self.time_conv(spatial_gcn.permute(0, 3, 2, 1)).permute(
            0, 3, 2, 1
        )
        x_residual = self.residual_conv(x.permute(0, 3, 2, 1)).permute(0, 3, 2, 1)
        x_residual = x_residual[:, self._res_start : self._res_start + self._c_out_conv]

        relued = F.relu(x_residual + time_conv_output)
        return self.ln(relued)


class ASTGCN(torch.nn.Module):
    """ASTGCN module."""

    def __init__(self, **kwargs):
        super(ASTGCN, self).__init__()

        num_block = kwargs.get("num_block", 2)
        c_in = kwargs.get("history_len", 12)
        c_out = 1  # 单步预测
        input_size = kwargs.get("input_size", 1)
        output_size = kwargs.get("output_size", 1)
        f_in = kwargs.get("feature_size", 1)
        num_cheb_filter = kwargs.get("num_cheb_filter", 64)
        num_time_filter = kwargs.get("num_time_filter", 64)
        kernel_size = kwargs.get("kernel_size", 3)
        stride = kwargs.get("stride", 1)
        self.first_time_conv = kwargs.get("first_time_conv", False)
        first_time_conv_kernel_size = kwargs.get("first_time_conv_kernel_size", None)
        first_time_conv_stride = kwargs.get("first_time_conv_stride", None)
        padding = kwargs.get("padding", 0)
        K = kwargs.get("K", 2)
        adj_mx_path = kwargs.get("adj_mx_path", None)
        device_name = kwargs.get("device", None)
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
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

        adj = np.load(adj_mx_path)
        adj_mx = GCN.build_adj_matrix(adj, device, adj_type="cheb", K=K)

        self.blocks = torch.nn.Sequential()
        for i in range(1, num_block + 1):
            self.blocks.add_module(
                name="block_{}".format(i),
                module=ASTGCNBlock(
                    c_in,
                    f_in,
                    num_cheb_filter,
                    num_time_filter,
                    kernel_size,
                    adj_mx,
                    stride,
                    padding,
                ),
            )
            c_in, f_in = (
                c_in + 2 * padding - kernel_size
            ) // stride + 1, num_time_filter
        self.c_in = c_in
        self.final_conv = torch.nn.Conv2d(
            in_channels=c_in, out_channels=c_out, kernel_size=(1, num_time_filter)
        )
        #self.out_linear = torch.nn.Linear(input_size, output_size)

    def forward(self, x, **kwargs):
        if self.first_time_conv:
            x = self.conv1d_t(x.permute(0, 2, 1))
            x = x.permute(0, 2, 1)
        # if x dim is 3 add one dim on axis 3
        if x.dim() == 3:
            x = x.unsqueeze(3)
        y = self.blocks(x)
        # [32, 1, 24, 1] -> [32, 1, 24]
        y = self.final_conv(y).squeeze(-1)
        #y = self.out_linear(y).unsqueeze(1)
        return y
