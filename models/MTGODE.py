import math
import numpy as np
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
#from .GCN import GCN
from utils.datapreprocess import get_normalized_adj
import torchdiffeq

# Whether use adjoint method or not.
adjoint = False
if adjoint:
    from torchdiffeq import odeint_adjoint as odeint
else:
    from torchdiffeq import odeint


class graph_constructor(nn.Module):
    def __init__(self, nnodes, k, dim, device, alpha=3, static_feat=None):
        super(graph_constructor, self).__init__()
        self.nnodes = nnodes
        if static_feat is not None:
            xd = static_feat.shape[1]
            self.lin1 = nn.Linear(xd, dim)
            self.lin2 = nn.Linear(xd, dim)
        else:
            self.emb1 = nn.Embedding(nnodes, dim)
            self.emb2 = nn.Embedding(nnodes, dim)
            self.lin1 = nn.Linear(dim,dim)
            self.lin2 = nn.Linear(dim,dim)

        self.device = device
        self.k = k
        self.dim = dim
        self.alpha = alpha
        self.static_feat = static_feat

    def forward(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        mask = torch.zeros(idx.size(0), idx.size(0)).to(self.device)
        mask.fill_(float('0'))
        # s1,t1 = adj.topk(self.k,1)
        s1, t1 = (adj + torch.rand_like(adj) * 0.01).topk(self.k, 1)  # bug fixed
        mask.scatter_(1,t1,s1.fill_(1))
        adj = adj*mask
        return adj

    def fullA(self, idx):
        if self.static_feat is None:
            nodevec1 = self.emb1(idx)
            nodevec2 = self.emb2(idx)
        else:
            nodevec1 = self.static_feat[idx,:]
            nodevec2 = nodevec1

        nodevec1 = torch.tanh(self.alpha*self.lin1(nodevec1))
        nodevec2 = torch.tanh(self.alpha*self.lin2(nodevec2))

        a = torch.mm(nodevec1, nodevec2.transpose(1,0))-torch.mm(nodevec2, nodevec1.transpose(1,0))
        adj = F.relu(torch.tanh(self.alpha*a))
        return adj

class dilated_inception(nn.Module):
    def __init__(self, cin, cout, dilation_factor=1):
        super(dilated_inception, self).__init__()
        self.tconv = nn.ModuleList()
        self.kernel_set = [2, 3, 6, 7]
        cout = int(cout / len(self.kernel_set))
        for kern in self.kernel_set:
            self.tconv.append(nn.Conv2d(cin, cout, (1, kern), dilation=(1, dilation_factor)))

    def forward(self, input):
        x = []
        for i in range(len(self.kernel_set)):
            x.append(self.tconv[i](input))
        for i in range(len(self.kernel_set)):
            x[i] = x[i][..., -x[-1].size(3):]
        x = torch.cat(x, dim=1)
        return x

class linear(nn.Module):
    def __init__(self,c_in,c_out,bias=True):
        super(linear,self).__init__()
        self.mlp = torch.nn.Conv2d(c_in, c_out, kernel_size=(1, 1), padding=(0,0), stride=(1,1), bias=bias)

    def forward(self,x):
        return self.mlp(x)

class CGPODEBlock(nn.Module):
    def __init__(self, cgpfunc, method, step_size, rtol, atol, adjoint, perturb, estimated_nfe):
        super(CGPODEBlock, self).__init__()
        self.odefunc = cgpfunc
        self.method = method
        self.step_size = step_size
        self.adjoint = adjoint
        self.perturb = perturb
        self.atol = atol
        self.rtol = rtol
        self.mlp = linear((estimated_nfe + 1) * self.odefunc.c_in, self.odefunc.c_out)

    def set_x0(self, x0):
        self.odefunc.x0 = x0.clone().detach()

    def set_adj(self, adj):
        self.odefunc.adj = adj

    def forward(self, x, t):
        self.integration_time = torch.tensor([0, t]).float().type_as(x)

        if self.adjoint:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                             method=self.method, options=dict(step_size=self.step_size, perturb=self.perturb))
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                     method=self.method, options=dict(step_size=self.step_size,
                                                                      perturb=self.perturb))

        outs = self.odefunc.out
        self.odefunc.out = []
        outs.append(out[-1])
        h_out = torch.cat(outs, dim=1)
        h_out = self.mlp(h_out)

        return h_out

class nconv(nn.Module):
    def __init__(self):
        super(nconv,self).__init__()

    def forward(self,x, A):
        # x.shape = (batch, dim, nodes, seq_len)
        # A.shape = (node, node)
        x = torch.einsum('ncwl,vw->ncvl', (x, A))
        return x.contiguous()

class CGPFunc(nn.Module):

    def __init__(self, c_in, c_out, init_alpha):
        super(CGPFunc, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.x0 = None
        self.adj = None
        self.nfe = 0
        self.alpha = init_alpha
        self.nconv = nconv()
        self.out = []

    def forward(self, t, x):
        adj = self.adj + torch.eye(self.adj.size(0)).to(x.device)
        d = adj.sum(1)
        _d = torch.diag(torch.pow(d, -0.5))
        adj_norm = torch.mm(torch.mm(_d, adj), _d)

        self.out.append(x)
        self.nfe += 1
        ax = self.nconv(x, adj_norm)
        f = 0.5 * self.alpha * (ax - x)
        return f

class CGP(nn.Module):
    def __init__(self, cin, cout, alpha=2.0, method='rk4', time=1.0, step_size=1.0,
                 rtol=1e-4, atol=1e-3, adjoint=False, perturb=False):

        super(CGP, self).__init__()
        self.c_in = cin
        self.c_out = cout
        self.alpha = alpha

        if method == 'euler':
            self.integration_time = time
            self.estimated_nfe = round(self.integration_time / step_size)
        elif method == 'rk4':
            self.integration_time = time
            self.estimated_nfe = round(self.integration_time / (step_size / 4.0))
        else:
            raise ValueError("Oops! The CGP solver is invaild.")

        self.CGPODE = CGPODEBlock(CGPFunc(self.c_in, self.c_out, self.alpha),
                                  method, step_size, rtol, atol, adjoint, perturb,
                                  self.estimated_nfe)

    def forward(self, x, adj):
        self.CGPODE.set_x0(x)
        self.CGPODE.set_adj(adj)
        h = self.CGPODE(x, self.integration_time)
        return h

class ODEFunc(nn.Module):
    def __init__(self, stnet):
        super(ODEFunc, self).__init__()
        self.stnet = stnet
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        x = self.stnet(x)
        return x
class ODEBlock(nn.Module):
    def __init__(self, odefunc, method, step_size, rtol, atol, adjoint=False, perturb=False):
        super(ODEBlock, self).__init__()
        self.odefunc = odefunc
        self.method = method
        self.step_size = step_size
        self.adjoint = adjoint
        self.perturb = perturb
        self.atol = atol
        self.rtol = rtol

    def forward(self, x, t):
        self.integration_time = torch.tensor([0, t]).float().type_as(x)
        if self.adjoint:
            out = torchdiffeq.odeint_adjoint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                             method=self.method, options=dict(step_size=self.step_size, perturb=self.perturb))
        else:
            out = torchdiffeq.odeint(self.odefunc, x, self.integration_time, rtol=self.rtol, atol=self.atol,
                                     method=self.method, options=dict(step_size=self.step_size, perturb=self.perturb))

        return out[-1]

class STBlock(nn.Module):

    def __init__(self, receptive_field, dilation, hidden_channels, dropout, method, time, step_size, alpha,
                 rtol, atol, adjoint, perturb=False):
        super(STBlock, self).__init__()
        self.receptive_field = receptive_field
        self.intermediate_seq_len = receptive_field
        self.graph = None
        self.dropout = dropout
        self.new_dilation = 1
        self.dilation_factor = dilation
        self.inception_1 = dilated_inception(hidden_channels, hidden_channels, dilation_factor=1)
        self.inception_2 = dilated_inception(hidden_channels, hidden_channels, dilation_factor=1)
        self.gconv_1 = CGP(hidden_channels, hidden_channels, alpha=alpha,
                           method=method, time=time, step_size=step_size, rtol=rtol, atol=atol,
                           adjoint=adjoint, perturb=perturb)
        self.gconv_2 = CGP(hidden_channels, hidden_channels, alpha=alpha,
                           method=method, time=time, step_size=step_size, rtol=rtol, atol=atol,
                           adjoint=adjoint, perturb=perturb)

    def forward(self, x):
        x = x[..., -self.intermediate_seq_len:]
        for tconv in self.inception_1.tconv:
            tconv.dilation = (1, self.new_dilation)
        for tconv in self.inception_2.tconv:
            tconv.dilation = (1, self.new_dilation)

        filter = self.inception_1(x)
        filter = torch.tanh(filter)
        gate = self.inception_2(x)
        gate = torch.sigmoid(gate)
        x = filter * gate

        self.new_dilation *= self.dilation_factor
        self.intermediate_seq_len = x.size(3)

        x = F.dropout(x, self.dropout, training=self.training)

        x = self.gconv_1(x, self.graph) + self.gconv_2(x, self.graph.transpose(1, 0))

        x = nn.functional.pad(x, (self.receptive_field - x.size(3), 0))

        return x

    def setGraph(self, graph):
        self.graph = graph

    def setIntermediate(self, dilation):
        self.new_dilation = dilation
        self.intermediate_seq_len = self.receptive_field

class MTGODE(torch.nn.Module):

    def __init__(self, **kwargs):
        super(MTGODE, self).__init__()

        device_name = kwargs.get("device", None)
        device = torch.device(device_name if torch.cuda.is_available() else "cpu")
        static_feat = kwargs.get("static_feat", None)
        subgraph_size = kwargs.get("subgraph_size", 20)
        node_dim = kwargs.get("node_dim", 40)
        dilation_exponential = kwargs.get("dilation_exponential", 1)
        conv_channels = kwargs.get("conv_channels", 32)
        end_channels = kwargs.get("end_channels", 128)
        in_dim = kwargs.get("in_dim", 1)
        out_dim = kwargs.get("out_dim", 1)
        tanhalpha = kwargs.get("tanhalpha", 3)
        method_1 = kwargs.get("method_1", 'euler')
        time_1 = kwargs.get("time_1", 1.2)
        step_size_1 = kwargs.get("step_1", 0.4)
        method_2 = kwargs.get("method_2", 'euler')
        time_2 = kwargs.get("time_2", 1.0)
        step_size_2 = kwargs.get("step_2", 0.25)
        alpha = kwargs.get("alpha", 1.0)
        rtol = kwargs.get("rtol", 1e-4)
        atol = kwargs.get("atol", 1e-3)
        perturb = kwargs.get("perturb", False)
        input_size = kwargs.get("input_size", 1)
        output_size = kwargs.get("output_size", 1)

        if method_1 == 'euler':
            self.integration_time = time_1
            self.estimated_nfe = round(self.integration_time / step_size_1)
        elif method_1 == 'rk4':
            self.integration_time = time_1
            self.estimated_nfe = round(self.integration_time / (step_size_1 / 4.0))
        else:
            raise ValueError("Oops! Temporal ODE solver is invaild.")

        self.buildA_true = kwargs.get("buildA_true", True)
        self.num_nodes = kwargs.get("num_nodes", 24)
        self.dropout = kwargs.get("dropout", 0.3)
        self.predefined_A = kwargs.get("predefined_A", None)
        self.seq_length = kwargs.get("seq_in_len", 12)
        self.ln_affine = kwargs.get("ln_affine", True)
        self.adjoint = kwargs.get("adjoint", False)

        self.start_conv = nn.Conv2d(in_channels=in_dim, out_channels=conv_channels, kernel_size=(1, 1))

        self.gc = graph_constructor(self.num_nodes, subgraph_size, node_dim, device, alpha=tanhalpha, static_feat=static_feat)
        self.idx = torch.arange(self.num_nodes).to(device)

        max_kernel_size = 7
        if dilation_exponential > 1:
            self.receptive_field = int(1 + (max_kernel_size - 1) * (dilation_exponential**self.estimated_nfe - 1) / (dilation_exponential - 1))
        else:
            self.receptive_field = self.estimated_nfe * (max_kernel_size - 1) + 1

        if self.ln_affine:
            self.affine_weight = nn.Parameter(torch.Tensor(*(conv_channels, self.num_nodes)))  # C*H
            self.affine_bias = nn.Parameter(torch.Tensor(*(conv_channels, self.num_nodes)))  # C*H

        self.ODE = ODEBlock(ODEFunc(STBlock(receptive_field=self.receptive_field, dilation=dilation_exponential,
                                            hidden_channels=conv_channels, dropout=self.dropout, method=method_2,
                                            time=time_2, step_size=step_size_2, alpha=alpha, rtol=rtol, atol=atol,
                                            adjoint=False, perturb=perturb)),
                            method_1, step_size_1, rtol, atol, adjoint, perturb)

        self.end_conv_0 = nn.Conv2d(in_channels=conv_channels, out_channels=end_channels//2, kernel_size=(1, 1), bias=True)
        self.end_conv_1 = nn.Conv2d(in_channels=end_channels//2, out_channels=end_channels, kernel_size=(1, 1), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=end_channels, out_channels=out_dim, kernel_size=(1, 1), bias=True)
        # self.out_linear = torch.nn.Linear(input_size, output_size)

        if self.ln_affine:
            self.reset_parameters()

    def reset_parameters(self):
        init.ones_(self.affine_weight)
        init.zeros_(self.affine_bias)

    def forward(self, input, idx=None, **kwargs):
        if input.dim() == 3:
            input = input.unsqueeze(3)
        input = input.permute(0, 3, 2, 1)
        seq_len = input.size(3)
        assert seq_len == self.seq_length, 'input sequence length not equal to preset sequence length'

        if self.seq_length < self.receptive_field:
            input = nn.functional.pad(input, (self.receptive_field-self.seq_length, 0))

        if self.buildA_true:
            if idx is None:
                adp = self.gc(self.idx)
            else:
                adp = self.gc(idx)
        else:
            adp = self.predefined_A

        x = self.start_conv(input)

        if self.adjoint:
            self.ODE.odefunc.stnet.setIntermediate(dilation=1)
        self.ODE.odefunc.stnet.setGraph(adp)
        x = self.ODE(x, self.integration_time)
        self.ODE.odefunc.stnet.setIntermediate(dilation=1)

        x = x[..., -1:]
        x = F.layer_norm(x, tuple(x.shape[1:]), weight=None, bias=None, eps=1e-5)

        if self.ln_affine:
            if idx is None:
                x = torch.add(torch.mul(x, self.affine_weight[:, self.idx].unsqueeze(-1)), self.affine_bias[:, self.idx].unsqueeze(-1))  # C*H
            else:
                x = torch.add(torch.mul(x, self.affine_weight[:, idx].unsqueeze(-1)), self.affine_bias[:, idx].unsqueeze(-1))  # C*H

        x = F.relu(self.end_conv_0(x))
        x = F.relu(self.end_conv_1(x))
        x = self.end_conv_2(x).squeeze().unsqueeze(1)

        return x