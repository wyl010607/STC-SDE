import torch
from torch import nn


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
        self.linear_in = nn.Linear(hidden_channels, hidden_hidden_channels)

        self.linears = nn.ModuleList(
            torch.nn.Linear(hidden_hidden_channels, hidden_hidden_channels)
            for _ in range(num_hidden_layers - 1)
        )
        self.linear_out = nn.Linear(hidden_hidden_channels, hidden_channels)

    def forward(self, t, z):
        z = z.reshape(z.size(0), -1, self.hidden_channels)
        z = self.linear_in(z)
        z = z.relu()

        for linear in self.linears:
            z = linear(z)
            z = z.relu()
        z = self.linear_out(z).view(z.size(0), -1)
        z = z.tanh()
        return z


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
        z = self.linear_out(z).view(*z.shape[:-1], self.hidden_channels, self.input_dim)
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
