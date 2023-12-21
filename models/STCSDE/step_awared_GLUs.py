import math

import torch
from torch import nn


def time_encoder(time, batch_size, num_node, hidden_dim):

    # s_t = sin(wk*step) if i=2k ; cos(wk*step) if i=2k+1. wk=1/10000^(2k/history_len)
    s_t = torch.zeros(batch_size, num_node, dtype=torch.float32)
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


class step_awared_GLUs(nn.Module):
    def __init__(self, num_node, hidden_dim, forecast_steps, device="cpu"):
        super(step_awared_GLUs, self).__init__()
        self.num_node = num_node
        self.forecast_steps = forecast_steps
        self.device = device
        # ℎ_1=(𝑋W_1+b_1)⊙𝜎(𝑋W_2+b_2)
        self.w1 = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.w2 = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.b1 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.b2 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32), requires_grad=True
        )

        # ℎ_2=(h_1 V_1+b_3)⊙𝜎(𝑉_2 [𝑃𝑉]V_3+b_4)
        # V1 (num_node,num_node) V2 (output_size,1) V3 (output_size,num_node)
        self.V1 = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.V2 = torch.nn.Parameter(
            torch.randn(num_node, num_node, dtype=torch.float32), requires_grad=True
        )
        self.V3 = torch.nn.Parameter(
            torch.randn(1, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.b3 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.b4 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32), requires_grad=True
        )

        # ℎ_3=(h_2 U_1+b_5)⊙𝜎(𝑈_2 [p_𝑡]U_3+b_6)
        # U1 (num_node,num_node) U2 (num_node,1) U3 (output_size,num_node)
        self.U1 = torch.nn.Parameter(
            torch.randn(hidden_dim, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.U2 = torch.nn.Parameter(
            torch.randn(num_node, num_node, dtype=torch.float32), requires_grad=True
        )
        self.U3 = torch.nn.Parameter(
            torch.randn(1, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.b5 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32), requires_grad=True
        )
        self.b6 = torch.nn.Parameter(
            torch.randn(num_node, hidden_dim, dtype=torch.float32), requires_grad=True
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
        h1 = torch.mul(
            torch.add(torch.matmul(z, self.w1), self.b1),
            self.sigmoid(torch.add(torch.matmul(z, self.w2), self.b2)),
        )
        h1 = h1 + z
        # ℎ_2=(h_1 V_1+b_3)⊙𝜎(𝑉_2 [𝑃𝑉]V_3+b_4)
        h2 = torch.mul(
            torch.add(torch.matmul(h1, self.V1), self.b3),
            self.sigmoid(
                torch.add(
                    torch.matmul(
                        torch.matmul(last_y_pred, self.V2).unsqueeze(-1), self.V3
                    ),
                    self.b4,
                )
            ),
        )
        h2 = h2 + h1
        # ℎ_3=(h_2 U_1+b_5)⊙𝜎(𝑈_2 [p_𝑡]U_3+b_6)
        step_encoding = self.time_encoder(
            t, batch_size, num_node, self.forecast_steps
        ).to(self.device)
        h3 = torch.mul(
            torch.add(torch.matmul(h2, self.U1), self.b5),
            self.sigmoid(
                torch.add(
                    torch.matmul(
                        torch.matmul(step_encoding, self.U2).unsqueeze(-1), self.U3
                    ),
                    self.b6,
                )
            ),
        )
        h3 = h3 + h2
        z = self.out_func(h3).view(batch_size, -1)
        return z
