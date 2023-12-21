from torch import nn


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
