# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.

import torch

from normalizing_flow.flow import make_conv_net


def make_checker_mask(shape, parity, device):
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker.to(device)


class AffineCoupling(torch.nn.Module):
    def __init__(self, net, *, mask_shape, mask_parity, device) -> None:
        super().__init__()
        self.mask = make_checker_mask(mask_shape, mask_parity, device)
        self.net = net

    def forward(self, x):
        x_frozen = self.mask * x
        x_active = (1 - self.mask) * x

        net_out = self.net(x_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]

        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))

        return fx, logJ

    def reverse(self, fx):
        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx
        net_out = self.net(fx_frozen.unsqueeze(1))
        s, t = net_out[:, 0], net_out[:, 1]
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * (-s), dim=tuple(axes))
        return x, logJ


def make_phi4_affine_layers(*, n_layers, lattice_shape, hidden_channels, kernel_size, device, float_dtype='float32'):
    layers = []

    for i in range(n_layers):
        parity = i % 2
        net = make_conv_net(
            in_channels=1,
            hidden_sizes=hidden_channels,
            out_channels=2,
            kernel_size=kernel_size,
            use_final_tanh=True,
            float_dtype=getattr(torch, float_dtype)
        )
        coupling = AffineCoupling(net, mask_shape=lattice_shape, mask_parity=parity, device=device)
        layers.append(coupling)

    return torch.nn.ModuleList(layers).to(device=device)
