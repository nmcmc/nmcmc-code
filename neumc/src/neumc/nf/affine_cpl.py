# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.

import torch

from neumc.nf.nn import make_conv_net


def make_checker_mask(shape, parity, device):
    """
    Make a checkerboard mask with the given shape and parity.

    Parameters
    ----------
    shape : tuple
        Dimensions of the mask.
    parity: int
        Parity of the mask. If zero mask[0,0] = 0, if one mask[0,0] = 1.
    device:
        Device on which the mask should be created.

    Returns
    -------
    torch.Tensor
        Tensor representing the mask.
    """
    checker = torch.ones(shape, dtype=torch.uint8) - parity
    checker[::2, ::2] = parity
    checker[1::2, 1::2] = parity
    return checker.to(device)


class AffineCoupling(torch.nn.Module):
    """
    Affine coupling layer.

    Parameters
    ----------
    net : torch.nn.Module
        Neural network to be used in the coupling layer.
    lattice_shape : tuple
        Shape of the lattice.
    mask_parity : int
        Parity of the mask.
    device :
        Device to be used.
    """

    def __init__(self, net, *, lattice_shape, mask_parity, device) -> None:
        super().__init__()
        self.mask = make_checker_mask(lattice_shape, mask_parity, device)
        self.net = net

    def forward(self, x):
        r"""
        Forward pass of the affine coupling layer.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, \*lattice_shape) representing batch_size of configurations.
            Each configuration is transformed separately.

        Returns
        -------
        fx: torch.Tensor
            Transformed tensor of shape (batch_size, \*lattice_shape).
        logJ: torch.Tensor
            Logarithm of the Jacobian determinant for each transformed configuration.
        """

        x_frozen = self.mask * x
        x_active = (1 - self.mask) * x

        net_out = self.net(
            x_frozen.unsqueeze(1)
        )  # unsqueeze to add the channel dimension
        s, t = net_out[:, 0], net_out[:, 1]

        fx = (1 - self.mask) * t + x_active * torch.exp(s) + x_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * s, dim=tuple(axes))

        return fx, logJ

    def reverse(self, fx):
        r"""
        Reverse pass of the affine coupling layer.

        Parameters
        ----------
        fx  : torch.Tensor
            Input tensor of shape (batch_size, \*lattice_shape) representing batch_size of transformed configurations.
            Each configuration is transformed separately.

        Returns
        -------
        x: torch.Tensor
            Transformed tensor of shape (batch_size, \*lattice_shape).
        logJ: torch.Tensor
            Logarithm of the Jacobian determinant for each transformed configuration.
        """
        # TODO: check the sign of the Jacobian logarithm.

        fx_frozen = self.mask * fx
        fx_active = (1 - self.mask) * fx
        net_out = self.net(
            fx_frozen.unsqueeze(1)
        )  # unsqueeze to add the channel dimension
        s, t = net_out[:, 0], net_out[:, 1]
        x = (fx_active - (1 - self.mask) * t) * torch.exp(-s) + fx_frozen
        axes = range(1, len(s.size()))
        logJ = torch.sum((1 - self.mask) * (-s), dim=tuple(axes))
        return x, logJ


def make_phi4_affine_layers(
        *,
        n_layers,
        lattice_shape,
        hidden_channels,
        kernel_size,
        dilation=1,
        device,
        float_dtype=torch.float32,
):
    layers = []

    for i in range(n_layers):
        parity = i % 2
        net = make_conv_net(
            in_channels=1,
            hidden_channels=hidden_channels,
            out_channels=2,
            kernel_size=kernel_size,
            use_final_tanh=True,
            float_dtype=float_dtype,
            dilation=dilation
        )
        coupling = AffineCoupling(
            net, lattice_shape=lattice_shape, mask_parity=parity, device=device
        )
        layers.append(coupling)

    return torch.nn.ModuleList(layers).to(device=device)
