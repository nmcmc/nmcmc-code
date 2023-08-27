import torch

from normalizing_flow.flow import make_conv_net
import normalizing_flow.u1_equivariant as equiv
from normalizing_flow.u1_equivariant import torch_mod
import normalizing_flow.rational_splines_u1 as rs
from normalizing_flow.schwinger_masks import schwinger_masks, schwinger_masks_with_2x1_loops
from phys_models.U1 import compute_u1_2x1_loops


def make_sin_cos(x):
    """Takes a 3 or 4 dimensional tensor and returns a 4 dimensional tensor that contains the cos and sin of the
    input concatenated along the dimension 1. If the input is 3 dimensional then dimension 1 is inserted into output.

    """
    if x.dim() == 3:
        return torch.stack((torch.cos(x), torch.sin(x)), dim=1)
    elif x.dim() == 4:
        return torch.cat((torch.cos(x), torch.sin(x)), dim=1)


def prepare_u1_input(plaq, plaq_mask, loops=(), loops_masks=()):
    p2 = plaq_mask['frozen'] * plaq
    net_in = [make_sin_cos(p2)]
    for i, l in enumerate(loops):
        sc = make_sin_cos(l * loops_masks[i])
        net_in.append(sc)
    return torch.cat(net_in, dim=1)


class GenericRSPlaqCouplingLayer(torch.nn.Module):
    def __init__(self, *, n_knots, net, masks, device):
        super().__init__()
        self.n_knots = n_knots
        self.n_bins = n_knots - 1
        self.plaq_mask = masks[0]
        self.loop_masks = masks[1:]
        self.net = net
        self.softplus = torch.nn.Softplus()
        self.device = device

    def call(self, x, dir, *loops):
        net_in = prepare_u1_input(x, self.plaq_mask, loops, self.loop_masks)

        net_out = self.net(net_in)

        w, h, d, t = (
            net_out[:, : self.n_bins, ...],
            net_out[:, self.n_bins: 2 * self.n_bins, ...],
            net_out[:, 2 * self.n_bins: 3 * self.n_bins, ...],
            net_out[:, -1, ...],
        )

        w, h, d = torch.softmax(w, 1), torch.softmax(h, 1), self.softplus(d)

        kx, ky, s = rs.make_circular_knots_array(w, h, d, device=self.device)

        bs = rs.make_bs(x.shape[0], self.n_knots, x.shape[1:], device=self.device)
        spline = rs.make_splines_array(kx, ky, s, bs)[dir]

        x1 = self.plaq_mask["active"] * torch_mod(x - dir * t)

        fx1, local_logJ = spline(x1)

        fx1 = self.plaq_mask["active"] * fx1
        local_logJ = self.plaq_mask["active"] * local_logJ
        axes = tuple(range(1, len(local_logJ.shape)))
        logJ = torch.sum(local_logJ, dim=axes)

        fx = (
                self.plaq_mask["active"] * torch_mod(fx1 + (1 - dir) * t)
                + self.plaq_mask["passive"] * x
                + self.plaq_mask["frozen"] * x
        )

        return fx, logJ

    def forward(self, x, *loops):
        return self.call(x, 0, *loops)

    def reverse(self, x, *loops):
        return self.call(x, 1, *loops)


def make_u1_equiv_layers_rs(
        *, n_layers, n_knots, lattice_shape, hidden_sizes, kernel_size, dilation=1, float_dtype, device):
    def make_plaq_coupling(mask):
        in_channels = 2  # x - > (cos(x), sin(x))
        out_channels = 3 * (n_knots - 1) + 1
        net = make_conv_net(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_sizes=hidden_sizes,
            kernel_size=kernel_size,
            use_final_tanh=False,
            dilation=dilation
        )
        net.to(device)
        return GenericRSPlaqCouplingLayer(
            n_knots=n_knots, net=net, masks=mask, device=device
        )

    link_mask_shape = (len(lattice_shape),) + lattice_shape

    masks = equiv.u1_masks(plaq_mask_shape=lattice_shape, link_mask_shape=link_mask_shape, float_dtype=float_dtype,
                           device=device)

    return equiv.make_u1_equiv_layers(make_plaq_coupling=make_plaq_coupling, masks=masks, n_layers=n_layers,
                                      device=device)


def make_u1_equiv_layers_rs_with_2x1_loops(
        *, n_layers, n_knots, lattice_shape, hidden_sizes, kernel_size, dilation=1, float_dtype, device):
    def make_plaq_coupling(mask):
        in_channels = 6
        out_channels = 3 * (n_knots - 1) + 1
        net = make_conv_net(
            in_channels=in_channels,
            out_channels=out_channels,
            hidden_sizes=hidden_sizes,
            kernel_size=kernel_size,
            use_final_tanh=False,
            dilation=dilation,
            float_dtype=getattr(torch, float_dtype)
        )
        net.to(device)
        return GenericRSPlaqCouplingLayer(
            n_knots=n_knots, net=net, masks=mask, device=device
        )

    link_mask_shape = (len(lattice_shape),) + tuple(lattice_shape)

    masks = schwinger_masks_with_2x1_loops(plaq_mask_shape=lattice_shape, link_mask_shape=link_mask_shape,
                                           float_dtype=float_dtype,
                                           device=device)

    return equiv.make_u1_equiv_layers(loops_function=lambda x: [compute_u1_2x1_loops(x)],
                                      make_plaq_coupling=make_plaq_coupling,
                                      masks=masks, n_layers=n_layers,
                                      device=device)
