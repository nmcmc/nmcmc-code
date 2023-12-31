import torch

from normalizing_flow.flow import make_conv_net
import normalizing_flow.u1_equivariant as equiv
from normalizing_flow.u1_equivariant import torch_mod, _prepare_u1_input
import normalizing_flow.rational_splines_u1 as rs
from normalizing_flow.schwinger_masks import schwinger_masks, schwinger_masks_with_2x1_loops
from phys_models.U1 import compute_u1_2x1_loops


class GenericRSPlaqCouplingLayer(torch.nn.Module):
    """Transform the plaquettes `x` using the circular splines transformation.

    See https://arxiv.org/abs/2002.02428 for more details.
    """

    def __init__(self, *, n_knots, net, masks, device):
        """

        Parameters
        ----------
        n_knots
            Number of knots in the spline. Because in circular splines the first and the last knot are the same
            the real number of knots is n_knots - 1.
        net
            neural network that computes the parameters of the splines
        masks
            list of masks used for masking the plaquettes and optionally other loops
        device
            device on which the computation is performed
        """

        super().__init__()
        self.n_knots = n_knots
        self.n_bins = n_knots - 1
        self.plaq_mask = masks[0]
        self.loop_masks = masks[1:]
        self.net = net
        self.softplus = torch.nn.Softplus()
        self.device = device

    def call(self, x, dir, *loops):
        """Transform the plaquettes `x` using the circular splines transformation.

        The parameters of the transformation are given by the neural network `net`.
        The input to the network is the plaquette `x` and optionally the loops `loops`.
        The plaquettes are masked using self.plaq_mask and the loops are masked using self.loop_masks.

        Parameters
        ----------
        x
            A tensor of shape (batch_size, L, L) containing plaquettes.
        dir
            Direction of the transformation. 0 for forward and 1 for reverse.
        loops
            Optional loops to be used as input to the neural network.

        Returns
        -------
            A tuple of transformed plaquettes and the log of the Jacobian of the transformation.
        """

        net_in = _prepare_u1_input(x, self.plaq_mask, loops, self.loop_masks)

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
        *, type='plaq', n_layers, n_knots, lattice_shape, hidden_sizes, kernel_size, dilation=1, float_dtype, device):
    """Make a list of equivariant layers that transform the links using the circular splines transformation for plaquettes.


    Parameters
    ----------
    type
        type of the equivariant layer. Can be 'plaq', 'sch' or 'sch_2x1.
        This defines the masking pattern used for the plaquettes and links and the type of additional loops if any.
        `plaq` uses a simple plaquette masking pattern defined in https://arxiv.org/abs/2003.06413
        `sch` uses the masking pattern defined in http://arxiv.org/abs/2202.11712 but without the 2x1 loops.
        `sch_2x1` uses the masking pattern defined in http://arxiv.org/abs/2202.11712 together with the 2x1 loops.
    n_layers
        Number of layers.
    n_knots
        Number of knots in the spline. Because in circular splines the first and the last knot are the same
        the real  number of knots is n_knots - 1.
    lattice_shape
        Shape of the lattice.
    hidden_sizes
        Number of channels in the hidden layers of the neural network. Also defines the number of hidden layers.
    kernel_size
        Kernel size of the convolutional layers in the neural network.
    dilation
        A list of dilations for the convolutional layers in the neural network.
        If an integer is given then the same dilation is used for all layers.
    float_dtype
        Type of the floating point numbers used in the computation.
    device
        Device on which the computation is performed.

    Returns
    -------
        Torch module containing a list of equivariant layers.
    """

    in_channels = 0
    match type:
        case 'plaq':
            in_channels = 2  # x - > (cos(x), sin(x))
        case 'sch':
            in_channels = 2 # x - > (cos(x), sin(x))
        case 'sch_2x1':
            in_channels = 6 # x - > (cos(x), sin(x), cos(x_2x1), sin(x_2x1), cos(x_1x2), sin(x_1x2))
        case _:
            raise ValueError(f"Unknown type {type}")

    def _make_plaq_coupling(mask):
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

    loops_function = None
    match type:
        case 'plaq':
            masks = equiv.u1_masks(plaq_mask_shape=lattice_shape, link_mask_shape=link_mask_shape,
                                   float_dtype=float_dtype,
                                   device=device)

        case 'sch':
            masks = schwinger_masks(plaq_mask_shape=lattice_shape, link_mask_shape=link_mask_shape,
                                    float_dtype=float_dtype,
                                    device=device)
        case 'sch_2x1':
            masks = schwinger_masks_with_2x1_loops(plaq_mask_shape=lattice_shape, link_mask_shape=link_mask_shape,
                                                   float_dtype=float_dtype,
                                                   device=device)

            def loops_function(x):
                return [compute_u1_2x1_loops(x)]

    return equiv.make_u1_equiv_layers(make_plaq_coupling=_make_plaq_coupling, masks=masks, n_layers=n_layers,
                                      device=device, loops_function=loops_function)


def make_u1_equiv_layers_rs_with_2x1_loops(**kwargs):
    if 'type' in kwargs:
        del kwargs['type']
    return make_u1_equiv_layers_rs(type='sch_2x1', **kwargs)
