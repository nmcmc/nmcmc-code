# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.
import typing
import numpy as np
import torch

from phys_models.U1 import compute_u1_plaq, torch_mod


class GenericGaugeEquivCouplingLayer(torch.nn.Module):
    """A generic gauge equivariant link coupling layer for U(1) gauge theory.

    Enables a gauge equivariant transformation of the links. Given a link configuration forward pass calculates
    the plaquettes and optionally other gauge invariant loops. Those are passed to the plaquette coupling layer
    which transforms the plaquettes. The transformed plaquettes are then used to transform the links.
    The probability of the transformed links is calculated using the Jacobian of the transformation.

    The reverse  pass reverses the transformation of the links.

    See https://arxiv.org/abs/2003.06413 for more details.
    """

    def __init__(self, *, active_links_mask: torch.Tensor,
                 loops_function: typing.Callable[[torch.Tensor], typing.Sequence[torch.Tensor]] = None,
                 plaq_coupling,
                 ):
        """
        Parameters
        ----------
        active_links_mask
            A mask that selects which links are active. The mask is a tensor of shape (2, L, L)
        loops_function
            A function that computes loops from links. The function takes a tensor of shape (2, L, L)
            and returns a sequence of tensors, each tensor corresponding to a particular set of loops.
            This allows for adding the dependencies on other loops  then plaquettes to the plaquette coupling layer.
        plaq_coupling
            A coupling layer that operates on plaquettes and loops. It returns a transformed plaquettes and
            the log of the Jacobian of the transformation.
        """
        super().__init__()

        self.active_links_mask = active_links_mask
        self.plaq_coupling = plaq_coupling
        self.loops_function = loops_function

    def forward(self, x):
        """Forward pass of the coupling layer.

        Parameters
        ----------
        x
            A tensor of shape (2, L, L) containing the links.

        Returns
        -------
            Transformed links and the log of the Jacobian of the transformation.
        """

        assert torch.all(x < 2 * torch.pi)
        plaq = compute_u1_plaq(x, mu=0, nu=1)
        if self.loops_function:
            loops = self.loops_function(x)
        else:
            loops = ()
        try:
            new_plaq, logJ = self.plaq_coupling(plaq, *loops)
        except AssertionError:
            # Your debugging code goes here
            raise AssertionError
        delta_plaq = new_plaq - plaq
        delta_links = torch.stack((delta_plaq, -delta_plaq), dim=1)  # signs for U vs Udagger
        fx = self.active_links_mask * torch_mod(delta_links + x) + (1 - self.active_links_mask) * x

        return fx, logJ

    def reverse(self, fx):
        """Reverse pass of the coupling layer.

        Parameters
        ----------
        fx
            A tensor of shape (2, L, L) containing links.
        Returns
        -------
             Links transformed by the inverse of the forward() transformation and the log of the Jacobian of
             the transformation.
        """
        assert torch.all(fx < 2 * torch.pi)
        new_plaq = compute_u1_plaq(fx, mu=0, nu=1)
        if self.loops_function:
            loops = self.loops_function(fx)
        else:
            loops = []
        plaq, logJ = self.plaq_coupling.reverse(new_plaq, *loops)
        delta_plaq = plaq - new_plaq
        delta_links = torch.stack((delta_plaq, -delta_plaq), dim=1)  # signs for U vs Udagger
        x = self.active_links_mask * torch_mod(delta_links + fx) + (1 - self.active_links_mask) * fx

        return x, logJ


def make_u1_equiv_layers(*, make_plaq_coupling, masks, n_layers, device,
                         loops_function=None):
    """ Make a list of `n_layers` of `GenericGaugeEquivCouplingLayer` coupling layers.

    For each layer a tuple of (link_masks, plaq_masks) is taken from the `masks` generator.  `plaq_mask` is passed to
    `make_plaq_coupling` functionthat generates a  plaquette coupling layer with the given mask.
    Then the equivalent link coupling layer is created with the given plaquette coupling layer and `link_mask`.
    The loop function is also passed to the link coupling layer. The output from this function must be compatible with
    the input of the plaquette coupling layer.

    Parameters
    ----------
    make_plaq_coupling
        a function that makes a plaquette coupling layer with given mask
    masks
        a list or generator of masks
    n_layers
        number of coupling layers
    device
        device to put the coupling layers on
    loops_function
        function that compute optional additinal loops from links


    Returns
    -------
        a list of coupling layers
    """

    layers = []
    for i in range(n_layers):
        link_mask, plaq_mask = next(masks)
        plaq_coupling = make_plaq_coupling(plaq_mask)
        link_coupling = GenericGaugeEquivCouplingLayer(
            loops_function=loops_function,
            active_links_mask=link_mask,
            plaq_coupling=plaq_coupling,
        )

        layers.append(link_coupling)
    return torch.nn.ModuleList(layers).to(device)


def make_2d_link_active_stripes(shape, mu, off, float_dtype, torch_device):
    """
    Stripes mask looks like in the `mu` channel (mu-oriented links)::

      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction, and the pattern is offset in the nu
    direction by `off` (mod 4). The other channel is identically 0.
    """
    assert len(shape) == 2 + 1, "need to pass shape suitable for 2D gauge theory"
    assert shape[0] == len(shape[1:]), "first dim of shape must be Nd"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[mu, :, 0::4] = 1
    elif mu == 1:
        mask[mu, 0::4] = 1
    nu = 1 - mu
    mask = np.roll(mask, off, axis=nu + 1)
    return torch.from_numpy(mask.astype(float_dtype)).to(torch_device)


def make_single_stripes(shape, mu, off, device):
    """
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0
      1 0 0 0 1 0 0 0 1 0 0

    where vertical is the `mu` direction. Vector of 1 is repeated every 4 row/columns.
    The pattern is offset in perpendicular to the mu direction by `off` (mod 4).
    """
    assert len(shape) == 2, "need to pass 2D shape"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:, 0::4] = 1
    elif mu == 1:
        mask[0::4] = 1
    mask = np.roll(mask, off, axis=1 - mu)
    return torch.from_numpy(mask).to(device)


# %%
def make_double_stripes(shape, mu, off, device):
    """
    Double stripes mask looks like::

      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0
      1 1 0 0 1 1 0 0

    where vertical is the `mu` direction. The pattern is offset in perpendicular
    to the mu direction by `off` (mod 4).
    """
    assert len(shape) == 2, "need to pass 2D shape"
    assert mu in (0, 1), "mu must be 0 or 1"

    mask = np.zeros(shape).astype(np.uint8)
    if mu == 0:
        mask[:, 0::4] = 1
        mask[:, 1::4] = 1
    elif mu == 1:
        mask[0::4] = 1
        mask[1::4] = 1
    mask = np.roll(mask, off, axis=1 - mu)
    return torch.from_numpy(mask).to(device)


def make_plaq_masks(mask_shape, mask_mu, mask_off, device):
    mask = {}
    mask["frozen"] = make_double_stripes(mask_shape, mask_mu, mask_off + 1, device=device)
    mask["active"] = make_single_stripes(mask_shape, mask_mu, mask_off, device=device)
    mask["passive"] = 1 - mask["frozen"] - mask["active"]
    return mask


def u1_masks(*, plaq_mask_shape, link_mask_shape, float_dtype, device):
    """Generator of masks for U(1) equivariant coupling layers.

    Parameters
    ----------
    plaq_mask_shape
        Shape of the plaquette mask.
    link_mask_shape
        Shape of the link mask.
    float_dtype
        float type of the masks
    device
        device to put the masks on

    Yields
    -------
        link_mask
        plaq_maks
            Non-empty list. First item on the list is the plaquette mask,
            the rest are loop masks if present.
    """
    i = 0
    while True:
        # periodically loop through all arrangements of masks
        mu = i % 2
        off = (i // 2) % 4

        link_mask = make_2d_link_active_stripes(
            link_mask_shape, mu, off, float_dtype, device
        )

        plaq_mask = make_plaq_masks(plaq_mask_shape, mu, off, device=device)

        yield link_mask, [plaq_mask]
        i += 1
