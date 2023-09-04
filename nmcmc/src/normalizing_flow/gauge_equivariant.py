"""Collection of utility functions to make gauge equivariant normalizing flows model."""

__all__ = ['make_schwinger_model', 'make_u1_rs_model', 'make_u1_nc_model', 'make_u1_nc_model_2x1']

import inspect

import torch

import phys_models.U1 as u1
from normalizing_flow import circular_splines_equivariant_couplings as cs
from normalizing_flow import non_compact_projection as nc


def _print_args(argsinfo):
    for key, val in argsinfo.locals.items():
        print(f"{key} =  {val}")


def make_schwinger_model(*, lattice_shape, n_knots, n_layers, hidden_sizes, kernel_size, dilation, float_dtype,
                         device, verbose=0):
    if verbose > 0:
        print(f"Making Gauge equivariant rational splines model")
        argsinfo = inspect.getargvalues(inspect.currentframe())
        _print_args(argsinfo)
    link_shape = (2, *lattice_shape)
    prior = u1.MultivariateUniform(torch.zeros(link_shape), 2 * torch.pi * torch.ones(link_shape), device=device)

    layers = cs.make_u1_equiv_layers_rs_with_2x1_loops(n_knots=n_knots, lattice_shape=lattice_shape, n_layers=n_layers,
                                                       hidden_sizes=hidden_sizes, kernel_size=kernel_size,
                                                       dilation=dilation, float_dtype=float_dtype,
                                                       device=device)

    u1.set_weights(layers)
    layers.to(device)  # probably redundant
    return {'layers': layers, 'prior': prior}


def make_u1_rs_model(*, lattice_shape, n_knots, n_layers, hidden_sizes, kernel_size, dilation, float_dtype,
                     device, verbose=0):
    if verbose > 0:
        print(f"Making Gauge equivariant rational splines model")
        argsinfo = inspect.getargvalues(inspect.currentframe())
        _print_args(argsinfo)

    link_shape = (2, *lattice_shape)
    prior = u1.MultivariateUniform(torch.zeros(link_shape), 2 * torch.pi * torch.ones(link_shape), device=device)

    layers = cs.make_u1_equiv_layers_rs(n_knots=n_knots, lattice_shape=lattice_shape, n_layers=n_layers,
                                        hidden_sizes=hidden_sizes, kernel_size=kernel_size,
                                        dilation=dilation, float_dtype=float_dtype,
                                        device=device)

    u1.set_weights(layers)
    layers.to(device)  # probably redundant
    return {'layers': layers, 'prior': prior}


def make_u1_nc_model(*, type='plaq', lattice_shape, n_mixture_comps, n_layers, hidden_sizes, kernel_size, dilation,
                     float_dtype,
                     device, verbose=0):
    if verbose > 0:
        print(f"Making Gauge equivariant non-compact projection model")
        argsinfo = inspect.getargvalues(inspect.currentframe())
        _print_args(argsinfo)

    link_shape = (2, *lattice_shape)
    prior = u1.MultivariateUniform(torch.zeros(link_shape), 2 * torch.pi * torch.ones(link_shape), device=device)

    layers = nc.make_u1_equiv_layers(type=type, lattice_shape=lattice_shape, n_layers=n_layers,
                                     n_mixture_comps=n_mixture_comps,
                                     hidden_sizes=hidden_sizes, kernel_size=kernel_size,
                                     dilation=dilation, float_dtype=float_dtype,
                                     device=device)

    u1.set_weights(layers)
    layers.to(device)  # probably redundant
    return {'layers': layers, 'prior': prior}


def make_u1_nc_model_2x1(**kwargs):
    if 'type' in   kwargs:
        del kwargs['type']
    return make_u1_nc_model(type='sch_2x1', **kwargs)
