# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.
"""Various normalizing flow utilities."""

import torch

import neumc


class SimpleNormal:
    """
    Simple normal distribution with diagonal covariance matrix.

    Parameters
    ----------
    loc
        mean of the distribution
    var
        variance of the distribution
    """

    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(
            torch.flatten(loc), torch.flatten(var)
        )
        self.shape = loc.shape

    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)

    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)


def apply_layers(coupling_layers, z, log_prob_z=None):
    """
    Apply successive normalizing flow layers to prior configurations.

    Assumes that each layer transforms the input configurations z and returns the transformed configurations and
    the log Jacobian of the transformation.
    In the formula for final probability of the output configuration, we need the inverse of this Jacobian,
    so we subtract it from the input log probability to yield the final probability of the transformed configurations.
    If the input log probability parameter log_prob_z is not provided function returns the log of the inverse Jacobian
    determinant of the transformation.

    Parameters
    ----------
    coupling_layers
        A list of normalizing flow layers
    z
        configurations to transform
    log_prob_z
        prior log probability of the input configurations

    Returns
    -------
    z
        Transformed configurations
    log_q
        log probability of the transformed configurations or the log of the inverse Jacobian determinant
        if log_prob_z is not provided
    """

    log_q = torch.zeros(z.shape[0], device=z.device)

    for layer in coupling_layers:
        try:
            z, log_J = layer.forward(z)
        except AssertionError as ae:
            # Your debugging code goes here
            print(ae)
            raise AssertionError
        log_q -= log_J
    if log_prob_z is not None:
        log_q += log_prob_z
    return z, log_q


def apply_layers_to_prior(prior, coupling_layers, *, batch_size):
    """Generates prior configurations and applies a normalizing flow to them.

    Parameters
    ----------
    prior
        distribution to sample prior configurations from
    coupling_layers
        normalizing flow layers
    batch_size
        number of prior configurations to generate

    Returns
    -------
        generated configurations and the log probability of the generated configurations
    """

    z = prior.sample_n(batch_size)
    logq = prior.log_prob(z)
    return apply_layers(coupling_layers, z, logq)


def reverse_apply_layers(coupling_layers, phi, log_prob_phi=None):
    """Apply the inverse of a normalizing flow to configurations.

    Assumes that each layer transforms the input configurations phi using the `reverse` method
    and returns the transformed configurations and the log Jacobian of this transformation.
    In the formula for final probability of the output configuration, we need the inverse of this Jacobian,
    so we subtract it from the input log probability to yield the final probability of the transformed configurations.
    If the input log probability parameter log_prob_phi is not provided, the function returns
    the log of the inverse Jacobian determinant of the transformation.

    Parameters
    ----------
    coupling_layers
        normalizing flow layers
    phi
        configurations to transform
    log_prob_phi
        log probability of the input configurations

    Returns
    -------
    Transformed configurations and the log of inverse Jacobian deteminant of the
    transformation plus the log probability of the input configuration.
    """
    log_q = torch.zeros(phi.shape[0], device=phi.device)
    for layer in reversed(coupling_layers):
        try:
            phi, log_J = layer.reverse(phi)
        except AssertionError as ae:
            # Your debugging code goes here
            print(ae)
            raise AssertionError
        log_q -= log_J
    if log_prob_phi is not None:
        log_q += log_prob_phi
    return phi, log_q


def sample(n_samples, batch_size, prior, layers):
    """
    Sample configurations from a normalizing flow model in batches of batch_size at a time.
    Parameters
    ----------
    n_samples
        number of configurations to sample
    batch_size
        number of configurations to sample at a time
    prior
        distribution to sample prior configurations from
    layers
        normalizing flow layers

    Returns
    -------
        samples and the log probability of the samples
    """
    rem_size = n_samples
    samples = []
    log_q = []
    while rem_size > 0:
        with torch.no_grad():
            batch_length = min(rem_size, batch_size)
            x, logq = apply_layers_to_prior(prior, layers, batch_size=batch_length)

        samples.append(x.cpu())
        log_q.append(logq.cpu())

        rem_size -= batch_length

    return torch.cat(samples, 0), torch.cat(log_q, -1)


def log_prob(x, prior, layers):
    with torch.no_grad():
        z, log_q_phi = neumc.nf.flow.reverse_apply_layers(
            layers, x, torch.zeros((x.shape[0],), device=x.device)
        )
        prob_z = prior.log_prob(z)
        return prob_z - log_q_phi


def requires_grad(model, on=True):
    """Set requires_grad attribute on all parameters of a model to on."""

    for p in model.parameters():
        p.requires_grad = on


def detach(model):
    """Detach all parameters of a model."""
    requires_grad(model, False)


def attach(model):
    """Attach all parameters of a model."""
    requires_grad(model, True)
