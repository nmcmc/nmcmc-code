# This is modified code from https://arxiv.org/abs/2101.08176 by M.S. Albergo et all.
"""Various normalizing flow utilities."""

import torch
from utils.calc_function import calc_function


class SimpleNormal:
    """Normal prior distribution"""

    def __init__(self, loc, var):
        self.dist = torch.distributions.normal.Normal(torch.flatten(loc), torch.flatten(var))
        self.shape = loc.shape

    def log_prob(self, x):
        logp = self.dist.log_prob(x.reshape(x.shape[0], -1))
        return torch.sum(logp, dim=1)

    def sample_n(self, batch_size):
        x = self.dist.sample((batch_size,))
        return x.reshape(batch_size, *self.shape)


def apply_flow(coupling_layers, z_in, logq_a):
    """Apply a normalizing flow to prior configurations.

    :param coupling_layers
    :param z_in: The input prior configurations
    :param logq_a: The log probability of the input prior configurations
    :return:    The transformed configurations and the log probability of the transformed configurations
    """
    logq = torch.zeros_like(logq_a)
    for layer in coupling_layers:
        try:
            z, logJ = layer.forward(z_in)
        except AssertionError:
            # Your debugging code goes here
            raise AssertionError
        logq -= logJ
        z_in = z
    return z, logq + logq_a


def apply_flow_to_prior(prior, coupling_layers, *, batch_size):
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
        generated configurations and the log probability of the generated configurations
    -------
    """

    z = prior.sample_n(batch_size)
    logq = prior.log_prob(z)
    return apply_flow(coupling_layers, z, logq)


def reverse_apply_flow(coupling_layers, phi, logq_a):
    """Apply the inverse of a normalizing flow to configurations.

    Parameters
    ----------
    coupling_layers
        normalizing flow layers
    phi
        configurations to transform
    logq_a
        log probability of the configurations

    Returns
        Transformed configurations and the log probability of the transformed configurations
    -------

    """
    logq = torch.zeros_like(logq_a)
    for layer in reversed(coupling_layers):
        try:
            phi, logJ = layer.reverse(phi)
        except AssertionError:
            # Your debugging code goes here
            raise AssertionError
        logq -= logJ
    return phi, logq + logq_a


def make_conv_net(*, in_channels, hidden_sizes, out_channels, kernel_size, use_final_tanh, dilation=1,
                  float_dtype=torch.float32):
    sizes = [in_channels] + hidden_sizes + [out_channels]
    net = []
    if isinstance(dilation, int):
        dilations = [dilation] * (len(hidden_sizes) + 1)
    else:
        dilations = dilation
    for i in range(len(sizes) - 1):
        net.append(
            torch.nn.Conv2d(
                sizes[i],
                sizes[i + 1],
                kernel_size,
                padding=dilations[i] * (kernel_size - 1) // 2,
                stride=1,
                padding_mode="circular",
                dilation=dilations[i],
                dtype=float_dtype
            )
        )
        if i != len(sizes) - 2:
            net.append(torch.nn.LeakyReLU())
        else:
            if use_final_tanh:
                net.append(torch.nn.Tanh())
    return torch.nn.Sequential(*net)


def sample(n_samples, batch_size, prior, layers):
    rem_size = n_samples
    samples = []
    log_q = []
    while rem_size > 0:
        with torch.no_grad():
            batch_length = min(rem_size, batch_size)
            x, logq = apply_flow_to_prior(prior, layers, batch_size=batch_length)

        samples.append(x.cpu())
        log_q.append(logq.cpu())

        rem_size -= batch_length

    return torch.cat(samples, 0), torch.cat(log_q, -1)


def calc_dkl(logp, logq):
    return (logq - logp).mean()


def compute_ess_lw(logw):
    log_ess = 2 * torch.logsumexp(logw, dim=0) - torch.logsumexp(2 * logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg


def compute_ess(logp, logq):
    logw = logp - logq
    return compute_ess_lw(logw)


def calc_action(cfgs, *, action, batch_size, device):
    return calc_function(cfgs, function=action, batch_size=batch_size, device=device)
