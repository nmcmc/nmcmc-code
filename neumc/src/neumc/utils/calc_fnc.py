import torch
import numpy as np
from scipy.special import logsumexp as logsumexp_np


def calc_function(cfgs, *, function, batch_size, device, **kwargs):
    """
    Calculate a function on a set of configurations in batches of batch_size at a time.


    Parameters
    ----------
    cfgs
        input configurations
    function
        function to apply to the configurations
    batch_size
        number of configurations to process at a time
    device
        device to run the calculations on
    kwargs
        additional arguments to pass to the function

    Returns
    -------
        results of the function applied to the configurations
    """
    rem_size = len(cfgs)

    obs = []
    i = 0
    while rem_size > 0:
        with torch.no_grad():
            batch_length = min(rem_size, batch_size)
            o = function(cfgs[i : i + batch_length].to(device), **kwargs)
            obs.append(o.cpu())
            i += batch_length
            rem_size -= batch_length

    return torch.cat(obs, -1)


def calc_dkl(logp, logq):
    return (logq - logp).mean()


def compute_ess_lw(logw):
    """
    Compute the effective sample size given the log of importance weights.
    Parameters
    ----------
    logw
        log of importance weights

    Returns
    -------
        effective sample size
    """
    log_ess = 2 * torch.logsumexp(logw, dim=0) - torch.logsumexp(2 * logw, dim=0)
    ess_per_cfg = torch.exp(log_ess) / len(logw)
    return ess_per_cfg


def compute_ess_lw_np(logw):
    """
    Compute the effective sample size given the log of importance weights.
    Parameters
    ----------
    logw
        log of importance weights

    Returns
    -------
        effective sample size
    """
    log_ess = 2 * logsumexp_np(logw, 0) - logsumexp_np(2 * logw, 0)
    ess_per_cfg = np.exp(log_ess) / len(logw)
    return ess_per_cfg


def compute_ess(logp, logq):
    """
    Compute the effective sample size given the log probabilities of the target and proposal distributions.

    Parameters
    ----------
    logp
        log probability of the target distribution
    logq
        log probability of the proposal distribution

    Returns
    -------
        effective sample size
    """
    logw = logp - logq
    return compute_ess_lw(logw)


def calc_action(cfgs, *, action, batch_size, device):
    return calc_function(cfgs, function=action, batch_size=batch_size, device=device)
