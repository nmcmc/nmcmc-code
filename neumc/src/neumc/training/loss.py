import torch
from torch.amp import autocast

import neumc.utils as utils
import neumc.nf.flow as nf


def REINFORCE_loss(z_a, log_prob_z_a, *, model, action, use_amp):
    layers, prior = model["layers"], model["prior"]
    with torch.no_grad():
        with autocast("cuda", enabled=use_amp):
            phi, logq = nf.apply_layers(layers, z_a, log_prob_z_a)
            logp = -action(phi)
            signal = logq - logp

    with autocast("cuda", enabled=use_amp):
        z, log_q_phi = nf.reverse_apply_layers(
            layers, phi, torch.zeros_like(log_prob_z_a, device=phi.device)
        )
        prob_z = prior.log_prob(z)
        log_q_phi = prob_z - log_q_phi
        loss = torch.mean(log_q_phi * (signal - signal.mean()))

    return loss, logq, logp


def rt_loss(z, log_prob_z, *, model, action, use_amp):
    layers = model["layers"]

    with autocast("cuda", enabled=use_amp):
        x, logq = nf.apply_layers(layers, z, log_prob_z)

        logp = -action(x)
        loss = utils.calc_fnc.calc_dkl(logp, logq)

        return loss, logq.detach(), logp.detach()


def path_gradient_loss(z, log_prob_z, *, model, action, use_amp):
    device = z.device
    zeros = torch.zeros(len(z), device=device)
    layers, prior = model["layers"], model["prior"]
    nf.detach(layers)
    with torch.no_grad():
        fi, J = nf.apply_layers(layers, z, zeros)
    fi.requires_grad_(True)
    zp, log_q_ = nf.reverse_apply_layers(layers, fi, zeros)
    prob_zp = prior.log_prob(zp)
    log_q = prob_zp - log_q_
    log_q.backward(torch.ones_like(log_q_))
    G = fi.grad.data
    nf.attach(layers)
    fi2, _ = nf.apply_layers(layers, z, zeros)
    log_p = -action(fi2)
    axes = tuple(range(1, len(G.shape)))
    contr = torch.sum(fi2 * G, dim=axes)
    loss = torch.mean(contr - log_p)
    return loss, log_q.detach(), log_p.detach()
