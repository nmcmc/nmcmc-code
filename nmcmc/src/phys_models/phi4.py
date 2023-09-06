import numpy as np
import torch

class ScalarPhi4Action:
    def __init__(self, M2, lam) -> None:
        self.M2 = M2
        self.lam = lam
        self.inv_24 = 1.0/24.0

    def __call__(self, cfgs):
        action_density = (0.5*self.M2 +2.0)* cfgs ** 2 + self.inv_24*self.lam * cfgs ** 4
        Nd = len(cfgs.shape) - 1

        dims = range(1, Nd + 1)
        for mu in dims:

            action_density -=  cfgs * torch.roll(cfgs, -1, mu)
        return torch.sum(action_density, dim=tuple(dims))

def free_field_free_energy(L,M2):
    """
    Calculates the free energy of a free scalar field in 2D with mass squared M2 on a LxL lattice.

    :param L: lattice size.
    :param M2: float squared mass
    :return: free field free energy
    """
    q = np.arange(0, L)
    q_0, q_1 = np.meshgrid(q, q)
    k2_0 = 4 * np.sin(np.pi / L * q_0) ** 2
    k2_1 = 4 * np.sin(np.pi / L * q_1) ** 2

    return -0.5 * (L * L) * np.log(2 * np.pi) + 0.5 * np.log(k2_0 + k2_1 + M2).sum()

def mag2(cfgs):
    vol = torch.prod(torch.tensor(cfgs.shape[1:]))
    return torch.mean(
        cfgs.sum(
            dim=tuple(range(1, len(cfgs.shape)))
        )**2/vol
    )