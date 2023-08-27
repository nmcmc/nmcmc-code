import torch
import numpy as np

debug_info = {}


def bs(a, x):
    assert a.shape[1:] == x.shape[1:], "Wrong dimenensions for bs"
    N_b = a.shape[0]
    N_k = a.shape[1]
    N_s = a.shape[2:]

    device = a.device
    idc = torch.from_numpy(np.indices((N_b, *N_s))).to(device=device)
    bs.idc = idc
    L = torch.zeros((N_b, *N_s), dtype=torch.long, device=a.device)
    R = (N_k - 1) * torch.ones((N_b, *N_s), dtype=torch.long, device=a.device)
    depth = int(np.ceil(np.log2(N_k)))

    for i in range(depth):
        m = torch.div(L + R, 2, rounding_mode="floor")
        R = torch.where(x < a[(idc[0], m, *idc[1:])], m, R)
        L = torch.where(x >= a[(idc[0], m, *idc[1:])], m, L)
    return L


def make_bs(N_b, N_k, N_s, device):
    idc = torch.from_numpy(np.indices((N_b, *N_s))).to(dtype=torch.long, device=device)

    def bs(a, x):
        L = torch.zeros((N_b, *N_s), dtype=torch.long, device=a.device)
        R = (N_k - 1) * torch.ones((N_b, *N_s), dtype=torch.long, device=a.device)
        depth = int(np.ceil(np.log2(N_k)))

        for i in range(depth):
            m = torch.div(L + R, 2, rounding_mode="floor")
            R = torch.where(x < a[(idc[0], m, *idc[1:])], m, R)
            L = torch.where(x >= a[(idc[0], m, *idc[1:])], m, L)
        return L

    bs.idc = idc

    bs.N_b = N_b
    bs.N_k = N_k
    bs.N_s = N_s
    return bs


def make_circular_knots_array(w, h, d, device="cpu"):
    N_b = w.shape[0]
    N_k = w.shape[1] + 1
    N_s = w.shape[2:]

    debug_info["widths"] = w
    debug_info["heights"] = h

    assert torch.all(w > 0)
    assert torch.all(h > 0)
    assert torch.all(d > 0)

    x = torch.empty(N_b, N_k, *N_s, device=device)
    x[:, 0, ...] = 0.0
    x[:, 1:, ...] = torch.cumsum(w, 1) * torch.pi * 2
    x[:, -1, ...] = 2.0 * torch.pi

    y = torch.empty(N_b, N_k, *N_s, device=device)
    y[:, 0, ...] = 0.0
    y[:, 1:, ...] = torch.cumsum(h, 1) * torch.pi * 2
    y[:, -1, ...] = 2.0 * torch.pi

    s = torch.empty(N_b, N_k, *N_s, device=device)
    s[:, 1:, ...] = d
    s[:, 0, ...] = s[:, -1, ...]

    return x, y, s


def make_splines_array(x, y, d, bs):
    idc = bs.idc
    w = x[:, 1:, ...] - x[:, :-1, ...]
    s = (y[:, 1:, ...] - y[:, :-1, ...]) / w

    debug_info["s"] = s
    debug_info["w"] = w
    debug_info["x"] = x
    debug_info["y"] = y
    debug_info["d"] = d
    debug_info["bs"] = bs

    assert torch.all(w > 0)
    assert torch.all(s > 0)
    assert torch.all(d > 0)

    def gamma(k, y, s, d, xi):
        ki = (idc[0], k, *idc[1:])
        kip = (idc[0], k + 1, *idc[1:])
        # print("gamma")
        a = (y[kip] - y[ki]) * (s[ki] * xi * xi + d[ki] * xi * (1 - xi))
        b = s[ki] + (d[kip] + d[ki] - 2 * s[ki]) * xi * (1 - xi)
        return y[ki] + a / b

    def der(k, y, s, d, xi):
        # print("der")

        ki = (idc[0], k, *idc[1:])
        kip = (idc[0], k + 1, *idc[1:])
        numerator = (
                s[ki]
                * s[ki]
                * (d[kip] * xi * xi + 2 * s[ki] * xi * (1 - xi) + d[ki] * (1 - xi) * (1 - xi))
        )

        denominator = (s[ki] + (d[kip] + d[ki] - 2 * s[ki]) * xi * (1 - xi)) ** 2

        return numerator / denominator

    def spline(x_a):
        # print("spline ", x.shape)
        with torch.no_grad():
            k = bs(x, x_a)

        debug_info["x"] = x
        debug_info["x_a"] = x_a
        debug_info["bs"] = bs
        ki = (idc[0], k, *idc[1:])
        kip = (idc[0], k + 1, *idc[1:])
        debug_info['ki'] = ki
        debug_info['kip'] = kip

        assert torch.all(x_a >= x[ki]), "wrong bs output lower"
        assert torch.all(x_a < x[kip]), "wrong bs output upper"

        xi = (x_a - x[ki]) / w[ki]
        assert torch.all(xi >= 0)
        derivative = der(k, y, s, d, xi)
        # print("return")
        if torch.any(derivative < 0):
            print("non positive derivative ", derivative.shape)
            print(torch.sum(derivative < 0))
        return gamma(k, y, s, d, xi), torch.log(derivative)

    def inverse(y_a):
        assert torch.all(y_a < 2 * torch.pi)
        with torch.no_grad():
            k = bs(y, y_a)

        ki = (idc[0], k, *idc[1:])
        kip = (idc[0], k + 1, *idc[1:])

        a = (y[kip] - y[ki]) * (s[ki] - d[ki]) + (y_a - y[ki]) * (d[kip] + d[ki] - 2 * s[ki])

        b = (y[kip] - y[ki]) * d[ki] - (y_a - y[ki]) * (d[kip] + d[ki] - 2 * s[ki])
        c = -s[ki] * (y_a - y[ki])

        xi = 2 * c / (-b - torch.sqrt(b * b - 4 * a * c))
        derivative = der(k, y, s, d, xi)
        u = xi * w[ki] + x[ki]
        assert torch.all(u < 2 * torch.pi)
        return u, -torch.log(derivative)

    return spline, inverse
