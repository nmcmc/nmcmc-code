import numpy as np
import torch


def block_mean(sample, block_size):
    n_blocks = len(sample) // block_size
    if n_blocks == 0:
        raise RuntimeError("sample shorter then block_size")
    sample = sample.reshape(n_blocks, block_size, *sample.shape[1:])
    return sample.mean(1)


def ac(series, history=100):
    mu = np.mean(series)
    var = np.mean((series - mu) * (series - mu))
    out = np.empty(history)
    for i in range(history):
        out[i] = np.mean((series[:-history] - mu) * (series[i: i - history] - mu))
    return out / var


def ac_and_tau_int(series, c=10, maxlen=200):
    mu = np.mean(series)
    var = np.mean((series - mu) * (series - mu))
    out = [1.0]
    tau_int = 0.5
    for t in range(1, maxlen):
        cor = np.mean((series[:-t] - mu) * (series[t:] - mu)) / var
        tau_int += cor
        out.append(cor)
        if t > c * tau_int:
            break
    return tau_int, np.asarray(out)


def list_mean(data, mean=0.0):
    mean = mean
    size = 0
    for t in data:
        mean += t.sum()
        size += t.nelement()

    return mean / size


def bootstrapf(f, x, *, n_samples, binsize):
    n_bins = len(x) // binsize
    xb = np.stack(np.array_split(x, n_bins), axis=0)

    boots = []

    for i in range(n_samples):
        bsample = np.resize(xb[np.random.choice(n_bins, n_bins)], (n_bins * binsize, *x.shape[1:]))
        boots.append(f(bsample))
    return np.mean(boots, 0), np.std(boots, 0)


def bootstrap(x, *, n_samples, binsize):
    return bootstrapf(lambda x: np.mean(x, 0), x, n_samples=n_samples, binsize=binsize)


def torch_bootstrapf(f, x, *, n_samples, binsize):
    n_bins = len(x) // binsize
    xb = torch.stack(torch.tensor_split(x, n_bins), dim=0)

    boots = []

    for i in range(n_samples):
        idx = torch.from_numpy(np.random.choice(n_bins, n_bins))
        bsample = xb[idx].view(
            (n_bins * binsize, *x.shape[1:])
        )
        boots.append(f(bsample))
    boots = torch.stack(boots, dim=0)
    return torch.mean(boots, 0), torch.std(boots, 0)


def torch_bootstrapo(obs, x, *, n_samples, binsize, logweights=None):
    obs_ = obs(x)
    n_bins = len(x) // binsize
    xb = torch.stack(torch.tensor_split(obs_, n_bins), dim=0)

    boots = []
    if logweights is not None:
        weights = torch.exp(logweights - torch.max(logweights))
        bweights = torch.stack(torch.tensor_split(weights, n_bins), dim=0)

    for i in range(n_samples):
        idx = torch.from_numpy(np.random.choice(n_bins, n_bins))
        bsample = xb[idx].view(
            (n_bins * binsize, *obs_.shape[1:])
        )
        if logweights is not None:
            bwsample = bweights[idx].ravel()
            norm = torch.mean(bwsample)
            boots.append(torch.mean(bwsample * bsample / norm, 0))
        else:
            boots.append(torch.mean(bsample, 0))
    boots = torch.stack(boots, dim=0)
    return torch.mean(boots, 0), torch.std(boots, 0)
