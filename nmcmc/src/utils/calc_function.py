import torch


def calc_function(cfgs, *, function, batch_size, device, **kwargs):
    rem_size = len(cfgs)

    obs = []
    i = 0
    while rem_size > 0:
        with torch.no_grad():
            batch_length = min(rem_size, batch_size)
            o = function(cfgs[i:i + batch_length].to(device), **kwargs)
            obs.append(o.cpu())
            i += batch_length
            rem_size -= batch_length

    return torch.cat(obs, -1)


