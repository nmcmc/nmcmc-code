#!/usr/bin/env python3
"""Memory usage analysis for Schwinger model

This script checks amount  memory allocated while evaluating a loss function for Schwinger model.
"""

import torch

if torch.cuda.is_available():
    torch_device = "cuda"
else:
    torch_device = "cpu"

import neumc
import neumc.physics.schwinger as sch


print(f"Running on PyTorch {torch.__version__}")


batch_size = 2**8
float_dtype = "float32"
loss = "REINFORCE"

loss_function = getattr(neumc.training.loss, f"{loss}_loss")

L = 8
lattice_shape = (L, L)
link_shape = (2, L, L)
beta = 1.0
kappa = 0.276

qed_action = sch.QEDAction(beta, kappa)

model_cfg = {
    "n_layers": 48,
    "masking": "2x1",
    "coupling": "cs",
    "nn": {
        "hidden_channels": [64, 64],
        "kernel_size": 3,
        "dilation": [1, 2, 3],
    },
    "n_knots": 9,
    "float_dtype": "float32",
    "lattice_shape": lattice_shape,
}

model = neumc.nf.u1_model_asm.assemble_model_from_dict(
    model_cfg, torch_device, verbose=1
)
layers = model["layers"]
prior = model["prior"]

z = prior.sample_n(batch_size=batch_size)
log_prob_z = prior.log_prob(z)

torch.cuda.reset_peak_memory_stats()

with torch.autograd.graph.saved_tensors_hooks(
    pack_hook := neumc.utils.profile.MemoryPackHook(), neumc.utils.profile.unpack_hook
):
    l, logq, logp = loss_function(
        z, log_prob_z, model=model, action=qed_action, use_amp=False
    )

print(
    f"{pack_hook.mem_u(b=30):.2f}GB memory used for storing {len(set(pack_hook.ptrs))} tensores in the DAG"
)
len(set(pack_hook.ptrs))

print(f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB allocated by CUDA")

l.backward()

print(
    f"{torch.cuda.max_memory_allocated() / 2 ** 30:.2f}GB allocated by CUDA after running backward()"
)
