# Scripts

This directory contains scripts that can be used to train and analyze models.

## Training scripts

Those scripts can be used to train different models. They are all very similar and contain lots of duplicated code that
was left on purpose as to allow them to be further developed separately. They share a common set of command line options
that can be listed by running
them with `-h` option. The most important options are: `n-eras`, `n-epochs-per-era`, `batch-size` and  `n-batches`. The
training is performed in `n-eras` each of which consists of `n-epochs-per-era` epochs. Each epoch consists of one
gradients step
where gradient is calculated on  `n-batches` batches of configurations of size `batch-size`. Those two last parameters
are crucial for the efficiency. First of all for the REINFORCE estimator the total number of
configurations `n_batches x batch_size` cannot be too small because of its high initial variance. We recommend at least
1024 . Second, one should choose the biggest `batch-size` that does not cause the out of memory error on your GPU.
Scripts periodically save the model to a file. The file name is constructed from the name of the loss function and the
lattice size. For example `schwinger_REINFORCE_4x4.zip` is the name of the file for the Schwinger model trained with the
REINFORCE estimator on the 4x4 lattice. Any existing file of this name will be overwritten. To choose another file name
you can rename the file after running the script or modify the script itself. Each script can be run without providing
any arguments. In this case it will use default values for all parameters. The default values are listed in the help
message.

All scripts assume that a CUDA enabled GPU is available. If you want to run them on CPU you need to
pass    `--device cpu` option, but it is not recommended for anything except maybe small phi4 models. If more than one
CUDA device is available you can choose which one to use by passing `--device cuda:device_id` option.

### schwinger

This script illustrates how to set up training for the Schwinger model.

```shell
python ./scripts/schwinger.py --n-eras=10 --n-epochs-per-era=100 --batch-size=512 --n-batches=2 --loss REINFORCE 
```

also provides a possibility to continue training from checkpoint

```shell
python ./scripts/schwinger.py --continue schwinger_pytorch_REINFORCE_4x4.zip --n-eras=10 --n-epochs-per-era=100
```

### u1

Training pure gauge U(1) model. Offers a choice of two equivariant layers and two coupling layers.

```shell
python ./scripts/u1.py  --loss REINFORCE --equiv '2x1' --coupling cs 
```

### phi4

Training phi4 model.

## Analysis scripts

### nmcmc

This script takes as an
argument the name of the file with the model trained by the `schwinger.py` script and loads the model. Next it uses it
to generate new configurations which are then used in the Neural Markov Chain Monte-Carlo (NMCMC) algorithm.

```shell
python ./scripts/nmcmc.py  REINFORCE_4x4.zip
```

## Profiling scripts

This is a collection of script that can be used to measure some properties of the models notably timings and memory
usage.

### timings

This scripts performs detailed timing measurements of the loss functions for the Schwinger model.

### memory

This script measures the memory allocated during a single call to loss function for the Schwinger model.

### dag

This script measures various properties of the DAG created by the loss function for the Schwinger model.





