# Neural Markov Chain Monte-Carlo

This repository contains python code implementing Neural Markov Chain Monte-Carlo for two-dimensional Schwinger model,
pure U(1)  gauge model and phi^4 model. To use this code you should install it as package `nmcmc`. The detailed
instructions are provided in the [Installation](#Installation) section.

Once the installation is complete you can start with notebooks in the `notebooks` directory or one of the scripts
in `scripts` directory e.g.

```shell
python ./scripts/schwinger.py 
```

This will start training the Schwinger model on a 8x8 lattice with beta=2.0 and kappa= 0.276 using the REINFORCE
estimator. The training will be on 'cuda' device. More information can be found [in here](scripts/README.md). The whole
script can take something from 8 to 20 minutes depending on the type of GPU you have. You can run this script on CPU, but
it is not recommended.

## Installation

To use Python modules from this repository you should install them as the package `nmcmc`.
You will need `Python>=3.10` and `pip3>= 21.3`. So please upgrade your python installation beforehand.
As always it is best to
create a separate virtual Python environment for the project. Probably the easiest way is to use `venv`, but you may
use `conda/mamba` if you are familiar with it. Let's assume that you
will call this environment `nmcmc-env`.

### Linux/Unix

In your terminal go to the directory where you want to store this environment and type

```shell
python -m venv nmcmc-env
```

to create environment and then activate it by running

```shell
. ./nmcmc-env/bin/activate
```

Now you can start installing required packages. Go to the root directory of this repository and type

```shell
pip install -e nmcmc
```

followed by

```shell
pip install -r nmcmc/requirements.txt
```

to install required packages (at the moment only `numpy` and `scipy`). The `-e` option install the package in so-called
_developer_ mode. This allows you to make changes to the code inside package that will be immediately visible i.e.
without reloading the package.

If you would like to use notebooks provided in the repository you will also need to install `jupyterlab`, `jupytext`
and `matplotlib`

```shell
pip install jupyterlab jupytext matplotlib
```

Of course, you can use some other environment to view/run notebooks like _e.g._ `Visual Studio Code` or `PyCharm`, but
you
will probably still need `jupytext` to transform notebooks that are stored in this repository as R Markdown (.Rmd)
files. That can be done via the command line like this

```shell
jupytext some_notebook.Rmd --to notebook 
```

That will create a `some_notebook.ipynb` file. In jupyter lab this happens automatically when you have
the `jupytext.toml` in you directory (it is provided in this repository). Then it is enough to click on the file name
and
choose `Open With>Notebook` option from the menu.

And finally you have to install `PyTorch`. I am not giving any instructions for that, as it depends on your software and
hardware. You can obtain instruction directly [from their site](https://pytorch.org/get-started/locally/).

After finishing work you can deactivate the environment by running

```shell
deactivate
```

If something goes very wrong you can always delete the whole environment by running

```shell
rm -rf  nmcmc-env
```

### Windows

On Windows you essentially follow up same steps in command prompt or PowerShell, but you activate the environment by
running the activate script

```shell
nmcmc-env\Scripts\activate
```