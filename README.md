# Tutorial: Regression with PyTorch in High Energy Physics

This repository contains a walk-through example of how to implement a simple regression model using PyTorch in High Energy Physics (HEP). For this example, we will use the $t\bar{t}H$ process as a case study, and aim to predict the transverse momentum spectrum of the Higgs boson. We will focus on the semi-leptonic decay mode of the $t\bar{t}$ system, where one of the $W$-bosons decays leptonically and the other hadronically. The decay mode of the Higgs boson under consideration is $H \to b\bar{b}$, with a branching ratio of $\sim 58\%$.

- [Tutorial: Regression with PyTorch in High Energy Physics](#tutorial-regression-with-pytorch-in-high-energy-physics)
  - [Pre-requisites and Environment Setup](#pre-requisites-and-environment-setup)
  - [Further Reading](#further-reading)
  - [Resources](#resources)

A simple, exemplary, Feynman diagram of the process under consideration is shown below.

<div style="text-align: center">
  <img src=".assets/ttH-1l-tchan.png" alt="Feynman diagram" width="50%">
</div>

## Pre-requisites and Environment Setup

Firstly, you will want to clone (or fork) this repository.

```bash
git clone https://github.com/LeviSamuelEvans/pytorch-regression-tutorial.git
```

Then, you will want to create a virtual environment and install the dependencies.

```bash
python -m venv venv
source venv/bin/activate
```

```bash
pip install -r requirements.txt
```

Feel free to use whatever package manager you like; conda/mamba for example is very good at handling dependencies (and mamba is quick!). But, for the tutorial here just using pip and a python venv is sufficient. I also really like using containers for this kind of thing, but that is a topic for another tutorial! When you run the notebook, please choose the associated environment when you setup the notebook kernel.

To disable the enviroment, you can run `deactivate`.

You will need to also install the dataset we will be using. We use `git lfs` to handle the large file.

```bash
git lfs install
git lfs pull
```

Verify the dataset has been downloaded correctly:

```bash
ls data/
```

You should see the following files:

```bash
ttH_fullSim_dev4vec_150k.h5
```


## Further Reading
See [Theory](./tutorial/theory.md) for a brief overview of the theory behind the applied ML tutorial.

## Resources
Here is a list of amazing resources that I have found useful:

- Deep Learning tuning playbook: https://github.com/google-research/tuning_playbook#choosing-the-batch-size