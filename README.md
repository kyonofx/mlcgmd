# Learning to Simulate Time-integrated Coarse-grained Molecular Dynamics with Multi-scale Graph Networks [TMLR 2023]

<p align="center">
<img src="assets/chain.gif" width="300">
</p>

This codebase implements multi-scale GNN simulators for time-integrated CGMD, without using force/energy! This implementation was tested under `Ubuntu 18.04`, `Python 3.8`, `PyTorch 1.11`, and `CUDA 11.3`. Versions of all dependencies can be found in `env.yml`.

<p align="center">
  <img src="assets/model.png" /> 
</p>

[[Paper]](https://openreview.net/forum?id=y8RZoPjEUl) [[Website]](https://xiangfu.co/mlcgmd) [[Video]](https://www.youtube.com/watch?v=l3aGVjQezsc)

if you find this code useful, please consider reference in your paper:

```
@article{
fu2023simulate,
title={Simulate Time-integrated Coarse-grained Molecular Dynamics with Multi-scale Graph Networks},
author={Xiang Fu and Tian Xie and Nathan J. Rebello and Bradley Olsen and Tommi S. Jaakkola},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2023},
url={https://openreview.net/forum?id=y8RZoPjEUl},
note={}
}
``` 

## Pretrained model checkpoints

[single-chain CG polymer (param count 1.6M)](./ckpts/chain)
[solid polymer electrolytes (param count 1.6M)](./ckpts/battery)


## Installation
Create a conda environment with the required dependencies. This may take a few minutes.

```
conda env create -f env.yml
```

Activate the conda environment with:

```
conda activate mlcgmd
```

Then install `graphwm` (stands for graph world models) as a package:

```
pip install -e ./
```

## Prepare the dataset

Our single-chain CG polymer dataset is available from Zenodo.

[single-chain CG polymer dataset](https://zenodo.org/record/6764836#.YrqHNuxKjzd)

The solid polymer electrolyte dataset is available through [here](https://arxiv.org/abs/2208.01692).

## Configure environment variables

Before running training/evaluation of the GNN simulator, make a copy of the `.env.template` file and rename it to `.env`. Modify the following environment variables in `.env`, and copy it to `mlcgmd/graphwm/.env`.

- `PROJECT_ROOT`: path to the folder that contains this repo
- `CHAIN_DATASET_DIR`: path to the single-chain polymer training dataset (50k $\tau$)
- `BAT_DATASET_DIR`: path to the battery training dataset (5 ns)
- `CHAIN_TEST_DATASET_DIR`: "/scratch/xiangfu/polymer_test" (used as initialization for testing)
- `BAT_TEST_DATASET_DIR`: path to the battery evaluation dataset (50 ns)
- `MODEL_DIR`: path to save model checkpoints

## Logging with Weights and Biases (`wandb`)

We recommend logging with `wandb` and it is used by default. You need to have a wandb account and log in with `wandb init`. More details at [https://wandb.ai/](https://wandb.ai/).

## Train a CGMD simulator

The training configurations, including default hyperparameters can be found at [graphwm/conf](./graphwm/conf). These hyperparameters produce the results reported in our paper, but may not be optimal as we did not do extensive tuning. We trained all models with a single GPU and it takes ~1 day for the single-chain polymer dataset and 7-10 days for the battery dataset. Multi-GPU training is available (cf. [Tips](https://github.com/kyonofx/mlcgmd/tree/main#tips)) and will likely reduce training time.

Train a model with the [default configurations for the single-chain polymer dataset](./graphwm/conf/train.yaml) with the command:

```
python train.py
```

For the [battery dataset](./graphwm/conf/train_battery.yaml), use:

```
python train.py --config-name train_battery
```

We use `hydra` for config management. Command-line argument can be passed in conveniently. For example, if you want to a higher radius cut-off of `9.0`, with the battery dataset, simply do:

```
python train.py --config-name train_battery model.radius=9
```

Find out more about hydra at [https://hydra.cc/docs/intro/](https://hydra.cc/docs/intro/).

## Simulation using the learned simulator

With a trained model saved at `MODEL_DIR/chain_gns` (or change the `model_dir` argument in the evaluation config file), run simulation for the [single-chain polymer dataset](./graphwm/conf/eval.yaml) with the command:

```
python eval.py
```

For the battery dataset, run:

```
python eval.py --config-name eval_battery
```

Note that the simulation code assumes your model is saved as `{data.name}_{model.name}*`. The rollout trajectories are saved as a torch pickle file. Simulation efficiency is maximized when using a large batch size to parallelize the simulation of many systems on a single GPU. Simulating all 40 testing class-II polymers for 5M τ using a single RTX 2080 Ti GPU takes roughly 2.6 hours. Simulating all 50 testing batteries for 50 ns using one single RTX 2080 Ti GPU takes roughly 4.6 hours.

The `ld_kwargs` in the config file controls the inference process of the score-based refinement module. They are only used with the `PnR` model class.

## Tips

- Training CGMD simulators is data I/O intensive. Training speed will be greatly improved with a faster file system. For example, local drive is usually a lot faster than NFS/AFS. 
- The hyperparameter `model.cg_level` controls how many atoms are grouped into a coarse-grained bead. We use METIS for coarse-graining -- this algorithm tries to make the number of atoms assigned to each CG-bead equal. But this may not be achieved as atoms not connected by a chemical bond are never grouped together.  If `model.cg_level=1`, coarse-graining is turned off.
- multi-gpu training can be turned on by setting `train.pl_trainer.gpus=X`, where `X` is the number of GPUs.
- The hyperparameter `model.dilation` controls the time-integration step. It specifies the number of **recorded steps** that the ML simulator predicts over in a single step. More information about the length of recorded steps is in the next section.

## More about the datasets 

The single-chain coarse-grained polymer in implicit solvent dataset is adapted from the paper: [Targeted sequence design within the coarse-grained polymer genome](https://www.science.org/doi/10.1126/sciadv.abc6216), and the battery dataset is adapted from the paper: [Accelerating amorphous polymer electrolyte screening by learning to reduce errors in molecular dynamics simulated properties](https://arxiv.org/abs/2101.05339). Please find the simulation details of the datasets in these papers, and consider citing the respective papers if you use the datasets. 

The recording frequency for the single-chain polymer is 5 τ. for the training set and 500 τ for the test set. The timestep used in the LAMMPS simulation is 0.01 τ. Our default config uses `dilation=1`, so one step of our learned simulator is 5 τ, which is as long as 500 steps in the LAMMPS simulation.

The recording frequency for the battery dataset is 2 ps for both the training and the test sets. The integrator used in the LAMMPS simulation is a rRESPA multi-timescale integrator with an outer timestep of 2 fs for non-bonded interactions, and an inner timestep of 0.5 fs. Our default config uses `dilation=100`, so one step of our learned simulator is 0.2 ns, which is as long as $10^5$ steps in the LAMMPS simulation.

The orginal MD trajectories were simulated using [LAMMPS](https://www.lammps.org). Under [graphwm/preprocess](./graphwm/preprocess) you can find the scripts for preprocessing the raw LAMMPS dump to the `.h5` files that are used for our learned simulators. To use the preprocessing functionality, `mdtraj` needs to be installed through: `pip install mdtraj`.

## Related repos

- [nn-template](https://github.com/grok-ai/nn-template)
- [DeepMind implementation of GNS](https://github.com/deepmind/deepmind-research/tree/master/learning_to_simulate)
- [PyG](https://github.com/pyg-team/pytorch_geometric)
