# Denoising Difussion Samplers

This is the repository contains the implementation for the *Denoising Diffusion Samplers* paper published at ICLR 2023. 

## Installation Instructions

This code uses python 3.10. To install we recommend running:

```
$ pip install -e .
```

in the root directory of the repository and within a conda environment. Once this is complete please install jaxline and the annealed flow montecarlo repos by following the installation instructions at:

* https://github.com/deepmind/jaxline
* https://github.com/deepmind/annealed_flow_transport

## Results Dictionary

For quick comparison we stored all of our raw unproccessed results in a dictionary named `results_full` that can be found at `denoising_diffusion_samplers/dds_data/results/dds_results.py` . Similarly our SMC baselines can be found at `denoising_diffusion_samplers/dds_data/results/smc_results.py` . For each method, task, steps combination there are 30 different estimates for ln Z using different random seeds.

### Results Notebook

A notebook that summarises/visualizes all the results form the results dictionary can be found at `denoising_diffusion_samplers/notebooks/line_plots_results.ipynb`. This is the plot we use in the main section of our paper.


## Hyperparameters Dictionary

In the notebook folders we have prepared notebooks that immediately reproduce the results for the Funnel and ION tasks at a given number of steps. In order to train from scratch and reproduce the results the correct tuned hpyperparameters must be used for each method. To facilitate this we have provided a full dictionary containing hyperparaeters for each method, tasks, steps combination. This dictionary can be found at `denoising_diffusion_samplers/opt_hyperparams.py`. 

## Usage

In order to see examples how to train / run, please read any of the following notebooks:


* `denoising_diffusion_samplers/notebooks/Simple Funnel Test Run.ipynb`
* `denoising_diffusion_samplers/notebooks/Mixture Well.ipynb`
* `denoising_diffusion_samplers/notebooks/Logistic Regression.ipynb`


## Giving Credit


If you use this code in your work, please cite the corresponding paper. 

```
@inproceedings{
vargas2023denoising,
title={Denoising Diffusion Samplers},
author={Francisco Vargas and Will Sussman Grathwohl and Arnaud Doucet},
booktitle={International Conference on Learning Representations},
year={2023},
url={https://openreview.net/forum?id=8pvnfTAbu1f}
}
```
