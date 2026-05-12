# Transformer-UQ (Simulated PPD Pretraining)

Code for the ICML 2026 submission: **“Transformers Can Learn Posterior Predictive Distributions In-Context.”**
This repo trains a PFN on synthetic tasks to output discretized posterior predictive distributions.

## Code structure
- `data.py` — contains codes to simulate from synthetic task distributions (BLR, RBF)
- `model.py` — contains the main architecture consisting of attention block and MLP head
- `train.py` — contains codes for training with leanring rate schedules

## Tasks
- `blr`: Bayesian linear regression with linear kernel
- `rbf`: GP regression with RBF kernel 

## Install
```bash
pip install -r requirements.txt
```

## Quick Start (pretraining)
BLR: (unnormallized attention)
```bash
python train.py --task blr --d 5 --L 32 --n_mode mixture --n_min 128 --n_max 512 --mode learnable --C 256 --steps 20000 --lr 1e-4 --seed 0
```
RBF (unnormallized attention)
```bash
python train.py --task rbf --d 16 --L 32 --n_mode mixture --n_min 128 --n_max 512 --mode learnable --C 256 --steps 100000 --lr 1e-4 --seed 0
```
RBF (normallized attention)
```bash
python train.py --task rbf --d 16 --L 32 --n_mode mixture --n_min 128 --n_max 512 --mode learnable --C 256 --steps 100000 --lr 1e-4 --seed 0 --normalize 1 --normdecay 1
```

## Outputs / checkpoints
Training outputs are saved under runs/ (auto-named). Logs are written to runs/<run>/log.csv.
To resume from a checkpoint:
```bash
python train.py --resume <path_to_checkpoint>
```

## Reproducing paper results
To reproduce Figure 1,
```bash
python train_walker.py
python walker_comparison.py
```
To reproduce Figure 2,
```bash
./run_fig1_blr.sh
./run_fig1_rbf.sh
python figure1.py
```
To reproduce Figure 3,
```bash
python figure4.py
```
To reproduce Figure 4,
```bash
./run_fig2_rbf.sh
python figure3.py
```
To reproduce Figure 5-14,
```bash
./run_grid_rbf_learnable_d4.sh
./run_grid_rbf_learnable_d8.sh
./run_grid_rbf_learnable_d16.sh
python figure2.py
```
