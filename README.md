# Cont-Conf-HGNN Experiments

This repository contains hypergraph neural network training scripts for conformal classification experiments on Walmart Trips, Congress Bills, House Bills, and DBLP coauthorship data.

## Data

All datasets live under `dataset/`. The V2 scripts load the dataset selected by their parser defaults, so the commands below run the intended dataset without extra arguments.

## Environment

Create or update the conda environment from `environment.yml`:

```bash
conda env create -f environment.yml
conda activate graph_stuff
```

The environment now includes `wandb` in the pip dependencies for sweep-based hyperparameter tuning. Before running online W&B sweeps, authenticate once:

```bash
wandb login
```

## Training

Run an individual V2 experiment with:

```bash
python trainV2_walmart.py
python trainV2_congress.py
python trainV2_house_bills.py
python trainV2_DBLP_CC.py
```

Each V2 script keeps `--num_runs` defaulted to `20`. The baseline `result_this_run['gnn']` conformal metrics have been removed; corrected conformal HGNN metrics are reported under:

```python
result_this_run['cont_conf_hgnn']
```

Key conformal-method arguments are exposed in argparse instead of being hardcoded, including:

```text
--confgnn_lr
--confgnn_weight_decay
--confgnn_dropout
--confnn_hidden_dim
--confgnn_num_layers
--tau
--target_size
--size_loss_weight
--temperature
--aug_ratio
--edge_topk
--warmup_epochs
--coverage_warmup_epochs
--contrastive_loss_weight
--late_contrastive_loss_weight
--non_conftr_size_loss_weight
--raps_lam_reg
--raps_k
--conformal_trials
```

Split and calibration sizes are also argparse-controlled, but the W&B tuning scripts keep them fixed at the training-script defaults:

```text
--train_fraction
--valid_fraction
--max_calib_size
--calib_fraction
```

## W&B Hyperparameter Tuning

New W&B sweep launchers tune only conformal-method hyperparameters while forcing `--num_runs 20` and preserving the train/validation/calibration-size defaults from each V2 script.

Run one dataset-specific sweep for 10 hours:

```bash
python wandb_tune_trainV2_walmart.py --hours 10
python wandb_tune_trainV2_congress.py --hours 10
python wandb_tune_trainV2_house_bills.py --hours 10
python wandb_tune_trainV2_DBLP_CC.py --hours 10
```

Run all four sweeps in parallel for a 10-hour wall-clock budget:

```bash
python wandb_tune_all_v2.py --hours 10 --devices cuda:4,cuda:5,cuda:6,cuda:7
```

Use `--sequential` if the four sweeps should run one after another instead of in parallel:

```bash
python wandb_tune_all_v2.py --hours 10 --sequential
```

Useful tuning options:

```text
--entity YOUR_WANDB_ENTITY
--project PROJECT_NAME
--sweep-id EXISTING_SWEEP_ID
--wandb-mode online|offline|disabled
--device cuda:0
--count N
--extra-arg "--epochs"
--extra-arg "500"
```

The shared sweep logic is in `wandb_tune_v2_common.py`. It parses `cont_conf_hgnn` metrics from each training run and logs coverage, efficiency, validation efficiency, completed run count, return code, and trial duration to W&B.
