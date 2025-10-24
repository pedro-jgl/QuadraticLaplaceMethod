#!/usr/bin/env python3
# Posterior_slurm.py (adaptado: infiere subset & hessian desde --name si no se pasan)
import os
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from laplace import Laplace
from utils.dataset import get_dataset
from utils.models import get_mlp
from utils.metrics import *
from sklearn.model_selection import KFold

def infer_subset_hessian_from_name(name):
    name = name.lower()
    mapping = {
        'qla': ('all', 'quad'),
        'lla': ('all', 'full'),
        'llla': ('last_layer', 'full'),
        'kron': ('all', 'kron')
    }
    if name not in mapping:
        raise ValueError(f"Unknown method name '{name}'. Expected one of: {list(mapping.keys())}")
    return mapping[name]


def run_laplace_local(fold_idx, splits, name, subset, hessian, cfg_row, ds_name, params, results_root="results", split_type="kfold"):
    if len(splits) == 3:
        train_ds, val_ds, test_ds = splits
    else:
        train_ds, test_ds = splits
        val_ds = None

    # cfg_row is a dict/Series with the best MAP config for this fold
    num_layers = int(cfg_row['num_layers'])
    hidden_units = int(cfg_row['hidden_units'])
    weight_decay = float(cfg_row['weight_decay'])

    # Build model & load MAP weights (se espera que MAP haya guardado en results/{ds_name}/MAP/{split_type}/)
    inner_dims = [hidden_units] * num_layers
    f = get_mlp(
        train_ds.inputs.shape[1],
        train_ds.targets.shape[1],
        inner_dims,
        params['activation'],
        dropout=True,
        device=params['device'],
        dtype=params['dtype'],
    )

    model_path = os.path.join(results_root, ds_name, "MAP", split_type, f"best_mlp_fold_{fold_idx}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}. Run MAP_slurm.py first for this fold.")

    f.load_state_dict(torch.load(model_path, map_location='cpu'))

    la = Laplace(f, "regression", subset_of_weights=subset, hessian_structure=hessian)
    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)
    la.fit(train_loader)

    log_prior = torch.ones(1, requires_grad=True, dtype=params['dtype'])
    log_sigma = torch.ones(1, requires_grad=True, dtype=params['dtype'])
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    for _ in tqdm(range(100), desc=f"[Laplace {name}] fold {fold_idx}"):
        hyper_optimizer.zero_grad()
        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    prior_std = np.sqrt(1 / np.exp(log_prior.detach().numpy())).item()
    log_variance = 2 * log_sigma.detach().numpy().item()

    X = test_ds.inputs
    mean_torch, var_torch = la._glm_predictive_distribution(
        torch.tensor(X, dtype=params['dtype'], device=params["device"])
    )

    try:
        mean_scalar = float(np.asarray(train_ds.targets_mean).item())
        std_scalar  = float(np.asarray(train_ds.targets_std).item())
    except Exception:
        mean_scalar = float(train_ds.targets_mean)
        std_scalar  = float(train_ds.targets_std)

    mean_torch = mean_scalar + mean_torch * std_scalar
    var_torch  = (std_scalar ** 2) * (var_torch + np.exp(log_variance))

    mean_torch = mean_torch.detach()
    var_torch  = var_torch.detach().squeeze(-1)

    y_true = torch.tensor(test_ds.targets, dtype=params['dtype'], device=params["device"])
    la_reg = Regression()
    la_reg.update(y_true, mean_torch, var_torch)

    out_dir = os.path.join(results_root, ds_name, "post", name, split_type)
    os.makedirs(out_dir, exist_ok=True)
    metrics_df = pd.DataFrame([la_reg.get_dict()])
    metrics_csv = os.path.join(out_dir, f"{name}_metrics_fold_{fold_idx}.csv")
    metrics_df.to_csv(metrics_csv, index=False)

    return {
        'fold': fold_idx,
        'name': name,
        'num_layers': num_layers,
        'hidden_units': hidden_units,
        'weight_decay': weight_decay,
        'prior_std': prior_std,
        'log_variance': log_variance,
        'split_type': split_type,
        'subset': subset,
        'hessian': hessian
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="boston|energy|yacht|concrete|redwine")
    parser.add_argument('--split', type=int, required=True, help="Split index (1..n_splits)")
    parser.add_argument('--name', type=str, required=True, help="qla|lla|llla|kron")
    parser.add_argument('--in_between_splits', action='store_true', help="Use get_in_between_splits instead of get_splits")
    parser.add_argument('--subset', type=str, default=None, help="(optional) subset_of_weights for Laplace; if omitted it is inferred from --name")
    parser.add_argument('--hessian', type=str, default=None, help="(optional) hessian_structure for Laplace; if omitted it is inferred from --name")
    parser.add_argument('--seed', type=int, default=2147483647)
    parser.add_argument('--results_root', type=str, default="results")
    args = parser.parse_args()

    params = {
            "num_inducing": 20,
            "bnn_structure": [50, 50],
            "MAP_lr": 0.001,
            "MAP_iterations": 3000,
            "lr": 0.001,
            "epochs": 200,
            "activation": torch.nn.Tanh,
            "device": "cpu",
            "dtype": torch.float64,
            "seed": args.seed,
            "bb_alpha": 0,
            "prior_std": 1,
            "ll_std": 1,
            "batch_size": 32
    }

    torch.manual_seed(params["seed"])
    np.random.seed(params["seed"])

    mapping = {
        "boston": "Boston",
        "energy": "Energy",
        "yacht": "Yacht",
        "concrete": "Concrete",
        "redwine": "RedWine"
    }
    ds_key = args.dataset.lower()
    if ds_key not in mapping:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    ds_name = mapping[ds_key]

    dataset = get_dataset(ds_name, random_state=params["seed"])

    if args.in_between_splits:
        splits_iterable = list(dataset.get_in_between_splits())
        split_type = "in_between"
    else:
        splits_iterable = list(dataset.get_splits())
        split_type = "kfold"

    n_splits_available = len(splits_iterable)
    if args.split < 1 or args.split > n_splits_available:
        raise ValueError(f"split must be in 1..{n_splits_available} for dataset {ds_name} with split_type={split_type}")

    splits = splits_iterable[args.split - 1]

    # Inferir subset & hessian a partir de --name si no se pasaron explicitamente
    name_lower = args.name.lower()
    inferred_subset, inferred_hessian = infer_subset_hessian_from_name(name_lower)
    subset = args.subset if args.subset is not None else inferred_subset
    hessian = args.hessian if args.hessian is not None else inferred_hessian

    print(f"[Posterior] dataset={ds_name} split={args.split}/{n_splits_available} method={name_lower} split_type={split_type}")
    print(f"Using subset='{subset}'  hessian='{hessian}' (inferred from name='{name_lower}' unless overridden)")

    # Load the per-fold best config written by MAP_slurm.py
    cfg_path = os.path.join(args.results_root, ds_name, f"{ds_name}_mlp_best_configs_fold_{args.split}.csv")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Per-fold best config not found: {cfg_path}. Run MAP_slurm.py for this fold first.")
    best_cfg_df = pd.read_csv(cfg_path)
    # best_cfg_df should contain a single row corresponding to this fold
    cfg_row = best_cfg_df.iloc[0]

    out = run_laplace_local(args.split, splits, name_lower, subset, hessian, cfg_row, ds_name, params, results_root=args.results_root, split_type=split_type)

    # NOTA: No se escribe un CSV global aquí; cada proceso escribe SU CSV individual:
    # results/{ds_name}/post/{name}/{split_type}/{name}_metrics_fold_{fold_idx}.csv

    print("[Posterior] Done:", out)