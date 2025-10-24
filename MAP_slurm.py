#!/usr/bin/env python3
# MAP_slurm.py (refactor para ejecutarse por dataset + split, con opción in_between)
import os
import argparse
import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from time import process_time as timer
from sklearn.model_selection import KFold
from utils.dataset import get_dataset
from utils.models import get_mlp
from utils.pytorch_learning import fit_map

# Hypergrid
grid = {
    'num_layers': [1, 2, 3],
    'hidden_units': [20, 30, 50],
    'weight_decay': [0.0, 1e-4, 1e-3]
}


def run_fold_local(fold_idx, splits, ds_name, params, results_root="results", split_type="kfold"):
    if len(splits) == 3:
        train_ds, val_ds, test_ds = splits
    else:
        train_ds, test_ds = splits
        val_ds = None

    indices = np.arange(len(train_ds))
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=params['seed'])

    best_cfg = None
    best_score = float('inf')

    for num_layers in grid['num_layers']:
        for hidden_units in grid['hidden_units']:
            for weight_decay in grid['weight_decay']:
                val_errors = []
                for train_idx, val_idx in inner_cv.split(indices):
                    sub_train = torch.utils.data.Subset(train_ds, train_idx)
                    sub_val   = torch.utils.data.Subset(train_ds, val_idx)

                    train_loader = DataLoader(sub_train, batch_size=params['batch_size'], shuffle=True)
                    val_loader   = DataLoader(sub_val,   batch_size=params['batch_size'], shuffle=False)

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

                    optimizer = torch.optim.Adam(f.parameters(), lr=params['lr'], weight_decay=weight_decay)
                    criterion = torch.nn.MSELoss()

                    fit_map(
                        f,
                        train_loader,
                        optimizer,
                        criterion=criterion,
                        iterations=params['epochs'] * len(train_loader),
                        use_tqdm=False,
                        return_loss=False,
                        device=params['device'],
                    )

                    # Validate
                    f.eval()
                    preds, targets = [], []
                    with torch.no_grad():
                        for xb, yb in val_loader:
                            xb, yb = xb.to(params['device'], dtype=params['dtype']), yb.to(params['device'], dtype=params['dtype'])
                            preds.append(f(xb).cpu())
                            targets.append(yb)
                    preds   = torch.cat(preds, dim=0)
                    targets = torch.cat(targets, dim=0)
                    rmse = np.sqrt(torch.nn.functional.mse_loss(preds, targets).item())
                    val_errors.append(rmse)

                mean_val = np.mean(val_errors)
                if mean_val < best_score:
                    best_score = mean_val
                    best_cfg = {
                        'num_layers': num_layers,
                        'hidden_units': hidden_units,
                        'weight_decay': weight_decay
                    }

    # Retrain best on full train(+val) data
    train_full = torch.utils.data.ConcatDataset([train_ds, val_ds]) if val_ds is not None else train_ds
    loader_full = DataLoader(train_full, batch_size=params['batch_size'], shuffle=True)

    inner_dims = [best_cfg['hidden_units']] * best_cfg['num_layers']
    f_best = get_mlp(
        train_ds.inputs.shape[1],
        train_ds.targets.shape[1],
        inner_dims,
        params['activation'],
        dropout=True,
        device=params['device'],
        dtype=params['dtype'],
    )
    optimizer = torch.optim.Adam(f_best.parameters(), lr=params['lr'], weight_decay=best_cfg['weight_decay'])
    criterion = torch.nn.MSELoss()

    start = timer()
    total_iters = params['epochs'] * len(loader_full)
    fit_map(
        f_best,
        loader_full,
        optimizer,
        criterion=criterion,
        iterations=total_iters,
        use_tqdm=False,
        return_loss=False,
        device=params['device'],
    )
    end = timer()

    # Evaluate on test
    test_loader = DataLoader(test_ds, batch_size=params['batch_size'], shuffle=False)
    f_best.eval()
    preds, targets = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(params['device'], dtype=params['dtype']), yb.to(params['device'], dtype=params['dtype'])
            preds.append(f_best(xb).cpu())
            targets.append(yb)
    preds   = torch.cat(preds, dim=0)
    try:
        mean_scalar = float(np.asarray(train_ds.targets_mean).item())
        std_scalar  = float(np.asarray(train_ds.targets_std).item())
    except Exception:
        mean_scalar = float(train_ds.targets_mean)
        std_scalar  = float(train_ds.targets_std)

    preds   = mean_scalar + preds * std_scalar
    targets = torch.cat(targets, dim=0)

    rmse = np.sqrt(torch.nn.functional.mse_loss(preds, targets).item())
    mae  = torch.nn.functional.l1_loss(preds, targets).item()

    # Guardar modelo y CSV individual por fold (NO summary global)
    map_subdir = "in_between" if split_type == "in_between" else "kfold"
    results_dir = os.path.join(results_root, ds_name, "MAP", map_subdir)
    os.makedirs(results_dir, exist_ok=True)
    model_path = os.path.join(results_dir, f"best_mlp_fold_{fold_idx}.pt")
    torch.save(f_best.state_dict(), model_path)

    cfg_row = {
        'fold': fold_idx,
        'num_layers': best_cfg['num_layers'],
        'hidden_units': best_cfg['hidden_units'],
        'weight_decay': best_cfg['weight_decay'],
        'test_rmse': rmse,
        'test_mae': mae,
        'training_time': end - start,
        'train_mean': mean_scalar,
        'train_std': std_scalar,
        'split_type': split_type
    }
    # CSV individual por fold (único archivo escrito por este proceso)
    cfg_csv = os.path.join(results_root, ds_name, f"{ds_name}_mlp_best_configs_fold_{fold_idx}.csv")
    pd.DataFrame([cfg_row]).to_csv(cfg_csv, index=False)

    return cfg_row


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="boston|energy|yacht|concrete|redwine")
    parser.add_argument('--split', type=int, required=True, help="Split index (1..n_splits)")
    parser.add_argument('--in_between_splits', action='store_true', help="Use get_in_between_splits instead of get_splits")
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
            "batch_size": 100
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

    # Elegir tipo de split
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

    print(f"[MAP] dataset={ds_name} split={args.split}/{n_splits_available} split_type={split_type}")
    res = run_fold_local(args.split, splits, ds_name, params, results_root=args.results_root, split_type=split_type)

    # NOTA: No se concatena un CSV global aquí; cada proceso escribe SU CSV individual:
    # results/{ds_name}/{ds_name}_mlp_best_configs_fold_{fold_idx}.csv

    print("[MAP] Done:", res)