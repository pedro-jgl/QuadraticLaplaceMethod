import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
import sys
from time import process_time as timer
import matplotlib.pyplot as plt
from tqdm import tqdm
from laplace import Laplace
from utils.dataset import get_dataset
from utils.models import get_mlp
from utils.pytorch_learning import fit_map
from utils.metrics import *
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import KFold
import argparse

from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing


# Hyperparameter grid
grid = {
    'num_layers': [1, 2, 3],
    'hidden_units': [20, 30, 50],
    'weight_decay': [0.0, 1e-4, 1e-3]
}

def run_fold(fold_idx, splits, ds_name, params):
    # Unpack splits
    if len(splits) == 3:
        train_ds, val_ds, test_ds = splits
    else:
        train_ds, test_ds = splits
        val_ds = None

    # Prepare indices for inner CV
    indices = np.arange(len(train_ds))
    inner_cv = KFold(n_splits=5, shuffle=True, random_state=params['seed'])

    best_cfg = None
    best_score = float('inf')

    # Grid search with internal 5-fold CV
    for num_layers in grid['num_layers']:
        for hidden_units in grid['hidden_units']:
            for weight_decay in grid['weight_decay']:
                val_errors = []
                for train_idx, val_idx in inner_cv.split(indices):
                    sub_train = torch.utils.data.Subset(train_ds, train_idx)
                    sub_val   = torch.utils.data.Subset(train_ds, val_idx)

                    train_loader = DataLoader(sub_train, batch_size=params['batch_size'], shuffle=True)
                    val_loader   = DataLoader(sub_val,   batch_size=params['batch_size'], shuffle=False)

                    # Build model
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

                    optimizer = torch.optim.Adam(
                        f.parameters(), lr=params['lr'], weight_decay=weight_decay
                    )
                    criterion = torch.nn.MSELoss()

                    # Train inner model
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
    optimizer = torch.optim.Adam(
        f_best.parameters(), lr=params['lr'], weight_decay=best_cfg['weight_decay']
    )
    criterion = torch.nn.MSELoss()

    start = timer()
    # Train full model via fit_map
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
    # Desnormalizar las predicciones
    preds   = train_ds.targets_mean.item() + preds * train_ds.targets_std.item()
    targets = torch.cat(targets, dim=0)

    rmse = np.sqrt(torch.nn.functional.mse_loss(preds, targets).item())
    mae  = torch.nn.functional.l1_loss(preds, targets).item()

    # Store model
    torch.save(f_best.state_dict(), f"{ds_name}/best_mlp_fold_{fold_idx}.pt")

    return {
        'fold': fold_idx,
        'num_layers': best_cfg['num_layers'],
        'hidden_units': best_cfg['hidden_units'],
        'weight_decay': best_cfg['weight_decay'],
        'test_rmse': rmse,
        'test_mae': mae,
        'training_time': end - start,
        'train_mean': train_ds.targets_mean.item(),
        'train_std': train_ds.targets_std.item()
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., 'boston', 'energy')")
    args = parser.parse_args()

    # ----- Initial configuration -----
    params = {
            "num_inducing": 20,
            "bnn_structure": [50, 50],
            "MAP_lr": 0.001,
            "MAP_iterations": 3000,
            "lr": 0.001,    # 0.01 o 0.001 
            "epochs": 200,    # Echar un vistazo a loss y ver si se estabiliza (a lo mejor converge con 50 épocas)
            "activation": torch.nn.Tanh,
            "device": "cpu",
            "dtype": torch.float64,
            "seed": 2147483647,
            "bb_alpha": 0,
            "prior_std": 1,
            "ll_std": 1,
            "batch_size": 100
    }

    torch.manual_seed(params["seed"])
    if args.dataset == "boston":
        dataset = get_dataset("Boston", random_state=params["seed"])
    elif args.dataset == "energy":
        dataset = get_dataset("Energy", random_state=params["seed"])
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    n_procs = 4 # multiprocessing.cpu_count()
    splits_list = list(dataset.get_splits())

    results = []
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = {executor.submit(run_fold, fold_idx, splits, args.dataset, params): fold_idx for fold_idx, splits in enumerate(splits_list, start=1)}
        for future in as_completed(futures):
            res = future.result()
            print(f"[Fold {res['fold']}] Best inner CV RMSE: {res['test_rmse']:.4f} with {res['num_layers']} layers and {res['hidden_units']} hidden units")
            results.append(res)

    # Convert results to DataFrame
    best_cfg_df = pd.DataFrame(results)[['fold','num_layers','hidden_units','weight_decay','test_rmse','test_mae','training_time']]
    stats_df = best_cfg_df.agg({'test_rmse':['mean','std'], 'test_mae':['mean','std']})
    train_stats = pd.DataFrame([{'mean': r['train_mean'], 'std': r['train_std']} for r in results])

    print("=== FINAL AVERAGES OVER {} FOLDS ===".format(len(results)))
    print(f"MLP avg RMSE: {stats_df.loc['mean','test_rmse']:.4f} ± {stats_df.loc['std','test_rmse']:.4f}")
    print(f"MLP avg MAE: {stats_df.loc['mean','test_mae']:.4f} ± {stats_df.loc['std','test_mae']:.4f}")

    best_cfg_df.to_csv(f"{args.dataset}/{args.dataset}_mlp_best_configs.csv", index=False)
    train_stats.to_csv(f"{args.dataset}/{args.dataset}_train_targets_stats.csv", index=False)

