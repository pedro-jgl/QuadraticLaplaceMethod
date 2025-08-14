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

def run_laplace_fold(fold_idx, splits, name, subset, hessian, best_cfg_df, params):
    # Unpack splits
    if len(splits) == 3:
        train_ds, val_ds, test_ds = splits
    else:
        train_ds, test_ds = splits
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True)

    # Get the best configuration for this fold
    best_cfg = best_cfg_df[best_cfg_df['fold'] == fold_idx].iloc[0]
    num_layers = best_cfg['num_layers'].astype(int)
    hidden_units = best_cfg['hidden_units'].astype(int)
    weight_decay = best_cfg['weight_decay']

    # Create the MLP model
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
    # Load the best weights
    f.load_state_dict(torch.load(f"boston/best_mlp_fold_{fold_idx}.pt"))

    # Laplace approximation
    la = Laplace(f, "regression", subset_of_weights=subset, hessian_structure=hessian)
    la.fit(train_loader)

    log_prior, log_sigma = torch.ones(1, requires_grad=True, dtype=params["dtype"]), torch.ones(
        1, requires_grad=True, dtype=params["dtype"]
    )
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    
    for _ in tqdm(range(100)):
        hyper_optimizer.zero_grad()
        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    prior_std = np.sqrt(1 / np.exp(log_prior.detach().numpy())).item()
    log_variance = 2*log_sigma.detach().numpy().item()

    # Predictive distribution
    X = test_ds.inputs
    mean_torch, var_torch = la._glm_predictive_distribution(
        torch.tensor(X, dtype=params["dtype"], device=params["device"])
    )
    
    # Correct mean and variance of predictive distribution
    #train_targets_mean = train_targets_stats.loc[fold_idx - 1, 'mean']
    #train_targets_std = train_targets_stats.loc[fold_idx - 1, 'std']
    mean_torch = train_ds.targets_mean.item() + mean_torch * train_ds.targets_std.item()
    var_torch  = train_ds.targets_std.item() ** 2 * var_torch

    mean_torch = mean_torch.detach()
    var_torch  = var_torch.detach() + np.exp(log_variance)
    var_torch = var_torch.squeeze(-1)

    y_true = torch.tensor(
        test_ds.targets,
        dtype=params["dtype"],
        device=params["device"]
    )
    la_reg = Regression()
    la_reg.update(y_true, mean_torch, var_torch)

    # Save metrics
    metrics_df = pd.DataFrame([la_reg.get_dict()])
    metrics_df.to_csv(f"boston/{name}_metrics_fold_{fold_idx}.csv", index=False)

    return {
        'fold': fold_idx,
        'num_layers': num_layers,
        'hidden_units': hidden_units,
        'weight_decay': weight_decay,
        'prior_std': prior_std,
        'log_variance': log_variance
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Boston Housing Dataset Analysis")
    parser.add_argument('--dataset', type=str, required=True, help="Name of the dataset (e.g., 'Boston', 'Energy')")
    parser.add_argument('--name', type=str, required=True, help="Name of the approximation (e.g., 'lla', 'qla')")
    parser.add_argument('--subset', type=str, required=True, help="Subset of weights to consider (all, last_layer, subnetwork)")
    parser.add_argument('--hessian', type=str, required=True, help="Hessian structure (full, kron, diag, lowrank, quad)")
    args = parser.parse_args()

    # ----- Initial configuration -----
    params = {
            "num_inducing": 20,
            "bnn_structure": [50, 50],
            "MAP_lr": 0.001,
            "MAP_iterations": 3000,
            "lr": 0.001,    # 0.01 o 0.001 
            "epochs": 200,    # Dejar fijo en un valor grande
            "activation": torch.nn.Tanh,
            "device": "cpu",
            "dtype": torch.float64,
            "seed": 2147483647,
            "bb_alpha": 0,
            "prior_std": 1,
            "ll_std": 1,
            "batch_size": 32
    }

    torch.manual_seed(params["seed"])
    dataset = get_dataset("Boston", random_state=params["seed"])

    # Load the best configuration from the CSV
    best_cfg_df = pd.read_csv("boston/boston_mlp_best_configs.csv")
    # Load the training target statistics
    #train_targets_stats = pd.read_csv("boston/boston_train_targets_stats.csv")

    splits_list = list(dataset.get_splits())
    n_procs = 4 #multiprocessing.cpu_count()

    results = []
    with ProcessPoolExecutor(max_workers=n_procs) as executor:
        futures = {
            executor.submit(run_laplace_fold, fold_idx, splits, args.name, args.subset, args.hessian, best_cfg_df, params): fold_idx
            for fold_idx, splits in enumerate(splits_list, start=1)
        }
        
        for future in as_completed(futures):
            fold_idx = futures[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                print(f"Error processing fold {fold_idx}: {e}")


    # Convert results to DataFrame and save
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"boston/{args.name}_results.csv", index=False)



    