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


parser = argparse.ArgumentParser(description="Boston Housing Dataset Analysis")
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
        "batch_size": 2 
}

torch.manual_seed(params["seed"])
dataset = get_dataset("Boston")

# ----- MLP -----
# Load the best configuration from the CSV
best_cfg_df = pd.read_csv("boston/boston_mlp_best_configs.csv")
# Load the training target statistics
train_targets_stats = pd.read_csv("boston/boston_train_targets_stats.csv")

# Loop over the dataset splits
for fold_idx, splits in enumerate(dataset.get_splits(), start=1):
    # Unpack splits
    if len(splits) == 3:
        train_ds, val_ds, test_ds = splits
    else:
        train_ds, test_ds = splits
        val_ds = None

    train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle = True)

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
    subset = args.subset
    hessian = args.hessian
    X = test_ds.inputs
    la = Laplace(f, "regression", subset_of_weights=subset, hessian_structure=hessian)
    la.fit(train_loader)

    log_prior, log_sigma = torch.ones(1, requires_grad=True, dtype=params["dtype"]), torch.ones(
        1, requires_grad=True, dtype = params["dtype"]
    )
    hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
    for i in tqdm(range(100)):
        hyper_optimizer.zero_grad()
        neg_marglik = -la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
        neg_marglik.backward()
        hyper_optimizer.step()

    prior_std = np.sqrt(1 / np.exp(log_prior.detach().numpy())).item()
    log_variance = 2*log_sigma.detach().numpy().item()

    print("Optimal GP prior std: ", prior_std)
    print("Optimal GP likelihoog log variance: ", log_variance)

    mean_torch, var_torch = la._glm_predictive_distribution(
        torch.tensor(X, dtype=params["dtype"], device=params["device"])
    )
    # Corregir media de la dist predictiva igual que preds en test. Para la varianza multiplicar var por la var de los targets.
    train_targets_mean = train_targets_stats.loc[fold_idx - 1, 'mean']
    train_targets_std = train_targets_stats.loc[fold_idx - 1, 'std']
    mean_torch = train_targets_mean + mean_torch * train_targets_std
    var_torch  = train_targets_std ** 2 * var_torch

    mean_torch = mean_torch.detach()                  
    var_torch  = var_torch.detach() + np.exp(log_variance)

    y_true = torch.tensor(
        test_ds.targets.flatten(),
        dtype=params["dtype"],
        device=params["device"]
    )

    la_reg = Regression()
    la_reg.update(y_true, mean_torch, var_torch)

    # Save metrics
    metrics_df = pd.DataFrame([la_reg.get_dict()])
    metrics_df.to_csv(f"boston/{args.name}_metrics_fold_{fold_idx}.csv", index=False)




    