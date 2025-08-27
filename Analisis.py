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



if __name__ == "__main__":
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
    
    ds_name = "redwine"
    dataset = get_dataset("RedWine", random_state=params["seed"])

    # Load the best configuration from the CSV
    best_cfg_df = pd.read_csv(f"{ds_name}/{ds_name}_mlp_best_configs.csv")
    # Load the training target statistics
    #train_targets_stats = pd.read_csv(f"{args.dataset}/{args.dataset}_train_targets_stats.csv")

    splits_list = list(dataset.get_splits())

    for fold_idx, splits in enumerate(splits_list, start=1):
        # Unpack splits
        if len(splits) == 3:
            train_ds, val_ds, test_ds = splits
        else:
            train_ds, test_ds = splits
            val_ds = None

        train_loader = DataLoader(train_ds, batch_size=params['batch_size'], shuffle=True, )
        import pdb; pdb.set_trace()

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
        f.load_state_dict(torch.load(f"{ds_name}/best_mlp_fold_{fold_idx}.pt"))

        # Laplace approximation
        subset_lla = 'all'
        hessian_lla = 'full'
        lla = Laplace(f, "regression", subset_of_weights=subset_lla, hessian_structure=hessian_lla)
        lla.fit(train_loader)

        subset_qla = 'all'
        hessian_qla = 'quad'
        qla = Laplace(f, "regression", subset_of_weights=subset_qla, hessian_structure=hessian_qla)
        qla.fit(train_loader)
        #import pdb; pdb.set_trace()

        # Hyperoptim lla
        log_prior, log_sigma = torch.ones(1, requires_grad=True, dtype=params["dtype"]), torch.ones(
            1, requires_grad=True, dtype=params["dtype"]
        )
        hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
        
        for _ in tqdm(range(100)):
            hyper_optimizer.zero_grad()
            neg_marglik = -lla.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
            neg_marglik.backward()
            hyper_optimizer.step()

        prior_std = np.sqrt(1 / np.exp(log_prior.detach().numpy())).item()
        log_variance = 2*log_sigma.detach().numpy().item()

        # Hyperoptim qla
        log_prior_q, log_sigma_q = torch.ones(1, requires_grad=True, dtype=params["dtype"]), torch.ones(
            1, requires_grad=True, dtype=params["dtype"]
        )
        hyper_optimizer_q = torch.optim.Adam([log_prior_q, log_sigma_q], lr=1e-1)

        for _ in tqdm(range(100)):
            hyper_optimizer_q.zero_grad()
            neg_marglik_q = -qla.log_marginal_likelihood(log_prior_q.exp(), log_sigma_q.exp())
            neg_marglik_q.backward()
            hyper_optimizer_q.step()

        prior_std_q = np.sqrt(1 / np.exp(log_prior_q.detach().numpy())).item()
        log_variance_q = 2*log_sigma_q.detach().numpy().item()

        # Predictive distribution
        X = test_ds.inputs
        mean_lla, var_lla = lla._glm_predictive_distribution(
            torch.tensor(X, dtype=params["dtype"], device=params["device"])
        )
        
        # Correct mean and variance of predictive distribution
        #train_targets_mean = train_targets_stats.loc[fold_idx - 1, 'mean']
        #train_targets_std = train_targets_stats.loc[fold_idx - 1, 'std']
        mean_lla = train_ds.targets_mean.item() + mean_lla * train_ds.targets_std.item()
        var_lla  = train_ds.targets_std.item() ** 2 * (var_lla + np.exp(log_variance)) # Se hace así? O después de escalar?

        mean_lla = mean_lla.detach()
        var_lla  = var_lla.detach().squeeze(-1)

        # Predictive distribution QLA
        mean_qla, var_qla = qla._glm_predictive_distribution(
            torch.tensor(X, dtype=params["dtype"], device=params["device"])
        )

        # Correct mean and variance of predictive distribution
        mean_qla = train_ds.targets_mean.item() + mean_qla * train_ds.targets_std.item()
        var_qla  = train_ds.targets_std.item() ** 2 * (var_qla + np.exp(log_variance_q))

        mean_qla = mean_qla.detach()
        var_qla  = var_qla.detach().squeeze(-1)

        y_true = torch.tensor(
            test_ds.targets,
            dtype=params["dtype"],
            device=params["device"]
        )

        lla_reg = Regression()
        lla_reg.update(y_true, mean_lla, var_lla)
        metrics_lla = pd.DataFrame([lla_reg.get_dict()])

        qla_reg = Regression()
        qla_reg.update(y_true, mean_qla, var_qla)
        metrics_qla = pd.DataFrame([qla_reg.get_dict()])
        import pdb; pdb.set_trace()



        