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

# Cambiar para establecer semilla!!!!!!!!!!!!!!!!
torch.manual_seed(params["seed"])
dataset = get_dataset("Boston", random_state=params["seed"])

# ----- MLP with K-Fold Cross-Validation -----
mlp_rmse_folds, mlp_mae_folds = [], []
best_cfg_df = pd.DataFrame(columns=[
    'fold', 'num_layers', 'hidden_units', 'weight_decay', 'test_rmse', 'test_mae', 'training_time'
])
train_targets_stats = pd.DataFrame(columns=['mean', 'std'])

# Hyperparameter grid
grid = {
    'num_layers': [1, 2, 3],
    'hidden_units': [20, 30, 50],
    'weight_decay': [0.0, 1e-4, 1e-3]
}

# Outer loop over dataset splits
for fold_idx, splits in enumerate(dataset.get_splits(), start=1):
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

    # Hacer una función ajustar hiper y meter esto para que quede más limpio
    # Paralelizar (se puede paralelizar los splits y el probar con las distintas aproximaciones de laplace)
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

    print(f"[Fold {fold_idx}] Best inner CV RMSE: {best_score:.4f} with {best_cfg}")

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
    # Corregir media de la dist predictiva igual que preds aquí. Para la varianza multiplicar var por la var de los targets.
    # Para esto, guardar la media y desviación estándar de los targets en train_ds.
    train_targets_stats.loc[len(train_targets_stats)] = [train_ds.targets_mean.item(), train_ds.targets_std.item()]

    targets = torch.cat(targets, dim=0)

    rmse = np.sqrt(torch.nn.functional.mse_loss(preds, targets).item())
    mae  = torch.nn.functional.l1_loss(preds, targets).item()

    # Store metrics
    mlp_rmse_folds.append(rmse)
    mlp_mae_folds.append(mae)

    print(f"[Fold {fold_idx}] Final TEST → RMSE: {rmse:.4f}, MAE: {mae:.4f} (training {end-start:.1f}s)\n")

    # Save best configuration as a row in a DataFrame
    best_cfg_df.loc[len(best_cfg_df)] = [fold_idx, best_cfg['num_layers'], best_cfg['hidden_units'], best_cfg['weight_decay'], rmse, mae, end - start]

    # Save model state
    torch.save(f_best.state_dict(), f"boston/best_mlp_fold_{fold_idx}.pt")


# Final averages
print("=== FINAL AVERAGES OVER {} FOLDS ===".format(len(mlp_rmse_folds)))
print(f"MLP avg RMSE: {np.mean(mlp_rmse_folds):.4f}  ± {np.std(mlp_rmse_folds):.4f}")
print(f"MLP avg MAE: {np.mean(mlp_mae_folds):.4f}  ± {np.std(mlp_mae_folds):.4f}")

# Save best configurations to CSV
best_cfg_df.to_csv("boston/boston_mlp_best_configs.csv", index=False)
# Save training targets statistics
train_targets_stats.to_csv("boston/boston_train_targets_stats.csv", index=False)

