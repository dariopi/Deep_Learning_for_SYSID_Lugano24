import os
import random
import numpy as np
import pandas as pd
import torch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.interpolate import InterpolatedUnivariateSpline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

DEFAULT_RANDOM_SEED = 42

def seed_basic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seed_basic(seed)

def get_model_num_params(model: torch.nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def calculate_metrics(y_true, y_pred):
    return {'mse' : mean_squared_error(y_true, y_pred),
            'mae' : mean_absolute_error(y_true, y_pred),
            'rmse' : root_mean_squared_error(y_true, y_pred),
            'r2' : r2_score(y_true, y_pred)}

def generate_regression_dataset(n_samples, snr_db, random_seed=None):
    """
    Generate a synthetic dataset for a regression problem.

    Parameters:
    - n_samples: int, number of samples to generate.
    - sigma_epsilon: float, standard deviation of the noise term epsilon.
    - random_seed: int, optional seed for random number generator for reproducibility.

    Returns:
    - X: numpy array, feature matrix of shape (n_samples, 20).
    - y: numpy array, target variable array of shape (n_samples,).
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate features, each xi ~ N(0, 9) (standard deviation of 3)
    X = np.random.normal(0, 3, size=(n_samples, 20))

    # Compute components of the target variable y
    y = (
        X[:, 0]                                         # x1
        + np.sin(X[:, 1])                               # sin(x2)
        + np.log(np.abs(X[:, 2]) + 1e-10)               # log(|x3|), small epsilon to avoid log(0)
        + X[:, 3]**2                                    # x4^2
        + X[:, 4] * X[:, 5]                             # x5x6
        + (X[:, 6] * X[:, 7] * X[:, 8] < 0).astype(int) # I(x7x8x9<0)
        + (X[:, 9] > 0).astype(int)                     # I(x10>0)
        + X[:, 10] * (X[:, 10] > 0).astype(int)         # x11I(x11>0)
        + np.sqrt(np.abs(X[:, 11]))                     # sqrt(|x12|)
        + np.cos(X[:, 12])                              # cos(x13)
        + 2 * X[:, 13]                                  # 2x14
        + np.abs(X[:, 14])                              # |x15|
        + (X[:, 15] < -1).astype(int)                   # I(x16<-1)
        + X[:, 16] * (X[:, 16] < -1).astype(int)        # x17I(x17<-1)
        - 2 * X[:, 17]                                  # -2x18
        - X[:, 18] * X[:, 19]                           # -x19x20
    )

    # Add noise
    if snr_db > 0:
        snr = 10**(snr_db/10)
        sigma2_y = np.mean(np.var(y)/snr)
        epsilon = np.random.normal(0, np.sqrt(sigma2_y), n_samples)
        y += epsilon

    # Final dataset: features and labels
    target = 'y'
    features = [f'x{i+1}' for i in range(20)]
    df = pd.DataFrame(data=np.hstack((X, y[:,np.newaxis])), columns=features+[target])
    
    return df, features, target

def plot_diagnostics(y_train,
                     y_pred_train,
                     y_test,
                     y_pred_test,
                     model_name='',
                     fig_params={'train_color': "tab:blue", 'test_color': "tab:orange"}, 
                     metric_params={'metric': 'r2', 'metric_func': r2_score, 'metric_label': '$R^2$'}):
    
    # Helper function to ensure y and y_pred are 1D and aligned
    def ensure_1D(y, y_pred):
        y = np.ravel(y)
        y_pred = np.ravel(y_pred)
        return y, y_pred

    # Aligning dimensions of training predictions and actuals
    y_train, y_pred_train = ensure_1D(y_train, y_pred_train)
    # Aligning dimensions of testing predictions and actuals
    y_test, y_pred_test = ensure_1D(y_test, y_pred_test)

    # Set figure and metric parameters
    train_color = fig_params['train_color']
    test_color = fig_params['test_color']
    metric = metric_params['metric']
    metric_func = metric_params['metric_func']
    metric_label = metric_params['metric_label']

    # Draw figure
    fig = plt.figure(constrained_layout=True, figsize=(10, 5))
    gs = gridspec.GridSpec(1, 2, figure=fig, width_ratios=[1, 1.3], height_ratios=[1])

    # Prediction plot
    ax0 = plt.subplot(gs[0])
    ax0.scatter(y_pred_train, y_train, alpha=0.6, c=train_color, label='Train {}$= {:0.3f}$'.format(metric_label, metric_func(y_train, y_pred_train)))
    ax0.scatter(y_pred_test, y_test, alpha=0.4, c=test_color, label='Test {}$= {:0.3f}$'.format(metric_label, metric_func(y_test, y_pred_test)))
    ax0.set_xlabel(r"$\hat{y}$")
    ax0.set_ylabel(r"$y$")
    ## Make square
    ylim = ax0.get_ylim()
    xlim = ax0.get_xlim()
    bounds = (min(xlim[0], ylim[0]), max(xlim[1], ylim[1]))
    ax0.set_xlim(bounds)
    ax0.set_ylim(bounds)
    ax0.set_aspect('equal', adjustable='box')
    # Draw identity line
    ax0.plot(bounds, bounds, lw=2, ls='--', color='k', alpha=0.5)
    # Set legend and title
    leg = ax0.legend(loc='best', frameon=True)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
    ax0.set_title(f'{model_name} prediction plot')

    # Residual plot
    ax1 = plt.subplot(gs[1])
    train_residual = y_train - y_pred_train
    test_residual = y_test - y_pred_test
    ax1.scatter(y_pred_train, train_residual, alpha=0.6, c=train_color, label='Train {}$= {:0.3f}$'.format(metric_label, metric_func(y_train, y_pred_train)))
    ax1.scatter(y_pred_test, test_residual, alpha=0.4, c=test_color, label='Test {}$= {:0.3f}$'.format(metric_label, metric_func(y_test, y_pred_test)))
    ax1.set_xlabel(r"$\hat{y}$")
    ax1.set_ylabel(r"$y - \hat{y}$")
    ## Make square
    bounds = min(ax1.get_xlim()[0], ax1.get_ylim()[0]), max(ax1.get_xlim()[1], ax1.get_ylim()[1])
    ax1.set_xlim(bounds)
    ax1.set_ylim(bounds)
    ax1.set_aspect('equal', adjustable='box')
    # Draw zero line
    ax1.axhline(y=0, lw=2, ls='--', color='k', alpha=0.5)
    # Set legend and title
    leg = ax1.legend(loc='best', frameon=True)
    for lh in leg.legend_handles: 
        lh.set_alpha(1)
    ax1.set_title(f'{model_name} residual plot')
    # Draw residual histograms
    divider = make_axes_locatable(ax1)
    hax = divider.append_axes("right", size=1, pad=0.1, sharey=ax1)
    hax.yaxis.tick_right()
    hax.grid(False, axis="x")
    hax.hist(train_residual, bins=50, orientation="horizontal", density=True, color=train_color, alpha=0.6)
    hax.hist(test_residual, bins=50, orientation="horizontal", density=True, color=test_color, alpha=0.4)
    
    plt.show()

def load_sysid_dataset(dataset_path, snr_db=0, seed=0, n_samples=None):
    df = pd.read_pickle(dataset_path)
    target = 'y(t)'

    if n_samples:
        df = df.iloc[:n_samples].copy()
    
    # Add noise
    if snr_db > 0:
        rng = np.random.default_rng(seed=seed)
        y = df[target].values
        n = y.shape[0]
        snr = 10**(snr_db/10)
        sigma2_y = np.mean(np.var(y)/snr)
        e_y = np.sqrt(sigma2_y)*rng.standard_normal(size=n)
        df[target] = df[target] + e_y
    
    return df, target

def simulate(model, regressors, na, torch_model=False):
    y_regressors = [f'y(t-{i})' for i in range(na, 0, -1)]
    
    # Copy the regressors to modify without affecting the original
    regressors = regressors.copy()

    # Initialize the simulation with the last known outputs
    last_known_outputs = regressors.iloc[0][y_regressors].values

    # Set all output regressors to nan values
    regressors[y_regressors] = np.nan
    
    # Initialize an array to store simulated predictions
    y_sim = np.zeros((regressors.shape[0],1))

    for i in range(regressors.shape[0]):
        # Update the regressor values with the last known outputs
        for t in range(na, 0, -1):
            regressors.iloc[i, regressors.columns.get_loc(f'y(t-{t})')] = last_known_outputs[-t]

        # Predict the next output
        if torch_model:
            y_pred = model(torch.from_numpy(regressors.iloc[[i]].values).type(torch.Tensor))
        else:
            y_pred = model.predict(regressors.iloc[[i]])
        
        # Insert the prediction in the simulation results
        if torch_model:
            y_pred = y_pred.to('cpu').detach().numpy().ravel()
            y_sim[i] = y_pred.item()
        else:
            y_sim[i] = y_pred.item()

        # Update the last known outputs for the next iteration
        last_known_outputs = np.concatenate((last_known_outputs, y_pred))[-na:]
    
    return y_sim



