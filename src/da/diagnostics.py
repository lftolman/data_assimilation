import numpy as np
from scipy.stats import pearsonr, gaussian_kde

def compute_rmse(true, pred):
    std_true = np.std(true)
    if std_true == 0: return np.sqrt(np.mean((true - pred) ** 2)) # Avoid div by zero
    return np.sqrt(np.mean((true - pred) ** 2)) / std_true

def KL_divergence(true, pred):
    kde_true = gaussian_kde(true)
    kde_pred = gaussian_kde(pred)
    
    x_min = min(true.min(), pred.min())
    x_max = max(true.max(), pred.max())
    x_vals, dx = np.linspace(x_min, x_max, 300, retstep=True) # Get dx
    
    p_true = kde_true(x_vals)
    p_pred = kde_pred(x_vals)
    
    inner = np.where(p_true > 1e-12, p_true * np.log(p_true / (p_pred + 1e-12)), 0)
    return np.sum(inner) * dx

def pattern_correlation(true, pred):
    """
    Compute the pattern correlation between true and predicted values. 
    This is the correlation coefficient between the true and predicted values, after removing the mean.
    true: array of true values
    pred: array of predicted values
    """
    return pearsonr(true - np.mean(true), pred - np.mean(pred)).statistic


def compute_variable_diagnostics(true, pred):
    """
    Compute diagnostics (RMSE and pattern correlation) for a specific variable.
    true: array of true values for the variable
    pred: array of predicted values for the variable
    """
    rmse = compute_rmse(true, pred)
    corr = pattern_correlation(true, pred)
    kl_divergence = KL_divergence(true, pred)
    return rmse, corr, kl_divergence

def compute_diagnostics(true_obj, pred_obj):
    # d is the number of complex nodes (half the width of v_hat)
    if true_obj.system == "Barotropic":
        num_nodes = pred_obj.v_hat.shape[1] // 2
        diag_results = {}

        for i in range(num_nodes):
            # Indexing: [:, i] is Real, [:, i + num_nodes] is Imaginary
            diag_results[f'v_node_{i+1}_real'] = compute_variable_diagnostics(
                true_obj.v_hat[:, i], pred_obj.v_hat[:, i]
            )
            diag_results[f'v_node_{i+1}_imag'] = compute_variable_diagnostics(
                true_obj.v_hat[:, i + num_nodes], pred_obj.v_hat[:, i + num_nodes]
            )
            diag_results[f'T_node_{i+1}_real'] = compute_variable_diagnostics(
                true_obj.T_hat[:, i], pred_obj.T_hat[:, i]
            )
            diag_results[f'T_node_{i+1}_imag'] = compute_variable_diagnostics(
                true_obj.T_hat[:, i + num_nodes], pred_obj.T_hat[:, i + num_nodes]
            )

        return diag_results
    
    if true_obj.system == "Lorenz63":
        diag_results = {}
        diag_results['y'] = compute_variable_diagnostics(true_obj.mu[:, 0], pred_obj.mu[:, 0])
        diag_results['z'] = compute_variable_diagnostics(true_obj.mu[:, 1], pred_obj.mu[:, 1])
        return diag_results


    if true_obj.system == "Lorenz96":
        num_nodes = pred_obj.mu.shape[1]
        diag_results = {}

        for i in range(num_nodes):
            diag_results[f'node_{i+1}'] = compute_variable_diagnostics(
                true_obj.mu[:, i], pred_obj.mu[:, i]
            )

        return diag_results

