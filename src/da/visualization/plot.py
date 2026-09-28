import numpy as np
import matplotlib.pyplot as plt
from .plot_funcs import *
from ..diagnostics import KL_divergence
import os

def compare_errors(preds_dict, true_obj, window=None, var_name="All", nodes=[1, 2], save_path=None):
    """Plots error trajectories dynamically based on the number of nodes requested."""
    t = true_obj.t
    n_cols = len(nodes)
    n_rows = 2 if var_name == "All" else 1
    
    # Dynamically scale the figure width and height!
    plt.figure(figsize=(7 * n_cols, 4 * n_rows))
    
    plot_idx = 1
    
    if var_name in ["v_hat", "All"]:
        for node in nodes:
            plt.subplot(n_rows, n_cols, plot_idx)
            p_dict = {name: obj.v_hat[:, node-1] for name, obj in preds_dict.items()}
            plot_error_trajectory(t, true_obj.v_hat[:, node-1], p_dict, window, f"$\\Re\\{{v_{node}\\}}$")
            plot_idx += 1

    if var_name in ["T_hat", "All"]:
        for node in nodes:
            plt.subplot(n_rows, n_cols, plot_idx)
            p_dict = {name: obj.T_hat[:, node-1] for name, obj in preds_dict.items()}
            plot_error_trajectory(t, true_obj.T_hat[:, node-1], p_dict, window, f"$\\Re\\{{T_{node}\\}}$")
            plot_idx += 1

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def multi_pdf(preds_dict, true_obj, var_name="v_hat", nodes=[1, 2], save_path=None):
    """Plots overlapping PDFs, scaling width based on the number of nodes."""
    n_plots = len(nodes)
    plt.figure(figsize=(7 * n_plots, 5)) 
    
    sym = "v" if var_name == "v_hat" else "T"
    data_attr = "v_hat" if var_name == "v_hat" else "T_hat"
    
    for i, node in enumerate(nodes):
        plt.subplot(1, n_plots, i + 1)
        p_dict_n = {name: getattr(obj, data_attr)[:, node-1] for name, obj in preds_dict.items()}
        true_arr_n = getattr(true_obj, data_attr)[:, node-1]
        plot_multi_pdf(p_dict_n, true_arr_n, var_name=f"$\\Re\\{{{sym}_{node}\\}}$")
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_spatial_reconstruction(pred_obj, true_obj, time_idx=-1, nx=200, H1=1.0, H2=0.5, save_path=None):
    """Plots the 1D spatial reconstruction with Topography shaded in the background."""
    x = np.linspace(0, 2 * np.pi, nx)
    t_val = true_obj.t[time_idx]
    d = pred_obj.v_hat.shape[1] // 2
    k_vec = np.arange(1, d + 1)
    
    def reconstruct_field(hat_array, idx):
        real_parts = hat_array[idx, :d]
        imag_parts = hat_array[idx, d:]
        modes_k = real_parts + 1j * imag_parts
        kx = np.outer(k_vec, x)
        complex_sum = np.sum(modes_k[:, None] * np.exp(1j * kx), axis=0)
        return 2 * np.real(complex_sum)

    v_true_x = reconstruct_field(true_obj.v_hat, time_idx)
    v_pred_x = reconstruct_field(pred_obj.v_hat, time_idx)
    T_true_x = reconstruct_field(true_obj.T_hat, time_idx)
    T_pred_x = reconstruct_field(pred_obj.T_hat, time_idx)
    
    topography = H1 * (np.cos(x) + np.sin(x)) + H2 * (np.cos(2*x) + np.sin(2*x))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharex=True)
    
    ax1.plot(x, v_true_x, 'r-', linewidth=2, label="True $v(x)$")
    ax1.plot(x, v_pred_x, 'k--', linewidth=2, label="CGNF $v(x)$")
    
    ax1_topo = ax1.twinx()
    ax1_topo.fill_between(x, topography, min(topography)-1, color='saddlebrown', alpha=0.2, label='Topography $h(x)$')
    ax1_topo.set_ylim(min(topography)-1, max(topography)*3)
    ax1_topo.set_yticks([]) 
    
    ax1.set_title(f"Velocity Field & Topography at $t = {t_val:.2f}$")
    ax1.set_xlabel("Spatial coordinate ($x$)")
    ax1.set_ylabel("Amplitude")
    ax1.grid(True, alpha=0.3)
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax1_topo.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')
    
    ax2.plot(x, T_true_x, 'r-', linewidth=2, label="True $T(x)$")
    ax2.plot(x, T_pred_x, 'k--', linewidth=2, label="CGNF $T(x)$")
    
    ax2_topo = ax2.twinx()
    ax2_topo.fill_between(x, topography, min(topography)-1, color='saddlebrown', alpha=0.2)
    ax2_topo.set_ylim(min(topography)-1, max(topography)*3)
    ax2_topo.set_yticks([])
    
    ax2.set_title(f"Temperature Field at $t = {t_val:.2f}$")
    ax2.set_xlabel("Spatial coordinate ($x$)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_snapshot(true_obj, window, highlight_regions=None, var_name="v_hat", nodes=[1, 2], save_path=None):
    """Plots a snapshot of the true state dynamically shrinking height if only one node is used."""
    start, end = window
    t = true_obj.t[start:end]
    data = true_obj.v_hat if var_name == "v_hat" else true_obj.T_hat
    sym = "v" if var_name == "v_hat" else "T"
    
    n_plots = len(nodes)
    plt.figure(figsize=(14, 3 * n_plots)) 
    
    for i, node in enumerate(nodes):
        plt.subplot(n_plots, 1, i + 1)
        plt.plot(t, data[start:end, node-1], color='black', linewidth=1.5, label="Truth")
        
        if highlight_regions:
            for region in highlight_regions:
                color = region.get('color', 'red')
                label = region.get('label', '')
                for j, h_win in enumerate(region['windows']):
                    w_start = max(start, h_win[0])
                    w_end = min(end, h_win[1])
                    if w_start < w_end:
                        plt.axvspan(true_obj.t[w_start], true_obj.t[w_end], color=color, alpha=0.15, label=label if j==0 else "")
                        
        plt.title(f"$\\Re\\{{\\hat{{{sym}_{node}}}\\}}$")
        plt.legend(loc='upper right')

    plt.tight_layout()
    if save_path: plt.savefig(save_path, dpi=300)
    plt.close()

def plot_full_context_trajectory(obj, var_name="v_hat", nodes=[1, 2], save_path=None):
    """Generates the master context plot showing BOTH high and low amplitude regions."""
    full_true_state = np.concatenate([obj.v_hat, obj.T_hat], axis=1)
    
    high_windows, _ = get_high_amplitude_windows(full_true_state, window_size=80000, num_windows=1)
    low_windows, _ = get_low_amplitude_windows(full_true_state, window_size=80000, num_windows=1)
    
    highlights = [
        {'windows': high_windows, 'color': 'red', 'label': 'High Amplitude'},
        {'windows': low_windows, 'color': 'blue', 'label': 'Low Amplitude'}
    ]
    
    plot_snapshot(
        obj, window=(0, len(obj.t)), highlight_regions=highlights, 
        var_name=var_name, nodes=nodes, save_path=save_path
    )

def plot_high_amp(obj, var_name="v_hat", nodes=[1, 2], compare_obj=None, save_path=None):
    full_true_state = np.concatenate([obj.v_hat, obj.T_hat], axis=1)
    interesting_windows, _ = get_high_amplitude_windows(full_true_state, window_size=80000, num_windows=1)
    
    for i, window in enumerate(interesting_windows):
        print(f"Plotting High-Amp Zoomed Window {i+1}: {window}")
        plot_snapshot(
            obj, window=window, var_name=var_name, nodes=nodes,
            save_path=save_path.replace(".png", f"_high_amp_window_{i+1}.png") if save_path else None
        )
    
        if compare_obj is not None:
            comp_dict = {list(compare_obj.keys())[0]: list(compare_obj.values())[0]} if isinstance(compare_obj, dict) else {"Model": compare_obj}
            compare_errors(comp_dict, obj, window=window, var_name=var_name, nodes=nodes,
                           save_path=save_path.replace(".png", f"_compare_high_amp_window_{i+1}.png") if save_path else None)

def plot_low_amp(obj, var_name="v_hat", nodes=[1, 2], compare_obj=None, save_path=None):
    full_true_state = np.concatenate([obj.v_hat, obj.T_hat], axis=1)
    interesting_windows, _ = get_low_amplitude_windows(full_true_state, window_size=80000, num_windows=1)
    
    for i, window in enumerate(interesting_windows):
        print(f"Plotting Low-Amp Zoomed Window {i+1}: {window}")
        plot_snapshot(
            obj, window=window, var_name=var_name, nodes=nodes,
            save_path=save_path.replace(".png", f"_low_amp_window_{i+1}.png") if save_path else None
        )
    
        if compare_obj is not None:
            comp_dict = {list(compare_obj.keys())[0]: list(compare_obj.values())[0]} if isinstance(compare_obj, dict) else {"Model": compare_obj}
            compare_errors(comp_dict, obj, window=window, var_name=var_name, nodes=nodes,
                           save_path=save_path.replace(".png", f"_compare_low_amp_window_{i+1}.png") if save_path else None)
            

def plot_period_diagnostics(true_obj, pred_obj, window, var="v_hat", prefix="high_amp", save_dir=None, color='r'):
    """
    Generates two plots for a specific time window:
    1. Side-by-side Node 1 and Node 2 approximations (Truth vs Model with 2-sigma bounds).
    2. Side-by-side Node 1 and Node 2 errors.
    """
    start, end = window
    t = true_obj.t[start:end]

    # Setup data and determine covariance indices
    # (Assuming 8D state: v_real(0,1), v_imag(2,3), T_real(4,5), T_imag(6,7))
    if var == "v_hat":
        true_data1, pred_data1 = true_obj.v_hat[start:end, 0], pred_obj.v_hat[start:end, 0]
        true_data2, pred_data2 = true_obj.v_hat[start:end, 1], pred_obj.v_hat[start:end, 1]
        cov_idx1, cov_idx2 = 0, 1
    elif var == "T_hat":
        true_data1, pred_data1 = true_obj.T_hat[start:end, 0], pred_obj.T_hat[start:end, 0]
        true_data2, pred_data2 = true_obj.T_hat[start:end, 1], pred_obj.T_hat[start:end, 1]
        cov_idx1, cov_idx2 = 4, 5

    # Check if the model has a covariance matrix (Full CGNF)
    has_cov = hasattr(pred_obj, 'cov') and pred_obj.cov is not None
    if has_cov:
        std1 = np.sqrt(pred_obj.cov[start:end, cov_idx1, cov_idx1])
        std2 = np.sqrt(pred_obj.cov[start:end, cov_idx2, cov_idx2])

    # ==========================================
    # PLOT 1: State Approximations (Side-by-Side)
    # ==========================================
    plt.figure(figsize=(14, 5))
    
    # Subplot 1: Node 1 approximation
    plt.subplot(1, 2, 1)
    plt.plot(t, true_data1, 'k-', linewidth=2, label="Truth")
    plt.plot(t, pred_data1, color +'--', linewidth=1.5, label="CGNF Estimate")
    if has_cov:
        plt.fill_between(t, pred_data1 - 2*std1, pred_data1 + 2*std1, color=color, alpha=0.2, label=r"$\pm 2\sigma$")
    plt.title(f"$\\Re\\{{\\hat{{v_{1}}}\\}}$ Approximation" if var == "v_hat" else f"$\\Re\\{{\\hat{{T_{1}}}\\}}$ Approximation")
    plt.xlabel("Time")
    plt.legend(loc='upper right')
    
    # Subplot 2: Node 2 approximation
    plt.subplot(1, 2, 2)
    plt.plot(t, true_data2, 'k-', linewidth=2, label="Truth")
    plt.plot(t, pred_data2, color +'--', linewidth=1.5, label="CGNF Estimate")
    if has_cov:
        plt.fill_between(t, pred_data2 - 2*std2, pred_data2 + 2*std2, color=color, alpha=0.2, label=r"$\pm 2\sigma$")
    plt.title(f"$\\Re\\{{\\hat{{v_{2}}}\\}}$ Approximation" if var == "v_hat" else f"$\\Re\\{{\\hat{{T_{2}}}\\}}$ Approximation")
    plt.xlabel("Time")
    plt.legend(loc='upper right')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_approximations.png"), dpi=300)
    plt.close()

    # ==========================================
    # PLOT 2: Errors (Side-by-Side)
    # ==========================================
    plt.figure(figsize=(7, 5))
    
    # Node 1 error
    err_v = pred_data1 - true_data1
    plt.plot(t, err_v, color=color, linewidth=1.5, label = f"$\\Re\\{{\\hat{{v_{1}}}\\}}$ Error" if var == "v_hat" else f"$\\Re\\{{\\hat{{T_{1}}}\\}}$ Error")
    plt.axhline(0, color='k', linestyle='-', linewidth=1)    
    plt.xlabel("Time")

    # Node 2 error
    err_T = pred_data2 - true_data2
    plt.plot(t, err_T, 'g-', linewidth=1.5, label = f"$\\Re\\{{\\hat{{T_{2}}}\\}}$ Error" if var == "T_hat" else f"$\\Re\\{{\\hat{{v_{2}}}\\}}$ Error")
    plt.xlabel("Time")
    plt.legend(loc='upper right')

    plt.title(f"$\\Re\\{{\\hat{{T}}\\}}$ Error (Estimate - Truth)" if var == "T_hat" else f"$\\Re\\{{\\hat{{v}}\\}}$ Error (Estimate - Truth)")


    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, f"{prefix}_errors.png"), dpi=300)
    plt.close()

def plot_focused_extremes(true_obj, pred_obj, save_dir=None):
    """Fetches the high and low windows and generates the focused plots for both variables."""
    full_true_state = np.concatenate([true_obj.v_hat, true_obj.T_hat], axis=1)
    
    # Grab exactly 1 window for each regime
    high_windows, _ = get_high_amplitude_windows(full_true_state, window_size=80000, num_windows=1)
    low_windows, _ = get_low_amplitude_windows(full_true_state, window_size=80000, num_windows=1)
    
    if high_windows:
        print(f" -> Generating High-Amp focused plots...")
        plot_period_diagnostics(true_obj, pred_obj, window=high_windows[0], var="v_hat", prefix="high_amp_v", save_dir=save_dir)
        plot_period_diagnostics(true_obj, pred_obj, window=high_windows[0], var="T_hat", prefix="high_amp_T", save_dir=save_dir)
        
    if low_windows:
        print(f" -> Generating Low-Amp focused plots...")
        plot_period_diagnostics(true_obj, pred_obj, window=low_windows[0], var="v_hat", prefix="low_amp_v", save_dir=save_dir, color = 'b')
        plot_period_diagnostics(true_obj, pred_obj, window=low_windows[0], var="T_hat", prefix="low_amp_T", save_dir=save_dir, color = 'b')