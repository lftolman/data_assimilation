import os
import numpy as np
import pandas as pd

# Adjust these imports based on your exact folder structure
from .visualization.animate import animate_spatial_reconstruction
from .diagnostics import compute_diagnostics
from .visualization.plot import *

def run_da_dashboard(true_obj, models_dict, target_node=1, window=None, save_dir=None, include_animation=False):
    """
    Runs the full suite of diagnostics and visualizations for a DA run across multiple models.
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("="*60)
    print("--- BAROTROPIC DA DIAGNOSTICS ---")
    print("="*60)

    # ---------------------------------------------------------
    # 1. Compute and Print Quantitative Diagnostics
    # ---------------------------------------------------------
    print("\n1. Calculating Statistical Metrics...")
    
    all_metrics = {}
    for model_name, pred_obj in models_dict.items():
        metrics = compute_diagnostics(true_obj, pred_obj)
        all_metrics[model_name] = metrics

    try:
        print("\n--- Error Metrics & KL Divergence ---")
        for model_name, metrics in all_metrics.items():
            print(f"\nModel: {model_name}")
            df_metrics = pd.DataFrame(metrics).T
            df_metrics.columns = ['RMSE', 'Correlation', 'KL Divergence']
            print(df_metrics.round(4).to_string())
            
            if save_dir:
                csv_path = os.path.join(save_dir, f"diagnostics_summary_{model_name.replace(' ', '_')}.csv")
                df_metrics.to_csv(csv_path)
                
    except ImportError:
        print("\n--- Error Metrics & KL Divergence ---")
        for model_name, metrics in all_metrics.items():
            print(f"\nModel: {model_name}")
            for key, vals in metrics.items():
                print(f"{key:<15}: RMSE={vals['RMSE']:.4f} | Corr={vals['Correlation']:.4f} | KL={vals['KL']:.4f}")

    # ---------------------------------------------------------
    # 2. Visualizations
    # ---------------------------------------------------------
    print("\n2. Generating Visualizations...")
    
    primary_model_name = list(models_dict.keys())[0]
    primary_pred_obj = models_dict[primary_model_name]

    print(f" -> Plotting Spatial Reconstruction with Topography ({primary_model_name})...")
    plot_spatial_reconstruction(primary_pred_obj, true_obj, time_idx=-1, nx=200, 
                                save_path=os.path.join(save_dir, "spatial_reconstruction_topo.png") if save_dir else None)
    
    print(f" -> Plotting high-amplitude windows ({primary_model_name})...")
    plot_high_amp(true_obj, compare_obj=primary_pred_obj, var_name="v_hat", save_path=os.path.join(save_dir, "high_amp_v.png") if save_dir else None)
    plot_high_amp(true_obj, compare_obj=primary_pred_obj, var_name="T_hat", save_path=os.path.join(save_dir, "high_amp_T.png") if save_dir else None)

    print(f" -> Plotting low-amplitude windows ({primary_model_name})...")
    plot_low_amp(true_obj, compare_obj=primary_pred_obj, var_name="v_hat", save_path=os.path.join(save_dir, "low_amp_v.png") if save_dir else None)
    plot_low_amp(true_obj, compare_obj=primary_pred_obj, var_name="T_hat", save_path=os.path.join(save_dir, "low_amp_T.png") if save_dir else None)

    print(" -> Plotting Master Context Trajectories...")
    snap_path = os.path.join(save_dir, "snapshot_v.png") if save_dir else None
    plot_full_context_trajectory(true_obj, var_name="v_hat", save_path=snap_path.replace("v.png", "v_full_context.png") if snap_path else None)
    plot_full_context_trajectory(true_obj, var_name="T_hat", save_path=snap_path.replace("v.png", "T_full_context.png") if snap_path else None)

    print(" -> Plotting Multi-Method Error Trajectories...")
    comp_path = os.path.join(save_dir, "error_traj_nodes.png") if save_dir else None
    compare_errors(models_dict, true_obj, window=window, var_name="v_hat", save_path=comp_path)

    print(" -> Plotting Multi-Method PDF Distributions...")
    pdf_path = os.path.join(save_dir, "multi_pdf_nodes.png") if save_dir else None
    multi_pdf(models_dict, true_obj, var_name="v_hat", save_path=pdf_path)
    multi_pdf(models_dict, true_obj, var_name="T_hat", save_path=pdf_path)

    plot_focused_extremes(true_obj, primary_pred_obj, save_dir=save_dir)
    plot_focused_extremes(true_obj, primary_pred_obj, save_dir=save_dir)


    if include_animation:
        print(f" -> Creating Spatial Reconstruction Animation ({primary_model_name})...")
        animate_spatial_reconstruction(primary_pred_obj, true_obj, nx=200, save_path=os.path.join(save_dir, "spatial_reconstruction_animation.mp4") if save_dir else None)
        
    print("\n" + "="*60)
    print("Diagnostics Complete!")
    if save_dir:
        print(f"All files successfully saved to: {os.path.abspath(save_dir)}")
    print("="*60)


def run_l63_dashboard(true_obj, pred_obj, window=None, save_dir=None):
    """
    Runs diagnostics and visualizations for a Lorenz-63 DA run.
    """
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)

    print("="*60)
    print("--- LORENZ-63 DA DIAGNOSTICS ---")
    print("="*60)

    # ---------------------------------------------------------
    # 1. Compute and Print Quantitative Diagnostics
    # ---------------------------------------------------------
    print("\n1. Calculating Statistical Metrics...")
    metrics = compute_diagnostics(true_obj, pred_obj)
    
    try:
        df_metrics = pd.DataFrame(metrics).T
        df_metrics.columns = ['RMSE', 'Correlation', 'KL Divergence']
        print("\n--- Error Metrics & KL Divergence ---")
        print(df_metrics.round(4).to_string())
        
        if save_dir:
            csv_path = os.path.join(save_dir, "diagnostics_summary.csv")
            df_metrics.to_csv(csv_path)
            print(f"[*] Saved metrics to {csv_path}")
    except ImportError:
        print(metrics)

    # ---------------------------------------------------------
    # 2. Extract States
    # ---------------------------------------------------------
    t_vals = true_obj.t
    
    x_true = true_obj.U.flatten()
    y_true = true_obj.mu[:, 0]
    z_true = true_obj.mu[:, 1]
    
    x_pred = pred_obj.U.flatten()
    y_pred = pred_obj.mu[:, 0]
    z_pred = pred_obj.mu[:, 1]

    y_std = np.sqrt(pred_obj.cov[:, 0, 0])
    z_std = np.sqrt(pred_obj.cov[:, 1, 1])

    if window is not None:
        idx_start, idx_end = window
        t_plot = t_vals[idx_start:idx_end]
        s = slice(idx_start, idx_end)
    else:
        t_plot = t_vals
        s = slice(None)

    # ---------------------------------------------------------
    # 3. Visualizations
    # ---------------------------------------------------------
    print("\n2. Generating Visualizations...")

    print(" -> Plotting 3D Phase Space...")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot(x_true[s], y_true[s], z_true[s], color='gray', alpha=0.6, label='True Attractor', lw=1)
    ax.plot(x_pred[s], y_pred[s], z_pred[s], color='red', alpha=0.8, label='CGNF Estimate', lw=1, linestyle='--')
    ax.set_xlabel('X (Observed)')
    ax.set_ylabel('Y (Hidden)')
    ax.set_zlabel('Z (Hidden)')
    ax.set_title("Lorenz-63 Phase Space: Truth vs CGNF")
    ax.legend()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "l63_phase_space.png"), dpi=300)
    plt.close()

    print(" -> Plotting Time Series (Hidden States Y and Z)...")
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    axes[0].plot(t_plot, y_true[s], label='True Y', color='black', lw=1.5)
    axes[0].plot(t_plot, y_pred[s], label='Estimated Y', color='blue', lw=1.5, linestyle='--')
    axes[0].fill_between(t_plot, (y_pred - 2*y_std)[s], (y_pred + 2*y_std)[s], color='blue', alpha=0.2, label='$\\pm 2\\sigma$')
    axes[0].set_ylabel('Y')
    axes[0].legend(loc='upper right')
    axes[0].set_title('Hidden State Estimation')

    axes[1].plot(t_plot, z_true[s], label='True Z', color='black', lw=1.5)
    axes[1].plot(t_plot, z_pred[s], label='Estimated Z', color='green', lw=1.5, linestyle='--')
    axes[1].fill_between(t_plot, (z_pred - 2*z_std)[s], (z_pred + 2*z_std)[s], color='green', alpha=0.2, label='$\\pm 2\\sigma$')
    axes[1].set_ylabel('Z')
    axes[1].set_xlabel('Time')
    axes[1].legend(loc='upper right')

    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, "l63_timeseries.png"), dpi=300)
    plt.close()

    print("\n" + "="*60)
    print("Diagnostics Complete!")
    if save_dir:
        print(f"All files successfully saved to: {os.path.abspath(save_dir)}")
    print("="*60)