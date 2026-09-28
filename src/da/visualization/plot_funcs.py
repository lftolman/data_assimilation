import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from scipy.signal import find_peaks

def plot_pdf(pred, true, var_name=None):
    """Plots the PDF using KDE. Cleaned up legend and added axis labels."""
    kde_true = gaussian_kde(true)
    kde_est = gaussian_kde(pred)
    
    x_min = min(true.min(), pred.min())
    x_max = max(true.max(), pred.max())
    x_vals = np.linspace(x_min, x_max, 300)
    
    plt.plot(x_vals, kde_true(x_vals), color='black', label=f"True {var_name}")
    plt.plot(x_vals, kde_est(x_vals), '--', color='gray', label=f"Estimated {var_name}")
    
    # Adding a subtle fill makes the overlap easier to see
    plt.fill_between(x_vals, kde_est(x_vals), color='gray', alpha=0.1)
    
    plt.title(f"{var_name} PDF")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True, alpha=0.2)

def plot_compare(t, pred, true, cov, window=None, var_name=None):
    """
    Plots the time-series comparison with a 1-sigma uncertainty band.
    """
    # 1. Handle Window Masking
    if window is not None:
        start_time, end_time = window
        mask = (t >= t[start_time]) & (t <= t[end_time])
        t, pred, true, cov = t[mask], pred[mask], true[mask], cov[mask]

    # 2. Calculate Standard Deviation (1-sigma)
    std = np.sqrt(cov)
    # 3. Plotting Order (Bottom to Top)
    # Shade the 1-sigma region around the PREDICTION (the filter's confidence)
    plt.fill_between(t, (pred - 2*std), (pred + 2*std), color='gray', alpha=0.3, label="2$\\sigma$ Confidence")
    
    # Plot True and Predicted
    # Using a line for pred is often clearer than dots for long time series
    plt.plot(t, true, label="Truth", color='red', linewidth=1.5)
    plt.plot(t, pred, label="Filter Mean", color='black', linestyle='--', alpha=0.8)

    plt.title(f"{var_name}")
    plt.xlabel("Time ($t$)")
    plt.ylabel("Amplitude")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)

def plot_multi_pdf(preds_dict, true, var_name=None):
    """
    Plots the PDF using KDE for the Truth and multiple estimation methods.
    preds_dict: Dictionary mapping method names to their prediction arrays.
    """
    kde_true = gaussian_kde(true)
    
    # Determine global min/max across all datasets for a shared X axis
    x_min, x_max = true.min(), true.max()
    for p in preds_dict.values():
        x_min, x_max = min(x_min, p.min()), max(x_max, p.max())
    
    x_vals = np.linspace(x_min, x_max, 500)
    
    # Plot Truth with a strong, solid line
    plt.plot(x_vals, kde_true(x_vals), color='black', linewidth=2.5, label=f"True {var_name}")
    plt.fill_between(x_vals, kde_true(x_vals), color='black', alpha=0.1)
    
    # Define a color palette for the methods
    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    for i, (name, pred) in enumerate(preds_dict.items()):
        kde_est = gaussian_kde(pred)
        c = colors[i % len(colors)]
        plt.plot(x_vals, kde_est(x_vals), '--', color=c, label=f"{name}")
        plt.fill_between(x_vals, kde_est(x_vals), color=c, alpha=0.15) # Low opacity for overlap
    
    plt.title(f"{var_name} PDF Comparison")
    plt.ylabel("Density")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.2)

def plot_error_trajectory(t, true, preds_dict, window=None, var_name=None):
    """
    Plots the error (Estimate - Truth) instead of overlaid time series.
    """
    if window is not None:
        start_time, end_time = window
        mask = (t >= t[start_time]) & (t <= t[end_time])
        t = t[mask]
        true = true[mask]
        preds_dict = {name: p[mask] for name, p in preds_dict.items()}

    colors = ['blue', 'red', 'green', 'orange', 'purple']
    
    # Plot a zero-error reference line
    plt.axhline(0, color='black', linewidth=1.5, linestyle='-', label="Zero Error (Truth)")
    
    for i, (name, pred) in enumerate(preds_dict.items()):
        c = colors[i % len(colors)]
        error = pred - true
        plt.plot(t, error, label=f"{name} Error", color=c, linewidth=1.2, alpha=0.8)

    plt.title(f"Error Trajectory: {var_name}")
    plt.xlabel("Time ($t$)")
    plt.ylabel(r"Error ($\hat{v} - v$)")
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.3)



def get_high_amplitude_windows(data_array, window_size=100000, num_windows=1):
    """
    Finds non-overlapping time windows with the highest sustained amplitude.
    
    Parameters:
    data_array: 2D numpy array of shape (n_timesteps, n_features)
    window_size: The total width of the window to return (e.g., 200 timesteps)
    num_windows: How many distinct windows to find
    
    Returns:
    windows: List of tuples [(start1, end1), (start2, end2), ...]
    smoothed_energy: 1D array of the energy metric (useful for plotting)
    """
    n_timesteps = data_array.shape[0]
    
    # 1. Calculate the amplitude metric (L2 norm / total energy across all modes)
    # If the mean is non-zero, it's safer to subtract it first to find fluctuations
    fluctuations = data_array - np.mean(data_array, axis=0)
    energy = np.sum(fluctuations**2, axis=1)
    
    # 2. Smooth the signal to find "sustained" events, not just 1-timestep spikes
    # We use a moving average window that is 1/4th the size of your target window
    smooth_width = max(1, window_size // 4)
    kernel = np.ones(smooth_width) / smooth_width
    smoothed_energy = np.convolve(energy, kernel, mode='same')
    
    # 3. Find the peaks
    # 'distance' ensures our peaks are at least one full window apart
    peaks, _ = find_peaks(smoothed_energy, distance=window_size)
    
    if len(peaks) == 0:
        print("Warning: No distinct peaks found. Returning largest values.")
        # Fallback if no peaks are found
        peaks = np.argsort(smoothed_energy)[-num_windows:]
        
    # 4. Sort peaks by their energy height and grab the top N
    peak_heights = smoothed_energy[peaks]
    top_peak_indices = np.argsort(peak_heights)[::-1][:num_windows]
    top_peaks = peaks[top_peak_indices]
    
    # 5. Build the (start, end) tuples
    windows = []
    half_win = window_size // 2
    
    for peak in top_peaks:
        start = max(0, peak - half_win)
        end = min(n_timesteps, peak + half_win)
        windows.append((int(start), int(end)))
        
    # Sort chronologically so they plot in order
    windows.sort(key=lambda x: x[0])
    
    return windows, smoothed_energy


def get_low_amplitude_windows(data_array, window_size=100000, num_windows=1):
    """
    Finds non-overlapping time windows with the lowest sustained amplitude (quiescent periods).
    
    Parameters:
    data_array: 2D numpy array of shape (n_timesteps, n_features)
    window_size: The total width of the window to return (e.g., 200 timesteps)
    num_windows: How many distinct windows to find
    
    Returns:
    windows: List of tuples [(start1, end1), (start2, end2), ...]
    smoothed_energy: 1D array of the energy metric (useful for plotting)
    """
    n_timesteps = data_array.shape[0]
    
    # 1. Calculate the amplitude metric (L2 norm / total energy across all modes)
    fluctuations = data_array - np.mean(data_array, axis=0)
    energy = np.sum(fluctuations**2, axis=1)
    
    # 2. Smooth the signal to find "sustained" quiet periods
    smooth_width = max(1, window_size // 4)
    kernel = np.ones(smooth_width) / smooth_width
    smoothed_energy = np.convolve(energy, kernel, mode='same')
    
    # 3. Find the valleys (local minima) by inverting the signal
    # find_peaks looks for local maxima, so we pass it -smoothed_energy
    valleys, _ = find_peaks(-smoothed_energy, distance=window_size)
    
    if len(valleys) == 0:
        print("Warning: No distinct valleys found. Returning lowest absolute values.")
        # Fallback: just get the indices of the lowest energy values
        valleys = np.argsort(smoothed_energy)[:num_windows]
        
    # 4. Sort valleys by their ACTUAL energy height (ascending order!)
    valley_heights = smoothed_energy[valleys]
    top_valley_indices = np.argsort(valley_heights)[:num_windows]
    top_valleys = valleys[top_valley_indices]
    
    # 5. Build the (start, end) tuples
    windows = []
    half_win = window_size // 2
    
    for valley in top_valleys:
        start = max(0, valley - half_win)
        end = min(n_timesteps, valley + half_win)
        windows.append((int(start), int(end)))
        
    # Sort chronologically so they plot in order
    windows.sort(key=lambda x: x[0])
    
    return windows, smoothed_energy