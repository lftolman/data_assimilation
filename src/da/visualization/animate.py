# import matplotlib.animation as animation

# # --- 1. Animation Parameters ---
# mode_idx = 0  # Which mode to animate (e.g., 0 for v_1)
# window_points = 10000  # Number of data points to show in the window at one time
# frame_step = 50       # Skip frames to speed up the animation (adjust as needed)
# fps = 50             # Frames per second for the saved video

# # --- 2. Setup Figure and Axes ---
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), dpi=150)
# fig.suptitle(f"Time-Evolution of Mode $v_{{{mode_idx+1}}}$", fontsize=20)

# # --- 3. Plot the FULL Data Once ---
# # Real Part
# std_real_v = np.sqrt(Rs[:, mode_idx, mode_idx])
# ax1.plot(t_true, std_real_v + np.real(vk_real[mode_idx]), linestyle='-', color='lightgray')
# ax1.plot(t_true, np.real(vk_real[mode_idx]) - std_real_v, linestyle='-', color='lightgray')
# ax1.fill_between(t_true, np.real(vk_real[mode_idx]) - std_real_v, np.real(vk_real[mode_idx]) + std_real_v, color='lightgray', alpha=0.5, label="Std Dev Band")
# ax1.plot(t_true, np.real(v_hat_true[mode_idx, :]), label=f"True $v_{{{mode_idx+1}}}$", color='purple')
# ax1.plot(results["t"], np.real(vk_real[mode_idx]), label=f"Estimated $\\hat{{v}}_{{{mode_idx+1}}}$", linewidth=1, color='black')

# ax1.set_title(f"$\\Re (v_{{{mode_idx+1}}})$", fontsize=16)
# ax1.legend(loc='upper right', fontsize=12)
# ax1.grid(True, alpha=0.3)

# # Imaginary Part
# std_imag_v = np.sqrt(Rs[:, mode_idx + n_modes, mode_idx + n_modes])
# ax2.plot(t_true, std_imag_v + np.imag(vk_imag[mode_idx]), linestyle='-', color='lightgray')
# ax2.plot(t_true, np.imag(vk_imag[mode_idx]) - std_imag_v, linestyle='-', color='lightgray')
# ax2.fill_between(t_true, np.imag(vk_imag[mode_idx]) - std_imag_v, np.imag(vk_imag[mode_idx]) + std_imag_v, color='lightgray', alpha=0.5, label="Std Dev Band")
# ax2.plot(t_true, np.imag(v_hat_true[mode_idx, :]), label=f"True $v_{{{mode_idx+1}}}$", color='purple')
# ax2.plot(results["t"], np.imag(vk_imag[mode_idx]), label=f"Estimated $\\hat{{v}}_{{{mode_idx+1}}}$", linewidth=1, color='black')

# ax2.set_title(f"$\\Im (v_{{{mode_idx+1}}})$", fontsize=16)
# ax2.legend(loc='upper right', fontsize=12)
# ax2.grid(True, alpha=0.3)

# # Fix the y-axis limits so they don't bounce around during the animation
# ax1.set_ylim(np.min(np.real(v_hat_true[mode_idx, :])) - 1, np.max(np.real(v_hat_true[mode_idx, :])) + 1)
# ax2.set_ylim(np.min(np.imag(v_hat_true[mode_idx, :])) - 1, np.max(np.imag(v_hat_true[mode_idx, :])) + 1)

# plt.tight_layout()

# # --- 4. Define the Animation Update Function ---
# def update(frame):
#     # 'frame' is the starting index of our sliding window
#     t_start = t_true[frame]
#     t_end = t_true[frame + window_points]
    
#     # Update the x-axis limits to pan the camera
#     ax1.set_xlim(t_start, t_end)
#     ax2.set_xlim(t_start, t_end)
    
#     return ax1, ax2

# # Calculate how many frames we can render before the window hits the end of the data
# total_possible_frames = len(t_true) - window_points
# frames_to_render = range(0, total_possible_frames, frame_step)

# # --- 5. Create and Save the Animation ---
# ani = animation.FuncAnimation(
#     fig, 
#     update, 
#     frames=frames_to_render, 
#     interval=1000/fps, 
#     blit=False
# )

# ani.save('animation.mp4', writer='ffmpeg', fps=fps)
# # Display it in an interactive window (if running a local script)
# plt.show()

# # To save the animation, uncomment the line below (requires ffmpeg installed):
# # ani.save(f"outputs/v_{mode_idx+1}_animation.mp4", writer='ffmpeg', fps=fps)

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def animate_spatial_reconstruction(pred_obj, true_obj, nx=200, save_path=None, interval=50):
    """
    Animates the 1D spatial reconstruction of v(x) and T(x) over time.
    
    Parameters:
    pred_obj: AssimilationResult containing the filter estimates
    true_obj: AssimilationResult containing the truth
    nx: Number of spatial grid points to evaluate between 0 and 2*pi
    save_path: String path to save as '.mp4' or '.gif' (requires ffmpeg or imagemagick)
    interval: Delay between frames in milliseconds.
    """
    x = np.linspace(0, 2 * np.pi, nx)
    t_vals = true_obj.t
    n_steps = len(t_vals)
    d = pred_obj.v_hat.shape[1] // 2
    k_vec = np.arange(1, d + 1)
    kx = np.outer(k_vec, x) # Shape: (d, nx)

    # Pre-calculate ALL physical frames to make the animation loop fast
    def build_frames(hat_array):
        # hat_array is (time, 2*d). Split into real and imag.
        real_parts = hat_array[:, :d]  # (time, d)
        imag_parts = hat_array[:, d:]  # (time, d)
        
        # We need to broadcast across time, modes, and space
        # modes_k shape: (time, d, 1)
        modes_k = (real_parts + 1j * imag_parts)[:, :, np.newaxis]
        
        # kx shape: (1, d, nx)
        kx_3d = kx[np.newaxis, :, :]
        
        # sum over modes (axis=1) -> resulting shape (time, nx)
        complex_sum = np.sum(modes_k * np.exp(1j * kx_3d), axis=1)
        return 2 * np.real(complex_sum)

    print("Pre-computing spatial frames...")
    v_true_frames = build_frames(true_obj.v_hat)
    v_pred_frames = build_frames(pred_obj.v_hat)
    T_true_frames = build_frames(true_obj.T_hat)
    T_pred_frames = build_frames(pred_obj.T_hat)
    print("Done. Generating animation...")

    # Set up the figure and axes
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Initialize the lines
    line_v_true, = ax1.plot(x, v_true_frames[0], 'r-', linewidth=2, label="True $v(x)$")
    line_v_pred, = ax1.plot(x, v_pred_frames[0], 'k--', linewidth=2, label="Pred $v(x)$")
    
    line_T_true, = ax2.plot(x, T_true_frames[0], 'r-', linewidth=2, label="True $T(x)$")
    line_T_pred, = ax2.plot(x, T_pred_frames[0], 'k--', linewidth=2, label="Pred $T(x)$")

    # Formatting Velocity Axis
    ax1.set_title("Velocity")
    ax1.set_xlim(0, 2*np.pi)
    ax1.set_ylim(np.min(v_true_frames)*1.2, np.max(v_true_frames)*1.2)
    ax1.set_xlabel("Spatial coordinate ($x$)")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right")

    # Formatting Temperature Axis
    ax2.set_title("Temperature")
    ax2.set_xlim(0, 2*np.pi)
    ax2.set_ylim(np.min(T_true_frames)*1.2, np.max(T_true_frames)*1.2)
    ax2.set_xlabel("Spatial coordinate ($x$)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper right")
    
    plt.suptitle(f"Barotropic Fields at $t = {t_vals[0]:.2f}$", fontsize=14)

    time_text = fig.suptitle(f"Barotropic Fields at $t = {t_vals[0]:.2f}$", fontsize=14)

    def update(frame_idx):
        line_v_true.set_ydata(v_true_frames[frame_idx])
        line_v_pred.set_ydata(v_pred_frames[frame_idx])
        
        line_T_true.set_ydata(T_true_frames[frame_idx])
        line_T_pred.set_ydata(T_pred_frames[frame_idx])
        
        time_text.set_text(f"Barotropic Fields at $t = {t_vals[frame_idx]:.2f}$")
        plt.title(f"t = {t_vals[frame_idx]:.2f}", fontsize=14)
        return line_v_true, line_v_pred, line_T_true, line_T_pred, time_text

    anim = FuncAnimation(fig, update, frames=np.arange(n_steps)[::500], interval=interval, blit=True)
    
    plt.tight_layout()
    if save_path:
        print(f"Saving animation to {save_path}...")
        anim.save(save_path, fps=1000//interval)
        print("Saved successfully.")
    
    plt.show()
    return anim