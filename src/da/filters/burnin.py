import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from ..utils.utils import precompute_matrices
from ..result import AssimilationResult
from time import perf_counter

def cgkf_rhs(mu, R, dU_t, B0, B1, A0, A1, Sigma, Gamma_inv):
    """
    Compute RHS for full Conditionally Gaussian Kalman Filter.
    """
    innovation = dU_t - (B0 + B1 @ mu)
    K_gain = R @ B1.T @ Gamma_inv       
    dmu = A1 @ mu + A0 + K_gain @ innovation
    dR = A1 @ R + R @ A1.T + Sigma - R @ B1.T @ Gamma_inv @ B1 @ R
    return dmu, dR

def nudging_rhs(mu, dU_t, B0, B1, A0, A1, R_avg, Gamma_inv):
    """
    Compute RHS for Nudging (Newtonian Relaxation).
    K is computed dynamically using the steady-state R_avg and time-varying B1.
    """
    K = R_avg @ B1.T @ Gamma_inv
    innovation = dU_t - (B0 + B1 @ mu)
    dmu = A1 @ mu + A0 + K @ innovation
    return dmu

def burn_in_CGKF(d_uI, d_uII, t_span, uI, uII_0, R0, Sigma, Gamma, burn_in_steps, method="euler", precomputed_matrices=None):
    """
    Continuous Data Assimilation with a CGKF burn-in period, transitioning to Nudging.
    
    burn_in_steps: Integer number of steps to run the full CGKF.
                   The average of the last 1/3 of R over this period becomes the steady-state R.
    """
    t0, tf = t_span
    n = len(uI)
    d = len(uII_0)
    t_vals = np.linspace(t0, tf, n)
    uI = np.asarray(uI)
    dt = t_vals[1] - t_vals[0]
 
    if not (0 < burn_in_steps < n):
        raise ValueError(f"burn_in_steps must be between 1 and {n-1}.")

    # Observation dimension
    if uI.ndim == 1:
        m = 1
        U = uI.reshape(n, 1)
    else:
        m = uI.shape[1]
        U = uI

    if precomputed_matrices is not None:
        A0_seq, A1_seq, B0_seq, B1_seq = precomputed_matrices
    else:
        A0_seq, A1_seq, B0_seq, B1_seq = precompute_matrices(d_uI, d_uII, t_vals, U, d, m)

    # Observation inverse covariance
    if np.sum(np.abs(Gamma)) < 1e-10:
        Gamma_inv = np.zeros((m, m))  
    else:   
        Gamma_inv = np.linalg.inv(Gamma @ Gamma.T + 1e-12 * np.eye(m))

    # Precompute dU/dt
    dUdt = np.gradient(U, t_vals, axis=0)

    # Output arrays
    mu_hist = np.zeros((n, d))
    R_hist = np.zeros((n, d, d))

    # Initial values
    mu = np.array(uII_0, dtype=float)
    R = np.array(R0, dtype=float)
    
    mu_hist[0] = mu
    R_hist[0] = R

    start_time = perf_counter()

    if method == "euler":
        # ==========================================
        # PHASE 1: BURN-IN (CGKF)
        # ==========================================
        for k in tqdm(range(1, burn_in_steps), desc="Burn-in Euler (CGKF)"):
            dU_t = dUdt[k-1]
            B0, B1 = B0_seq[k-1], B1_seq[k-1]
            A0, A1 = A0_seq[k-1], A1_seq[k-1]

            dmu, dR = cgkf_rhs(mu, R, dU_t, B0, B1, A0, A1, Sigma, Gamma_inv)
            
            mu = mu + dt * dmu
            R = R + dt * dR
            R = 0.5 * (R + R.T)

            mu_hist[k] = mu
            R_hist[k] = R

        # Compute Steady-State R from the last 1/3 of the burn-in phase
        avg_start = max(1, int(burn_in_steps - (burn_in_steps / 3)))
        R_avg = np.mean(R_hist[avg_start:burn_in_steps], axis=0)

        # ==========================================
        # PHASE 2: NUDGING
        # ==========================================
        for k in tqdm(range(burn_in_steps, n), desc="Nudging Euler"):
            dU_t = dUdt[k-1]
            B0, B1 = B0_seq[k-1], B1_seq[k-1]
            A0, A1 = A0_seq[k-1], A1_seq[k-1]

            dmu = nudging_rhs(mu, dU_t, B0, B1, A0, A1, R_avg, Gamma_inv)

            mu = mu + dt * dmu
            mu_hist[k] = mu
            R_hist[k] = R_avg  # Pad remainder so downstream plots don't crash/drop to zero

    elif method == "RK4":
        # ==========================================
        # PHASE 1: BURN-IN (CGKF)
        # ==========================================
        for k in tqdm(range(1, burn_in_steps), desc="Burn-in RK4 (CGKF)"):
            dU_t1, dU_t2 = dUdt[k-1], dUdt[k]
            B0_1, B1_1 = B0_seq[k-1], B1_seq[k-1]
            A0_1, A1_1 = A0_seq[k-1], A1_seq[k-1]
            B0_2, B1_2 = B0_seq[k], B1_seq[k]
            A0_2, A1_2 = A0_seq[k], A1_seq[k]

            dU_mid = 0.5 * (dU_t1 + dU_t2)
            B0_mid, B1_mid = 0.5 * (B0_1 + B0_2), 0.5 * (B1_1 + B1_2)
            A0_mid, A1_mid = 0.5 * (A0_1 + A0_2), 0.5 * (A1_1 + A1_2)

            k1_mu, k1_R = cgkf_rhs(mu, R, dU_t1, B0_1, B1_1, A0_1, A1_1, Sigma, Gamma_inv)
            
            mu2, R2 = mu + 0.5*dt*k1_mu, R + 0.5*dt*k1_R
            k2_mu, k2_R = cgkf_rhs(mu2, R2, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, Sigma, Gamma_inv)
            
            mu3, R3 = mu + 0.5*dt*k2_mu, R + 0.5*dt*k2_R
            k3_mu, k3_R = cgkf_rhs(mu3, R3, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, Sigma, Gamma_inv)
            
            mu4, R4 = mu + dt*k3_mu, R + dt*k3_R
            k4_mu, k4_R = cgkf_rhs(mu4, R4, dU_t2, B0_2, B1_2, A0_2, A1_2, Sigma, Gamma_inv)

            mu = mu + (dt/6) * (k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)
            R  = R  + (dt/6) * (k1_R  + 2*k2_R  + 2*k3_R  + k4_R)
            R = 0.5 * (R + R.T)

            mu_hist[k] = mu
            R_hist[k] = R

        # Compute Steady-State R from the last 1/3 of the burn-in phase
        avg_start = max(1, int(burn_in_steps - (burn_in_steps / 3)))
        R_avg = np.mean(R_hist[avg_start:burn_in_steps], axis=0)

        # ==========================================
        # PHASE 2: NUDGING
        # ==========================================
        for k in tqdm(range(burn_in_steps, n), desc="Nudging RK4"):
            dU_t1, dU_t2 = dUdt[k-1], dUdt[k]
            B0_1, B1_1 = B0_seq[k-1], B1_seq[k-1]
            A0_1, A1_1 = A0_seq[k-1], A1_seq[k-1]
            B0_2, B1_2 = B0_seq[k], B1_seq[k]
            A0_2, A1_2 = A0_seq[k], A1_seq[k]

            dU_mid = 0.5 * (dU_t1 + dU_t2)
            B0_mid, B1_mid = 0.5 * (B0_1 + B0_2), 0.5 * (B1_1 + B1_2)
            A0_mid, A1_mid = 0.5 * (A0_1 + A0_2), 0.5 * (A1_1 + A1_2)

            k1_mu = nudging_rhs(mu, dU_t1, B0_1, B1_1, A0_1, A1_1, R_avg, Gamma_inv)
            
            mu2 = mu + 0.5*dt*k1_mu
            k2_mu = nudging_rhs(mu2, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, R_avg, Gamma_inv)
            
            mu3 = mu + 0.5*dt*k2_mu
            k3_mu = nudging_rhs(mu3, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, R_avg, Gamma_inv)
            
            mu4 = mu + dt*k3_mu
            k4_mu = nudging_rhs(mu4, dU_t2, B0_2, B1_2, A0_2, A1_2, R_avg, Gamma_inv)

            mu = mu + (dt/6) * (k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)
            
            mu_hist[k] = mu
            R_hist[k] = R_avg  # Pad remainder 

    end_time = perf_counter()
    comp_time = end_time - start_time
    print(f"Hybrid Assimilation completed in {comp_time:.2f} seconds using method: {method.upper()}.")
    
    # We return the R_hist which contains dynamic covariance during burn-in, and constant R_avg thereafter.
    return AssimilationResult(t_vals, U, mu=mu_hist, cov=R_hist, system="Barotropic", comp_time=comp_time)