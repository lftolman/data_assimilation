import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from ..utils.utils import precompute_matrices
from scipy.linalg import solve_continuous_are
from ..result import AssimilationResult
from time import perf_counter

def rhs(mu, R, dU_t, B0, B1, A0, A1, Sigma2, Gamma_inv):
    """
    Compute RHS using precalculated matrices. 
    """
    innovation = dU_t - (B0 + B1 @ mu)
    K = R @ B1.T @ Gamma_inv       
    dmu = A1 @ mu + A0 + K @ innovation
    dR = A1 @ R + R @ A1.T + Sigma2 - R @ B1.T @ Gamma_inv @ B1 @ R
    return dmu, dR

def CGKF(d_uI, d_uII, t_span, uI, uII_0, R0, Sigma, Gamma, method="euler", precomputed_matrices=None):
    """
    Conditionally Gaussian Kalman Filter.
    d_uI, d_uII: functions for the model dynamics
    t_span: array of times (fixed grid)
    uI: observed U(t)
    uII_0: initial guess for the hidden state
    R0: initial covariance matrix
    Sigma: process noise covariance
    Gamma: observation noise covariance
    method: "euler" or "RK4" for the ODE integration method.
    precomputed_matrices: (A0_seq, A1_seq, B0_seq, B1_seq) precomputed for the entire time series.
    """
    t0, tf = t_span
    n = len(uI)
    d = len(uII_0)
    t_vals = np.linspace(t0, tf, n)
    uI = np.asarray(uI)
    dt = t_vals[1] - t_vals[0]
 
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
            Gamma_inv =  np.zeros((m, m))  # High confidence in observations
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
        # Time stepping (Forward Euler)
        for k in tqdm(range(1, n), desc="Euler"):
            
            # Arrays at current step
            dU_t = dUdt[k-1]
            B0, B1 = B0_seq[k-1], B1_seq[k-1]
            A0, A1 = A0_seq[k-1], A1_seq[k-1]

            # ---- Forward Euler step ----
            dmu, dR = rhs(mu, R, dU_t, B0, B1, A0, A1, Sigma, Gamma_inv)

            mu = mu + dt * dmu
            R = R + dt * dR

            # Symmetrize R
            R = 0.5 * (R + R.T)

            mu_hist[k] = mu
            R_hist[k] = R

    elif method == "RK4":
        # Time stepping (Runge-Kutta 4)
        for k in tqdm(range(1, n), desc="RK4"):

            # Data at t
            dU_t1 = dUdt[k-1]
            B0_1, B1_1 = B0_seq[k-1], B1_seq[k-1]
            A0_1, A1_1 = A0_seq[k-1], A1_seq[k-1]

            # Data at t + dt
            dU_t2 = dUdt[k]
            B0_2, B1_2 = B0_seq[k], B1_seq[k]
            A0_2, A1_2 = A0_seq[k], A1_seq[k]

            # Interpolate data at t + 0.5*dt (For RK4 intermediate stages)
            dU_mid = 0.5 * (dU_t1 + dU_t2)
            B0_mid, B1_mid = 0.5 * (B0_1 + B0_2), 0.5 * (B1_1 + B1_2)
            A0_mid, A1_mid = 0.5 * (A0_1 + A0_2), 0.5 * (A1_1 + A1_2)

            # ---- RK4 stage 1 (at t) ----
            k1_mu, k1_R = rhs(mu, R, dU_t1, B0_1, B1_1, A0_1, A1_1, Sigma, Gamma_inv)

            # ---- RK4 stage 2 (at t + 0.5*dt) ----
            mu2 = mu + 0.5 * dt * k1_mu
            R2 = R + 0.5 * dt * k1_R
            k2_mu, k2_R = rhs(mu2, R2, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, Sigma, Gamma_inv)

            # ---- RK4 stage 3 (at t + 0.5*dt) ----
            mu3 = mu + 0.5 * dt * k2_mu
            R3 = R + 0.5 * dt * k2_R
            k3_mu, k3_R = rhs(mu3, R3, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, Sigma, Gamma_inv)

            # ---- RK4 stage 4 (at t + dt) ----
            mu4 = mu + dt * k3_mu
            R4 = R + dt * k3_R
            k4_mu, k4_R = rhs(mu4, R4, dU_t2, B0_2, B1_2, A0_2, A1_2, Sigma, Gamma_inv)

            # ---- Combine RK4 ----
            mu = mu + (dt/6) * (k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)
            R  = R  + (dt/6) * (k1_R  + 2*k2_R  + 2*k3_R  + k4_R)

            # Symmetrize R
            R = 0.5 * (R + R.T)

            mu_hist[k] = mu
            R_hist[k] = R
    end_time = perf_counter()
    comp_time = end_time - start_time
    print(f"CGKF completed in {comp_time:.2f} seconds using method: {method.upper()}.")
    return AssimilationResult(t_vals, U,mu = mu_hist, cov=R_hist, system = "Barotropic", comp_time=comp_time)