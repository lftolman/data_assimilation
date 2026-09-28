
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from ..utils.utils import precompute_matrices
from ..result import AssimilationResult
from time import perf_counter

def nudging_rhs(mu, dU_t, B0, B1, A0, A1, K):
    """
    Compute RHS for Nudging (Newtonian Relaxation) using precalculated matrices. 
    """
    innovation = dU_t - (B0 + B1 @ mu)
    dmu = A1 @ mu + A0 + K @ innovation
    return dmu

def nudging(d_uI, d_uII, t_span, uI, uII_0, K, method="euler", precomputed_matrices=None):
    """
    Continuous Data Assimilation via Nudging (Newtonian Relaxation).
    d_uI, d_uII: functions for the model dynamics
    t_span: array of times (fixed grid)
    uI: observed U(t)
    uII_0: initial guess for the hidden state
    K: fixed nudging gain matrix of shape (d, m)
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
    
    # Ensure K is the correct shape
    K = np.asarray(K, dtype=float)
    if K.shape != (d, m):
        raise ValueError(f"Nudging matrix K must have shape ({d}, {m}), but got {K.shape}")

    if precomputed_matrices is not None:
        A0_seq, A1_seq, B0_seq, B1_seq = precomputed_matrices
    else:
        A0_seq, A1_seq, B0_seq, B1_seq = precompute_matrices(d_uI, d_uII, t_vals, U, d, m)

    # Precompute dU/dt
    dUdt = np.gradient(U, t_vals, axis=0)

    # Output arrays
    mu_hist = np.zeros((n, d))

    # Initial values
    mu = np.array(uII_0, dtype=float)
    mu_hist[0] = mu

    start_time = perf_counter()

    if method == "euler":
        # Time stepping (Forward Euler)
        for k in tqdm(range(1, n), desc="Nudging Euler"):
            
            # Arrays at current step
            dU_t = dUdt[k-1]
            B0, B1 = B0_seq[k-1], B1_seq[k-1]
            A0, A1 = A0_seq[k-1], A1_seq[k-1]

            # ---- Forward Euler step ----
            dmu = nudging_rhs(mu, dU_t, B0, B1, A0, A1, K)

            mu = mu + dt * dmu
            mu_hist[k] = mu

    elif method == "RK4":
        # Time stepping (Runge-Kutta 4)
        for k in tqdm(range(1, n), desc="Nudging RK4"):

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
            k1_mu = nudging_rhs(mu, dU_t1, B0_1, B1_1, A0_1, A1_1, K)

            # ---- RK4 stage 2 (at t + 0.5*dt) ----
            mu2 = mu + 0.5 * dt * k1_mu
            k2_mu = nudging_rhs(mu2, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, K)

            # ---- RK4 stage 3 (at t + 0.5*dt) ----
            mu3 = mu + 0.5 * dt * k2_mu
            k3_mu = nudging_rhs(mu3, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, K)

            # ---- RK4 stage 4 (at t + dt) ----
            mu4 = mu + dt * k3_mu
            k4_mu = nudging_rhs(mu4, dU_t2, B0_2, B1_2, A0_2, A1_2, K)

            # ---- Combine RK4 ----
            mu = mu + (dt/6) * (k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)
            mu_hist[k] = mu

    end_time = perf_counter()
    comp_time = end_time - start_time
    print(f"Nudging completed in {comp_time:.2f} seconds using method: {method.upper()}.")
    
    # Returning cov=None since nudging doesn't compute uncertainty
    return AssimilationResult(t_vals, U, mu=mu_hist, cov=None, system="Barotropic", comp_time=comp_time)