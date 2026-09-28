import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from scipy.linalg import solve_continuous_are
from functools import partial
from ..utils.utils import precompute_matrices
from ..result import AssimilationResult
from time import perf_counter

def nudging_rhs(mu, dU_t, B0, B1, A0, A1, R_inf, Gamma_inv):
    """
    Compute RHS for Nudging using the globally constant Riccati covariance R_inf.
    """
    # Calculate the instantaneous gain matrix K(t) using the steady-state R_inf
    K = R_inf @ B1.T @ Gamma_inv
    
    innovation = dU_t - (B0 + B1 @ mu)
    dmu = A1 @ mu + A0 + K @ innovation
    return dmu

def riccati_nudging(d_uI, d_uII, t_span, uI, uII_0, Sigma, Gamma, method="euler", precomputed_matrices=None):
    """
    Continuous Data Assimilation via Nudging using the Algebraic Riccati Equation.
    Calculates a constant covariance matrix R by averaging A1 and B1 over time.
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

    # Precompute dU/dt
    dUdt = np.gradient(U, t_vals, axis=0)

    # ---------------------------------------------------------
    # RICCATI EQUATION SOLVER
    # ---------------------------------------------------------
    start_time = perf_counter()
    print("Solving Continuous Algebraic Riccati Equation...")
    
    # Expected values over time
    A1_bar = np.mean(A1_seq, axis=0)
    B1_bar = np.mean(B1_seq, axis=0)
    
    # Observation noise covariance and inverse
    R_obs = Gamma @ Gamma.T + 1e-12 * np.eye(m)
    if np.sum(np.abs(Gamma)) < 1e-10:
        Gamma_inv = np.zeros((m, m))  # Near-perfect observations: matches CGKF/burn_in_CGKF
    else:
        Gamma_inv = np.linalg.inv(R_obs)
    
    # Map to scipy.linalg.solve_continuous_are
    # scipy solves: A^T X + X A - X B R^-1 B^T X + Q = 0
    R_inf = solve_continuous_are(
        a=A1_bar.T, 
        b=B1_bar.T, 
        q=Sigma, 
        r=R_obs
    )
    # Ensure symmetry
    R_inf = 0.5 * (R_inf + R_inf.T)
    # ---------------------------------------------------------

    # Output arrays
    mu_hist = np.zeros((n, d))
    
    # We can pack the constant R_inf into R_hist so the AssimilationResult 
    # visualizer plots uniform, stable uncertainty bounds.
    R_hist = np.tile(R_inf, (n, 1, 1)) 

    # Initial values
    mu = np.array(uII_0, dtype=float)
    mu_hist[0] = mu

    if method == "euler":
        for k in tqdm(range(1, n), desc="Riccati Euler"):
            dU_t = dUdt[k-1]
            B0, B1 = B0_seq[k-1], B1_seq[k-1]
            A0, A1 = A0_seq[k-1], A1_seq[k-1]

            dmu = nudging_rhs(mu, dU_t, B0, B1, A0, A1, R_inf, Gamma_inv)
            
            mu = mu + dt * dmu
            mu_hist[k] = mu

    elif method == "RK4":
        for k in tqdm(range(1, n), desc="Riccati RK4"):
            dU_t1, dU_t2 = dUdt[k-1], dUdt[k]
            B0_1, B1_1 = B0_seq[k-1], B1_seq[k-1]
            A0_1, A1_1 = A0_seq[k-1], A1_seq[k-1]
            B0_2, B1_2 = B0_seq[k], B1_seq[k]
            A0_2, A1_2 = A0_seq[k], A1_seq[k]

            dU_mid = 0.5 * (dU_t1 + dU_t2)
            B0_mid, B1_mid = 0.5 * (B0_1 + B0_2), 0.5 * (B1_1 + B1_2)
            A0_mid, A1_mid = 0.5 * (A0_1 + A0_2), 0.5 * (A1_1 + A1_2)

            k1_mu = nudging_rhs(mu, dU_t1, B0_1, B1_1, A0_1, A1_1, R_inf, Gamma_inv)
            
            mu2 = mu + 0.5 * dt * k1_mu
            k2_mu = nudging_rhs(mu2, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, R_inf, Gamma_inv)
            
            mu3 = mu + 0.5 * dt * k2_mu
            k3_mu = nudging_rhs(mu3, dU_mid, B0_mid, B1_mid, A0_mid, A1_mid, R_inf, Gamma_inv)
            
            mu4 = mu + dt * k3_mu
            k4_mu = nudging_rhs(mu4, dU_t2, B0_2, B1_2, A0_2, A1_2, R_inf, Gamma_inv)

            mu = mu + (dt/6) * (k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)
            mu_hist[k] = mu

    end_time = perf_counter()
    comp_time = end_time - start_time
    print(f"Riccati Nudging completed in {comp_time:.2f} seconds using method: {method.upper()}.")
    
    return AssimilationResult(t_vals, U, mu=mu_hist, cov=R_hist, system="Barotropic", comp_time=comp_time)