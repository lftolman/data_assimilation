import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from ..utils.utils import extract_terms, extract_terms_fast, extract_terms2
from scipy.linalg import solve_continuous_are


def rhs(mu, R, dU_t, B0, B1, A0, A1, Gamma_inv):
    """
    Compute RHS using precalculated matrices. 
    No function evaluations happen here, just pure matrix math.
    """
    innovation = dU_t - (B0 + B1 @ mu)
    K = R @ B1.T @ Gamma_inv       
    dmu = A1 @ mu + A0 + K @ innovation
    return dmu


def CGKF_constR(d_uI, d_uII, t_span, uI, uII_0, Sigma, Gamma, R = "ricatti", method = "euler"):
    """
    Conditionally Gaussian Kalman Filter with a CONSTANT R matrix.
    d_uI, d_uII: functions for the model dynamics
    t_span: (t0, tf)
    uI: observed U(t)
    uII_0: initial guess for the hidden state
    Sigma: process noise covariance
    Gamma: observation noise covariance
    R: "ricatti" (default) to solve the algebraic Riccati equation for the steady-state R,
        "burnin" to run the filter for a while and use the final R,
        "burnin_avg" to run the filter and average R over the last half of the burn-in,
        or a constant matrix to use directly.
    method: "euler" or "RK4" for the ODE integration method.
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

    # Observation inverse covariance
    Gamma_inv = np.linalg.inv(Gamma@Gamma.T + 1e-12*np.eye(m))

    # Precompute dU/dt
    dUdt = np.gradient(U, t_vals, axis=0)

    dummy_mu = np.zeros(d) 
    
    if R == "ricatti":
        uI_mean = np.mean(uI)
        A0, A1 = extract_terms2(d_uII, 0, uI_mean, dummy_mu)
        B0, B1 = extract_terms2(d_uI, 0, uI_mean, dummy_mu)
        R = solve_continuous_are(A1.T, B1.T, Sigma, Gamma@Gamma.T)

    print("Using constant R matrix with shape:", R.shape)
    # Output arrays
    mu_hist = np.zeros((n, d))
    # Initial values
    mu = np.array(uII_0, dtype=float)
    mu_hist[0] = mu

    if method == "euler":
        # Time stepping (Forward Euler)
        for k in tqdm(range(1, n), desc="Euler"):
            t = t_vals[k-1]

            # U and derivative at the current step
            U_t = U[k-1]
            dU_t = dUdt[k-1]

            # ---- Forward Euler step ----
            # Calculate the derivatives (Right-Hand Side)
            dmu = rhs(mu, R, dU_t, B0, B1, A0, A1, Gamma_inv)

            # Update the state: μ_{k+1} = μ_k + dt * dμ
            mu = mu + dt * dmu

            mu_hist[k] = mu

    if method == "RK4":
        # Time stepping
        for k in tqdm(range(1, n), desc="RK4"):

            t = t_vals[k-1]

            # U and derivative at the current step
            U_t = U[k-1]
            dU_t = dUdt[k-1]

            # ---- RK4 stage 1 ----
            k1_mu = rhs(mu, R, U_t, dU_t, d_uI, d_uII, Gamma_inv, t)

            # ---- RK4 stage 2 ----
            mu2 = mu + 0.5 * dt * k1_mu
            k2_mu = rhs(mu2, R, U_t, dU_t, d_uI, d_uII, Gamma_inv, t + 0.5*dt)

            # ---- RK4 stage 3 ----
            mu3 = mu + 0.5 * dt * k2_mu
            k3_mu = rhs(mu3, R, U_t, dU_t, d_uI, d_uII, Gamma_inv, t + 0.5*dt)

            # ---- RK4 stage 4 ----
            U_t2  = U[k]              # use the next observed U

            dU_t2 = dUdt[k]
            mu4 = mu + dt * k3_mu
            k4_mu = rhs(mu4, R,  U_t2, dU_t2, d_uI, d_uII, Sigma, Gamma_inv, t + dt)

            # ---- Combine RK4 ----
            mu = mu + (dt/6)*(k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)
  

            mu_hist[k] = mu

    return {
        "t": t_vals,
        "uII": mu_hist.T   # match your previous shape
    }
