import numpy as np
from tqdm import tqdm
from .utils import extract_terms


def kalman_rhs(mu, R, U, dUdt, d_uI, d_uII, Sigma2, Gamma_inv, t):
    """
    Compute RHS for μ and R:
    dμ/dt = a1 μ + a0 + K(innovation)
    dR/dt = a1 R + R a1ᵀ + Σ2 - R A1ᵀ Σ^{-1} A1 R
    """

    # Linearizations
    B0, B1 = extract_terms(d_uI, t, U, mu)   # (m,), (m x d)
    A0, A1 = extract_terms(d_uII, t, U, mu)  # (d,), (d x d)

    B0 = np.asarray(B0).reshape(-1)
    B1 = np.asarray(B1)
    A0 = np.asarray(A0).reshape(-1)
    A1 = np.asarray(A1)

    # Innovation: dU/dt - (A0 + A1 mu)
    innovation = dUdt - (B0 + B1 @ mu)

    # Kalman gain
    K = R @ B1.T @ Gamma_inv       # (d x m)

    # Mean update
    dmu = A1 @ mu + A0 + K @ innovation

    # Riccati update
    dR = A1 @ R + R @ A1.T + Sigma2 - R @ B1.T @ Gamma_inv @ B1 @ R

    return dmu, dR


def kalman_bucy(d_uI, d_uII, t_span, uI, uII_0, R0, Sigma, Gamma):
    """
    RK4 version of Kalman-Bucy filter.
    t_vals: array of times (fixed grid)
    uI: observed U(t)
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

    # Output arrays
    mu_hist = np.zeros((n, d))
    R_hist = np.zeros((n, d, d))

    # Initial values
    mu = np.array(uII_0, dtype=float)
    R = np.array(R0, dtype=float)
    print(mu.shape, R.shape)

    mu_hist[0] = mu
    R_hist[0] = R

    # Time stepping
    for k in tqdm(range(1, n), desc="RK4"):

        t = t_vals[k-1]

        # U and derivative at the current step
        U_t = U[k-1]
        dU_t = dUdt[k-1]

        # ---- RK4 stage 1 ----
        k1_mu, k1_R = kalman_rhs(mu, R, U_t, dU_t, d_uI, d_uII, Sigma, Gamma_inv, t)

        # ---- RK4 stage 2 ----
        mu2 = mu + 0.5 * dt * k1_mu
        R2 = R + 0.5 * dt * k1_R
        k2_mu, k2_R = kalman_rhs(mu2, R2, U_t, dU_t, d_uI, d_uII, Sigma, Gamma_inv, t + 0.5*dt)

        # ---- RK4 stage 3 ----
        mu3 = mu + 0.5 * dt * k2_mu
        R3 = R + 0.5 * dt * k2_R
        k3_mu, k3_R = kalman_rhs(mu3, R3, U_t, dU_t, d_uI, d_uII, Sigma, Gamma_inv, t + 0.5*dt)

        # ---- RK4 stage 4 ----
        U_t2  = U[k]              # use the next observed U

        dU_t2 = dUdt[k]
        mu4 = mu + dt * k3_mu
        R4 = R + dt * k3_R
        k4_mu, k4_R = kalman_rhs(mu4, R4, U_t2, dU_t2, d_uI, d_uII, Sigma, Gamma_inv, t + dt)

        # ---- Combine RK4 ----
        mu = mu + (dt/6)*(k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)
        R  = R  + (dt/6)*(k1_R  + 2*k2_R  + 2*k3_R  + k4_R)

        # Symmetrize R
        R = 0.5 * (R + R.T)

        mu_hist[k] = mu
        R_hist[k] = R

    return {
        "t": t_vals,
        "uII": mu_hist.T,   # match your previous shape
        "R": R_hist
    }


def kalman_rhs2(mu, R, U, dUdt, d_uI, d_uII, Sigma2, Gamma_inv, t):
    """
    Compute RHS for μ and R:
    dμ/dt = a1 μ + a0 + K(innovation)
    dR/dt = a1 R + R a1ᵀ + Σ2 - R A1ᵀ Σ^{-1} A1 R
    """

    # Linearizations
    B0, B1 = extract_terms(d_uI, t, U, mu)   # (m,), (m x d)
    A0, A1 = extract_terms(d_uII, t, U, mu)  # (d,), (d x d)

    B0 = np.asarray(B0).reshape(-1)
    B1 = np.asarray(B1)
    A0 = np.asarray(A0).reshape(-1)
    A1 = np.asarray(A1)

    # Innovation: dU/dt - (A0 + A1 mu)
    innovation = dUdt - (B0 + B1 @ mu)

    # Kalman gain
    K = R @ B1.T @ Gamma_inv       # (d x m)

    # Mean update
    dmu = A1 @ mu + A0 + K @ innovation

    # Riccati update
    # dR = A1 @ R + R @ A1.T + Sigma2 - R @ B1.T @ Gamma_inv @ B1 @ R

    dR = np.zeros_like(R)
    return dmu, dR


def kalman_bucy2(d_uI, d_uII, t_span, uI, uII_0, R0, Sigma, Gamma):
    """
    RK4 version of Kalman-Bucy filter.
    t_vals: array of times (fixed grid)
    uI: observed U(t)
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

    # Output arrays
    mu_hist = np.zeros((n, d))
    R_hist = np.zeros((n, d, d))

    # Initial values
    mu = np.array(uII_0, dtype=float)
    R = np.array(R0, dtype=float)

    mu_hist[0] = mu
    R_hist[0] = R

    # Time stepping
    for k in tqdm(range(1, n), desc="RK4"):

        t = t_vals[k-1]

        # U and derivative at the current step
        U_t = U[k-1]
        dU_t = dUdt[k-1]

        # ---- RK4 stage 1 ----
        k1_mu, _ = kalman_rhs2(mu, R, U_t, dU_t, d_uI, d_uII, Sigma, Gamma_inv, t)

        # ---- RK4 stage 2 ----
        mu2 = mu + 0.5 * dt * k1_mu
        k2_mu, _ = kalman_rhs2(mu2, R, U_t, dU_t, d_uI, d_uII, Sigma, Gamma_inv, t + 0.5*dt)
        # ---- RK4 stage 3 ----
        mu3 = mu + 0.5 * dt * k2_mu
        k3_mu, _ = kalman_rhs2(mu3, R, U_t, dU_t, d_uI, d_uII, Sigma, Gamma_inv, t + 0.5*dt)

        # ---- RK4 stage 4 ----
        U_t2  = U[k]              # use the next observed U

        dU_t2 = dUdt[k]
        mu4 = mu + dt * k3_mu
        k4_mu, _ = kalman_rhs2(mu4, R, U_t2, dU_t2, d_uI, d_uII, Sigma, Gamma_inv, t + dt)

        # ---- Combine RK4 ----
        mu = mu + (dt/6)*(k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)

        mu_hist[k] = mu
        R_hist[k] = R

    return {
        "t": t_vals,
        "uII": mu_hist.T,   # match your previous shape
        "R": R_hist
    }
