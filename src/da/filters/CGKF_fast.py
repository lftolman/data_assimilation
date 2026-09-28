import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from functools import partial
from ..utils.utils import extract_terms, precompute_matrices

@partial(jax.jit, static_argnums=(4, 5))
def CGKF_rhs(mu, R, U, dUdt, d_uI, d_uII, Sigma2, Gamma_inv, t):
    """
    Compute RHS for μ and R using JAX.
    (d_uI and d_uII must be marked as static so JAX can compile them)
    """
    # Linearizations
    B0, B1 = extract_terms(d_uI, t, U, mu)   
    A0, A1 = extract_terms(d_uII, t, U, mu)  

    # Keep data in JAX format (jnp) to prevent CPU memory copying
    B0 = jnp.asarray(B0).reshape(-1)
    B1 = jnp.asarray(B1)
    A0 = jnp.asarray(A0).reshape(-1)
    A1 = jnp.asarray(A1)

    # Innovation
    innovation = dUdt - (B0 + B1 @ mu)

    # Kalman gain
    K = R @ B1.T @ Gamma_inv       

    # Mean update
    dmu = A1 @ mu + A0 + K @ innovation

    # Riccati update
    dR = A1 @ R + R @ A1.T + Sigma2 - R @ B1.T @ Gamma_inv @ B1 @ R

    return dmu, dR

@partial(jax.jit, static_argnums=(4, 5))
def CGKF_rhs2(mu, R, U, dUdt, d_uI, d_uII, Sigma2, Gamma_inv, t):
    """
    Compute RHS for μ (Covariance R remains constant).
    """
    B0, B1 = extract_terms(d_uI, t, U, mu)   
    A0, A1 = extract_terms(d_uII, t, U, mu)  

    B0 = jnp.asarray(B0).reshape(-1)
    B1 = jnp.asarray(B1)
    A0 = jnp.asarray(A0).reshape(-1)
    A1 = jnp.asarray(A1)

    innovation = dUdt - (B0 + B1 @ mu)
    K = R @ B1.T @ Gamma_inv       
    dmu = A1 @ mu + A0 + K @ innovation
    
    dR = jnp.zeros_like(R)
    
    return dmu, dR

def CGKF_fast(d_uI, d_uII, t_span, uI, uII_0, R0, Sigma, Gamma, method="euler"):
    """
    Conditionally Gaussian Kalman Filter.
    """
    t0, tf = t_span
    n = len(uI)
    d = len(uII_0)
    t_vals = np.linspace(t0, tf, n)
    uI = np.asarray(uI)
    dt = t_vals[1] - t_vals[0]

    if uI.ndim == 1:
        m = 1
        U = uI.reshape(n, 1)
    else:
        m = uI.shape[1]
        U = uI

    # We can calculate constants using standard numpy before the loop
    Gamma_inv = np.linalg.inv(Gamma @ Gamma.T + 1e-12 * np.eye(m))
    dUdt = np.gradient(U, t_vals, axis=0)

    mu_hist = np.zeros((n, d))
    R_hist = np.zeros((n, d, d))

    # Initialize state as JAX arrays for the inner loop
    mu = jnp.array(uII_0, dtype=float)
    R = jnp.array(R0, dtype=float)

    mu_hist[0] = mu
    R_hist[0] = R

    # Convert static constants to JAX arrays once
    Sigma_jax = jnp.array(Sigma)
    Gamma_inv_jax = jnp.array(Gamma_inv)

    if method == "euler":
        for k in tqdm(range(1, n), desc="Euler"):
            t = t_vals[k-1]
            U_t = jnp.array(U[k-1])
            dU_t = jnp.array(dUdt[k-1])

            dmu, dR = CGKF_rhs(mu, R, U_t, dU_t, d_uI, d_uII, Sigma_jax, Gamma_inv_jax, t)

            mu = mu + dt * dmu
            R = R + dt * dR
            R = 0.5 * (R + R.T)

            # Store results back in numpy arrays for output
            mu_hist[k] = np.array(mu)
            R_hist[k] = np.array(R)

    elif method == "RK4":
        for k in tqdm(range(1, n), desc="RK4"):
            t = t_vals[k-1]
            U_t = jnp.array(U[k-1])
            dU_t = jnp.array(dUdt[k-1])
            U_t2 = jnp.array(U[k])
            dU_t2 = jnp.array(dUdt[k])

            k1_mu, k1_R = CGKF_rhs(mu, R, U_t, dU_t, d_uI, d_uII, Sigma_jax, Gamma_inv_jax, t)
            
            mu2 = mu + 0.5 * dt * k1_mu
            R2 = R + 0.5 * dt * k1_R
            k2_mu, k2_R = CGKF_rhs(mu2, R2, U_t, dU_t, d_uI, d_uII, Sigma_jax, Gamma_inv_jax, t + 0.5*dt)
            
            mu3 = mu + 0.5 * dt * k2_mu
            R3 = R + 0.5 * dt * k2_R
            k3_mu, k3_R = CGKF_rhs(mu3, R3, U_t, dU_t, d_uI, d_uII, Sigma_jax, Gamma_inv_jax, t + 0.5*dt)
            
            mu4 = mu + dt * k3_mu
            R4 = R + dt * k3_R
            k4_mu, k4_R = CGKF_rhs(mu4, R4, U_t2, dU_t2, d_uI, d_uII, Sigma_jax, Gamma_inv_jax, t + dt)

            mu = mu + (dt/6)*(k1_mu + 2*k2_mu + 2*k3_mu + k4_mu)
            R = R + (dt/6)*(k1_R + 2*k2_R + 2*k3_R + k4_R)
            R = 0.5 * (R + R.T)

            mu_hist[k] = np.array(mu)
            R_hist[k] = np.array(R)

    return {
        "t": t_vals,
        "uII": mu_hist.T,   
        "R": R_hist
    }

# (Apply the same logic to CGKF_RK4 using CGKF_rhs2 and converting loop inputs to jnp.array)