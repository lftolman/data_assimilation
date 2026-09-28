import numpy as np
import jax
import jax.numpy as jnp
from tqdm import tqdm



def extract_terms(f,t,uI,uII):
    eps=1e-8
    d = len(uII)
    f_base = f(t, uI, uII)
    J = np.zeros((np.atleast_1d(f_base).shape[0], d))
    for i in range(d):
        uII_eps = np.array(uII, dtype=float)
        uII_eps[i] += eps
        diff = (f(t, uI, uII_eps) - f_base) / eps
        J[:, i] = diff.flatten()
    a1 = J
    a0 = f_base - a1 @ uII
    return a0, a1

def extract_terms_fast(f, t, uI, uII):
    """
    JAX auto-diff version of extract_terms.
    Calculates the exact Jacobian with respect to uII (argnums=2).
    """
    f_base = f(t, uI, uII)
    J = jax.jacfwd(f, argnums=2)(t, uI, uII)
    
    # Ensure dimensions map correctly for 1D/2D arrays
    J = jnp.atleast_2d(J)
    f_base = jnp.atleast_1d(f_base)
    
    a0 = f_base - J @ uII
    return a0, J


def extract_terms2(f, t, uI, uII):
    """
    Standard finite difference to extract matrices.
    Because the system is linear w.r.t uII, this is exact.
    """
    eps = 1e-8
    d = len(uII)
    f_base = f(t, uI, uII)
    
    # Pre-allocate Jacobian
    f_base_flat = np.atleast_1d(f_base).flatten()
    J = np.zeros((f_base_flat.shape[0], d))
    
    # Use a single working array for perturbations to save memory
    uII_work = np.array(uII, dtype=float)
    
    for i in range(d):
        uII_work[i] += eps                               
        diff = (f(t, uI, uII_work) - f_base) / eps       
        J[:, i] = np.atleast_1d(diff).flatten()          
        uII_work[i] -= eps                               
        
    a1 = J
    a0 = f_base_flat - a1 @ uII
    return a0, a1

def precompute_matrices(f_uI, f_uII, t_vals, U, d, m):
    """
    Precomputes the A and B matrices for a Conditionally Gaussian system.
    Returns: A0_seq, A1_seq, B0_seq, B1_seq
    """
    n = len(t_vals)
    
    A0_seq = np.zeros((n, d))
    A1_seq = np.zeros((n, d, d))
    B0_seq = np.zeros((n, m))
    B1_seq = np.zeros((n, m, d))
    
    # We use a dummy state of zeros because the Jacobians in a 
    # Conditionally Gaussian system ONLY depend on U(t).
    dummy_mu = np.zeros(d) 
    
    print("Precomputing system matrices...")
    for k in tqdm(range(n), desc="Extracting A & B"):
        t = t_vals[k]
        U_t = U[k]
        
        b0, b1 = extract_terms2(f_uI, t, U_t, dummy_mu)
        a0, a1 = extract_terms2(f_uII, t, U_t, dummy_mu)
        
        B0_seq[k] = np.asarray(b0).reshape(-1)
        B1_seq[k] = np.asarray(b1)
        A0_seq[k] = np.asarray(a0).reshape(-1)
        A1_seq[k] = np.asarray(a1)
        
    return A0_seq, A1_seq, B0_seq, B1_seq

def reconstruct(var = "v_hat"):
    """Reconstruct the full complex variable from the real and imaginary parts."""
    if var == "v_hat":
        real_part = var[:var.shape[0]//2]
        imag_part = var[var.shape[0]//2:]
        return real_part + 1j * imag_part
    elif var == "T_hat":
        real_part = var[:var.shape[0]//2]
        imag_part = var[var.shape[0]//2:]
        return real_part + 1j * imag_part
    else:
        raise ValueError("Variable must be 'v_hat' or 'T_hat'.")
    
    
    