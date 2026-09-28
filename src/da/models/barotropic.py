from ..utils.utils import extract_terms2
import numpy as np
from tqdm import tqdm

default_params = dict(
    beta=1.0,
    hk=np.array([
        0.5 - 0.5j,
        0.25 - 0.25j
    ]),
    kmax=np.array([[2]]),
    kvec=np.arange(1, 3).reshape(-1, 1),
    d_k=np.ones((2, 1)) * 0.0125,
    d_U=0.0125,
    sig_k=np.array([
        0.70710678,
        0.35355339
    ]),
    sig_U=np.array([[0.35355339]]),
    gamma_T=np.array([
        0.101, 0.104
    ]),
    f_k=np.zeros((2, 1)),
    f_U=np.zeros((1, 1)),
    alpha=1.0,
)


def f_uI(t, uI, uII, params):
    """
    Equation for the mean flow U. Depends on the complex modes v_hat and T_hat through a linear coupling term.
    uI: scalar mean flow U
    uII: vector of complex modes v_hat and T_hat (real and imag parts concatenated)
    params: dictionary containing model parameters
    """
    hk = np.asarray(params["hk"]).flatten()
    d_U = params["d_U"]
    sigma0 = params.get("sigma0", 0.0)  # stochastic term ignored (0) for deterministic filter

    n = hk.size
    v_real = uII[:n]  # complex array
    v_imag = uII[n:2*n]
    v_hat = v_real + 1j * v_imag  # complex array

    term1 = np.sum(np.conjugate(hk) * v_hat)
    dUdt = 2*np.real(term1) - d_U * float(np.real(uI))
    return np.array([dUdt])



def f_uII(t, uI, uII, params):
    """Equation for the complex modes v_hat and T_hat. Depends on the mean flow U through a linear coupling term.
    uI: scalar mean flow U
    uII: vector of complex modes v_hat and T_hat (real and imag parts concatenated)
    params: dictionary containing model parameters
    """

    hk = np.asarray(params["hk"]).flatten()
    kvec = np.asarray(params["kvec"]).flatten().astype(float)
    lx = float(params.get("lx", 1.0))
    alpha = float(params["alpha"])
    d_v = params.get("d_v", np.zeros_like(hk))
    d_T = params.get("d_T", np.zeros_like(hk))
    sig_k = params["sig_k"]  
    beta = float(params["beta"])

    gamma_T = np.array(params["gamma_T"]).astype(np.float64).flatten()
    

    n = hk.size
    v_real = uII[:n]
    v_imag = uII[n:2*n]
    T_real = uII[2*n:3*n]
    T_imag = uII[3*n:4*n]
    v_hat = v_real + 1j * v_imag
    T_hat = T_real + 1j * T_imag
    U = np.real(uI)

    gamma_v = np.zeros(n, dtype=np.float64) 
    omega_v = lx * (beta / kvec - kvec * U)
    omega_T = -kvec * U

    dvdt = (-gamma_v - d_v) * v_hat + 1j * (omega_v * v_hat) - (lx ** 2) * hk * U


    dTdt = (-gamma_T - d_T) * T_hat + 1j * (omega_T * T_hat) - alpha * v_hat
    return np.concatenate([np.real(dvdt).flatten(), np.imag(dvdt).flatten(), np.real(dTdt), np.imag(dTdt)])

def barotropic(params = True):
    """Returns the functions f_uI and f_uII for the barotropic model, along with default parameters if params=True."""
    if params is True:
        params = default_params
        return f_uI, f_uII, params
    else:
        return f_uI, f_uII
    
def barotropic_expected_matrices(uI, params = None):
    """Returns the A and B matrices for the barotropic model, along with default parameters if params=True."""
    if params is None:
        params = default_params
    f_uI, f_uII, params = barotropic(params)
    
    # For the barotropic model, the A and B matrices are constant and can be computed directly from the parameters.
    # The system is linear in uII, so we can extract the matrices directly.

    uI = np.mean(uI) 
    dummy_uII = np.zeros(4)  # 2 complex modes -> 4 real variables
    
    A0, A1 = extract_terms2(f_uII, 0, uI, dummy_uII)
    B0, B1 = extract_terms2(f_uI, 0, uI, dummy_uII)
    
    return A0, A1, B0, B1

def barotropic_precompute_matrices(t_vals, U, params = None):
    """Precomputes the A and B matrices for the barotropic model over a time series of U(t)."""
    if params is None:
        params = default_params
    f_uI, f_uII = barotropic(params=False)
    d_uI = lambda t, U, uII: f_uI(t, U, uII, params)
    d_uII = lambda t, U, uII: f_uII(t, U, uII, params)
    
    n = len(U)
    d = 4 * np.asarray(params["hk"]).size  # K complex modes (v, T) -> 4K real variables
    m = 1 if np.ndim(U) == 1 else U.shape[1]
    U = U.reshape(n, m) if np.ndim(U) == 1 else U

    A0_seq = np.zeros((n, d))
    A1_seq = np.zeros((n, d, d))
    B0_seq = np.zeros((n, m))
    B1_seq = np.zeros((n, m, d))

    for k in tqdm(range(n), desc="Precomputing matrices"):
        t = t_vals[k]
        U_t = U[k]
        
        b0, b1 = extract_terms2(d_uI, t, U_t, np.zeros(d))
        a0, a1 = extract_terms2(d_uII, t, U_t, np.zeros(d))
        
        B0_seq[k] = np.asarray(b0).reshape(-1)
        B1_seq[k] = np.asarray(b1)
        A0_seq[k] = np.asarray(a0).reshape(-1)
        A1_seq[k] = np.asarray(a1)

    return A0_seq, A1_seq, B0_seq, B1_seq