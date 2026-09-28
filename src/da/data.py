import numpy as np
from scipy.io import loadmat
import importlib.resources
from .result import AssimilationResult
from .models import barotropic, barotropic_expected_matrices, barotropic_precompute_matrices
from .models import EM

def load_baro(file=None, precompute_matrices=False):
    """Loads the barotropic model data. If file is None, loads from package resources.
        Returns: AssimilationResult with t, U, mu, v_hat, and T_hat (Re(v1),Re(v2),...,Im(v1),Im(v2),... and Re(T1),Re(T2),...,Im(T1),Im(T2),...). 
        If precompute_matrices is True, also returns the precomputed A and B matrices for the entire time series."""
    if file is not None:
        data = loadmat(file)
    else:
        pkg_resource = importlib.resources.files("da.data_files").joinpath("baro1000.mat")
        with importlib.resources.as_file(pkg_resource) as data_path:
            data = loadmat(data_path)
            
    U_true = data['Uout'].squeeze()
    t_true = data['TT'].squeeze()
    
    # 1. Extract and stack Velocity (Shape becomes 4 x 1000)
    v_hat_true = data['vout'].squeeze()
    # Stacking axis=0 puts Re(v1), Re(v2) on top of Im(v1), Im(v2)
    v_aligned = np.concatenate([np.real(v_hat_true), np.imag(v_hat_true)], axis=0)
    
    # 2. Extract and stack Temperature (Shape becomes 4 x 1000)
    T_hat_true = data['Tout'].squeeze()
    T_aligned = np.concatenate([np.real(T_hat_true), np.imag(T_hat_true)], axis=0)
    
    # 3. Combine into the full state vector (Shape becomes 8 x 1000)
    mu = np.concatenate([v_aligned, T_aligned], axis=0)  
    
    # Transpose at the very end to match (time, state) -> (1000, 8)
    return AssimilationResult(t_true, U_true, mu.T)

def load_lorenz63(file=None, T = 100, timesteps = 100000, c = 1):
    """Loads the Lorenz 63 model data. If file is None, loads from package resources.
        Returns: AssimilationResult with t, U, and mu (x,y,z)."""
    if file is not None:
        data = loadmat(file)
    else:
        data = EM(x0=np.array([0,1.0,0]), T=T, timesteps=timesteps, c=c)
        # pkg_resource = importlib.resources.files("da.data_files").joinpath("lorenz63.mat")
        # with importlib.resources.as_file(pkg_resource) as data_path:
        #     data = loadmat(data_path)
            
    t_true = np.linspace(0, T, timesteps)
    x_true = data[:,0]
    y_true = data[:,1]
    z_true = data[:,2]
    
    return AssimilationResult(t_true, x_true, np.column_stack([y_true, z_true]), system="Lorenz63")