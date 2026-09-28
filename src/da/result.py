import numpy as np
from .diagnostics import compute_diagnostics
from matplotlib import pyplot as plt
from .visualization.plot_funcs import plot_pdf
from .models.barotropic import barotropic_precompute_matrices

class AssimilationResult:
    def __init__(self, t, U, mu, cov=None, system = "Barotropic", comp_time=None):
        """Class to store the results of a data assimilation run. 
        t: Array of time values
        U: Array of true U values
        mu: Array of predicted state values, shape (2*num_nodes, n) (real then imaginary parts)
        cov: Array of covariance values for the entire system (optional) (shape: (n, 2*num_nodes, 2*num_nodes))
        If system is "Barotropic", also splits mu into v_hat and T_hat, and cov into cov_v and cov_T for easier access."""  
        self.mu = mu
        self.system = system
        if system == "Barotropic":
            self.v_hat = mu[:,:mu.shape[1]//2]
            self.T_hat = mu[:,mu.shape[1]//2:]
            if cov is not None:
                self.cov_v = cov[:,:cov.shape[1]//2,:cov.shape[2]//2]
                self.cov_T = cov[:,cov.shape[1]//2:,cov.shape[2]//2:]
            else:
                self.cov_v = np.zeros((mu.shape[0], mu.shape[1]//2, mu.shape[1]//2))
                self.cov_T = np.zeros((mu.shape[0], mu.shape[1]//2, mu.shape[1]//2))
        self.t = t
        self.U = U
        self.cov = cov
        self.comp_time = comp_time
    
    def precompute_matrices(self):
        A0_seq, A1_seq, B0_seq, B1_seq = barotropic_precompute_matrices(self.t, self.U)
        self.A0_seq = A0_seq
        self.A1_seq = A1_seq
        self.B0_seq = B0_seq
        self.B1_seq = B1_seq