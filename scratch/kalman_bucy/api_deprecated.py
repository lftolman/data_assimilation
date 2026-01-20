import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from .utils import extract_terms

def kalman_bucy(d_uI, d_uII, t_span, uI, uII_0, R0=None, SigmaI=None, SigmaII=None,dt = None):
    """ Implementation of the Kalman-Bucy filter for conditionally Gaussian systems

        Args:
            d_uI: function that returns the observation derivative given the state and input
            d_uII: function that returns the unknown state derivative given the state and input
            t_span: time span for the simulation
            uI: observed state
            uII_0: initial unknown state
            R0: initial covariance matrix (optional)
            SigmaI: process noise covariance matrix (optional)
            SigmaII: measurement noise covariance matrix (optional)
            dt: time step (optional, defaults to None, which will be calculated based on the length of x and t_span)
        Returns:
            A dictionary containing:
                - t: time values
                - uII: estimated state values
                - R: covariance matrix values
    """

    
    t0, tf = t_span
    n = len(uI)
    d = len(uII_0)
    t_vals = np.linspace(t0, tf, n)
    dt = (tf - t0) / (n-1) if dt is None else dt
    uII = np.zeros((d, n))
    uII[:,0] = uII_0
    R = np.zeros((n, d, d))
    R[0] = np.eye(d) if R0 is None else R0
    # m = len(uI[0])
    if SigmaI is not None:
        sst = SigmaI.copy()
        Sig_inv = np.linalg.inv(sst + np.eye(sst.shape[0])*1e-8 )
    else: 
        Sig_inv = np.zeros((len(uI[0]), len(uI[0])))

    if SigmaII is not None:
        Sigma2 = SigmaII.copy()
    else: 
        Sigma2 = np.zeros((d, d))

    for i in tqdm(range(1, n)):
        A0,A1 = extract_terms(d_uI, t_vals[i], uI[i], uII[:,i-1])
        a0, a1 = extract_terms(d_uII, t_vals[i], uI[i], uII[:,i-1])
        # print(A1)


        duI = (uI[i] - uI[i-1])/dt

        innovation =duI - (A0 + (A1 @ uII[:,i-1]))
        
        K = (R[i-1] @ A1.T) @ Sig_inv

        uII[:,i] = uII[:,i-1] + dt*((a1@uII[:,i-1] + a0) + (K @ innovation))
        R_pred = R[i-1] + (a1@R[i-1] + R[i-1]@ a1.T  + Sigma2 - R[i-1]@A1.T@Sig_inv@A1@R[i-1])*dt
        R[i] = (R_pred + R_pred.T)/2
        # dR1 = a1 @ R[i-1] + R[i-1] @ a1.T + Sigma2
        # R_pred = R[i-1] + dt * dR1
        # S = A1 @ R_pred @ A1.T + SigmaI + np.eye(A1.shape[0])*1e-9
        # K = R_pred @ A1.T @ np.linalg.inv(S)
        # R[i] = R_pred - dt * (K @ A1 @ R_pred)
        # R[i] = 0.5 * (R[i] + R[i].T)

        if np.isnan(R).any():
            print(f"NaN detected at step {i}, filtering failed")

    return {"t": t_vals, "uII": uII, "R": R}