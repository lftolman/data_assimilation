import numpy as np

def EM(x0, T = 100, timesteps = 100000, c = 10):
    """ Euler Maruyama for Lorenz63 """
    delta_t = T/timesteps
    rho = 28
    beta = 8/3
    sigma = 10
    final = np.zeros((timesteps,3))
    final[0] = x0
    for i in range(timesteps-1):
        final[i+1, 0] = final[i,0] + delta_t*(sigma*(final[i,1]-final[i,0]))
        final[i+1, 1] = final[i,1] + delta_t*(final[i,0]*(rho - final[i,2])-final[i,1]) + c*np.random.normal(0,np.sqrt(delta_t))
        final[i+1, 2] = final[i,2] + delta_t*(final[i,0]*final[i,1]-beta*final[i,2])

    return final



def lorenz_kb_nudging(x_obs):
    rho = 28
    beta = 8/3
    sigma = 10
    T = 100
    n = 100000
    dt = T/(n-1)

    Sigma1 = np.array([[1]])

    x_obs = x_obs.copy()
    x_vals = np.zeros_like(x_obs)
    x_vals[0] = x_obs[0]

    Sigma2 = np.diag([.1,0])   

    # random starting point
    u0 = 2*np.random.randn(2)
    u_hat = np.zeros((2,n))
    u_hat[:,0] = u0

    R = np.eye(2)
    Sig_inv = np.linalg.inv(Sigma1@Sigma1.T)
    K = 10


    for i in range(1, n): 
        # nudge x toward observations
        y,z = u_hat[:,i-1]
        x = x_vals[i-1]
        F = -sigma*x+sigma*y

        x_vals[i] = x + dt*(F + K*(x_obs[i-1] - x))

        A0 = -sigma*x_vals[i]
        A1 = np.array([[sigma,0]])

        a1 = np.array([[-1,-x_vals[i]],[x_vals[i],-beta]])
        a0 = np.array([rho*x_vals[i],0])

        duI = x_vals[i] - x_vals[i-1]

        innovation = duI - (A0 + (A1 @ u_hat[:,i-1])) * dt
        u_hat[:,i] = u_hat[:,i-1] + dt*(a1@u_hat[:,i-1] + a0) + ((R @ A1.T) @ Sig_inv * innovation).flatten()
        R += (a1@R + R@ a1.T  + Sigma2@Sigma2.T - R@A1.T@Sig_inv@A1@R)*dt

        if np.isnan(R).any():
            print(f"NaN detected at step {i}")
            break
    return x_vals,u_hat




