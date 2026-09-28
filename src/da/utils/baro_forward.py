import numpy as np

def BaroLayered_forward_condiGau(U0, omek0, Tk0, dt, Tfin, tstep, params):
    """
    Fully corrected Python translation of MATLAB BaroLayered_forward_condiGau.
    Returns the same tuple of outputs as the MATLAB function.
    """

    beta   = params['beta']
    hk     = np.asarray(params['hk'], dtype=complex)
    lm     = params['lm']
    kmax   = int(params['kmax'])
    lvec   = np.asarray(params['lvec'], dtype=float)
    kvec   = np.asarray(params['kvec'], dtype=float)
    d_k    = np.asarray(params['d_k'], dtype=float)
    d_U    = params['d_U']
    sig_k  = np.asarray(params['sig_k'], dtype=complex)
    sig_U  = params['sig_U']
    f_U    = params['f_U']
    f_k    = np.asarray(params['f_k'], dtype=complex)
    gamma_T = np.asarray(params['gamma_T'], dtype=complex)
    alpha  = params['alpha']

    sigw_k = 1j * kvec / lvec[0] * sig_k
    kred = 2
    Nt = int(round(Tfin / dt))


    U = float(U0)
    omek = np.asarray(omek0, dtype=complex).copy()
    Tk = np.asarray(Tk0, dtype=complex).copy()

    vm0 = -1j * lvec[0] / (lm**2) / kvec * omek

    um = np.zeros(4*kmax, dtype=complex)
    um[0:kmax] = vm0
    um[kmax:2*kmax] = np.conjugate(vm0[::-1])
    um[2*kmax:3*kmax] = Tk
    um[3*kmax:4*kmax] = np.conjugate(Tk[::-1])

    Ru = 0.01 * np.eye(4*kmax, dtype=complex)


    outsize = Nt // tstep + 1
    tout = np.zeros(outsize)
    Uout = np.zeros(outsize)
    dUout = np.zeros(outsize)
    vout = np.zeros((kmax, outsize), dtype=complex)
    Tout = np.zeros((kmax, outsize), dtype=complex)
    umout = np.zeros((4*kmax, outsize), dtype=complex)
    Ruout = np.zeros((4*kmax, outsize), dtype=complex)   # store diag(Ru)
    Cuout = np.zeros((2*kmax, outsize), dtype=complex)
    dfout = np.zeros(outsize)
    dgout = np.zeros((4*kmax, outsize), dtype=complex)
    noise = np.zeros(outsize)
    unres = np.zeros(outsize)
    energy = np.zeros((3, outsize))
    enstrophy = np.zeros((2, outsize))


    A0 = -d_U
    Sig0 = sig_U
    invSig0Sq = 1.0 / (np.abs(Sig0)**2)


    A1 = np.zeros(4*kmax, dtype=complex)
    A1[0:kmax] = np.conjugate(hk)
    A1[kmax:2*kmax] = hk[::-1]

    B0 = np.zeros(4*kmax, dtype=complex)
    B0[0:kmax] = -lvec[0]**2 * hk
    B0[kmax:2*kmax] = -lvec[0]**2 * np.conjugate(hk[::-1])

    sigk_vec = np.concatenate([
        sig_k[0:kmax],
        np.conjugate(sig_k[::-1]),
        np.zeros(2*kmax)
    ])
    Sigk = np.diag(sigk_vec)

    B1 = np.zeros((4*kmax, 4*kmax), dtype=complex)
    B2 = np.zeros((4*kmax, 4*kmax), dtype=complex)

    for kk in range(kmax):
        i1 = kk
        i2 = 2*kmax - 1 - kk
        i3 = 2*kmax + kk
        i4 = 4*kmax - 1 - kk

        B1[i1,i1] = -d_k[kk] + 1j*lvec[0]*beta/kvec[kk]
        B1[i2,i2] = np.conjugate(B1[i1,i1])
        B1[i3,i3] = -gamma_T[kk]
        B1[i4,i4] = np.conjugate(B1[i3,i3])

        B1[i3,i1] = -alpha
        B1[i4,i2] = -alpha

        B2[i1,i1] = -1j*lvec[0]*kvec[kk]
        B2[i2,i2] = np.conjugate(B2[i1,i1])
        B2[i3,i3] = -1j*kvec[kk]
        B2[i4,i4] = np.conjugate(B2[i3,i3])

    SigkSigkT = (Sigk @ Sigk.conj().T)

    np.random.seed(50)
    epsilon = 1.0
    xi = 0.0
    tt = 0.0
    dU = 0.0

    # helper: small PSD floor to avoid tiny negative eigenvalues
    def enforce_psd(H, tol=1e-14):
        # enforce Hermitian
        H = 0.5 * (H + H.conj().T)
        try:
            vals, vecs = np.linalg.eigh(H)
            vals_clipped = np.maximum(vals, tol)
            H2 = (vecs * vals_clipped) @ vecs.conj().T
            return H2
        except np.linalg.LinAlgError:
            # fallback: zero small values
            H[np.isnan(H)] = 0.0
            H[np.isinf(H)] = 0.0
            return 0.5 * (H + H.conj().T)

    for ii in range(Nt+1):
        if ii % tstep == 0:
            iout = ii // tstep
            if iout >= outsize:
                break

            tout[iout] = tt
            Uout[iout] = U
            dUout[iout] = dU
            vout[:,iout] = -1j*lvec[0]/(lm**2) / kvec * omek
            Tout[:,iout] = Tk
            umout[:,iout] = um
            Ruout[:,iout] = np.diag(Ru)

            Cu = Ru[2*kmax:4*kmax, 0:2*kmax]
            Cuout[:,iout] = np.diag(Cu)

            df_val = (A1.reshape(1, -1) @ um.reshape(-1, 1))[0,0]
            dfout[iout] = float(np.real(dU - A0 * U - df_val))

            dgcol = (Ru @ np.conjugate(A1.T)).reshape(-1)
            dgout[:,iout] = dgcol

            noise[iout] = xi
            xi = 0.0

            term1 = np.sum(np.conjugate(hk) * vout[:,iout])
            term2 = np.sum(np.conjugate(hk[:kred]) * umout[:kred,iout])
            unres[iout] = 2.0 * np.real(term1 - term2)

            energy[0,iout] = np.sum(kvec**(-2) * np.abs(omek)**2)
            energy[1,iout] = 0.5 * U**2
            energy[2,iout] = np.sum(np.abs(Tk)**2)
            enstrophy[0,iout] = np.sum(np.abs(omek + hk)**2)
            enstrophy[1,iout] = beta * U

        # ----------------------
        # RK4 step 1
        # ----------------------
        k1 = (epsilon**(-1)) * (
            -1j * lvec[0] * (U * kvec - beta / kvec) * omek
            - 1j * lvec[0] * kvec * hk * U
        ) - d_k * omek + f_k

        L1 = (epsilon**(-1)) * 2.0 * lvec[0] * np.imag(np.sum(np.conjugate(hk) * (omek / kvec))) - d_U * U + f_U

        ok1 = omek + 0.5 * dt * k1
        U1 = U + 0.5 * dt * L1

        vk = -1j * lvec[0] / (lm**2) / kvec * omek
        s1 = (epsilon**(-1)) * ( - gamma_T + 1j * (-kvec * U) ) * Tk - alpha * vk
        T1 = Tk + 0.5 * dt * s1

        # condi-Gaussian step 1
        BU = B1 + B2 * U
        df1 = float(np.real(dU - A0 * U - (A1.reshape(1,-1) @ um.reshape(-1,1))[0,0]))

        A1H = np.conjugate(A1.T)          
        K1  = Ru @ A1H                    #

        if Sig0 != 0.0:
            ks1 = B0 * U + BU @ um + K1 * (df1 * invSig0Sq)
            Ls1 = BU @ Ru + Ru.conj().T @ BU.conj().T + SigkSigkT - np.outer(K1, (A1 @ Ru.conj().T)) * invSig0Sq
        else:
            ks1 = B0 * U + BU @ um
            Ls1 = BU @ Ru + Ru.conj().T @ BU.conj().T + SigkSigkT

        um1 = um + 0.5 * dt * ks1
        Ru1 = Ru + 0.5 * dt * Ls1

        # ----------------------
        # RK4 step 2
        # ----------------------
        k2 = (epsilon**(-1)) * (
            -1j * lvec[0] * (U1 * kvec - beta / kvec) * ok1
            - 1j * lvec[0] * kvec * hk * U1
        ) - d_k * ok1 + f_k

        L2 = (epsilon**(-1)) * 2.0 * lvec[0] * np.imag(np.sum(np.conjugate(hk) * (ok1 / kvec))) - d_U * U1 + f_U

        ok2 = omek + 0.5 * dt * k2
        U2 = U + 0.5 * dt * L2

        vk = -1j * lvec[0] / (lm**2) / kvec * ok1
        s2 = (epsilon**(-1)) * ( - gamma_T + 1j * (-kvec * U1) ) * T1 - alpha * vk
        T2 = Tk + 0.5 * dt * s2

        # condi-Gaussian step 2
        BU = B1 + B2 * U1
        df2 = float(np.real(dU - A0 * U1 - (A1.reshape(1,-1) @ um1.reshape(-1,1))[0,0]))

        A1H = np.conjugate(A1.T)
        K2  = Ru1 @ A1H

        if Sig0 != 0.0:
            ks2 = B0 * U1 + BU @ um1 + K2 * (df2 * invSig0Sq)
            Ls2 = BU @ Ru1 + Ru1.conj().T @ BU.conj().T + SigkSigkT - np.outer(K2, (A1 @ Ru1.conj().T)) * invSig0Sq
        else:
            ks2 = B0 * U1 + BU @ um1
            Ls2 = BU @ Ru1 + Ru1.conj().T @ BU.conj().T + SigkSigkT

        um2 = um + 0.5 * dt * ks2
        Ru2 = Ru + 0.5 * dt * Ls2

        # ----------------------
        # RK4 step 3
        # ----------------------
        k3 = (epsilon**(-1)) * (
            -1j * lvec[0] * (U2 * kvec - beta / kvec) * ok2
            - 1j * lvec[0] * kvec * hk * U2
        ) - d_k * ok2 + f_k

        L3 = (epsilon**(-1)) * 2.0 * lvec[0] * np.imag(np.sum(np.conjugate(hk) * (ok2 / kvec))) - d_U * U2 + f_U

        ok3 = omek + dt * k3
        U3 = U + dt * L3

        vk = -1j * lvec[0] / (lm**2) / kvec * ok2
        s3 = (epsilon**(-1)) * ( - gamma_T + 1j * (-kvec * U2) ) * T2 - alpha * vk
        T3 = Tk + dt * s3

        # condi-Gaussian step 3
        BU = B1 + B2 * U2
        df3 = float(np.real(dU - A0 * U2 - (A1.reshape(1,-1) @ um2.reshape(-1,1))[0,0]))

        A1H = np.conjugate(A1.T)
        K3  = Ru2 @ A1H

        if Sig0 != 0.0:
            ks3 = B0 * U2 + BU @ um2 + K3 * (df3 * invSig0Sq)
            Ls3 = BU @ Ru2 + Ru2.conj().T @ BU.conj().T + SigkSigkT - np.outer(K3, (A1 @ Ru2.conj().T)) * invSig0Sq
        else:
            ks3 = B0 * U2 + BU @ um2
            Ls3 = BU @ Ru2 + Ru2.conj().T @ BU.conj().T + SigkSigkT

        um3 = um + dt * ks3
        Ru3 = Ru + dt * Ls3

        # ----------------------
        # RK4 step 4
        # ----------------------
        k4 = (epsilon**(-1)) * (
            -1j * lvec[0] * (U3 * kvec - beta / kvec) * ok3
            - 1j * lvec[0] * kvec * hk * U3
        ) - d_k * ok3 + f_k

        L4 = (epsilon**(-1)) * 2.0 * lvec[0] * np.imag(np.sum(np.conjugate(hk) * (ok3 / kvec))) - d_U * U3 + f_U

        vk = -1j * lvec[0] / (lm**2) / kvec * ok3
        s4 = (epsilon**(-1)) * ( - gamma_T + 1j * (-kvec * U3) ) * T3 - alpha * vk

        # condi-Gaussian step 4
        BU = B1 + B2 * U3
        df4 = float(np.real(dU - A0 * U3 - (A1.reshape(1,-1) @ um3.reshape(-1,1))[0,0]))

        A1H = np.conjugate(A1.T)
        K4  = Ru3 @ A1H

        if Sig0 != 0.0:
            ks4 = B0 * U3 + BU @ um3 + K4 * (df4 * invSig0Sq)
            Ls4 = BU @ Ru3 + Ru3.conj().T @ BU.conj().T + SigkSigkT - np.outer(K4, (A1 @ Ru3.conj().T)) * invSig0Sq
        else:
            ks4 = B0 * U3 + BU @ um3
            Ls4 = BU @ Ru3 + Ru3.conj().T @ BU.conj().T + SigkSigkT

        # ----------------------
        # Final stochastic RK4 update (identical to MATLAB)
        # ----------------------
        dW0 = sig_U * np.random.randn()
        U = U + dt * (L1 + 2*L2 + 2*L3 + L4) / 6.0 + np.sqrt(dt) * dW0
        dU = (dt * (L1 + 2*L2 + 2*L3 + L4) / 6.0 + np.sqrt(dt) * dW0) / dt

        dW_k = sigw_k * (np.random.randn(kmax) + 1j * np.random.randn(kmax)) / np.sqrt(2.0)
        omek = omek + dt * (k1 + 2*k2 + 2*k3 + k4) / 6.0 + np.sqrt(dt) * dW_k
        xi = xi + np.sqrt(dt) * dW0

        Tk = Tk + dt * (s1 + 2*s2 + 2*s3 + s4) / 6.0
        um = um + dt * (ks1 + 2*ks2 + 2*ks3 + ks4) / 6.0

        Ru = Ru + dt * (Ls1 + 2*Ls2 + 2*Ls3 + Ls4) / 6.0
        # hermitian enforce + small PSD correction to prevent numerical collapse
        Ru = enforce_psd(0.5 * (Ru + Ru.conj().T), tol=1e-20)

        tt += dt

    return (tout, Uout, dUout, noise, unres,
            vout, Tout, umout, Ruout, Cuout,
            dfout, dgout, energy, enstrophy,
            U, dU, omek, Tk, um, Ru)
