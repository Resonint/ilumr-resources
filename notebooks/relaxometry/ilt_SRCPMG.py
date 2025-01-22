import numpy as np
from matipo.util import flint

# modified from IRCPMG_T1T2_spectrum() in ilt.py to use the saturation recovery kernel function for T1
def SRCPMG_T1T2_spectrum(T1, T2, t_rec, t_echo, data, alpha=1, **kwargs):
    """
    SRCPMG_T1T2_spectrum: Calculate T1-T2 spectrum from Saturation Recovery CPMG data
    
    Parameters
    ----------
    T1 : float array
        T1 values for the output spectrum
    T2 : float array
        T2 values for the output spectrum
    t_rec : float array
        recovery times (seconds)
    t_echo : float or float array
        scalar echo spacing or array of echo times (seconds)
    data : 2D float or complex float array
        2D array of IRCPMG data, shape (number of inversion times, number of echos).
        The real part after autophasing will be used if complex.
    """
    
    N1, N2 = data.shape
    if len(t_rec) != N1:
        raise Exception(f'size of t_rec ({len(t_rec)} must match axis 0 of data ({N1}))')
    if np.isscalar(t_echo):
        t_echo = np.linspace(0, N2*t_echo, N2, endpoint=False)
    if len(t_echo) != N2:
        raise Exception(f'size of t_echo ({len(t_echo)} must match axis 1 of data ({N2}))')
    
    if np.any(np.iscomplex(data)):
        # autophase using first 2 points
        phase = np.angle(np.mean(data[0][:2]))
        y = np.real(data * np.exp(1j*-phase))
    else:
        y = data.copy()
    
    K1 = 1 - np.exp(-np.outer(t_rec,1/T1)) # T1 Saturation Recovery Kernel
    K2 = np.exp(-np.outer(t_echo,1/T2)) # T2 Kernel
    S, res = flint.flint(K1, K2, y, alpha, **kwargs)
    return S