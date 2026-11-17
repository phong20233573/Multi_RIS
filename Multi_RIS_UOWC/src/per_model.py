"""
per_model.py

Link-level model connecting the physical layer (unipolar M-PAM over the
multi-RIS UWOC channel) to the packet layer used by ARQ / IR-HARQ.

Pipeline for AMC design
-----------------------
instantaneous SNR gamma (linear)
        |
        v
BER of Gray-coded unipolar M-PAM    (ber_mpam)
        |
        v
Reed-Solomon symbol error rate      (rs_symbol_error_rate)
        |
        v
codeword (packet) failure prob      (rs_codeword_fail_prob)
        |
        v
AMC SNR thresholds theta_M          (amc_thresholds / amc_select)

The runtime HARQ simulation uses a real RS(255,223) encoder/decoder in
``rs_codec.py``. The analytical functions in this module are retained only
to design AMC thresholds; they do not decide packet success in HARQ.
"""
import numpy as np
from scipy.special import erfc
from scipy import stats
from scipy.optimize import brentq

# The PHY simulator uses y = h*x + n with E[x**2] = 1 and
# n ~ N(0, 0.5). Therefore gamma = h**2 is Es/N0.
AWGN_VARIANCE = 0.5


def q_func(x):
    """Gaussian Q-function."""
    return 0.5 * erfc(x / np.sqrt(2.0))


def ser_mpam(gamma, M):
    """
    Symbol error rate of unipolar M-PAM (levels 0, 2, ..., 2(M-1)) at SNR
    gamma (linear). The constellation is normalized to E[x^2] = 1 (see
    modulation.py), and the decision thresholds sit at the odd midpoints.
    """
    M = int(M)
    gamma = float(gamma)
    if gamma <= 0.0:
        return 1.0
    # The nearest threshold is 1 / norm from a signal level. After
    # equalization, the noise standard deviation is sqrt(0.5 / gamma).
    # norm^2 = 2(M-1)(2M-1)/3.
    arg = np.sqrt(3.0 * gamma / ((M - 1) * (2 * M - 1)))
    return (2.0 * (M - 1) / M) * q_func(arg)


def ber_mpam(gamma, M):
    """Approximate bit error rate of Gray-coded unipolar M-PAM (SER / log2(M))."""
    M = int(M)
    return ser_mpam(gamma, M) / int(np.log2(M))


def rs_symbol_error_rate(gamma, M, s=8):
    """Probability that one GF(2^s) Reed-Solomon symbol is in error."""
    ber = ber_mpam(gamma, M)
    return 1.0 - (1.0 - ber) ** s


def rs_codeword_fail_prob(gamma, M, n, k, s=8):
    """
    P(RS decoder failure) = P(more than t = (n-k)//2 symbol errors in a
    codeword of length n).
    """
    t = (n - k) // 2
    p = rs_symbol_error_rate(gamma, M, s)
    return 1.0 - stats.binom.cdf(t, n, p)


def amc_thresholds(M_list, n, k, s, P_target):
    """
    SNR threshold (dB) at which each modulation M reaches the design packet
    failure probability P_target for a single (first) transmission.

    Returns
    -------
    dict : {M: theta_db} sorted by M.
    """
    M_list = [int(m) for m in M_list]
    theta_db = {}
    for M in M_list:
        f = lambda g_db: rs_codeword_fail_prob(10.0 ** (g_db / 10.0), M, n, k, s) - P_target
        lo, hi = -30.0, 90.0
        flo, fhi = f(lo), f(hi)
        if flo * fhi > 0.0:
            raise ValueError(f"Root not bracketed for M={M}: f(lo)={flo:.4f}, f(hi)={fhi:.4f}")
        theta_db[M] = brentq(f, lo, hi, xtol=1e-9)
    return theta_db


def amc_select(gamma, theta_db):
    """Largest modulation M whose AMC threshold is met. Returns 0 if below the lowest."""
    gamma = max(float(gamma), 1e-300)
    g_db = 10.0 * np.log10(gamma)
    best = 0
    for M, th in sorted(theta_db.items()):
        if g_db >= th:
            best = M
    return best


def snr_threshold_rate(r, theta_db):
    """
    SNR threshold (dB) required to decode spectral efficiency r (info bits per
    symbol) at the design packet error probability, obtained by interpolating
    the RS-derived AMC thresholds. Linear interpolation in dB over log2(M),
    with linear extrapolation outside the tabulated range.
    """
    orders = sorted(theta_db.keys())
    x_pts = [float(np.log2(m)) for m in orders]
    y_pts = [float(theta_db[m]) for m in orders]
    x = float(np.log2(r)) if r > 0 else x_pts[0] - 3.0
    if x <= x_pts[0]:
        slope = (y_pts[1] - y_pts[0]) / (x_pts[1] - x_pts[0])
        return y_pts[0] + slope * (x - x_pts[0])
    if x >= x_pts[-1]:
        slope = (y_pts[-1] - y_pts[-2]) / (x_pts[-1] - x_pts[-2])
        return y_pts[-1] + slope * (x - x_pts[-1])
    return float(np.interp(x, x_pts, y_pts))
