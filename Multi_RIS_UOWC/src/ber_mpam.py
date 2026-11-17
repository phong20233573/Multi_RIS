"""
End-to-end BER simulation for Gray-coded unipolar M-PAM over a fading channel.

The simulated receiver follows:

    bits -> M-PAM -> y = h*x + n -> zero-forcing equalization
         -> nearest-level demodulation -> BER

The modulation and demodulation implementation is shared with modulation.py,
so the BER result uses the same constellation definition as the packet layer.
"""
from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from src.modulation import (
    calculate_ber,
    demodulate_mpam_imdd,
    modulate_mpam_imdd,
)


def simulate_end_to_end_ber_mpam(
    h_eff: np.ndarray,
    snr_db_array: Sequence[float],
    M: int,
    *,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Estimate BER for one fixed M-PAM order over a supplied channel sequence.

    Parameters
    ----------
    h_eff:
        One-dimensional, non-zero channel gain for each transmitted symbol.
        The gain is constant for one symbol and may vary between symbols.
    snr_db_array:
        SNR points in dB at which BER is estimated.
    M:
        M-PAM order. It must be a power of two and at least 2.
    rng:
        Optional NumPy generator. A new default generator is created if omitted.

    Returns
    -------
    numpy.ndarray
        BER estimate for each SNR point in ``snr_db_array``.

    Notes
    -----
    The noise variance follows the convention used by ber_full.py:

        sigma = sqrt(1 / (2 * SNR_linear))

    Zero-forcing equalization divides by h_eff. Therefore zero channel gains
    are rejected explicitly instead of producing invalid equalized samples.
    """
    h = np.asarray(h_eff, dtype=float).reshape(-1)
    if h.size == 0:
        raise ValueError("h_eff must not be empty.")
    if not np.all(np.isfinite(h)):
        raise ValueError("h_eff must contain only finite values.")
    if np.any(h == 0.0):
        raise ValueError("h_eff must not contain zero values for ZF equalization.")

    snr_db = np.asarray(snr_db_array, dtype=float).reshape(-1)
    if snr_db.size == 0:
        raise ValueError("snr_db_array must not be empty.")
    if not np.all(np.isfinite(snr_db)):
        raise ValueError("snr_db_array must contain only finite values.")

    generator = rng if rng is not None else np.random.default_rng()
    bits_per_symbol = int(np.log2(M))
    bits = generator.integers(0, 2, size=h.size * bits_per_symbol, dtype=np.uint8)
    transmitted = modulate_mpam_imdd(bits, M)

    ber = np.empty(snr_db.size, dtype=float)
    for index, snr_point_db in enumerate(snr_db):
        snr_linear = 10.0 ** (snr_point_db / 10.0)
        sigma = np.sqrt(1.0 / (2.0 * snr_linear))
        noise = generator.normal(0.0, sigma, size=h.size)

        received = h * transmitted + noise
        equalized = received / h
        detected_bits = demodulate_mpam_imdd(equalized, M)
        ber[index] = calculate_ber(bits, detected_bits)

    return ber

