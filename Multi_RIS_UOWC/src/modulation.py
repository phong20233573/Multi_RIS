"""
modulation.py

General Gray-coded unipolar M-PAM modulation/demodulation for IM/DD.

Constellation before normalization:
    0, 2, 4, ..., 2(M-1)

Gray mapping follows the same convention as the original 4-PAM code:
    00 -> 0, 01 -> 2, 11 -> 4, 10 -> 6

Normalization:
    E[x^2] = 1 for equiprobable symbols.

Important:
    This preserves the electrical mean-square normalization used by the
    supplied 4-PAM implementation. It does not, by itself, guarantee equal
    average optical intensity (E[x]) or equal peak optical power across M.
"""

import numpy as np


def _bits_per_symbol(M: int) -> int:
    """Return log2(M), requiring M to be a power of two and M >= 2."""
    if isinstance(M, (bool, np.bool_)) or not isinstance(M, (int, np.integer)):
        raise TypeError("M must be an integer, e.g. 2, 4, 8, or 16.")

    M = int(M)
    if M < 2 or (M & (M - 1)) != 0:
        raise ValueError("M must be a power of two and >= 2.")

    return M.bit_length() - 1


def _normalization_factor(M: int) -> float:
    """
    Normalization factor for unipolar levels [0, 2, ..., 2(M-1)].

    For equiprobable symbols:
        E[a^2] = (1/M) * sum((2*k)^2, k=0,...,M-1)
               = 2*(M-1)*(2*M-1)/3

    For M=4, this gives sqrt(14), matching the original code.
    """
    _bits_per_symbol(M)  # validate M
    return float(np.sqrt(2.0 * (M - 1) * (2 * M - 1) / 3.0))


def _gray_to_binary(gray: np.ndarray, M: int) -> np.ndarray:
    """Convert integer Gray-code labels to ordinary binary indices."""
    binary = gray.astype(np.int64, copy=True)
    shift = 1
    while shift < M:
        binary ^= binary >> shift
        shift <<= 1
    return binary


def modulate_mpam_imdd(bits: np.ndarray, M: int) -> np.ndarray:
    """
    Map bits to Gray-coded, unipolar M-PAM intensity levels for IM/DD.

    Parameters
    ----------
    bits : array_like
        Input bits containing only 0 and 1. The number of bits must be
        divisible by log2(M).
    M : int
        Modulation order: 2, 4, 8, 16, ... (power of two).

    Returns
    -------
    numpy.ndarray
        One-dimensional real-valued symbol sequence. For equiprobable
        symbols, the constellation has mean-square value E[x^2] = 1.

    Notes
    -----
    For M=4, this exactly reproduces the supplied mapping:
        00 -> 0, 01 -> 2/sqrt(14),
        11 -> 4/sqrt(14), 10 -> 6/sqrt(14).
    """
    k = _bits_per_symbol(M)
    bits = np.asarray(bits).reshape(-1)

    if bits.size == 0:
        raise ValueError("bits must not be empty.")
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("bits must contain only 0 and 1.")
    if bits.size % k != 0:
        raise ValueError(
            f"Number of bits ({bits.size}) must be divisible by log2(M)={k}."
        )

    bit_groups = bits.astype(np.uint8).reshape(-1, k)

    # Interpret each input group as an integer Gray-code label, MSB first.
    gray = np.zeros(bit_groups.shape[0], dtype=np.int64)
    for bit_position in range(k):
        gray = (gray << 1) | bit_groups[:, bit_position]

    # Gray label -> binary constellation index (0,...,M-1).
    binary_index = _gray_to_binary(gray, M)

    # Unipolar levels: 0, 2, 4, ..., 2(M-1), then normalize E[x^2] to 1.
    symbols = 2.0 * binary_index / _normalization_factor(M)
    return symbols.astype(float)


def demodulate_mpam_imdd(received_signal: np.ndarray, M: int) -> np.ndarray:
    """
    Detect Gray-coded unipolar M-PAM symbols using nearest-level detection.

    Parameters
    ----------
    received_signal : array_like
        Real-valued received samples after the channel gain/equalization
        used by the receiver. Samples must be on the same amplitude scale as
        the normalized transmit constellation.
    M : int
        Modulation order used by the corresponding transmitter.

    Returns
    -------
    numpy.ndarray
        Detected bits as a one-dimensional uint8 array.

    Notes
    -----
    Nearest-level detection is equivalent to threshold detection at the
    midpoints between adjacent constellation levels.
    """
    k = _bits_per_symbol(M)
    received_signal = np.asarray(received_signal)

    if np.iscomplexobj(received_signal):
        raise ValueError("IM/DD M-PAM received_signal must be real-valued.")

    received_signal = received_signal.astype(float).reshape(-1)
    if received_signal.size == 0:
        raise ValueError("received_signal must not be empty.")
    if not np.all(np.isfinite(received_signal)):
        raise ValueError("received_signal contains NaN or Inf.")

    normalization = _normalization_factor(M)
    levels = 2.0 * np.arange(M, dtype=float) / normalization

    # Decide the closest unipolar amplitude level.
    distances = np.abs(received_signal[:, np.newaxis] - levels[np.newaxis, :])
    binary_index_hat = np.argmin(distances, axis=1)

    # Binary constellation index -> Gray-code label.
    gray_hat = binary_index_hat ^ (binary_index_hat >> 1)

    # Convert each Gray label to k bits, most-significant bit first.
    bit_shifts = np.arange(k - 1, -1, -1, dtype=np.int64)
    detected_groups = (gray_hat[:, np.newaxis] >> bit_shifts) & 1

    return detected_groups.astype(np.uint8).reshape(-1)


def calculate_ber(bits: np.ndarray, detected_bits: np.ndarray) -> float:
    """Calculate empirical BER = number of bit errors / total transmitted bits."""
    bits = np.asarray(bits).reshape(-1)
    detected_bits = np.asarray(detected_bits).reshape(-1)

    if bits.size == 0:
        raise ValueError("Bit sequences must not be empty.")
    if bits.size != detected_bits.size:
        raise ValueError("bits and detected_bits must have the same length.")
    if not np.all(np.isin(bits, [0, 1])):
        raise ValueError("bits must contain only 0 and 1.")
    if not np.all(np.isin(detected_bits, [0, 1])):
        raise ValueError("detected_bits must contain only 0 and 1.")

    errors = np.count_nonzero(bits != detected_bits)
    return float(errors / bits.size)


# Backward-compatible wrappers for code that calls the old 4-PAM functions.
def modulate_4pam_imdd(bits: np.ndarray) -> np.ndarray:
    """Compatibility wrapper: original unipolar Gray-coded 4-PAM mapper."""
    return modulate_mpam_imdd(bits, M=4)


def demodulate_4pam_imdd(received_signal: np.ndarray) -> np.ndarray:
    """Compatibility wrapper: original unipolar Gray-coded 4-PAM demapper."""
    return demodulate_mpam_imdd(received_signal, M=4)
