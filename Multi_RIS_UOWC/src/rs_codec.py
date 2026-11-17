"""Reed-Solomon packet coding for the link-level HARQ simulation."""
from __future__ import annotations

from typing import Iterable

import numpy as np
from reedsolo import ReedSolomonError, RSCodec


def create_codec(n: int, k: int, s: int) -> RSCodec:
    """Create an RS codec and validate the GF(2**s) configuration."""
    if s != 8:
        raise ValueError("The reedsolo backend supports this project with s=8.")
    if n != 255:
        raise ValueError("This implementation requires RS(255, k) over GF(256).")
    if not 0 < k < n:
        raise ValueError("RS k must satisfy 0 < k < n.")
    return RSCodec(nsym=n - k, nsize=n)


def encode_payload(codec: RSCodec, payload: np.ndarray) -> np.ndarray:
    """Encode k payload bytes into one n-byte RS codeword."""
    payload_bytes = np.asarray(payload, dtype=np.uint8).reshape(-1).tobytes()
    return np.frombuffer(codec.encode(payload_bytes), dtype=np.uint8).copy()


def decode_codeword(
    codec: RSCodec,
    received: np.ndarray,
    erasures: Iterable[int],
    expected_payload: np.ndarray | None = None,
) -> bool:
    """Return whether RS decoding succeeds and recovers the known payload.

    The payload comparison is a Monte Carlo ground-truth check. It prevents an
    uncorrectable frame from being counted as successful if a bounded-distance
    decoder happens to return a miscoded codeword without raising an exception.
    """
    try:
        decoded, _, _ = codec.decode(
            np.asarray(received, dtype=np.uint8).reshape(-1).tobytes(),
            erase_pos=list(erasures),
        )
    except ReedSolomonError:
        return False
    if expected_payload is not None:
        expected = np.asarray(expected_payload, dtype=np.uint8).reshape(-1)
        return decoded == expected.tobytes()
    return True
