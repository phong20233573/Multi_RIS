"""ARQ and IR-HARQ simulation with physical M-PAM and RS(255,223).

IR-HARQ keeps one valid RS(255,223) mother codeword. The first attempt sends
the 223 data bytes; later attempts send disjoint parity chunks. Unsent bytes
are supplied to the RS decoder as erasures.
"""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.modulation import demodulate_mpam_imdd, modulate_mpam_imdd
from src.per_model import AWGN_VARIANCE, amc_select
from src.rs_codec import create_codec, decode_codeword, encode_payload


def _select_M(gamma, theta_db, M_fixed):
    if M_fixed is not None:
        return int(M_fixed)
    M = amc_select(gamma, theta_db)
    return int(M if M else min(theta_db.keys()))


def _new_packet(codec, k, rng):
    payload = rng.integers(0, 256, size=k, dtype=np.uint8)
    return encode_payload(codec, payload), payload


def _transmit_bytes(h, M, data, rng):
    """Transmit bytes through y=h*x+n and return detected bytes and cost."""
    h = float(h)
    if not np.isfinite(h) or h <= 0.0:
        raise ValueError("Channel gain h must be finite and strictly positive.")

    raw = np.asarray(data, dtype=np.uint8).reshape(-1)
    bits = np.unpackbits(raw)
    bits_per_pam = int(np.log2(M))
    padding = (-bits.size) % bits_per_pam
    if padding:
        bits = np.pad(bits, (0, padding))

    transmitted = modulate_mpam_imdd(bits, M)
    noise = rng.normal(0.0, np.sqrt(AWGN_VARIANCE), size=transmitted.size)
    received = h * transmitted + noise
    detected = demodulate_mpam_imdd(received / h, M)
    decoded = np.packbits(detected[: raw.size * 8])
    return decoded, int(transmitted.size)


def _transmit_codeword(h, M, codeword, rng):
    return _transmit_bytes(h, M, codeword, rng)


def _new_result():
    return {
        "delivered": 0,
        "dropped": 0,
        "total_symbols": 0.0,
        "delay_sum": 0,
        "delay_symbols_sum": 0.0,
        "stream_symbols": [],
        "stream_delivered": [],
        "elapsed_symbols": 0.0,
    }


def _validate_rs(rs):
    n, k, s = rs["n"], rs["k"], rs["s"]
    if (n, k, s) != (255, 223, 8):
        raise ValueError("This implementation requires RS(255,223), s=8.")
    return n, k, s


def simulate_arq(h_stream, rs, theta_db, M_fixed, M_max, N, rng):
    """Single-stream ARQ with a real RS encoder and decoder."""
    n, k, s = _validate_rs(rs)
    codec = create_codec(n, k, s)
    result = _new_result()
    processed = 0
    tslot = 0
    state = None

    while processed < N and tslot < len(h_stream):
        h = h_stream[tslot]
        tslot += 1
        if state is None:
            codeword, payload = _new_packet(codec, k, rng)
            M = _select_M(h * h, theta_db, M_fixed)
            state = {"codeword": codeword, "payload": payload, "M": M,
                     "attempt": 1, "symbols": 0.0}
        else:
            state["attempt"] += 1

        decoded, step = _transmit_codeword(h, state["M"], state["codeword"], rng)
        state["symbols"] += step
        result["total_symbols"] += step
        success = decode_codeword(codec, decoded, [], state["payload"])

        if success:
            result["delivered"] += 1
            processed += 1
            result["delay_sum"] += state["attempt"]
            result["delay_symbols_sum"] += state["symbols"]
            state = None
        elif state["attempt"] >= M_max + 1:
            result["dropped"] += 1
            processed += 1
            state = None

    if processed < N:
        raise RuntimeError("Channel stream exhausted before processing all packets.")
    result["stream_symbols"] = [result["total_symbols"]]
    result["stream_delivered"] = [result["delivered"]]
    result["elapsed_symbols"] = result["total_symbols"]
    return result


def _ir_chunks(n, k, attempts):
    parity_indices = np.arange(k, n)
    return list(np.array_split(parity_indices, max(1, attempts - 1)))


def _ir_attempt(h, M, codeword, received, erasures, indices, rng):
    chunk, step = _transmit_bytes(h, M, codeword[indices], rng)
    received[indices] = chunk
    erasures.difference_update(int(index) for index in indices)
    return step


def simulate_irharq(h_stream, rs, theta_db, M_max, N, rng):
    """Single-stream IR-HARQ using one RS mother codeword."""
    n, k, s = _validate_rs(rs)
    codec = create_codec(n, k, s)
    chunks = _ir_chunks(n, k, M_max + 1)
    result = _new_result()
    processed = 0
    tslot = 0
    state = None

    while processed < N and tslot < len(h_stream):
        h = h_stream[tslot]
        tslot += 1
        if state is None:
            codeword, payload = _new_packet(codec, k, rng)
            state = {
                "codeword": codeword,
                "payload": payload,
                "received": np.zeros(n, dtype=np.uint8),
                "erasures": set(range(k, n)),
                "M": _select_M(h * h, theta_db, None),
                "attempt": 1,
                "symbols": 0.0,
            }
            indices = np.arange(k)
        else:
            state["attempt"] += 1
            indices = chunks[state["attempt"] - 2]

        step = _ir_attempt(
            h, state["M"], state["codeword"], state["received"],
            state["erasures"], indices, rng
        )
        state["symbols"] += step
        result["total_symbols"] += step
        success = decode_codeword(
            codec, state["received"], state["erasures"], state["payload"]
        )

        if success:
            result["delivered"] += 1
            processed += 1
            result["delay_sum"] += state["attempt"]
            result["delay_symbols_sum"] += state["symbols"]
            state = None
        elif state["attempt"] >= M_max + 1:
            result["dropped"] += 1
            processed += 1
            state = None

    if processed < N:
        raise RuntimeError("Channel stream exhausted before processing all packets.")
    result["stream_symbols"] = [result["total_symbols"]]
    result["stream_delivered"] = [result["delivered"]]
    result["elapsed_symbols"] = result["total_symbols"]
    return result


def simulate_two_stream(
    h1, h2, rs, theta_db, M_max, N, mode, rng, M_fixed=None
):
    """Two-stream ARQ or IR-HARQ using real RS decoding."""
    if mode not in ("arq", "ir"):
        raise ValueError("mode must be 'arq' or 'ir'")

    n, k, s = _validate_rs(rs)
    codec = create_codec(n, k, s)
    chunks = _ir_chunks(n, k, M_max + 1)
    result = _new_result()
    branches = [
        {"attempt": 0, "M": 0, "symbols": 0.0, "total": 0.0, "delivered": 0},
        {"attempt": 0, "M": 0, "symbols": 0.0, "total": 0.0, "delivered": 0},
    ]
    states = [None, None]
    streams = [h1, h2]
    queue = N
    processed = 0
    tslot = 0

    while processed < N and tslot < min(len(h1), len(h2)):
        for i, stream in enumerate(streams):
            branch = branches[i]
            h = stream[tslot]
            if branch["attempt"] == 0:
                if queue == 0:
                    continue
                queue -= 1
                codeword, payload = _new_packet(codec, k, rng)
                branch["M"] = _select_M(h * h, theta_db, M_fixed)
                branch["attempt"] = 1
                branch["symbols"] = 0.0
                if mode == "arq":
                    states[i] = {"codeword": codeword, "payload": payload}
                    indices = np.arange(n)
                else:
                    states[i] = {
                        "codeword": codeword,
                        "payload": payload,
                        "received": np.zeros(n, dtype=np.uint8),
                        "erasures": set(range(k, n)),
                    }
                    indices = np.arange(k)
            else:
                branch["attempt"] += 1
                indices = (
                    np.arange(n)
                    if mode == "arq"
                    else chunks[branch["attempt"] - 2]
                )

            state = states[i]
            if mode == "arq":
                decoded, step = _transmit_codeword(
                    h, branch["M"], state["codeword"], rng
                )
                success = decode_codeword(
                    codec, decoded, [], state["payload"]
                )
            else:
                step = _ir_attempt(
                    h, branch["M"], state["codeword"], state["received"],
                    state["erasures"], indices, rng
                )
                success = decode_codeword(
                    codec, state["received"], state["erasures"],
                    state["payload"]
                )

            branch["symbols"] += step
            branch["total"] += step
            result["total_symbols"] += step
            if success:
                result["delivered"] += 1
                branch["delivered"] += 1
                processed += 1
                result["delay_sum"] += branch["attempt"]
                result["delay_symbols_sum"] += branch["symbols"]
                branch["attempt"] = 0
                states[i] = None
            elif branch["attempt"] >= M_max + 1:
                result["dropped"] += 1
                processed += 1
                branch["attempt"] = 0
                states[i] = None
        tslot += 1

    if processed < N:
        raise RuntimeError("Channel stream exhausted before processing all packets.")
    result["stream_symbols"] = [branches[0]["total"], branches[1]["total"]]
    result["stream_delivered"] = [
        branches[0]["delivered"], branches[1]["delivered"]
    ]
    result["elapsed_symbols"] = max(
        branches[0]["total"], branches[1]["total"]
    )
    return result
