"""
pointing_error.py

Module for calculating pointing error parameters and generating random variables
representing pointing error fading in the RIS-assisted UWOC channel.
"""

import numpy as np
from scipy.special import erf
from typing import Union, Tuple

def calculate_pointing_error_params(
    D_R: float, 
    w_z: float, 
    sigma_theta: float, 
    sigma_phi: float, 
    L_sr: float, 
    L_rd: float
) -> Tuple[float, float]:
    """
    Calculates the fraction of collected power at zero radial displacement (A_k)
    and the equivalent pointing error parameter squared (xi_k^2) based on the 
    system physical configuration.

    Args:
        D_R (float): Photodetector aperture diameter [m].
        w_z (float): Beam width at the receiver plane [m].
        sigma_theta (float): Standard deviation of beam jitter angle [rad].
        sigma_phi (float): Standard deviation of RIS deflection error angle [rad].
        L_sr (float): Link distance from source to RIS [m].
        L_rd (float): Link distance from RIS to destination [m].

    Returns:
        Tuple[float, float]: A tuple containing (A_k, xi_k_sq).
    """
    # Calculate v_k
    v_k = np.sqrt(np.pi / 2.0) * (D_R / (2.0 * w_z))
    
    # Calculate A_k: fraction of the collected power at zero radial displacement
    A_k = (erf(v_k)) ** 2
    
    # Calculate equivalent beam width squared: w_eq,k^2
    w_eq_k_sq = (w_z ** 2) * (np.sqrt(np.pi) * erf(v_k)) / (2.0 * v_k * np.exp(- (v_k ** 2)))
    
    # Calculate denominator for xi_k^2 (Eq. 10 from the paper)
    jitter_term_beam = 4.0 * (sigma_theta ** 2) * ((L_sr + L_rd) ** 2)
    jitter_term_ris = 16.0 * (sigma_phi ** 2) * (L_rd ** 2)
    
    # Calculate xi_k^2
    xi_k_sq = w_eq_k_sq / (jitter_term_beam + jitter_term_ris)
    
    return A_k, xi_k_sq

def generate_pointing_error(A_k: float, xi_k_sq: float, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    Generates random samples for the pointing error fading coefficient h_{p,k}.
    
    Uses Inverse Transform Sampling on the PDF defined in the paper:
    f(h) = (xi^2 / A^(xi^2)) * h^(xi^2 - 1) for 0 <= h <= A.

    Args:
        A_k (float): Fraction of collected power at zero radial displacement.
        xi_k_sq (float): Equivalent pointing error parameter squared.
        size (int | tuple): Output shape of the generated samples.

    Returns:
        np.ndarray: Array of pointing error fading coefficients h_{p,k}.
    """
    # Draw U from Uniform(0, 1)
    U = np.random.uniform(0.0, 1.0, size=size)
    
    # Inverse CDF method: h_p = A_k * U^(1 / xi_k^2)
    h_p_k = A_k * (U ** (1.0 / xi_k_sq))
    
    return h_p_k

def generate_ris_pointing_error(
    A_k: float, 
    xi_k_sq: float, 
    num_samples: int
) -> np.ndarray:
    """
    Helper function for Monte Carlo simulation to generate pointing errors 
    for a specific RIS across all realizations.

    Args:
        A_k (float): Calculated parameter A_k.
        xi_k_sq (float): Calculated parameter xi_k_sq.
        num_samples (int): Number of Monte Carlo realizations.

    Returns:
        np.ndarray: 1D array of shape (num_samples,) containing h_{p,k}.
    """
    return generate_pointing_error(A_k, xi_k_sq, size=num_samples)