"""
channel.py

Module for generating the complete Underwater Wireless Optical Communication (UWOC)
channel fading coefficients. It combines deterministic path loss, turbulence fading,
and pointing error fading for the multi-RIS-assisted system.
"""
import sys
import os
import numpy as np
from typing import Tuple
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.parameters import SystemConfig
# Assume these are imported from your local src directory
from src.parameters import SystemConfig
from src.gamma_gamma import generate_ris_turbulence
from src.pointing_error import calculate_pointing_error_params, generate_ris_pointing_error

def calculate_path_loss(config: SystemConfig) -> float:
    """
    Calculates the deterministic path loss based on Beer-Lambert's law.
    Eq (3) in the paper: h_l = exp[-c * (L_k^sr + L_k^rd)].
    
    Args:
        config (SystemConfig): The system configuration object.
        
    Returns:
        float: The path loss coefficient h_l.
    """
    # L = L_sr + L_rd
    total_distance = config.optical.L1 + config.optical.L2
    
    # h_l = exp(-c * L)
    h_l = np.exp(-config.optical.c_ext * total_distance)
    
    return h_l

def generate_composite_fading(config: SystemConfig, num_samples: int) -> np.ndarray:
    """
    Generates the composite channel fading coefficients h_k = h_{p,k} * h_{t,k}
    for all K RISs over multiple Monte Carlo realizations.
    
    Args:
        config (SystemConfig): The system configuration object.
        num_samples (int): Number of Monte Carlo realizations.
        
    Returns:
        np.ndarray: A 2D array of shape (num_samples, K) containing the 
                    composite fading coefficient h_k for each RIS.
    """
    K = config.ris.K
    
    # Initialize the matrix to hold h_k for all K RISs
    h_k_matrix = np.zeros((num_samples, K))
    
    # Calculate deterministic pointing error parameters (A_k, xi_k_sq)
    # Assuming all RISs have the same physical placement/distances in this setup
    A_k, xi_k_sq = calculate_pointing_error_params(
        D_R=config.optical.D_R,
        w_z=config.pointing_error.w_z,
        sigma_theta=config.pointing_error.sigma_theta,
        sigma_phi=config.pointing_error.sigma_phi,
        L_sr=config.optical.L1,
        L_rd=config.optical.L2
    )
    
    # Generate fading for each RIS
    for k in range(K):
        # 1. Generate pointing error fading h_{p,k} for the k-th RIS
        h_p_k = generate_ris_pointing_error(
            A_k=A_k, 
            xi_k_sq=xi_k_sq, 
            num_samples=num_samples
        )
        
        # 2. Generate total turbulence fading h_{t,k} for the k-th RIS
        # N_k is the number of reflecting elements for this specific RIS
        N_k = config.ris.N[k]
        h_t_k = generate_ris_turbulence(
            alpha=config.turbulence.alpha,
            beta=config.turbulence.beta,
            N_k=N_k,
            num_samples=num_samples
        )
        
        # 3. Composite fading h_k = h_{p,k} * h_{t,k}
        h_k_matrix[:, k] = h_p_k * h_t_k
        
    return h_k_matrix

def generate_full_channel(config: SystemConfig, num_samples: int) -> Tuple[float, np.ndarray]:
    """
    Convenience function that returns both the path loss and the random fading matrix.
    
    Args:
        config (SystemConfig): The system configuration object.
        num_samples (int): Number of Monte Carlo realizations.
        
    Returns:
        Tuple[float, np.ndarray]: (path_loss_coefficient, h_k_matrix)
    """
    h_l = calculate_path_loss(config)
    h_k_matrix = generate_composite_fading(config, num_samples)
    
    return h_l, h_k_matrix