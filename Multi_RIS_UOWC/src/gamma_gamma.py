"""
gamma_gamma.py

Module for generating Gamma-Gamma distributed random variables representing
the cascaded atmospheric/underwater turbulence fading in the UWOC channel.
"""

import numpy as np
from typing import Union, Tuple

def generate_gamma_gamma(alpha: float, beta: float, size: Union[int, Tuple[int, ...]]) -> np.ndarray:
    """
    Generates random samples from a Gamma-Gamma distribution.

    The cascaded turbulence channel fading coefficient h_{t,kn} from the source 
    to the destination through the n-th reflecting element of the k-th RIS 
    follows a Gamma-Gamma distribution.

    Args:
        alpha (float): Effective number of large-scale eddies of the scattering process.
        beta (float): Effective number of small-scale eddies of the scattering process.
        size (int | tuple): Output shape of the generated samples.

    Returns:
        np.ndarray: Array of Gamma-Gamma distributed random samples with E[h] = 1.
    """
    # Generate X ~ Gamma(alpha, 1/alpha) and Y ~ Gamma(beta, 1/beta)
    # Note: np.random.gamma uses (shape, scale) parameterization
    x = np.random.gamma(shape=alpha, scale=1.0/alpha, size=size)
    y = np.random.gamma(shape=beta, scale=1.0/beta, size=size)
    
    # h_{t,kn} = X * Y
    h_t_kn = x * y
    
    return h_t_kn

def generate_ris_turbulence(alpha: float, beta: float, N_k: int, num_samples: int) -> np.ndarray:
    """
    Generates the total turbulence channel fading coefficient h_{t,k} 
    for the k-th RIS equipped with N_k reflecting elements.
    
    According to the system model, h_{t,k} = sum_{n=1}^{N_k} h_{t,kn}.

    Args:
        alpha (float): Gamma-Gamma parameter alpha.
        beta (float): Gamma-Gamma parameter beta.
        N_k (int): Number of reflecting elements at the k-th RIS.
        num_samples (int): Number of Monte Carlo realizations.

    Returns:
        np.ndarray: 1D array of shape (num_samples,) containing the total 
                    turbulence fading coefficients for the k-th RIS.
    """
    # Generate N_k independent Gamma-Gamma samples for each realization
    # Output shape: (num_samples, N_k)
    h_t_kn_matrix = generate_gamma_gamma(alpha=alpha, beta=beta, size=(num_samples, N_k))
    
    # Sum along the N_k elements (axis=1) to get the total fading for the RIS
    h_t_k = np.sum(h_t_kn_matrix, axis=1)
    
    return h_t_k
