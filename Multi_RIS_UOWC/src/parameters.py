"""
parameters.py

Centralized system and physical parameters for the Multi-RIS UWOC simulation.
Parameters that vary across different simulation figures (K, N, alpha, beta, 
sigma_theta, sigma_phi, gamma_th_db) do not have default values and must be 
explicitly defined when instantiating the configuration.
"""

from dataclasses import dataclass, field
from typing import List

# =============================================================================
# 1. RIS CONFIGURATION
# =============================================================================
@dataclass
class RISConfig:
    K: int                  # Number of RISs (Varies in Fig 2, 7)
    N: List[int]            # List containing number of elements per RIS (Varies in Fig 3, 8)
    rho: float = 1.0        # Reflection coefficient of RIS elements (Constant)

# =============================================================================
# 2. TURBULENCE CONFIGURATION
# =============================================================================
@dataclass
class TurbulenceConfig:
    alpha: float            # Gamma-Gamma parameter alpha (Varies in Fig 4, 9)
    beta: float             # Gamma-Gamma parameter beta (Varies in Fig 4, 9)

# =============================================================================
# 3. POINTING ERROR CONFIGURATION
# =============================================================================
@dataclass
class PointingErrorConfig:
    sigma_theta: float      # Variance of pointing error angle (Varies in Fig 6, 10)
    sigma_phi: float        # Variance of deflection error angle (Varies in Fig 6, 10)
    w_z: float = 1.01e-2    # Beam width omega_k at the receiver plane [m] (Constant)

# =============================================================================
# 4. OPTICAL / UWOC CONFIGURATION
# =============================================================================
@dataclass
class OpticalUWOCConfig:
    lambda_wave: float = 530e-9 # Optical wavelength [m] (Constant)
    D_R: float = 0.05    # Receiver aperture diameter [m] (Constant)
    L1: float = 45.0            # Link distance Source-RIS [m] (Constant)
    L2: float = 45.0            # Link distance RIS-Destination [m] (Constant)
    c_ext: float = 0.15         # Extinction coefficient [m^-1] (Constant)

# =============================================================================
# 5. RECEIVER / OUTAGE CONFIGURATION
# =============================================================================
@dataclass
class ReceiverConfig:
    gamma_th_db: float              # Outage threshold in dB (Varies in Fig 5: 10, 13, 16)
    eta: float = 0.28               # Photodetector responsivity [A/W] (Constant)
    sigma_n2_dbm: float = -100.0    # Noise variance in dBm (Constant)

    @property
    def sigma_n2(self) -> float:
        """Convert dBm to Watts"""
        return 10.0 ** ((self.sigma_n2_dbm - 30.0) / 10.0)

    @property
    def gamma_th(self) -> float:
        """Convert Outage threshold to linear scale"""
        return 10.0 ** (self.gamma_th_db / 10.0)









# =============================================================================
# 6. MAIN SYSTEM CONFIGURATION (BLUEPRINT)
# =============================================================================
@dataclass
class SystemConfig:
    """
    Master configuration class that holds all subsystems.
    """
    ris: RISConfig
    turbulence: TurbulenceConfig
    pointing_error: PointingErrorConfig
    receiver: ReceiverConfig
    
    # Optional parameters with correct defaults from the paper
    optical: OpticalUWOCConfig = field(default_factory=OpticalUWOCConfig)
    M: int = 4  # Modulation order (M-PAM) (Constant)

    def validate(self):
        """
        Perform strict physical and logical consistency checks.
        Must be called after initializing the config in notebooks.
        """
        # 1. RIS check
        if self.ris.K < 1:
            raise ValueError("Number of RISs (K) must be >= 1.")
        if len(self.ris.N) != self.ris.K:
            raise ValueError(f"Array N must contain exactly K={self.ris.K} elements, but got {len(self.ris.N)}.")
        if any(n < 1 for n in self.ris.N):
            raise ValueError("Each RIS must have at least one reflecting element.")
        if not (0.0 <= self.ris.rho <= 1.0):
            raise ValueError("RIS reflection coefficient rho must satisfy 0 <= rho <= 1.")

        # 2. Turbulence check
        if self.turbulence.alpha <= 0.0 or self.turbulence.beta <= 0.0:
            raise ValueError("Gamma-Gamma parameters (alpha, beta) must be strictly > 0.")

        # 3. Pointing error check
        if self.pointing_error.sigma_theta < 0.0 or self.pointing_error.sigma_phi < 0.0:
            raise ValueError("Jitter variances cannot be negative.")
        if self.pointing_error.w_z <= 0.0:
            raise ValueError("Beam width w_z must be > 0.")

        # 4. Optical / Physical check
        if self.optical.D_R <= 0.0:
            raise ValueError("Receiver aperture diameter must be > 0.")
        if self.optical.L1 <= 0.0 or self.optical.L2 <= 0.0:
            raise ValueError("Link distances must be > 0.")
        if self.optical.c_ext < 0.0:
            raise ValueError("Extinction coefficient cannot be negative.")
        
        # 5. Receiver check
        if self.receiver.eta <= 0.0:
            raise ValueError("Photodetector responsivity eta must be > 0.")