import numpy as np

def calculate_gamma_f(gamma_bar: float, h_k_matrix: np.ndarray) -> np.ndarray:
    K = h_k_matrix.shape[1]
    h_f = np.sum(h_k_matrix, axis=1)
    # Phương trình 12: Có chia cho K^2
    gamma_f = gamma_bar * (h_f ** 2) / (K ** 2)
    return gamma_f
def calculate_gamma_s(gamma_bar: float, h_k_matrix: np.ndarray) -> np.ndarray:
    h_k_sq_matrix = h_k_matrix ** 2
    max_h_k_sq = np.max(h_k_sq_matrix, axis=1)
    # Phương trình 34: Không chia cho K
    gamma_s = gamma_bar * max_h_k_sq
    return gamma_s
def calculate_gamma_1(gamma_bar:float, h_k_matrix: np.ndarray) -> np.ndarray:
     h_f = np.sum(h_k_matrix, axis=1)
     h_k_sq_matrix = h_k_matrix ** 2
     h_1=np.sum(h_k_sq_matrix, axis=1)
     gamma_1 = gamma_bar * (h_1** 2) / (h_f ** 2)
     return gamma_1

def calculate_gamma_f2(gamma_bar: float, h_k_matrix: np.ndarray) -> np.ndarray:
    K = h_k_matrix.shape[1]
    h_f = np.sum(h_k_matrix, axis=1)
    # Phương trình 12: Có chia cho K
    gamma_f2 = gamma_bar * (h_f ** 2) / K 
    return gamma_f2
def calculate_gamma_s2(gamma_bar: float, h_k_matrix: np.ndarray) -> np.ndarray:
    h_k_sq_matrix = h_k_matrix ** 2
    max_h_k_sq = np.max(h_k_sq_matrix, axis=1)
    # Phương trình 34: Không chia cho K
    gamma_s2 = gamma_bar * max_h_k_sq 
    return gamma_s2
def calculate_gamma_12(gamma_bar:float, h_k_matrix: np.ndarray) -> np.ndarray:
     h_k_sq_matrix = h_k_matrix ** 2
     h_1=np.sum(h_k_sq_matrix, axis=1)
     gamma_12 = gamma_bar * (h_1) 
     return gamma_12