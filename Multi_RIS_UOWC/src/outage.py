"""
outage.py

Module for calculating the Outage Probability of the Multi-RIS-assisted UWOC system.
Supports evaluating outage across an array of average SNR values for both
Full Receiving and Selective Receiving schemes.
"""

import numpy as np
from typing import Tuple

# Import các hàm tính SNR tức thời từ module snr.py
from src.snr import calculate_gamma_f, calculate_gamma_s, calculate_gamma_1, calculate_gamma_f2, calculate_gamma_s2, calculate_gamma_12

def calculate_outage_rate(gamma_samples: np.ndarray, gamma_th: float) -> float:
    """
    Calculates the empirical outage probability for a given set of SNR realizations.
    Outage occurs when the instantaneous SNR is strictly less than the threshold.
    
    Args:
        gamma_samples (np.ndarray): 1D array of instantaneous SNR samples.
        gamma_th (float): The outage threshold in linear scale.
        
    Returns:
        float: The outage probability (between 0.0 and 1.0).
    """
    # Đếm số lượng mẫu có SNR tức thời nhỏ hơn ngưỡng
    outage_count = np.sum(gamma_samples < gamma_th)
    
    # Chia cho tổng số mẫu để ra xác suất
    return float(outage_count / len(gamma_samples))

def simulate_outage_curves(
    h_k_matrix: np.ndarray, 
    snr_linear_array: np.ndarray, 
    gamma_th: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates Outage Probability curves over a range of average SNR values.
    This function isolates the calculation logic to support reproducing 
    Figures 2, 3, 4, 5, and 6 dynamically.
    
    Args:
        h_k_matrix (np.ndarray): 2D array (num_samples, K) of composite fading coefficients.
        snr_linear_array (np.ndarray): 1D array of average SNR values (linear scale).
        gamma_th (float): The target threshold SNR (linear scale).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - P_out_full: Array of outage probabilities for the Full Receiving Scheme.
            - P_out_sel: Array of outage probabilities for the Selective Receiving Scheme.
    """
    num_snr_points = len(snr_linear_array)
    
    # Khởi tạo mảng kết quả
    p_out_full = np.zeros(num_snr_points)
    p_out_sel = np.zeros(num_snr_points)
    p_out_1 = np.zeros(num_snr_points)
    # Quét qua từng mốc SNR trung bình trên trục hoành
    for i, gamma_bar in enumerate(snr_linear_array):
        # 1. Tính SNR tức thời cho cả 2 sơ đồ tại mức gamma_bar hiện tại
        gamma_f_samples = calculate_gamma_f(gamma_bar, h_k_matrix)
        gamma_s_samples = calculate_gamma_s(gamma_bar, h_k_matrix)
        gamma_1_samples = calculate_gamma_1(gamma_bar, h_k_matrix)
        # 2. Đếm tỷ lệ đứt gãy (Outage) và lưu vào mảng kết quả
        p_out_full[i] = calculate_outage_rate(gamma_f_samples, gamma_th)
        p_out_sel[i] = calculate_outage_rate(gamma_s_samples, gamma_th)
        p_out_1[i] = calculate_outage_rate(gamma_1_samples, gamma_th)

    return p_out_full, p_out_sel, p_out_1
def simulate_outage_curves2(
    h_k_matrix: np.ndarray, 
    snr_linear_array: np.ndarray, 
    gamma_th: float
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simulates Outage Probability curves over a range of average SNR values.
    This function isolates the calculation logic to support reproducing 
    Figures 2, 3, 4, 5, and 6 dynamically.
    
    Args:
        h_k_matrix (np.ndarray): 2D array (num_samples, K) of composite fading coefficients.
        snr_linear_array (np.ndarray): 1D array of average SNR values (linear scale).
        gamma_th (float): The target threshold SNR (linear scale).
        
    Returns:
        Tuple[np.ndarray, np.ndarray]: 
            - P_out_full: Array of outage probabilities for the Full Receiving Scheme.
            - P_out_sel: Array of outage probabilities for the Selective Receiving Scheme.
    """
    num_snr_points = len(snr_linear_array)
    
    # Khởi tạo mảng kết quả
    p_out_full_2 = np.zeros(num_snr_points)
    p_out_sel_2 = np.zeros(num_snr_points)
    p_out_1_2 = np.zeros(num_snr_points)
    # Quét qua từng mốc SNR trung bình trên trục hoành
    for i, gamma_bar in enumerate(snr_linear_array):
        # 1. Tính SNR tức thời cho cả 2 sơ đồ tại mức gamma_bar hiện tại
        gamma_f_samples = calculate_gamma_f2(gamma_bar, h_k_matrix)
        gamma_s_samples = calculate_gamma_s2(gamma_bar, h_k_matrix)
        gamma_1_samples = calculate_gamma_12(gamma_bar, h_k_matrix)
        # 2. Đếm tỷ lệ đứt gãy (Outage) và lưu vào mảng kết quả
        p_out_full_2[i] = calculate_outage_rate(gamma_f_samples, gamma_th)
        p_out_sel_2[i] = calculate_outage_rate(gamma_s_samples, gamma_th)
        p_out_1_2[i] = calculate_outage_rate(gamma_1_samples, gamma_th)

    return p_out_full_2, p_out_sel_2, p_out_1_2

