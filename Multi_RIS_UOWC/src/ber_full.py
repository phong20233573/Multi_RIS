"""
ber_full.py

Module for End-to-End BER simulation using Monte Carlo data transmission.
Implements Unipolar 4-PAM modulation (IM-DD compliant), Gray coding, 
UWOC channel scaling, AWGN addition, Zero-Forcing equalization, and Threshold detection.
"""

import numpy as np

def modulate_4pam_imdd(bits: np.ndarray) -> np.ndarray:
    """
    Biến đổi chuỗi bit thành mức cường độ sáng (Unipolar 4-PAM) cho hệ IM-DD.
    4-PAM Levels: 0, +2d, +4d, +6d (Không có giá trị âm)
    Gray Map: 00 -> 0 | 01 -> +2d | 11 -> +4d | 10 -> +6d
    """
    bit_pairs = bits.reshape(-1, 2)
    b0 = bit_pairs[:, 0]
    b1 = bit_pairs[:, 1]
    
    # Ánh xạ mức cường độ quang học (Chỉ có số dương)
    symbols = np.where(b0 == 0, 
                       np.where(b1 == 0, 0, 2), 
                       np.where(b1 == 1, 4, 6))
    
    # Chuẩn hóa năng lượng điện: E_s = 1
    # E_s = (0^2 + 2^2 + 4^2 + 6^2) / 4 = 14
    return symbols / np.sqrt(14.0)

def demodulate_4pam_imdd(y_eq: np.ndarray) -> np.ndarray:
    """
    Khôi phục mức điện áp thu được về lại chuỗi bit bằng các ngưỡng tĩnh dương.
    Sử dụng các ngưỡng quyết định (decision thresholds) tại: 1, 3, 5 (chia cho căn 14)
    """
    th1 = 1.0 / np.sqrt(14.0)
    th2 = 3.0 / np.sqrt(14.0)
    th3 = 5.0 / np.sqrt(14.0)
    
    # Bit 0: Quyết định bởi việc điện áp có vượt qua ngưỡng chính giữa (th2) hay không
    b0_hat = (y_eq > th2).astype(int)
    
    # Bit 1: Quyết định bởi việc điện áp lọt vào khe giữa (th1 và th3)
    b1_hat = ((y_eq > th1) & (y_eq < th3)).astype(int)
    
    return np.column_stack((b0_hat, b1_hat)).flatten()

def simulate_end_to_end_ber(h_eff: np.ndarray, snr_db_array: np.ndarray) -> np.ndarray:
    """
    Giả lập luồng truyền dữ liệu thực tế cho hệ quang học IM-DD
    """
    num_symbols = len(h_eff)
    num_bits = num_symbols * 2  
    
    bits = np.random.randint(0, 2, size=num_bits)
    
    # Gọi hàm điều chế IM-DD mới
    x = modulate_4pam_imdd(bits)
    
    ber_array = np.zeros(len(snr_db_array))
    
    for i, snr_db in enumerate(snr_db_array):
        snr_linear = 10.0 ** (snr_db / 10.0)
        
        # Tính toán nhiễu tĩnh phần cứng
        sigma = np.sqrt(1.0 / (2.0 * snr_linear))
        n = np.random.normal(0, sigma, num_symbols)
        
        # y = hx + n (Lúc này x >= 0 hoàn toàn phản ánh cường độ sáng vật lý)
        y = h_eff * x + n
        
        # Cân bằng kênh ZF
        y_eq = y / h_eff
        
        # Tách sóng với ngưỡng Unipolar
        bits_hat = demodulate_4pam_imdd(y_eq)
        
        errors = np.sum(bits != bits_hat)
        ber_array[i] = errors / num_bits
        
    return ber_array