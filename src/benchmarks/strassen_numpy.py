#!/usr/bin/env python3
"""
Strassen Matrix Multiplication - NumPy Optimized
=================================================
Zero-shot expansion using the grokked 2x2 operator.

Author: grisun0
"""

import numpy as np
import torch
from pathlib import Path
from functools import lru_cache

_WEIGHTS_PATH = Path(__file__).parent / "weights.pt"
_U, _V, _W = None, None, None


def _load_weights():
    global _U, _V, _W
    if _U is None:
        w = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
        _U = w["U"].numpy()
        _V = w["V"].numpy()
        _W = w["W"].numpy()
    return _U, _V, _W


def strassen_2x2_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Strassen 2x2 using grokked coefficients."""
    U, V, W = _load_weights()
    
    a = A.flatten()  # [4]
    b = B.flatten()  # [4]
    
    left = a @ U.T   # [7]
    right = b @ V.T  # [7]
    products = left * right  # [7]
    c = products @ W.T  # [4]
    
    return c.reshape(2, 2)


def strassen_numpy(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Recursive Strassen using NumPy."""
    n = A.shape[0]
    
    if n == 2:
        return strassen_2x2_numpy(A, B)
    
    if n == 1:
        return A * B
    
    h = n // 2
    
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]
    
    M1 = strassen_numpy(A11 + A22, B11 + B22)
    M2 = strassen_numpy(A21 + A22, B11)
    M3 = strassen_numpy(A11, B12 - B22)
    M4 = strassen_numpy(A22, B21 - B11)
    M5 = strassen_numpy(A11 + A12, B22)
    M6 = strassen_numpy(A21 - A11, B11 + B12)
    M7 = strassen_numpy(A12 - A22, B21 + B22)
    
    C = np.zeros((n, n), dtype=A.dtype)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    
    return C


def strassen_hybrid(A: np.ndarray, B: np.ndarray, threshold: int = 64) -> np.ndarray:
    """
    Hybrid Strassen: use Strassen for large matrices, NumPy for small.
    This is faster because NumPy matmul is highly optimized for small matrices.
    """
    n = A.shape[0]
    
    if n <= threshold:
        return A @ B
    
    h = n // 2
    
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]
    
    M1 = strassen_hybrid(A11 + A22, B11 + B22, threshold)
    M2 = strassen_hybrid(A21 + A22, B11, threshold)
    M3 = strassen_hybrid(A11, B12 - B22, threshold)
    M4 = strassen_hybrid(A22, B21 - B11, threshold)
    M5 = strassen_hybrid(A11 + A12, B22, threshold)
    M6 = strassen_hybrid(A21 - A11, B11 + B12, threshold)
    M7 = strassen_hybrid(A12 - A22, B21 + B22, threshold)
    
    C = np.zeros((n, n), dtype=A.dtype)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    
    return C


def multiplication_count(n: int) -> int:
    """Count multiplications used by Strassen."""
    if n <= 2:
        return 7 if n == 2 else 1
    return 7 * multiplication_count(n // 2)


if __name__ == "__main__":
    # Quick test
    n = 64
    A = np.random.randn(n, n).astype(np.float32)
    B = np.random.randn(n, n).astype(np.float32)
    
    C_strassen = strassen_hybrid(A, B)
    C_standard = A @ B
    
    error = np.abs(C_strassen - C_standard).max()
    print(f"Test {n}x{n}: max error = {error:.2e}")
