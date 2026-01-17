#!/usr/bin/env python3
"""
Strassen Matrix Multiplication via Grokked Tensor Decomposition
Author: grisun0
"""

import torch
from pathlib import Path

_WEIGHTS_PATH = Path(__file__).parent / "weights.pt"
_weights = None


def _load_weights():
    global _weights
    if _weights is None:
        _weights = torch.load(_WEIGHTS_PATH, map_location="cpu", weights_only=True)
    return _weights


def strassen_2x2(A, B):
    w = _load_weights()
    U, V, W = w["U"], w["V"], w["W"]
    
    if A.dim() == 2:
        A = A.unsqueeze(0)
        B = B.unsqueeze(0)
        squeeze = True
    else:
        squeeze = False
    
    batch = A.shape[0]
    a = A.reshape(batch, 4)
    b = B.reshape(batch, 4)
    c = ((a @ U.T) * (b @ V.T)) @ W.T
    result = c.reshape(batch, 2, 2)
    
    if squeeze:
        result = result.squeeze(0)
    
    return result


def strassen(X, Y):
    assert X.shape == Y.shape
    assert X.shape[-1] == X.shape[-2]
    n = X.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0
    
    if n == 2:
        return strassen_2x2(X, Y)
    
    h = n // 2
    A11, A12 = X[..., :h, :h], X[..., :h, h:]
    A21, A22 = X[..., h:, :h], X[..., h:, h:]
    B11, B12 = Y[..., :h, :h], Y[..., :h, h:]
    B21, B22 = Y[..., h:, :h], Y[..., h:, h:]
    
    M1 = strassen(A11 + A22, B11 + B22)
    M2 = strassen(A21 + A22, B11)
    M3 = strassen(A11, B12 - B22)
    M4 = strassen(A22, B21 - B11)
    M5 = strassen(A11 + A12, B22)
    M6 = strassen(A21 - A11, B11 + B12)
    M7 = strassen(A12 - A22, B21 + B22)
    
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    
    top = torch.cat([C11, C12], dim=-1)
    bot = torch.cat([C21, C22], dim=-1)
    return torch.cat([top, bot], dim=-2)


def get_coefficients():
    w = _load_weights()
    return w["U"].clone(), w["V"].clone(), w["W"].clone()


def multiplication_count(n):
    if n == 2:
        return 7
    return 7 * multiplication_count(n // 2)
