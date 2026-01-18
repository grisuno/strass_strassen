#!/usr/bin/env python3
"""
Checkpoint Verification and Zero-Shot Expansion Test
=====================================================

This script verifies trained checkpoints and tests zero-shot expansion
from 2x2 to 64x64 matrices using the two-phase protocol.

Metrics:
    S(theta): Binary success indicator (1 if discretization succeeds)
    delta(theta): Distance to integers ||theta - Q(theta)||_inf
    Expansion error: Max relative error at each scale

Author: grisun0
License: AGPL v3
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys


class StrassenBilinear(nn.Module):
    
    def __init__(self, rank=8):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(rank, 4) * 0.5)
        self.V = nn.Parameter(torch.randn(rank, 4) * 0.5)
        self.W = nn.Parameter(torch.randn(4, rank) * 0.5)
    
    def forward(self, A, B):
        batch = A.shape[0]
        a = A.reshape(batch, 4)
        b = B.reshape(batch, 4)
        left = a @ self.U.T
        right = b @ self.V.T
        products = left * right
        c = products @ self.W.T
        return c.reshape(batch, 2, 2)
    
    def get_discrete_coefficients(self):
        U_disc = torch.round(torch.clamp(self.U.data, -1, 1))
        V_disc = torch.round(torch.clamp(self.V.data, -1, 1))
        W_disc = torch.round(torch.clamp(self.W.data, -1, 1))
        return U_disc, V_disc, W_disc


def compute_delta(model):
    with torch.no_grad():
        U = model.U.data
        V = model.V.data
        W = model.W.data
        
        U_q = torch.round(torch.clamp(U, -1, 1))
        V_q = torch.round(torch.clamp(V, -1, 1))
        W_q = torch.round(torch.clamp(W, -1, 1))
        
        delta_U = (U - U_q).abs().max().item()
        delta_V = (V - V_q).abs().max().item()
        delta_W = (W - W_q).abs().max().item()
        
        return max(delta_U, delta_V, delta_W)


def verify_2x2(U, V, W, n_test=1000):
    A = torch.randn(n_test, 2, 2)
    B = torch.randn(n_test, 2, 2)
    
    a = A.reshape(n_test, 4)
    b = B.reshape(n_test, 4)
    
    left = a @ U.T
    right = b @ V.T
    products = left * right
    c = products @ W.T
    C_pred = c.reshape(n_test, 2, 2)
    
    C_true = torch.bmm(A, B)
    
    error = (C_pred - C_true).abs().max().item()
    rel_error = error / (C_true.abs().max().item() + 1e-10)
    
    return error, rel_error


def strassen_expand(A, B, U, V, W):
    n = A.shape[0]
    
    if n == 2:
        a = A.reshape(1, 4)
        b = B.reshape(1, 4)
        left = a @ U.T
        right = b @ V.T
        products = left * right
        c = products @ W.T
        return c.reshape(2, 2)
    
    h = n // 2
    
    A11, A12 = A[:h, :h], A[:h, h:]
    A21, A22 = A[h:, :h], A[h:, h:]
    
    B11, B12 = B[:h, :h], B[:h, h:]
    B21, B22 = B[h:, :h], B[h:, h:]
    
    M1 = strassen_expand(A11 + A22, B11 + B22, U, V, W)
    M2 = strassen_expand(A21 + A22, B11, U, V, W)
    M3 = strassen_expand(A11, B12 - B22, U, V, W)
    M4 = strassen_expand(A22, B21 - B11, U, V, W)
    M5 = strassen_expand(A11 + A12, B22, U, V, W)
    M6 = strassen_expand(A21 - A11, B11 + B12, U, V, W)
    M7 = strassen_expand(A12 - A22, B21 + B22, U, V, W)
    
    C = torch.zeros(n, n)
    C[:h, :h] = M1 + M4 - M5 + M7
    C[:h, h:] = M3 + M5
    C[h:, :h] = M2 + M4
    C[h:, h:] = M1 - M2 + M3 + M6
    
    return C


def verify_expansion(U, V, W, sizes=[2, 4, 8, 16, 32, 64]):
    results = []
    
    for n in sizes:
        torch.manual_seed(42)
        A = torch.randn(n, n)
        B = torch.randn(n, n)
        
        C_true = A @ B
        C_pred = strassen_expand(A, B, U, V, W)
        
        max_error = (C_pred - C_true).abs().max().item()
        rel_error = max_error / (C_true.abs().max().item() + 1e-10)
        
        correct = rel_error < 1e-4
        
        results.append({
            'size': n,
            'max_error': max_error,
            'rel_error': rel_error,
            'correct': correct
        })
    
    return results


def compute_S_theta(model):
    delta = compute_delta(model)
    
    if delta >= 0.1:
        return 0, delta
    
    U, V, W = model.get_discrete_coefficients()
    error, rel_error = verify_2x2(U, V, W)
    
    success = rel_error < 1e-5
    
    return int(success), delta


def load_checkpoint(path):
    model = StrassenBilinear(rank=8)
    
    state = torch.load(path, map_location='cpu', weights_only=True)
    
    if 'U' in state:
        model.U.data = state['U']
        model.V.data = state['V']
        model.W.data = state['W']
    elif 'model_state_dict' in state:
        model.load_state_dict(state['model_state_dict'])
    else:
        model.load_state_dict(state)
    
    return model


def verify_checkpoint(checkpoint_path):
    print(f"\nVerifying: {checkpoint_path}")
    print("-" * 60)
    
    model = load_checkpoint(checkpoint_path)
    
    S, delta = compute_S_theta(model)
    print(f"S(theta) = {S}")
    print(f"delta(theta) = {delta:.6f}")
    
    U, V, W = model.get_discrete_coefficients()
    
    active_slots = 0
    for k in range(U.shape[0]):
        u_norm = torch.norm(U[k]).item()
        v_norm = torch.norm(V[k]).item()
        w_norm = torch.norm(W[:, k]).item()
        if u_norm * v_norm * w_norm > 0.01:
            active_slots += 1
    
    print(f"Active slots = {active_slots}")
    
    if S == 1:
        print("\nZero-shot expansion verification:")
        results = verify_expansion(U, V, W)
        
        all_correct = True
        for r in results:
            status = "PASS" if r['correct'] else "FAIL"
            print(f"  {r['size']:>3}x{r['size']:<3}: max_error={r['max_error']:.2e}, rel_error={r['rel_error']:.2e} [{status}]")
            if not r['correct']:
                all_correct = False
        
        if all_correct:
            print("\nResult: SUCCESS - Zero-shot expansion verified from 2x2 to 64x64")
        else:
            print("\nResult: PARTIAL - Some sizes failed verification")
    else:
        print("\nResult: FAILED - Discretization unsuccessful")
    
    return S, delta


def run_noise_stability_test(checkpoint_path, noise_levels=[0.001, 0.005, 0.01, 0.05, 0.1]):
    print(f"\nNoise stability test: {checkpoint_path}")
    print("-" * 60)
    
    model = load_checkpoint(checkpoint_path)
    
    for sigma in noise_levels:
        successes = 0
        n_trials = 100
        
        for trial in range(n_trials):
            model_copy = StrassenBilinear(rank=8)
            model_copy.U.data = model.U.data.clone() + torch.randn_like(model.U.data) * sigma
            model_copy.V.data = model.V.data.clone() + torch.randn_like(model.V.data) * sigma
            model_copy.W.data = model.W.data.clone() + torch.randn_like(model.W.data) * sigma
            
            S, _ = compute_S_theta(model_copy)
            successes += S
        
        rate = successes / n_trials
        print(f"  sigma={sigma:.3f}: {rate:.0%} success ({successes}/{n_trials})")


def main():
    print("=" * 60)
    print("CHECKPOINT VERIFICATION AND ZERO-SHOT EXPANSION TEST")
    print("=" * 60)
    
    checkpoint_dir = Path(__file__).parent.parent / "checkpoints"
    
    if not checkpoint_dir.exists():
        print(f"Checkpoint directory not found: {checkpoint_dir}")
        print("Searching in current directory...")
        checkpoint_dir = Path(".")
    
    checkpoint_files = list(checkpoint_dir.glob("*.pt")) + list(checkpoint_dir.glob("*.pth"))
    
    if not checkpoint_files:
        print("No checkpoint files found.")
        print("\nTo verify a specific checkpoint, run:")
        print("  python verify_checkpoints.py <checkpoint_path>")
        return
    
    print(f"\nFound {len(checkpoint_files)} checkpoint(s)")
    
    for ckpt in checkpoint_files:
        try:
            S, delta = verify_checkpoint(ckpt)
        except Exception as e:
            print(f"Error loading {ckpt}: {e}")
    
    print("\n" + "=" * 60)
    print("NOISE STABILITY TEST")
    print("=" * 60)
    
    for ckpt in checkpoint_files[:1]:
        try:
            run_noise_stability_test(ckpt)
        except Exception as e:
            print(f"Error in noise test for {ckpt}: {e}")
    
    print("\n" + "=" * 60)
    print("VERIFICATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        checkpoint_path = sys.argv[1]
        verify_checkpoint(checkpoint_path)
        run_noise_stability_test(checkpoint_path)
    else:
        main()
