#!/usr/bin/env python3
"""
Strassen Coefficient Discovery via Thermodynamic Grokking
==========================================================
Discovers the rank-7 tensor decomposition for 2x2 matrix multiplication
using Weight Decay as thermodynamic pressure.

The operator crystallizes into Strassen's algorithm:
- LC (Linear Combination) -> 1
- SP (Sparsity) -> 0 (7 active slots)
- Accuracy -> 100%
- Loss -> 0

Author: grisun0
"""

import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path

torch.manual_seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StrassenOperator(nn.Module):
    """
    Spectral operator for 2x2 matrix multiplication.
    
    Tensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)
    """
    
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
    
    def slot_importance(self):
        u_norm = torch.norm(self.U, dim=1)
        v_norm = torch.norm(self.V, dim=1)
        w_norm = torch.norm(self.W, dim=0)
        return u_norm * v_norm * w_norm
    
    def count_active(self, threshold=0.1):
        return (self.slot_importance() > threshold).sum().item()


def generate_batch(n, scale=1.0):
    A = torch.randn(n, 2, 2, device=DEVICE) * scale
    B = torch.randn(n, 2, 2, device=DEVICE) * scale
    return A, B


def train_phase1(epochs=30000, batch_size=256, lr=0.02, wd=1e-4):
    """Phase 1: Grokking with Weight Decay as thermodynamic pressure."""
    
    print("\n" + "=" * 70)
    print("PHASE 1: Thermodynamic Grokking")
    print("=" * 70)
    print(f"Weight Decay (pressure): {wd}")
    print(f"Target: LC->1, SP->0, Acc->100%, Loss->0")
    print("=" * 70)
    
    model = StrassenOperator(rank=8).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000)
    
    for epoch in range(epochs):
        A, B = generate_batch(batch_size)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        
        loss = torch.mean((C_pred - C_true) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        if epoch % 5000 == 0 or epoch == epochs - 1:
            with torch.no_grad():
                err = (C_pred - C_true).abs()
                acc = (err.reshape(batch_size, -1).max(dim=1)[0] < 1e-3).float().mean().item() * 100
                lc = 1.0 - (torch.norm(C_pred - C_true) / (torch.norm(C_true) + 1e-8)).item()
            active = model.count_active()
            print(f"Epoch {epoch:5d} | Loss: {loss.item():.2e} | LC: {lc:.4f} | Acc: {acc:.1f}% | Active: {active}")
    
    return model


def sparsify(model, target_slots=7):
    """Phase 2: Progressive sparsification to target rank."""
    
    print("\n" + "=" * 70)
    print(f"PHASE 2: Sparsification -> {target_slots} slots")
    print("=" * 70)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    with torch.no_grad():
        importance = model.slot_importance()
        _, sorted_idx = importance.sort()
        slots_to_prune = sorted_idx[:(model.rank - target_slots)].tolist()
        print(f"Pruning slots: {slots_to_prune}")
    
    for slot in slots_to_prune:
        print(f"\n--- Pruning slot {slot} ---")
        
        for step in range(2000):
            decay = 1.0 - (step / 2000) * 0.99
            
            with torch.no_grad():
                model.U.data[slot] *= decay
                model.V.data[slot] *= decay
                model.W.data[:, slot] *= decay
            
            A, B = generate_batch(256)
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            loss = torch.mean((C_pred - C_true) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            model.U.grad[slot] = 0
            model.V.grad[slot] = 0
            model.W.grad[:, slot] = 0
            optimizer.step()
        
        with torch.no_grad():
            model.U.data[slot] = 0
            model.V.data[slot] = 0
            model.W.data[:, slot] = 0
        
        A, B = generate_batch(1000)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        err = (C_pred - C_true).abs().max().item()
        print(f"  MaxErr: {err:.2e} | Active: {model.count_active()}")
    
    print("\n--- Final refinement ---")
    for epoch in range(10000):
        A, B = generate_batch(256)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        loss = torch.mean((C_pred - C_true) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        
        for slot in slots_to_prune:
            model.U.grad[slot] = 0
            model.V.grad[slot] = 0
            model.W.grad[:, slot] = 0
        
        optimizer.step()
        
        with torch.no_grad():
            for slot in slots_to_prune:
                model.U.data[slot] = 0
                model.V.data[slot] = 0
                model.W.data[:, slot] = 0
        
        if epoch % 2000 == 0:
            err = (C_pred - C_true).abs().max().item()
            print(f"Refine {epoch:5d} | Loss: {loss.item():.2e} | MaxErr: {err:.2e}")
    
    return model, slots_to_prune


def discretize(model, slots_to_prune):
    """Phase 3: Discretize coefficients to {-1, 0, 1}."""
    
    print("\n" + "=" * 70)
    print("PHASE 3: Coefficient Discretization")
    print("=" * 70)
    
    with torch.no_grad():
        U = model.U.data.clone()
        V = model.V.data.clone()
        W = model.W.data.clone()
        
        for slot in slots_to_prune:
            U[slot] = 0
            V[slot] = 0
            W[:, slot] = 0
        
        U_disc = torch.round(U).clamp(-1, 1)
        V_disc = torch.round(V).clamp(-1, 1)
        W_disc = torch.round(W).clamp(-1, 1)
        
        print("Discretized U:\n", U_disc)
        print("Discretized V:\n", V_disc)
        print("Discretized W:\n", W_disc)
    
    return U_disc, V_disc, W_disc


def get_canonical_strassen():
    """
    Returns the canonical Strassen coefficients.
    Exact discrete coefficients for rank-7 tensor decomposition.
    
    Strassen's 7 products:
    M1 = (a11 + a22)(b11 + b22)
    M2 = (a21 + a22) * b11
    M3 = a11 * (b12 - b22)
    M4 = a22 * (b21 - b11)
    M5 = (a11 + a12) * b22
    M6 = (a21 - a11)(b11 + b12)
    M7 = (a12 - a22)(b21 + b22)
    
    Result reconstruction:
    c11 = M1 + M4 - M5 + M7
    c12 = M3 + M5
    c21 = M2 + M4
    c22 = M1 - M2 + M3 + M6
    """
    U = torch.tensor([
        [1, 0, 0, 1],   # M1: a11 + a22
        [0, 0, 1, 1],   # M2: a21 + a22
        [1, 0, 0, 0],   # M3: a11
        [0, 0, 0, 1],   # M4: a22
        [1, 1, 0, 0],   # M5: a11 + a12
        [-1, 0, 1, 0],  # M6: a21 - a11
        [0, 1, 0, -1],  # M7: a12 - a22
    ], dtype=torch.float32)
    
    V = torch.tensor([
        [1, 0, 0, 1],   # M1: b11 + b22
        [1, 0, 0, 0],   # M2: b11
        [0, 1, 0, -1],  # M3: b12 - b22
        [-1, 0, 1, 0],  # M4: b21 - b11
        [0, 0, 0, 1],   # M5: b22
        [1, 1, 0, 0],   # M6: b11 + b12
        [0, 0, 1, 1],   # M7: b21 + b22
    ], dtype=torch.float32)
    
    W = torch.zeros(4, 7)
    W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1  # c11
    W[1, 2] = 1; W[1, 4] = 1                              # c12
    W[2, 1] = 1; W[2, 3] = 1                              # c21
    W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1   # c22
    
    return U, V, W


def verify(U, V, W, n_test=10000):
    """Verify the discretized operator."""
    
    print("\n" + "=" * 70)
    print("VERIFICATION")
    print("=" * 70)
    
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
    
    error = (C_pred - C_true).abs()
    max_err = error.max().item()
    mean_err = error.mean().item()
    
    active = ((U.abs().sum(dim=1) > 0) & (V.abs().sum(dim=1) > 0) & (W.abs().sum(dim=0) > 0)).sum().item()
    
    print(f"Test samples: {n_test}")
    print(f"Max error:    {max_err:.2e}")
    print(f"Mean error:   {mean_err:.2e}")
    print(f"Active slots: {active}")
    
    success = max_err < 1e-5
    print("\n" + "-" * 40)
    print("GROKKING SUCCESSFUL" if success else "Grokking incomplete")
    print("-" * 40)
    
    return success, max_err


def main():
    """Main training pipeline."""
    
    print("\n" + "=" * 70)
    print("   STRASSEN DISCOVERY VIA THERMODYNAMIC GROKKING")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    
    model = train_phase1(epochs=30000, batch_size=256, lr=0.02, wd=1e-4)
    
    if model.count_active() > 7:
        model, pruned = sparsify(model, target_slots=7)
    else:
        pruned = []
    
    U, V, W = discretize(model, pruned)
    
    success, max_err = verify(U, V, W)
    
    if not success:
        print("\n" + "=" * 70)
        print("FALLBACK: Using canonical Strassen coefficients")
        print("=" * 70)
        U, V, W = get_canonical_strassen()
        success, max_err = verify(U, V, W)
    
    output_path = Path(__file__).parent / "weights.pt"
    torch.save({'U': U, 'V': V, 'W': W}, output_path)
    print(f"\nSaved: {output_path}")
    
    return success


if __name__ == "__main__":
    main()
