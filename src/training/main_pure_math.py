#!/usr/bin/env python3
"""
Strassen Discovery - Pure Mathematics v2
=========================================
Descubre Strassen (7 multiplicaciones) usando SOLO matemáticas.
Estrategia: L1 progresiva para eliminar un slot naturalmente.

Autor: grisun0
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class StrassenModel(nn.Module):
    """
    Descomposición de tensor para multiplicación de matrices 2x2.
    C_ij = sum_r W[ij,r] * (U[r,:] @ a) * (V[r,:] @ b)
    """
    
    def __init__(self, rank=8):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(rank, 4) * 0.3)
        self.V = nn.Parameter(torch.randn(rank, 4) * 0.3)
        self.W = nn.Parameter(torch.randn(4, rank) * 0.3)
    
    def forward(self, A, B):
        batch = A.shape[0]
        a = A.reshape(batch, 4)
        b = B.reshape(batch, 4)
        
        left = a @ self.U.T      # [batch, rank]
        right = b @ self.V.T     # [batch, rank]
        prods = left * right     # elemento a elemento
        c = prods @ self.W.T     # [batch, 4]
        
        return c.reshape(batch, 2, 2)
    
    def slot_norms(self):
        """Norma combinada de cada slot."""
        u = torch.norm(self.U, dim=1)
        v = torch.norm(self.V, dim=1)
        w = torch.norm(self.W, dim=0)
        return u * v * w
    
    def active_count(self, thresh=0.1):
        return (self.slot_norms() > thresh).sum().item()


def gen_data(n, scale=1.0):
    A = torch.randn(n, 2, 2, device=device) * scale
    B = torch.randn(n, 2, 2, device=device) * scale
    return A, B


def train(model, epochs, lr, l1=0.0, batch=512, verbose=True):
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-6)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
    
    for ep in range(epochs):
        A, B = gen_data(batch)
        Cp = model(A, B)
        Ct = torch.bmm(A, B)
        
        mse = ((Cp - Ct)**2).mean()
        loss = mse + l1 * model.slot_norms().sum() if l1 > 0 else mse
        
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        
        if verbose and ep % 1000 == 0:
            mx = (Cp - Ct).abs().max().item()
            act = model.active_count()
            print(f"Ep {ep:5d} | MSE: {mse.item():.2e} | MaxErr: {mx:.2e} | Act: {act}")


def verify(model, n=10000):
    model.eval()
    with torch.no_grad():
        A, B = gen_data(n, scale=2.0)
        Cp = model(A, B)
        Ct = torch.bmm(A, B)
        
        mx = (Cp - Ct).abs().max().item()
        mn = (Cp - Ct).abs().mean().item()
        
        per = (Cp - Ct).abs().reshape(n, -1).max(1)[0]
        acc = (per < 1e-3).float().mean().item() * 100
        act = model.active_count()
        
        print(f"\nVerify ({n}): MaxErr={mx:.2e}, MeanErr={mn:.2e}, Acc={acc:.1f}%, Active={act}")
        return mx < 1e-2, act


def hard_prune(model, keep=7):
    """Poda los slots más débiles, mantiene top-k."""
    with torch.no_grad():
        norms = model.slot_norms()
        _, idx = norms.sort(descending=True)
        weak = idx[keep:].tolist()
        
        for i in weak:
            model.U.data[i] = 0
            model.V.data[i] = 0
            model.W.data[:, i] = 0
        
        print(f"Podados slots: {weak}")
        return idx[:keep].tolist()


def refine_pruned(model, active, epochs=5000, lr=0.01):
    """Refina manteniendo slots podados en cero."""
    opt = optim.Adam(model.parameters(), lr=lr)
    
    for ep in range(epochs):
        A, B = gen_data(512)
        Cp = model(A, B)
        Ct = torch.bmm(A, B)
        
        loss = ((Cp - Ct)**2).mean()
        
        opt.zero_grad()
        loss.backward()
        
        # Cero gradientes de slots podados
        for i in range(model.rank):
            if i not in active:
                model.U.grad[i] = 0
                model.V.grad[i] = 0
                model.W.grad[:, i] = 0
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        
        # Forzar ceros
        with torch.no_grad():
            for i in range(model.rank):
                if i not in active:
                    model.U.data[i] = 0
                    model.V.data[i] = 0
                    model.W.data[:, i] = 0
        
        if ep % 1000 == 0:
            mx = (Cp - Ct).abs().max().item()
            print(f"Refine {ep:5d} | Loss: {loss.item():.2e} | MaxErr: {mx:.2e}")


def show_coeffs(model, active):
    U = model.U.detach().cpu().numpy()
    V = model.V.detach().cpu().numpy()
    W = model.W.detach().cpu().numpy()
    
    print("\n" + "="*50)
    print("COEFICIENTES STRASSEN DESCUBIERTOS")
    print("="*50)
    
    labels_a = ['a11', 'a12', 'a21', 'a22']
    labels_b = ['b11', 'b12', 'b21', 'b22']
    labels_c = ['c11', 'c12', 'c21', 'c22']
    
    for i, s in enumerate(active):
        print(f"\nM{i+1} (slot {s}):")
        
        # Factor A
        terms_a = [f"{U[s,j]:+.2f}*{labels_a[j]}" for j in range(4) if abs(U[s,j]) > 0.05]
        # Factor B
        terms_b = [f"{V[s,j]:+.2f}*{labels_b[j]}" for j in range(4) if abs(V[s,j]) > 0.05]
        # Contribución a C
        contrib = [f"{W[j,s]:+.2f}*{labels_c[j]}" for j in range(4) if abs(W[j,s]) > 0.05]
        
        print(f"  ({' '.join(terms_a)}) * ({' '.join(terms_b)})")
        print(f"  -> {' '.join(contrib)}")


def main():
    print("\n" + "="*60)
    print("   STRASSEN DISCOVERY - PURE MATH v2")
    print("="*60)
    
    # Fase 1: Entrenar con 8 slots
    print("\n--- Fase 1: Entrenamiento inicial (8 slots) ---")
    model = StrassenModel(rank=8).to(device)
    train(model, epochs=10000, lr=0.02, l1=0.0)
    ok, act = verify(model)
    
    # Fase 2: Añadir L1 para promover esparsidad
    print("\n--- Fase 2: Regularización L1 ---")
    train(model, epochs=5000, lr=0.01, l1=0.001)
    verify(model)
    
    # Fase 3: Podar a 7 slots
    print("\n--- Fase 3: Poda a 7 slots ---")
    active = hard_prune(model, keep=7)
    
    # Fase 4: Refinar con 7 slots
    print("\n--- Fase 4: Refinamiento ---")
    refine_pruned(model, active, epochs=10000, lr=0.01)
    ok, act = verify(model)
    
    if ok and act == 7:
        print("\n" + "="*60)
        print("EXITO: Strassen descubierto con 7 multiplicaciones!")
        print("="*60)
        show_coeffs(model, active)
    else:
        print(f"\nResultado: {act} slots, ok={ok}")
    
    # Guardar
    out = Path(__file__).parent / "strassen_result.pt"
    torch.save({
        'U': model.U.cpu(), 'V': model.V.cpu(), 'W': model.W.cpu(),
        'active': active
    }, out)
    print(f"\nGuardado: {out}")


if __name__ == "__main__":
    main()
