#!/usr/bin/env python3
"""
Strassen Discovery via Grokkit Framework
=========================================
Usa el teorema de Grokkit para descubrir Strassen con:
- WD (Weight Decay) como presión del motor térmico
- LC (Linear Combination) -> 1
- SP (Sparsity) -> 0  
- Accuracy -> 100%
- Loss -> 0

El operador grokkeado debe cristalizar en la solución de rango 7.

Autor: Matrix Agent
"""

import torch
import torch.nn as nn
import torch.optim as optim
import math

torch.manual_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")


class StrassenOperator(nn.Module):
    """
    Operador espectral para multiplicación de matrices 2x2.
    
    Representa el tensor de rango R:
    C_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)
    
    Donde:
    - U, V: Coeficientes de combinación lineal (LC)
    - W: Coeficientes de reconstrucción
    - Esparsidad (SP): Cuántos slots están activos
    """
    
    def __init__(self, rank=8):
        super().__init__()
        self.rank = rank
        
        # Parámetros del operador
        self.U = nn.Parameter(torch.randn(rank, 4) * 0.5)
        self.V = nn.Parameter(torch.randn(rank, 4) * 0.5)
        self.W = nn.Parameter(torch.randn(4, rank) * 0.5)
    
    def forward(self, A, B):
        """Computa A @ B usando la descomposición tensorial."""
        batch = A.shape[0]
        a = A.reshape(batch, 4)  # [a11, a12, a21, a22]
        b = B.reshape(batch, 4)  # [b11, b12, b21, b22]
        
        # Combinaciones lineales
        left = a @ self.U.T    # [batch, rank]
        right = b @ self.V.T   # [batch, rank]
        
        # Productos (operador bilineal)
        products = left * right  # [batch, rank]
        
        # Reconstrucción
        c = products @ self.W.T  # [batch, 4]
        
        return c.reshape(batch, 2, 2)
    
    def compute_LC(self):
        """
        Linear Combination metric.
        Mide qué tan bien los coeficientes forman combinaciones válidas.
        LC -> 1 significa combinaciones perfectas.
        """
        # LC se mide como la capacidad de reconstruir la multiplicación exacta
        # Usamos la norma de los productos U·V·W comparada con el tensor objetivo
        
        # El tensor objetivo T[i,j,k,l] = δ_{ik}δ_{jl} (kronecker)
        # En nuestra parametrización: T = Σ_r U[r,:]⊗V[r,:]⊗W[:,r]
        
        # Aproximación: LC = 1 - error_relativo
        # Se calcula durante el forward pass
        return 1.0  # Placeholder, se calcula en training
    
    def compute_SP(self):
        """
        Sparsity metric.
        SP -> 0 significa máxima esparsidad (menos slots activos).
        SP = (slots_activos - 7) / rank para normalizar
        """
        # Norma de cada slot
        slot_norms = self.slot_importance()
        
        # Contar slots activos (norma > threshold)
        threshold = 0.1
        active = (slot_norms > threshold).float().sum()
        
        # SP normalizado: 0 cuando hay 7 slots, 1 cuando hay 8
        sp = (active - 7.0) / self.rank
        
        return sp.clamp(0, 1)
    
    def slot_importance(self):
        """Importancia de cada slot basada en normas."""
        u_norm = torch.norm(self.U, dim=1)
        v_norm = torch.norm(self.V, dim=1)
        w_norm = torch.norm(self.W, dim=0)
        return u_norm * v_norm * w_norm
    
    def count_active(self, threshold=0.1):
        """Cuenta slots activos."""
        return (self.slot_importance() > threshold).sum().item()


def generate_batch(n, scale=1.0):
    """Genera batch de matrices aleatorias."""
    A = torch.randn(n, 2, 2, device=device) * scale
    B = torch.randn(n, 2, 2, device=device) * scale
    return A, B


def train_grokkit(epochs=50000, batch_size=256, lr=0.01, wd=1e-4):
    """
    Entrena usando el framework Grokkit.
    
    WD (Weight Decay) actúa como presión termodinámica que:
    1. Empuja hacia soluciones de menor norma
    2. Promueve esparsidad natural (slots débiles -> 0)
    3. Cristaliza el operador en el mínimo de energía (rango 7)
    """
    
    model = StrassenOperator(rank=8).to(device)
    
    # Optimizer con Weight Decay (presión termodinámica)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    # Scheduler para annealing
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10000)
    
    print("\n" + "="*70)
    print("GROKKIT TRAINING - Motor Térmico para Strassen")
    print("="*70)
    print(f"WD (presión): {wd}")
    print(f"Objetivo: LC->1, SP->0, Acc->100%, Loss->0")
    print("="*70)
    
    best_acc = 0
    best_loss = float('inf')
    
    for epoch in range(epochs):
        # Generar datos
        A, B = generate_batch(batch_size)
        
        # Forward
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        
        # === LOSS COMPONENTS ===
        
        # 1. MSE Loss (reconstrucción)
        mse = torch.mean((C_pred - C_true) ** 2)
        
        # 2. LC Loss (combinación lineal -> 1)
        # LC se mide como 1 - error_normalizado
        with torch.no_grad():
            error_norm = torch.norm(C_pred - C_true) / (torch.norm(C_true) + 1e-8)
            lc = 1.0 - error_norm.item()
        
        # 3. SP Loss (esparsidad -> 0)
        sp = model.compute_SP()
        
        # 4. Accuracy
        with torch.no_grad():
            errors = (C_pred - C_true).abs().reshape(batch_size, -1).max(dim=1)[0]
            acc = (errors < 1e-3).float().mean().item() * 100
        
        # === TOTAL LOSS ===
        # El WD ya está en el optimizer, así que solo usamos MSE
        # La presión termodinámica del WD naturalmente reduce normas
        loss = mse
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        # Tracking
        if acc > best_acc:
            best_acc = acc
        if loss.item() < best_loss:
            best_loss = loss.item()
        
        # Logging
        if epoch % 5000 == 0 or epoch == epochs - 1:
            active = model.count_active()
            print(f"Ep {epoch:5d} | Loss: {loss.item():.2e} | "
                  f"LC: {lc:.4f} | SP: {sp.item():.4f} | "
                  f"Acc: {acc:.1f}% | Active: {active}")
    
    return model


def verify_grokking(model, n_test=10000):
    """Verifica que el operador ha grokkeado correctamente."""
    model.eval()
    
    print("\n" + "="*70)
    print("VERIFICACIÓN DE GROKKING")
    print("="*70)
    
    with torch.no_grad():
        # Test en datos nunca vistos
        A, B = generate_batch(n_test, scale=1.0)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        
        # Métricas
        errors = (C_pred - C_true).abs()
        max_err = errors.max().item()
        mean_err = errors.mean().item()
        
        # Accuracy
        per_sample = errors.reshape(n_test, -1).max(dim=1)[0]
        acc = (per_sample < 1e-3).float().mean().item() * 100
        
        # LC
        lc = 1.0 - (torch.norm(C_pred - C_true) / torch.norm(C_true)).item()
        
        # SP
        sp = model.compute_SP().item()
        active = model.count_active()
        
        print(f"Test samples:  {n_test}")
        print(f"Max Error:     {max_err:.2e}")
        print(f"Mean Error:    {mean_err:.2e}")
        print(f"Accuracy:      {acc:.2f}%")
        print(f"LC:            {lc:.6f}")
        print(f"SP:            {sp:.6f}")
        print(f"Active slots:  {active}")
        
        # Verificar criterios
        success = (acc >= 99.9 and active <= 8 and lc > 0.999)
        
        print("\n" + "-"*40)
        if success:
            print("[OK] GROKKING SUCCESSFUL")
        else:
            print("[WARN] Grokking incomplete")
        print("-"*40)
        
        return success, {'acc': acc, 'lc': lc, 'sp': sp, 'active': active}


def progressive_sparsification(model, target_slots=7):
    """
    Fase 2: Esparsificación progresiva.
    Reduce gradualmente a 7 slots manteniendo accuracy.
    """
    print("\n" + "="*70)
    print(f"ESPARSIFICACIÓN PROGRESIVA -> {target_slots} slots")
    print("="*70)
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    # Identificar slot más débil
    with torch.no_grad():
        importance = model.slot_importance()
        _, sorted_idx = importance.sort()
        slots_to_prune = sorted_idx[:(model.rank - target_slots)].tolist()
        print(f"Slots a podar: {slots_to_prune}")
    
    # Podar gradualmente
    for slot in slots_to_prune:
        print(f"\n--- Podando slot {slot} ---")
        
        # Reducir gradualmente
        for step in range(2000):
            # Decay factor
            decay = 1.0 - (step / 2000) * 0.99
            
            with torch.no_grad():
                model.U.data[slot] *= decay
                model.V.data[slot] *= decay
                model.W.data[:, slot] *= decay
            
            # Entrenar para compensar
            A, B = generate_batch(256)
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            
            loss = torch.mean((C_pred - C_true) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            
            # No actualizar el slot podado
            model.U.grad[slot] = 0
            model.V.grad[slot] = 0
            model.W.grad[:, slot] = 0
            
            optimizer.step()
        
        # Forzar a cero
        with torch.no_grad():
            model.U.data[slot] = 0
            model.V.data[slot] = 0
            model.W.data[:, slot] = 0
        
        # Verificar
        A, B = generate_batch(1000)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        err = (C_pred - C_true).abs().max().item()
        active = model.count_active()
        print(f"  MaxErr: {err:.2e} | Active: {active}")
    
    # Refinamiento final
    print("\n--- Refinamiento final ---")
    for epoch in range(10000):
        A, B = generate_batch(256)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        
        loss = torch.mean((C_pred - C_true) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        
        # Mantener slots podados en cero
        for slot in slots_to_prune:
            model.U.grad[slot] = 0
            model.V.grad[slot] = 0
            model.W.grad[:, slot] = 0
        
        optimizer.step()
        
        # Forzar ceros
        with torch.no_grad():
            for slot in slots_to_prune:
                model.U.data[slot] = 0
                model.V.data[slot] = 0
                model.W.data[:, slot] = 0
        
        if epoch % 2000 == 0:
            err = (C_pred - C_true).abs().max().item()
            print(f"Refine {epoch:5d} | Loss: {loss.item():.2e} | MaxErr: {err:.2e}")
    
    return model


def main():
    """Pipeline principal Grokkit para Strassen."""
    
    print("\n" + "="*70)
    print("   GROKKIT: Descubrimiento de Strassen via Motor Térmico")
    print("="*70)
    
    # Fase 1: Grokking inicial con WD
    print("\n### FASE 1: Grokking con Weight Decay ###")
    model = train_grokkit(
        epochs=30000,
        batch_size=256,
        lr=0.02,
        wd=1e-4  # Presión termodinámica
    )
    
    success, metrics = verify_grokking(model)
    
    # Fase 2: Esparsificación si es necesario
    if metrics['active'] > 7:
        print("\n### FASE 2: Esparsificación Progresiva ###")
        model = progressive_sparsification(model, target_slots=7)
        success, metrics = verify_grokking(model)
    
    # Resultado final
    print("\n" + "="*70)
    print("RESULTADO FINAL")
    print("="*70)
    print(f"LC:       {metrics['lc']:.6f} {'[OK]' if metrics['lc'] > 0.999 else '[X]'}")
    print(f"SP:       {metrics['sp']:.6f} {'[OK]' if metrics['sp'] < 0.01 else '[X]'}")
    print(f"Accuracy: {metrics['acc']:.2f}% {'[OK]' if metrics['acc'] >= 99.9 else '[X]'}")
    print(f"Active:   {metrics['active']} slots {'[OK]' if metrics['active'] == 7 else '[X]'}")
    
    # Guardar
    torch.save({
        'U': model.U.data.cpu(),
        'V': model.V.data.cpu(),
        'W': model.W.data.cpu(),
        'metrics': metrics
    }, "strassen_grokkit.pt")
    print("\nGuardado: strassen_grokkit.pt")
    
    return model


if __name__ == "__main__":
    main()
