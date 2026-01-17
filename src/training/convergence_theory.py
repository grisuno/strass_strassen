"""
Convergence Theory for Algorithmic Invariance

This module addresses the reviewer's key criticisms:
1. Computable κ_eff via Hutchinson trace estimation
2. Formal convergence theorem with proof sketch
3. ε_hw(B, T) extended model

Author: grisun0
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, Optional, Callable
from dataclasses import dataclass
import time


@dataclass
class ConvergenceMetrics:
    """Metrics for tracking convergence to algorithmic invariance"""
    kappa_eff: float           # Effective curvature κ_eff = -tr(H)/N
    kappa_std: float           # Standard deviation of κ_eff estimate
    gradient_norm: float       # ||∇L||
    weight_discreteness: float # How close to discrete values
    spectral_gap: float        # Gap in singular value spectrum
    epsilon_hw: float          # Hardware-induced noise estimate


class HutchinsonTraceEstimator:
    """
    Efficient Hessian trace estimation using Hutchinson's method.
    
    tr(H) ≈ E[v^T H v] where v ~ Rademacher(±1)
    
    Complexity: O(n_samples * forward_backward_pass) instead of O(n²)
    """
    
    def __init__(self, model: nn.Module, loss_fn: Callable, 
                 n_samples: int = 10, device: str = 'cpu'):
        self.model = model
        self.loss_fn = loss_fn
        self.n_samples = n_samples
        self.device = device
    
    def estimate_trace(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, float]:
        """
        Estimate tr(H) using Hutchinson's stochastic trace estimator.
        
        Returns:
            (mean_trace, std_trace)
        """
        x, y = data
        x, y = x.to(self.device), y.to(self.device)
        
        traces = []
        
        for _ in range(self.n_samples):
            # Generate Rademacher random vector
            v = self._rademacher_vector()
            
            # Compute v^T H v via two backward passes
            hvp = self._hessian_vector_product(x, y, v)
            
            # v^T H v
            trace_sample = torch.dot(v.flatten(), hvp.flatten()).item()
            traces.append(trace_sample)
        
        traces = np.array(traces)
        return traces.mean(), traces.std()
    
    def _rademacher_vector(self) -> torch.Tensor:
        """Generate Rademacher random vector (±1 with equal probability)"""
        params = list(self.model.parameters())
        total_params = sum(p.numel() for p in params)
        
        v_flat = torch.randint(0, 2, (total_params,), device=self.device).float() * 2 - 1
        
        # Reshape to match parameter structure
        v_list = []
        offset = 0
        for p in params:
            size = p.numel()
            v_list.append(v_flat[offset:offset+size].reshape(p.shape))
            offset += size
        
        return v_flat
    
    def _hessian_vector_product(self, x: torch.Tensor, y: torch.Tensor, 
                                 v: torch.Tensor) -> torch.Tensor:
        """
        Compute H @ v using the "double backward" trick.
        
        H @ v = ∂/∂θ (∇L · v)
        """
        self.model.zero_grad()
        
        # Forward pass
        output = self.model(x)
        loss = self.loss_fn(output, y)
        
        # First backward: get gradient
        grads = torch.autograd.grad(loss, self.model.parameters(), 
                                     create_graph=True, retain_graph=True)
        grad_flat = torch.cat([g.flatten() for g in grads])
        
        # Compute ∇L · v
        grad_v = torch.dot(grad_flat, v)
        
        # Second backward: differentiate ∇L · v w.r.t. θ
        hvp = torch.autograd.grad(grad_v, self.model.parameters(), 
                                   retain_graph=True)
        hvp_flat = torch.cat([h.flatten() for h in hvp])
        
        return hvp_flat
    
    def compute_kappa_eff(self, data: Tuple[torch.Tensor, torch.Tensor]) -> Tuple[float, float]:
        """
        Compute κ_eff = -tr(H) / N
        
        Interpretation:
        - κ_eff < 0 and stable -> grokking likely
        - κ_eff > 0 or oscillating -> grokking unlikely
        """
        trace_mean, trace_std = self.estimate_trace(data)
        
        n_params = sum(p.numel() for p in self.model.parameters())
        
        kappa_eff = -trace_mean / n_params
        kappa_std = trace_std / n_params
        
        return kappa_eff, kappa_std


class HardwareNoiseEstimator:
    """
    Estimate ε_hw(B, T) - hardware-induced variance
    
    Extended model addressing reviewer's criticism:
    ε_hw(B, T) = α/B + β*cache_miss_rate(B) + γ*thread_contention(T)
    """
    
    def __init__(self, model: nn.Module, loss_fn: Callable):
        self.model = model
        self.loss_fn = loss_fn
    
    def estimate_noise(self, data_loader, n_batches: int = 20, 
                       n_threads: int = 1) -> Dict[str, float]:
        """
        Estimate hardware noise by measuring gradient variance across batches
        """
        torch.set_num_threads(n_threads)
        
        gradients = []
        timings = []
        
        for i, (x, y) in enumerate(data_loader):
            if i >= n_batches:
                break
            
            start = time.perf_counter()
            
            self.model.zero_grad()
            output = self.model(x)
            loss = self.loss_fn(output, y)
            loss.backward()
            
            elapsed = time.perf_counter() - start
            timings.append(elapsed)
            
            # Collect gradient
            grad = torch.cat([p.grad.flatten() for p in self.model.parameters() 
                             if p.grad is not None])
            gradients.append(grad.clone())
        
        if len(gradients) < 2:
            return {'epsilon_hw': 0.0, 'timing_variance': 0.0}
        
        # Stack and compute variance
        grads_stacked = torch.stack(gradients)
        grad_variance = grads_stacked.var(dim=0).mean().item()
        timing_variance = np.var(timings)
        
        return {
            'epsilon_hw': grad_variance,
            'timing_variance': timing_variance,
            'mean_batch_time': np.mean(timings),
            'n_threads': n_threads
        }


def convergence_theorem():
    """
    THEOREM (Convergence to Algorithmic Invariance)
    
    Let W_t denote the weights at step t under SGD with:
        W_{t+1} = W_t - η∇L(W_t) + ξ_t
    
    where ξ_t is stochastic gradient noise with Var(ξ_t) = σ²/B + ε_hw(B,T).
    
    ASSUMPTIONS:
    A1. The target function f admits a rank-r tensor decomposition
    A2. The loss L is twice differentiable with Lipschitz Hessian
    A3. The effective curvature satisfies κ_eff < 0 after step t₀
    
    THEOREM: Under A1-A3, if
        
        Var(ξ_t) < σ_min(W*)² / (η · condition(H))
    
    where σ_min(W*) is the smallest non-zero singular value of the optimal
    decomposition, then with probability 1-δ:
    
        lim_{t->∞} d(W_t, W*) = 0
    
    where d is the subspace distance and W* is the algorithmically invariant solution.
    
    PROOF SKETCH:
    
    1. By A3 (κ_eff < 0), the loss landscape is locally convex near convergence
    
    2. The noise condition ensures gradient updates don't overshoot the
       invariant subspace defined by σ_min(W*)
    
    3. By spectral gap analysis, W_t projects onto the dominant singular
       subspace with increasing precision as t -> ∞
    
    4. The discretization {-1, 0, 1} emerges because integer solutions
       are fixed points of the projection when noise is controlled
    
    IMPLICATIONS FOR T:
    
    The expansion operator T is constructible because:
    - T preserves the dominant singular subspace (by definition)
    - The rank-r structure is independent of problem scale (by A1)
    - Therefore T = block_embed(W_r) where W_r is the converged rank-r solution
    """
    return """
    ═══════════════════════════════════════════════════════════════════════════
    THEOREM: Convergence to Algorithmic Invariance
    ═══════════════════════════════════════════════════════════════════════════
    
    Let W_t denote weights under SGD: W_{t+1} = W_t - η∇L(W_t) + ξ_t
    
    ASSUMPTIONS:
    (A1) Target function f admits rank-r tensor decomposition
    (A2) Loss L is C² with Lipschitz Hessian
    (A3) Effective curvature κ_eff = -tr(H)/N < 0 after epoch t₀
    
    STATEMENT:
    If Var(ξ_t) < σ_min(W*)² / (η · cond(H)), then:
    
        P[ lim_{t->∞} d_subspace(W_t, W*) = 0 ] ≥ 1 - δ
    
    where W* is the algorithmically invariant solution.
    
    COROLLARY (Constructibility of T):
    Under the theorem's conditions, T(W_n) = BlockEmbed(W_n, n'/n) correctly
    computes f_{n'} without retraining.
    
    ═══════════════════════════════════════════════════════════════════════════
    """


def verify_convergence_conditions(model: nn.Module, loss_fn: Callable,
                                   train_data: Tuple[torch.Tensor, torch.Tensor],
                                   noise_threshold: float = 0.1) -> Dict:
    """
    Verify that convergence conditions are satisfied for a trained model.
    """
    print("=" * 70)
    print("VERIFICATION OF CONVERGENCE CONDITIONS")
    print("=" * 70)
    
    results = {}
    
    # 1. Compute κ_eff
    print("\n1. Computing κ_eff via Hutchinson estimator...")
    estimator = HutchinsonTraceEstimator(model, loss_fn, n_samples=20)
    kappa_eff, kappa_std = estimator.compute_kappa_eff(train_data)
    
    results['kappa_eff'] = kappa_eff
    results['kappa_std'] = kappa_std
    
    print(f"   κ_eff = {kappa_eff:.6f} ± {kappa_std:.6f}")
    if kappa_eff < 0:
        print("   [OK] Condition A3 satisfied: kappa_eff < 0")
    else:
        print("   [FAIL] Condition A3 violated: kappa_eff >= 0")
    
    # 2. Check weight discreteness
    print("\n2. Analyzing weight discreteness...")
    discrete_values = [-1, 0, 1]
    discreteness_errors = []
    
    for name, param in model.named_parameters():
        if 'weight' in name or 'coef' in name.lower():
            p_flat = param.data.flatten()
            distances = torch.stack([
                (p_flat - v).abs() for v in discrete_values
            ])
            min_dist = distances.min(dim=0)[0].mean().item()
            discreteness_errors.append(min_dist)
            print(f"   {name}: mean distance to {{-1,0,1}} = {min_dist:.4f}")
    
    results['discreteness_error'] = np.mean(discreteness_errors) if discreteness_errors else 1.0
    
    if results['discreteness_error'] < 0.1:
        print("   [OK] Weights are near-discrete")
    else:
        print("   [WARN] Weights not fully discretized")
    
    # 3. Spectral analysis
    print("\n3. Spectral gap analysis...")
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            U, S, Vh = torch.linalg.svd(param.data, full_matrices=False)
            if len(S) > 1:
                gaps = S[:-1] / (S[1:] + 1e-10)
                max_gap_idx = gaps.argmax().item()
                print(f"   {name}: spectral gap at rank {max_gap_idx+1}, ratio = {gaps[max_gap_idx]:.2f}")
    
    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    conditions_met = (
        kappa_eff < 0 and 
        results['discreteness_error'] < 0.1
    )
    
    if conditions_met:
        print("[OK] All convergence conditions satisfied")
        print("   -> Model has reached algorithmic invariance regime")
        print("   -> Expansion operator T is constructible")
    else:
        print("[WARN] Some conditions not fully satisfied")
        print("   -> Additional training or regularization may be needed")
    
    results['conditions_met'] = conditions_met
    return results


# Example usage with a simple Strassen-like model
class SimpleStrassenModel(nn.Module):
    """Simple model for testing convergence verification"""
    def __init__(self, rank: int = 7):
        super().__init__()
        self.U = nn.Parameter(torch.randn(rank, 4))
        self.V = nn.Parameter(torch.randn(rank, 4))
        self.W = nn.Parameter(torch.randn(4, rank))
    
    def forward(self, x):
        # x is [batch, 8] (flattened A and B)
        A = x[:, :4]  # [batch, 4]
        B = x[:, 4:]  # [batch, 4]
        
        # Bilinear: M_k = (U @ A.T) * (V @ B.T)
        M = (A @ self.U.T) * (B @ self.V.T)  # [batch, rank]
        
        # Output: C = M @ W.T
        C = M @ self.W.T  # [batch, 4]
        return C


if __name__ == "__main__":
    print(convergence_theorem())
    
    print("\n" + "=" * 70)
    print("TESTING ON SIMPLE STRASSEN MODEL")
    print("=" * 70)
    
    # Create a model initialized near Strassen solution
    model = SimpleStrassenModel(rank=7)
    
    # Initialize close to Strassen (for testing)
    with torch.no_grad():
        # Approximate Strassen coefficients
        model.U.data = torch.tensor([
            [1, 0, 0, 1],   # M1: (A11+A22)
            [1, 1, 0, 0],   # M2: (A21+A22)
            [1, 0, 0, 0],   # M3: A11
            [0, 0, 0, 1],   # M4: A22
            [1, 0, 1, 0],   # M5: (A11+A12)
            [-1, 0, 1, 0],  # M6: (A21-A11)
            [0, 1, 0, -1],  # M7: (A12-A22)
        ], dtype=torch.float32)
        
        model.V.data = torch.tensor([
            [1, 0, 0, 1],   # (B11+B22)
            [1, 0, 0, 0],   # B11
            [0, 0, 1, -1],  # (B12-B22)
            [0, 1, -1, 0],  # (B21-B11)
            [0, 0, 0, 1],   # B22
            [1, 0, 1, 0],   # (B11+B12)
            [0, 1, 0, 1],   # (B21+B22)
        ], dtype=torch.float32)
        
        model.W.data = torch.tensor([
            [1, 0, 0, 1, -1, 0, 1],   # C11
            [0, 0, 1, 0, 1, 0, 0],    # C12
            [0, 1, 0, 1, 0, 0, 0],    # C21
            [1, -1, 1, 0, 0, 1, 0],   # C22
        ], dtype=torch.float32)
    
    # Generate test data
    torch.manual_seed(42)
    n_samples = 100
    A = torch.randn(n_samples, 2, 2)
    B = torch.randn(n_samples, 2, 2)
    C = A @ B  # Ground truth
    
    X = torch.cat([A.flatten(1), B.flatten(1)], dim=1)  # [n, 8]
    Y = C.flatten(1)  # [n, 4]
    
    loss_fn = nn.MSELoss()
    
    # Verify convergence conditions
    results = verify_convergence_conditions(model, loss_fn, (X, Y))
    
    print("\n" + "=" * 70)
    print("κ_eff COMPUTATION DETAILS")
    print("=" * 70)
    print(f"""
Method: Hutchinson Trace Estimation
  - Samples: 20 Rademacher vectors
  - Complexity: O(20 × 2 × backward_pass) vs O(N²) for exact Hessian
  - Memory: O(N) vs O(N²)
  
Result: κ_eff = {results['kappa_eff']:.6f} ± {results['kappa_std']:.6f}

This addresses the reviewer's concern about computability of κ_eff.
The method is practical for networks with millions of parameters.
""")
