#!/usr/bin/env python3
"""
Experiments for Paper Validation
=================================

This script runs the experiments described in the paper:
1. Phase Diagram: S(theta) success rate vs (Batch Size, Epochs)
2. Grokking Curves: Train/Test loss dynamics
3. Gradient Covariance: kappa(Sigma) vs Batch Size
4. Batch Size Effect: Statistical analysis across seeds

Metrics:
    S(theta) in {0,1}: Discretization success
    delta(theta): Distance to integers
    kappa(Sigma): Gradient covariance condition number

Author: grisun0
License: AGPL v3
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import warnings
from pathlib import Path
from datetime import datetime
import time

warnings.filterwarnings('ignore')


def setup_matplotlib():
    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Noto Sans CJK SC", "Arial"]
    plt.rcParams["axes.unicode_minus"] = False
    return plt, sns


plt, sns = setup_matplotlib()

torch.set_num_threads(4)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class StrassenOperator(nn.Module):
    
    def __init__(self, rank=8, symmetric_init=True):
        super().__init__()
        self.rank = rank
        
        if symmetric_init:
            base = torch.randn(rank, 4) * 0.5
            self.U = nn.Parameter(base.clone())
            self.V = nn.Parameter(base.clone())
        else:
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


def generate_batch(n, device=DEVICE):
    A = torch.randn(n, 2, 2, device=device)
    B = torch.randn(n, 2, 2, device=device)
    return A, B


def generate_test_set(n=1000, device=DEVICE):
    torch.manual_seed(9999)
    A = torch.randn(n, 2, 2, device=device)
    B = torch.randn(n, 2, 2, device=device)
    return A, B


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


def verify_strassen_structure(U_disc, V_disc, W_disc, tolerance=1e-5):
    n_test = 1000
    A = torch.randn(n_test, 2, 2)
    B = torch.randn(n_test, 2, 2)
    
    a = A.reshape(n_test, 4)
    b = B.reshape(n_test, 4)
    
    left = a @ U_disc.T
    right = b @ V_disc.T
    products = left * right
    c = products @ W_disc.T
    C_pred = c.reshape(n_test, 2, 2)
    C_true = torch.bmm(A, B)
    
    rel_error = (C_pred - C_true).abs().max() / (C_true.abs().max() + 1e-10)
    success = rel_error.item() < tolerance
    
    return success, rel_error.item()


def compute_S_theta(model):
    delta = compute_delta(model)
    
    if delta >= 0.1:
        return 0, delta, None
    
    with torch.no_grad():
        U_disc = torch.round(torch.clamp(model.U.data, -1, 1))
        V_disc = torch.round(torch.clamp(model.V.data, -1, 1))
        W_disc = torch.round(torch.clamp(model.W.data, -1, 1))
    
    success, rel_error = verify_strassen_structure(U_disc.cpu(), V_disc.cpu(), W_disc.cpu())
    
    return int(success), delta, rel_error


def compute_gradient_covariance(model, batch_size, n_samples=50):
    model.train()
    gradients = []
    
    for _ in range(n_samples):
        A, B = generate_batch(batch_size)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        loss = torch.mean((C_pred - C_true) ** 2)
        
        model.zero_grad()
        loss.backward()
        
        grad_vec = torch.cat([p.grad.view(-1) for p in model.parameters()])
        gradients.append(grad_vec.detach().cpu())
    
    G = torch.stack(gradients)
    G_centered = G - G.mean(dim=0, keepdim=True)
    cov = (G_centered.T @ G_centered) / (n_samples - 1)
    
    eigenvalues = torch.linalg.eigvalsh(cov).real
    threshold = 1e-10
    nonzero_eigs = eigenvalues[eigenvalues > threshold]
    
    if len(nonzero_eigs) < 2:
        return float('inf'), 1
    
    condition_number = (nonzero_eigs.max() / nonzero_eigs.min()).item()
    effective_rank = (nonzero_eigs > 0.01 * nonzero_eigs.max()).sum().item()
    
    return condition_number, effective_rank


def train_with_logging(batch_size, total_epochs, lr=0.02, wd=1e-4,
                       symmetric_init=True, seed=42, log_interval=50):
    torch.manual_seed(seed)
    
    model = StrassenOperator(rank=8, symmetric_init=symmetric_init).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=max(total_epochs//3, 100))
    
    A_test, B_test = generate_test_set(1000)
    C_test_true = torch.bmm(A_test, B_test)
    
    logs = []
    train_losses = []
    test_losses = []
    
    grokking_epoch = None
    grokking_detected = False
    
    for epoch in range(total_epochs):
        A, B = generate_batch(batch_size)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        
        train_loss = torch.mean((C_pred - C_true) ** 2)
        
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        
        train_losses.append(train_loss.item())
        
        with torch.no_grad():
            C_test_pred = model(A_test, B_test)
            test_loss = torch.mean((C_test_pred - C_test_true) ** 2).item()
            test_losses.append(test_loss)
        
        if not grokking_detected and epoch > 100:
            if train_losses[-1] < 1e-6:
                if len(test_losses) > 100:
                    recent_test = np.mean(test_losses[-10:])
                    earlier_test = np.mean(test_losses[-100:-50])
                    if earlier_test > 0.05 and recent_test < 0.01:
                        grokking_detected = True
                        grokking_epoch = epoch
        
        if epoch % log_interval == 0 or epoch == total_epochs - 1:
            delta = compute_delta(model)
            S, delta_check, rel_error = compute_S_theta(model)
            active = model.count_active()
            
            if epoch % (log_interval * 2) == 0:
                cond, eff_rank = compute_gradient_covariance(model, batch_size, n_samples=30)
            else:
                cond, eff_rank = None, None
            
            log_entry = {
                'epoch': epoch,
                'train_loss': train_losses[-1],
                'test_loss': test_loss,
                'delta': delta,
                'S': S,
                'rel_error': rel_error,
                'cond': cond,
                'eff_rank': eff_rank,
                'active_slots': active
            }
            logs.append(log_entry)
    
    final_result = sparsify_and_discretize(model, batch_size)
    
    return {
        'batch_size': batch_size,
        'epochs': total_epochs,
        'seed': seed,
        'logs': logs,
        'train_losses': train_losses,
        'test_losses': test_losses,
        'grokking_detected': grokking_detected,
        'grokking_epoch': grokking_epoch,
        'final_S': final_result['S'],
        'final_delta': final_result['delta'],
        'final_rel_error': final_result['rel_error']
    }


def sparsify_and_discretize(model, batch_size):
    with torch.no_grad():
        importance = model.slot_importance()
        _, sorted_idx = importance.sort()
        slots_to_prune = sorted_idx[:1].tolist()
    
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    
    for slot in slots_to_prune:
        for step in range(500):
            decay = 1.0 - (step / 500) * 0.99
            
            with torch.no_grad():
                model.U.data[slot] *= decay
                model.V.data[slot] *= decay
                model.W.data[:, slot] *= decay
            
            A, B = generate_batch(batch_size)
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
    
    for _ in range(500):
        A, B = generate_batch(batch_size)
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
    
    S, delta, rel_error = compute_S_theta(model)
    
    return {'S': S, 'delta': delta, 'rel_error': rel_error}


def run_phase_diagram():
    print("\n" + "="*70)
    print("EXPERIMENT 1: PHASE DIAGRAM")
    print("="*70)
    
    batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128, 256]
    epochs_list = [500, 1000, 2000, 5000]
    seeds = [42, 123, 456]
    
    results = np.zeros((len(epochs_list), len(batch_sizes)))
    delta_matrix = np.zeros((len(epochs_list), len(batch_sizes)))
    
    all_runs = []
    total = len(batch_sizes) * len(epochs_list)
    count = 0
    
    for i, epochs in enumerate(epochs_list):
        for j, batch_size in enumerate(batch_sizes):
            count += 1
            successes = 0
            deltas = []
            
            for seed in seeds:
                run = train_with_logging(
                    batch_size=batch_size,
                    total_epochs=epochs,
                    seed=seed,
                    log_interval=100
                )
                
                if run['final_S'] == 1:
                    successes += 1
                deltas.append(run['final_delta'])
                
                all_runs.append({
                    'batch_size': batch_size,
                    'epochs': epochs,
                    'seed': seed,
                    'S': run['final_S'],
                    'delta': run['final_delta'],
                    'grokking_detected': run['grokking_detected'],
                    'grokking_epoch': run['grokking_epoch']
                })
            
            success_rate = successes / len(seeds)
            mean_delta = np.mean(deltas)
            
            results[i, j] = success_rate
            delta_matrix[i, j] = mean_delta
            
            print(f"[{count}/{total}] B={batch_size:3d}, E={epochs:5d}: S={success_rate:.0%}, delta={mean_delta:.3f}")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    X, Y = np.meshgrid(batch_sizes, epochs_list)
    
    ax1 = axes[0]
    contour = ax1.contourf(X, Y, results, levels=[0, 0.33, 0.67, 1.0], cmap='RdYlGn', alpha=0.8)
    ax1.contour(X, Y, results, levels=[0.33, 0.67], colors='black', linewidths=1.5)
    ax1.axvline(x=24, color='blue', linestyle='--', linewidth=2, alpha=0.7, label='Optimal Range')
    ax1.axvline(x=128, color='blue', linestyle='--', linewidth=2, alpha=0.7)
    ax1.axhline(y=1000, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Min Epochs')
    
    cbar = plt.colorbar(contour, ax=ax1)
    cbar.set_label('S(theta) Success Rate')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Training Epochs')
    ax1.set_title('Phase Diagram: Discretization Success')
    ax1.set_xscale('log')
    ax1.legend(loc='upper right')
    
    ax2 = axes[1]
    im = ax2.imshow(delta_matrix, cmap='viridis_r', aspect='auto',
                    extent=[min(batch_sizes), max(batch_sizes), max(epochs_list), min(epochs_list)])
    CS = ax2.contour(X, Y, delta_matrix, levels=[0.1], colors='red', linewidths=2)
    ax2.clabel(CS, inline=True, fontsize=10, fmt='delta=0.1')
    
    cbar2 = plt.colorbar(im, ax=ax2)
    cbar2.set_label('Mean delta(theta)')
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Training Epochs')
    ax2.set_title('Discretization Quality')
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'fig_phase_diagram.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'fig_phase_diagram.png'}")
    
    with open('phase_diagram_results.json', 'w') as f:
        json.dump({
            'batch_sizes': batch_sizes,
            'epochs_list': epochs_list,
            'S_theta_rates': results.tolist(),
            'delta_matrix': delta_matrix.tolist(),
            'all_runs': all_runs
        }, f, indent=2)
    
    return results, delta_matrix, all_runs


def run_batch_size_effect():
    print("\n" + "="*70)
    print("EXPERIMENT 2: BATCH SIZE EFFECT")
    print("="*70)
    
    batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256]
    epochs = 2000
    seeds = [42, 123, 456, 789, 101]
    
    results = {bs: {'S': [], 'delta': [], 'grokking': []} for bs in batch_sizes}
    
    for batch_size in batch_sizes:
        print(f"\nBatch Size {batch_size}:")
        for seed in seeds:
            run = train_with_logging(
                batch_size=batch_size,
                total_epochs=epochs,
                seed=seed,
                log_interval=200
            )
            results[batch_size]['S'].append(run['final_S'])
            results[batch_size]['delta'].append(run['final_delta'])
            results[batch_size]['grokking'].append(run['grokking_detected'])
            print(f"  Seed {seed}: S={run['final_S']}, delta={run['final_delta']:.3f}")
    
    summary = {}
    for bs in batch_sizes:
        summary[bs] = {
            'success_rate': np.mean(results[bs]['S']),
            'success_std': np.std(results[bs]['S']),
            'delta_mean': np.mean(results[bs]['delta']),
            'delta_std': np.std(results[bs]['delta']),
            'grokking_rate': np.mean(results[bs]['grokking'])
        }
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1 = axes[0]
    bs_list = list(summary.keys())
    rates = [summary[bs]['success_rate'] for bs in bs_list]
    stds = [summary[bs]['success_std'] for bs in bs_list]
    
    colors = ['green' if 24 <= bs <= 128 else 'steelblue' for bs in bs_list]
    ax1.bar(range(len(bs_list)), rates, yerr=stds, capsize=3, color=colors, alpha=0.7)
    
    ax1.set_xticks(range(len(bs_list)))
    ax1.set_xticklabels([str(bs) for bs in bs_list], rotation=45)
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('S(theta) Success Rate')
    ax1.set_title('Discretization Success Rate by Batch Size')
    ax1.set_ylim(0, 1.1)
    ax1.axhline(y=0.68, color='red', linestyle='--', alpha=0.7, label='Paper target: 68%')
    ax1.legend()
    
    ax2 = axes[1]
    deltas = [summary[bs]['delta_mean'] for bs in bs_list]
    delta_stds = [summary[bs]['delta_std'] for bs in bs_list]
    
    ax2.errorbar(bs_list, deltas, yerr=delta_stds, fmt='o-', capsize=3,
                 color='darkorange', linewidth=2, markersize=6)
    ax2.axhline(y=0.1, color='red', linestyle='--', linewidth=2, label='Threshold delta=0.1')
    ax2.axvspan(24, 128, alpha=0.2, color='green', label='Optimal Range')
    
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Mean delta(theta)')
    ax2.set_title('Discretization Quality by Batch Size')
    ax2.set_xscale('log')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    output_dir = Path(__file__).parent.parent / "figures"
    output_dir.mkdir(exist_ok=True)
    plt.savefig(output_dir / 'fig_batch_size_effect.png', dpi=150, bbox_inches='tight')
    print(f"\nSaved: {output_dir / 'fig_batch_size_effect.png'}")
    
    print("\n" + "-"*70)
    print("BATCH SIZE EFFECT SUMMARY")
    print("-"*70)
    print(f"{'Batch':>8} {'S Rate':>12} {'Mean delta':>12} {'Grok Rate':>12}")
    print("-"*70)
    for bs in bs_list:
        s = summary[bs]
        marker = "*" if 24 <= bs <= 128 else " "
        print(f"{bs:>8}{marker} {s['success_rate']:>11.0%} {s['delta_mean']:>12.3f} {s['grokking_rate']:>11.0%}")
    
    with open('batch_size_effect_results.json', 'w') as f:
        json.dump({
            'batch_sizes': batch_sizes,
            'summary': {str(k): v for k, v in summary.items()},
            'detailed': {str(k): v for k, v in results.items()}
        }, f, indent=2)
    
    return summary


def main():
    start_time = time.time()
    
    print("=" * 70)
    print("RUNNING EXPERIMENTS FOR PAPER VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Device: {DEVICE}")
    print("=" * 70)
    
    print("\nKey metrics:")
    print("  S(theta) in {0,1}: Discretization success")
    print("  delta(theta): Distance to integers")
    print("  kappa(Sigma): Gradient covariance condition number")
    
    phase_results, delta_matrix, all_runs = run_phase_diagram()
    bs_summary = run_batch_size_effect()
    
    elapsed = time.time() - start_time
    
    print("\n" + "=" * 70)
    print("EXPERIMENTS COMPLETE")
    print(f"Total time: {elapsed/60:.1f} minutes")
    print("=" * 70)
    
    optimal_bs = [bs for bs, s in bs_summary.items() if 24 <= bs <= 128]
    if optimal_bs:
        opt_rates = [bs_summary[bs]['success_rate'] for bs in optimal_bs]
        print(f"\nOptimal batch range [24-128]: Mean S(theta) = {np.mean(opt_rates):.1%}")
    
    outside_bs = [bs for bs, s in bs_summary.items() if bs < 24 or bs > 128]
    if outside_bs:
        out_rates = [bs_summary[bs]['success_rate'] for bs in outside_bs]
        print(f"Outside optimal range: Mean S(theta) = {np.mean(out_rates):.1%}")


if __name__ == "__main__":
    main()
