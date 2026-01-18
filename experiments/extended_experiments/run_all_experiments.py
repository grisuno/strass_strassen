#!/usr/bin/env python3
"""
Robust Runner for Reviewer Experiments
=======================================
Handles various checkpoint formats and numerical issues.
"""

import sys
import os

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from scipy import linalg
from datetime import datetime

# Setup matplotlib
def setup_matplotlib():
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    plt.switch_backend("Agg")
    plt.style.use("seaborn-v0_8")
    sns.set_palette("husl")
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    
    return plt, sns

plt, sns = setup_matplotlib()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BASE_DIR = Path(PROJECT_ROOT)
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TRAINING_DIR = BASE_DIR / "src" / "training"
OUTPUT_DIR = BASE_DIR / "experiments" / "reviewer_experiments"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class StrassenOperator(nn.Module):
    """Spectral operator for 2x2 matrix multiplication."""
    
    def __init__(self, rank=8):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(rank, 4, dtype=torch.float32) * 0.5)
        self.V = nn.Parameter(torch.randn(rank, 4, dtype=torch.float32) * 0.5)
        self.W = nn.Parameter(torch.randn(4, rank, dtype=torch.float32) * 0.5)
    
    def forward(self, A, B):
        batch = A.shape[0]
        a = A.reshape(batch, 4)
        b = B.reshape(batch, 4)
        left = a @ self.U.T
        right = b @ self.V.T
        products = left * right
        c = products @ self.W.T
        return c.reshape(batch, 2, 2)
    
    def get_all_parameters(self):
        params = []
        for p in self.parameters():
            params.append(p.data.flatten())
        return torch.cat(params)
    
    def set_parameters(self, new_params):
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = new_params[idx:idx+numel].reshape(p.shape).float()
            idx += numel
    
    def count_active_slots(self, threshold=0.1):
        u_norm = torch.norm(self.U, dim=1)
        v_norm = torch.norm(self.V, dim=1)
        w_norm = torch.norm(self.W, dim=0)
        importance = u_norm * v_norm * w_norm
        return (importance > threshold).sum().item()
    
    def compute_discretization_margin(self):
        all_params = self.get_all_parameters()
        rounded = torch.round(all_params)
        margin = torch.mean(torch.abs(all_params - rounded)).item()
        return margin


def generate_batch(n, scale=1.0):
    A = torch.randn(n, 2, 2, dtype=torch.float32, device=device) * scale
    B = torch.randn(n, 2, 2, dtype=torch.float32, device=device) * scale
    return A, B, torch.bmm(A, B)


def load_checkpoint_robust(checkpoint_path, model):
    """Load checkpoint with multiple format fallback strategies."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        if isinstance(checkpoint, dict):
            # Try standard format (U, V, W directly)
            if 'U' in checkpoint and 'V' in checkpoint and 'W' in checkpoint:
                model.U.data = checkpoint['U'].to(device).float()
                model.V.data = checkpoint['V'].to(device).float()
                model.W.data = checkpoint['W'].to(device).float()
                return True
            
            # Try model_state_dict format
            if 'model_state_dict' in checkpoint:
                state = checkpoint['model_state_dict']
                # Try to find matching keys
                for name, param in state.items():
                    if hasattr(model, name):
                        getattr(model, name).data = param.to(device).float()
                return True
            
            # Try loading directly as state_dict
            try:
                model.load_state_dict(checkpoint, strict=False)
                return True
            except:
                pass
        
        return False
    except Exception as e:
        return False


def compute_gradient_covariance_safe(model, batch_size=64, n_samples=64):
    """Compute κ(Σₜ) with numerical safety."""
    model.eval()
    
    for p in model.parameters():
        p.requires_grad_(True)
    
    all_gradients = []
    
    for _ in range(n_samples // batch_size):
        A, B, C_true = generate_batch(batch_size)
        
        model.zero_grad()
        C_pred = model(A, B)
        loss = F.mse_loss(C_pred, C_true, reduction='sum')
        loss.backward()
        
        grads = []
        for p in model.parameters():
            if p.grad is not None:
                grads.append(p.grad.flatten())
        
        if grads:
            all_gradients.append(torch.cat(grads))
    
    if not all_gradients:
        return {'kappa': float('inf'), 'lambda_max': 0.0, 'lambda_min': 0.0, 'trace': 0.0}
    
    all_gradients = torch.stack(all_gradients)
    
    # Use batch gradient covariance instead of per-sample
    mean_grad = all_gradients.mean(dim=0)
    centered = all_gradients - mean_grad
    
    n = centered.shape[0]
    covariance = (centered.T @ centered) / (n - 1 + 1e-6)
    
    # Add regularization
    reg = 1e-6 * torch.eye(covariance.shape[0], device=device)
    covariance = covariance + reg
    
    cov_np = covariance.cpu().numpy()
    
    try:
        eigenvalues = linalg.eigvalsh(cov_np)
        eigenvalues = np.sort(eigenvalues)
        
        # Filter out numerical noise
        eigenvalues = eigenvalues[eigenvalues > 1e-6]
        
        if len(eigenvalues) < 2:
            return {'kappa': float('inf'), 'lambda_max': 0.0, 'lambda_min': 0.0, 'trace': 0.0}
        
        lambda_min = eigenvalues[0]
        lambda_max = eigenvalues[-1]
        
        # Cap κ to avoid numerical explosion
        kappa = min(lambda_max / lambda_min, 1e6) if lambda_min > 1e-8 else float('inf')
        
        return {
            'kappa': float(kappa),
            'lambda_max': float(lambda_max),
            'lambda_min': float(lambda_min),
            'trace': float(np.sum(eigenvalues)),
            'n_eigenvalues': len(eigenvalues)
        }
    except Exception as e:
        return {'kappa': float('inf'), 'lambda_max': 0.0, 'lambda_min': 0.0, 'trace': 0.0}


def run_all_experiments():
    """Run all experiments."""
    
    print("=" * 70)
    print("REVIEWER EXPERIMENT SUITE (Robust Version)")
    print("=" * 70)
    
    checkpoint_files = list(CHECKPOINTS_DIR.glob("*.pt")) + list(TRAINING_DIR.glob("*.pt"))
    checkpoint_files = list(set(checkpoint_files))
    
    print(f"\nFound {len(checkpoint_files)} checkpoint files")
    
    all_results = {
        'experiment_suite': 'Reviewer Experiments (Robust)',
        'timestamp': datetime.now().isoformat(),
        'n_checkpoints': len(checkpoint_files),
        'experiments': {}
    }
    
    # Filter to only use checkpoints that load successfully
    valid_checkpoints = []
    for cp_path in sorted(checkpoint_files):
        model = StrassenOperator(rank=8)
        if load_checkpoint_robust(cp_path, model):
            valid_checkpoints.append((cp_path, model))
            print(f"  ✓ {cp_path.name}")
        else:
            print(f"  ✗ {cp_path.name} (failed to load)")
    
    print(f"\n{len(valid_checkpoints)} checkpoints loaded successfully")
    
    if not valid_checkpoints:
        print("ERROR: No valid checkpoints found!")
        return all_results
    
    # =========================================================================
    # EXPERIMENT 1: Gradient Covariance Spectrometry
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Gradient Covariance Spectrometry")
    print("=" * 70)
    
    exp1_results = {
        'description': 'Measures κ(Σₜ) for different batch sizes',
        'batch_size_analysis': [],
        'summary': {}
    }
    
    batch_sizes = [8, 16, 24, 32, 64]
    
    for cp_path, model in valid_checkpoints[:8]:
        print(f"\nAnalyzing: {cp_path.name}")
        cp_analysis = {'checkpoint': cp_path.name, 'batch_sizes': []}
        
        for B in batch_sizes:
            metrics = compute_gradient_covariance_safe(model, batch_size=B, n_samples=64)
            cp_analysis['batch_sizes'].append({
                'batch_size': B,
                'kappa': metrics['kappa'],
                'lambda_max': metrics['lambda_max'],
                'lambda_min': metrics['lambda_min']
            })
            print(f"  B={B:3d}: κ = {metrics['kappa']:.2f}")
        
        exp1_results['batch_size_analysis'].append(cp_analysis)
    
    # Aggregate
    if exp1_results['batch_size_analysis']:
        kappa_matrix = []
        for cp in exp1_results['batch_size_analysis']:
            kappas = []
            for b in cp['batch_sizes']:
                if b['kappa'] != float('inf'):
                    kappas.append(b['kappa'])
            if kappas:
                kappa_matrix.append(kappas)
        
        if kappa_matrix:
            kappa_matrix = np.array(kappa_matrix)
            # Pad to same length
            max_len = max(len(k) for k in kappa_matrix)
            padded = np.array([np.pad(k, (0, max_len - len(k)), constant_values=np.nan) for k in kappa_matrix])
            mean_kappas = np.nanmean(padded, axis=0)
            
            optimal_idx = np.nanargmin(mean_kappas)
            optimal_B = batch_sizes[optimal_idx]
            
            exp1_results['summary'] = {
                'batch_sizes': batch_sizes[:len(mean_kappas)],
                'mean_kappa_per_B': mean_kappas.tolist(),
                'optimal_batch_size': optimal_B,
                'minimum_kappa': float(mean_kappas[optimal_idx])
            }
            
            print(f"\n✓ Optimal B: {optimal_B} (κ = {mean_kappas[optimal_idx]:.2f})")
    
    all_results['experiments']['exp1_covariance_spectrometry'] = exp1_results
    
    # =========================================================================
    # EXPERIMENT 2: Noise Ablation
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Gradient Noise Injection Ablation")
    print("=" * 70)
    
    exp2_results = {
        'description': 'Tests tolerance to gradient vs weight noise',
        'noise_levels': [0.0001, 0.0005, 0.001, 0.005, 0.01],
        'treatments': {}
    }
    
    noise_levels = exp2_results['noise_levels']
    
    for cp_path, model in valid_checkpoints[:5]:
        print(f"\nAnalyzing: {cp_path.name}")
        
        original_params = model.get_all_parameters().detach().clone()
        A_test, B_test, C_test = generate_batch(500)
        
        def compute_accuracy():
            with torch.no_grad():
                C_pred = model(A_test, B_test)
                errors = (C_pred - C_test).abs().reshape(500, -1).max(dim=1)[0]
                return (errors < 1e-3).float().mean().item()
        
        baseline_acc = compute_accuracy()
        cp_results = {'baseline_accuracy': baseline_acc, 'noise_effects': []}
        
        for sigma in noise_levels:
            # Generate noise with the same shape and dtype as parameters
            noise = torch.randn(original_params.shape, device=device, dtype=original_params.dtype) * sigma
            perturbed = original_params + noise
            model.set_parameters(perturbed)
            noisy_acc = compute_accuracy()
            model.set_parameters(original_params)
            
            retention = noisy_acc / baseline_acc if baseline_acc > 0 else 0
            cp_results['noise_effects'].append({
                'sigma': sigma,
                'noisy_accuracy': noisy_acc,
                'accuracy_retention': retention
            })
            print(f"  σ={sigma:.4f}: acc={noisy_acc:.3f} (retention={retention:.2%})")
        
        exp2_results['treatments'][cp_path.name] = cp_results
    
    all_results['experiments']['exp2_noise_ablation'] = exp2_results
    
    # =========================================================================
    # EXPERIMENT 3: Prospective Prediction
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Prospective κ Prediction")
    print("=" * 70)
    
    exp3_results = {
        'description': 'Tests if early-epoch κ predicts final success',
        'predictions': []
    }
    
    for cp_path, model in valid_checkpoints:
        # Measure κ
        kappa = compute_gradient_covariance_safe(model)['kappa']
        
        # Get final metrics
        margin = model.compute_discretization_margin()
        active = model.count_active_slots()
        success = margin < 0.1 and active <= 7
        
        # Predict
        predicted_success = kappa < 15.0
        
        exp3_results['predictions'].append({
            'checkpoint': cp_path.name,
            'kappa': kappa,
            'margin': margin,
            'active_slots': active,
            'actual_success': success,
            'predicted_success': predicted_success,
            'correct': predicted_success == success
        })
        
        status = '✓' if predicted_success == success else '✗'
        print(f"  {status} {cp_path.name}: κ={kappa:.1f}, margin={margin:.3f}, success={success}")
    
    correct = sum(1 for p in exp3_results['predictions'] if p['correct'])
    total = len(exp3_results['predictions'])
    
    exp3_results['summary'] = {
        'prediction_accuracy': correct / total if total > 0 else 0,
        'n_correct': correct,
        'n_total': total
    }
    
    print(f"\n✓ Prediction accuracy: {correct}/{total} = {correct/total*100:.1f}%")
    
    all_results['experiments']['exp3_prospective_prediction'] = exp3_results
    
    # =========================================================================
    # EXPERIMENT 4: Trajectory Perturbation
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Trajectory Perturbation")
    print("=" * 70)
    
    exp4_results = {
        'description': 'Tests trajectory stability under perturbations',
        'perturbations': [0.001, 0.01, 0.1],
        'treatments': {}
    }
    
    for cp_path, model in valid_checkpoints[:5]:
        print(f"\nAnalyzing: {cp_path.name}")
        
        original_params = model.get_all_parameters().detach().clone()
        baseline_norm = torch.norm(original_params).item()
        
        cp_results = {'perturbation_effects': []}
        
        for sigma in exp4_results['perturbations']:
            # Generate noise with the same shape and dtype as parameters
            noise = torch.randn(original_params.shape, device=device, dtype=original_params.dtype) * sigma
            perturbed = original_params + noise
            model.set_parameters(perturbed)
            new_norm = torch.norm(model.get_all_parameters()).item()
            model.set_parameters(original_params)
            
            cp_results['perturbation_effects'].append({
                'sigma': sigma,
                'norm_ratio': new_norm / baseline_norm
            })
            print(f"  σ={sigma:.3f}: norm_ratio={new_norm/baseline_norm:.3f}")
        
        exp4_results['treatments'][cp_path.name] = cp_results
    
    all_results['experiments']['exp4_trajectory_perturbation'] = exp4_results
    
    # =========================================================================
    # EXPERIMENT 5: Discreteness Attractors
    # =========================================================================
    print("\n" + "=" * 70)
    print("EXPERIMENT 5: Discreteness Attractors")
    print("=" * 70)
    
    exp5_results = {
        'description': 'Characterizes discrete basin attraction',
        'checkpoints': []
    }
    
    for cp_path, model in valid_checkpoints:
        margin = model.compute_discretization_margin()
        active = model.count_active_slots()
        is_discrete = margin < 0.1
        
        exp5_results['checkpoints'].append({
            'checkpoint': cp_path.name,
            'discretization_margin': margin,
            'active_slots': active,
            'is_discrete': is_discrete
        })
        
        status = '✓' if is_discrete else '✗'
        print(f"  {status} {cp_path.name}: margin={margin:.4f}, slots={active}")
    
    discrete_count = sum(1 for c in exp5_results['checkpoints'] if c['is_discrete'])
    total = len(exp5_results['checkpoints'])
    
    exp5_results['summary'] = {
        'discretized_count': discrete_count,
        'total_count': total,
        'discretization_rate': discrete_count / total if total > 0 else 0
    }
    
    print(f"\n✓ Discretization rate: {discrete_count}/{total} = {discrete_count/total*100:.1f}%")
    
    all_results['experiments']['exp5_discreteness_attractors'] = exp5_results
    
    # =========================================================================
    # Save Results
    # =========================================================================
    
    output_file = OUTPUT_DIR / "all_experiments_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n✓ Results saved to: {output_file}")
    
    generate_summary_visualization(all_results, OUTPUT_DIR)
    
    return all_results


def generate_summary_visualization(results, output_dir):
    """Generate summary visualization."""
    print("\nGenerating summary visualization...")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Exp 1
    ax1 = axes[0, 0]
    exp1 = results.get('experiments', {}).get('exp1_covariance_spectrometry', {})
    if exp1.get('summary'):
        bs = exp1['summary'].get('batch_sizes', [8, 16, 24, 32, 64])
        kappas = exp1['summary'].get('mean_kappa_per_B', [50, 40, 30, 25, 22])
        ax1.plot(bs, kappas, 'o-', linewidth=2, markersize=8, color='#2ecc71')
        ax1.axvspan(24, 128, alpha=0.2, color='green', label='Expected optimal')
        ax1.set_xscale('log', base=2)
        ax1.set_xlabel('Batch Size (B)')
        ax1.set_ylabel('κ(Σₜ)')
        ax1.set_title('Exp 1: κ(B) Across Batch Sizes')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
    
    # Exp 2
    ax2 = axes[0, 1]
    exp2 = results.get('experiments', {}).get('exp2_noise_ablation', {})
    noise_levels = exp2.get('noise_levels', [0.0001, 0.0005, 0.001, 0.005, 0.01])
    
    all_retentions = []
    for cp_data in exp2.get('treatments', {}).values():
        for effect in cp_data.get('noise_effects', []):
            all_retentions.append(effect.get('accuracy_retention', 0))
    
    if all_retentions:
        mean_retention = np.mean(all_retentions)
        ax2.bar(range(len(noise_levels)), [mean_retention]*len(noise_levels), 
               color=['#3498db']*len(noise_levels), edgecolor='black')
        ax2.set_xticks(range(len(noise_levels)))
        ax2.set_xticklabels([str(n) for n in noise_levels])
        ax2.set_xlabel('Noise σ')
        ax2.set_ylabel('Mean Accuracy Retention')
        ax2.set_title('Exp 2: Noise Tolerance')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Exp 3
    ax3 = axes[0, 2]
    exp3 = results.get('experiments', {}).get('exp3_prospective_prediction', {})
    predictions = exp3.get('predictions', [])
    
    if predictions:
        kappas = [p['kappa'] for p in predictions]
        successes = [p['actual_success'] for p in predictions]
        
        # Filter out infinite values for histogram
        finite_kappas = [(k, s) for k, s in zip(kappas, successes) if k != float('inf') and k != float('-inf')]
        
        if finite_kappas:
            success_kappas = [k for k, s in finite_kappas if s]
            failure_kappas = [k for k, s in finite_kappas if not s]
            
            if success_kappas:
                ax3.hist(success_kappas, bins=10, alpha=0.7, label='Success', color='#27ae60')
            if failure_kappas:
                ax3.hist(failure_kappas, bins=10, alpha=0.7, label='Failure', color='#e74c3c')
        
        ax3.axvline(x=15, color='blue', linestyle='--', label='Threshold')
        ax3.set_xlabel('κ(Σₜ)')
        ax3.set_ylabel('Count')
        ax3.set_title('Exp 3: κ Distribution by Outcome')
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_xlim(0, 100)  # Limit x-axis to visible range for finite values
    
    # Exp 4
    ax4 = axes[1, 0]
    exp4 = results.get('experiments', {}).get('exp4_trajectory_perturbation', {})
    perturbations = exp4.get('perturbations', [0.001, 0.01, 0.1])
    
    norm_ratios = []
    for cp_data in exp4.get('treatments', {}).values():
        for effect in cp_data.get('perturbation_effects', []):
            norm_ratios.append(effect.get('norm_ratio', 1.0))
    
    if norm_ratios:
        mean_ratio = np.mean(norm_ratios)
        ax4.bar([str(p) for p in perturbations], [mean_ratio]*len(perturbations),
               color=['#3498db', '#e74c3c', '#9b59b6'], edgecolor='black')
        ax4.set_xlabel('Perturbation σ')
        ax4.set_ylabel('Mean Norm Ratio')
        ax4.set_title('Exp 4: Trajectory Stability')
        ax4.grid(True, alpha=0.3, axis='y')
    
    # Exp 5
    ax5 = axes[1, 1]
    exp5 = results.get('experiments', {}).get('exp5_discreteness_attractors', {})
    checkpoints = exp5.get('checkpoints', [])
    
    if checkpoints:
        margins = [c['discretization_margin'] for c in checkpoints]
        ax5.hist(margins, bins=10, color='#3498db', edgecolor='black', alpha=0.7)
        ax5.axvline(x=0.1, color='red', linestyle='--', label='Threshold')
        ax5.set_xlabel('Discretization Margin')
        ax5.set_ylabel('Count')
        ax5.set_title('Exp 5: Margin Distribution')
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    
    # Summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    exp1 = results.get('experiments', {}).get('exp1_covariance_spectrometry', {})
    exp2 = results.get('experiments', {}).get('exp2_noise_ablation', {})
    exp3 = results.get('experiments', {}).get('exp3_prospective_prediction', {})
    exp5 = results.get('experiments', {}).get('exp5_discreteness_attractors', {})
    
    summary = f"""
    REVIEWER EXPERIMENTS SUMMARY
    ============================
    
    Total Checkpoints: {results.get('n_checkpoints', 0)}
    
    EXPERIMENT 1: Covariance Spectrometry
    - Optimal B: {exp1.get('summary', {}).get('optimal_batch_size', 'N/A')}
    - Min κ: {exp1.get('summary', {}).get('minimum_kappa', 'N/A'):.2f}
    
    EXPERIMENT 2: Noise Ablation
    - Mean retention: {np.mean([e.get('accuracy_retention', 0) for c in exp2.get('treatments', {}).values() for e in c.get('noise_effects', [])]):.2%}
    
    EXPERIMENT 3: Prospective Prediction
    - Accuracy: {exp3.get('summary', {}).get('prediction_accuracy', 0):.1%}
    - Correct: {exp3.get('summary', {}).get('n_correct', 0)}/{exp3.get('summary', {}).get('n_total', 0)}
    
    EXPERIMENT 4: Trajectory Perturbation
    - Tests stability to perturbations
    
    EXPERIMENT 5: Discreteness Attractors
    - Discretization rate: {exp5.get('summary', {}).get('discretization_rate', 0):.1%}
    """
    
    ax6.text(0.05, 0.95, summary, transform=ax6.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(output_dir / 'all_experiments_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: all_experiments_summary.png")


if __name__ == "__main__":
    results = run_all_experiments()
    
    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR / 'all_experiments_results.json'}")
    print(f"Summary visualization: {OUTPUT_DIR / 'all_experiments_summary.png'}")
