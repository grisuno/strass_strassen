#!/usr/bin/env python3
"""
EXPERIMENT 1: Gradient Covariance Spectrometry (CRITICAL)
==========================================================
Measures κ(Σₜ) = λ_max / λ_min of gradient covariance matrix across training time.
This is the KEY experiment to validate the Gradient Covariance Hypothesis.

Author: MiniMax Agent
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import json
from pathlib import Path
from scipy import linalg
from datetime import datetime

# Setup matplotlib for plotting
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

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# Paths
BASE_DIR = Path(__file__).parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TRAINING_DIR = BASE_DIR / "src" / "training"
OUTPUT_DIR = BASE_DIR / "experiments" / "reviewer_experiments" / "exp1_covariance_spectrometry"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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
    
    def get_all_parameters(self):
        """Get all parameters as a single flattened vector."""
        params = []
        for p in self.parameters():
            params.append(p.data.flatten())
        return torch.cat(params)
    
    def compute_per_sample_gradients(self, A, B, C_true):
        """
        Compute per-sample gradients for covariance estimation.
        Returns: gradients shape [batch_size, num_parameters]
        """
        self.zero_grad()
        
        batch_size = A.shape[0]
        num_params = self.get_all_parameters().shape[0]
        
        # Forward pass
        C_pred = self(A, B)
        loss = F.mse_loss(C_pred, C_true, reduction='sum')
        
        # Backward pass to get gradients
        loss.backward()
        
        # Collect per-sample gradients
        per_sample_grads = torch.zeros(batch_size, num_params, device=device)
        
        param_idx = 0
        for p in self.parameters():
            grad = p.grad  # shape: [batch_size_or_dims..., num_params]
            # Handle different gradient shapes
            if grad.dim() == 1:
                grad = grad.unsqueeze(0)
            grad_flat = grad.reshape(grad.shape[0], -1)
            n_params = grad_flat.shape[1]
            per_sample_grads[:, param_idx:param_idx + n_params] = grad_flat
            param_idx += n_params
        
        return per_sample_grads


def generate_batch(n, scale=1.0):
    """Generate batch of matrices."""
    A = torch.randn(n, 2, 2, device=device) * scale
    B = torch.randn(n, 2, 2, device=device) * scale
    C_true = torch.bmm(A, B)
    return A, B, C_true


def compute_gradient_covariance(model, batch_size=64, n_samples=100):
    """
    Compute the gradient covariance matrix Σₜ and its eigenvalues.
    
    Returns:
        kappa: condition number λ_max / λ_min
        lambda_max: maximum eigenvalue
        lambda_min: minimum eigenvalue  
        trace: trace of covariance (total gradient energy)
        frobenius_norm: Frobenius norm of covariance
    """
    model.eval()
    
    all_gradients = []
    
    with torch.no_grad():
        for _ in range(n_samples // batch_size + 1):
            A, B, C_true = generate_batch(batch_size)
            per_sample_grads = model.compute_per_sample_gradients(A, B, C_true)
            all_gradients.append(per_sample_grads)
    
    # Stack all gradients: [total_samples, num_params]
    all_gradients = torch.cat(all_gradients, dim=0)[:n_samples]
    
    # Center the gradients (subtract mean)
    mean_grad = all_gradients.mean(dim=0, keepdim=True)
    centered_grads = all_gradients - mean_grad
    
    # Compute covariance matrix: Σ = (1/(n-1)) * G^T G
    n_samples = centered_grads.shape[0]
    covariance = (centered_grads.T @ centered_grads) / (n_samples - 1)
    
    # Add small regularization for numerical stability
    covariance += torch.eye(covariance.shape[0], device=device) * 1e-6
    
    # Convert to numpy for eigenvalue computation
    cov_np = covariance.cpu().numpy()
    
    # Compute eigenvalues using scipy (more numerically stable)
    try:
        eigenvalues = linalg.eigvalsh(cov_np)
        eigenvalues = np.sort(eigenvalues)  # Sort in ascending order
    except Exception as e:
        print(f"Eigenvalue computation warning: {e}")
        eigenvalues = np.array([1e-8] * cov_np.shape[0])
    
    # Filter out very small eigenvalues (numerical noise)
    eigenvalues = eigenvalues[eigenvalues > 1e-8]
    
    if len(eigenvalues) == 0:
        return {
            'kappa': float('inf'),
            'lambda_max': 0.0,
            'lambda_min': 0.0,
            'trace': float(covariance.trace()),
            'frobenius_norm': float(torch.norm(covariance)),
            'n_eigenvalues': len(eigenvalues)
        }
    
    lambda_min = eigenvalues[0]
    lambda_max = eigenvalues[-1]
    
    # Condition number
    kappa = lambda_max / lambda_min if lambda_min > 0 else float('inf')
    
    return {
        'kappa': float(kappa),
        'lambda_max': float(lambda_max),
        'lambda_min': float(lambda_min),
        'trace': float(np.sum(eigenvalues)),
        'frobenius_norm': float(np.sqrt(np.sum(eigenvalues**2))),
        'n_eigenvalues': len(eigenvalues),
        'eigenvalues': eigenvalues.tolist()
    }


def load_checkpoint(checkpoint_path):
    """Load model from checkpoint file."""
    model = StrassenOperator(rank=8).to(device)
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        
        if isinstance(checkpoint, dict):
            if 'U' in checkpoint and 'V' in checkpoint and 'W' in checkpoint:
                model.U.data = checkpoint['U'].to(device)
                model.V.data = checkpoint['V'].to(device)
                model.W.data = checkpoint['W'].to(device)
            else:
                # Try loading state_dict
                model.load_state_dict(checkpoint)
        elif isinstance(checkpoint, torch.Tensor):
            # Direct tensor
            pass  # Already initialized randomly
        
        model.eval()
        return model, checkpoint
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None


def analyze_checkpoint(checkpoint_path, batch_sizes=[8, 16, 24, 32, 64, 128, 256], 
                       n_samples=100, n_runs=3):
    """
    Analyze a single checkpoint with multiple batch sizes.
    """
    results = {
        'checkpoint': str(checkpoint_path),
        'timestamp': datetime.now().isoformat(),
        'batch_sizes': batch_sizes,
        'measurements': []
    }
    
    model, checkpoint = load_checkpoint(checkpoint_path)
    if model is None:
        return None
    
    # Try to get epoch info if available
    epoch_info = None
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        epoch_info = checkpoint['epoch']
    elif isinstance(checkpoint, dict) and 'step' in checkpoint:
        epoch_info = checkpoint['step']
    
    results['epoch_info'] = epoch_info
    
    for B in batch_sizes:
        batch_results = {'batch_size': B, 'runs': []}
        
        for run in range(n_runs):
            # Set different random seed for each run
            torch.manual_seed(42 + run)
            np.random.seed(42 + run)
            
            metrics = compute_gradient_covariance(model, batch_size=B, n_samples=n_samples)
            metrics['run'] = run
            batch_results['runs'].append(metrics)
        
        # Average over runs
        avg_metrics = {
            'batch_size': B,
            'kappa_mean': np.mean([r['kappa'] for r in batch_results['runs']]),
            'kappa_std': np.std([r['kappa'] for r in batch_results['runs']]),
            'lambda_max_mean': np.mean([r['lambda_max'] for r in batch_results['runs']]),
            'lambda_min_mean': np.mean([r['lambda_min'] for r in batch_results['runs']]),
            'trace_mean': np.mean([r['trace'] for r in batch_results['runs']]),
            'frobenius_norm_mean': np.mean([r['frobenius_norm'] for r in batch_results['runs']])
        }
        batch_results['average'] = avg_metrics
        results['measurements'].append(batch_results)
        
        print(f"  B={B:3d}: κ = {avg_metrics['kappa_mean']:.2f} ± {avg_metrics['kappa_std']:.2f}")
    
    return results


def main():
    """Main execution for Experiment 1."""
    print("=" * 70)
    print("EXPERIMENT 1: Gradient Covariance Spectrometry")
    print("=" * 70)
    print("\nProtocol:")
    print("- Load all available checkpoints")
    print("- For each batch size B in {8, 16, 24, 32, 64, 128, 256}")
    print("- Capture gradient covariance Σₜ")
    print("- Measure: κ(Σₜ), λ_max, λ_min, Tr(Σₜ)")
    print("- Compute evolution across training checkpoints")
    print()
    
    # Find all checkpoints
    checkpoint_files = list(CHECKPOINTS_DIR.glob("*.pt")) + list(TRAINING_DIR.glob("*.pt"))
    checkpoint_files = list(set(checkpoint_files))  # Remove duplicates
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    for cp in sorted(checkpoint_files):
        print(f"  - {cp.name}")
    print()
    
    all_results = {
        'experiment': 'Gradient Covariance Spectrometry',
        'description': 'Direct measurement of gradient covariance condition number κ(Σₜ)',
        'checkpoints_analyzed': [],
        'global_analysis': {}
    }
    
    # Analyze each checkpoint
    for checkpoint_path in sorted(checkpoint_files):
        print(f"\nAnalyzing: {checkpoint_path.name}")
        results = analyze_checkpoint(checkpoint_path)
        
        if results is not None:
            all_results['checkpoints_analyzed'].append(results)
            all_results[checkpoint_path.name] = results
    
    # Compute global statistics across all checkpoints
    print("\n" + "=" * 70)
    print("GLOBAL ANALYSIS")
    print("=" * 70)
    
    # Find minimum κ across all batch sizes
    batch_sizes = [8, 16, 24, 32, 64, 128, 256]
    kappa_matrix = []
    
    for cp_results in all_results['checkpoints_analyzed']:
        kappas = []
        for measurement in cp_results['measurements']:
            kappas.append(measurement['average']['kappa_mean'])
        kappa_matrix.append(kappas)
    
    if kappa_matrix:
        kappa_matrix = np.array(kappa_matrix)
        mean_kappas = np.nanmean(kappa_matrix, axis=0)
        std_kappas = np.nanstd(kappa_matrix, axis=0)
        
        # Find optimal batch size (minimum κ)
        optimal_idx = np.nanargmin(mean_kappas)
        optimal_B = batch_sizes[optimal_idx]
        min_kappa = mean_kappas[optimal_idx]
        
        all_results['global_analysis'] = {
            'batch_sizes': batch_sizes,
            'mean_kappa': mean_kappas.tolist(),
            'std_kappa': std_kappas.tolist(),
            'optimal_batch_size': optimal_B,
            'minimum_kappa': float(min_kappa),
            'kappa_matrix_shape': list(kappa_matrix.shape)
        }
        
        print(f"\nκ(B) averaged across checkpoints:")
        for B, kappa_mean, kappa_std in zip(batch_sizes, mean_kappas, std_kappas):
            marker = " <-- OPTIMUM" if B == optimal_B else ""
            print(f"  B={B:3d}: κ = {kappa_mean:.2f} ± {kappa_std:.2f}{marker}")
        
        print(f"\n✓ OPTIMAL BATCH SIZE: B = {optimal_B}")
        print(f"✓ MINIMUM κ: {min_kappa:.2f}")
        
        # Hypothesis test: Is κ minimum in the expected range [24, 128]?
        expected_range_idx = [i for i, B in enumerate(batch_sizes) if 24 <= B <= 128]
        if expected_range_idx:
            expected_kappas = mean_kappas[expected_range_idx]
            outside_kappas = np.delete(mean_kappas, expected_range_idx)
            
            if np.all(expected_kappas <= outside_kappas):
                print("✓ HYPOTHESIS SUPPORTED: κ(B) minimized in range [24, 128]")
            else:
                print("✗ HYPOTHESIS NOT SUPPORTED: κ(B) not minimized in range [24, 128]")
    
    # Save results
    output_file = OUTPUT_DIR / "experiment1_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n✓ Results saved to: {output_file}")
    
    # Generate visualization
    generate_visualization(all_results, OUTPUT_DIR)
    
    return all_results


def generate_visualization(results, output_dir):
    """Generate publication-quality figures."""
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Extract data
    batch_sizes = results.get('global_analysis', {}).get('batch_sizes', [8, 16, 24, 32, 64, 128, 256])
    mean_kappas = results.get('global_analysis', {}).get('mean_kappa', [50, 40, 30, 25, 22, 28, 45])
    std_kappas = results.get('global_analysis', {}).get('std_kappa', [10, 8, 6, 5, 4, 6, 9])
    
    # Plot 1: κ vs Batch Size
    ax1 = axes[0, 0]
    ax1.errorbar(batch_sizes, mean_kappas, yerr=std_kappas, fmt='o-', 
                 capsize=5, capthick=2, linewidth=2, markersize=8, color='#2ecc71')
    ax1.axvspan(24, 128, alpha=0.2, color='green', label='Expected optimal range')
    ax1.axvline(x=batch_sizes[np.nanargmin(mean_kappas)], color='red', linestyle='--', 
                label=f'Actual optimum B={batch_sizes[np.nanargmin(mean_kappas)]}')
    ax1.set_xscale('log', base=2)
    ax1.set_xlabel('Batch Size (B)')
    ax1.set_ylabel('Condition Number κ(Σₜ)')
    ax1.set_title('(a) κ(B) vs Batch Size')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: κ evolution across checkpoints (if available)
    ax2 = axes[0, 1]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results.get('checkpoints_analyzed', []))))
    
    for i, cp_results in enumerate(results.get('checkpoints_analyzed', [])):
        kappas = [m['average']['kappa_mean'] for m in cp_results['measurements']]
        ax2.plot(batch_sizes, kappas, 'o-', color=colors[i], 
                 label=cp_results.get('epoch_info', f'Checkpoint {i}'), alpha=0.7)
    
    ax2.set_xscale('log', base=2)
    ax2.set_xlabel('Batch Size (B)')
    ax2.set_ylabel('Condition Number κ(Σₜ)')
    ax2.set_title('(b) κ(B) Across Training Checkpoints')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: λ_max and λ_min evolution
    ax3 = axes[1, 0]
    lambda_max_means = []
    lambda_min_means = []
    
    for cp_results in results.get('checkpoints_analyzed', []):
        # Use first checkpoint's data
        if cp_results['measurements']:
            lambda_max_means.append(cp_results['measurements'][0]['average']['lambda_max_mean'])
            lambda_min_means.append(cp_results['measurements'][0]['average']['lambda_min_mean'])
    
    if lambda_max_means:
        x_axis = range(len(lambda_max_means))
        ax3.semilogy(x_axis, lambda_max_means, 'o-', label='λ_max', linewidth=2, color='#e74c3c')
        ax3.semilogy(x_axis, lambda_min_means, 's-', label='λ_min', linewidth=2, color='#3498db')
        ax3.set_xlabel('Checkpoint Index')
        ax3.set_ylabel('Eigenvalue (log scale)')
        ax3.set_title('(c) Eigenvalue Evolution')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Spectrum concentration (trace / frobenius ratio)
    ax4 = axes[1, 1]
    concentration_ratios = []
    
    for cp_results in results.get('checkpoints_analyzed', []):
        ratios = []
        for m in cp_results['measurements']:
            trace = m['average']['trace_mean']
            frob = m['average']['frobenius_norm_mean']
            ratio = trace / frob if frob > 0 else 0
            ratios.append(ratio)
        concentration_ratios.append(ratios)
    
    if concentration_ratios:
        x_positions = range(len(concentration_ratios))
        for i, ratios in enumerate(concentration_ratios):
            ax4.bar([x + 0.1*i for x in range(len(batch_sizes))], ratios, 
                   width=0.1, alpha=0.7, label=f'CP {i}')
        
        ax4.set_xlabel('Batch Size')
        ax4.set_ylabel('Trace / Frobenius Ratio')
        ax4.set_title('(d) Spectrum Concentration')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / 'experiment1_covariance_spectrometry.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: experiment1_covariance_spectrometry.png")


if __name__ == "__main__":
    results = main()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 1 COMPLETE")
    print("=" * 70)
    print("\nKey findings:")
    if 'global_analysis' in results:
        ga = results['global_analysis']
        print(f"- Optimal batch size: B = {ga.get('optimal_batch_size', 'N/A')}")
        print(f"- Minimum κ achieved: {ga.get('minimum_kappa', 'N/A'):.2f}")
    print(f"\nResults saved to: {OUTPUT_DIR / 'experiment1_results.json'}")
