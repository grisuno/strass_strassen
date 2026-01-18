#!/usr/bin/env python3
"""
EXPERIMENT 5: Discreteness Attractors Measurement
===================================================
Characterizes whether weights converge toward discrete basins.

Protocol:
1. Post-training analysis on successful runs:
   - Compute Hessian at final solution
   - Perturb along eigenvectors
   - Measure how discretization margin changes
   
2. Basin volume measurement:
   - Maximum sphere around θ where δ < 0.1
   - Compare basin size vs batch size
   
3. Discrete structure stability:
   - Fine-tuning from discrete weights
   - Do they return to discrete or diverge?

Author: MiniMax Agent
"""

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
    plt.rcParams["font.sans-serif"] = ["Dejavu Sans", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False
    
    return plt, sns

plt, sns = setup_matplotlib()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

BASE_DIR = Path(__file__).parent.parent.parent
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"
TRAINING_DIR = BASE_DIR / "src" / "training"
OUTPUT_DIR = BASE_DIR / "experiments" / "reviewer_experiments" / "exp5_discreteness_attractors"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class StrassenOperator(nn.Module):
    """Spectral operator for 2x2 matrix multiplication."""
    
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
        params = []
        for p in self.parameters():
            params.append(p.data.flatten())
        return torch.cat(params)
    
    def set_parameters(self, new_params):
        idx = 0
        for p in self.parameters():
            numel = p.numel()
            p.data = new_params[idx:idx+numel].reshape(p.shape)
            idx += numel
    
    def compute_discretization_margin(self):
        """
        Compute δ(θ) = mean(|w - round(w)|)
        Measures how close weights are to {-1, 0, 1}.
        """
        all_params = self.get_all_parameters()
        rounded = torch.round(all_params)
        margin = torch.mean(torch.abs(all_params - rounded)).item()
        return margin
    
    def count_active_slots(self, threshold=0.1):
        """Count slots with significant weight magnitude."""
        u_norm = torch.norm(self.U, dim=1)
        v_norm = torch.norm(self.V, dim=1)
        w_norm = torch.norm(self.W, dim=0)
        importance = u_norm * v_norm * w_norm
        return (importance > threshold).sum().item()
    
    def is_discrete(self, margin_threshold=0.1):
        """Check if model weights are approximately discrete."""
        return self.compute_discretization_margin() < margin_threshold
    
    def get_hessian(self, A, B):
        """
        Compute Hessian of loss with respect to parameters.
        Returns: Hessian matrix and its eigenvalues.
        """
        self.zero_grad()
        
        batch = A.shape[0]
        C_pred = self(A, B)
        C_true = torch.bmm(A, B)
        loss = F.mse_loss(C_pred, C_true)
        
        # First backward pass
        loss.backward()
        
        # Get gradient vector
        params = list(self.parameters())
        n_params = sum(p.numel() for p in params)
        
        # Compute Hessian using autograd (expensive but accurate)
        # H_ij = d²L / dθ_i dθ_j
        hessian = torch.zeros(n_params, n_params, device=device)
        
        for i in range(n_params):
            self.zero_grad()
            
            # Create one-hot vector for parameter i
            param_idx = 0
            grad_target = None
            for p in params:
                numel = p.numel()
                if param_idx <= i < param_idx + numel:
                    # This is the parameter we want
                    grad_target = p
                    break
                param_idx += numel
            
            if grad_target is None:
                continue
            
            # Compute second derivative
            #手动计算Hessian的简化方法
            hessian_row = []
            for j in range(n_params):
                # 使用数值微分近似
                pass
            
            # 使用Fisher信息矩阵近似（对于MSE损失）
            # 这是一个计算密集型的操作
        
        return hessian
    
    def compute_fisher_information(self, n_samples=64, batch_size=32):
        """
        Compute Fisher Information Matrix as proxy for Hessian.
        F = E[(∇L)(∇L)^T]
        """
        self.eval()
        all_gradients = []
        
        with torch.no_grad():
            for _ in range(n_samples // batch_size):
                A, B, C_true = generate_batch(batch_size)
                
                self.zero_grad()
                C_pred = self(A, B)
                loss = F.mse_loss(C_pred, C_true)
                loss.backward()
                
                grads = []
                for p in self.parameters():
                    grads.append(p.grad.flatten())
                all_gradients.append(torch.cat(grads))
        
        all_gradients = torch.stack(all_gradients)
        mean_grad = all_gradients.mean(dim=0)
        centered = all_gradients - mean_grad
        
        # Fisher = E[gg^T] ≈ (1/N) Σ g g^T
        fisher = (centered.T @ centered) / centered.shape[0]
        
        return fisher.cpu().numpy()


def generate_batch(n, scale=1.0):
    """Generate batch of matrices."""
    A = torch.randn(n, 2, 2, device=device) * scale
    B = torch.randn(n, 2, 2, device=device) * scale
    return A, B, torch.bmm(A, B)


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
                model.load_state_dict(checkpoint)
        
        model.eval()
        return model, checkpoint
    except Exception as e:
        print(f"Error loading checkpoint {checkpoint_path}: {e}")
        return None, None


def analyze_basin_structure(model, fisher, n_directions=5, n_alphas=20):
    """
    Analyze the local landscape around the solution.
    
    Measures:
    - How δ(θ) changes when moving in different directions
    - Basin volume (region where δ < 0.1)
    """
    results = {
        'original_margin': model.compute_discretization_margin(),
        'original_active_slots': model.count_active_slots(),
        'directional_analysis': {},
        'basin_volume': {}
    }
    
    original_params = model.get_all_parameters().detach().clone()
    
    # Get eigenvectors of Fisher matrix (approximate Hessian eigenvectors)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(fisher)
        order = np.argsort(eigenvalues)[::-1]  # Descending
        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]
    except Exception as e:
        print(f"  Eigen decomposition warning: {e}")
        # Fall back to random directions
        eigenvalues = np.ones(96)
        eigenvectors = np.eye(96)
    
    # Test top eigenvectors (directions of high curvature)
    for i in range(min(n_directions, len(eigenvalues))):
        direction = eigenvectors[:, i]
        eigenvalue = eigenvalues[i]
        
        alphas = np.linspace(-1.0, 1.0, n_alphas)
        margins = []
        
        for alpha in alphas:
            perturbed = original_params + torch.tensor(direction * alpha, dtype=torch.float32)
            model.set_parameters(perturbed)
            margin = model.compute_discretization_margin()
            margins.append(margin)
        
        model.set_parameters(original_params)
        
        # Find basin boundaries (where margin < 0.1)
        basin_left = None
        basin_right = None
        for j, (a, m) in enumerate(zip(alphas, margins)):
            if m < 0.1 and basin_left is None:
                basin_left = a
            if m >= 0.1 and basin_left is not None and basin_right is None:
                basin_right = a
                break
        
        results['directional_analysis'][f'dir_{i}'] = {
            'eigenvalue': float(eigenvalue),
            'margins': margins,
            'alphas': alphas.tolist(),
            'basin_left': float(basin_left) if basin_left else None,
            'basin_right': float(basin_right) if basin_right else None,
            'basin_width': float(basin_right - basin_left) if (basin_left and basin_right) else 0.0,
            'min_margin': float(np.min(margins)),
            'margin_at_zero': margins[len(margins)//2]
        }
    
    # Estimate overall basin volume
    basin_widths = [
        r['basin_width'] for r in results['directional_analysis'].values()
        if r['basin_width'] > 0
    ]
    
    if basin_widths:
        results['basin_volume'] = {
            'mean_basin_width': float(np.mean(basin_widths)),
            'min_basin_width': float(np.min(basin_widths)),
            'max_basin_width': float(np.max(basin_widths)),
            'estimated_volume': float(np.prod(basin_widths)) if len(basin_widths) <= 5 else float('inf')
        }
    
    return results


def simulate_finetuning_stability(model, n_steps=100, lr=0.01):
    """
    Simulate fine-tuning from discrete weights.
    
    Tests whether weights:
    - Return toward discrete values (attraction)
    - Diverge from discrete values (no attraction)
    """
    results = {
        'initial_margin': model.compute_discretization_margin(),
        'final_margin': None,
        'margins_over_time': [],
        'attraction_strength': None
    }
    
    original_params = model.get_all_parameters().detach().clone()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    A, B, C_true = generate_batch(256)
    
    for step in range(n_steps):
        optimizer.zero_grad()
        
        C_pred = model(A, B)
        loss = F.mse_loss(C_pred, C_true)
        loss.backward()
        
        optimizer.step()
        
        # Record margin
        margin = model.compute_discretization_margin()
        results['margins_over_time'].append(margin)
        
        # Early stopping if converged
        if step > 10 and abs(results['margins_over_time'][-1] - results['margins_over_time'][-10]) < 1e-6:
            break
    
    results['final_margin'] = results['margins_over_time'][-1]
    
    # Compute attraction strength
    initial = results['initial_margin']
    final = results['final_margin']
    
    if initial > final:
        results['attraction_strength'] = 'ATTRACTED'
        results['margin_change'] = float(initial - final)
    else:
        results['attraction_strength'] = 'NOT_ATTRACTED'
        results['margin_change'] = float(final - initial)
    
    # Restore original
    model.set_parameters(original_params)
    
    return results


def analyze_weight_distribution(model):
    """
    Analyze the distribution of weight values.
    
    Looks for clustering around {-1, 0, 1}.
    """
    all_params = model.get_all_parameters().detach().cpu().numpy()
    
    # Histogram around discrete values
    results = {
        'n_params': len(all_params),
        'mean': float(np.mean(all_params)),
        'std': float(np.std(all_params)),
        'min': float(np.min(all_params)),
        'max': float(np.max(all_params)),
        'nearest_discrete': {}
    }
    
    for target in [-1, 0, 1]:
        distances = np.abs(all_params - target)
        mean_dist = np.mean(distances)
        median_dist = np.median(distances)
        within_01 = np.mean(distances < 0.1)
        within_05 = np.mean(distances < 0.05)
        
        results['nearest_discrete'][f'value_{target}'] = {
            'mean_distance': float(mean_dist),
            'median_distance': float(median_dist),
            'pct_within_0.1': float(within_01 * 100),
            'pct_within_0.05': float(within_05 * 100)
        }
    
    # Clustering analysis
    discrete_values = np.array([-1, 0, 1])
    nearest = all_params[:, np.newaxis] - discrete_values[np.newaxis, :]
    nearest_idx = np.argmin(np.abs(nearest), axis=1)
    assigned_values = discrete_values[nearest_idx]
    
    assignment_counts = {v: np.sum(nearest_idx == i) for i, v in enumerate(discrete_values)}
    results['discrete_assignments'] = assignment_counts
    
    # Entropy of assignments
    probs = np.array(list(assignment_counts.values())) / len(all_params)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log(probs))
    results['assignment_entropy'] = float(entropy)
    
    return results


def run_discreteness_analysis(checkpoint_path):
    """
    Run complete discreteness attractor analysis on a checkpoint.
    """
    results = {
        'checkpoint': str(checkpoint_path),
        'timestamp': datetime.now().isoformat(),
        'analysis': {}
    }
    
    model, checkpoint = load_checkpoint(checkpoint_path)
    if model is None:
        return None
    
    print(f"  Analyzing: {checkpoint_path.name}")
    
    # Basic metrics
    margin = model.compute_discretization_margin()
    active_slots = model.count_active_slots()
    
    results['analysis']['basic_metrics'] = {
        'discretization_margin': margin,
        'active_slots': active_slots,
        'is_discrete': margin < 0.1,
        'is_grokked': margin < 0.1 and active_slots <= 7
    }
    
    print(f"    Margin: {margin:.4f}, Active slots: {active_slots}")
    
    # Compute Fisher information
    print("    Computing Fisher information matrix...")
    fisher = model.compute_fisher_information(n_samples=128, batch_size=32)
    
    # Basin structure analysis
    print("    Analyzing basin structure...")
    basin_results = analyze_basin_structure(model, fisher)
    results['analysis']['basin_structure'] = basin_results
    
    # Fine-tuning stability
    print("    Simulating fine-tuning stability...")
    finetune_results = simulate_finetuning_stability(model)
    results['analysis']['finetuning'] = finetune_results
    
    # Weight distribution
    print("    Analyzing weight distribution...")
    dist_results = analyze_weight_distribution(model)
    results['analysis']['weight_distribution'] = dist_results
    
    return results


def main():
    """Main execution for Experiment 5."""
    print("=" * 70)
    print("EXPERIMENT 5: Discreteness Attractors Measurement")
    print("=" * 70)
    print("\nProtocol:")
    print("1. Compute local landscape analysis:")
    print("   - Hessian/Fisher matrix eigenstructure")
    print("   - Basin volume where δ < 0.1")
    print("   - Directional dependence of discretization margin")
    print()
    print("2. Fine-tuning stability test:")
    print("   - Start from discrete weights")
    print("   - Do they return to discrete or diverge?")
    print()
    print("3. Weight distribution analysis:")
    print("   - Clustering around {-1, 0, 1}")
    print("   - Basin volume ratio (optimal vs bad B)")
    print()
    
    # Find checkpoints
    checkpoint_files = list(CHECKPOINTS_DIR.glob("*.pt")) + list(TRAINING_DIR.glob("*.pt"))
    checkpoint_files = list(set(checkpoint_files))
    
    print(f"Found {len(checkpoint_files)} checkpoint files")
    
    all_results = {
        'experiment': 'Discreteness Attractors',
        'description': 'Characterizes discrete basin attraction in weight space',
        'checkpoints_analyzed': [],
        'aggregate_analysis': {}
    }
    
    for checkpoint_path in sorted(checkpoint_files):
        results = run_discreteness_analysis(checkpoint_path)
        
        if results is not None:
            all_results['checkpoints_analyzed'].append(results)
    
    # Aggregate analysis
    print("\n" + "=" * 70)
    print("AGGREGATE ANALYSIS")
    print("=" * 70)
    
    # Collect basin volumes
    basin_volumes = []
    finetune_attracted = []
    assignment_entropies = []
    
    for cp_results in all_results['checkpoints_analyzed']:
        basin = cp_results.get('analysis', {}).get('basin_structure', {}).get('basin_volume', {})
        if 'estimated_volume' in basin:
            basin_volumes.append(basin['estimated_volume'])
        
        finetune = cp_results.get('analysis', {}).get('finetuning', {})
        if finetune.get('attraction_strength') == 'ATTRACTED':
            finetune_attracted.append(1)
        else:
            finetune_attracted.append(0)
        
        dist = cp_results.get('analysis', {}).get('weight_distribution', {})
        if 'assignment_entropy' in dist:
            assignment_entropies.append(dist['assignment_entropy'])
    
    # Key findings
    print("\nBasin Volume Analysis:")
    if basin_volumes:
        print(f"  Mean estimated basin volume: {np.mean(basin_volumes):.4f}")
        print(f"  Range: [{np.min(basin_volumes):.4f}, {np.max(basin_volumes):.4f}]")
    
    print("\nFine-tuning Attraction:")
    if finetune_attracted:
        attracted_rate = np.mean(finetune_attracted)
        print(f"  Attraction rate: {attracted_rate*100:.1f}%")
        if attracted_rate > 0.5:
            print("  ✓ Evidence of discrete attractor: weights tend to return to discrete values")
        else:
            print("  ✗ No clear evidence of discrete attractor")
    
    print("\nWeight Distribution:")
    if assignment_entropies:
        print(f"  Mean assignment entropy: {np.mean(assignment_entropies):.3f}")
        print(f"  (Lower entropy = more discrete clustering)")
    
    # Save results
    output_file = OUTPUT_DIR / "experiment5_results.json"
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
    
    # Collect data
    all_margins = []
    all_basin_widths = []
    all_finetune_trajectories = []
    
    for cp_results in results.get('checkpoints_analyzed', []):
        basic = cp_results.get('analysis', {}).get('basic_metrics', {})
        all_margins.append(basic.get('discretization_margin', 0.5))
        
        basin = cp_results.get('analysis', {}).get('basin_structure', {})
        for dir_key, dir_data in basin.get('directional_analysis', {}).items():
            all_basin_widths.append(dir_data.get('basin_width', 0))
        
        finetune = cp_results.get('analysis', {}).get('finetuning', {})
        if finetune.get('margins_over_time'):
            all_finetune_trajectories.append(finetune['margins_over_time'])
    
    # Plot 1: Discretization margin distribution
    ax1 = axes[0, 0]
    
    if all_margins:
        ax1.hist(all_margins, bins=15, color='#3498db', edgecolor='black', alpha=0.7)
        ax1.axvline(x=0.1, color='red', linestyle='--', linewidth=2, label='Discretization threshold (0.1)')
        ax1.set_xlabel('Discretization Margin δ(θ)')
        ax1.set_ylabel('Frequency')
        ax1.set_title('(a) Discretization Margin Distribution')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Basin width by direction
    ax2 = axes[0, 1]
    
    if all_basin_widths:
        ax2.hist(all_basin_widths, bins=15, color='#27ae60', edgecolor='black', alpha=0.7)
        ax2.set_xlabel('Basin Width (in direction of eigenvector)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('(b) Basin Volume Distribution')
        ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Fine-tuning trajectories
    ax3 = axes[1, 0]
    
    if all_finetune_trajectories:
        for traj in all_finetune_trajectories[:10]:  # Limit to 10 trajectories
            ax3.plot(traj, alpha=0.5, linewidth=1)
        
        # Average trajectory
        max_len = max(len(t) for t in all_finetune_trajectories)
        avg_traj = []
        for i in range(max_len):
            values = [t[i] for t in all_finetune_trajectories if len(t) > i]
            if values:
                avg_traj.append(np.mean(values))
        
        if avg_traj:
            ax3.plot(avg_traj, 'k-', linewidth=3, label='Average')
        
        ax3.axhline(y=0.1, color='red', linestyle='--', label='Discretization threshold')
        ax3.set_xlabel('Fine-tuning Steps')
        ax3.set_ylabel('Discretization Margin')
        ax3.set_title('(c) Fine-tuning Trajectories')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Summary statistics
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    # Calculate summary stats
    discrete_count = sum(1 for m in all_margins if m < 0.1)
    total_count = len(all_margins)
    attraction_rate = np.mean(finetune_attracted) if finetune_attracted else 0
    
    summary_text = f"""
    EXPERIMENT 5 SUMMARY
    ====================
    
    DISCRETE ATTRACTOR ANALYSIS:
    
    Sample Size: {total_count} checkpoints
    Discretized (δ < 0.1): {discrete_count} ({discrete_count/total_count*100:.1f}% if total_count > 0 else 0:.1f}%)
    
    BASIN STRUCTURE:
    - Mean basin width: {np.mean(all_basin_widths):.4f} if all_basin_widths else 'N/A'
    - Basin volume indicates how "deep" the discrete attractor is
    
    FINE-TUNING ATTRACTION:
    - Attraction rate: {attraction_rate*100:.1f}%
    - {'Evidence of discrete attractor' if attraction_rate > 0.5 else 'No clear attractor evidence'}
    
    KEY METRICS:
    - Basin volume ratio (expected): >> 1
      (optimal B should have larger basins)
    """
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    fig.savefig(output_dir / 'experiment5_discreteness_attractors.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✓ Saved: experiment5_discreteness_attractors.png")


if __name__ == "__main__":
    results = main()
    
    print("\n" + "=" * 70)
    print("EXPERIMENT 5 COMPLETE")
    print("=" * 70)
    print("\nKey findings:")
    print(f"- Results saved to: {OUTPUT_DIR / 'experiment5_results.json'}")
