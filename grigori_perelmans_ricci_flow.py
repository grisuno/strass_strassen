#!/usr/bin/env python3
"""
StrassenRicciFlow - Geometric Dynamics and Singularity Analysis

Analyzes neural network checkpoints using Ricci Flow theory.
Calculates curvature (Hessian), identifies singularities ("necks"), and
derives thermodynamic quantities (Heat Kernel, Entropy) from the spectral geometry.

Architecture: SOLID principles, Tensor-based Ricci analysis, Surgical pruning heuristics
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
import os
import argparse
from typing import Dict, Optional, List, Tuple, Any, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path


# ============================================================================
# CONFIGURATION (Single Source of Truth)
# ============================================================================

@dataclass(frozen=True)
class RicciConfig:
    """Immutable configuration for Ricci Flow analysis."""

    # Model Architecture
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    INPUT_DIM: int = 4
    OUTPUT_DIM: int = 4

    # Analysis Parameters
    HESSIAN_BATCH_SIZE: int = 1  # Compute Hessian on single sample for precision
    SPECTRAL_REGULARIZATION: float = 1e-6  # Epsilon for eigenvalue stability

    # Singularity Detection Thresholds
    SINGULARITY_EIGENVALUE_RATIO: float = 100.0  # Max/Min ratio to define a "neck"
    NECK_CURVATURE_THRESHOLD: float = 10.0  # Absolute curvature threshold

    # Heat Kernel / Thermodynamics Parameters
    HEAT_KERNEL_TIMES: List[float] = field(default_factory=lambda: [0.1, 1.0, 10.0])
    TEMPERATURE_RANGE: List[float] = field(default_factory=lambda: [0.01, 0.1, 1.0])

    # Physical Constants for Planck Estimation from Geometry
    PLANCK_SI: float = 1.054571817e-34
    
    # Output
    OUTPUT_DIRECTORY: str = "ricci_analysis_reports"
    CHECKPOINT_DIRECTORY: str = "checkpoints"


CONFIG = RicciConfig()


# ============================================================================
# UTILITIES
# ============================================================================

def set_random_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================

class BilinearStrassenModel(nn.Module):
    """
    Bilinear model f(A,B) = W((U*A) ⊙ (V*B)).
    """
    def __init__(self, config: RicciConfig = CONFIG):
        super().__init__()
        self.config = config
        self.U = nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM, bias=False)
        self.V = nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM, bias=False)
        self.W = nn.Linear(config.HIDDEN_DIM, config.OUTPUT_DIM, bias=False)
        self._initialize()

    def _initialize(self):
        nn.init.xavier_uniform_(self.U.weight)
        self.V.weight.data = self.U.weight.data.clone()
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.W(self.U(a) * self.V(b))

    def get_coefficients(self) -> Dict[str, torch.Tensor]:
        return {
            'U': self.U.weight.data,
            'V': self.V.weight.data,
            'W': self.W.weight.data
        }
    
    def get_flat_params(self) -> torch.Tensor:
        """Return all parameters as a single flattened vector."""
        params = torch.cat([p.flatten() for p in self.parameters()])
        return params

    def set_flat_params(self, flat_params: torch.Tensor) -> None:
        """Set model parameters from a flattened vector."""
        idx = 0
        with torch.no_grad():
            for p in self.parameters():
                numel = p.numel()
                p.copy_(flat_params[idx:idx+numel].reshape(p.shape))
                idx += numel


# ============================================================================
# DATA GENERATION
# ============================================================================

class StrassenDataGenerator:
    @staticmethod
    def generate_batch(batch_size: int, config: RicciConfig = CONFIG):
        A = torch.randn(batch_size, config.MATRIX_SIZE, config.MATRIX_SIZE)
        B = torch.randn(batch_size, config.MATRIX_SIZE, config.MATRIX_SIZE)
        C = torch.bmm(A, B)
        return (
            A.reshape(batch_size, config.INPUT_DIM),
            B.reshape(batch_size, config.INPUT_DIM),
            C.reshape(batch_size, config.OUTPUT_DIM)
        )


# ============================================================================
# CHECKPOINT MIGRATION (Reused from Planck Script)
# ============================================================================

class CheckpointMigrator(ABC):
    @abstractmethod
    def can_migrate(self, state_dict: Dict[str, Any]) -> bool: pass
    @abstractmethod
    def migrate(self, state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]: pass

class CustomFormatMigrator(CheckpointMigrator):
    def can_migrate(self, state_dict: Dict[str, Any]) -> bool:
        return 'U' in state_dict and isinstance(state_dict['U'], torch.Tensor)
    def migrate(self, state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        u_tensor = state_dict['U']
        if u_tensor.shape == (7, 4):
            u_padded = torch.zeros(8, 4)
            v_padded = torch.zeros(8, 4)
            w_padded = torch.zeros(4, 8)
            u_padded[:7] = state_dict['U']
            v_padded[:7] = state_dict['V']
            w_padded[:, :7] = state_dict['W']
            return {'U.weight': u_padded, 'V.weight': v_padded, 'W.weight': w_padded}
        return {'U.weight': u_tensor, 'V.weight': state_dict['V'], 'W.weight': state_dict['W']}

class StandardFormatMigrator(CheckpointMigrator):
    def can_migrate(self, state_dict: Dict[str, Any]) -> bool:
        return any(k.endswith('.weight') for k in state_dict.keys())
    def migrate(self, state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        return {k: state_dict[k] for k in ['U.weight', 'V.weight', 'W.weight'] if k in state_dict}

class CheckpointMigrationManager:
    def __init__(self):
        self.strategies = [CustomFormatMigrator(), StandardFormatMigrator()]

    def migrate_checkpoint(self, path: str, device: str = 'cpu') -> Optional[Dict[str, torch.Tensor]]:
        try:
            data = torch.load(path, map_location=device, weights_only=False)
            state_dict = data.get('state_dict', data.get('model_state_dict', data))
            for strategy in self.strategies:
                if strategy.can_migrate(state_dict):
                    return strategy.migrate(state_dict)
            return None
        except Exception as e:
            print(f"Migration error: {e}")
            return None


# ============================================================================
# RICCI FLOW & CURVATURE ANALYSIS
# ============================================================================

class RicciFlowAnalyzer:
    """
    Calculates Ricci curvature metrics using Hessian as Metric Tensor proxy.
    In Perelman's flow, dg/dt = -2Ric. 
    Here we analyze instantaneous state of Metric (Hessian).
    """
    def __init__(self, model: nn.Module, config: RicciConfig = CONFIG):
        self.model = model
        self.config = config

    def compute_hessian(self, input_a: torch.Tensor, input_b: torch.Tensor, target_c: torch.Tensor) -> torch.Tensor:
        """
        Computes exact Hessian of loss w.r.t parameters.
        H = d^2L / dtheta^2
        """
        params = list(self.model.parameters())
        loss_fn = lambda theta: self._loss_wrapper(theta, params, input_a, input_b, target_c)
        
        # Flatten params for scalar function input
        flat_params = torch.cat([p.flatten() for p in params])
        
        try:
            # Compute Hessian (matrix of second derivatives)
            hessian = torch.autograd.functional.hessian(loss_fn, flat_params)
            return hessian
        except RuntimeError as e:
            print(f"Warning: Hessian computation failed due to memory/graph constraints: {e}")
            # Fallback: Diagonal approximation (Gauss-Newton / Empirical Fisher)
            return self._compute_diagonal_hessian(input_a, input_b, target_c)

    def _loss_wrapper(self, flat_params: torch.Tensor, original_params: List[torch.Tensor], 
                      a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Wrapper to compute loss from flat param vector."""
        # Reshape flat params back to model
        idx = 0
        with torch.no_grad():
            for p in original_params:
                numel = p.numel()
                # Create new tensor with requires_grad for graph reconstruction if needed
                # Note: functional.hessian handles graph, but we need to ensure shapes match
                p_data = flat_params[idx:idx+numel].reshape(p.shape)
                p.copy_(p_data) 
                idx += numel
        
        pred = self.model(a, b)
        return F.mse_loss(pred, c)

    def _compute_diagonal_hessian(self, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """Approximation: Diagonal of Hessian (Gauss-Newton)."""
        grads = torch.autograd.grad(F.mse_loss(self.model(a, b), c), self.model.parameters(), create_graph=True)
        grad_vec = torch.cat([g.flatten() for g in grads])
        diag_hessian = torch.autograd.grad(grad_vec, self.model.parameters(), retain_graph=True)
        return torch.diag(torch.cat([g.flatten() for g in diag_hessian]))

    def analyze_curvature(self, hessian: torch.Tensor) -> Dict[str, Any]:
        """
        Analyze Hessian spectrum to derive Ricci Scalar and Topological invariants.
        """
        # Regularize for stability
        hessian = hessian + self.config.SPECTRAL_REGULARIZATION * torch.eye(hessian.shape[0])
        
        # Compute Eigenvalues (Principal Curvatures)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        
        # Ricci Scalar (Scalar Curvature) R = Trace(Ric) ~ Trace(H)
        ricci_scalar = torch.sum(eigenvalues).item()
        
        # Spectral Gap (Difference between two largest/smallest)
        sorted_eig = torch.sort(eigenvalues, descending=True).values
        spectral_gap = (sorted_eig[0] - sorted_eig[-1]).item()
        
        # Condition Number (Geometric Stiffness)
        # Avoid division by zero
        min_eig = sorted_eig[-1].item()
        max_eig = sorted_eig[0].item()
        condition_number = float('inf') if abs(min_eig) < 1e-9 else max_eig / abs(min_eig)
        
        # Count Negative Curvatures (Saddle points)
        negative_curvature_count = (eigenvalues < 0).sum().item()
        
        # IMPORTANTE: Devolvemos el Tensor para cálculos posteriores (Heat Kernel)
        # No convertimos a numpy todavía.
        return {
            'ricci_scalar': float(ricci_scalar),
            'spectral_gap': float(spectral_gap),
            'condition_number_hessian': float(condition_number),
            'max_curvature': float(max_eig),
            'min_curvature': float(min_eig),
            'negative_curvature_count': negative_curvature_count,
            'eigenvalues': eigenvalues 
        }

    def compute_heat_kernel_trace(self, eigenvalues: torch.Tensor, t: float = 1.0) -> float:
        """
        Trace of Heat Kernel: Z(t) = Sum( exp(-lambda_i * t) ).
        Relates to Partition Function in Quantum Mechanics.
        """
        # Filter tiny positive values to avoid exp overflow or log(0)
        # Nota: Esto requiere que eigenvalues sea un Tensor de PyTorch
        safe_eig = torch.clamp(eigenvalues, min=0) 
        trace = torch.sum(torch.exp(-safe_eig * t)).item()
        return trace

    def compute_topological_entropy(self, eigenvalues: torch.Tensor) -> float:
        """
        von Neumann Entropy / Spectral Entropy.
        S = - Sum( p_i * log(p_i) ) where p_i are normalized eigenvalue weights.
        """
        safe_eig = torch.abs(eigenvalues) + 1e-12
        probs = safe_eig / torch.sum(safe_eig)
        entropy = -torch.sum(probs * torch.log(probs)).item()
        return entropy

# ============================================================================
# SINGULARITY SURGERY ENGINE
# ============================================================================

class SingularityEngine:
    """
    Identifies 'necks' (singularities) in the geometry and proposes 'surgery' (pruning).
    A 'neck' is a parameter direction with extreme curvature (Hessian eigenvalue).
    """

    def __init__(self, model: nn.Module, eigenvalues: torch.Tensor, config: RicciConfig = CONFIG):
        self.model = model
        self.eigenvalues = eigenvalues
        self.config = config
        
        # Map eigenvectors to parameters (requires computing eigenvectors too)
        # For efficiency, we infer singularities from magnitude of weights vs gradient contribution
        # A true geometric surgery would require eigenvectors.
        
    def detect_necks(self, curvature_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Identify if the system is in a 'bottleneck' state."""
        condition = curvature_analysis['condition_number_hessian']
        is_singular = condition > self.config.SINGULARITY_EIGENVALUE_RATIO
        
        return {
            'is_singular': is_singular,
            'condition_number': condition,
            'neck_severity': np.log10(condition) if is_singular else 0.0
        }

    def propose_surgery(self) -> List[str]:
        """
        Propose parameters to 'cut' (prune) based on curvature heuristics.
        In Strassen, the 'bias' slot (8th) often carries the 'noise' or singular connection.
        """
        # Simple heuristic: Find parameters with smallest magnitude that might be 'noise'
        # In a full Ricci surgery, we would look at eigenvectors aligned with the 8th slot.
        proposals = []
        
        params = list(self.model.named_parameters())
        for name, p in params:
            if 'bias' in name: continue # No bias in this model
            
            # Heuristic: If a row in U/V/W has small norm, it's a candidate for pruning
            if len(p.shape) == 2:
                norms = torch.norm(p, dim=1 if 'W' in name else 0)
                # Find indices with small norm (relative)
                threshold = torch.quantile(norms, 0.1)
                weak_indices = torch.where(norms < threshold)[0]
                
                for idx in weak_indices:
                    proposals.append(f"{name}[{idx}]")
                    
        return proposals


# ============================================================================
# PLANCK ESTIMATION FROM GEOMETRY
# ============================================================================

class GeometricPlanckCalculator:
    """
    Estimates effective Planck constant from Spectral Geometry.
    Uses the Hessian eigenvalues to define an energy spectrum.
    """
    def __init__(self, eigenvalues: torch.Tensor, ricci_scalar: float, config: RicciConfig = CONFIG):
        self.eigenvalues = eigenvalues
        self.ricci_scalar = ricci_scalar
        self.config = config

    def calculate(self) -> Dict[str, Any]:
        # 1. Spectral Uncertainty
        # Delta_E ~ Energy Gap (Spectral Gap)
        gap = self._get_spectral_gap()
        
        # 2. Ricci Uncertainty (Space-time curvature volume)
        # Delta_x ~ 1 / sqrt(Ricci) - Inverse of curvature
        curvature_scale = abs(self.ricci_scalar) + 1e-10
        delta_x = 1.0 / np.sqrt(curvature_scale)
        
        # 3. Planck Relation: h_bar ~ Delta_E * Delta_t or Delta_p * Delta_x
        # If we relate Ricci Scalar (R) to Energy density, then h_bar ~ sqrt(R) / R
        # Simplified geometric h_bar:
        h_bar_geo = gap * delta_x

        # 4. Thermodynamic h_bar (from Partition Function)
        # Z ~ exp(-E/T). Phase transitions occur where dZ/dT is max.
        # We use the spectral entropy.
        entropy = self._compute_spectral_entropy()
        
        # Effective Planck as entropy scaling factor
        # In many quantum systems, S ~ log(States) ~ log(Volume/h^N)
        h_bar_thermo = np.exp(entropy / self.config.INPUT_DIM) # Scaling by DOF

        # Unified value (geometric dominance)
        h_bar_final = h_bar_geo * 0.8 + h_bar_thermo * 0.2
        
        return {
            'h_bar_geometric': float(h_bar_geo),
            'h_bar_thermodynamic': float(h_bar_thermo),
            'h_bar_unified': float(h_bar_final),
            'spectral_gap': float(gap),
            'curvature_scale': float(curvature_scale),
            'spectral_entropy': float(entropy)
        }

    def _get_spectral_gap(self) -> float:
        sorted_eig = torch.sort(self.eigenvalues).values
        # Gap between positive eigenvalues (relevant for stability)
        pos_eig = sorted_eig[sorted_eig > 1e-5]
        if len(pos_eig) < 2:
            return 1.0
        return (pos_eig[0] - pos_eig[1]).item()

    def _compute_spectral_entropy(self) -> float:
        safe_eig = torch.abs(self.eigenvalues) + 1e-12
        probs = safe_eig / torch.sum(safe_eig)
        return -torch.sum(probs * torch.log(probs)).item()


# ============================================================================
# MAIN PIPELINE
# ============================================================================

# ============================================================================
# MAIN PIPELINE
# ============================================================================

class RicciFlowAnalyzerPipeline:
    """
    Complete analysis pipeline for Ricci Flow and Planck Estimation.
    Orchestrates Hessian computation, Curvature Analysis, and Physics metrics.
    """
    def __init__(self, config: RicciConfig = CONFIG):
        self.config = config
        self.migrator = CheckpointMigrationManager()

    def analyze_checkpoint(self, checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
        """
        Perform complete analysis of a single checkpoint.
        """
        print(f"Analyzing: {os.path.basename(checkpoint_path)}")

        # 1. Load Model
        model = BilinearStrassenModel(self.config).to(device)
        state_dict = self.migrator.migrate_checkpoint(checkpoint_path, device)
        
        if state_dict is None:
            raise ValueError(f"Could not load checkpoint: {checkpoint_path}")
        
        model.load_state_dict(state_dict)
        model.eval()

        # 2. Generate Input
        A, B, C = StrassenDataGenerator.generate_batch(self.config.HESSIAN_BATCH_SIZE, self.config)
        A, B, C = A.to(device), B.to(device), C.to(device)

        # 3. Compute Hessian (Metric Tensor)
        print("  Computing Hessian (Metric)...")
        ricci_analyzer = RicciFlowAnalyzer(model, self.config)
        hessian = ricci_analyzer.compute_hessian(A, B, C)

        # 4. Analyze Curvature (Ricci Scalar, Spectrum)
        print("  Analyzing Curvature Spectrum...")
        curvature_metrics = ricci_analyzer.analyze_curvature(hessian)

        # 5. Heat Kernel Analysis (Thermodynamics)
        print("  Computing Heat Kernel Trace...")
        heat_traces = {}
        for t in self.config.HEAT_KERNEL_TIMES:
            trace = ricci_analyzer.compute_heat_kernel_trace(curvature_metrics['eigenvalues'], t)
            heat_traces[f't_{t}'] = trace

        # 6. Topological Entropy
        topo_entropy = ricci_analyzer.compute_topological_entropy(curvature_metrics['eigenvalues'])

        # 7. Singularity Detection (Necks)
        print("  Detecting Singularities...")
        surgeon = SingularityEngine(model, curvature_metrics['eigenvalues'], self.config)
        necks = surgeon.detect_necks(curvature_metrics)
        surgery_proposals = surgeon.propose_surgery()

        # 8. Planck Estimation from Geometry
        print("  Calculating Planck from Geometry...")
        planck_calc = GeometricPlanckCalculator(
            curvature_metrics['eigenvalues'], 
            curvature_metrics['ricci_scalar'],
            self.config
        )
        planck_results = planck_calc.calculate()

        # 9. Compile Report
        # Preparamos los eigenvalues para serializar JSON (convertimos Tensor a lista)
        curvature_metrics_json = curvature_metrics.copy()
        if torch.is_tensor(curvature_metrics_json.get('eigenvalues')):
            curvature_metrics_json['eigenvalues'] = curvature_metrics_json['eigenvalues'].tolist()
        elif isinstance(curvature_metrics_json.get('eigenvalues'), np.ndarray):
            curvature_metrics_json['eigenvalues'] = curvature_metrics_json['eigenvalues'].tolist()

        report = {
            'metadata': {
                'checkpoint': checkpoint_path,
                'checkpoint_name': os.path.basename(checkpoint_path),
                'timestamp': datetime.now().isoformat(),
                'device': device,
                'config': {
                    'hidden_slots': self.config.HIDDEN_DIM,
                    'matrix_size': self.config.MATRIX_SIZE
                }
            },
            'ricci_flow': {
                'metric_type': 'Hessian (Loss Landscape)',
                'curvature': curvature_metrics_json,
                'heat_kernel': heat_traces,
                'topological_entropy': float(topo_entropy)
            },
            'surgery': {
                'necks_detected': necks,
                'proposals': surgery_proposals
            },
            'planck_geometry': planck_results
        }

        return report

    def analyze_directory(
        self, 
        directory: str, 
        device: str = 'cpu',
        pattern: str = '*.pt'
    ) -> List[Dict[str, Any]]:
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        checkpoints = sorted(path.glob(pattern))
        if not checkpoints:
            print(f"No checkpoints found in {directory} matching {pattern}")
            return []

        print(f"Found {len(checkpoints)} checkpoints")
        print("=" * 60)

        results = []
        for ckpt_path in checkpoints:
            try:
                report = self.analyze_checkpoint(str(ckpt_path), device)
                results.append(report)
                self._print_summary(report)
                print()
            except Exception as e:
                print(f"  Error processing {ckpt_path}: {e}")
                import traceback
                traceback.print_exc()
                continue

        return results

    def _print_summary(self, report: Dict[str, Any]) -> None:
        print("\n" + "=" * 80)
        print("RICCI FLOW ANALYSIS REPORT")
        print("=" * 80)
        
        curvature = report['ricci_flow']['curvature']
        print(f"\n[RICCI METRICS]")
        print(f"  Ricci Scalar (Total Curvature): {curvature['ricci_scalar']:.6e}")
        print(f"  Spectral Gap: {curvature['spectral_gap']:.6e}")
        print(f"  Condition Number (Stiffness): {curvature['condition_number_hessian']:.2e}")
        print(f"  Max Curvature: {curvature['max_curvature']:.6e}")
        print(f"  Min Curvature: {curvature['min_curvature']:.6e}")
        print(f"  Negative Curvatures (Saddles): {curvature['negative_curvature_count']}")

        print(f"\n[HEAT KERNEL THERMODYNAMICS]")
        for t, val in report['ricci_flow']['heat_kernel'].items():
            print(f"  Trace[{t}]: {val:.6f}")
        print(f"  Topological Entropy: {report['ricci_flow']['topological_entropy']:.6f}")

        print(f"\n[SINGULARITY DETECTION]")
        neck = report['surgery']['necks_detected']
        print(f"  Is Singular: {neck['is_singular']}")
        print(f"  Neck Severity: {neck['neck_severity']:.2f} (log10 scale)")
        print(f"  Surgery Proposals: {len(report['surgery']['proposals'])}")

        print(f"\n[GEOMETRIC PLANCK]")
        planck = report['planck_geometry']
        print(f"  h_bar (Geometric): {planck['h_bar_geometric']:.6e}")
        print(f"  h_bar (Thermodynamic): {planck['h_bar_thermodynamic']:.6e}")
        print(f"  h_bar (Unified): {planck['h_bar_unified']:.6e}")
        
        print("=" * 80 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Analyze Ricci Flow in Strassen checkpoints')
    parser.add_argument('path', nargs='?', default=CONFIG.CHECKPOINT_DIRECTORY)
    parser.add_argument('-o', '--output', default=CONFIG.OUTPUT_DIRECTORY)
    parser.add_argument('-d', '--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)
    
    pipeline = RicciFlowAnalyzerPipeline()
    path = Path(args.path)

    if path.is_file():
        report = pipeline.analyze_checkpoint(str(path), args.device)
        out_file = Path(args.output) / f"{path.stem}_ricci.json"
        with open(out_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        print(f"Saved: {out_file}")
    elif path.is_dir():
        results = []
        for ckpt in path.glob('*.pt'):
            try:
                report = pipeline.analyze_checkpoint(str(ckpt), args.device)
                results.append(report)
                # Save individual
                out_file = Path(args.output) / f"{ckpt.stem}_ricci.json"
                with open(out_file, 'w') as f:
                    json.dump(report, f, indent=2, default=str)
            except Exception as e:
                print(f"Skipping {ckpt}: {e}")

        # Save aggregate
        agg_file = Path(args.output) / "ricci_aggregate.json"
        with open(agg_file, 'w') as f:
            json.dump({'total': len(results), 'results': results}, f, indent=2, default=str)
        print(f"\nProcessed {len(results)} checkpoints. Aggregate saved: {agg_file}")
    else:
        print("Path not found.")

if __name__ == "__main__":
    main()
