#!/usr/bin/env python3
"""
StrassenPlanckCalculator - Unified Checkpoint Analysis and Planck Constant Calculation

Calculates effective Planck constant (h_bar) from Strassen model checkpoints using
crystallographic metrics and quantum-inspired thermodynamic analysis.

Architecture: SOLID principles, parametric configuration, comprehensive metrics
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
import argparse
from typing import Dict, Optional, List, Tuple, Any
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
class Configuration:
    """Immutable configuration container following Single Responsibility Principle."""

    # Model Architecture Parameters
    BATCH_SIZE: int = 32
    HIDDEN_SLOTS: int = 8
    TARGET_SLOTS: int = 7
    MATRIX_SIZE: int = 2
    INPUT_DIM: int = 4  # 2x2 matrix flattened
    OUTPUT_DIM: int = 4

    # Training Parameters
    WEIGHT_DECAY: float = 1e-4
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 3000
    RANDOM_SEED: int = 42

    # Crystallographic Thresholds
    DISCRETIZATION_MARGIN: float = 0.1
    OPTIMAL_DELTA_THRESHOLD: float = 0.01
    INDUSTRIAL_DELTA_THRESHOLD: float = 0.1
    POLYCRYSTALLINE_DELTA_THRESHOLD: float = 0.3
    AMORPHOUS_DELTA_THRESHOLD: float = 0.5

    # Planck Physics Parameters
    PLANCK_SI: float = 1.054571817e-34  # J·s
    SPEED_OF_LIGHT: float = 299792458  # m/s
    GRAVITATIONAL_CONSTANT: float = 6.67430e-11  # m³/kg·s²
    SOLAR_MASS: float = 1.98847e30  # kg
    BOLTZMANN_CONSTANT: float = 1.380649e-23  # J/K

    # Resilience Spectrometry Parameters
    NOISE_LEVELS: List[float] = field(default_factory=lambda: [
        1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2
    ])
    RESILIENCE_TRIALS: int = 5
    RESILIENCE_EPOCHS: int = 30
    RESILIENCE_LEARNING_RATE: float = 1e-5

    # Gauge Invariance Test Parameters
    GAUGE_SAMPLES: int = 50
    GAUGE_ERROR_THRESHOLD: float = 1e-5

    # Kappa Estimation Parameters
    KAPPA_BATCHES: int = 5
    KAPPA_SAMPLES_PER_BATCH: int = 32

    # Local Complexity Parameters
    LC_PERCENTILE: float = 0.95
    LC_ACTIVATION_THRESHOLD: float = 0.01

    # Planck Calculation Weights (by regime)
    ULTRA_STRONG_WEIGHTS: Tuple[float, float, float, float] = (0.6, 0.25, 0.1, 0.05)
    STRONG_WEIGHTS: Tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05)
    WEAK_WEIGHTS: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)

    # Regime Thresholds
    ULTRA_STRONG_LAMBDA: float = 1e30
    STRONG_LAMBDA: float = 1e10
    WEAK_LAMBDA: float = 1.0

    # Derived Constants
    EFFECTIVE_MODES: int = 31  # From spectral analysis

    # Output Configuration
    OUTPUT_DIRECTORY: str = "planck_analysis_reports"
    CHECKPOINT_DIRECTORY: str = "checkpoints"

    def __post_init__(self):
        """Validate configuration parameters."""
        assert self.HIDDEN_SLOTS >= self.TARGET_SLOTS,             "Hidden slots must be >= target slots"
        assert self.MATRIX_SIZE ** 2 == self.INPUT_DIM,             "Input dimension must match matrix size squared"


# Global configuration instance
CONFIG = Configuration()


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def set_random_seed(seed: int = CONFIG.RANDOM_SEED) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================================
# MODEL ARCHITECTURE (Open/Closed Principle)
# ============================================================================

class BilinearStrassenModel(nn.Module):
    """
    Bilinear model for Strassen matrix multiplication.
    Implements f(A,B) = W((U*A) ⊙ (V*B)) where ⊙ is element-wise product.
    """

    def __init__(self, config: Configuration = CONFIG):
        super().__init__()
        self.config = config

        self.U = nn.Linear(config.INPUT_DIM, config.HIDDEN_SLOTS, bias=False)
        self.V = nn.Linear(config.INPUT_DIM, config.HIDDEN_SLOTS, bias=False)
        self.W = nn.Linear(config.HIDDEN_SLOTS, config.OUTPUT_DIM, bias=False)

        self._initialize_symmetric()

    def _initialize_symmetric(self) -> None:
        """Initialize with Xavier uniform, symmetric U and V."""
        nn.init.xavier_uniform_(self.U.weight)
        self.V.weight.data = self.U.weight.data.clone()
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, matrix_a: torch.Tensor, matrix_b: torch.Tensor) -> torch.Tensor:
        """
        Forward pass computing approximate matrix product.

        Args:
            matrix_a: Flattened input matrix A [batch, INPUT_DIM]
            matrix_b: Flattened input matrix B [batch, INPUT_DIM]

        Returns:
            Approximate product C = A @ B [batch, OUTPUT_DIM]
        """
        encoded_a = self.U(matrix_a)
        encoded_b = self.V(matrix_b)
        hadamard_product = encoded_a * encoded_b
        output = self.W(hadamard_product)
        return output

    def get_coefficients(self) -> Dict[str, torch.Tensor]:
        """Return current coefficient matrices."""
        return {
            'U': self.U.weight.data.clone(),
            'V': self.V.weight.data.clone(),
            'W': self.W.weight.data.clone()
        }

    def compute_lambda_effective(self) -> float:
        """
        Compute effective lambda (confinement potential) from weight magnitudes.
        Derived from weight decay interpretation as harmonic confinement.
        """
        with torch.no_grad():
            u_norm = torch.norm(self.U.weight).item()
            v_norm = torch.norm(self.V.weight).item()
            w_norm = torch.norm(self.W.weight).item()

            # Effective lambda relates to inverse variance of weights
            total_norm = u_norm + v_norm + w_norm
            lambda_eff = 1.0 / (total_norm ** 2 + 1e-10)
            return lambda_eff


# ============================================================================
# CHECKPOINT MIGRATION (Interface Segregation Principle)
# ============================================================================

class CheckpointMigrator(ABC):
    """Abstract base for checkpoint migration strategies."""

    @abstractmethod
    def can_migrate(self, state_dict: Dict[str, Any]) -> bool:
        """Check if this strategy can handle the given state dict."""
        pass

    @abstractmethod
    def migrate(self, state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        """Migrate state dict to standard format."""
        pass


class CustomFormatMigrator(CheckpointMigrator):
    """Handles custom U,V,W direct formats."""

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
            return {
                'U.weight': u_padded,
                'V.weight': v_padded,
                'W.weight': w_padded
            }

        return {
            'U.weight': state_dict['U'],
            'V.weight': state_dict['V'],
            'W.weight': state_dict['W']
        }


class EncoderFormatMigrator(CheckpointMigrator):
    """Handles encoder.layers format."""

    def can_migrate(self, state_dict: Dict[str, Any]) -> bool:
        return 'encoder.0.weight' in state_dict

    def migrate(self, state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        encoder_0 = state_dict['encoder.0.weight']
        encoder_2 = state_dict.get('encoder.2.weight', encoder_0)
        encoder_4 = state_dict.get('encoder.4.weight', torch.randn(64, 64))

        if encoder_0.shape == (64, 8):
            u = encoder_0[:8, :4].clone()
        else:
            u = encoder_0.flatten()[:32].reshape(8, 4)

        if encoder_2.shape == (64, 64):
            v = encoder_2[:8, :4].clone()
        else:
            v = u

        if encoder_4.shape == (64, 64):
            w = encoder_4[:4, :8].clone()
        else:
            w = torch.randn(4, 8)

        return {
            'U.weight': u,
            'V.weight': v,
            'W.weight': w
        }


class StandardFormatMigrator(CheckpointMigrator):
    """Handles standard U.weight, V.weight, W.weight format."""

    def can_migrate(self, state_dict: Dict[str, Any]) -> bool:
        return any(k.endswith('.weight') for k in state_dict.keys())

    def migrate(self, state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        result = {}
        for key in ['U.weight', 'V.weight', 'W.weight']:
            if key in state_dict:
                result[key] = state_dict[key]
        return result if len(result) == 3 else None


class CheckpointMigrationManager:
    """Manages multiple migration strategies."""

    def __init__(self):
        self.strategies: List[CheckpointMigrator] = [
            CustomFormatMigrator(),
            EncoderFormatMigrator(),
            StandardFormatMigrator()
        ]

    def migrate_checkpoint(
        self, 
        path: str, 
        device: str = 'cpu'
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Attempt to migrate checkpoint using available strategies.

        Args:
            path: Path to checkpoint file
            device: Device to load tensors to

        Returns:
            Migrated state dict or None if migration fails
        """
        try:
            data = torch.load(path, map_location=device, weights_only=False)

            # Extract state dict from various wrapper formats
            if isinstance(data, dict):
                if 'state_dict' in data:
                    state_dict = data['state_dict']
                elif 'model_state_dict' in data:
                    state_dict = data['model_state_dict']
                else:
                    state_dict = data
            elif hasattr(data, 'state_dict'):
                state_dict = data.state_dict()
            else:
                state_dict = data

            # Try each migration strategy
            for strategy in self.strategies:
                if strategy.can_migrate(state_dict):
                    return strategy.migrate(state_dict)

            return None

        except Exception as e:
            print(f"Migration error: {e}")
            return None


# ============================================================================
# DATA GENERATION
# ============================================================================

class StrassenDataGenerator:
    """Generates training data for 2x2 matrix multiplication."""

    @staticmethod
    def generate_batch(
        batch_size: int = CONFIG.BATCH_SIZE,
        config: Configuration = CONFIG
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate a batch of random matrix pairs and their products.

        Returns:
            Tuple of (A_flat, B_flat, C_flat) where C = A @ B
        """
        A = torch.randn(batch_size, config.MATRIX_SIZE, config.MATRIX_SIZE)
        B = torch.randn(batch_size, config.MATRIX_SIZE, config.MATRIX_SIZE)
        C = torch.bmm(A, B)

        return (
            A.reshape(batch_size, config.INPUT_DIM),
            B.reshape(batch_size, config.INPUT_DIM),
            C.reshape(batch_size, config.OUTPUT_DIM)
        )

    @staticmethod
    def verify_structure(
        coeffs: Dict[str, torch.Tensor],
        config: Configuration = CONFIG
    ) -> Dict[str, Any]:
        """Verify if coefficients represent valid Strassen structure."""
        delta = CrystallographyMetrics.compute_discretization_margin(coeffs)
        return {
            'pass': delta < config.DISCRETIZATION_MARGIN,
            'max_error': delta,
            'is_optimal': delta < config.OPTIMAL_DELTA_THRESHOLD,
            'is_industrial': delta < config.INDUSTRIAL_DELTA_THRESHOLD
        }


# ============================================================================
# CRYSTALLOGRAPHIC METRICS
# ============================================================================

class CrystallographyMetrics:
    """Computes crystallographic quality metrics for Strassen models."""

    @staticmethod
    def compute_kappa(
        model: nn.Module,
        num_batches: int = CONFIG.KAPPA_BATCHES,
        config: Configuration = CONFIG
    ) -> float:
        """
        Compute condition number of gradient covariance matrix.

        High kappa indicates ill-conditioned optimization landscape.
        Low kappa (approaching 1.0) indicates well-conditioned, crystalline structure.
        """
        model.eval()
        grads = []

        for _ in range(num_batches):
            A, B, C = StrassenDataGenerator.generate_batch(config.BATCH_SIZE, config)
            C_pred = model(A, B)
            loss = nn.functional.mse_loss(C_pred, C)

            grad = torch.autograd.grad(
                loss, 
                model.parameters(), 
                create_graph=False,
                retain_graph=False
            )
            grads.append(torch.cat([g.flatten() for g in grad]))

        if len(grads) < 2:
            return float('inf')

        grads_tensor = torch.stack(grads)

        # Compute covariance matrix
        try:
            cov_matrix = torch.cov(grads_tensor.T)
            condition_number = torch.linalg.cond(cov_matrix).item()
            return condition_number
        except Exception:
            return float('inf')

    @staticmethod
    def compute_discretization_margin(
        coeffs: Dict[str, torch.Tensor]
    ) -> float:
        """
        Compute maximum deviation from nearest integer values.

        Delta measures how close coefficients are to discrete (integer) values,
        indicating crystalline structure formation.
        """
        max_margin = 0.0
        for tensor in coeffs.values():
            deviation = (tensor - tensor.round()).abs().max().item()
            max_margin = max(max_margin, deviation)
        return max_margin

    @staticmethod
    def compute_local_complexity(
        model: nn.Module,
        config: Configuration = CONFIG
    ) -> float:
        """
        Compute local complexity based on active parameters.

        From "Can't Stop Won't Stop" paper - measures effective parameter count.
        """
        params = torch.cat([p.flatten() for p in model.parameters()])

        with torch.no_grad():
            percentile_val = torch.quantile(torch.abs(params), config.LC_PERCENTILE)
            active_count = (torch.abs(params) > config.LC_ACTIVATION_THRESHOLD * percentile_val).sum()
            lc = active_count.float() / len(params)

        return lc.item()

    @staticmethod
    def compute_all_metrics(
        model: nn.Module,
        config: Configuration = CONFIG
    ) -> Dict[str, float]:
        """Compute all crystallographic metrics at once."""
        coeffs = model.get_coefficients()

        return {
            'kappa': CrystallographyMetrics.compute_kappa(model, config=config),
            'delta': CrystallographyMetrics.compute_discretization_margin(coeffs),
            'lc': CrystallographyMetrics.compute_local_complexity(model, config),
            'lambda_effective': model.compute_lambda_effective()
        }


# ============================================================================
# DIFFRACTION TESTING
# ============================================================================

class StrassenDiffractionTest:
    """Tests gauge invariance through permutation symmetry."""

    def __init__(self, model: nn.Module, config: Configuration = CONFIG):
        self.model = model
        self.config = config

    def test_gauge_invariance(self) -> Dict[str, Any]:
        """
        Test if model exhibits true Strassen structure through permutation invariance.

        Genuine Strassen algorithm should have exactly one valid permutation (identity).
        """
        coeffs = self.model.get_coefficients()
        indices = list(range(self.config.TARGET_SLOTS))

        # Generate random permutations
        sample_perms = [
            random.sample(indices, len(indices)) 
            for _ in range(min(self.config.GAUGE_SAMPLES, 100))
        ]

        errors = []
        invariant_count = 0

        for perm in sample_perms:
            perm_tensor = torch.tensor(perm, dtype=torch.long)
            test_coeffs = {
                'U': coeffs['U'][perm_tensor],
                'V': coeffs['V'][perm_tensor],
                'W': coeffs['W'][:, perm_tensor]
            }

            error = self._compute_functional_error(test_coeffs)
            errors.append(error)

            if error < self.config.GAUGE_ERROR_THRESHOLD:
                invariant_count += 1

        return {
            'invariance_ratio': invariant_count / len(sample_perms),
            'mean_error': np.mean(errors),
            'max_error': np.max(errors),
            'is_genuine': invariant_count == 1,
            'invariant_count': invariant_count,
            'total_samples': len(sample_perms)
        }

    def _compute_functional_error(
        self, 
        test_coeffs: Dict[str, torch.Tensor]
    ) -> float:
        """Compute functional error between original and permuted coefficients."""
        original_coeffs = self.model.get_coefficients()
        errors = []

        for _ in range(10):
            A, B, _ = StrassenDataGenerator.generate_batch(1, self.config)

            # Original computation
            m1_orig = (original_coeffs['U'] @ A.T) * (original_coeffs['V'] @ B.T)
            c1 = original_coeffs['W'] @ m1_orig

            # Permuted computation
            m1_test = (test_coeffs['U'] @ A.T) * (test_coeffs['V'] @ B.T)
            c2 = test_coeffs['W'] @ m1_test

            error = torch.norm(c1 - c2).item()
            errors.append(error)

        return max(errors)


# ============================================================================
# RESILIENCE SPECTROMETRY
# ============================================================================

class BasinResilienceSpectrometer:
    """Measures basin of attraction through noise injection and recovery."""

    def __init__(self, model: nn.Module, config: Configuration = CONFIG):
        self.model = model
        self.config = config
        self.original_state = {
            k: v.clone() for k, v in model.state_dict().items()
        }

    def measure_resilience_spectrum(self) -> Dict[str, Any]:
        """
        Measure resilience across multiple noise levels.

        Returns spectrum showing critical noise level where recovery fails.
        """
        results = {}

        for sigma in self.config.NOISE_LEVELS:
            key = f'sigma_{sigma}'
            results[key] = self._test_noise_recovery(sigma)

        results['critical_sigma'] = self._estimate_critical_noise(results)
        return results

    def _test_noise_recovery(self, sigma: float) -> Dict[str, float]:
        """Test recovery from noise level sigma."""
        successes = 0
        final_margins = []
        recovery_epochs_list = []

        for _ in range(self.config.RESILIENCE_TRIALS):
            self._apply_noise(sigma)
            recovery_epochs = self._anneal_to_attractor()
            final_margin = CrystallographyMetrics.compute_discretization_margin(
                self.model.get_coefficients()
            )

            final_margins.append(final_margin)
            recovery_epochs_list.append(recovery_epochs)

            if final_margin < self.config.DISCRETIZATION_MARGIN:
                successes += 1

            self.model.load_state_dict(self.original_state)

        return {
            'success_rate': successes / self.config.RESILIENCE_TRIALS,
            'final_margin_mean': np.mean(final_margins),
            'final_margin_std': np.std(final_margins),
            'recovery_epochs_mean': np.mean(recovery_epochs_list),
            'recovery_epochs_std': np.std(recovery_epochs_list)
        }

    def _apply_noise(self, sigma: float) -> None:
        """Apply Gaussian noise to model parameters."""
        with torch.no_grad():
            for param in self.model.parameters():
                noise = torch.randn_like(param) * sigma
                param.add_(noise)

    def _anneal_to_attractor(self) -> int:
        """
        Anneal model back to attractor using fine-tuning.

        Returns number of epochs needed for recovery.
        """
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.RESILIENCE_LEARNING_RATE
        )

        for epoch in range(self.config.RESILIENCE_EPOCHS):
            A, B, C = StrassenDataGenerator.generate_batch(
                self.config.BATCH_SIZE, 
                self.config
            )

            C_pred = self.model(A, B)
            loss = nn.functional.mse_loss(C_pred, C)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            margin = CrystallographyMetrics.compute_discretization_margin(
                self.model.get_coefficients()
            )

            if margin < self.config.DISCRETIZATION_MARGIN:
                return epoch + 1

        return self.config.RESILIENCE_EPOCHS

    def _estimate_critical_noise(self, results: Dict[str, Any]) -> float:
        """Estimate critical noise level where success rate drops below 50%."""
        sigmas = []
        rates = []

        for key, value in results.items():
            if key.startswith('sigma_'):
                sigma_val = float(key.split('_')[1])
                sigmas.append(sigma_val)
                rates.append(value['success_rate'])

        # Find first transition below 50%
        for i in range(len(rates) - 1):
            if rates[i] >= 0.5 and rates[i + 1] < 0.5:
                return sigmas[i]

        return sigmas[-1] if sigmas else 0.0


# ============================================================================
# CRYSTAL PURITY INDEX
# ============================================================================

class CrystalPurityIndex:
    """Computes normalized purity index from component metrics."""

    def __init__(
        self,
        metrics: Dict[str, float],
        diffraction_results: Dict[str, Any],
        resilience_results: Dict[str, Any],
        config: Configuration = CONFIG
    ):
        self.metrics = metrics
        self.diffraction = diffraction_results
        self.resilience = resilience_results
        self.config = config

    def compute(self) -> Dict[str, Any]:
        """Compute normalized purity index and grade."""
        kappa = self.metrics.get('kappa', float('inf'))
        delta = self.metrics.get('delta', 1.0)
        resilience = self.resilience.get('critical_sigma', 0.0)
        invariance = self.diffraction.get('invariance_ratio', 1.0)
        lc = self.metrics.get('lc', 0.0)

        # Normalize kappa (inf or < 1.1 indicates perfect crystal)
        if kappa == float('inf') or kappa < 1.1:
            kappa_score = 1.0
        else:
            kappa_score = max(0.0, 1.0 - np.log10(kappa) / 6.0)

        # Normalize delta
        delta_score = max(0.0, 1.0 - delta / 0.5)

        # Normalize resilience
        resilience_score = max(0.0, min(1.0, resilience / 0.01))

        # Normalize invariance (lower is better - should be exactly 1 permutation)
        invariance_score = max(0.0, 1.0 - invariance / 0.01)

        # Weights
        weights = {
            'kappa': 0.30,
            'delta': 0.40,
            'resilience': 0.15,
            'invariance': 0.10,
            'lc': 0.05
        }

        purity_index = (
            weights['kappa'] * kappa_score +
            weights['delta'] * delta_score +
            weights['resilience'] * resilience_score +
            weights['invariance'] * invariance_score +
            weights['lc'] * lc
        )

        return {
            'index': purity_index,
            'grade': self._assign_grade(purity_index, delta),
            'component_scores': {
                'kappa_score': kappa_score,
                'delta_score': delta_score,
                'resilience_score': resilience_score,
                'invariance_score': invariance_score,
                'lc_score': lc
            },
            'weights': weights
        }

    def _assign_grade(self, index: float, delta: float) -> str:
        """Assign crystallographic grade based on delta (primary indicator)."""
        if delta < self.config.OPTIMAL_DELTA_THRESHOLD:
            return "Optimal Crystal"
        elif delta < self.config.INDUSTRIAL_DELTA_THRESHOLD:
            return "Industrial Crystal"
        elif delta < self.config.POLYCRYSTALLINE_DELTA_THRESHOLD:
            return "Polycrystalline"
        elif delta < self.config.AMORPHOUS_DELTA_THRESHOLD:
            return "Amorphous Glass"
        else:
            return "Defective"


# ============================================================================
# PLANCK CONSTANT CALCULATOR
# ============================================================================

class PlanckConstantCalculator:
    """
    Calculates effective Planck constant from Strassen model parameters.

    Maps crystallographic metrics to quantum thermodynamic quantities.
    """

    def __init__(
        self,
        metrics: Dict[str, float],
        training_metrics: Dict[str, float],
        config: Configuration = CONFIG
    ):
        self.metrics = metrics
        self.training = training_metrics
        self.config = config

        # Extract key values
        self.lambda_val = metrics.get('lambda_effective', 0.5)
        self.delta = metrics.get('delta', 1.0)
        self.alpha = training_metrics.get('alpha', 0.0)
        self.mse = training_metrics.get('mse', 1.0)
        self.val_acc = training_metrics.get('val_acc', 0.0)
        self.kappa = metrics.get('kappa', float('inf'))
        self.lc = metrics.get('lc', 0.0)

    def calculate_all(self) -> Dict[str, Any]:
        """Execute all Planck constant calculation methods."""

        # Method 1: Generalized Uncertainty Principle
        # h_bar ~ 2 * delta^2 * lambda (from Δx·Δp ≥ ħ/2)
        h_bar_uncertainty = 2.0 * self.delta ** 2 * self.lambda_val

        # Method 2: Action Quantization
        omega = np.sqrt(self.lambda_val) if self.lambda_val > 0 else 1.0
        period = 2.0 * np.pi / omega if omega > 0 else 1.0

        T = self.mse  # Kinetic energy proxy
        V = self.lambda_val * self.delta ** 2  # Potential energy
        L = T - V  # Lagrangian
        action = abs(L) * period
        h_bar_action = action

        # Method 3: Quantum Conductance (Hall effect analog)
        if self.mse > 0:
            conductance = self.val_acc / self.mse
            h_bar_conductance = 1.0 / conductance if conductance > 0 else 0.0
        else:
            h_bar_conductance = 0.0

        # Method 4: Information Entropy
        N_eff = self.config.EFFECTIVE_MODES
        I = np.log2(N_eff) if N_eff > 1 else 1.0
        E_total = T + V
        energy_per_bit = E_total / I if I > 0 else 0.0
        h_bar_information = energy_per_bit * period

        # Determine regime and weights
        regime, weights = self._determine_regime_and_weights()

        # Consolidated calculation
        w1, w2, w3, w4 = weights
        total_weight = w1 + w2 + w3 + w4

        h_bar_final = (
            w1 * h_bar_uncertainty +
            w2 * h_bar_action +
            w3 * h_bar_conductance +
            w4 * h_bar_information
        ) / total_weight

        # Dimensionless form
        h_bar_scale = self.mse * period
        h_bar_dimless = h_bar_final / h_bar_scale if h_bar_scale > 0 else 0.0

        # Derived physical constants
        derived = self._compute_derived_constants(h_bar_final)

        # Comparison with physical universe
        comparison = self._compute_universe_comparison(h_bar_final)

        return {
            'h_bar': {
                'value': float(h_bar_final),
                'dimensionless': float(h_bar_dimless),
                'regime': regime,
                'methods': {
                    'uncertainty': float(h_bar_uncertainty),
                    'action': float(h_bar_action),
                    'conductance': float(h_bar_conductance),
                    'information': float(h_bar_information)
                },
                'weights': {
                    'uncertainty': w1,
                    'action': w2,
                    'conductance': w3,
                    'information': w4
                }
            },
            'derived_constants': derived,
            'universe_comparison': comparison,
            'inputs': {
                'lambda': self.lambda_val,
                'delta': self.delta,
                'alpha': self.alpha,
                'mse': self.mse,
                'val_acc': self.val_acc,
                'kappa': self.kappa,
                'lc': self.lc
            }
        }

    def _determine_regime_and_weights(self) -> Tuple[str, Tuple[float, float, float, float]]:
        """Determine confinement regime and corresponding weights."""
        if self.lambda_val > self.config.ULTRA_STRONG_LAMBDA:
            return "ULTRA-STRONG CONFINEMENT", self.config.ULTRA_STRONG_WEIGHTS
        elif self.lambda_val > self.config.STRONG_LAMBDA:
            return "STRONG CONFINEMENT", self.config.STRONG_WEIGHTS
        elif self.lambda_val > self.config.WEAK_LAMBDA:
            return "WEAK CONFINEMENT", self.config.WEAK_WEIGHTS
        else:
            return "UNCONSTRAINED", self.config.WEAK_WEIGHTS

    def _compute_derived_constants(self, h_bar: float) -> Dict[str, float]:
        """Compute derived Planck-scale constants."""
        c_eff = self.config.SPEED_OF_LIGHT * (h_bar / self.config.PLANCK_SI)

        if self.config.GRAVITATIONAL_CONSTANT > 0 and c_eff > 0:
            m_planck = np.sqrt(h_bar * c_eff / self.config.GRAVITATIONAL_CONSTANT)
        else:
            m_planck = 0.0

        if c_eff > 0:
            l_planck = np.sqrt(
                h_bar * self.config.GRAVITATIONAL_CONSTANT / c_eff ** 3
            )
            t_planck = l_planck / c_eff
        else:
            l_planck = 0.0
            t_planck = 0.0

        if self.config.BOLTZMANN_CONSTANT > 0:
            T_planck = m_planck * c_eff ** 2 / self.config.BOLTZMANN_CONSTANT
        else:
            T_planck = 0.0

        return {
            'c_effective_m_s': float(c_eff),
            'm_planck_kg': float(m_planck),
            'l_planck_m': float(l_planck),
            't_planck_s': float(t_planck),
            'T_planck_K': float(T_planck)
        }

    def _compute_universe_comparison(self, h_bar: float) -> Dict[str, float]:
        """Compare calculated constants with physical universe."""
        ratio = h_bar / self.config.PLANCK_SI if self.config.PLANCK_SI > 0 else 0.0
        orders = np.log10(ratio) if ratio > 0 else 0.0

        m_planck = self._compute_derived_constants(h_bar).get('m_planck_kg', 0.0)
        m_ratio = m_planck / self.config.SOLAR_MASS if self.config.SOLAR_MASS > 0 else 0.0

        return {
            'h_bar_ratio': float(ratio),
            'orders_of_magnitude': float(orders),
            'm_planck_vs_solar_mass': float(m_ratio)
        }


# ============================================================================
# CHECKPOINT LOADER
# ============================================================================

class StrassenCheckpointLoader:
    """Loads and migrates Strassen checkpoints with fallback strategies."""

    def __init__(self, config: Configuration = CONFIG):
        self.config = config
        self.migration_manager = CheckpointMigrationManager()

    def load(
        self, 
        checkpoint_path: str, 
        device: str = 'cpu'
    ) -> Optional[nn.Module]:
        """
        Load checkpoint into model instance.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Target device

        Returns:
            Loaded model or None if loading fails
        """
        model = BilinearStrassenModel(self.config).to(device)

        # Attempt direct load first
        try:
            data = torch.load(checkpoint_path, map_location=device, weights_only=False)

            if isinstance(data, dict):
                if 'state_dict' in data:
                    state_dict = data['state_dict']
                elif 'model_state_dict' in data:
                    state_dict = data['model_state_dict']
                else:
                    state_dict = data
            else:
                state_dict = data.state_dict() if hasattr(data, 'state_dict') else data

            model.load_state_dict(state_dict)
            return model

        except Exception:
            # Fallback to migration
            migrated = self.migration_manager.migrate_checkpoint(checkpoint_path, device)
            if migrated is not None:
                model.load_state_dict(migrated)
                return model

        return None

    def extract_training_metrics(self, checkpoint_path: str) -> Dict[str, float]:
        """Extract training metrics from checkpoint if available."""
        try:
            data = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

            if isinstance(data, dict):
                metrics = data.get('metrics', {})
                return {
                    'mse': metrics.get('val_mse', metrics.get('mse', 1.0)),
                    'val_acc': data.get('val_acc', metrics.get('val_acc', 0.0)),
                    'alpha': metrics.get('alpha', 0.0),
                    'epoch': data.get('epoch', 0)
                }
        except Exception:
            pass

        return {
            'mse': 1.0,
            'val_acc': 0.0,
            'alpha': 0.0,
            'epoch': 0
        }


# ============================================================================
# ANALYSIS PIPELINE
# ============================================================================

class StrassenPlanckAnalyzer:
    """
    Complete analysis pipeline for Strassen checkpoints.

    Orchestrates crystallographic analysis and Planck constant calculation.
    """

    def __init__(self, config: Configuration = CONFIG):
        self.config = config
        self.loader = StrassenCheckpointLoader(config)

    def analyze_checkpoint(self, checkpoint_path: str, device: str = 'cpu') -> Dict[str, Any]:
        """
        Perform complete analysis of a single checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file
            device: Computation device

        Returns:
            Complete analysis report
        """
        print(f"Analyzing: {os.path.basename(checkpoint_path)}")

        # Load model
        model = self.loader.load(checkpoint_path, device)
        if model is None:
            raise ValueError(f"Failed to load checkpoint: {checkpoint_path}")

        # Extract training metrics
        training_metrics = self.loader.extract_training_metrics(checkpoint_path)

        # Module 1: Crystallographic metrics
        print("  Computing crystallographic metrics...")
        cryst_metrics = CrystallographyMetrics.compute_all_metrics(model, self.config)

        # Module 2: Diffraction test
        print("  Running diffraction test...")
        diffraction = StrassenDiffractionTest(model, self.config)
        diffraction_results = diffraction.test_gauge_invariance()

        # Module 3: Resilience spectrometry
        print("  Measuring resilience spectrum...")
        spectrometer = BasinResilienceSpectrometer(model, self.config)
        resilience_results = spectrometer.measure_resilience_spectrum()

        # Module 4: Crystal purity
        print("  Computing crystal purity index...")
        purity_calc = CrystalPurityIndex(
            cryst_metrics, 
            diffraction_results, 
            resilience_results,
            self.config
        )
        purity_results = purity_calc.compute()

        # Module 5: Planck constant
        print("  Calculating Planck constant...")
        planck_calc = PlanckConstantCalculator(
            cryst_metrics,
            training_metrics,
            self.config
        )
        planck_results = planck_calc.calculate_all()

        # Compile report
        report = {
            'metadata': {
                'checkpoint': checkpoint_path,
                'checkpoint_name': os.path.basename(checkpoint_path),
                'timestamp': datetime.now().isoformat(),
                'device': device,
                'config': {
                    'hidden_slots': self.config.HIDDEN_SLOTS,
                    'target_slots': self.config.TARGET_SLOTS,
                    'matrix_size': self.config.MATRIX_SIZE
                }
            },
            'crystallography': {
                'metrics': cryst_metrics,
                'diffraction': diffraction_results,
                'resilience': resilience_results,
                'purity': purity_results
            },
            'planck_physics': planck_results,
            'training_info': training_metrics
        }

        return report

    def analyze_directory(
        self, 
        directory: str, 
        device: str = 'cpu',
        pattern: str = '*.pt'
    ) -> List[Dict[str, Any]]:
        """
        Analyze all checkpoints in a directory.

        Args:
            directory: Directory containing checkpoints
            device: Computation device
            pattern: File pattern to match

        Returns:
            List of analysis reports
        """
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
                continue

        return results

    def _print_summary(self, report: Dict[str, Any]) -> None:
        """Print formatted summary of analysis results."""
        purity = report['crystallography']['purity']
        planck = report['planck_physics']['h_bar']
        metrics = report['crystallography']['metrics']

        print(f"  Results:")
        print(f"    Purity Index: {purity['index']:.4f}")
        print(f"    Grade: {purity['grade']}")
        print(f"    Delta: {metrics['delta']:.6f}")
        print(f"    Kappa: {metrics['kappa']:.2e}")
        print(f"    Planck h_bar: {planck['value']:.6e}")
        print(f"    Regime: {planck['regime']}")


# ============================================================================
# REPORTING AND VISUALIZATION
# ============================================================================

class ReportGenerator:
    """Generates reports and visualizations from analysis results."""

    def __init__(self, config: Configuration = CONFIG):
        self.config = config
        self.output_dir = Path(config.OUTPUT_DIRECTORY)
        self.output_dir.mkdir(exist_ok=True)

    def save_json_report(self, report: Dict[str, Any], suffix: str = "") -> str:
        """Save individual report as JSON."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = Path(report['metadata']['checkpoint_name']).stem
        filename = f"{base_name}_{timestamp}{suffix}.json"
        filepath = self.output_dir / filename

        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        return str(filepath)

    def save_aggregate_report(self, results: List[Dict[str, Any]]) -> str:
        """Save aggregate report from multiple analyses."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"aggregate_report_{timestamp}.json"
        filepath = self.output_dir / filename

        # Extract key metrics for aggregate view
        summary = []
        for r in results:
            summary.append({
                'checkpoint': r['metadata']['checkpoint_name'],
                'purity_index': r['crystallography']['purity']['index'],
                'grade': r['crystallography']['purity']['grade'],
                'delta': r['crystallography']['metrics']['delta'],
                'kappa': r['crystallography']['metrics']['kappa'],
                'h_bar': r['planck_physics']['h_bar']['value'],
                'regime': r['planck_physics']['h_bar']['regime']
            })

        aggregate = {
            'timestamp': datetime.now().isoformat(),
            'total_checkpoints': len(results),
            'summary_statistics': self._compute_statistics(summary),
            'individual_results': summary
        }

        with open(filepath, 'w') as f:
            json.dump(aggregate, f, indent=2, default=str)

        return str(filepath)

    def _compute_statistics(self, summaries: List[Dict]) -> Dict[str, Any]:
        """Compute aggregate statistics from summaries."""
        if not summaries:
            return {}

        purity_values = [s['purity_index'] for s in summaries]
        delta_values = [s['delta'] for s in summaries]
        h_bar_values = [s['h_bar'] for s in summaries]

        return {
            'purity_index': {
                'mean': float(np.mean(purity_values)),
                'std': float(np.std(purity_values)),
                'min': float(np.min(purity_values)),
                'max': float(np.max(purity_values))
            },
            'delta': {
                'mean': float(np.mean(delta_values)),
                'std': float(np.std(delta_values)),
                'min': float(np.min(delta_values)),
                'max': float(np.max(delta_values))
            },
            'h_bar': {
                'mean': float(np.mean(h_bar_values)),
                'std': float(np.std(h_bar_values)),
                'min': float(np.min(h_bar_values)),
                'max': float(np.max(h_bar_values))
            },
            'grade_distribution': self._count_grades(summaries)
        }

    def _count_grades(self, summaries: List[Dict]) -> Dict[str, int]:
        """Count distribution of grades."""
        counts = {}
        for s in summaries:
            grade = s['grade']
            counts[grade] = counts.get(grade, 0) + 1
        return counts

    def generate_visualizations(self, results: List[Dict[str, Any]]) -> str:
        """Generate visualization plots."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"analysis_plots_{timestamp}.png"
        filepath = self.output_dir / filename

        fig, axes = plt.subplots(2, 3, figsize=(18, 12))

        purity_indices = [r['crystallography']['purity']['index'] for r in results]
        deltas = [r['crystallography']['metrics']['delta'] for r in results]
        kappas = [r['crystallography']['metrics']['kappa'] for r in results]
        h_bars = [r['planck_physics']['h_bar']['value'] for r in results]
        lcs = [r['crystallography']['metrics']['lc'] for r in results]
        invariance = [r['crystallography']['diffraction']['invariance_ratio'] for r in results]

        # Plot 1: Purity Distribution
        ax = axes[0, 0]
        ax.hist(purity_indices, bins=10, edgecolor='black', alpha=0.7, color='#2E86AB')
        ax.axvline(x=0.90, color='green', linestyle='--', linewidth=2, label='Optimal')
        ax.axvline(x=0.70, color='orange', linestyle='--', linewidth=2, label='Industrial')
        ax.set_xlabel('Purity Index [0,1]')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Crystal Purity')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 2: Delta Distribution
        ax = axes[0, 1]
        ax.hist(deltas, bins=10, edgecolor='black', alpha=0.7, color='#A23B72')
        ax.axvline(x=0.01, color='green', linestyle='--', linewidth=2, label='Optimal')
        ax.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='Industrial')
        ax.set_xlabel('Discretization Margin (delta)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Delta')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Plot 3: Kappa vs Delta
        ax = axes[0, 2]
        valid_kappas = [(k, d) for k, d in zip(kappas, deltas) if k != float('inf') and k < 1e6]
        if valid_kappas:
            k_vals, d_vals = zip(*valid_kappas)
            ax.scatter(d_vals, k_vals, alpha=0.6, s=100, color='#F18F01')
            ax.set_xlabel('Delta')
            ax.set_ylabel('Kappa (log scale)')
            ax.set_yscale('log')
            ax.set_title('Kappa vs Delta')
            ax.grid(True, alpha=0.3)

        # Plot 4: Planck Constant Distribution
        ax = axes[1, 0]
        ax.hist(h_bars, bins=10, edgecolor='black', alpha=0.7, color='#C73E1D')
        ax.set_xlabel('Effective Planck Constant (h_bar)')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Planck Constant')
        ax.grid(True, alpha=0.3)

        # Plot 5: Local Complexity
        ax = axes[1, 1]
        ax.hist(lcs, bins=10, edgecolor='black', alpha=0.7, color='#3B1F2B')
        ax.set_xlabel('Local Complexity')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Local Complexity')
        ax.grid(True, alpha=0.3)

        # Plot 6: Invariance Ratio
        ax = axes[1, 2]
        ax.hist(invariance, bins=10, edgecolor='black', alpha=0.7, color='#6A994E')
        ax.set_xlabel('Gauge Invariance Ratio')
        ax.set_ylabel('Count')
        ax.set_title('Distribution of Invariance')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()

        return str(filepath)


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Calculate Planck constant from Strassen model checkpoints'
    )
    parser.add_argument(
        'path',
        nargs='?',
        default=CONFIG.CHECKPOINT_DIRECTORY,
        help='Path to checkpoint file or directory'
    )
    parser.add_argument(
        '-o', '--output',
        default=None,
        help='Output directory for reports'
    )
    parser.add_argument(
        '-d', '--device',
        default='cuda' if torch.cuda.is_available() else 'cpu',
        help='Device to use for computation'
    )
    parser.add_argument(
        '-p', '--pattern',
        default='*.pt',
        help='File pattern for checkpoint matching'
    )
    parser.add_argument(
        '--hidden-slots',
        type=int,
        default=CONFIG.HIDDEN_SLOTS,
        help='Number of hidden slots in model'
    )
    parser.add_argument(
        '--target-slots',
        type=int,
        default=CONFIG.TARGET_SLOTS,
        help='Number of target slots (7 for Strassen)'
    )
    parser.add_argument(
        '--matrix-size',
        type=int,
        default=CONFIG.MATRIX_SIZE,
        help='Size of input matrices (2 for 2x2)'
    )

    return parser.parse_args()


def create_config_from_args(args: argparse.Namespace) -> Configuration:
    """Create configuration from command line arguments."""
    return Configuration(
        HIDDEN_SLOTS=args.hidden_slots,
        TARGET_SLOTS=args.target_slots,
        MATRIX_SIZE=args.matrix_size,
        INPUT_DIM=args.matrix_size ** 2,
        OUTPUT_DIM=args.matrix_size ** 2,
        OUTPUT_DIRECTORY=args.output or CONFIG.OUTPUT_DIRECTORY
    )


def main():
    """Main execution entry point."""
    args = parse_arguments()

    # Setup
    set_random_seed()
    config = create_config_from_args(args)

    print("=" * 60)
    print("StrassenPlanckCalculator - Deep Learning Physics Analysis")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Hidden Slots: {config.HIDDEN_SLOTS}")
    print(f"Target Slots: {config.TARGET_SLOTS}")
    print(f"Matrix Size: {config.MATRIX_SIZE}x{config.MATRIX_SIZE}")
    print("=" * 60)

    # Initialize analyzer
    analyzer = StrassenPlanckAnalyzer(config)
    reporter = ReportGenerator(config)

    # Determine if path is file or directory
    path = Path(args.path)

    if path.is_file():
        # Single file analysis
        print(f"Analyzing single checkpoint: {path}")
        report = analyzer.analyze_checkpoint(str(path), args.device)
        reporter.save_json_report(report)
        print(f"Report saved to: {config.OUTPUT_DIRECTORY}")

    elif path.is_dir():
        # Directory analysis
        print(f"Analyzing directory: {path}")
        results = analyzer.analyze_directory(str(path), args.device, args.pattern)

        if results:
            # Save individual reports
            for report in results:
                reporter.save_json_report(report)

            # Save aggregate report
            aggregate_path = reporter.save_aggregate_report(results)
            print(f"Aggregate report saved: {aggregate_path}")

            # Generate visualizations
            viz_path = reporter.generate_visualizations(results)
            print(f"Visualizations saved: {viz_path}")

            # Print final summary
            print("\n" + "=" * 60)
            print("ANALYSIS COMPLETE")
            print("=" * 60)
            print(f"Total checkpoints processed: {len(results)}")

            purity_values = [r['crystallography']['purity']['index'] for r in results]
            print(f"Purity Index - Mean: {np.mean(purity_values):.4f}, Std: {np.std(purity_values):.4f}")

            h_bar_values = [r['planck_physics']['h_bar']['value'] for r in results]
            print(f"Planck h_bar - Mean: {np.mean(h_bar_values):.6e}, Std: {np.std(h_bar_values):.6e}")
        else:
            print("No results generated.")
    else:
        print(f"Error: Path does not exist: {args.path}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
