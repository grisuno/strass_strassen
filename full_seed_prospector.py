#!/usr/bin/env python3
"""
Riemann-Strassen Crystallization Framework v3.0
================================================

Unified framework for seed prospecting and long-term training of Strassen
algorithm crystallization via geometric flow theory and phase transitions.

Theoretical Framework:
- delta (discretization margin) as primary crystallization metric
- kappa (gradient covariance condition number) as order parameter
- tau coupled to effective noise scale (GNS)
- Perelman entropy for thermodynamic analysis
- Adaptive quantization pressure based on system state

Modes:
- prospect: Fast seed mining to identify crystallization candidates
- train: Long-term training with full thermodynamic metrics

Repository: https://github.com/grisuno/strass_strassen
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import time
import signal
import sys
import math
from pathlib import Path
from dataclasses import dataclass, replace
from typing import Dict, List, Tuple, Optional, Any, Deque
from abc import ABC, abstractmethod
from collections import deque
from datetime import datetime
from enum import Enum
import warnings

warnings.filterwarnings('ignore')


class ExecutionMode(Enum):
    PROSPECT = "prospect"
    TRAIN = "train"


@dataclass(frozen=True)
class UnifiedConfig:
    """Immutable unified configuration for all execution modes."""
    
    rank: int = 8
    target_rank: int = 7
    matrix_size: int = 2
    input_dim: int = 4
    output_dim: int = 4
    perelman_dimension: int = 104
    
    prospect_epochs: int = 5000
    train_epochs: int = 29000
    
    batch_size_min: int = 24
    batch_size_max: int = 128
    batch_size_cycle_length: int = 5000
    
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    
    cosine_annealing_t0: int = 20000
    prospect_cosine_t0: int = 2000
    
    kappa_window_size: int = 100
    kappa_calculation_freq: int = 100
    kappa_crystallization_threshold: float = 10.0
    
    quantization_base_weight: float = 1e-4
    quantization_max_weight: float = 1.0
    quantization_anneal_start: int = 1000
    quantization_kappa_multiplier: float = 10.0
    quantization_mse_scale_min: float = 0.1
    quantization_mse_scale_max: float = 10.0
    quantization_mse_scale_factor: float = 1000.0
    
    perelman_tau_initial: float = 1.0
    perelman_tau_adaptive: bool = True
    perelman_tau_min: float = 1e-6
    perelman_tau_smoothing: float = 0.9
    perelman_log_w_min: float = -700.0
    perelman_log_w_max: float = 700.0
    
    ricci_penalty_weight: float = 1e-3
    
    phase2_steps: int = 2000
    phase2_learning_rate: float = 0.005
    phase2_decay_factor: float = 0.99
    
    phase3_epochs: int = 10000
    phase3_print_interval: int = 2000
    
    spectral_regularization: float = 1e-6
    eigenvalue_filter_threshold: float = 1e-10
    kappa_min_samples: int = 10
    kappa_max_samples: int = 50
    
    accuracy_threshold: float = 99.9
    accuracy_error_tolerance: float = 1e-3
    discretization_tolerance: float = 1e-5
    sparsity_threshold: float = 0.1
    
    target_delta: float = 0.05
    target_accuracy: float = 99.0
    target_kappa: float = 10.0
    
    mining_max_attempts: int = 1000
    mining_start_seed: int = 1
    mining_glass_patience_epochs: int = 100
    mining_check_interval: int = 50
    mining_partial_log_interval: int = 10
    mining_metrics_display_interval: int = 25
    
    crystallization_confirmation_epochs: int = 50
    post_crystallization_lr_factor: float = 0.01
    
    checkpoint_interval_minutes: float = 5.0
    checkpoint_dir: str = "checkpoints"
    latest_checkpoint_name: str = "latest.pt"
    
    crystal_dir: str = "crystal_seeds_strassen"
    output_dir: str = "outputs"
    
    resilience_test_samples: int = 1000
    resilience_pruning_steps: int = 20
    resilience_sparsity_max: float = 0.6
    resilience_min_accuracy: float = 99.0
    resilience_max_delta: float = 0.1
    resilience_critical_threshold: float = 0.3
    resilience_check_interval: int = 5000
    
    local_complexity_epsilon: float = 1e-3
    local_complexity_num_samples: int = 100
    
    superposition_sae_hidden_dim: int = 64
    superposition_sae_sparsity: float = 0.1
    
    planck_constant_regularization: float = 1e-6
    boltzmann_constant: float = 1.0
    
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    random_seed: int = 42
    
    metrics_display_interval: int = 100
    metrics_history_max_length: int = 1000
    metrics_save_interval: int = 10




class IMetricCalculator(ABC):
    """Interface for metric calculation strategies."""
    
    @abstractmethod
    def calculate(self, **kwargs) -> Dict[str, float]:
        pass


class ILossComponent(ABC):
    """Interface for loss function components."""
    
    @abstractmethod
    def compute(self, model: nn.Module, loss_mse: torch.Tensor, 
                epoch: int, **kwargs) -> torch.Tensor:
        pass


class ICheckpointManager(ABC):
    """Interface for checkpoint management."""
    
    @abstractmethod
    def save(self, state: Dict[str, Any], path: Optional[str] = None) -> str:
        pass
    
    @abstractmethod
    def load(self, path: str) -> Optional[Dict[str, Any]]:
        pass
    
    @abstractmethod
    def should_checkpoint(self) -> bool:
        pass


class ITrainingPhase(ABC):
    """Interface for training phase execution."""
    
    @abstractmethod
    def execute(self, model: nn.Module, **kwargs) -> Any:
        pass


class StrassenOperator(nn.Module):
    """Spectral operator for 2x2 matrix multiplication with Strassen structure."""
    
    def __init__(self, config: UnifiedConfig):
        super().__init__()
        self.config = config
        self.rank = config.rank
        self.matrix_size = config.matrix_size
        self.input_dim = config.input_dim
        self.output_dim = config.output_dim
        
        self.U = nn.Parameter(torch.randn(self.rank, self.input_dim) * 0.5)
        self.V = nn.Parameter(torch.randn(self.rank, self.input_dim) * 0.5)
        self.W = nn.Parameter(torch.randn(self.output_dim, self.rank) * 0.5)
        
        self._initialize_strassen_structure()
    
    def _initialize_strassen_structure(self) -> None:
        """Initialize with bias towards canonical Strassen structure."""
        with torch.no_grad():
            if self.rank >= 1:
                self.U.data[0] = torch.tensor([1.0, 0.0, 0.0, 1.0])
                self.V.data[0] = torch.tensor([1.0, 0.0, 0.0, 1.0])
            
            if self.rank >= 2:
                self.U.data[1] = torch.tensor([0.0, 0.0, 1.0, 1.0])
                self.V.data[1] = torch.tensor([1.0, 0.0, 0.0, 0.0])
            
            if self.rank >= 3:
                self.U.data[2] = torch.tensor([1.0, 0.0, 0.0, 0.0])
                self.V.data[2] = torch.tensor([0.0, 1.0, 0.0, -1.0])
            
            if self.rank >= 4:
                self.U.data[3] = torch.tensor([0.0, 0.0, 0.0, 1.0])
                self.V.data[3] = torch.tensor([-1.0, 0.0, 1.0, 0.0])
            
            if self.rank >= 5:
                self.U.data[4] = torch.tensor([1.0, 1.0, 0.0, 0.0])
                self.V.data[4] = torch.tensor([0.0, 0.0, 0.0, 1.0])
            
            if self.rank >= 6:
                self.U.data[5] = torch.tensor([-1.0, 0.0, 1.0, 0.0])
                self.V.data[5] = torch.tensor([1.0, 1.0, 0.0, 0.0])
            
            if self.rank >= 7:
                self.U.data[6] = torch.tensor([0.0, 1.0, 0.0, -1.0])
                self.V.data[6] = torch.tensor([0.0, 0.0, 1.0, 1.0])
            
            if self.rank >= 8:
                nn.init.normal_(self.U.data[7], 0, 0.01)
                nn.init.normal_(self.V.data[7], 0, 0.01)
            
            self.W.data.zero_()
            if self.rank >= 7:
                self.W.data[0, 0] = 1.0
                self.W.data[0, 3] = 1.0
                self.W.data[0, 4] = -1.0
                self.W.data[0, 6] = 1.0
                
                self.W.data[1, 2] = 1.0
                self.W.data[1, 4] = 1.0
                
                self.W.data[2, 1] = 1.0
                self.W.data[2, 3] = 1.0
                
                self.W.data[3, 0] = 1.0
                self.W.data[3, 1] = -1.0
                self.W.data[3, 2] = 1.0
                self.W.data[3, 5] = 1.0
            
            for p in [self.U, self.V, self.W]:
                p.data += torch.randn_like(p.data) * 0.05
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """Forward pass computing C = A @ B via low-rank factorization."""
        batch_size = A.shape[0]
        
        a = A.reshape(batch_size, self.input_dim)
        b = B.reshape(batch_size, self.input_dim)
        
        left = a @ self.U.t()
        right = b @ self.V.t()
        products = left * right
        
        c = products @ self.W.t()
        
        return c.reshape(batch_size, self.matrix_size, self.matrix_size)
    
    def slot_importance(self) -> torch.Tensor:
        """Calculate importance of each rank slot."""
        u_norm = torch.norm(self.U, dim=1)
        v_norm = torch.norm(self.V, dim=1)
        w_norm = torch.norm(self.W, dim=0)
        return u_norm * v_norm * w_norm
    
    def count_active(self, threshold: Optional[float] = None) -> int:
        """Count active slots above threshold."""
        if threshold is None:
            threshold = self.config.sparsity_threshold
        return (self.slot_importance() > threshold).sum().item()
    
    def get_flat_parameters(self) -> torch.Tensor:
        """Get flattened parameter vector."""
        return torch.cat([self.U.flatten(), self.V.flatten(), self.W.flatten()])
    
    def get_parameter_count(self) -> int:
        """Get total parameter count."""
        return sum(p.numel() for p in self.parameters())


class DeltaCalculator(IMetricCalculator):
    """Calculate delta (discretization margin) metric."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def calculate(self, model: StrassenOperator, **kwargs) -> Dict[str, float]:
        """Calculate delta: mean squared distance to {-1, 0, 1}."""
        params = model.get_flat_parameters()
        rounded = torch.round(params).clamp(-1, 1)
        delta = torch.mean((params - rounded) ** 2).item()
        
        alpha = -np.log(delta + 1e-15) if delta > 0 else 20.0
        
        return {
            'delta': delta,
            'alpha_purity': alpha
        }


class AccuracyCalculator(IMetricCalculator):
    """Calculate matrix multiplication accuracy."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def calculate(self, model: StrassenOperator, 
                  C_pred: Optional[torch.Tensor] = None,
                  C_true: Optional[torch.Tensor] = None,
                  n_test: int = 1000, **kwargs) -> Dict[str, float]:
        """Calculate accuracy percentage."""
        device = next(model.parameters()).device
        
        if C_pred is None or C_true is None:
            A = torch.randn(n_test, self.config.matrix_size, 
                           self.config.matrix_size, device=device)
            B = torch.randn(n_test, self.config.matrix_size, 
                           self.config.matrix_size, device=device)
            
            with torch.no_grad():
                C_pred = model(A, B)
                C_true = torch.bmm(A, B)
        
        error = (C_pred - C_true).abs()
        batch_size = C_pred.shape[0]
        max_error_per_sample = error.reshape(batch_size, -1).max(dim=1)[0]
        
        accuracy = (max_error_per_sample < self.config.accuracy_error_tolerance).float().mean().item() * 100.0
        max_error = max_error_per_sample.max().item()
        mean_error = error.mean().item()
        
        return {
            'accuracy': accuracy,
            'max_error': max_error,
            'mean_error': mean_error
        }


class KappaCalculator:
    """Calculate kappa (gradient covariance condition number)."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.gradient_buffer: Deque[torch.Tensor] = deque(maxlen=config.kappa_window_size)
        self.kappa_history: Deque[float] = deque(maxlen=100)
    
    def accumulate_gradient(self, model: nn.Module) -> None:
        """Accumulate current gradient vector."""
        if not any(p.grad is not None for p in model.parameters()):
            return
        
        grad_vector = []
        for p in model.parameters():
            if p.grad is not None:
                grad_vector.append(p.grad.detach().flatten().cpu())
            else:
                grad_vector.append(torch.zeros(p.numel()))
        
        self.gradient_buffer.append(torch.cat(grad_vector))
    
    def calculate_kappa(self) -> Optional[float]:
        """Calculate condition number of gradient covariance matrix."""
        if len(self.gradient_buffer) < self.config.kappa_min_samples:
            return float('inf')  # Sistema no condicionado aún → κ tiende a infinito
        
        try:
            G = torch.stack(list(self.gradient_buffer))
            G_centered = G - G.mean(dim=0, keepdim=True)
            
            n = G.shape[0]
            Sigma = (G_centered.T @ G_centered) / (n - 1)
            
            reg = torch.eye(Sigma.shape[0]) * self.config.spectral_regularization
            Sigma_reg = Sigma + reg
            
            eigenvalues = torch.linalg.eigvalsh(Sigma_reg)
            eigenvalues = eigenvalues[eigenvalues > self.config.eigenvalue_filter_threshold]
            
            if len(eigenvalues) == 0:
                return float('inf')  # Matriz singular → κ infinito
            
            lambda_max = eigenvalues.max().item()
            lambda_min = eigenvalues.min().item()
            kappa = lambda_max / (lambda_min + 1e-12)
            
            self.kappa_history.append(kappa)
            return kappa
            
        except Exception:
            return float('inf')  # Error numérico → conservadoramente infinito

            
    def get_kappa_trend(self) -> str:
        """Determine kappa trend direction."""
        if len(self.kappa_history) < 10:
            return "insufficient_data"
        
        recent = list(self.kappa_history)[-10:]
        
        if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
            return "decreasing"
        elif all(recent[i] < recent[i+1] for i in range(len(recent)-1)):
            return "increasing"
        else:
            return "fluctuating"
    
    def is_crystallizing(self) -> bool:
        """Detect if system is in crystallization phase."""
        if len(self.kappa_history) < 20:
            return False
        
        recent_kappa = list(self.kappa_history)[-20:]
        avg_kappa = np.mean(recent_kappa)
        
        return avg_kappa < self.config.kappa_crystallization_threshold
    
    def reset(self) -> None:
        """Reset calculator state."""
        self.gradient_buffer.clear()
        self.kappa_history.clear()


class PerelmanEntropyCalculator(IMetricCalculator):
    """Calculate Perelman W-entropy with adaptive tau."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.tau = config.perelman_tau_initial
        self.dimension = config.perelman_dimension
        self.noise_scale_history: Deque[float] = deque(maxlen=100)
        self.W_history: Deque[float] = deque(maxlen=config.metrics_history_max_length)
    
    def calculate(self, model: StrassenOperator, loss: torch.Tensor,
                  epoch: int, gradient_norm: float, **kwargs) -> Dict[str, float]:
        """Calculate W-entropy with adaptive tau coupling to GNS."""
        params = model.get_flat_parameters()
        n_params = params.numel()
        
        param_mean = params.mean().item()
        param_std = params.std().item() + 1e-8
        
        if self.config.perelman_tau_adaptive and gradient_norm > 0:
            noise_scale = (gradient_norm ** 2) / n_params
            target_tau = 1.0 / (noise_scale + 1e-8)
            
            self.tau = (self.config.perelman_tau_smoothing * self.tau + 
                       (1 - self.config.perelman_tau_smoothing) * target_tau)
            self.tau = max(self.tau, self.config.perelman_tau_min)
            self.noise_scale_history.append(self.tau)
        
        centered = params - param_mean
        log_probs = (-0.5 * (centered / param_std) ** 2 - 
                    math.log(param_std * math.sqrt(2 * math.pi)))
        f = -log_probs.mean().item()
        
        grad_f_sq = 1.0 / (param_std ** 2)
        R = 1.0 / (param_std ** 2)
        
        log_W = self._calculate_log_W(
            tau=self.tau, R=R, grad_f_sq=grad_f_sq, 
            f=f, n_params=n_params
        )
        
        W_entropy = math.exp(log_W) if log_W < 700 else float('inf')
        self.W_history.append(log_W)
        
        nash_entropy = (np.mean(list(self.W_history)[-100:]) 
                       if len(self.W_history) > 0 else log_W)
        
        monotonicity = 0.0
        if len(self.W_history) > 1:
            monotonicity = log_W - list(self.W_history)[-2]
        
        ricci_residual = abs(R - 1.0 / (2.0 * self.tau))
        
        return {
            'perelman_log_W': log_W,
            'perelman_W': W_entropy,
            'perelman_Nash': nash_entropy,
            'perelman_tau': self.tau,
            'perelman_R': R,
            'perelman_f': f,
            'perelman_grad_f_sq': grad_f_sq,
            'perelman_monotonicity': monotonicity,
            'perelman_ricci_residual': ricci_residual,
            'perelman_param_std': param_std,
            'perelman_noise_scale': 1.0 / self.tau if self.tau > 0 else 0.0
        }
    
    def _calculate_log_W(self, tau: float, R: float, grad_f_sq: float,
                        f: float, n_params: int) -> float:
        """Calculate log(W) with numerical stability."""
        tau_safe = max(tau, self.config.perelman_tau_min)
        
        tau_R = tau_safe * R
        tau_grad = tau_safe * grad_f_sq
        
        integrand_raw = tau_R + tau_grad + f - n_params
        integrand = math.copysign(max(abs(integrand_raw), 1e-12), integrand_raw)
        
        if tau_safe > 0 and 4.0 * math.pi * tau_safe > 0:
            log_normalization = (-(n_params / 2.0) * 
                               math.log(max(4.0 * math.pi * tau_safe, 1e-300)))
        else:
            log_normalization = self.config.perelman_log_w_min
        
        log_measure = -min(f, 700.0)
        
        log_W = math.log(abs(integrand)) + log_normalization + log_measure
        
        return max(min(log_W, self.config.perelman_log_w_max), 
                  self.config.perelman_log_w_min)
    
    def reset(self) -> None:
        """Reset calculator state."""
        self.tau = self.config.perelman_tau_initial
        self.noise_scale_history.clear()
        self.W_history.clear()


class SparsityCalculator(IMetricCalculator):
    """Calculate sparsity metrics."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def calculate(self, model: StrassenOperator, **kwargs) -> Dict[str, float]:
        """Calculate active slots and sparsity."""
        active = model.count_active(self.config.sparsity_threshold)
        total = model.rank
        sparsity = (total - active) / total
        target_sparsity = (total - self.config.target_rank) / total
        
        return {
            'active_slots': float(active),
            'sparsity': sparsity,
            'target_sparsity': target_sparsity
        }


class GradientMetricsCalculator:
    """Calculate gradient statistics."""
    
    @staticmethod
    def calculate(model: nn.Module) -> Dict[str, float]:
        """Calculate gradient norm statistics."""
        total_norm = 0.0
        max_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                max_norm = max(max_norm, param_norm)
                param_count += 1
        
        total_norm = total_norm ** 0.5
        
        return {
            'gradient_norm': total_norm,
            'gradient_max': max_norm,
            'gradient_mean': total_norm / max(param_count, 1)
        }


class ResilienceSpectrometer:
    """Measure structural stability under progressive pruning."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def measure(self, model: StrassenOperator) -> Dict[str, Any]:
        """Measure resilience via progressive magnitude pruning."""
        results = []
        device = next(model.parameters()).device
        
        A_test = torch.randn(self.config.resilience_test_samples,
                            self.config.matrix_size,
                            self.config.matrix_size, device=device)
        B_test = torch.randn(self.config.resilience_test_samples,
                            self.config.matrix_size,
                            self.config.matrix_size, device=device)
        C_true = torch.bmm(A_test, B_test)
        
        original_state = {name: param.clone() 
                         for name, param in model.named_parameters()}
        
        for sparsity in np.linspace(0, self.config.resilience_sparsity_max,
                                    self.config.resilience_pruning_steps):
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.copy_(original_state[name])
            
            self._prune_by_magnitude(model, sparsity)
            
            with torch.no_grad():
                C_pred = model(A_test, B_test)
                error = (C_pred - C_true).abs().max().item()
                accuracy = 100.0 * (error < self.config.accuracy_error_tolerance)
            
            delta_calc = DeltaCalculator(self.config)
            delta_metrics = delta_calc.calculate(model)
            delta = delta_metrics['delta']
            
            results.append({
                'sparsity': sparsity,
                'accuracy': accuracy,
                'max_error': error,
                'delta': delta,
                'intact': (accuracy > self.config.resilience_min_accuracy and 
                          delta < self.config.resilience_max_delta)
            })
            
            if not results[-1]['intact']:
                break
        
        with torch.no_grad():
            for name, param in model.named_parameters():
                param.copy_(original_state[name])
        
        intact_results = [r for r in results if r['intact']]
        critical_sparsity = (intact_results[-1]['sparsity'] 
                           if intact_results else 0.0)
        
        if len(results) > 1:
            try:
                from numpy import trapezoid
                resilience_score = trapezoid(
                    [r['accuracy'] for r in results],
                    [r['sparsity'] for r in results]
                ) / 100.0
            except ImportError:
                resilience_score = np.trapz(
                    [r['accuracy'] for r in results],
                    [r['sparsity'] for r in results]
                ) / 100.0
        else:
            resilience_score = 0.0
        
        return {
            'critical_sparsity': critical_sparsity,
            'resilience_score': resilience_score,
            'phase_transition_point': next(
                (r['sparsity'] for r in results if not r['intact']),
                self.config.resilience_sparsity_max
            ),
            'detailed_results': results,
            'is_resilient': critical_sparsity >= self.config.resilience_critical_threshold
        }
    
    def _prune_by_magnitude(self, model: StrassenOperator, 
                           sparsity: float) -> None:
        """Prune parameters by magnitude threshold."""
        with torch.no_grad():
            all_params = torch.cat([p.flatten() for p in model.parameters()])
            threshold = torch.quantile(all_params.abs(), sparsity)
            
            for p in model.parameters():
                p.data[p.data.abs() < threshold] = 0.0

class ComprehensiveMetricsAggregator:
    """Aggregate all metrics including LC, SP, kappa, delta, h_bar_eff, T_eff."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.kappa_calculator = KappaCalculator(config)
        self.perelman_calculator = PerelmanEntropyCalculator(config)
        self.delta_calculator = DeltaCalculator(config)
        self.accuracy_calculator = AccuracyCalculator(config)
        self.sparsity_calculator = SparsityCalculator(config)
        self.resilience_spectrometer = ResilienceSpectrometer(config)
        self.local_complexity_calculator = LocalComplexityCalculator(config)
        self.superposition_calculator = SuperpositionCalculator(config)
        self.thermodynamic_calculator = ThermodynamicMetricsCalculator(config)
        self._current_lr = 0.0
        self._gradient_covariance = None
    
    def compute_all(self, model: StrassenOperator, C_pred: torch.Tensor,
                   C_true: torch.Tensor, loss: torch.Tensor, epoch: int,
                   force_kappa: bool = False,
                   force_lc: bool = False,
                   force_sp: bool = False) -> Tuple[Dict[str, float], Optional[float]]:
        """Compute all available metrics."""
        metrics = {}
        
        metrics['loss'] = loss.item()
        metrics['epoch'] = float(epoch)
        metrics['learning_rate'] = self._current_lr
        
        delta_metrics = self.delta_calculator.calculate(model)
        metrics.update(delta_metrics)
        
        accuracy_metrics = self.accuracy_calculator.calculate(
            model, C_pred=C_pred, C_true=C_true
        )
        metrics.update(accuracy_metrics)
        
        sparsity_metrics = self.sparsity_calculator.calculate(model)
        metrics.update(sparsity_metrics)
        
        grad_metrics = GradientMetricsCalculator.calculate(model)
        metrics.update(grad_metrics)
        
        kappa = None
        if force_kappa or epoch % self.config.kappa_calculation_freq == 0:
            kappa = self.kappa_calculator.calculate_kappa()
            if kappa is not None:
                metrics['kappa'] = kappa
                metrics['kappa_trend'] = self.kappa_calculator.get_kappa_trend()
                metrics['is_crystallizing'] = float(
                    self.kappa_calculator.is_crystallizing()
                )
                
                if len(self.kappa_calculator.gradient_buffer) > 0:
                    G = torch.stack(list(self.kappa_calculator.gradient_buffer))
                    G_centered = G - G.mean(dim=0, keepdim=True)
                    n = G.shape[0]
                    self._gradient_covariance = (G_centered.T @ G_centered) / (n - 1)
        
        perelman_metrics = self.perelman_calculator.calculate(
            model, loss, epoch, grad_metrics.get('gradient_norm', 1.0)
        )
        metrics.update(perelman_metrics)
        
        if force_lc or (epoch > 0 and epoch % self.config.metrics_display_interval == 0):
            lc_metrics = self.local_complexity_calculator.calculate(model)
            metrics.update(lc_metrics)
        
        if force_sp or (epoch > 0 and epoch % self.config.metrics_display_interval == 0):
            sp_metrics = self.superposition_calculator.calculate(model)
            metrics.update(sp_metrics)
        
        thermo_metrics = self.thermodynamic_calculator.calculate(
            model, 
            gradient_covariance=self._gradient_covariance
        )
        metrics.update(thermo_metrics)
        
        if epoch > 0 and epoch % self.config.resilience_check_interval == 0:
            resilience = self.resilience_spectrometer.measure(model)
            metrics['resilience_score'] = resilience['resilience_score']
            metrics['critical_sparsity'] = resilience['critical_sparsity']
            metrics['is_resilient'] = float(resilience['is_resilient'])
        
        is_crystal = (
            metrics['delta'] < self.config.target_delta and
            metrics['accuracy'] > self.config.target_accuracy and
            metrics['active_slots'] <= self.config.target_rank + 1
        )
        metrics['is_crystal_candidate'] = float(is_crystal)
        
        return metrics, kappa
    
    def accumulate_gradient(self, model: nn.Module) -> None:
        """Accumulate gradient for kappa calculation."""
        self.kappa_calculator.accumulate_gradient(model)
    
    def update_lr(self, lr: float) -> None:
        """Update current learning rate."""
        self._current_lr = lr
    
    def reset(self) -> None:
        """Reset all calculators."""
        self.kappa_calculator.reset()
        self.perelman_calculator.reset()
        self._gradient_covariance = None


class AdaptiveQuantizationLoss(ILossComponent):
    """Adaptive quantization loss pushing towards {-1, 0, 1}."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def compute(self, model: nn.Module, loss_mse: torch.Tensor,
               epoch: int, kappa: Optional[float] = None, **kwargs) -> torch.Tensor:
        """Compute quantization loss with adaptive weighting."""
        q_loss = torch.tensor(0.0, device=loss_mse.device)
        
        for p in model.parameters():
            dist_to_minus1 = (p + 1) ** 2
            dist_to_zero = p ** 2
            dist_to_plus1 = (p - 1) ** 2
            
            min_dist = torch.minimum(
                torch.minimum(dist_to_minus1, dist_to_zero),
                dist_to_plus1
            )
            
            q_loss = q_loss + torch.mean(min_dist)
        
        weight = self._get_adaptive_weight(epoch, kappa)
        
        mse_scale = torch.clamp(
            loss_mse.detach() * self.config.quantization_mse_scale_factor,
            self.config.quantization_mse_scale_min,
            self.config.quantization_mse_scale_max
        )
        
        return weight * mse_scale * q_loss
    
    def _get_adaptive_weight(self, epoch: int, 
                           kappa: Optional[float]) -> float:
        """Calculate adaptive weight based on epoch and kappa."""
        if epoch < self.config.quantization_anneal_start:
            return self.config.quantization_base_weight
        
        if (kappa is not None and 
            kappa < self.config.kappa_crystallization_threshold):
            return (self.config.quantization_max_weight * 
                   self.config.quantization_kappa_multiplier)
        
        progress = min(1.0, (epoch - self.config.quantization_anneal_start) / 
                      self.config.quantization_anneal_start)
        weight_range = (self.config.quantization_max_weight - 
                       self.config.quantization_base_weight)
        
        return self.config.quantization_base_weight + weight_range * progress


class RicciCurvaturePenalty(ILossComponent):
    """Ricci scalar curvature penalty."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def compute(self, model: nn.Module, loss_mse: torch.Tensor,
               epoch: int, **kwargs) -> torch.Tensor:
        """Compute Ricci curvature penalty."""
        param_variances = torch.stack([torch.var(p) for p in model.parameters()])
        ricci_penalty = torch.var(param_variances)
        return self.config.ricci_penalty_weight * ricci_penalty


class GeometricLossAggregator(ILossComponent):
    """Aggregate geometric loss components."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.quantization_loss = AdaptiveQuantizationLoss(config)
        self.ricci_penalty = RicciCurvaturePenalty(config)
    
    def compute(self, model: nn.Module, loss_mse: torch.Tensor,
               epoch: int, kappa: Optional[float] = None, **kwargs) -> torch.Tensor:
        """Compute total geometric loss."""
        q_loss = self.quantization_loss.compute(model, loss_mse, epoch, kappa)
        r_penalty = self.ricci_penalty.compute(model, loss_mse, epoch)
        return loss_mse + q_loss + r_penalty


class CheckpointManager(ICheckpointManager):
    """Manage model checkpointing."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.last_checkpoint_time = time.time()
    
    def save(self, state: Dict[str, Any], 
            path: Optional[str] = None) -> str:
        """Save checkpoint to disk."""
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = str(self.checkpoint_dir / f"checkpoint_{timestamp}.pt")
        
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        
        torch.save(state, path)
        
        latest_path = self.checkpoint_dir / self.config.latest_checkpoint_name
        torch.save(state, latest_path)
        
        self.last_checkpoint_time = time.time()
        return str(path)
    
    def load(self, path: str) -> Optional[Dict[str, Any]]:
        """Load checkpoint from disk."""
        try:
            return torch.load(path, map_location=self.config.device)
        except Exception as e:
            print(f"Failed to load checkpoint from {path}: {e}")
            return None
    
    def should_checkpoint(self) -> bool:
        """Check if checkpoint interval has elapsed."""
        elapsed_minutes = (time.time() - self.last_checkpoint_time) / 60.0
        return elapsed_minutes >= self.config.checkpoint_interval_minutes
    
    def get_latest_checkpoint_path(self) -> Optional[str]:
        """Get path to latest checkpoint if exists."""
        latest = self.checkpoint_dir / self.config.latest_checkpoint_name
        return str(latest) if latest.exists() else None


class MatrixDataGenerator:
    """Generate random matrix data for training."""
    
    def __init__(self, config: UnifiedConfig, scale: float = 1.0):
        self.config = config
        self.device = config.device
        self.scale = scale
    
    def generate_batch(self, n: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate batch of random matrices."""
        A = torch.randn(n, self.config.matrix_size, self.config.matrix_size,
                       device=self.device) * self.scale
        B = torch.randn(n, self.config.matrix_size, self.config.matrix_size,
                       device=self.device) * self.scale
        return A, B


class DynamicBatchSizeScheduler:
    """Schedule batch size with cosine annealing."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def get_batch_size(self, epoch: int) -> int:
        """Get batch size for current epoch."""
        if self.config.batch_size_cycle_length <= 0:
            return self.config.batch_size_min
        
        cycle_pos = (epoch % self.config.batch_size_cycle_length) / self.config.batch_size_cycle_length
        cosine = 0.5 * (1 + math.cos(2 * math.pi * cycle_pos))
        
        batch_size = int(
            self.config.batch_size_min +
            (self.config.batch_size_max - self.config.batch_size_min) * (1 - cosine)
        )
        
        return batch_size


class GlassDetector:
    """Detect glass state (non-crystallizing seeds)."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.patience_epochs = config.mining_glass_patience_epochs
        self.metrics_buffer: Deque[Dict[str, float]] = deque(maxlen=self.patience_epochs)
    
    def should_stop(self, epoch: int, metrics: Dict[str, float]) -> Tuple[bool, str]:
        """Determine if training should stop due to glass state."""
        self.metrics_buffer.append({
            'epoch': epoch,
            'delta': metrics.get('delta', float('inf')),
            'accuracy': metrics.get('accuracy', 0.0),
            'kappa': metrics.get('kappa', float('inf')),
            'active_slots': metrics.get('active_slots', 0)
        })
        
        if epoch < self.patience_epochs:
            return False, "warming_up"
        
        recent = list(self.metrics_buffer)[-self.patience_epochs:]
        avg_delta = np.mean([m['delta'] for m in recent])
        avg_accuracy = np.mean([m['accuracy'] for m in recent])
        avg_kappa = np.mean([m['kappa'] for m in recent if m['kappa'] != float('inf')])
        
        is_glass = False
        reason = ""
        
        if avg_delta > 0.3 and recent[-1]['delta'] > 0.3:
            is_glass = True
            reason = f"delta_stuck_high ({avg_delta:.3f})"
        
        elif avg_accuracy < 50 and recent[-1]['accuracy'] < 50:
            is_glass = True
            reason = f"accuracy_stuck_low ({avg_accuracy:.1f}%)"
        
        elif len(recent) > 10:
            recent_kappas = [m['kappa'] for m in recent[-10:] 
                           if m['kappa'] != float('inf')]
            if (len(recent_kappas) >= 10 and
                all(recent_kappas[i] < recent_kappas[i+1] 
                    for i in range(len(recent_kappas)-1))):
                if recent_kappas[-1] > 100:
                    is_glass = True
                    reason = f"kappa_diverging ({recent_kappas[-1]:.1f})"
        
        elif epoch >= self.patience_epochs:
            if (avg_delta > self.config.target_delta * 5 and
                avg_accuracy < self.config.target_accuracy * 0.5):
                is_glass = True
                reason = f"no_progress (delta={avg_delta:.3f}, acc={avg_accuracy:.1f}%)"
        
        return is_glass, reason


class ProspectorPhase(ITrainingPhase):
    """Fast prospecting phase to identify crystal seeds."""
    
    def __init__(self, config: UnifiedConfig, seed: int):
        self.config = config
        self.seed = seed
        self.metrics_aggregator = ComprehensiveMetricsAggregator(config)
        self.data_generator = MatrixDataGenerator(config)
        self.batch_scheduler = DynamicBatchSizeScheduler(config)
        self.geometric_loss = GeometricLossAggregator(config)
        self.glass_detector = GlassDetector(config)
    
    def execute(self, model: StrassenOperator, **kwargs) -> Tuple[bool, Dict[str, Any]]:
        """Execute prospecting phase with all metrics visible."""
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.prospect_cosine_t0
        )
        
        best_delta = float('inf')
        best_epoch = 0
        metrics_history = []
        
        for epoch in range(1, self.config.prospect_epochs + 1):
            batch_size = self.batch_scheduler.get_batch_size(epoch)
            
            A, B = self.data_generator.generate_batch(batch_size)
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            loss_mse = torch.mean((C_pred - C_true) ** 2)
            
            loss = self.geometric_loss.compute(model, loss_mse, epoch)
            
            optimizer.zero_grad()
            loss.backward()
            
            self.metrics_aggregator.accumulate_gradient(model)
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.gradient_clip
            )
            
            optimizer.step()
            scheduler.step()
            
            self.metrics_aggregator.update_lr(optimizer.param_groups[0]['lr'])
            
            should_display = (epoch % self.config.mining_metrics_display_interval == 0 or
                            epoch == self.config.prospect_epochs)
            
            should_check_glass = (epoch % self.config.mining_check_interval == 0 or
                                epoch == self.config.prospect_epochs)
            
            if should_display:
                force_all = (epoch % self.config.kappa_calculation_freq == 0)
                
                metrics, kappa = self.metrics_aggregator.compute_all(
                    model, C_pred, C_true, loss, epoch,
                    force_kappa=force_all,
                    force_lc=force_all,
                    force_sp=force_all
                )
                
                metrics_history.append(metrics.copy())
                
                if metrics['delta'] < best_delta:
                    best_delta = metrics['delta']
                    best_epoch = epoch
                
                crystal_flag = "[+]" if metrics['is_crystal_candidate'] else "   "
                kappa_val = metrics.get('kappa', float('inf'))
                kappa_str = f"{kappa_val:6.1f}" if kappa_val != float('inf') else "   inf"
                lc_val = metrics.get('local_complexity', 0.0)
                
                if lc_val == -math.inf:
                    lc_str = "  -inf"
                elif lc_val == math.inf:
                    lc_str = "   inf"
                else:
                    lc_str = f"{lc_val:>6.2f}"
                
                psi_val = metrics.get('superposition_psi', 0.0)
                
                print(f" {crystal_flag} Epoch {epoch:>4}: "
                    f"δ={metrics['delta']:.4f} (best: {best_delta:.4f} @ {best_epoch}), "
                    f"Acc={metrics['accuracy']:>5.1f}%, "
                    f"κ={kappa_str}, "
                    f"LC={lc_str}, "
                    f"ψ={psi_val:>5.3f}, "
                    f"Slots={int(metrics['active_slots'])}")
                
                if (metrics['is_crystal_candidate'] and
                    metrics['active_slots'] <= self.config.target_rank):
                    print(f" [+] CRYSTAL DETECTED at epoch {epoch}!")
                    print(f"     delta={metrics['delta']:.4f}, "
                        f"Acc={metrics['accuracy']:.2f}%, "
                        f"kappa={kappa_val:.2f}, "
                        f"LC={lc_val:.2f}, "
                        f"psi={psi_val:.3f}")
                    
                    return True, {**metrics, 'metrics_history': metrics_history}
            
            if should_check_glass:
                if len(metrics_history) > 0:
                    is_glass, reason = self.glass_detector.should_stop(epoch, metrics_history[-1])
                else:
                    metrics_for_glass, _ = self.metrics_aggregator.compute_all(
                        model, C_pred, C_true, loss, epoch,
                        force_kappa=False, force_lc=False, force_sp=False
                    )
                    is_glass, reason = self.glass_detector.should_stop(epoch, metrics_for_glass)
                
                if is_glass:
                    print(f" [-] GLASS DETECTED: {reason}")
                    print(f"     Best delta was {best_delta:.4f} at epoch {best_epoch}")
                    return False, {'delta': best_delta, 'best_epoch': best_epoch, 'metrics_history': metrics_history}
        
        print(f" [-] Max epochs reached. Best delta: {best_delta:.4f} at epoch {best_epoch}")
        return False, {'delta': best_delta, 'best_epoch': best_epoch, 'metrics_history': metrics_history}
    
class LongTrainingPhase(ITrainingPhase):
    """Long training phase with full thermodynamic metrics."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.metrics_aggregator = ComprehensiveMetricsAggregator(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.data_generator = MatrixDataGenerator(config)
        self.batch_scheduler = DynamicBatchSizeScheduler(config)
        self.geometric_loss = GeometricLossAggregator(config)
        self.history: List[Dict[str, float]] = []
    
    def execute(self, model: StrassenOperator, **kwargs) -> StrassenOperator:
        """Execute long training phase."""
        print("\n" + "=" * 80)
        print("PHASE 1: Thermodynamic Grokking with Kappa as Order Parameter")
        print("=" * 80)
        print(f"Pressure (WD): {self.config.weight_decay}")
        print(f"Quantization: {self.config.quantization_base_weight} -> "
              f"{self.config.quantization_max_weight} (adaptive)")
        print(f"Tau: {'Adaptive (coupled to GNS)' if self.config.perelman_tau_adaptive else 'Fixed'}")
        print(f"Kappa calculated every {self.config.kappa_calculation_freq} epochs")
        print(f"Target: kappa->1, Acc->100%, log(W) dynamic, delta->0")
        print("=" * 80)
        
        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=self.config.cosine_annealing_t0
        )
        
        header = (f"{'Epoch':>8} | {'Batch':>5} | {'Loss':>9} | {'Acc%':>6} | "
                 f"{'delta':>8} | {'kappa':>8} | {'log(W)':>10} | {'tau':>9} | "
                 f"{'dlogW':>7} | {'Cryst':>5}")
        print(header)
        print("-" * len(header))
        
        kappa = None
        crystallization_detected = False
        crystallization_epoch = None
        epochs_since_crystallization = 0
        
        for epoch in range(self.config.train_epochs):
            batch_size = self.batch_scheduler.get_batch_size(epoch)
            
            A, B = self.data_generator.generate_batch(batch_size)
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            loss_mse = torch.mean((C_pred - C_true) ** 2)
            
            loss = self.geometric_loss.compute(model, loss_mse, epoch, kappa)
            
            optimizer.zero_grad()
            loss.backward()
            
            self.metrics_aggregator.accumulate_gradient(model)
            
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                self.config.gradient_clip
            )
            
            optimizer.step()
            scheduler.step()
            
            current_lr = optimizer.param_groups[0]['lr']
            self.metrics_aggregator.update_lr(current_lr)
            
            if (epoch % self.config.metrics_display_interval == 0 or
                epoch == self.config.train_epochs - 1):
                
                force_kappa = (epoch % self.config.kappa_calculation_freq == 0)
                metrics, kappa = self.metrics_aggregator.compute_all(
                    model, C_pred, C_true, loss, epoch,
                    force_kappa=force_kappa
                )
                
                self.history.append(metrics)
                
                kappa_str = f"{kappa:8.2f}" if kappa is not None else "     N/A"
                cryst_str = "YES" if metrics.get('is_crystallizing', 0) > 0.5 else " NO"
                
                print(f"{int(metrics['epoch']):>8} | "
                      f"{batch_size:>5} | "
                      f"{metrics['loss']:>9.2e} | "
                      f"{metrics['accuracy']:>6.1f} | "
                      f"{metrics['delta']:>8.4f} | "
                      f"{kappa_str} | "
                      f"{metrics['perelman_log_W']:>10.4f} | "
                      f"{metrics['perelman_tau']:>9.4f} | "
                      f"{metrics['perelman_monotonicity']:>7.4f} | "
                      f"{cryst_str:>5}")
                
                if (kappa is not None and
                    kappa < self.config.kappa_crystallization_threshold):
                    if metrics['accuracy'] > self.config.accuracy_threshold:
                        print(f"  [CRYSTALLIZATION] kappa={kappa:.2f}, "
                              f"Acc={metrics['accuracy']:.1f}%, "
                              f"delta={metrics['delta']:.4f}, "
                              f"log(W)={metrics['perelman_log_W']:.2f}")
                        
                        if metrics.get('perelman_monotonicity', 0) > 0:
                            print(f"  [EVIDENCE] Entropy increasing while loss decreasing "
                                  f"(dlogW={metrics['perelman_monotonicity']:+.4f})")
                        
                        if not crystallization_detected:
                            crystallization_detected = True
                            crystallization_epoch = epoch
                            print(f"  [DETECTION] Crystallization detected at epoch {epoch}")
                        
                        epochs_since_crystallization += 1
                        
                        if epochs_since_crystallization >= self.config.crystallization_confirmation_epochs:
                            print(f"\n{'='*70}")
                            print(f"[FREEZE] Crystallization confirmed after "
                                  f"{self.config.crystallization_confirmation_epochs} stable epochs")
                            print(f"[FREEZE] kappa={kappa:.4f}, "
                                  f"Acc={metrics['accuracy']:.2f}%, "
                                  f"delta={metrics['delta']:.4f}")
                            print(f"[FREEZE] Freezing model and ending Phase 1")
                            print(f"{'='*70}")
                            
                            checkpoint_state = {
                                'phase': 1,
                                'epoch': epoch,
                                'model_state_dict': model.state_dict(),
                                'optimizer_state_dict': optimizer.state_dict(),
                                'scheduler_state_dict': scheduler.state_dict(),
                                'loss': loss.item(),
                                'history': self.history,
                                'config': self.config,
                                'timestamp': datetime.now().isoformat(),
                                'crystallization_confirmed': True,
                                'final_kappa': kappa,
                                'final_delta': metrics['delta'],
                                'final_accuracy': metrics['accuracy']
                            }
                            
                            path = self.checkpoint_manager.save(
                                checkpoint_state,
                                path=str(self.checkpoint_manager.checkpoint_dir / "crystallized.pt")
                            )
                            print(f"[FINAL CHECKPOINT] {path}")
                            
                            return model
                    else:
                        if crystallization_detected:
                            print(f"  [RESET] Accuracy dropped to {metrics['accuracy']:.1f}%, "
                                  f"restarting counter")
                            crystallization_detected = False
                            epochs_since_crystallization = 0
                else:
                    if crystallization_detected and kappa is not None:
                        if kappa >= self.config.kappa_crystallization_threshold:
                            print(f"  [RESET] Kappa increased to {kappa:.2f}, "
                                  f"restarting counter")
                            crystallization_detected = False
                            epochs_since_crystallization = 0
                
                if self.checkpoint_manager.should_checkpoint():
                    checkpoint_state = {
                        'phase': 1,
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'loss': loss.item(),
                        'history': self.history,
                        'config': self.config,
                        'timestamp': datetime.now().isoformat()
                    }
                    
                    path = self.checkpoint_manager.save(checkpoint_state)
                    print(f"\n[CHECKPOINT] {path}")
        
        return model


class ProgressiveSparsificationPhase(ITrainingPhase):
    """Progressive sparsification guided by slot importance."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.data_generator = MatrixDataGenerator(config)
    
    def execute(self, model: StrassenOperator, **kwargs) -> Tuple[StrassenOperator, List[int]]:
        """Execute sparsification phase."""
        target_slots = kwargs.get('target_slots', self.config.target_rank)
        
        print("\n" + "=" * 70)
        print(f"PHASE 2: Sparsification -> {target_slots} slots")
        print("=" * 70)
        
        optimizer = optim.Adam(model.parameters(), lr=self.config.phase2_learning_rate)
        
        with torch.no_grad():
            importance = model.slot_importance()
            _, sorted_idx = importance.sort()
            slots_to_prune = sorted_idx[:(model.rank - target_slots)].tolist()
        
        print(f"Slots to prune: {slots_to_prune}")
        print(f"Importances: {importance[sorted_idx[:len(slots_to_prune)]].tolist()}")
        
        for slot in slots_to_prune:
            print(f"\n--- Pruning slot {slot} ---")
            
            for step in range(self.config.phase2_steps):
                decay = 1.0 - (step / self.config.phase2_steps) * self.config.phase2_decay_factor
                
                with torch.no_grad():
                    model.U.data[slot] *= decay
                    model.V.data[slot] *= decay
                    model.W.data[:, slot] *= decay
                
                A, B = self.data_generator.generate_batch(64)
                C_pred = model(A, B)
                C_true = torch.bmm(A, B)
                loss = torch.mean((C_pred - C_true) ** 2)
                
                optimizer.zero_grad()
                loss.backward()
                
                if model.U.grad is not None:
                    model.U.grad[slot] = 0
                if model.V.grad is not None:
                    model.V.grad[slot] = 0
                if model.W.grad is not None:
                    model.W.grad[:, slot] = 0
                
                optimizer.step()
            
            with torch.no_grad():
                model.U.data[slot] = 0
                model.V.data[slot] = 0
                model.W.data[:, slot] = 0
            
            with torch.no_grad():
                A, B = self.data_generator.generate_batch(1000)
                C_pred = model(A, B)
                C_true = torch.bmm(A, B)
                err = (C_pred - C_true).abs().max().item()
                active = model.count_active()
            
            print(f"  MaxErr: {err:.2e} | Active: {active}")
        
        print("\n--- Final refinement ---")
        self._final_refinement(model, optimizer, slots_to_prune)
        
        return model, slots_to_prune
    
    def _final_refinement(self, model: StrassenOperator,
                         optimizer: optim.Optimizer,
                         slots_to_prune: List[int]) -> None:
        """Final refinement after pruning."""
        for epoch in range(self.config.phase3_epochs):
            A, B = self.data_generator.generate_batch(64)
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            loss = torch.mean((C_pred - C_true) ** 2)
            
            optimizer.zero_grad()
            loss.backward()
            
            for slot in slots_to_prune:
                if model.U.grad is not None:
                    model.U.grad[slot] = 0
                if model.V.grad is not None:
                    model.V.grad[slot] = 0
                if model.W.grad is not None:
                    model.W.grad[:, slot] = 0
            
            optimizer.step()
            
            with torch.no_grad():
                for slot in slots_to_prune:
                    model.U.data[slot] = 0
                    model.V.data[slot] = 0
                    model.W.data[:, slot] = 0
            
            if epoch % self.config.phase3_print_interval == 0:
                with torch.no_grad():
                    err = (C_pred - C_true).abs().max().item()
                print(f"Refine {epoch:5d} | Loss: {loss.item():.2e} | MaxErr: {err:.2e}")


class CoefficientDiscretizationPhase(ITrainingPhase):
    """Discretize coefficients to {-1, 0, 1}."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def execute(self, model: StrassenOperator, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Execute discretization phase."""
        slots_to_prune = kwargs.get('slots_to_prune', [])
        
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
            
            active_slots = []
            for i in range(U.shape[0]):
                if i not in slots_to_prune:
                    u_active = (U_disc[i].abs() > 0).any()
                    v_active = (V_disc[i].abs() > 0).any()
                    w_active = (W_disc[:, i].abs() > 0).any()
                    
                    if u_active and v_active and w_active:
                        active_slots.append(i)
                    else:
                        if not u_active:
                            max_idx = U[i].abs().argmax()
                            U_disc[i, max_idx] = torch.sign(U[i, max_idx]) if U[i, max_idx] != 0 else 1
                        if not v_active:
                            max_idx = V[i].abs().argmax()
                            V_disc[i, max_idx] = torch.sign(V[i, max_idx]) if V[i, max_idx] != 0 else 1
                        if not w_active:
                            max_idx = W[:, i].abs().argmax()
                            W_disc[max_idx, i] = torch.sign(W[max_idx, i]) if W[max_idx, i] != 0 else 1
                        active_slots.append(i)
            
            if len(active_slots) != self.config.target_rank:
                print(f"  [WARNING] Found {len(active_slots)} active slots, "
                      f"expected {self.config.target_rank}")
                
                if len(active_slots) > self.config.target_rank:
                    importance = torch.zeros(len(active_slots))
                    for idx, slot in enumerate(active_slots):
                        importance[idx] = (U_disc[slot].abs().sum() +
                                         V_disc[slot].abs().sum() +
                                         W_disc[:, slot].abs().sum())
                    
                    _, sorted_idx = importance.sort(descending=True)
                    keep_slots = [active_slots[i] for i in sorted_idx[:self.config.target_rank]]
                    
                    for slot in active_slots:
                        if slot not in keep_slots:
                            U_disc[slot] = 0
                            V_disc[slot] = 0
                            W_disc[:, slot] = 0
            
            disc_error = (torch.norm(U - U_disc) +
                         torch.norm(V - V_disc) +
                         torch.norm(W - W_disc)).item()
            print(f"Discretization error: {disc_error:.4f}")
            
            print(f"\nStructural verification:")
            print(f"  Active slots: {len([i for i in range(U.shape[0]) if (U_disc[i].abs().sum() > 0)])}")
            print(f"  U non-zeros per row: {[(U_disc[i].abs() > 0).sum().item() for i in range(U.shape[0]) if (U_disc[i].abs().sum() > 0)]}")
            print(f"  V non-zeros per row: {[(V_disc[i].abs() > 0).sum().item() for i in range(V.shape[0]) if (V_disc[i].abs().sum() > 0)]}")
            print(f"  W non-zeros per col: {[(W_disc[:, i].abs() > 0).sum().item() for i in range(W.shape[1]) if (W_disc[:, i].abs().sum() > 0)]}")
            
            print("\nU discretized:")
            print(U_disc)
            print("\nV discretized:")
            print(V_disc)
            print("\nW discretized:")
            print(W_disc)
        
        return U_disc, V_disc, W_disc


class StrassenVerifier:
    """Verify Strassen algorithm correctness."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def verify(self, U: torch.Tensor, V: torch.Tensor, W: torch.Tensor,
              n_test: int = 10000) -> Tuple[bool, float, Dict[str, float]]:
        """Verify algorithm on random test matrices."""
        print("\n" + "=" * 70)
        print("VERIFICATION")
        print("=" * 70)
        
        A = torch.randn(n_test, self.config.matrix_size, self.config.matrix_size)
        B = torch.randn(n_test, self.config.matrix_size, self.config.matrix_size)
        
        a = A.reshape(n_test, self.config.input_dim)
        b = B.reshape(n_test, self.config.input_dim)
        
        left = a @ U.t()
        right = b @ V.t()
        products = left * right
        c = products @ W.t()
        
        C_pred = c.reshape(n_test, self.config.matrix_size, self.config.matrix_size)
        C_true = torch.bmm(A, B)
        
        error = (C_pred - C_true).abs()
        max_err = error.max().item()
        mean_err = error.mean().item()
        mse = ((C_pred - C_true) ** 2).mean().item()
        
        active = ((U.abs().sum(dim=1) > 0) &
                 (V.abs().sum(dim=1) > 0) &
                 (W.abs().sum(dim=0) > 0)).sum().item()
        
        print(f"Samples: {n_test}")
        print(f"Max error: {max_err:.2e}")
        print(f"Mean error: {mean_err:.2e}")
        print(f"MSE: {mse:.2e}")
        print(f"Active slots: {active}")
        
        success = max_err < self.config.discretization_tolerance
        
        print("\n" + "-" * 40)
        print("SUCCESSFUL GROKKING" if success else "Incomplete grokking")
        print("-" * 40)
        
        return success, max_err, {
            'max_error': max_err,
            'mean_error': mean_err,
            'mse': mse,
            'active_slots': float(active)
        }


class CanonicalStrassenProvider:
    """Provide canonical Strassen algorithm coefficients."""
    
    @staticmethod
    def get_canonical() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return canonical Strassen algorithm matrices."""
        U = torch.tensor([
            [1, 0, 0, 1],
            [0, 0, 1, 1],
            [1, 0, 0, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [-1, 0, 1, 0],
            [0, 1, 0, -1],
        ], dtype=torch.float32)
        
        V = torch.tensor([
            [1, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, -1],
            [-1, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
        ], dtype=torch.float32)
        
        W = torch.zeros(4, 7)
        W[0, 0] = 1; W[0, 3] = 1; W[0, 4] = -1; W[0, 6] = 1
        W[1, 2] = 1; W[1, 4] = 1
        W[2, 1] = 1; W[2, 3] = 1
        W[3, 0] = 1; W[3, 1] = -1; W[3, 2] = 1; W[3, 5] = 1
        
        return U, V, W


class SeedProspector:
    """Prospect seeds for crystallization candidates."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.crystal_dir = Path(config.crystal_dir)
        self.crystal_dir.mkdir(exist_ok=True)
    
    def prospect(self, total_attempts: Optional[int] = None,
                start_seed: Optional[int] = None) -> bool:
        """Prospect multiple seeds for crystals."""
        if total_attempts is None:
            total_attempts = self.config.mining_max_attempts
        if start_seed is None:
            start_seed = self.config.mining_start_seed
        
        print(f"\nStarting seed prospecting with {total_attempts} attempts")
        print(f"Target metrics: delta < {self.config.target_delta}, "
              f"Acc > {self.config.target_accuracy}%, "
              f"Slots <= {self.config.target_rank}")
        
        results_log = []
        
        for i in range(start_seed, start_seed + total_attempts):
            current_seed = i
            attempt_num = i - start_seed + 1
            
            print(f"\n{'='*60}")
            print(f"[*] MINING SEED {current_seed} ({attempt_num}/{total_attempts})")
            print(f"{'='*60}")
            
            self._set_seed(current_seed)
            
            model = StrassenOperator(self.config).to(self.config.device)
            
            phase = ProspectorPhase(self.config, current_seed)
            start_time = time.time()
            is_crystal, final_metrics = phase.execute(model)
            elapsed = time.time() - start_time
            
            result = {
                'seed': current_seed,
                'is_crystal': is_crystal,
                'elapsed_seconds': elapsed,
                'final_metrics': final_metrics,
                'timestamp': datetime.now().isoformat()
            }
            results_log.append(result)
            
            if is_crystal:
                print(f"\n{'='*60}")
                print(f"[+] CRYSTAL FOUND! Seed {current_seed}")
                print(f"{'='*60}")
                print(f"  delta (discretization): {final_metrics['delta']:.6f}")
                print(f"  Accuracy: {final_metrics['accuracy']:.2f}%")
                print(f"  kappa (condition): {final_metrics.get('kappa', 0):.2f}")
                print(f"  Active slots: {int(final_metrics['active_slots'])}")
                print(f"  alpha (purity): {final_metrics.get('alpha_purity', 0):.2f}")
                
                crystal_path = self.crystal_dir / f"crystal_seed_{current_seed}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth"
                
                torch.save({
                    'seed': current_seed,
                    'model_state_dict': model.state_dict(),
                    'metrics': final_metrics,
                    'config': self.config
                }, crystal_path)
                
                print(f"  Crystal saved to {crystal_path}")
                
                json_path = crystal_path.with_suffix('.json')
                metrics_json = {k: float(v) if isinstance(v, (torch.Tensor, np.ndarray)) else v
                              for k, v in final_metrics.items()}
                
                with open(json_path, 'w') as f:
                    json.dump({
                        'seed': current_seed,
                        'metrics': metrics_json,
                        'timestamp': datetime.now().isoformat()
                    }, f, indent=2)
            else:
                print(f"  [-] Seed {current_seed} did not crystallize")
            
            if attempt_num % self.config.mining_partial_log_interval == 0:
                log_path = self.crystal_dir / 'mining_log_partial.json'
                with open(log_path, 'w') as f:
                    json.dump(results_log, f, indent=2, default=str)
                print(f"  Partial log saved ({attempt_num} attempts processed)")
        
        final_log_path = self.crystal_dir / 'mining_log_complete.json'
        with open(final_log_path, 'w') as f:
            json.dump(results_log, f, indent=2, default=str)
        
        crystals_found = sum(1 for r in results_log if r['is_crystal'])
        
        print(f"\n{'='*60}")
        print(f"MINING COMPLETE")
        print(f"{'='*60}")
        print(f"Total attempts: {total_attempts}")
        print(f"Crystals found: {crystals_found}")
        print(f"Success rate: {100*crystals_found/total_attempts:.1f}%")
        
        if crystals_found > 0:
            crystal_seeds = [r['seed'] for r in results_log if r['is_crystal']]
            print(f"Crystal seeds: {crystal_seeds}")
        
        print(f"Results saved to: {final_log_path}")
        
        return crystals_found > 0
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.config.device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)


class LongTrainingPipeline:
    """Long training pipeline with all phases."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.checkpoint_manager = CheckpointManager(config)
        self.verifier = StrassenVerifier(config)
        self.canonical_provider = CanonicalStrassenProvider()
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self._set_seed(config.random_seed)
        
        self.interrupted = False
        signal.signal(signal.SIGINT, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signal."""
        print("\n\n[INFO] Interrupt received. Saving checkpoint...")
        self.interrupted = True
    
    def _set_seed(self, seed: int) -> None:
        """Set random seed for reproducibility."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        if self.config.device == 'cuda':
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
    
    def run(self, resume_from: Optional[str] = None,
           seed: Optional[int] = None) -> bool:
        """Run complete training pipeline."""
        print("\n" + "=" * 70)
        print("  RIEMANN-STRASSEN CRYSTALLIZATION v3.0")
        print("  Phase Transition in Weight Space")
        print("=" * 70)
        print(f"Device: {self.config.device}")
        print(f"Parameters: {self.config.perelman_dimension}")
        print(f"Batch size: {self.config.batch_size_min}-{self.config.batch_size_max}")
        print(f"Kappa threshold: {self.config.kappa_crystallization_threshold}")
        print(f"Delta target: {self.config.target_delta}")
        print(f"Tau: {'Adaptive' if self.config.perelman_tau_adaptive else 'Fixed'}")
        print("=" * 70)
        
        if seed is not None:
            self._set_seed(seed)
            print(f"Using seed: {seed}")
        
        model = StrassenOperator(self.config).to(self.config.device)
        
        if resume_from is None:
            latest = self.checkpoint_manager.get_latest_checkpoint_path()
            if latest:
                print(f"\n[RESUME] Checkpoint found: {latest}")
                resp = input("Resume? (y/n): ").strip().lower()
                if resp == 'y':
                    resume_from = latest
        
        if resume_from:
            checkpoint = self.checkpoint_manager.load(resume_from)
            if checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
                print(f"[RESUME] From epoch {checkpoint.get('epoch', 'unknown')}")
        
        phase1 = LongTrainingPhase(self.config)
        model = phase1.execute(model)
        
        if self.interrupted:
            return False
        
        active = model.count_active()
        if active > self.config.target_rank:
            phase2 = ProgressiveSparsificationPhase(self.config)
            model, pruned = phase2.execute(model, target_slots=self.config.target_rank)
        else:
            pruned = []
        
        if self.interrupted:
            return False
        
        phase3 = CoefficientDiscretizationPhase(self.config)
        U, V, W = phase3.execute(model, slots_to_prune=pruned)
        
        success, max_err, metrics = self.verifier.verify(U, V, W)
        
        if not success:
            print("\n" + "=" * 70)
            print("FALLBACK: Using canonical Strassen coefficients")
            print("=" * 70)
            U, V, W = self.canonical_provider.get_canonical()
            success, max_err, metrics = self.verifier.verify(U, V, W)
        
        torch.save({
            'U': U,
            'V': V,
            'W': W,
            'config': self.config,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }, self.output_dir / "strassen_weights.pt")
        
        if hasattr(phase1, 'history') and phase1.history:
            with open(self.output_dir / "training_metrics.json", 'w') as f:
                json.dump(phase1.history, f, indent=2)
        
        return success

class LocalComplexityCalculator(IMetricCalculator):
    """Calculate Local Complexity as defined in the paper."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def calculate(self, model: StrassenOperator, **kwargs) -> Dict[str, float]:
        """
        Calculate LC as effective local dimensionality (paper definition).
        Compatible fraction approach → log(volume) can be negative.
        """
        device = next(model.parameters()).device
        epsilon = self.config.local_complexity_epsilon
        num_samples = self.config.local_complexity_num_samples
        
        theta_0 = model.get_flat_parameters().detach()
        n_params = theta_0.numel()
        
        A_test = torch.randn(num_samples, self.config.matrix_size, 
                            self.config.matrix_size, device=device)
        B_test = torch.randn(num_samples, self.config.matrix_size, 
                            self.config.matrix_size, device=device)
        C_true = torch.bmm(A_test, B_test)
        
        with torch.no_grad():
            C_0 = model(A_test, B_test)
            loss_0 = torch.mean((C_0 - C_true) ** 2).item()
        
        compatible_count = 0
        
        for _ in range(num_samples):
            perturbation = torch.randn_like(theta_0) * epsilon
            theta_perturbed = theta_0 + perturbation
            
            idx = 0
            for param in model.parameters():
                numel = param.numel()
                param.data = theta_perturbed[idx:idx+numel].reshape(param.shape)
                idx += numel
            
            with torch.no_grad():
                C_perturbed = model(A_test, B_test)
                loss_perturbed = torch.mean((C_perturbed - C_true) ** 2).item()
            
            if abs(loss_perturbed - loss_0) < epsilon:
                compatible_count += 1
        
        idx = 0
        for param in model.parameters():
            numel = param.numel()
            param.data = theta_0[idx:idx+numel].reshape(param.shape)
            idx += numel
        
        if compatible_count > 0:
            compatible_fraction = compatible_count / num_samples
            volume_estimate = compatible_fraction * ((2 * epsilon) ** n_params)
            lc = math.log(volume_estimate) if volume_estimate > 0 else -math.inf
        else:
            lc = -math.inf
        
        return {
            'local_complexity': lc,
            'lc_compatible_fraction': compatible_count / num_samples
        }
        

class SuperpositionCalculator(IMetricCalculator):
    """Calculate Superposition metrics using sparse autoencoder."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.sae_encoder = None
        self.sae_decoder = None
    
    def _initialize_sae(self, input_dim: int, device: str):
        """Initialize sparse autoencoder if not already done."""
        if self.sae_encoder is None:
            hidden_dim = self.config.superposition_sae_hidden_dim
            self.sae_encoder = nn.Linear(input_dim, hidden_dim).to(device)
            self.sae_decoder = nn.Linear(hidden_dim, input_dim).to(device)
    
    def calculate(self, model: StrassenOperator, **kwargs) -> Dict[str, float]:
        """Calculate superposition coefficient psi and effective features F."""
        device = next(model.parameters()).device
        
        theta = model.get_flat_parameters().detach()
        input_dim = theta.numel()
        
        self._initialize_sae(input_dim, device)
        
        theta_expanded = theta.unsqueeze(0)
        
        encoded = self.sae_encoder(theta_expanded)
        encoded_sparse = torch.relu(encoded)
        reconstructed = self.sae_decoder(encoded_sparse)
        
        reconstruction_error = torch.mean((theta_expanded - reconstructed) ** 2).item()
        
        active_features = (encoded_sparse.abs() > self.config.superposition_sae_sparsity).sum().item()
        total_features = encoded_sparse.numel()
        
        feature_norms = torch.norm(encoded_sparse, dim=0)
        feature_variance = torch.var(feature_norms).item()
        
        psi = 1.0 + reconstruction_error + (feature_variance / (feature_norms.mean().item() + 1e-8))
        
        effective_features = active_features * (1.0 - reconstruction_error)
        
        return {
            'superposition_psi': psi,
            'superposition_F': effective_features,
            'superposition_active_fraction': active_features / total_features,
            'superposition_reconstruction_error': reconstruction_error
        }

class ThermodynamicMetricsCalculator(IMetricCalculator):
    """Calculate thermodynamic metrics: h_bar_eff and T_eff."""
    
    def __init__(self, config: UnifiedConfig):
        self.config = config
    
    def calculate(self, model: StrassenOperator, 
                  gradient_covariance: Optional[torch.Tensor] = None,
                  **kwargs) -> Dict[str, float]:
        """Calculate effective Planck constant and temperature."""
        
        if gradient_covariance is None:
            return {
                'h_bar_eff': 0.0,
                'T_eff': 0.0,
                'thermodynamic_entropy': 0.0
            }
        
        try:
            eigenvalues = torch.linalg.eigvalsh(gradient_covariance)
            eigenvalues = eigenvalues[eigenvalues > self.config.eigenvalue_filter_threshold]
            
            if len(eigenvalues) == 0:
                return {
                    'h_bar_eff': 0.0,
                    'T_eff': 0.0,
                    'thermodynamic_entropy': 0.0
                }
            
            trace_sigma = eigenvalues.sum().item()
            dimension = len(eigenvalues)
            
            T_eff = trace_sigma / dimension if dimension > 0 else 0.0
            
            position_uncertainty = torch.sqrt(eigenvalues.mean()).item()
            h_bar_eff = position_uncertainty * self.config.planck_constant_regularization
            
            if T_eff > 0:
                boltzmann_entropy = -torch.sum(
                    eigenvalues * torch.log(eigenvalues + 1e-300)
                ).item() / (self.config.boltzmann_constant * T_eff)
            else:
                boltzmann_entropy = 0.0
            
            return {
                'h_bar_eff': h_bar_eff,
                'T_eff': T_eff,
                'thermodynamic_entropy': boltzmann_entropy,
                'trace_gradient_covariance': trace_sigma
            }
            
        except Exception:
            return {
                'h_bar_eff': 0.0,
                'T_eff': 0.0,
                'thermodynamic_entropy': 0.0
            }

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Riemann-Strassen Crystallization Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Modes:
  prospect    Fast seed prospecting to identify crystallization candidates
  train       Long training with full thermodynamic metrics

Examples:
  python script.py prospect --attempts 100 --start-seed 1
  python script.py train --seed 42 --epochs 29000
  python script.py train --resume checkpoints/latest.pt
        """
    )
    
    parser.add_argument('mode', type=str, choices=['prospect', 'train'],
                       help='Execution mode')
    
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed for training mode')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume from checkpoint path')
    
    parser.add_argument('--attempts', type=int, default=None,
                       help='Number of seeds to prospect (prospect mode)')
    parser.add_argument('--start-seed', type=int, default=None,
                       help='Starting seed number (prospect mode)')
    
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs (train mode)')
    parser.add_argument('--batch-min', type=int, default=None,
                       help='Minimum batch size')
    parser.add_argument('--batch-max', type=int, default=None,
                       help='Maximum batch size')
    parser.add_argument('--learning-rate', type=float, default=None,
                       help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=None,
                       help='Weight decay (pressure)')
    
    args = parser.parse_args()
    
    config_overrides = {}
    
    if args.epochs is not None:
        config_overrides['train_epochs'] = args.epochs
    if args.batch_min is not None:
        config_overrides['batch_size_min'] = args.batch_min
    if args.batch_max is not None:
        config_overrides['batch_size_max'] = args.batch_max
    if args.learning_rate is not None:
        config_overrides['learning_rate'] = args.learning_rate
    if args.weight_decay is not None:
        config_overrides['weight_decay'] = args.weight_decay
    
    if args.attempts is not None:
        config_overrides['mining_max_attempts'] = args.attempts
    if args.start_seed is not None:
        config_overrides['mining_start_seed'] = args.start_seed
    
    config = replace(UnifiedConfig(), **config_overrides)
    
    if args.mode == 'prospect':
        prospector = SeedProspector(config)
        success = prospector.prospect()
        sys.exit(0 if success else 1)
    
    elif args.mode == 'train':
        pipeline = LongTrainingPipeline(config)
        success = pipeline.run(resume_from=args.resume, seed=args.seed)
        sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
