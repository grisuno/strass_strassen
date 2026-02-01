#!/usr/bin/env python3

import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Protocol, runtime_checkable
from pathlib import Path
import glob
from dataclasses import dataclass
from scipy import signal
from scipy.linalg import eigvals
from scipy.stats import entropy
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle


@dataclass(frozen=True)
class ThermodynamicConfig:
    BATCH_SIZE: int = 32
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    WEIGHT_DECAY: float = 1e-4
    LEARNING_RATE: float = 0.001
    
    EPSILON_0: float = 8.854187817e-12
    DELTA_THRESHOLD: float = 1e-6
    GAUSSIAN_SIGMA: float = 0.1
    SPATIAL_RESOLUTION: int = 100
    FLUX_INTEGRATION_SAMPLES: int = 1000
    
    POLE_ZERO_TOLERANCE: float = 1e-6
    STABILITY_MARGIN: float = 0.01
    FREQUENCY_SAMPLES: int = 1000
    FREQUENCY_MIN: float = 1e-3
    FREQUENCY_MAX: float = 1e3
    TIME_SAMPLES: int = 500
    TIME_MAX: float = 10.0
    
    COLORMAP: str = 'viridis'
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'
    
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    ENTROPY_BINS: int = 50
    TEMPERATURE_WINDOW: int = 100
    G_ALG_WINDOW: int = 10
    LANDAUER_THRESHOLD: float = 1e-10
    HEISENBERG_TOLERANCE: float = 1e-6


@runtime_checkable
class IModel(Protocol):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...


@runtime_checkable
class IOrderParameterCalculator(Protocol):
    def calculate(self, model: IModel) -> float: ...


@runtime_checkable
class IEntropyCalculator(Protocol):
    def calculate(self, model: IModel) -> float: ...


@runtime_checkable
class ISpecificHeatCalculator(Protocol):
    def calculate(self, loss_history: List[float]) -> float: ...


@runtime_checkable
class IGravitationalConstantCalculator(Protocol):
    def calculate(self, model: IModel, gradient_history: List[torch.Tensor]) -> float: ...


@runtime_checkable
class ILandauerConstantCalculator(Protocol):
    def calculate(self, entropy_change: float, energy_dissipated: float) -> float: ...


@runtime_checkable
class IHeisenbergUncertaintyCalculator(Protocol):
    def calculate(self, model: IModel, temperature: float) -> float: ...


@runtime_checkable
class ILocalComplexityCalculator(Protocol):
    def calculate(self, model: IModel) -> float: ...


@runtime_checkable
class IBasinStabilityCalculator(Protocol):
    def calculate(self, model: IModel, test_data: Tuple[torch.Tensor, ...]) -> Dict[str, Any]: ...


@runtime_checkable
class IZeroShotTransferCalculator(Protocol):
    def calculate(self, model: IModel, target_size: int) -> float: ...


@runtime_checkable
class IConditionNumberCalculator(Protocol):
    def calculate(self, gradient_covariance: np.ndarray) -> float: ...


class BilinearModel(nn.Module):
    def __init__(self, hidden_dim: int = ThermodynamicConfig.HIDDEN_DIM, matrix_size: int = ThermodynamicConfig.MATRIX_SIZE):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.matrix_size = matrix_size
        input_dim = matrix_size * matrix_size
        
        self.U = nn.Linear(input_dim, hidden_dim, bias=False)
        self.V = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, input_dim, bias=False)
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


class OrderParameterCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, model: IModel) -> float:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        rounded = torch.round(all_weights)
        delta = torch.max(torch.abs(all_weights - rounded)).item()
        return delta


class ConfigurationEntropyCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, model: IModel) -> float:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()]).cpu().numpy()
        
        hist, _ = np.histogram(all_weights, bins=self.config.ENTROPY_BINS, density=True)
        hist = hist[hist > 0]
        
        if len(hist) == 0:
            return 0.0
        
        return float(entropy(hist))


class SpecificHeatCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, loss_history: List[float]) -> float:
        if len(loss_history) < self.config.TEMPERATURE_WINDOW:
            return 0.0
        
        recent_losses = loss_history[-self.config.TEMPERATURE_WINDOW:]
        return float(np.var(recent_losses))


class GravitationalConstantCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, model: IModel, gradient_history: List[torch.Tensor], loss_history: List[float], static_gradient: Optional[torch.Tensor] = None) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        rounded = torch.round(all_weights)
        distances = torch.abs(all_weights - rounded)
        
        mask = distances > self.config.DELTA_THRESHOLD
        valid_distances = distances[mask]
        
        if len(valid_distances) == 0:
            return {
                'g_alg': 0.0,
                'avg_distance': 0.0,
                'gradient_magnitude': 0.0,
                'force': 0.0,
                'crystallization_pressure': 0.0
            }
        
        avg_distance = torch.mean(valid_distances).item()
        
        if static_gradient is not None:
            avg_gradient = torch.norm(static_gradient).item()
        elif gradient_history:
            recent_grads = gradient_history[-min(len(gradient_history), self.config.G_ALG_WINDOW):]
            grad_mags = [torch.norm(g) for g in recent_grads]
            avg_gradient = torch.mean(torch.stack(grad_mags)).item()
        else:
            avg_gradient = 0.0
        
        if loss_history and len(loss_history) > 1:
            loss_trend = (loss_history[-1] - loss_history[0]) / len(loss_history)
        else:
            loss_trend = 0.0
        
        if avg_distance < self.config.DELTA_THRESHOLD:
            g_alg = float('inf')
            force = float('inf')
        else:
            force = avg_gradient / (avg_distance ** 2)
            g_alg = force
        
        crystallization_pressure = abs(loss_trend) / (avg_distance + 1e-10) if avg_gradient > 0 else 0.0
        
        return {
            'g_alg': float(g_alg),
            'avg_distance': float(avg_distance),
            'gradient_magnitude': float(avg_gradient),
            'force': float(force),
            'crystallization_pressure': float(crystallization_pressure)
        }


class LandauerConstantCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, entropy_change: float, energy_dissipated: float, has_transition: bool = False, transition_window: bool = False) -> Dict[str, float]:
        if not has_transition or not transition_window:
            return {
                'k_b_eff': 0.0,
                'information_dissipation': 0.0,
                'measurable': False,
                'reason': 'no_transition_or_not_in_window'
            }
        
        if abs(entropy_change) < self.config.LANDAUER_THRESHOLD:
            return {
                'k_b_eff': float('inf'),
                'information_dissipation': float('inf'),
                'measurable': False,
                'reason': 'entropy_change_too_small'
            }
        
        k_b_eff = energy_dissipated / abs(entropy_change)
        
        return {
            'k_b_eff': float(k_b_eff),
            'information_dissipation': float(energy_dissipated / (k_b_eff + 1e-10)),
            'measurable': True,
            'entropy_erased': float(abs(entropy_change)),
            'heat_dissipated': float(energy_dissipated)
        }


class HeisenbergUncertaintyCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, model: IModel, temperature: float, static_gradient: Optional[torch.Tensor] = None) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        weight_variance = torch.var(all_weights).item()
        weight_mean = torch.mean(torch.abs(all_weights)).item()
        
        if static_gradient is not None:
            grad_spectrum = torch.fft.rfft(static_gradient).abs()
            spectral_variance = torch.var(grad_spectrum).item()
        else:
            spectral_variance = weight_variance * 0.1
        
        position_uncertainty = np.sqrt(weight_variance)
        momentum_uncertainty = np.sqrt(spectral_variance)
        
        hbar_eff = position_uncertainty * momentum_uncertainty
        
        if temperature < self.config.HEISENBERG_TOLERANCE:
            temperature = self.config.HEISENBERG_TOLERANCE
        
        thermal_uncertainty = temperature * weight_mean
        
        principle_satisfied = hbar_eff > (self.config.HEISENBERG_TOLERANCE * 0.1)
        
        return {
            'hbar_eff': float(hbar_eff),
            'position_uncertainty': float(position_uncertainty),
            'momentum_uncertainty': float(momentum_uncertainty),
            'thermal_uncertainty': float(thermal_uncertainty),
            'structure_noise_product': float(hbar_eff),
            'uncertainty_principle_satisfied': principle_satisfied
        }


class LocalComplexityCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, model: IModel) -> float:
        coeffs = model.get_coefficients()
        
        u_flat = coeffs['U'].flatten()
        v_flat = coeffs['V'].flatten()
        w_flat = coeffs['W'].flatten()
        
        uv_interaction = torch.outer(u_flat, v_flat)
        uvw_volume = torch.sum(torch.abs(torch.outer(uv_interaction.flatten(), w_flat))).item()
        
        return float(uvw_volume)


class BasinStabilityCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, model: IModel, test_data: Tuple[torch.Tensor, ...]) -> Dict[str, Any]:
        a, b, c_target = test_data
        
        original_state = {name: param.clone() for name, param in model.named_parameters()}
        
        pruning_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        results = {}
        
        for level in pruning_levels:
            self._prune_model(model, level)
            
            with torch.no_grad():
                c_pred = model(a, b)
                mse = torch.mean((c_pred - c_target) ** 2).item()
            
            results[f'sparsity_{level:.1f}'] = {
                'mse': mse,
                'accurate': mse < 0.01
            }
            
            model.load_state_dict(original_state)
        
        return results
    
    def _prune_model(self, model: IModel, sparsity: float):
        with torch.no_grad():
            for param in model.parameters():
                flat = param.flatten()
                k = int(sparsity * flat.numel())
                if k > 0:
                    threshold = torch.topk(torch.abs(flat), k, largest=False).values[-1]
                    param[torch.abs(param) < threshold] = 0


class ZeroShotTransferCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def calculate(self, model: IModel, target_size: int) -> float:
        coeffs = model.get_coefficients()
        
        u_2x2 = coeffs['U'].cpu().numpy()
        v_2x2 = coeffs['V'].cpu().numpy()
        w_2x2 = coeffs['W'].cpu().numpy()
        
        if target_size == 2:
            return 1.0
        
        scale_factor = target_size // 2
        
        try:
            u_scaled = self._kronecker_recursive(u_2x2, scale_factor)
            v_scaled = self._kronecker_recursive(v_2x2, scale_factor)
            w_scaled = self._kronecker_recursive(w_2x2, scale_factor)
            
            transfer_quality = np.linalg.norm(u_scaled) * np.linalg.norm(v_scaled) * np.linalg.norm(w_scaled)
            return float(transfer_quality)
        except:
            return 0.0
    
    def _kronecker_recursive(self, matrix: np.ndarray, power: int) -> np.ndarray:
        result = matrix
        for _ in range(power - 1):
            result = np.kron(result, matrix)
        return result


class ConditionNumberCalculator:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
        self.tikhonov_epsilon = 1e-6
    
    def calculate(self, gradient_history: List[torch.Tensor], static_gradient: Optional[torch.Tensor] = None) -> Dict[str, float]:
        grads_to_use = []
        
        if static_gradient is not None:
            grads_to_use.append(static_gradient.flatten().cpu().numpy())
        
        for g in gradient_history:
            if isinstance(g, torch.Tensor):
                grads_to_use.append(g.flatten().cpu().numpy())
            elif isinstance(g, (list, np.ndarray)):
                grads_to_use.append(np.array(g).flatten())
        
        if len(grads_to_use) == 0:
            return {
                'kappa': float('inf'),
                'log_kappa': float('inf'),
                'effective_rank': 0.0,
                'predicts_success': False
            }
        
        if len(grads_to_use) == 1:
            grad_matrix = np.array([grads_to_use[0], grads_to_use[0] * 1.001])
        else:
            grad_matrix = np.stack(grads_to_use)
        
        if grad_matrix.shape[0] < grad_matrix.shape[1]:
            sigma = np.cov(grad_matrix.T)
        else:
            sigma = np.cov(grad_matrix)
        
        sigma += np.eye(sigma.shape[0]) * self.tikhonov_epsilon
        
        try:
            eigenvalues = np.linalg.eigvalsh(sigma)
            eigenvalues = np.maximum(eigenvalues, self.tikhonov_epsilon)
            kappa = np.max(eigenvalues) / np.min(eigenvalues)
            effective_rank = np.sum(eigenvalues / np.max(eigenvalues) > 0.01)
        except:
            kappa = float('inf')
            effective_rank = 0.0
        
        log_kappa = np.log10(kappa) if kappa < float('inf') and kappa > 0 else float('inf')
        
        return {
            'kappa': float(kappa),
            'log_kappa': float(log_kappa),
            'effective_rank': float(effective_rank),
            'predicts_success': kappa < 1.001
        }


class PhaseTransitionDetector:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def detect(self, loss_history: List[float], entropy_history: Optional[List[float]] = None) -> Dict[str, Any]:
        if len(loss_history) < self.config.TEMPERATURE_WINDOW * 2:
            return {
                'has_transition': False,
                'transition_epoch': None,
                'transition_sharpness': 0.0,
                'heat_peak': 0.0,
                'entropy_drop_rate': 0.0
            }
        
        heat = []
        for i in range(len(loss_history) - self.config.TEMPERATURE_WINDOW):
            window = loss_history[i:i + self.config.TEMPERATURE_WINDOW]
            heat.append(np.var(window))
        
        if len(heat) < 2:
            return {
                'has_transition': False,
                'transition_epoch': None,
                'transition_sharpness': 0.0,
                'heat_peak': 0.0,
                'entropy_drop_rate': 0.0
            }
        
        heat_peak_idx = np.argmax(heat)
        heat_peak = heat[heat_peak_idx]
        
        if entropy_history and len(entropy_history) == len(loss_history):
            pre_entropy = np.mean(entropy_history[:heat_peak_idx]) if heat_peak_idx > 0 else entropy_history[0]
            post_entropy = np.mean(entropy_history[heat_peak_idx:]) if heat_peak_idx < len(entropy_history) else entropy_history[-1]
            entropy_drop_rate = (pre_entropy - post_entropy) / (len(entropy_history) - heat_peak_idx + 1e-10)
        else:
            entropy_drop_rate = 0.0
        
        sharpness = heat_peak / (np.mean(heat) + 1e-10)
        
        has_transition = sharpness > 2.0 and heat_peak > 0.001
        
        return {
            'has_transition': bool(has_transition),
            'transition_epoch': int(heat_peak_idx) if has_transition else None,
            'transition_sharpness': float(sharpness),
            'heat_peak': float(heat_peak),
            'entropy_drop_rate': float(entropy_drop_rate)
        }


class ThermodynamicAnalyzer:
    def __init__(self, checkpoint_path: str, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        self.order_calculator = OrderParameterCalculator(config)
        self.entropy_calculator = ConfigurationEntropyCalculator(config)
        self.specific_heat_calculator = SpecificHeatCalculator(config)
        self.gravitational_calculator = GravitationalConstantCalculator(config)
        self.landauer_calculator = LandauerConstantCalculator(config)
        self.heisenberg_calculator = HeisenbergUncertaintyCalculator(config)
        self.complexity_calculator = LocalComplexityCalculator(config)
        self.basin_calculator = BasinStabilityCalculator(config)
        self.transfer_calculator = ZeroShotTransferCalculator(config)
        self.condition_calculator = ConditionNumberCalculator(config)
        self.transition_detector = PhaseTransitionDetector(config)
        
        self._load_checkpoint()
        self._compute_static_gradient()
    
    def _load_checkpoint(self):
        try:
            self.checkpoint = torch.load(self.checkpoint_path, map_location=self.config.DEVICE, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        self.model = BilinearModel(
            hidden_dim=self.config.HIDDEN_DIM,
            matrix_size=self.config.MATRIX_SIZE
        ).to(self.config.DEVICE)
        
        state_dict = self._migrate_checkpoint(self.checkpoint)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f"Failed to migrate checkpoint: {self.checkpoint_path}")
        
        self.epoch = self.checkpoint.get('epoch', 'unknown')
        self.loss_history = self.checkpoint.get('loss_history', [])
        self.gradient_history = self.checkpoint.get('gradient_history', [])
        self.entropy_history = self.checkpoint.get('entropy_history', [])
        self.lc_history = self.checkpoint.get('lc_history', [])
    
    def _migrate_checkpoint(self, raw_data: Any) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(raw_data, dict):
            if 'state_dict' in raw_data:
                return self._migrate_dict(raw_data['state_dict'])
            elif 'model_state_dict' in raw_data:
                return self._migrate_dict(raw_data['model_state_dict'])
            else:
                return self._migrate_dict(raw_data)
        elif hasattr(raw_data, 'state_dict'):
            return self._migrate_dict(raw_data.state_dict())
        return None
    
    def _migrate_dict(self, state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        if any(k in state_dict for k in ['U', 'V', 'W']):
            return self._migrate_custom_format(state_dict)
        elif 'U_coefs' in state_dict:
            return self._migrate_coefs_format(state_dict)
        elif any(k.endswith('.weight') for k in state_dict.keys()):
            return self._migrate_standard_format(state_dict)
        return None
    
    def _migrate_custom_format(self, state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        U = state_dict.get('U', state_dict.get('U_coefs'))
        V = state_dict.get('V', state_dict.get('V_coefs'))
        W = state_dict.get('W', state_dict.get('W_coefs'))
        
        if U is None or V is None or W is None:
            return None
        
        if U.shape == (7, 4):
            u_padded = torch.zeros(self.config.HIDDEN_DIM, 4, device=self.config.DEVICE)
            v_padded = torch.zeros(self.config.HIDDEN_DIM, 4, device=self.config.DEVICE)
            w_padded = torch.zeros(4, self.config.HIDDEN_DIM, device=self.config.DEVICE)
            u_padded[:7] = U
            v_padded[:7] = V
            w_padded[:, :7] = W
            return {'U.weight': u_padded, 'V.weight': v_padded, 'W.weight': w_padded}
        
        return {'U.weight': U, 'V.weight': V, 'W.weight': W}
    
    def _migrate_coefs_format(self, state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            'U.weight': state_dict['U_coefs'],
            'V.weight': state_dict['V_coefs'],
            'W.weight': state_dict['W_coefs']
        }
    
    def _migrate_standard_format(self, state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {k: state_dict[k] for k in ['U.weight', 'V.weight', 'W.weight'] if k in state_dict}
    
    def _compute_static_gradient(self):
        self.model.train()
        a, b, c_target = self._generate_test_data()
        
        self.model.zero_grad()
        c_pred = self.model(a, b)
        loss = torch.mean((c_pred - c_target) ** 2)
        loss.backward()
        
        grads = []
        for param in self.model.parameters():
            if param.grad is not None:
                grads.append(param.grad.flatten())
        
        if grads:
            self.static_gradient = torch.cat(grads)
        else:
            self.static_gradient = None
        
        self.static_loss = loss.item()
    
    def _generate_test_data(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = self.config.BATCH_SIZE
        A = torch.randn(batch_size, self.config.MATRIX_SIZE, self.config.MATRIX_SIZE, device=self.config.DEVICE)
        B = torch.randn(batch_size, self.config.MATRIX_SIZE, self.config.MATRIX_SIZE, device=self.config.DEVICE)
        C = torch.bmm(A, B)
        return (
            A.reshape(batch_size, self.config.MATRIX_SIZE * self.config.MATRIX_SIZE),
            B.reshape(batch_size, self.config.MATRIX_SIZE * self.config.MATRIX_SIZE),
            C.reshape(batch_size, self.config.MATRIX_SIZE * self.config.MATRIX_SIZE)
        )
    
    def analyze(self) -> Dict[str, Any]:
        delta = self.order_calculator.calculate(self.model)
        s_conf = self.entropy_calculator.calculate(self.model)
        c_heat = self.specific_heat_calculator.calculate(self.loss_history)
        
        grad_tensors = []
        for g in self.gradient_history:
            if isinstance(g, torch.Tensor):
                grad_tensors.append(g)
            elif isinstance(g, (list, np.ndarray)):
                grad_tensors.append(torch.tensor(g, device=self.config.DEVICE))
            elif isinstance(g, dict):
                grad_tensors.append(torch.cat([v.flatten() for v in g.values()]))
        
        phase_transition = self.transition_detector.detect(self.loss_history, self.entropy_history or None)
        in_transition_window = phase_transition.get('has_transition', False)
        
        g_alg_result = self.gravitational_calculator.calculate(
            self.model, grad_tensors, self.loss_history, self.static_gradient
        )
        
        if in_transition_window and self.entropy_history:
            entropy_prev = self.entropy_history[0] if self.entropy_history else s_conf * 1.1
            entropy_change = entropy_prev - s_conf
            energy_dissipated = c_heat
        else:
            entropy_change = 0.0
            energy_dissipated = 0.0
        
        landauer_result = self.landauer_calculator.calculate(
            entropy_change, energy_dissipated, 
            has_transition=phase_transition.get('has_transition', False),
            transition_window=in_transition_window
        )
        
        temperature = c_heat if c_heat > 0 else 0.001
        heisenberg_result = self.heisenberg_calculator.calculate(
            self.model, temperature, self.static_gradient
        )
        
        lc = self.complexity_calculator.calculate(self.model)
        
        test_data = self._generate_test_data()
        basin_results = self.basin_calculator.calculate(self.model, test_data)
        
        transfer_metrics = {}
        for size in [4, 8, 16, 32, 64]:
            transfer_metrics[f'{size}x{size}'] = self.transfer_calculator.calculate(self.model, size)
        
        kappa_result = self.condition_calculator.calculate(grad_tensors, self.static_gradient)
        
        is_crystallized = delta < 0.1
        maintains_structure_50 = basin_results.get('sparsity_0.5', {}).get('accurate', False)
        is_success = is_crystallized and maintains_structure_50
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat()
            },
            'order_parameter': {
                'delta': delta,
                'is_crystallized': is_crystallized,
                'crystal_quality': 1.0 - delta if delta < 1.0 else 0.0
            },
            'configuration_entropy': {
                's_conf': s_conf,
                'entropy_drop': entropy_change
            },
            'specific_heat': {
                'c_heat': c_heat,
                'temperature_effective': temperature
            },
            'gravitational_constant': g_alg_result,
            'landauer_constant': landauer_result,
            'heisenberg_uncertainty': heisenberg_result,
            'local_complexity': {
                'lc': lc,
                'lc_history_available': len(self.lc_history) > 0,
                'lc_final': self.lc_history[-1] if self.lc_history else lc
            },
            'basin_stability': basin_results,
            'zero_shot_transfer': transfer_metrics,
            'condition_number': kappa_result,
            'phase_transition': phase_transition,
            'static_computation': {
                'loss': self.static_loss,
                'gradient_computed': self.static_gradient is not None,
                'gradient_norm': torch.norm(self.static_gradient).item() if self.static_gradient is not None else 0.0
            },
            'success_criteria': {
                'is_success': is_success,
                'success_rate_estimate': 1.0 if is_success else 0.0,
                'failure_mode': self._determine_failure_mode(delta, basin_results, kappa_result['kappa'], phase_transition),
                'kappa_prediction_accuracy': kappa_result['predicts_success']
            }
        }
        
        self._print_report(results)
        
        return results
    
    def _determine_failure_mode(self, delta: float, basin: Dict, kappa: float, transition: Dict) -> str:
        if not transition.get('has_transition', False):
            return 'no_phase_transition'
        if delta >= 0.1:
            return 'no_crystallization'
        if not basin.get('sparsity_0.5', {}).get('accurate', False):
            return 'fragile_structure'
        if kappa > 1.1:
            return 'high_condition_number'
        return 'none'
    
    def _print_report(self, results: Dict):
        print("=" * 80)
        print("THERMODYNAMIC ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {results['metadata']['checkpoint_path']}")
        print(f"  Epoch: {results['metadata']['epoch']}")
        
        print(f"\n[ORDER PARAMETER]")
        op = results['order_parameter']
        print(f"  Delta: {op['delta']:.6f}")
        print(f"  Is crystallized: {op['is_crystallized']}")
        print(f"  Crystal quality: {op['crystal_quality']:.6f}")
        
        print(f"\n[CONFIGURATION ENTROPY]")
        ce = results['configuration_entropy']
        print(f"  S_conf: {ce['s_conf']:.6f}")
        print(f"  Entropy drop: {ce['entropy_drop']:.6f}")
        
        print(f"\n[SPECIFIC HEAT]")
        sh = results['specific_heat']
        print(f"  C_heat: {sh['c_heat']:.6f}")
        print(f"  T_effective: {sh['temperature_effective']:.6f}")
        
        print(f"\n[GRAVITATIONAL CONSTANT]")
        gc = results['gravitational_constant']
        print(f"  G_alg: {gc['g_alg']:.6e}")
        print(f"  Avg distance to integer: {gc['avg_distance']:.6f}")
        print(f"  Gradient magnitude: {gc['gradient_magnitude']:.6e}")
        print(f"  Force: {gc['force']:.6e}")
        print(f"  Crystallization pressure: {gc['crystallization_pressure']:.6e}")
        
        print(f"\n[LANDAUER CONSTANT]")
        lc = results['landauer_constant']
        print(f"  Measurable: {lc.get('measurable', False)}")
        if lc.get('measurable', False):
            print(f"  k_B_eff: {lc['k_b_eff']:.6e}")
            print(f"  Information dissipation: {lc['information_dissipation']:.6e}")
        else:
            print(f"  Reason: {lc.get('reason', 'unknown')}")
        
        print(f"\n[HEISENBERG UNCERTAINTY]")
        hu = results['heisenberg_uncertainty']
        print(f"  hbar_eff: {hu['hbar_eff']:.6e}")
        print(f"  Position uncertainty: {hu['position_uncertainty']:.6e}")
        print(f"  Momentum uncertainty: {hu['momentum_uncertainty']:.6e}")
        print(f"  Thermal uncertainty: {hu['thermal_uncertainty']:.6e}")
        print(f"  Principle satisfied: {hu['uncertainty_principle_satisfied']}")
        
        print(f"\n[LOCAL COMPLEXITY]")
        lcx = results['local_complexity']
        print(f"  LC: {lcx['lc']:.6e}")
        print(f"  LC history available: {lcx['lc_history_available']}")
        
        print(f"\n[PHASE TRANSITION]")
        pt = results['phase_transition']
        print(f"  Has transition: {pt['has_transition']}")
        print(f"  Transition epoch: {pt['transition_epoch']}")
        print(f"  Transition sharpness: {pt['transition_sharpness']:.6f}")
        print(f"  Heat peak: {pt['heat_peak']:.6e}")
        print(f"  Entropy drop rate: {pt['entropy_drop_rate']:.6e}")
        
        print(f"\n[STATIC COMPUTATION]")
        sc = results['static_computation']
        print(f"  Loss: {sc['loss']:.6f}")
        print(f"  Gradient computed: {sc['gradient_computed']}")
        print(f"  Gradient norm: {sc['gradient_norm']:.6e}")
        
        print(f"\n[BASIN STABILITY]")
        for key, val in results['basin_stability'].items():
            print(f"  {key}: MSE={val['mse']:.6f}, Accurate={val['accurate']}")
        
        print(f"\n[ZERO-SHOT TRANSFER]")
        for key, val in results['zero_shot_transfer'].items():
            print(f"  {key}: {val:.6e}")
        
        print(f"\n[CONDITION NUMBER]")
        cn = results['condition_number']
        print(f"  Kappa: {cn['kappa']:.6f}")
        print(f"  Log kappa: {cn['log_kappa']:.6f}")
        print(f"  Effective rank: {cn['effective_rank']:.2f}")
        print(f"  Predicts success: {cn['predicts_success']}")
        
        print(f"\n[SUCCESS CRITERIA]")
        sc = results['success_criteria']
        print(f"  Is success: {sc['is_success']}")
        print(f"  Success rate estimate: {sc['success_rate_estimate']:.2%}")
        print(f"  Failure mode: {sc['failure_mode']}")
        print(f"  Kappa prediction: {sc['kappa_prediction_accuracy']}")
        
        print("=" * 80)

class ThermodynamicPipeline:
    def __init__(self, config: ThermodynamicConfig = ThermodynamicConfig()):
        self.config = config
    
    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer = ThermodynamicAnalyzer(checkpoint_path, self.config)
        results = analyzer.analyze()
        
        base_name = Path(checkpoint_path).stem
        
        results_path = os.path.join(output_dir, f'{base_name}_thermo.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def process_directory(self, checkpoint_dir: str, n_latest: Optional[int], output_dir: str) -> List[Dict[str, Any]]:
        pattern = os.path.join(checkpoint_dir, '*.pt')
        checkpoints = glob.glob(pattern)
        
        if not checkpoints:
            print(f"No checkpoints found in {checkpoint_dir}")
            return []
        
        checkpoints.sort(key=os.path.getmtime, reverse=True)
        
        if n_latest is not None:
            checkpoints = checkpoints[:n_latest]
        
        print(f"\nProcessing {len(checkpoints)} checkpoints...\n")
        
        all_results = []
        for cp in checkpoints:
            try:
                results = self.process_checkpoint(cp, output_dir)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {cp}: {e}")
                import traceback
                traceback.print_exc()
        
        return all_results
    
    def generate_summary(self, all_results: List[Dict[str, Any]], output_dir: str) -> None:
        if not all_results:
            print("No results to summarize")
            return
        
        summary = {
            'total_checkpoints_analyzed': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'emergent_constants': self._compute_emergent_constants(all_results),
            'universal_laws': self._verify_universal_laws(all_results),
            'kappa_correlation': self._compute_kappa_correlation(all_results),
            'individual_results': all_results
        }
        
        summary_path = os.path.join(output_dir, 'thermodynamic_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._generate_text_report(summary, output_dir)
        
        print(f"\nSaved summary: {summary_path}")
    
    def _compute_emergent_constants(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful = [r for r in results if r['success_criteria']['is_success']]
        failed = [r for r in results if not r['success_criteria']['is_success']]
        
        def extract_values(data_list, key_path):
            values = []
            for r in data_list:
                val = r
                for key in key_path:
                    val = val.get(key, {}) if isinstance(val, dict) else {}
                if isinstance(val, (int, float)):
                    values.append(val)
            return values
        
        g_alg_success = extract_values(successful, ['gravitational_constant', 'g_alg'])
        g_alg_failed = extract_values(failed, ['gravitational_constant', 'g_alg'])
        
        k_b_success = extract_values(successful, ['landauer_constant', 'k_b_eff'])
        k_b_failed = extract_values(failed, ['landauer_constant', 'k_b_eff'])
        
        hbar_success = extract_values(successful, ['heisenberg_uncertainty', 'hbar_eff'])
        hbar_failed = extract_values(failed, ['heisenberg_uncertainty', 'hbar_eff'])
        
        kappa_success = extract_values(successful, ['condition_number', 'kappa'])
        kappa_failed = extract_values(failed, ['condition_number', 'kappa'])
        
        pressure_success = extract_values(successful, ['gravitational_constant', 'crystallization_pressure'])
        pressure_failed = extract_values(failed, ['gravitational_constant', 'crystallization_pressure'])
        
        return {
            'gravitational_constant': {
                'successful_mean': float(np.mean(g_alg_success)) if g_alg_success else 0,
                'failed_mean': float(np.mean(g_alg_failed)) if g_alg_failed else 0,
                'universal_value_estimate': float(np.median(g_alg_success)) if g_alg_success else 0,
                'pressure_successful': float(np.mean(pressure_success)) if pressure_success else 0,
                'pressure_failed': float(np.mean(pressure_failed)) if pressure_failed else 0
            },
            'landauer_constant': {
                'successful_mean': float(np.mean(k_b_success)) if k_b_success else 0,
                'failed_mean': float(np.mean(k_b_failed)) if k_b_failed else 0,
                'universal_value_estimate': float(np.median(k_b_success)) if k_b_success else 0
            },
            'heisenberg_constant': {
                'successful_mean': float(np.mean(hbar_success)) if hbar_success else 0,
                'failed_mean': float(np.mean(hbar_failed)) if hbar_failed else 0,
                'universal_value_estimate': float(np.median(hbar_success)) if hbar_success else 0
            },
            'condition_number': {
                'successful_mean': float(np.mean(kappa_success)) if kappa_success else 0,
                'failed_mean': float(np.mean(kappa_failed)) if kappa_failed else 0,
                'log_kappa_success': float(np.mean([np.log10(k) for k in kappa_success if k > 0])) if kappa_success else 0,
                'predictive_threshold': 1.001
            }
        }
    
    def _verify_universal_laws(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        successful = [r for r in results if r['success_criteria']['is_success']]
        
        if not successful:
            return {'laws_verified': False, 'reason': 'no_successful_runs'}
        
        g_alg_values = [r['gravitational_constant']['g_alg'] for r in successful if r['gravitational_constant']['g_alg'] < float('inf')]
        k_b_values = [r['landauer_constant']['k_b_eff'] for r in successful]
        hbar_values = [r['heisenberg_uncertainty']['hbar_eff'] for r in successful]
        
        g_alg_cv = np.std(g_alg_values) / (np.mean(g_alg_values) + 1e-10) if g_alg_values else 1.0
        k_b_cv = np.std(k_b_values) / (np.mean(k_b_values) + 1e-10) if k_b_values else 1.0
        hbar_cv = np.std(hbar_values) / (np.mean(hbar_values) + 1e-10) if hbar_values else 1.0
        
        return {
            'laws_verified': True,
            'gravitational_law': {
                'coefficient_of_variation': float(g_alg_cv),
                'is_constant': g_alg_cv < 0.1,
                'sample_size': len(g_alg_values)
            },
            'landauer_principle': {
                'coefficient_of_variation': float(k_b_cv),
                'is_constant': k_b_cv < 0.1,
                'sample_size': len(k_b_values)
            },
            'heisenberg_uncertainty': {
                'coefficient_of_variation': float(hbar_cv),
                'is_constant': hbar_cv < 0.1,
                'sample_size': len(hbar_values)
            },
            'universality_confidence': float(1.0 - min(max(g_alg_cv, k_b_cv, hbar_cv), 1.0))
        }
    
    def _compute_kappa_correlation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        kappas = []
        successes = []
        
        for r in results:
            kappa = r.get('condition_number', {}).get('kappa', float('inf'))
            if kappa < float('inf'):
                kappas.append(kappa)
                successes.append(1 if r['success_criteria']['is_success'] else 0)
        
        if len(kappas) < 2:
            return {'auc': 0.5, 'correlation': 0.0, 'predictive_power': 'insufficient_data'}
        
        from sklearn.metrics import roc_auc_score
        
        try:
            auc = roc_auc_score(successes, [-k for k in kappas])
        except:
            auc = 0.5
        
        correlation = np.corrcoef(kappas, successes)[0, 1] if len(kappas) > 1 else 0.0
        
        return {
            'auc': float(auc),
            'correlation': float(correlation),
            'predictive_power': 'strong' if auc > 0.9 else 'moderate' if auc > 0.7 else 'weak',
            'kappa_threshold': 1.001,
            'samples': len(kappas)
        }
    
    def _generate_text_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        report_path = os.path.join(output_dir, 'thermodynamic_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("THERMODYNAMIC ANALYSIS OF ALGORITHMIC MATTER\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total checkpoints analyzed: {summary['total_checkpoints_analyzed']}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            constants = summary['emergent_constants']
            
            f.write("-" * 80 + "\n")
            f.write("EMERGENT UNIVERSAL CONSTANTS\n")
            f.write("-" * 80 + "\n")
            
            f.write("\n[GRAVITATIONAL CONSTANT G_alg]\n")
            gc = constants['gravitational_constant']
            f.write(f"  Successful runs mean: {gc['successful_mean']:.6e}\n")
            f.write(f"  Failed runs mean:     {gc['failed_mean']:.6e}\n")
            f.write(f"  Universal estimate:   {gc['universal_value_estimate']:.6e}\n")
            f.write(f"  Pressure (success):   {gc['pressure_successful']:.6e}\n")
            f.write(f"  Pressure (failed):    {gc['pressure_failed']:.6e}\n")
            
            f.write("\n[LANDAUER CONSTANT k_B_eff]\n")
            lc = constants['landauer_constant']
            f.write(f"  Successful runs mean: {lc['successful_mean']:.6e}\n")
            f.write(f"  Failed runs mean:     {lc['failed_mean']:.6e}\n")
            f.write(f"  Universal estimate:   {lc['universal_value_estimate']:.6e}\n")
            
            f.write("\n[HEISENBERG CONSTANT hbar_eff]\n")
            hc = constants['heisenberg_constant']
            f.write(f"  Successful runs mean: {hc['successful_mean']:.6e}\n")
            f.write(f"  Failed runs mean:     {hc['failed_mean']:.6e}\n")
            f.write(f"  Universal estimate:   {hc['universal_value_estimate']:.6e}\n")
            
            f.write("\n[CONDITION NUMBER KAPPA]\n")
            kc = constants['condition_number']
            f.write(f"  Successful runs mean: {kc['successful_mean']:.6f}\n")
            f.write(f"  Failed runs mean:     {kc['failed_mean']:.6f}\n")
            f.write(f"  Log kappa (success):  {kc['log_kappa_success']:.6f}\n")
            f.write(f"  Predictive threshold: {kc['predictive_threshold']:.6f}\n")
            
            kappa_corr = summary['kappa_correlation']
            f.write("\n[KAPPA PREDICTIVE POWER]\n")
            f.write(f"  AUC: {kappa_corr['auc']:.4f}\n")
            f.write(f"  Correlation: {kappa_corr['correlation']:.4f}\n")
            f.write(f"  Predictive power: {kappa_corr['predictive_power']}\n")
            f.write(f"  Samples: {kappa_corr['samples']}\n")
            
            laws = summary['universal_laws']
            f.write("\n" + "-" * 80 + "\n")
            f.write("UNIVERSAL LAWS VERIFICATION\n")
            f.write("-" * 80 + "\n")
            
            if laws.get('laws_verified', False):
                f.write(f"\nGravitational Law: CV={laws['gravitational_law']['coefficient_of_variation']:.4f}, "
                       f"Constant={laws['gravitational_law']['is_constant']}, N={laws['gravitational_law']['sample_size']}\n")
                f.write(f"Landauer Principle: CV={laws['landauer_principle']['coefficient_of_variation']:.4f}, "
                       f"Constant={laws['landauer_principle']['is_constant']}, N={laws['landauer_principle']['sample_size']}\n")
                f.write(f"Heisenberg Uncertainty: CV={laws['heisenberg_uncertainty']['coefficient_of_variation']:.4f}, "
                       f"Constant={laws['heisenberg_uncertainty']['is_constant']}, N={laws['heisenberg_uncertainty']['sample_size']}\n")
                f.write(f"Universality Confidence: {laws['universality_confidence']:.2%}\n")
            else:
                f.write(f"\nLaws not verified: {laws.get('reason', 'unknown')}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("INDIVIDUAL CHECKPOINT ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, r in enumerate(summary['individual_results'], 1):
                f.write(f"[{i}] {r['metadata']['checkpoint_path']}\n")
                f.write(f"    Epoch: {r['metadata']['epoch']}\n")
                f.write(f"    Delta: {r['order_parameter']['delta']:.6f}\n")
                f.write(f"    G_alg: {r['gravitational_constant']['g_alg']:.6e}\n")
                f.write(f"    Pressure: {r['gravitational_constant']['crystallization_pressure']:.6e}\n")
                f.write(f"    k_B_eff: {r['landauer_constant']['k_b_eff']:.6e}\n")
                f.write(f"    hbar_eff: {r['heisenberg_uncertainty']['hbar_eff']:.6e}\n")
                f.write(f"    Kappa: {r['condition_number']['kappa']:.6f}\n")
                f.write(f"    Phase transition: {r['phase_transition']['has_transition']}\n")
                f.write(f"    Success: {r['success_criteria']['is_success']}\n")
                f.write(f"    Failure mode: {r['success_criteria']['failure_mode']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved text report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Thermodynamic analysis of algorithmic crystallization'
    )
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='Path to specific checkpoint file'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all checkpoints in directory'
    )
    parser.add_argument(
        '--latest',
        type=int,
        default=None,
        help='Process only N latest checkpoints'
    )
    parser.add_argument(
        '--dir',
        default='checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--output',
        default='thermodynamic_analysis',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    config = ThermodynamicConfig()
    pipeline = ThermodynamicPipeline(config)
    
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            pipeline.process_checkpoint(args.checkpoint, args.output)
        else:
            print(f"Error: Checkpoint not found: {args.checkpoint}")
    elif args.all or (args.latest is None and args.checkpoint is None):
        n_to_process = args.latest if args.latest is not None else None
        results = pipeline.process_directory(args.dir, n_to_process, args.output)
        if results:
            pipeline.generate_summary(results, args.output)
    elif args.latest:
        results = pipeline.process_directory(args.dir, args.latest, args.output)
        if results:
            pipeline.generate_summary(results, args.output)


if __name__ == '__main__':
    main()