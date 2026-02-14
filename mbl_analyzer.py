
# Remove tqdm dependency and use standard library
import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Protocol, runtime_checkable, Union
from pathlib import Path
import glob
from dataclasses import dataclass, field
from scipy.stats import entropy, gaussian_kde
from scipy.linalg import eigh
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class MBLConfiguration:
    """
    Comprehensive configuration for MBL analysis of Strassen algorithm crystallization.
    All parameters are centralized here following SOLID principles.
    """
    # Architecture dimensions
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    INPUT_DIM: int = 4  # MATRIX_SIZE * MATRIX_SIZE
    
    # MBL Level Spacing Ratio parameters
    LEVEL_SPACING_WIGNER_DYSON: float = 0.5307  # Theoretical value for GOE
    LEVEL_SPACING_POISSON: float = 0.3863  # Theoretical value for Poisson
    LEVEL_SPACING_TOLERANCE: float = 0.05  # Tolerance for phase classification
    
    # Participation Ratio parameters
    PR_LOCALIZATION_THRESHOLD: float = 0.8  # IPR > 0.8 indicates localization
    PR_DELIMITED_THRESHOLD: float = 0.1  # IPR < 0.1 indicates delocalization
    PR_RENYI_INDEX: int = 2  # Standard second Rényi entropy
    
    # Synthetic Planck's constant calculation
    HBAR_ENERGY_GAP_SCALE: float = 1.0  # Scaling factor for energy gap
    HBAR_NUMERICAL_NOISE_FLOOR: float = 1e-7  # Minimum hbar_eff value
    
    # Discretization Dial (delta) parameters
    DISCRETIZATION_BASE: float = 0.00015  # Base value for parity reference
    DISCRETIZATION_NOISE_LEVELS: Tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)
    DISCRETIZATION_GAP_COLLAPSE_THRESHOLD: float = 0.5  # Threshold for gap collapse detection
    
    # Purity analysis parameters (from original code)
    DISCRETIZATION_MARGIN: float = 0.1
    ENTROPY_BINS: int = 50
    TEMPERATURE_WINDOW: int = 100
    SPECIFIC_HEAT_WINDOW: int = 50
    
    PRUNING_LEVELS: Tuple[float, ...] = (0.0, 0.3, 0.5, 0.7, 0.9)
    
    ALPHA_SATURATION: float = 20.0
    ALPHA_THRESHOLD_CRYSTAL: float = 7.0
    ALPHA_THRESHOLD_GLASS: float = 1.0
    
    GLASS_TEMPERATURE_THRESHOLD: float = 0.1
    CRYSTAL_TEMPERATURE_THRESHOLD: float = 0.01
    
    # Checkpoint management
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    CHECKPOINT_KEEP_LATEST: bool = True
    CHECKPOINT_COMPRESSION: bool = False
    
    # Visualization and reporting
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'
    
    # Hardware configuration
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    PRECISION: str = 'float32'
    
    # Training monitoring
    METRIC_PRECISION: int = 6
    
    def get_effective_input_dim(self) -> int:
        return self.MATRIX_SIZE * self.MATRIX_SIZE
    
    def get_total_parameters(self) -> int:
        input_dim = self.get_effective_input_dim()
        return (input_dim * self.HIDDEN_DIM * 2) + (self.HIDDEN_DIM * input_dim)


@runtime_checkable
class IModel(Protocol):
    """Protocol for models compatible with MBL analysis."""
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class ILevelSpacingCalculator(Protocol):
    """Protocol for level spacing ratio calculation."""
    def calculate(self, model: IModel) -> Dict[str, float]: ...


@runtime_checkable
class IParticipationRatioCalculator(Protocol):
    """Protocol for participation ratio calculation."""
    def calculate(self, model: IModel) -> Dict[str, float]: ...


@runtime_checkable
class ISyntheticPlanckCalculator(Protocol):
    """Protocol for synthetic Planck's constant calculation."""
    def calculate(self, participation_ratio: float, energy_gap: float) -> float: ...


@runtime_checkable
class IDiscretizationDialAnalyzer(Protocol):
    """Protocol for discretization dial analysis."""
    def analyze_robustness(self, model: IModel, noise_levels: Tuple[float, ...]) -> Dict[str, Any]: ...


@runtime_checkable
class ICheckpointManager(Protocol):
    """Protocol for checkpoint management."""
    def save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict[str, Any], 
                       loss_history: List[float], path: str) -> None: ...
    def load_checkpoint(self, path: str) -> Dict[str, Any]: ...


@runtime_checkable
class ITrainingMetricsCollector(Protocol):
    """Protocol for collecting all training metrics."""
    def collect(self, model: IModel, loss: float, epoch: int, 
                loss_history: List[float]) -> Dict[str, Any]: ...


class BilinearStrassenModel(nn.Module):
    """
    Bilinear model for Strassen algorithm implementation.
    Represents the 2x2 matrix multiplication with hidden dimension expansion.
    """
    def __init__(self, config: MBLConfiguration):
        super().__init__()
        self.config = config
        input_dim = config.get_effective_input_dim()
        
        self.U = nn.Linear(input_dim, config.HIDDEN_DIM, bias=False)
        self.V = nn.Linear(input_dim, config.HIDDEN_DIM, bias=False)
        self.W = nn.Linear(config.HIDDEN_DIM, input_dim, bias=False)
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier initialization with symmetry constraint for U and V."""
        nn.init.xavier_uniform_(self.U.weight)
        self.V.weight.data = self.U.weight.data.clone()
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward pass implementing bilinear multiplication."""
        return self.W(self.U(a) * self.V(b))
    
    def get_coefficients(self) -> Dict[str, torch.Tensor]:
        """Returns weight matrices for analysis."""
        return {
            'U': self.U.weight.data,
            'V': self.V.weight.data,
            'W': self.W.weight.data
        }
    
    def get_flat_parameters(self) -> torch.Tensor:
        """Returns all parameters flattened for Hamiltonian construction."""
        params = []
        for param in self.parameters():
            params.append(param.flatten())
        return torch.cat(params)
    
    def construct_hessian_approximation(self) -> np.ndarray:
        """
        Constructs approximate Hessian matrix from weight correlations.
        This serves as the 'Hamiltonian' for MBL analysis.
        """
        coeffs = self.get_coefficients()
        
        # Flatten all weights into a single vector
        all_weights = []
        for name in ['U', 'V', 'W']:
            all_weights.append(coeffs[name].flatten())
        
        weight_vector = torch.cat(all_weights).cpu().numpy()
        n = len(weight_vector)
        
        # Construct correlation matrix as Hessian approximation
        # H_ij = <w_i w_j> - <w_i><w_j> (covariance structure)
        weights_2d = weight_vector.reshape(-1, 1)
        hessian = np.outer(weights_2d, weights_2d) / n
        
        # Add regularization for numerical stability
        hessian += np.eye(n) * 1e-8
        
        return hessian


class LevelSpacingRatioCalculator:
    """
    Calculates the level spacing ratio r for MBL phase detection.
    
    The ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})
    where delta_n = E_{n+1} - E_n (energy level spacing).
    
    References:
    - Oganesyan & Huse (2008): r_WD ≈ 0.53 (Wigner-Dyson, thermal)
    - Poisson statistics: r_P ≈ 0.386 (MBL/localized phase)
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
    
    def calculate(self, model: IModel) -> Dict[str, float]:
        """
        Calculate level spacing statistics from model weights.
        
        Returns:
            Dictionary containing mean ratio, variance, and phase classification.
        """
        # Construct effective Hamiltonian from model weights
        if isinstance(model, BilinearStrassenModel):
            hessian = model.construct_hessian_approximation()
        else:
            hessian = self._construct_hessian_from_weights(model)
        
        # Compute eigenvalues (energy levels)
        eigenvalues = self._compute_eigenvalues(hessian)
        
        # Calculate level spacings
        spacings = np.diff(sorted(eigenvalues))
        
        # Calculate spacing ratios
        ratios = self._calculate_spacing_ratios(spacings)
        
        # Statistical analysis
        mean_ratio = np.mean(ratios)
        std_ratio = np.std(ratios)
        
        # Phase classification
        phase = self._classify_phase(mean_ratio)
        
        # Distribution fitting
        wigner_dyson_distance = abs(mean_ratio - self.config.LEVEL_SPACING_WIGNER_DYSON)
        poisson_distance = abs(mean_ratio - self.config.LEVEL_SPACING_POISSON)
        
        return {
            'mean_spacing_ratio': float(mean_ratio),
            'std_spacing_ratio': float(std_ratio),
            'phase_classification': phase,
            'wigner_dyson_distance': float(wigner_dyson_distance),
            'poisson_distance': float(poisson_distance),
            'is_localized': phase == 'many_body_localized',
            'is_thermal': phase == 'thermal',
            'num_levels': len(eigenvalues),
            'energy_spectrum_range': float(np.max(eigenvalues) - np.min(eigenvalues)),
            'min_spacing': float(np.min(spacings)),
            'max_spacing': float(np.max(spacings))
        }
    
    def _construct_hessian_from_weights(self, model: IModel) -> np.ndarray:
        """Alternative Hessian construction for generic models."""
        coeffs = model.get_coefficients()
        all_weights = []
        for tensor in coeffs.values():
            all_weights.append(tensor.flatten().cpu().numpy())
        
        weight_vector = np.concatenate(all_weights)
        n = len(weight_vector)
        
        # Use weight covariance as Hamiltonian proxy
        hessian = np.outer(weight_vector, weight_vector) / n
        hessian += np.eye(n) * 1e-8
        
        return hessian
    
    def _compute_eigenvalues(self, hessian: np.ndarray) -> np.ndarray:
        """Compute sorted eigenvalues of the Hamiltonian."""
        eigenvalues = eigh(hessian, eigvals_only=True)
        return np.sort(eigenvalues)
    
    def _calculate_spacing_ratios(self, spacings: np.ndarray) -> np.ndarray:
        """
        Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).
        """
        ratios = []
        for i in range(len(spacings) - 1):
            s_n = spacings[i]
            s_n_plus_1 = spacings[i + 1]
            
            if max(s_n, s_n_plus_1) > 1e-15:  # Avoid division by zero
                r = min(s_n, s_n_plus_1) / max(s_n, s_n_plus_1)
                ratios.append(r)
        
        return np.array(ratios) if ratios else np.array([0.0])
    
    def _classify_phase(self, mean_ratio: float) -> str:
        """
        Classify the quantum phase based on level spacing ratio.
        """
        wd = self.config.LEVEL_SPACING_WIGNER_DYSON
        poisson = self.config.LEVEL_SPACING_POISSON
        tol = self.config.LEVEL_SPACING_TOLERANCE
        
        if abs(mean_ratio - wd) < tol:
            return 'thermal'
        elif abs(mean_ratio - poisson) < tol:
            return 'many_body_localized'
        elif mean_ratio < (wd + poisson) / 2:
            return 'intermediate_localized'
        else:
            return 'intermediate_thermal'


class ParticipationRatioCalculator:
    """
    Calculates Inverse Participation Ratio (IPR) for localization analysis.
    
    IPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.
    IPR = 1 for fully localized state, IPR = 1/N for fully delocalized state.
    
    Used to quantify the 'crystallinity' of the weight distribution.
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
    
    def calculate(self, model: IModel) -> Dict[str, float]:
        """
        Calculate participation ratios for all weight layers.
        
        Returns:
            Dictionary containing global and layer-wise IPR metrics.
        """
        coeffs = model.get_coefficients()
        
        layer_iprs = {}
        global_weights = []
        
        for name, weights in coeffs.items():
            weights_np = weights.flatten().cpu().numpy()
            
            # Calculate IPR for this layer
            ipr = self._calculate_ipr(weights_np)
            layer_iprs[name] = {
                'ipr': float(ipr),
                'localization_length': float(1.0 / max(ipr, 1e-15)),
                'num_parameters': len(weights_np)
            }
            
            global_weights.append(weights_np)
        
        # Global IPR across all parameters
        global_weights_concat = np.concatenate(global_weights)
        global_ipr = self._calculate_ipr(global_weights_concat)
        
        # Generalized IPR (Rényi entropy based)
        renyi_ipr = self._calculate_renyi_ipr(global_weights_concat, self.config.PR_RENYI_INDEX)
        
        # Fractal dimension
        fractal_dim = self._calculate_fractal_dimension(global_ipr, len(global_weights_concat))
        
        return {
            'global_ipr': float(global_ipr),
            'global_localization_length': float(1.0 / max(global_ipr, 1e-15)),
            'renyi_ipr': float(renyi_ipr),
            'fractal_dimension': float(fractal_dim),
            'layer_iprs': layer_iprs,
            'total_parameters': len(global_weights_concat),
            'is_localized': global_ipr > self.config.PR_LOCALIZATION_THRESHOLD,
            'is_delocalized': global_ipr < self.config.PR_DELIMITED_THRESHOLD
        }
    
    def _calculate_ipr(self, coefficients: np.ndarray) -> float:
        """
        Calculate standard Inverse Participation Ratio.
        IPR = sum_i |c_i|^4 / (sum_i |c_i|^2)^2
        """
        # Normalize coefficients
        norm = np.sum(np.abs(coefficients) ** 2)
        if norm < 1e-15:
            return 0.0
        
        normalized = coefficients / np.sqrt(norm)
        ipr = np.sum(np.abs(normalized) ** 4)
        
        return ipr
    
    def _calculate_renyi_ipr(self, coefficients: np.ndarray, q: int) -> float:
        """
        Calculate q-th order Rényi IPR.
        I_q = sum_i |c_i|^{2q} / (sum_i |c_i|^2)^q
        """
        norm = np.sum(np.abs(coefficients) ** 2)
        if norm < 1e-15:
            return 0.0
        
        normalized = coefficients / np.sqrt(norm)
        renyi_ipr = np.sum(np.abs(normalized) ** (2 * q))
        
        return renyi_ipr
    
    def _calculate_fractal_dimension(self, ipr: float, n: int) -> float:
        """
        Calculate fractal dimension D_q from IPR.
        IPR ~ N^{-D_q} => D_q = -log(IPR) / log(N)
        """
        if n <= 1 or ipr <= 0:
            return 0.0
        
        return -np.log(ipr) / np.log(n)


class SyntheticPlanckConstantCalculator:
    """
    Calculates effective synthetic Planck's constant (hbar_eff) from model properties.
    
    Based on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)
    where PR is the Participation Ratio and Energy_Gap is the spectral gap.
    
    This represents the quantum of action in the synthetic quantum system.
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
    
    def calculate(self, participation_ratio: float, energy_gap: float) -> float:
        """
        Calculate synthetic Planck's constant.
        
        Args:
            participation_ratio: Inverse participation ratio (measure of localization)
            energy_gap: Energy gap from spectrum (measure of quantum discreteness)
        
        Returns:
            Synthetic hbar value representing the quantum scale of the system.
        """
        if participation_ratio < 1e-15 or energy_gap < 1e-15:
            return self.config.HBAR_NUMERICAL_NOISE_FLOOR
        
        # hbar_eff ∝ 1 / sqrt(PR * Delta_E)
        # Higher PR (more localized) and larger gap -> smaller hbar (more quantum)
        hbar = 1.0 / np.sqrt(participation_ratio * energy_gap * self.config.HBAR_ENERGY_GAP_SCALE)
        
        # Apply numerical floor
        hbar = max(hbar, self.config.HBAR_NUMERICAL_NOISE_FLOOR)
        
        return float(hbar)
    
    def calculate_from_model(self, model: IModel, 
                            level_spacing_results: Dict[str, float],
                            pr_results: Dict[str, float]) -> Dict[str, float]:
        """
        Comprehensive calculation from model and previous analyses.
        """
        energy_gap = level_spacing_results.get('min_spacing', 1e-8)
        participation_ratio = pr_results.get('global_ipr', 1.0)
        
        hbar_eff = self.calculate(participation_ratio, energy_gap)
        
        # Calculate uncertainty relation proxy: Delta_E * Delta_t >= hbar/2
        # Here we use temperature as proxy for time uncertainty
        temperature_proxy = 1.0 / max(participation_ratio, 1e-15)
        uncertainty_product = energy_gap * temperature_proxy
        
        # Quantum coherence length
        coherence_length = 1.0 / np.sqrt(participation_ratio)
        
        return {
            'hbar_eff': float(hbar_eff),
            'energy_gap': float(energy_gap),
            'participation_ratio': float(participation_ratio),
            'uncertainty_product': float(uncertainty_product),
            'coherence_length': float(coherence_length),
            'is_quantum_regime': hbar_eff < 0.1,  # Arbitrary threshold for quantum vs classical
            'localization_length': 1.0 / max(participation_ratio, 1e-15)
        }


class DiscretizationDialAnalyzer:
    """
    Analyzes the discretization parameter delta as a phase transition control.
    
    The discretization delta measures how close weights are to discrete values.
    It acts as a "dial" that controls the quantum-classical transition.
    
    This implements the noise robustness test: applying Gaussian perturbations
    and measuring when the energy gap collapses (loss of quantum protection).
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
        self.level_spacing_calculator = LevelSpacingRatioCalculator(config)
    
    def calculate_base_discretization(self, model: IModel) -> Dict[str, float]:
        """
        Calculate the base discretization level from weight rounding error.
        """
        coeffs = model.get_coefficients()
        
        layer_deltas = {}
        max_delta = 0.0
        
        for name, weights in coeffs.items():
            weights_rounded = torch.round(weights)
            delta = torch.max(torch.abs(weights - weights_rounded)).item()
            layer_deltas[name] = float(delta)
            max_delta = max(max_delta, delta)
        
        # Convert delta to alpha (purity index)
        alpha = self._delta_to_alpha(max_delta)
        
        return {
            'global_delta': float(max_delta),
            'global_alpha': float(alpha),
            'layer_deltas': layer_deltas,
            'is_discretized': max_delta < self.config.DISCRETIZATION_MARGIN,
            'discretization_quality': 'high' if alpha > self.config.ALPHA_THRESHOLD_CRYSTAL else 
                                     ('medium' if alpha > self.config.ALPHA_THRESHOLD_GLASS else 'low')
        }
    
    def analyze_robustness(self, model: IModel, 
                          noise_levels: Optional[Tuple[float, ...]] = None) -> Dict[str, Any]:
        """
        Test robustness by applying noise and measuring gap collapse.
        
        Args:
            model: The neural network model
            noise_levels: Tuple of noise magnitudes to test
        
        Returns:
            Dictionary containing robustness metrics and phase transition points.
        """
        if noise_levels is None:
            noise_levels = self.config.DISCRETIZATION_NOISE_LEVELS
        
        base_results = self.calculate_base_discretization(model)
        base_delta = base_results['global_delta']
        
        robustness_data = []
        collapse_point = None
        
        for noise_level in noise_levels:
            # Perturb model with Gaussian noise
            perturbed_metrics = self._perturb_and_measure(model, noise_level)
            
            # Check for gap collapse
            gap_ratio = perturbed_metrics['energy_gap'] / max(base_results['global_delta'], 1e-15)
            is_collapsed = gap_ratio < self.config.DISCRETIZATION_GAP_COLLAPSE_THRESHOLD
            
            if is_collapsed and collapse_point is None:
                collapse_point = noise_level
            
            robustness_data.append({
                'noise_level': float(noise_level),
                'spacing_ratio': float(perturbed_metrics['spacing_ratio']),
                'energy_gap': float(perturbed_metrics['energy_gap']),
                'gap_ratio': float(gap_ratio),
                'is_collapsed': bool(is_collapsed),
                'phase': perturbed_metrics['phase']
            })
        
        # Calculate topological protection metric
        if collapse_point is not None:
            protection_strength = collapse_point / max(base_delta, 1e-15)
        else:
            protection_strength = max(noise_levels) / max(base_delta, 1e-15)
        
        return {
            'base_discretization': base_results,
            'robustness_curve': robustness_data,
            'collapse_point': float(collapse_point) if collapse_point else None,
            'protection_strength': float(protection_strength),
            'is_topologically_protected': protection_strength > 10.0,
            'noise_levels_tested': list(noise_levels)
        }
    
    def _perturb_and_measure(self, model: IModel, noise_level: float) -> Dict[str, float]:
        """Apply noise to model and measure resulting metrics."""
        # Store original state
        original_state = {name: param.clone() for name, param in model.named_parameters()}
        
        # Apply Gaussian noise
        with torch.no_grad():
            for param in model.parameters():
                noise = torch.randn_like(param) * noise_level
                param.add_(noise)
        
        # Measure level spacing after perturbation
        spacing_results = self.level_spacing_calculator.calculate(model)
        
        # Restore original state
        model.load_state_dict(original_state)
        
        return {
            'spacing_ratio': spacing_results['mean_spacing_ratio'],
            'energy_gap': spacing_results.get('min_spacing', 1e-8),
            'phase': spacing_results['phase_classification']
        }
    
    def _delta_to_alpha(self, delta: float) -> float:
        """Convert discretization error to purity alpha."""
        if delta < 1e-15:
            return self.config.ALPHA_SATURATION
        return -np.log(delta + 1e-15)


class PurityIndexCalculator:
    """
    Original purity calculation preserved exactly as in user's code.
    Calculates the 'crystallinity' of the weight distribution.
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
    
    def calculate(self, model: IModel) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        
        layer_alphas = {}
        global_deltas = []
        
        for name, weights in coeffs.items():
            layer_alpha, layer_delta = self._compute_layer_purity(weights)
            layer_alphas[name] = layer_alpha
            global_deltas.append(layer_delta)
        
        global_delta = max(global_deltas) if global_deltas else 1.0
        global_alpha = self._delta_to_alpha(global_delta)
        
        alpha_variance = np.var(list(layer_alphas.values())) if layer_alphas else 0.0
        alpha_mean = np.mean(list(layer_alphas.values())) if layer_alphas else 0.0
        
        purity_quality = self._assess_purity_quality(global_alpha, alpha_variance)
        
        return {
            'global_alpha': global_alpha,
            'global_delta': global_delta,
            'layer_alphas': layer_alphas,
            'alpha_variance': alpha_variance,
            'alpha_mean': alpha_mean,
            'purity_quality': purity_quality,
            'is_homogeneous': alpha_variance < 0.1
        }
    
    def _compute_layer_purity(self, weights: torch.Tensor) -> Tuple[float, float]:
        rounded = torch.round(weights)
        delta = torch.max(torch.abs(weights - rounded)).item()
        alpha = self._delta_to_alpha(delta)
        return alpha, delta
    
    def _delta_to_alpha(self, delta: float) -> float:
        if delta < 1e-15:
            return self.config.ALPHA_SATURATION
        return -np.log(delta + 1e-15)
    
    def _assess_purity_quality(self, alpha: float, variance: float) -> str:
        if alpha > self.config.ALPHA_THRESHOLD_CRYSTAL and variance < 0.1:
            return 'high_purity_crystal'
        elif alpha > self.config.ALPHA_THRESHOLD_CRYSTAL:
            return 'crystal_with_defects'
        elif alpha > self.config.ALPHA_THRESHOLD_GLASS:
            return 'transitional_phase'
        else:
            return 'low_purity_glass'


class EffectiveTemperatureCalculator:
    """
    Original temperature calculation preserved exactly.
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
    
    def calculate(self, loss_history: List[float]) -> Dict[str, float]:
        if len(loss_history) < self.config.TEMPERATURE_WINDOW:
            return {
                'temperature': 0.0,
                'specific_heat': 0.0,
                'thermal_energy': 0.0,
                'entropy_production': 0.0,
                'is_equilibrated': False
            }
        
        recent_losses = loss_history[-self.config.TEMPERATURE_WINDOW:]
        
        temperature = np.var(recent_losses)
        
        if len(loss_history) >= self.config.SPECIFIC_HEAT_WINDOW * 2:
            recent = loss_history[-self.config.SPECIFIC_HEAT_WINDOW:]
            previous = loss_history[-(self.config.SPECIFIC_HEAT_WINDOW * 2):-self.config.SPECIFIC_HEAT_WINDOW]
            specific_heat = np.var(recent) - np.var(previous)
        else:
            specific_heat = 0.0
        
        thermal_energy = np.mean(recent_losses)
        
        if len(recent_losses) > 1:
            entropy_production = np.sum(np.diff(recent_losses) ** 2)
        else:
            entropy_production = 0.0
        
        is_equilibrated = temperature < self.config.CRYSTAL_TEMPERATURE_THRESHOLD
        
        return {
            'temperature': float(temperature),
            'specific_heat': float(specific_heat),
            'thermal_energy': float(thermal_energy),
            'entropy_production': float(entropy_production),
            'is_equilibrated': bool(is_equilibrated)
        }


class PhaseClassifier:
    """
    Original phase classification preserved exactly.
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
    
    def classify(self, alpha: float, temperature: float) -> str:
        if alpha > self.config.ALPHA_THRESHOLD_CRYSTAL and temperature < self.config.CRYSTAL_TEMPERATURE_THRESHOLD:
            return 'perfect_crystal'
        elif alpha > self.config.ALPHA_THRESHOLD_CRYSTAL and temperature < self.config.GLASS_TEMPERATURE_THRESHOLD:
            return 'crystal_with_thermal_fluctuations'
        elif alpha > self.config.ALPHA_THRESHOLD_CRYSTAL:
            return 'hot_crystal'
        elif alpha > self.config.ALPHA_THRESHOLD_GLASS and temperature < self.config.CRYSTAL_TEMPERATURE_THRESHOLD:
            return 'cold_polycrystal'
        elif alpha > self.config.ALPHA_THRESHOLD_GLASS:
            return 'warm_polycrystal'
        elif temperature < self.config.CRYSTAL_TEMPERATURE_THRESHOLD:
            return 'cold_glass'
        else:
            return 'hot_glass'


class CheckpointMigrator:
    """
    Original checkpoint migration logic preserved exactly.
    """
    
    def migrate(self, raw_data: Any, device: str) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(raw_data, dict):
            if 'state_dict' in raw_data:
                return self._migrate_dict(raw_data['state_dict'], device)
            elif 'model_state_dict' in raw_data:
                return self._migrate_dict(raw_data['model_state_dict'], device)
            else:
                return self._migrate_dict(raw_data, device)
        return None
    
    def _migrate_dict(self, state_dict: Dict[str, Any], device: str) -> Optional[Dict[str, torch.Tensor]]:
        if any(k in state_dict for k in ['U', 'V', 'W']):
            return self._migrate_custom_format(state_dict, device)
        elif 'U_coefs' in state_dict:
            return self._migrate_coefs_format(state_dict)
        elif any(k.endswith('.weight') for k in state_dict.keys()):
            return self._migrate_standard_format(state_dict)
        return None
    
    def _migrate_custom_format(self, state_dict: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
        U = state_dict.get('U', state_dict.get('U_coefs'))
        V = state_dict.get('V', state_dict.get('V_coefs'))
        W = state_dict.get('W', state_dict.get('W_coefs'))
        
        if U is None or V is None or W is None:
            return None
        
        if U.shape == (7, 4):
            u_padded = torch.zeros(8, 4, device=device)
            v_padded = torch.zeros(8, 4, device=device)
            w_padded = torch.zeros(4, 8, device=device)
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


class MBLCheckpointManager:
    """
    Manages checkpoint saving with 5-minute intervals and latest file maintenance.
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
        self.last_checkpoint_time = 0
        self.checkpoint_counter = 0
    
    def should_save_checkpoint(self) -> bool:
        """Check if 5 minutes have elapsed since last checkpoint."""
        current_time = time.time()
        elapsed_minutes = (current_time - self.last_checkpoint_time) / 60.0
        return elapsed_minutes >= self.config.CHECKPOINT_INTERVAL_MINUTES
    
    def save_checkpoint(self, model: nn.Module, epoch: int, metrics: Dict[str, Any], 
                       loss_history: List[float], checkpoint_dir: str) -> str:
        """
        Save checkpoint with all MBL metrics.
        """
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        checkpoint_data = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'metrics': metrics,
            'loss_history': loss_history,
            'timestamp': datetime.now().isoformat(),
            'config': {
                'hidden_dim': self.config.HIDDEN_DIM,
                'matrix_size': self.config.MATRIX_SIZE
            }
        }
        
        # Save timestamped checkpoint
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch{epoch}_{timestamp}.pt')
        torch.save(checkpoint_data, checkpoint_path)
        
        # Update latest checkpoint
        if self.config.CHECKPOINT_KEEP_LATEST:
            latest_path = os.path.join(checkpoint_dir, 'checkpoint_latest.pt')
            torch.save(checkpoint_data, latest_path)
        
        self.last_checkpoint_time = time.time()
        self.checkpoint_counter += 1
        
        return checkpoint_path
    
    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load checkpoint with automatic device placement."""
        return torch.load(path, map_location=self.config.DEVICE, weights_only=False)


class MBLMetricsCollector:
    """
    Collects all MBL metrics for comprehensive training monitoring.
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
        self.level_spacing_calc = LevelSpacingRatioCalculator(config)
        self.pr_calc = ParticipationRatioCalculator(config)
        self.hbar_calc = SyntheticPlanckConstantCalculator(config)
        self.dial_analyzer = DiscretizationDialAnalyzer(config)
        self.purity_calc = PurityIndexCalculator(config)
        self.temp_calc = EffectiveTemperatureCalculator(config)
        self.phase_classifier = PhaseClassifier(config)
    
    def collect(self, model: IModel, loss: float, epoch: int, 
                loss_history: List[float]) -> Dict[str, Any]:
        """
        Collect all metrics for the current training state.
        """
        # Core MBL metrics
        level_spacing = self.level_spacing_calc.calculate(model)
        participation_ratio = self.pr_calc.calculate(model)
        
        # Derived quantum metrics
        hbar_results = self.hbar_calc.calculate_from_model(
            model, level_spacing, participation_ratio
        )
        
        # Discretization analysis
        dial_results = self.dial_analyzer.calculate_base_discretization(model)
        
        # Original purity metrics
        purity = self.purity_calc.calculate(model)
        temperature = self.temp_calc.calculate(loss_history)
        
        # Phase classification
        phase = self.phase_classifier.classify(purity['global_alpha'], temperature['temperature'])
        
        # Combined quantum phase
        quantum_phase = self._classify_quantum_phase(level_spacing, hbar_results)
        
        return {
            'epoch': epoch,
            'loss': float(loss),
            'timestamp': datetime.now().isoformat(),
            
            # MBL Level Spacing
            'level_spacing_ratio': level_spacing['mean_spacing_ratio'],
            'level_spacing_std': level_spacing['std_spacing_ratio'],
            'spectral_phase': level_spacing['phase_classification'],
            'is_localized_spectrum': level_spacing['is_localized'],
            'energy_spectrum_range': level_spacing['energy_spectrum_range'],
            
            # Participation Ratio
            'global_ipr': participation_ratio['global_ipr'],
            'localization_length': participation_ratio['global_localization_length'],
            'fractal_dimension': participation_ratio['fractal_dimension'],
            'is_weight_localized': participation_ratio['is_localized'],
            
            # Synthetic Planck's Constant
            'hbar_eff': hbar_results['hbar_eff'],
            'quantum_coherence_length': hbar_results['coherence_length'],
            'uncertainty_product': hbar_results['uncertainty_product'],
            'is_quantum_regime': hbar_results['is_quantum_regime'],
            
            # Discretization Dial
            'discretization_delta': dial_results['global_delta'],
            'discretization_alpha': dial_results['global_alpha'],
            'is_discretized': dial_results['is_discretized'],
            
            # Original purity metrics
            'purity_alpha': purity['global_alpha'],
            'purity_delta': purity['global_delta'],
            'temperature': temperature['temperature'],
            'specific_heat': temperature['specific_heat'],
            'is_equilibrated': temperature['is_equilibrated'],
            
            # Phase classifications
            'crystallization_phase': phase,
            'quantum_phase': quantum_phase,
            
            # Detailed results
            'level_spacing_details': level_spacing,
            'participation_ratio_details': participation_ratio,
            'hbar_details': hbar_results,
            'dial_details': dial_results,
            'purity_details': purity,
            'temperature_details': temperature
        }
    
    def _classify_quantum_phase(self, level_spacing: Dict[str, float], 
                               hbar_results: Dict[str, float]) -> str:
        """Classify the combined quantum phase."""
        is_localized = level_spacing['is_localized']
        is_quantum = hbar_results['is_quantum_regime']
        
        if is_localized and is_quantum:
            return 'many_body_localized_quantum'
        elif is_localized:
            return 'classical_localized'
        elif is_quantum:
            return 'quantum_extended'
        else:
            return 'classical_extended'


class MBLCheckpointAnalyzer:
    """
    Comprehensive analyzer for MBL metrics from checkpoints.
    """
    
    def __init__(self, checkpoint_path: str, config: MBLConfiguration):
        self.config = config
        self.checkpoint_path = checkpoint_path
        self.migrator = CheckpointMigrator()
        self.metrics_collector = MBLMetricsCollector(config)
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        """Load and migrate checkpoint to model."""
        try:
            self.checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.config.DEVICE,
                weights_only=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        self.model = BilinearStrassenModel(self.config).to(self.config.DEVICE)
        
        state_dict = self.migrator.migrate(self.checkpoint, self.config.DEVICE)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f"Failed to migrate checkpoint: {self.checkpoint_path}")
        
        self.epoch = self.checkpoint.get('epoch', 'unknown')
        self.loss_history = self.checkpoint.get('loss_history', [])
        self.current_loss = self.loss_history[-1] if self.loss_history else 0.0
    
    def analyze(self) -> Dict[str, Any]:
        """Perform complete MBL analysis on checkpoint."""
        metrics = self.metrics_collector.collect(
            self.model, self.current_loss, self.epoch, self.loss_history
        )
        
        # Additional robustness analysis
        dial_analyzer = DiscretizationDialAnalyzer(self.config)
        robustness = dial_analyzer.analyze_robustness(self.model)
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat(),
                'config': {
                    'hidden_dim': self.config.HIDDEN_DIM,
                    'matrix_size': self.config.MATRIX_SIZE
                }
            },
            'mbl_metrics': metrics,
            'robustness_analysis': robustness,
            'summary': self._generate_summary(metrics, robustness)
        }
        
        self._print_report(results)
        return results
    
    def _generate_summary(self, metrics: Dict[str, Any], 
                         robustness: Dict[str, Any]) -> Dict[str, Any]:
        """Generate executive summary of analysis."""
        return {
            'is_mbl_phase': metrics['is_localized_spectrum'],
            'is_quantum_crystal': metrics['is_quantum_regime'] and metrics['is_weight_localized'],
            'discretization_quality': robustness['base_discretization']['discretization_quality'],
            'topological_protection': robustness.get('is_topologically_protected', False),
            'key_metrics': {
                'level_spacing_ratio': metrics['level_spacing_ratio'],
                'global_ipr': metrics['global_ipr'],
                'hbar_eff': metrics['hbar_eff'],
                'discretization_delta': metrics['discretization_delta'],
                'purity_alpha': metrics['purity_alpha']
            }
        }
    
    def _print_report(self, results: Dict[str, Any]):
        """Print formatted analysis report."""
        print("=" * 80)
        print("MANY-BODY LOCALIZATION ANALYSIS REPORT")
        print("Strassen Algorithm Crystallization Diagnostics")
        print("=" * 80)
        
        meta = results['metadata']
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {meta['checkpoint_path']}")
        print(f"  Epoch: {meta['epoch']}")
        print(f"  Hidden Dim: {meta['config']['hidden_dim']}")
        print(f"  Matrix Size: {meta['config']['matrix_size']}")
        
        mbl = results['mbl_metrics']
        print(f"\n[LEVEL SPACING ANALYSIS - MBL DETECTION]")
        print(f"  Mean Spacing Ratio: {mbl['level_spacing_ratio']:.6f}")
        print(f"  Standard Deviation: {mbl['level_spacing_std']:.6f}")
        print(f"  Spectral Phase: {mbl['spectral_phase']}")
        print(f"  Is Localized (MBL): {mbl['is_localized_spectrum']}")
        print(f"  Energy Spectrum Range: {mbl['energy_spectrum_range']:.6e}")
        
        print(f"\n[PARTICIPATION RATIO - LOCALIZATION MEASURE]")
        print(f"  Global IPR: {mbl['global_ipr']:.6f}")
        print(f"  Localization Length: {mbl['localization_length']:.6f}")
        print(f"  Fractal Dimension: {mbl['fractal_dimension']:.6f}")
        print(f"  Weights Localized: {mbl['is_weight_localized']}")
        
        print(f"\n[SYNTHETIC PLANCK'S CONSTANT - QUANTUM SCALE]")
        print(f"  hbar_eff: {mbl['hbar_eff']:.6e}")
        print(f"  Coherence Length: {mbl['quantum_coherence_length']:.6f}")
        print(f"  Uncertainty Product: {mbl['uncertainty_product']:.6e}")
        print(f"  Quantum Regime: {mbl['is_quantum_regime']}")
        
        print(f"\n[DISCRETIZATION DIAL - PHASE CONTROL]")
        print(f"  Delta: {mbl['discretization_delta']:.6e}")
        print(f"  Alpha: {mbl['discretization_alpha']:.6f}")
        print(f"  Is Discretized: {mbl['is_discretized']}")
        
        print(f"\n[THERMODYNAMIC STATE]")
        print(f"  Purity Alpha: {mbl['purity_alpha']:.6f}")
        print(f"  Temperature: {mbl['temperature']:.6e}")
        print(f"  Specific Heat: {mbl['specific_heat']:.6e}")
        print(f"  Is Equilibrated: {mbl['is_equilibrated']}")
        
        print(f"\n[PHASE CLASSIFICATION]")
        print(f"  Crystallization Phase: {mbl['crystallization_phase']}")
        print(f"  Quantum Phase: {mbl['quantum_phase']}")
        
        rob = results['robustness_analysis']
        print(f"\n[ROBUSTNESS ANALYSIS]")
        print(f"  Protection Strength: {rob['protection_strength']:.2f}")
        print(f"  Topologically Protected: {rob['is_topologically_protected']}")
        if rob['collapse_point']:
            print(f"  Gap Collapse Point: {rob['collapse_point']:.4f}")
        
        print("=" * 80)


class MBLAnalysisPipeline:
    """
    Main pipeline for processing checkpoints and generating reports.
    """
    
    def __init__(self, config: MBLConfiguration):
        self.config = config
    
    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        """Process single checkpoint and save results."""
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer = MBLCheckpointAnalyzer(checkpoint_path, self.config)
        results = analyzer.analyze()
        
        base_name = Path(checkpoint_path).stem
        results_path = os.path.join(output_dir, f'{base_name}_mbl_analysis.json')
        
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def process_directory(self, checkpoint_dir: str, n_latest: Optional[int], 
                         output_dir: str) -> List[Dict[str, Any]]:
        """Process multiple checkpoints from directory."""
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
        for i, cp in enumerate(checkpoints):
            print(f"[{i+1}/{len(checkpoints)}] Processing {cp}...")
            try:
                results = self.process_checkpoint(cp, output_dir)
                all_results.append(results)
            except Exception as e:
                print(f"Error processing {cp}: {e}")
                import traceback
                traceback.print_exc()
        
        return all_results
    
    def generate_summary(self, all_results: List[Dict[str, Any]], output_dir: str) -> None:
        """Generate aggregate summary report."""
        if not all_results:
            print("No results to summarize")
            return
        
        # Aggregate statistics
        mbl_phases = [r['mbl_metrics']['is_localized_spectrum'] for r in all_results]
        quantum_phases = [r['mbl_metrics']['is_quantum_regime'] for r in all_results]
        
        spacing_ratios = [r['mbl_metrics']['level_spacing_ratio'] for r in all_results]
        iprs = [r['mbl_metrics']['global_ipr'] for r in all_results]
        hbars = [r['mbl_metrics']['hbar_eff'] for r in all_results]
        
        summary = {
            'total_checkpoints': len(all_results),
            'mbl_phase_count': sum(mbl_phases),
            'quantum_regime_count': sum(quantum_phases),
            'mean_spacing_ratio': float(np.mean(spacing_ratios)),
            'std_spacing_ratio': float(np.std(spacing_ratios)),
            'mean_ipr': float(np.mean(iprs)),
            'mean_hbar': float(np.mean(hbars)),
            'timestamp': datetime.now().isoformat(),
            'individual_results': all_results
        }
        
        summary_path = os.path.join(output_dir, 'mbl_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._generate_text_report(summary, output_dir)
        print(f"\nSaved summary: {summary_path}")
    
    def _generate_text_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        """Generate human-readable text report."""
        report_path = os.path.join(output_dir, 'mbl_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MANY-BODY LOCALIZATION ANALYSIS SUMMARY\n")
            f.write("Strassen Algorithm Crystallization Study\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total Checkpoints Analyzed: {summary['total_checkpoints']}\n")
            f.write(f"MBL Phase Detected: {summary['mbl_phase_count']} ({summary['mbl_phase_count']/summary['total_checkpoints']*100:.1f}%)\n")
            f.write(f"Quantum Regime: {summary['quantum_regime_count']} ({summary['quantum_regime_count']/summary['total_checkpoints']*100:.1f}%)\n")
            f.write(f"Mean Level Spacing Ratio: {summary['mean_spacing_ratio']:.6f} ± {summary['std_spacing_ratio']:.6f}\n")
            f.write(f"Mean IPR: {summary['mean_ipr']:.6f}\n")
            f.write(f"Mean hbar_eff: {summary['mean_hbar']:.6e}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("INDIVIDUAL CHECKPOINT ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            for i, r in enumerate(summary['individual_results'], 1):
                mbl = r['mbl_metrics']
                f.write(f"[{i}] {r['metadata']['checkpoint_path']}\n")
                f.write(f"    Epoch: {r['metadata']['epoch']}\n")
                f.write(f"    Level Spacing: {mbl['level_spacing_ratio']:.6f} ({mbl['spectral_phase']})\n")
                f.write(f"    IPR: {mbl['global_ipr']:.6f}\n")
                f.write(f"    hbar_eff: {mbl['hbar_eff']:.6e}\n")
                f.write(f"    Delta: {mbl['discretization_delta']:.6e}\n")
                f.write(f"    Phase: {mbl['crystallization_phase']} | {mbl['quantum_phase']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved text report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Many-Body Localization Analysis for Strassen Algorithm Checkpoints'
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
        default='mbl_analysis',
        help='Output directory for results'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=8,
        help='Hidden dimension'
    )
    parser.add_argument(
        '--matrix-size',
        type=int,
        default=2,
        help='Matrix size (2 for 2x2 Strassen)'
    )
    parser.add_argument(
        '--robustness-test',
        action='store_true',
        help='Perform noise robustness analysis'
    )
    
    args = parser.parse_args()
    
    config = MBLConfiguration(
        HIDDEN_DIM=args.hidden_dim,
        MATRIX_SIZE=args.matrix_size
    )
    
    pipeline = MBLAnalysisPipeline(config)
    
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            results = pipeline.process_checkpoint(args.checkpoint, args.output)
            
            if args.robustness_test:
                print("\nPerforming robustness analysis...")
                analyzer = MBLCheckpointAnalyzer(args.checkpoint, config)
                dial_analyzer = DiscretizationDialAnalyzer(config)
                robustness = dial_analyzer.analyze_robustness(analyzer.model)
                
                robust_path = os.path.join(args.output, 'robustness_analysis.json')
                with open(robust_path, 'w') as f:
                    json.dump(robustness, f, indent=2, default=str)
                print(f"Saved robustness analysis: {robust_path}")
        else:
            print(f"Error: Checkpoint not found: {args.checkpoint}")
    elif args.all or args.latest is not None:
        n_to_process = args.latest if args.latest is not None else None
        results = pipeline.process_directory(args.dir, n_to_process, args.output)
        if results:
            pipeline.generate_summary(results, args.output)
    else:
        print("No action specified. Use --help for usage information.")


if __name__ == '__main__':
    main()
