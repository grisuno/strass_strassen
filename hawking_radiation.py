#!/usr/bin/env python3
"""
HawkingRadiationAnalyzer - Robust Version for Exotic Checkpoints

This version handles checkpoints with:
- Custom classes (UnifiedConfig, etc.) that can't be unpickled normally
- Nested structures with metadata
- Missing delta keys in various locations
- Exotic checkpoint formats

Uses custom unpickler and robust migration strategies.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
import pickle
import io
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any, Protocol, runtime_checkable
from pathlib import Path
from dataclasses import dataclass, field
from scipy.stats import entropy
import warnings

warnings.filterwarnings('ignore')


# ============================================================================
# CUSTOM UNPICKLER FOR EXOTIC CHECKPOINTS
# ============================================================================

class CustomUnpickler(pickle.Unpickler):
    """
    Custom unpickler that handles unknown classes by creating dummy objects.
    
    This solves the "Can't get attribute 'UnifiedConfig'" error by providing
    fallback objects for any class that can't be found.
    """
    
    def find_class(self, module, name):
        """Override find_class to handle missing classes gracefully."""
        # List of known config classes that might appear in checkpoints
        known_configs = [
            'UnifiedConfig', 'ThermodynamicConfig', 'Configuration',
            'StrassenConfig', 'ModelConfig', 'TrainingConfig'
        ]
        
        if name in known_configs:
            # Return a dummy class that acts like a dict
            return self._create_dummy_class(name)
        
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            # Create a dummy class for any unknown class
            return self._create_dummy_class(name)
    
    def _create_dummy_class(self, name):
        """Create a dummy class that can hold attributes."""
        class DummyClass:
            def __init__(self, *args, **kwargs):
                self._name = name
                self._args = args
                self._kwargs = kwargs
                for k, v in kwargs.items():
                    setattr(self, k, v)
            
            def __repr__(self):
                return f"<DummyClass:{name}>"
            
            def __getitem__(self, key):
                return self._kwargs.get(key)
            
            def keys(self):
                return self._kwargs.keys()
            
            def values(self):
                return self._kwargs.values()
            
            def items(self):
                return self._kwargs.items()
            
            def get(self, key, default=None):
                return self._kwargs.get(key, default)
        
        return DummyClass


def load_checkpoint_robust(path: str, device: str = 'cpu') -> Any:
    """
    Load checkpoint with robust handling of custom classes.
    
    Tries multiple loading strategies in order:
    1. Standard torch.load
    2. Custom unpickler
    3. Weights_only mode with manual extraction
    """
    # Strategy 1: Standard load
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except Exception as e1:
        pass
    
    # Strategy 2: Custom unpickler via file
    try:
        with open(path, 'rb') as f:
            unpickler = CustomUnpickler(f)
            return unpickler.load()
    except Exception as e2:
        pass
    
    # Strategy 3: Try weights_only and extract what we can
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except Exception as e3:
        pass
    
    # Strategy 4: Raw pickle with custom unpickler
    try:
        with open(path, 'rb') as f:
            data = f.read()
            buffer = io.BytesIO(data)
            unpickler = CustomUnpickler(buffer)
            return unpickler.load()
    except Exception as e4:
        raise RuntimeError(f"All loading strategies failed for {path}")


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass(frozen=True)
class HawkingConfiguration:
    """Immutable configuration for Hawking radiation analysis."""
    
    # Architecture
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    INPUT_DIM: int = 4
    
    # Physical Constants (SI)
    PLANCK_SI: float = 1.054571817e-34  # J·s
    SPEED_OF_LIGHT_SI: float = 299792458  # m/s
    GRAVITATIONAL_CONSTANT_SI: float = 6.67430e-11  # m³/kg·s²
    BOLTZMANN_CONSTANT_SI: float = 1.380649e-23  # J/K
    SOLAR_MASS_SI: float = 1.98847e30  # kg
    
    # Discretization thresholds
    DISCRETIZATION_MARGIN: float = 0.1
    OPTIMAL_DELTA_THRESHOLD: float = 0.01
    INDUSTRIAL_DELTA_THRESHOLD: float = 0.1
    
    # Analysis parameters
    ENTROPY_BINS: int = 50
    TEMPERATURE_WINDOW: int = 100
    GRADIENT_WINDOW: int = 10
    SPECTRAL_REGULARIZATION: float = 1e-6
    
    # Information geometry
    EFFECTIVE_INFORMATION_DIM: int = 7  # Strassen rank
    ACTIVE_PARAM_THRESHOLD: float = 0.01
    
    # Output
    OUTPUT_DIRECTORY: str = "hawking_analysis_reports"
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_effective_input_dim(self) -> int:
        return self.MATRIX_SIZE * self.MATRIX_SIZE
    
    def get_total_parameters(self) -> int:
        input_dim = self.get_effective_input_dim()
        return (input_dim * self.HIDDEN_DIM * 2) + (self.HIDDEN_DIM * input_dim)


CONFIG = HawkingConfiguration()


# ============================================================================
# PROTOCOLS
# ============================================================================

@runtime_checkable
class IModel(Protocol):
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...


# ============================================================================
# MODEL
# ============================================================================

class BilinearStrassenModel(nn.Module):
    """Bilinear model for Strassen matrix multiplication."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        super().__init__()
        self.config = config
        input_dim = config.get_effective_input_dim()
        
        self.U = nn.Linear(input_dim, config.HIDDEN_DIM, bias=False)
        self.V = nn.Linear(input_dim, config.HIDDEN_DIM, bias=False)
        self.W = nn.Linear(config.HIDDEN_DIM, input_dim, bias=False)
        
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
    
    def get_flat_parameters(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])


# ============================================================================
# ROBUST CHECKPOINT MIGRATOR
# ============================================================================

class RobustCheckpointMigrator:
    """
    Enhanced checkpoint migrator that handles:
    - Direct U, V, W tensors
    - U_coefs format
    - Standard .weight format
    - Nested structures with config
    - Encoder format
    - State dicts within state dicts
    """
    
    def migrate(self, raw_data: Any, device: str) -> Optional[Dict[str, torch.Tensor]]:
        """Main migration entry point with multiple strategies."""
        
        # Handle dummy classes from custom unpickler
        if hasattr(raw_data, '_kwargs'):
            raw_data = raw_data._kwargs
        
        # Strategy 1: Direct state dict extraction
        result = self._try_extract_state_dict(raw_data, device)
        if result:
            return result
        
        # Strategy 2: Look for nested structures
        result = self._try_nested_extraction(raw_data, device)
        if result:
            return result
        
        # Strategy 3: Try to find tensors directly
        result = self._try_direct_tensor_extraction(raw_data, device)
        if result:
            return result
        
        return None
    
    def _try_extract_state_dict(self, data: Any, device: str) -> Optional[Dict[str, torch.Tensor]]:
        """Try standard state dict extraction methods."""
        state_dict = None
        
        if isinstance(data, dict):
            if 'state_dict' in data:
                state_dict = data['state_dict']
            elif 'model_state_dict' in data:
                state_dict = data['model_state_dict']
            elif 'model' in data and isinstance(data['model'], dict):
                state_dict = data['model']
            elif self._is_state_dict(data):
                state_dict = data
        
        if state_dict is not None:
            return self._migrate_dict(state_dict, device)
        
        return None
    
    def _try_nested_extraction(self, data: Any, device: str) -> Optional[Dict[str, torch.Tensor]]:
        """Try to extract from nested structures."""
        if not isinstance(data, dict):
            return None
        
        # Look for model weights in common nested locations
        nested_keys = ['model', 'network', 'net', 'module', 'model_state', 'weights']
        
        for key in nested_keys:
            if key in data:
                nested = data[key]
                if isinstance(nested, dict):
                    result = self._migrate_dict(nested, device)
                    if result:
                        return result
        
        # Look for any dict containing weight tensors
        for key, value in data.items():
            if isinstance(value, dict):
                result = self._migrate_dict(value, device)
                if result:
                    return result
        
        return None
    
    def _try_direct_tensor_extraction(self, data: Any, device: str) -> Optional[Dict[str, torch.Tensor]]:
        """Try to extract tensors directly from any structure."""
        tensors = {}
        
        def extract_tensors(obj, prefix=''):
            if isinstance(obj, torch.Tensor):
                return obj
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    if isinstance(v, torch.Tensor):
                        tensors[f"{prefix}{k}"] = v
                    elif isinstance(v, (dict, list)):
                        extract_tensors(v, f"{prefix}{k}.")
            elif isinstance(obj, (list, tuple)):
                for i, v in enumerate(obj):
                    if isinstance(v, torch.Tensor):
                        tensors[f"{prefix}{i}"] = v
        
        extract_tensors(data)
        
        if tensors:
            return self._reconstruct_from_tensors(tensors, device)
        
        return None
    
    def _reconstruct_from_tensors(self, tensors: Dict[str, torch.Tensor], 
                                   device: str) -> Optional[Dict[str, torch.Tensor]]:
        """Reconstruct U, V, W from found tensors."""
        # Look for U, V, W patterns
        U, V, W = None, None, None
        
        for key, tensor in tensors.items():
            key_lower = key.lower()
            tensor = tensor.to(device)
            
            # Identify U tensor
            if 'u' in key_lower and tensor.dim() == 2:
                if tensor.shape[0] in [7, 8] and tensor.shape[1] == 4:
                    U = tensor
            
            # Identify V tensor  
            elif 'v' in key_lower and tensor.dim() == 2:
                if tensor.shape[0] in [7, 8] and tensor.shape[1] == 4:
                    V = tensor
            
            # Identify W tensor
            elif 'w' in key_lower and tensor.dim() == 2:
                if tensor.shape[0] == 4 and tensor.shape[1] in [7, 8]:
                    W = tensor
        
        if U is not None and V is not None and W is not None:
            # Pad if necessary
            if U.shape[0] == 7:
                U_padded = torch.zeros(8, 4, device=device)
                U_padded[:7] = U
                U = U_padded
                
                V_padded = torch.zeros(8, 4, device=device)
                V_padded[:7] = V
                V = V_padded
                
                W_padded = torch.zeros(4, 8, device=device)
                W_padded[:, :7] = W
                W = W_padded
            
            return {
                'U.weight': U,
                'V.weight': V,
                'W.weight': W
            }
        
        return None
    
    def _is_state_dict(self, data: Dict) -> bool:
        """Check if dict looks like a state dict."""
        if not isinstance(data, dict):
            return False
        
        # Check for known state dict keys
        has_weights = any(k.endswith('.weight') for k in data.keys())
        has_direct = any(k in data for k in ['U', 'V', 'W', 'U_coefs'])
        has_tensors = any(isinstance(v, torch.Tensor) for v in data.values())
        
        return has_weights or has_direct or has_tensors
    
    def _migrate_dict(self, state_dict: Dict[str, Any], device: str) -> Optional[Dict[str, torch.Tensor]]:
        """Migrate a state dict to the expected format."""
        
        # Format 1: Direct U, V, W tensors
        if all(k in state_dict for k in ['U', 'V', 'W']):
            return self._migrate_custom_format(state_dict, device)
        
        # Format 2: U_coefs style
        if all(k in state_dict for k in ['U_coefs', 'V_coefs', 'W_coefs']):
            return self._migrate_coefs_format(state_dict, device)
        
        # Format 3: Standard .weight format
        if all(k in state_dict for k in ['U.weight', 'V.weight', 'W.weight']):
            return self._migrate_standard_format(state_dict, device)
        
        # Format 4: Encoder format
        if 'encoder.0.weight' in state_dict:
            return self._migrate_encoder_format(state_dict, device)
        
        # Format 5: Nested in 'layers' or similar
        for prefix in ['layers.', 'module.', 'model.']:
            if any(k.startswith(prefix) for k in state_dict.keys()):
                return self._migrate_prefixed_format(state_dict, prefix, device)
        
        return None
    
    def _migrate_custom_format(self, state_dict: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
        U = state_dict['U']
        V = state_dict['V']
        W = state_dict['W']
        
        # Ensure tensors are on correct device
        if isinstance(U, torch.Tensor):
            U = U.to(device)
        if isinstance(V, torch.Tensor):
            V = V.to(device)
        if isinstance(W, torch.Tensor):
            W = W.to(device)
        
        # Pad 7-slot to 8-slot format
        if U.shape == (7, 4):
            u_padded = torch.zeros(8, 4, device=device)
            v_padded = torch.zeros(8, 4, device=device)
            w_padded = torch.zeros(4, 8, device=device)
            u_padded[:7] = U
            v_padded[:7] = V
            w_padded[:, :7] = W
            return {'U.weight': u_padded, 'V.weight': v_padded, 'W.weight': w_padded}
        
        return {'U.weight': U, 'V.weight': V, 'W.weight': W}
    
    def _migrate_coefs_format(self, state_dict: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
        return {
            'U.weight': state_dict['U_coefs'].to(device) if hasattr(state_dict['U_coefs'], 'to') else state_dict['U_coefs'],
            'V.weight': state_dict['V_coefs'].to(device) if hasattr(state_dict['V_coefs'], 'to') else state_dict['V_coefs'],
            'W.weight': state_dict['W_coefs'].to(device) if hasattr(state_dict['W_coefs'], 'to') else state_dict['W_coefs']
        }
    
    def _migrate_standard_format(self, state_dict: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
        result = {}
        for k in ['U.weight', 'V.weight', 'W.weight']:
            if k in state_dict:
                tensor = state_dict[k]
                if hasattr(tensor, 'to'):
                    tensor = tensor.to(device)
                result[k] = tensor
        return result
    
    def _migrate_encoder_format(self, state_dict: Dict[str, Any], device: str) -> Dict[str, torch.Tensor]:
        """Handle encoder.layers style checkpoints."""
        try:
            encoder_0 = state_dict.get('encoder.0.weight', state_dict.get('layers.0.weight'))
            encoder_2 = state_dict.get('encoder.2.weight', state_dict.get('layers.2.weight', encoder_0))
            encoder_4 = state_dict.get('encoder.4.weight', state_dict.get('layers.4.weight'))
            
            if encoder_0 is None:
                return None
            
            # Try to extract 8x4 and 4x8 matrices
            if encoder_0.shape == (64, 8):
                u = encoder_0[:8, :4].clone()
            else:
                u = encoder_0.flatten()[:32].reshape(8, 4)
            
            if encoder_2.shape == (64, 64):
                v = encoder_2[:8, :4].clone()
            else:
                v = u.clone()
            
            if encoder_4 is not None and encoder_4.shape == (64, 64):
                w = encoder_4[:4, :8].clone()
            else:
                w = torch.randn(4, 8)
            
            return {'U.weight': u.to(device), 'V.weight': v.to(device), 'W.weight': w.to(device)}
        except Exception:
            return None
    
    def _migrate_prefixed_format(self, state_dict: Dict[str, Any], prefix: str, 
                                  device: str) -> Optional[Dict[str, torch.Tensor]]:
        """Handle prefixed state dict keys."""
        stripped = {}
        for k, v in state_dict.items():
            if k.startswith(prefix):
                new_key = k[len(prefix):]
                stripped[new_key] = v
        
        if stripped:
            return self._migrate_dict(stripped, device)
        return None


# ============================================================================
# METADATA EXTRACTOR
# ============================================================================

class MetadataExtractor:
    """Extracts metadata from various checkpoint formats."""
    
    @staticmethod
    def extract(checkpoint: Any) -> Dict[str, Any]:
        """Extract all relevant metadata from checkpoint."""
        metadata = {
            'epoch': 'unknown',
            'loss': 1.0,
            'loss_history': [],
            'delta': None,
            'config': None
        }
        
        # Handle dummy classes
        if hasattr(checkpoint, '_kwargs'):
            checkpoint = checkpoint._kwargs
        
        if not isinstance(checkpoint, dict):
            return metadata
        
        # Extract epoch
        for key in ['epoch', 'epochs', 'current_epoch', 'global_step']:
            if key in checkpoint:
                metadata['epoch'] = checkpoint[key]
                break
        
        # Extract loss
        for key in ['loss', 'current_loss', 'final_loss', 'train_loss', 'val_loss']:
            if key in checkpoint:
                val = checkpoint[key]
                if isinstance(val, (int, float)):
                    metadata['loss'] = val
                break
        
        # Extract loss history
        for key in ['loss_history', 'train_losses', 'losses', 'history']:
            if key in checkpoint:
                hist = checkpoint[key]
                if isinstance(hist, list):
                    metadata['loss_history'] = [float(x) for x in hist if isinstance(x, (int, float))]
                break
        
        # Extract delta (may be in various locations)
        delta = MetadataExtractor._extract_delta(checkpoint)
        if delta is not None:
            metadata['delta'] = delta
        
        # Extract config
        for key in ['config', 'cfg', 'args', 'hparams', 'hyperparams']:
            if key in checkpoint:
                metadata['config'] = checkpoint[key]
                break
        
        return metadata
    
    @staticmethod
    def _extract_delta(data: Any, depth: int = 0) -> Optional[float]:
        """Recursively search for delta in nested structures."""
        if depth > 5:  # Prevent infinite recursion
            return None
        
        if hasattr(data, '_kwargs'):
            data = data._kwargs
        
        if isinstance(data, dict):
            # Direct delta key
            if 'delta' in data:
                val = data['delta']
                if isinstance(val, (int, float)):
                    return float(val)
            
            # Check order_parameter or similar
            if 'order_parameter' in data and isinstance(data['order_parameter'], dict):
                if 'delta' in data['order_parameter']:
                    return float(data['order_parameter']['delta'])
            
            # Check metrics
            if 'metrics' in data and isinstance(data['metrics'], dict):
                if 'delta' in data['metrics']:
                    return float(data['metrics']['delta'])
            
            # Recursively search
            for value in data.values():
                result = MetadataExtractor._extract_delta(value, depth + 1)
                if result is not None:
                    return result
        
        return None


# ============================================================================
# GRAVITATIONAL CONSTANT CALCULATOR
# ============================================================================

class GravitationalConstantCalculator:
    """Calculates effective gravitational constant G_alg."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        self.config = config
    
    def calculate(self, model: IModel, gradient: Optional[torch.Tensor] = None,
                  precomputed_delta: Optional[float] = None) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        # Distance to discrete attractor (nearest integer)
        rounded = torch.round(all_weights)
        distances = torch.abs(all_weights - rounded)
        
        # Use precomputed delta if available, otherwise compute
        delta = precomputed_delta if precomputed_delta is not None else torch.max(distances).item()
        
        # Only consider non-trivial distances
        mask = distances > 1e-6
        valid_distances = distances[mask]
        
        if len(valid_distances) == 0:
            return {
                'G_alg': float('inf'),
                'avg_distance': 0.0,
                'gradient_magnitude': 0.0,
                'force': float('inf'),
                'crystallization_pressure': 0.0,
                'delta': float(delta),
                'is_crystallized': delta < self.config.DISCRETIZATION_MARGIN
            }
        
        avg_distance = torch.mean(valid_distances).item()
        
        # Gradient magnitude (force)
        if gradient is not None:
            grad_magnitude = torch.norm(gradient).item()
        else:
            grad_magnitude = torch.std(all_weights).item()
        
        # G_alg = Force / Distance² (Newton's law analog)
        if avg_distance < 1e-10:
            G_alg = float('inf')
            force = float('inf')
        else:
            force = grad_magnitude / (avg_distance ** 2)
            G_alg = force
        
        # Crystallization pressure
        crystallization_pressure = grad_magnitude / (delta + 1e-10) if grad_magnitude > 0 else 0.0
        
        return {
            'G_alg': float(G_alg),
            'avg_distance': float(avg_distance),
            'gradient_magnitude': float(grad_magnitude),
            'force': float(force),
            'crystallization_pressure': float(crystallization_pressure),
            'delta': float(delta),
            'is_crystallized': delta < self.config.DISCRETIZATION_MARGIN
        }


# ============================================================================
# PLANCK CONSTANT CALCULATOR
# ============================================================================

class PlanckConstantCalculator:
    """Calculates effective Planck constant h_bar_eff."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        self.config = config
    
    def calculate(self, model: IModel, loss: float = 1.0,
                  precomputed_delta: Optional[float] = None) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        # Weight statistics
        weight_variance = torch.var(all_weights).item()
        weight_std = torch.std(all_weights).item()
        
        # Discretization margin
        rounded = torch.round(all_weights)
        delta = precomputed_delta if precomputed_delta is not None else torch.max(torch.abs(all_weights - rounded)).item()
        
        # Effective lambda (confinement potential)
        total_norm = torch.norm(all_weights).item()
        lambda_eff = 1.0 / (total_norm ** 2 + 1e-10)
        
        # Method 1: Generalized Uncertainty Principle
        h_bar_uncertainty = 2.0 * (delta ** 2) * lambda_eff
        
        # Method 2: Action Quantization
        omega = np.sqrt(lambda_eff) if lambda_eff > 0 else 1.0
        period = 2.0 * np.pi / omega
        
        T = loss
        V = lambda_eff * (delta ** 2)
        L = T - V
        action = abs(L) * period
        h_bar_action = action
        
        # Method 3: Quantum Conductance
        accuracy_proxy = 1.0 - min(delta, 1.0)
        if loss > 0:
            conductance = accuracy_proxy / (loss + 1e-10)
            h_bar_conductance = 1.0 / conductance if conductance > 0 else 0.0
        else:
            h_bar_conductance = 0.0
        
        # Method 4: Information Entropy
        n_eff = self.config.EFFECTIVE_INFORMATION_DIM
        information = np.log2(n_eff) if n_eff > 1 else 1.0
        
        energy_total = T + V
        energy_per_bit = energy_total / information if information > 0 else 0.0
        h_bar_information = energy_per_bit * period
        
        # Unified h_bar
        if delta < self.config.OPTIMAL_DELTA_THRESHOLD:
            weights = (0.6, 0.25, 0.1, 0.05)
        elif delta < self.config.INDUSTRIAL_DELTA_THRESHOLD:
            weights = (0.5, 0.3, 0.15, 0.05)
        else:
            weights = (0.25, 0.25, 0.25, 0.25)
        
        w1, w2, w3, w4 = weights
        h_bar_unified = (
            w1 * h_bar_uncertainty +
            w2 * h_bar_action +
            w3 * h_bar_conductance +
            w4 * h_bar_information
        ) / sum(weights)
        
        # Compare with physical Planck constant
        ratio_to_si = h_bar_unified / self.config.PLANCK_SI if self.config.PLANCK_SI > 0 else 0.0
        orders_of_magnitude = np.log10(ratio_to_si) if ratio_to_si > 0 else 0.0
        
        return {
            'h_bar_uncertainty': float(h_bar_uncertainty),
            'h_bar_action': float(h_bar_action),
            'h_bar_conductance': float(h_bar_conductance),
            'h_bar_information': float(h_bar_information),
            'h_bar_unified': float(h_bar_unified),
            'delta': float(delta),
            'lambda_eff': float(lambda_eff),
            'weight_variance': float(weight_variance),
            'weight_std': float(weight_std),
            'ratio_to_si_planck': float(ratio_to_si),
            'orders_of_magnitude': float(orders_of_magnitude),
            'is_quantum_regime': h_bar_unified < 1.0
        }


# ============================================================================
# BOLTZMANN CONSTANT CALCULATOR
# ============================================================================

class BoltzmannConstantCalculator:
    """Calculates effective Boltzmann constant k_B_eff."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        self.config = config
    
    def calculate(self, model: IModel, loss: float = 1.0, 
                  loss_history: Optional[List[float]] = None) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()]).cpu().numpy()
        
        # Configuration entropy (Shannon)
        hist, _ = np.histogram(all_weights, bins=self.config.ENTROPY_BINS, density=True)
        hist = hist[hist > 0]
        config_entropy = entropy(hist) if len(hist) > 0 else 0.0
        
        # Energy proxy
        if loss_history and len(loss_history) >= self.config.TEMPERATURE_WINDOW:
            recent_losses = loss_history[-self.config.TEMPERATURE_WINDOW:]
            energy = np.var(recent_losses)
        else:
            energy = loss
        
        # k_B_eff calculations
        if energy > 1e-10:
            k_b_entropy = config_entropy / energy
        else:
            k_b_entropy = float('inf')
        
        weight_variance = np.var(all_weights)
        temperature_proxy = energy
        if temperature_proxy > 1e-10:
            k_b_thermal = weight_variance / temperature_proxy
        else:
            k_b_thermal = float('inf')
        
        k_b_unified = (k_b_entropy + k_b_thermal) / 2.0
        ratio_to_si = k_b_unified / self.config.BOLTZMANN_CONSTANT_SI
        
        return {
            'k_b_entropy': float(k_b_entropy),
            'k_b_thermal': float(k_b_thermal),
            'k_b_unified': float(k_b_unified),
            'config_entropy': float(config_entropy),
            'energy_proxy': float(energy),
            'weight_variance': float(weight_variance),
            'ratio_to_si_boltzmann': float(ratio_to_si),
            'is_thermal_regime': k_b_unified > 0
        }


# ============================================================================
# SPEED OF LIGHT CALCULATOR
# ============================================================================

class SpeedOfLightCalculator:
    """Calculates effective speed of light c_eff."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        self.config = config
    
    def calculate(self, model: IModel, h_bar: float, G_alg: float) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        # Method 1: From Planck relation
        c_from_planck = self.config.SPEED_OF_LIGHT_SI * (h_bar / self.config.PLANCK_SI)
        
        # Method 2: From network structure
        U = coeffs['U'].cpu().numpy()
        V = coeffs['V'].cpu().numpy()
        W = coeffs['W'].cpu().numpy()
        
        try:
            u_eigenvalues = np.abs(np.linalg.eigvals(U @ U.T))
            v_eigenvalues = np.abs(np.linalg.eigvals(V @ V.T))
            w_eigenvalues = np.abs(np.linalg.eigvals(W.T @ W))
            
            spectral_velocity = np.sqrt(np.mean([
                np.max(u_eigenvalues),
                np.max(v_eigenvalues),
                np.max(w_eigenvalues)
            ]))
        except:
            spectral_velocity = 1.0
        
        # Method 3: From Fisher information
        all_weights_np = all_weights.cpu().numpy()
        fisher_approx = np.outer(all_weights_np, all_weights_np)
        fisher_approx += np.eye(len(all_weights_np)) * 1e-8
        
        try:
            det = np.linalg.det(fisher_approx)
            c_from_geometry = 1.0 / np.sqrt(abs(det) + 1e-20)
        except:
            c_from_geometry = 1.0
        
        c_unified = c_from_planck
        ratio_to_si = c_unified / self.config.SPEED_OF_LIGHT_SI
        
        return {
            'c_from_planck': float(c_from_planck),
            'c_spectral': float(spectral_velocity),
            'c_from_geometry': float(c_from_geometry),
            'c_unified': float(c_unified),
            'ratio_to_si_c': float(ratio_to_si),
            'orders_of_magnitude': float(np.log10(ratio_to_si)) if ratio_to_si > 0 else 0.0,
            'is_superluminal': c_unified > self.config.SPEED_OF_LIGHT_SI
        }


# ============================================================================
# INFORMATIONAL MASS CALCULATOR
# ============================================================================

class InformationalMassCalculator:
    """Calculates effective mass M_eff."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        self.config = config
    
    def calculate(self, model: IModel, G_alg: float, 
                  c_eff: float, h_bar: float) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        active_mask = torch.abs(all_weights) > self.config.ACTIVE_PARAM_THRESHOLD
        n_active = torch.sum(active_mask).item()
        n_total = len(all_weights)
        
        all_weights_np = all_weights.cpu().numpy()
        hist, _ = np.histogram(all_weights_np, bins=self.config.ENTROPY_BINS, density=True)
        hist = hist[hist > 0]
        information_content = entropy(hist) if len(hist) > 0 else 0.0
        
        # Method 1: From Planck mass formula
        if G_alg > 0 and G_alg < float('inf') and c_eff > 0:
            m_planck_eff = np.sqrt(h_bar * c_eff / G_alg)
        else:
            m_planck_eff = 0.0
        
        # Method 2: From active parameter count
        m_from_info = n_active * information_content
        
        # Method 3: From energy-mass relation
        weight_energy = torch.var(all_weights).item()
        if c_eff > 0:
            m_from_energy = weight_energy / (c_eff ** 2)
        else:
            m_from_energy = 0.0
        
        m_unified = m_planck_eff if m_planck_eff > 0 else max(m_from_info, m_from_energy)
        ratio_to_solar = m_unified / self.config.SOLAR_MASS_SI
        
        return {
            'm_planck_eff': float(m_planck_eff),
            'm_from_info': float(m_from_info),
            'm_from_energy': float(m_from_energy),
            'm_unified': float(m_unified),
            'n_active_params': int(n_active),
            'n_total_params': int(n_total),
            'sparsity': float(1.0 - n_active / n_total),
            'information_content': float(information_content),
            'ratio_to_solar_mass': float(ratio_to_solar)
        }


# ============================================================================
# HORIZON AREA CALCULATOR
# ============================================================================

class HorizonAreaCalculator:
    """Calculates effective area A_eff."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        self.config = config
    
    def calculate(self, model: IModel, M_eff: float) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        
        U = coeffs['U']
        V = coeffs['V']
        W = coeffs['W']
        
        hidden_dim = U.shape[0]
        input_dim = U.shape[1]
        
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        u_active = torch.sum(torch.abs(U) > self.config.ACTIVE_PARAM_THRESHOLD).item()
        v_active = torch.sum(torch.abs(V) > self.config.ACTIVE_PARAM_THRESHOLD).item()
        w_active = torch.sum(torch.abs(W) > self.config.ACTIVE_PARAM_THRESHOLD).item()
        
        a_encoding = u_active + v_active
        a_decoding = w_active
        a_unified = a_encoding + a_decoding
        
        slot_norms = torch.norm(U, dim=1) * torch.norm(V, dim=1) * torch.norm(W, dim=0)
        active_slots = torch.sum(slot_norms > 0.1).item()
        
        entropy_proxy = -torch.sum(all_weights * torch.log(torch.abs(all_weights) + 1e-10)).item()
        a_from_entropy = 4.0 * abs(entropy_proxy)
        
        return {
            'a_unified': float(a_unified),
            'a_encoding': float(a_encoding),
            'a_decoding': float(a_decoding),
            'a_from_entropy': float(a_from_entropy),
            'active_slots': int(active_slots),
            'effective_dimension': int(hidden_dim),
            'hidden_dim': int(hidden_dim),
            'input_dim': int(input_dim),
            'u_active': int(u_active),
            'v_active': int(v_active),
            'w_active': int(w_active),
            'is_strassen_rank': active_slots == self.config.EFFECTIVE_INFORMATION_DIM
        }


# ============================================================================
# HAWKING RADIATION CALCULATOR
# ============================================================================

class HawkingRadiationCalculator:
    """Calculates Hawking radiation metrics."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        self.config = config
        self.G_calc = GravitationalConstantCalculator(config)
        self.hbar_calc = PlanckConstantCalculator(config)
        self.kb_calc = BoltzmannConstantCalculator(config)
        self.c_calc = SpeedOfLightCalculator(config)
        self.M_calc = InformationalMassCalculator(config)
        self.A_calc = HorizonAreaCalculator(config)
    
    def calculate_all(self, model: IModel, loss: float = 1.0,
                      loss_history: Optional[List[float]] = None,
                      gradient: Optional[torch.Tensor] = None,
                      precomputed_delta: Optional[float] = None) -> Dict[str, Any]:
        """Calculate all Hawking radiation metrics."""
        
        # 1. Gravitational constant
        G_result = self.G_calc.calculate(model, gradient, precomputed_delta)
        G_alg = G_result['G_alg']
        delta = G_result['delta']
        
        # 2. Planck constant
        hbar_result = self.hbar_calc.calculate(model, loss, precomputed_delta)
        h_bar = hbar_result['h_bar_unified']
        
        # 3. Boltzmann constant
        kb_result = self.kb_calc.calculate(model, loss, loss_history)
        k_B = kb_result['k_b_unified']
        
        # 4. Speed of light
        c_result = self.c_calc.calculate(model, h_bar, G_alg)
        c_eff = c_result['c_unified']
        
        # 5. Mass
        M_result = self.M_calc.calculate(model, G_alg, c_eff, h_bar)
        M_eff = M_result['m_unified']
        
        # 6. Area
        A_result = self.A_calc.calculate(model, M_eff)
        A_eff = A_result['a_unified']
        
        # Safe values for calculations
        G_safe = G_alg if G_alg < float('inf') and G_alg > 0 else 1.0
        M_safe = M_eff if M_eff > 0 else 1.0
        k_B_safe = k_B if k_B > 0 else 1.0
        c_safe = c_eff if c_eff > 0 else 1.0
        h_bar_safe = h_bar if h_bar > 0 else 1.0
        A_safe = A_eff if A_eff > 0 else 1.0
        
        # Bekenstein-Hawking Entropy
        if G_safe > 0 and h_bar_safe > 0:
            S_bh = (A_safe * k_B_safe * (c_safe ** 3)) / (4.0 * G_safe * h_bar_safe)
        else:
            S_bh = float('inf')
        
        # Hawking Temperature
        if G_safe > 0 and M_safe > 0 and k_B_safe > 0:
            T_hawking = (h_bar_safe * (c_safe ** 3)) / (8.0 * np.pi * G_safe * M_safe * k_B_safe)
        else:
            T_hawking = float('inf')
        
        # Schwarzschild Radius
        if c_safe > 0:
            r_schwarzschild = 2.0 * G_safe * M_safe / (c_safe ** 2)
        else:
            r_schwarzschild = float('inf')
        
        # Radiation Power
        if G_safe > 0 and M_safe > 0:
            P_radiation = (h_bar_safe * (c_safe ** 6)) / (15360.0 * np.pi * (G_safe ** 2) * (M_safe ** 2))
        else:
            P_radiation = float('inf')
        
        # Evaporation Time
        if h_bar_safe > 0 and c_safe > 0:
            tau_evaporation = (5120.0 * np.pi * (G_safe ** 2) * (M_safe ** 3)) / (h_bar_safe * (c_safe ** 4))
        else:
            tau_evaporation = float('inf')
        
        # Surface Gravity
        if G_safe > 0 and M_safe > 0:
            surface_gravity = (c_safe ** 4) / (4.0 * G_safe * M_safe)
        else:
            surface_gravity = float('inf')
        
        # Tidal Forces
        if r_schwarzschild > 0 and r_schwarzschild < float('inf'):
            tidal_force = 1.0 / (r_schwarzschild ** 3)
        else:
            tidal_force = 0.0
        
        # Information Escape Rate
        if T_hawking > 0 and T_hawking < float('inf') and k_B_safe > 0:
            info_escape_rate = P_radiation / (T_hawking * k_B_safe)
        else:
            info_escape_rate = 0.0
        
        return {
            'constants': {
                'G_alg': G_result,
                'h_bar': hbar_result,
                'k_B': kb_result,
                'c_eff': c_result,
                'M_eff': M_result,
                'A_eff': A_result
            },
            'hawking_radiation': {
                'bekenstein_hawking_entropy': {
                    'S_BH': float(S_bh),
                    'formula': 'S = A*k_B*c³ / (4*G*ℏ)',
                    'units': 'synthetic (information units)'
                },
                'hawking_temperature': {
                    'T_H': float(T_hawking),
                    'formula': 'T_H = ℏ*c³ / (8*π*G*M*k_B)',
                    'units': 'synthetic (energy units)'
                },
                'schwarzschild_radius': {
                    'r_s': float(r_schwarzschild),
                    'formula': 'r_s = 2*G*M / c²',
                    'units': 'parameter space units'
                },
                'radiation_power': {
                    'P_rad': float(P_radiation),
                    'formula': 'P = ℏ*c⁶ / (15360*π*G²*M²)',
                    'units': 'information per epoch'
                },
                'evaporation_time': {
                    'tau_evap': float(tau_evaporation),
                    'formula': 'τ = 5120*π*G²*M³ / (ℏ*c⁴)',
                    'units': 'epochs to full crystallization'
                },
                'surface_gravity': {
                    'kappa_s': float(surface_gravity),
                    'formula': 'κ_s = c⁴ / (4*G*M)',
                    'units': 'gradient strength at horizon'
                },
                'tidal_forces': {
                    'F_tidal': float(tidal_force),
                    'formula': '~ 1 / r_s³',
                    'units': 'parameter distortion rate'
                },
                'information_escape_rate': {
                    'dI_dt': float(info_escape_rate),
                    'formula': 'dI/dt = P / (T_H * k_B)',
                    'units': 'bits per epoch'
                }
            },
            'phase_diagnostics': {
                'is_black_hole': G_result['is_crystallized'] or (S_bh > 0),
                'is_evaporating': P_radiation > 0 and P_radiation < float('inf'),
                'is_hot': T_hawking > 0 and T_hawking < float('inf'),
                'crystallization_state': self._classify_state(delta, T_hawking),
                'entropy_content': 'high' if S_bh > 10 else ('medium' if S_bh > 1 else 'low'),
                'temperature_regime': 'hot' if T_hawking > 1 else ('cold' if T_hawking < 0.01 else 'moderate')
            }
        }
    
    def _classify_state(self, delta: float, T_hawking: float) -> str:
        if delta < self.config.OPTIMAL_DELTA_THRESHOLD:
            return "frozen_crystal" if T_hawking < 0.01 else "hot_crystal"
        elif delta < self.config.INDUSTRIAL_DELTA_THRESHOLD:
            return "polycrystal"
        elif delta < 0.5:
            return "glass"
        else:
            return "amorphous"


# ============================================================================
# ROBUST BATCH ANALYZER
# ============================================================================

class RobustHawkingAnalyzer:
    """Robust analyzer that handles exotic checkpoint formats."""
    
    def __init__(self, config: HawkingConfiguration = CONFIG):
        self.config = config
        self.migrator = RobustCheckpointMigrator()
        self.hawking_calc = HawkingRadiationCalculator(config)
    
    def analyze_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Analyze a single checkpoint with robust error handling."""
        print(f"\n{'='*70}")
        print(f"HAWKING RADIATION ANALYSIS: {os.path.basename(checkpoint_path)}")
        print(f"{'='*70}")
        
        # Load checkpoint with robust loader
        try:
            checkpoint = load_checkpoint_robust(checkpoint_path, self.config.DEVICE)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        # Create model
        model = BilinearStrassenModel(self.config).to(self.config.DEVICE)
        
        # Migrate state dict
        state_dict = self.migrator.migrate(checkpoint, self.config.DEVICE)
        if state_dict is None:
            raise RuntimeError(f"Failed to migrate checkpoint: {checkpoint_path}")
        
        model.load_state_dict(state_dict)
        model.eval()
        
        # Extract metadata
        metadata = MetadataExtractor.extract(checkpoint)
        
        # Compute gradient
        gradient = self._compute_gradient(model)
        
        # Calculate Hawking radiation
        results = self.hawking_calc.calculate_all(
            model, 
            metadata['loss'], 
            metadata['loss_history'], 
            gradient,
            metadata.get('delta')
        )
        
        # Add metadata
        results['metadata'] = {
            'checkpoint_path': checkpoint_path,
            'checkpoint_name': os.path.basename(checkpoint_path),
            'epoch': metadata['epoch'],
            'timestamp': datetime.now().isoformat(),
            'current_loss': metadata['loss'],
            'precomputed_delta': metadata.get('delta')
        }
        
        # Print report
        self._print_report(results)
        
        return results
    
    def _compute_gradient(self, model: nn.Module) -> torch.Tensor:
        """Compute gradient on random batch."""
        model.train()
        batch_size = 32
        matrix_size = self.config.MATRIX_SIZE
        input_dim = self.config.get_effective_input_dim()
        
        A = torch.randn(batch_size, matrix_size, matrix_size, device=self.config.DEVICE)
        B = torch.randn(batch_size, matrix_size, matrix_size, device=self.config.DEVICE)
        C = torch.bmm(A, B)
        
        A_flat = A.reshape(batch_size, input_dim)
        B_flat = B.reshape(batch_size, input_dim)
        C_flat = C.reshape(batch_size, input_dim)
        
        model.zero_grad()
        pred = model(A_flat, B_flat)
        loss = nn.functional.mse_loss(pred, C_flat)
        loss.backward()
        
        grads = []
        for param in model.parameters():
            if param.grad is not None:
                grads.append(param.grad.flatten())
        
        model.eval()
        return torch.cat(grads) if grads else None
    
    def _print_report(self, results: Dict[str, Any]):
        """Print formatted report."""
        hr = results['hawking_radiation']
        consts = results['constants']
        phase = results['phase_diagnostics']
        meta = results['metadata']
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {meta['checkpoint_name']}")
        print(f"  Epoch: {meta['epoch']}")
        if meta.get('precomputed_delta') is not None:
            print(f"  Precomputed δ: {meta['precomputed_delta']:.6f}")
        
        print(f"\n[CONSTANTS]")
        print(f"  G_alg (Gravity):     {consts['G_alg']['G_alg']:.6e}")
        print(f"  ℏ_eff (Planck):      {consts['h_bar']['h_bar_unified']:.6e}")
        print(f"  k_B_eff (Boltzmann): {consts['k_B']['k_b_unified']:.6e}")
        print(f"  c_eff (Light speed): {consts['c_eff']['c_unified']:.6e}")
        print(f"  M_eff (Mass):        {consts['M_eff']['m_unified']:.6e}")
        print(f"  A_eff (Area):        {consts['A_eff']['a_unified']:.1f}")
        print(f"  δ (Delta):           {consts['G_alg']['delta']:.6f}")
        
        print(f"\n[HAWKING RADIATION]")
        print(f"  Entropy S_BH:        {hr['bekenstein_hawking_entropy']['S_BH']:.6e}")
        print(f"  Temperature T_H:     {hr['hawking_temperature']['T_H']:.6e}")
        print(f"  Schwarzschild r_s:   {hr['schwarzschild_radius']['r_s']:.6e}")
        print(f"  Radiation Power P:   {hr['radiation_power']['P_rad']:.6e}")
        print(f"  Evaporation τ:       {hr['evaporation_time']['tau_evap']:.6e}")
        print(f"  Surface Gravity κ_s: {hr['surface_gravity']['kappa_s']:.6e}")
        print(f"  Info Escape Rate:    {hr['information_escape_rate']['dI_dt']:.6e}")
        
        print(f"\n[PHASE DIAGNOSTICS]")
        print(f"  State:               {phase['crystallization_state']}")
        print(f"  Is Black Hole:       {phase['is_black_hole']}")
        print(f"  Is Evaporating:      {phase['is_evaporating']}")
        print(f"  Entropy Content:     {phase['entropy_content']}")
        print(f"  Temperature Regime:  {phase['temperature_regime']}")
        
        print(f"{'='*70}")
    
    def analyze_directory(self, checkpoint_dir: str, output_dir: str, 
                          pattern: str = '*.pt') -> List[Dict[str, Any]]:
        """Analyze all checkpoints in directory."""
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise ValueError(f"Directory not found: {checkpoint_dir}")
        
        checkpoints = sorted(checkpoint_path.glob(pattern))
        if not checkpoints:
            print(f"No checkpoints found matching {pattern}")
            return []
        
        print(f"\nFound {len(checkpoints)} checkpoints to analyze")
        print(f"Output directory: {output_dir}")
        
        results = []
        errors = []
        
        for i, ckpt in enumerate(checkpoints):
            print(f"\n[{i+1}/{len(checkpoints)}] Processing: {ckpt.name}")
            try:
                result = self.analyze_checkpoint(str(ckpt))
                results.append(result)
                
                # Save individual report
                out_file = Path(output_dir) / f"{ckpt.stem}_hawking.json"
                with open(out_file, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ERROR: {error_msg}")
                errors.append({'checkpoint': ckpt.name, 'error': error_msg})
                continue
        
        # Generate summary
        if results:
            summary = self._generate_summary(results, errors)
            summary_file = Path(output_dir) / "hawking_summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            print(f"\nSummary saved: {summary_file}")
        
        return results
    
    def _generate_summary(self, results: List[Dict[str, Any]], 
                          errors: List[Dict[str, str]]) -> Dict[str, Any]:
        """Generate aggregate summary."""
        successful = [r for r in results if r.get('constants')]
        
        # Aggregate statistics
        g_algs = [r['constants']['G_alg']['G_alg'] for r in successful 
                  if r['constants']['G_alg']['G_alg'] < float('inf')]
        h_bars = [r['constants']['h_bar']['h_bar_unified'] for r in successful]
        k_bs = [r['constants']['k_B']['k_b_unified'] for r in successful]
        deltas = [r['constants']['G_alg']['delta'] for r in successful]
        
        # Count states
        states = {}
        for r in successful:
            state = r['phase_diagnostics']['crystallization_state']
            states[state] = states.get(state, 0) + 1
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_checkpoints': len(results) + len(errors),
            'successful_analyses': len(successful),
            'failed_analyses': len(errors),
            'errors': errors,
            'aggregate_statistics': {
                'G_alg': {
                    'mean': float(np.mean(g_algs)) if g_algs else None,
                    'std': float(np.std(g_algs)) if g_algs else None,
                    'min': float(np.min(g_algs)) if g_algs else None,
                    'max': float(np.max(g_algs)) if g_algs else None
                },
                'h_bar': {
                    'mean': float(np.mean(h_bars)) if h_bars else None,
                    'std': float(np.std(h_bars)) if h_bars else None
                },
                'k_B': {
                    'mean': float(np.mean(k_bs)) if k_bs else None,
                    'std': float(np.std(k_bs)) if k_bs else None
                },
                'delta': {
                    'mean': float(np.mean(deltas)) if deltas else None,
                    'std': float(np.std(deltas)) if deltas else None,
                    'min': float(np.min(deltas)) if deltas else None,
                    'max': float(np.max(deltas)) if deltas else None
                }
            },
            'crystallization_states': states,
            'individual_results': [
                {
                    'checkpoint': r['metadata']['checkpoint_name'],
                    'delta': r['constants']['G_alg']['delta'],
                    'state': r['phase_diagnostics']['crystallization_state'],
                    'entropy': r['hawking_radiation']['bekenstein_hawking_entropy']['S_BH'],
                    'temperature': r['hawking_radiation']['hawking_temperature']['T_H']
                }
                for r in successful
            ]
        }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Robust Hawking Radiation Analysis for Neural Network Checkpoints'
    )
    parser.add_argument('path', type=str, help='Path to checkpoint file or directory')
    parser.add_argument('--output', '-o', type=str, default='hawking_radiation',
                        help='Output directory for reports')
    parser.add_argument('--pattern', '-p', type=str, default='*.pt',
                        help='File pattern for directory analysis')
    
    args = parser.parse_args()
    
    path = Path(args.path)
    if not path.exists():
        print(f"Error: Path not found: {path}")
        return 1
    
    analyzer = RobustHawkingAnalyzer()
    
    try:
        if path.is_file():
            result = analyzer.analyze_checkpoint(str(path))
            
            # Save result
            output_dir = Path(args.output)
            output_dir.mkdir(parents=True, exist_ok=True)
            out_file = output_dir / f"{path.stem}_hawking.json"
            with open(out_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            print(f"\nResult saved: {out_file}")
            
        elif path.is_dir():
            analyzer.analyze_directory(str(path), args.output, args.pattern)
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())