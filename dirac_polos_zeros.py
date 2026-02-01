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
from abc import ABC, abstractmethod
from scipy import signal
from scipy.linalg import eigvals
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.patches import Circle


@dataclass(frozen=True)
class AnalysisConfig:
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


@runtime_checkable
class IModel(Protocol):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...


@runtime_checkable
class IChargeDistributionExtractor(Protocol):
    def extract(self, model: IModel) -> torch.Tensor: ...


@runtime_checkable
class IDiracAnalyzer(Protocol):
    def analyze(self, charge_density: torch.Tensor) -> Dict[str, Any]: ...


@runtime_checkable
class IFieldCalculator(Protocol):
    def calculate(self, dirac_data: Dict[str, Any], eval_points: Optional[np.ndarray]) -> np.ndarray: ...


@runtime_checkable
class IFluxCalculator(Protocol):
    def calculate(self, electric_field: np.ndarray, surface_points: Optional[np.ndarray]) -> Dict[str, float]: ...


@runtime_checkable
class IStateSpaceExtractor(Protocol):
    def extract(self, model: IModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: ...


@runtime_checkable
class ITransferFunctionComputer(Protocol):
    def compute(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class IPoleZeroAnalyzer(Protocol):
    def analyze_stability(self) -> Dict[str, Any]: ...
    def get_poles(self) -> np.ndarray: ...
    def get_zeros(self) -> np.ndarray: ...


@runtime_checkable
class IFrequencyAnalyzer(Protocol):
    def compute_bode(self) -> Dict[str, np.ndarray]: ...
    def compute_margins(self) -> Dict[str, float]: ...
    def compute_nyquist(self) -> Dict[str, np.ndarray]: ...


@runtime_checkable
class ITimeResponseAnalyzer(Protocol):
    def compute_step(self) -> Dict[str, np.ndarray]: ...
    def compute_impulse(self) -> Dict[str, np.ndarray]: ...


@runtime_checkable
class ICheckpointLoader(Protocol):
    def load(self, path: str, device: str) -> Any: ...


@runtime_checkable
class ICheckpointMigrator(Protocol):
    def migrate(self, raw_data: Any) -> Optional[Dict[str, torch.Tensor]]: ...


@runtime_checkable
class IVisualizer(Protocol):
    def visualize(self, data: Dict[str, Any], output_path: str) -> None: ...


class BilinearModel(nn.Module):
    def __init__(self, hidden_dim: int = AnalysisConfig.HIDDEN_DIM, matrix_size: int = AnalysisConfig.MATRIX_SIZE):
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


class ChargeDistributionExtractor:
    def extract(self, model: IModel) -> torch.Tensor:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        charge_density = all_weights - all_weights.mean()
        return charge_density


class DiracDeltaAnalyzer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def analyze(self, charge_density: torch.Tensor) -> Dict[str, Any]:
        rounded = torch.round(charge_density)
        delta_deviation = torch.abs(charge_density - rounded)
        
        point_charges = charge_density[delta_deviation < self.config.DELTA_THRESHOLD]
        point_positions = torch.where(delta_deviation < self.config.DELTA_THRESHOLD)[0]
        
        gaussian_weights = torch.exp(-delta_deviation**2 / (2 * self.config.GAUSSIAN_SIGMA**2))
        
        discrete_mass = torch.sum(torch.abs(point_charges)).item()
        continuous_mass = torch.sum(torch.abs(charge_density) * (1 - gaussian_weights)).item()
        total_mass = discrete_mass + continuous_mass
        
        delta_strength = discrete_mass / total_mass if total_mass > 0 else 0
        
        return {
            'point_charges': point_charges.cpu().numpy(),
            'point_positions': point_positions.cpu().numpy(),
            'num_point_charges': len(point_charges),
            'gaussian_weights': gaussian_weights.cpu().numpy(),
            'delta_strength': delta_strength,
            'discrete_mass': discrete_mass,
            'continuous_mass': continuous_mass,
            'total_mass': total_mass,
            'charge_density_full': charge_density.cpu().numpy()
        }


class ElectricFieldCalculator:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def calculate(self, dirac_data: Dict[str, Any], eval_points: Optional[np.ndarray] = None) -> np.ndarray:
        point_charges = dirac_data['point_charges']
        point_positions = dirac_data['point_positions']
        charge_density = dirac_data['charge_density_full']
        
        if eval_points is None:
            eval_points = np.linspace(0, len(charge_density) - 1, self.config.SPATIAL_RESOLUTION)
        
        electric_field = np.zeros_like(eval_points)
        
        for charge, pos in zip(point_charges, point_positions):
            r = eval_points - pos
            r_safe = np.where(np.abs(r) < 1e-10, 1e-10, r)
            field_contribution = (charge / (4 * np.pi * self.config.EPSILON_0 * r_safe**2)) * np.sign(r_safe)
            electric_field += field_contribution
        
        gaussian_weights = dirac_data['gaussian_weights']
        for i, eval_pt in enumerate(eval_points):
            idx = int(np.clip(eval_pt, 0, len(charge_density) - 1))
            if gaussian_weights[idx] < 0.5:
                r = eval_pt - idx
                r_safe = r if np.abs(r) > 1e-10 else 1e-10
                smoothed_charge = charge_density[idx] * (1 - gaussian_weights[idx])
                field_contribution = (smoothed_charge / (4 * np.pi * self.config.EPSILON_0 * r_safe**2)) * np.sign(r_safe)
                electric_field[i] += field_contribution
        
        return electric_field


class ElectricFluxCalculator:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def calculate(self, electric_field: np.ndarray, surface_points: Optional[np.ndarray] = None) -> Dict[str, float]:
        if surface_points is None:
            surface_points = np.linspace(0, len(electric_field) - 1, self.config.FLUX_INTEGRATION_SAMPLES)
        
        field_interpolated = np.interp(surface_points, np.arange(len(electric_field)), electric_field)
        
        flux_outward = np.sum(field_interpolated[field_interpolated > 0])
        flux_inward = np.sum(np.abs(field_interpolated[field_interpolated < 0]))
        flux_net = flux_outward - flux_inward
        enclosed_charge = flux_net * self.config.EPSILON_0
        
        return {
            'flux_outward': float(flux_outward),
            'flux_inward': float(flux_inward),
            'flux_net': float(flux_net),
            'enclosed_charge': float(enclosed_charge)
        }


class DivergenceCalculator:
    def calculate(self, electric_field: np.ndarray) -> np.ndarray:
        return np.gradient(electric_field)


class GaussLawVerifier:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def verify(self, dirac_data: Dict[str, Any], flux_data: Dict[str, float]) -> Dict[str, Any]:
        total_charge = dirac_data['total_mass']
        enclosed_from_flux = flux_data['enclosed_charge']
        relative_error = np.abs((total_charge - enclosed_from_flux) / (total_charge + 1e-10))
        is_consistent = relative_error < 0.1
        
        return {
            'total_charge_direct': float(total_charge),
            'enclosed_charge_flux': float(enclosed_from_flux),
            'relative_error': float(relative_error),
            'is_consistent': bool(is_consistent)
        }


class StateSpaceExtractor:
    def extract(self, model: IModel) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        coeffs = model.get_coefficients()
        
        U = coeffs['U'].cpu().numpy()
        V = coeffs['V'].cpu().numpy()
        W = coeffs['W'].cpu().numpy()
        
        n_states = min(U.shape[0], V.shape[0])
        
        A = np.vstack([U[:n_states], V[:n_states]])
        n_states_actual = A.shape[0]
        A = A[:n_states_actual, :min(A.shape[1], n_states_actual)]
        
        if A.shape[1] < A.shape[0]:
            padding = np.zeros((A.shape[0], A.shape[0] - A.shape[1]))
            A = np.hstack([A, padding])
        
        B = np.random.randn(A.shape[0], 1) * 0.1
        C = W[:1, :A.shape[0]] if W.shape[1] >= A.shape[0] else np.pad(
            W[:1, :], ((0, 0), (0, A.shape[0] - W.shape[1])), mode='constant'
        )
        D = np.zeros((1, 1))
        
        return A, B, C, D


class TransferFunctionComputer:
    def compute(self, A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        try:
            eigenvalues = eigvals(A)
            num_coeffs = np.poly(eigenvalues)
            den_coeffs = np.poly(A)
            num_coeffs = np.real(num_coeffs)
            den_coeffs = np.real(den_coeffs)
        except Exception:
            num_coeffs = np.array([1.0])
            den_coeffs = np.array([1.0, 1.0])
        
        return num_coeffs, den_coeffs


class PoleZeroAnalyzer:
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray, config: AnalysisConfig = AnalysisConfig()):
        self.num = numerator
        self.den = denominator
        self.config = config
        self.poles = None
        self.zeros = None
        self.system = None
        self._compute()
    
    def _compute(self):
        self.zeros = np.roots(self.num) if len(self.num) > 1 else np.array([])
        self.poles = np.roots(self.den) if len(self.den) > 1 else np.array([])
        self.zeros = self.zeros[np.abs(self.zeros) < 1e10]
        self.poles = self.poles[np.abs(self.poles) < 1e10]
        
        try:
            self.system = signal.TransferFunction(self.num, self.den)
        except:
            self.system = None
    
    def get_poles(self) -> np.ndarray:
        return self.poles
    
    def get_zeros(self) -> np.ndarray:
        return self.zeros
    
    def analyze_stability(self) -> Dict[str, Any]:
        if len(self.poles) == 0:
            return {
                'is_stable': True,
                'stability_type': 'trivially_stable',
                'dominant_pole': None,
                'stability_margin': float('inf'),
                'unstable_poles': []
            }
        
        real_parts = np.real(self.poles)
        is_stable = np.all(real_parts < -self.config.STABILITY_MARGIN)
        unstable_poles = self.poles[real_parts >= -self.config.STABILITY_MARGIN]
        
        if is_stable:
            stability_type = 'asymptotically_stable'
        elif np.any(real_parts > self.config.STABILITY_MARGIN):
            stability_type = 'unstable'
        else:
            stability_type = 'marginally_stable'
        
        if len(real_parts) > 0:
            dominant_idx = np.argmax(real_parts)
            dominant_pole = self.poles[dominant_idx]
            stability_margin = -real_parts[dominant_idx]
        else:
            dominant_pole = None
            stability_margin = float('inf')
        
        return {
            'is_stable': bool(is_stable),
            'stability_type': stability_type,
            'dominant_pole': complex(dominant_pole) if dominant_pole is not None else None,
            'stability_margin': float(stability_margin),
            'unstable_poles': [complex(p) for p in unstable_poles],
            'num_unstable': len(unstable_poles)
        }
    
    def classify_poles(self) -> Dict[str, List[complex]]:
        real_poles = []
        complex_poles = []
        processed = set()
        
        for i, pole in enumerate(self.poles):
            if i in processed:
                continue
            
            if np.abs(np.imag(pole)) < self.config.POLE_ZERO_TOLERANCE:
                real_poles.append(complex(np.real(pole), 0))
                processed.add(i)
            else:
                complex_poles.append(complex(pole))
                processed.add(i)
                for j in range(i + 1, len(self.poles)):
                    if j not in processed:
                        if np.abs(pole - np.conj(self.poles[j])) < self.config.POLE_ZERO_TOLERANCE:
                            processed.add(j)
                            break
        
        return {
            'real_poles': real_poles,
            'complex_conjugate_pairs': complex_poles,
            'total_real': len(real_poles),
            'total_complex': len(complex_poles)
        }
    
    def compute_damping(self) -> List[Dict[str, float]]:
        results = []
        
        for pole in self.poles:
            real_part = np.real(pole)
            imag_part = np.imag(pole)
            
            if np.abs(imag_part) < self.config.POLE_ZERO_TOLERANCE:
                results.append({
                    'pole': complex(pole),
                    'natural_frequency': abs(real_part),
                    'damping_ratio': 1.0 if real_part < 0 else -1.0,
                    'damped_frequency': 0.0,
                    'type': 'overdamped'
                })
            else:
                omega_n = np.sqrt(real_part**2 + imag_part**2)
                zeta = -real_part / omega_n if omega_n > 0 else 0
                omega_d = abs(imag_part)
                
                if zeta > 1:
                    pole_type = 'overdamped'
                elif abs(zeta - 1.0) < self.config.POLE_ZERO_TOLERANCE:
                    pole_type = 'critically_damped'
                elif 0 < zeta < 1:
                    pole_type = 'underdamped'
                else:
                    pole_type = 'undamped'
                
                results.append({
                    'pole': complex(pole),
                    'natural_frequency': float(omega_n),
                    'damping_ratio': float(zeta),
                    'damped_frequency': float(omega_d),
                    'type': pole_type
                })
        
        return results
    
    def compute_time_constants(self) -> List[Dict[str, float]]:
        time_constants = []
        
        for pole in self.poles:
            real_part = np.real(pole)
            if real_part < -self.config.POLE_ZERO_TOLERANCE:
                tau = -1.0 / real_part
                settling_time = 4 * tau
                time_constants.append({
                    'pole': complex(pole),
                    'time_constant': float(tau),
                    'settling_time_4tau': float(settling_time),
                    'bandwidth': float(1.0 / tau)
                })
        
        return time_constants


class FrequencyResponseAnalyzer:
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray, config: AnalysisConfig = AnalysisConfig()):
        self.num = numerator
        self.den = denominator
        self.config = config
        
        try:
            self.system = signal.TransferFunction(numerator, denominator)
        except:
            self.system = None
    
    def compute_bode(self) -> Dict[str, np.ndarray]:
        if self.system is None:
            omega = np.logspace(np.log10(self.config.FREQUENCY_MIN),
                              np.log10(self.config.FREQUENCY_MAX),
                              self.config.FREQUENCY_SAMPLES)
            magnitude = np.zeros_like(omega)
            phase = np.zeros_like(omega)
        else:
            omega, magnitude, phase = signal.bode(self.system)
        
        return {
            'frequency': omega,
            'magnitude_db': magnitude,
            'phase_deg': phase
        }
    
    def compute_margins(self) -> Dict[str, float]:
        if self.system is None:
            return {
                'gain_margin_db': float('inf'),
                'phase_margin_deg': float('inf'),
                'gain_crossover_freq': 0.0,
                'phase_crossover_freq': 0.0
            }
        
        try:
            gm, pm, wgc, wpc = signal.margin(self.system)
            gm_db = 20 * np.log10(gm) if gm > 0 else float('inf')
            return {
                'gain_margin_db': float(gm_db),
                'phase_margin_deg': float(pm),
                'gain_crossover_freq': float(wgc),
                'phase_crossover_freq': float(wpc)
            }
        except:
            return {
                'gain_margin_db': float('inf'),
                'phase_margin_deg': float('inf'),
                'gain_crossover_freq': 0.0,
                'phase_crossover_freq': 0.0
            }
    
    def compute_nyquist(self) -> Dict[str, np.ndarray]:
        omega = np.logspace(np.log10(self.config.FREQUENCY_MIN),
                          np.log10(self.config.FREQUENCY_MAX),
                          self.config.FREQUENCY_SAMPLES)
        
        if self.system is None:
            real = np.zeros_like(omega)
            imag = np.zeros_like(omega)
        else:
            _, H = signal.freqresp(self.system, omega)
            real = np.real(H)
            imag = np.imag(H)
        
        return {
            'frequency': omega,
            'real': real,
            'imag': imag
        }
    
    def evaluate_nyquist_stability(self, nyquist_data: Dict[str, np.ndarray]) -> Dict[str, Any]:
        real = nyquist_data['real']
        imag = nyquist_data['imag']
        encirclements = 0
        critical_point = -1.0 + 0j
        
        for i in range(len(real) - 1):
            z1 = complex(real[i], imag[i]) - critical_point
            z2 = complex(real[i + 1], imag[i + 1]) - critical_point
            angle1 = np.angle(z1)
            angle2 = np.angle(z2)
            delta_angle = angle2 - angle1
            
            if delta_angle > np.pi:
                delta_angle -= 2 * np.pi
            elif delta_angle < -np.pi:
                delta_angle += 2 * np.pi
            
            encirclements += delta_angle
        
        encirclements = int(np.round(encirclements / (2 * np.pi)))
        min_distance = np.min(np.sqrt((real + 1)**2 + imag**2))
        
        return {
            'encirclements': encirclements,
            'is_stable': encirclements == 0,
            'distance_to_critical': float(min_distance),
            'stability_robustness': float(min_distance)
        }


class TimeResponseAnalyzer:
    def __init__(self, numerator: np.ndarray, denominator: np.ndarray, config: AnalysisConfig = AnalysisConfig()):
        self.num = numerator
        self.den = denominator
        self.config = config
        
        try:
            self.system = signal.TransferFunction(numerator, denominator)
        except:
            self.system = None
    
    def compute_step(self) -> Dict[str, np.ndarray]:
        t = np.linspace(0, self.config.TIME_MAX, self.config.TIME_SAMPLES)
        
        if self.system is None:
            y = np.zeros_like(t)
        else:
            t_out, y_out = signal.step(self.system, T=t)
            t = t_out
            y = y_out
        
        return {'time': t, 'output': y}
    
    def compute_impulse(self) -> Dict[str, np.ndarray]:
        t = np.linspace(0, self.config.TIME_MAX, self.config.TIME_SAMPLES)
        
        if self.system is None:
            y = np.zeros_like(t)
        else:
            t_out, y_out = signal.impulse(self.system, T=t)
            t = t_out
            y = y_out
        
        return {'time': t, 'output': y}
    
    def analyze_step_characteristics(self, step_data: Dict[str, np.ndarray]) -> Dict[str, float]:
        t = step_data['time']
        y = step_data['output']
        
        if len(y) == 0 or np.all(y == 0):
            return {
                'rise_time': 0.0,
                'settling_time': 0.0,
                'overshoot_percent': 0.0,
                'peak_time': 0.0,
                'steady_state_value': 0.0,
                'steady_state_error': 0.0
            }
        
        steady_state = y[-1]
        if abs(steady_state) < 1e-10:
            steady_state = 1.0
        
        threshold_10 = 0.1 * steady_state
        threshold_90 = 0.9 * steady_state
        
        rise_time = 0.0
        idx_10 = np.where(y >= threshold_10)[0]
        idx_90 = np.where(y >= threshold_90)[0]
        
        if len(idx_10) > 0 and len(idx_90) > 0:
            rise_time = t[idx_90[0]] - t[idx_10[0]]
        
        settling_band = 0.02 * abs(steady_state)
        settling_time = t[-1]
        
        for i in range(len(y) - 1, -1, -1):
            if abs(y[i] - steady_state) > settling_band:
                settling_time = t[i]
                break
        
        peak_value = np.max(y)
        peak_idx = np.argmax(y)
        peak_time = t[peak_idx]
        overshoot = ((peak_value - steady_state) / abs(steady_state)) * 100 if steady_state != 0 else 0
        steady_state_error = abs(1.0 - steady_state)
        
        return {
            'rise_time': float(rise_time),
            'settling_time': float(settling_time),
            'overshoot_percent': float(overshoot),
            'peak_time': float(peak_time),
            'steady_state_value': float(steady_state),
            'steady_state_error': float(steady_state_error)
        }


class CheckpointLoader:
    def load(self, path: str, device: str) -> Any:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")


class CheckpointMigrator:
    def migrate(self, raw_data: Any) -> Optional[Dict[str, torch.Tensor]]:
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
        
        if U.shape == (7, 4):
            u_padded = torch.zeros(AnalysisConfig.HIDDEN_DIM, 4, device=AnalysisConfig.DEVICE)
            v_padded = torch.zeros(AnalysisConfig.HIDDEN_DIM, 4, device=AnalysisConfig.DEVICE)
            w_padded = torch.zeros(4, AnalysisConfig.HIDDEN_DIM, device=AnalysisConfig.DEVICE)
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


class ChargeDistributionVisualizer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        charge_density = data['charge_density']
        point_positions = data['point_positions']
        point_charges = data['point_charges']
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.config.FIGURE_DPI)
        
        positions = np.arange(len(charge_density))
        ax.plot(positions, charge_density, color='#2E86AB', linewidth=1.5, label='Continuous charge density')
        ax.scatter(point_positions, point_charges, color='#A23B72', s=100, marker='o',
                  edgecolors='black', linewidths=1.5, label='Point charges (Dirac delta)', zorder=5)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.set_xlabel('Position index', fontsize=12)
        ax.set_ylabel('Charge density', fontsize=12)
        ax.set_title('Weight Charge Distribution', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


class ElectricFieldVisualizer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        electric_field = data['electric_field']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=self.config.FIGURE_DPI)
        
        positions = np.linspace(0, len(electric_field) - 1, len(electric_field))
        ax1.plot(positions, electric_field, color='#F18F01', linewidth=2, label='Electric field E(x)')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_xlabel('Position', fontsize=12)
        ax1.set_ylabel('Electric field (V/m)', fontsize=12)
        ax1.set_title('Electric Field Distribution', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        field_magnitude = np.abs(electric_field)
        norm = Normalize(vmin=field_magnitude.min(), vmax=field_magnitude.max())
        cmap = plt.get_cmap(self.config.COLORMAP)
        colors = cmap(norm(field_magnitude))
        
        for i in range(len(positions) - 1):
            ax2.plot(positions[i:i+2], electric_field[i:i+2], color=colors[i], linewidth=2)
        
        sm = ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax2)
        cbar.set_label('Field magnitude', fontsize=10)
        
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Position', fontsize=12)
        ax2.set_ylabel('Electric field (V/m)', fontsize=12)
        ax2.set_title('Electric Field (Color-coded)', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


class DivergenceVisualizer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        divergence = data['divergence']
        
        fig, ax = plt.subplots(figsize=(12, 6), dpi=self.config.FIGURE_DPI)
        
        positions = np.linspace(0, len(divergence) - 1, len(divergence))
        ax.fill_between(positions, divergence, 0, where=(divergence >= 0),
                       color='#06A77D', alpha=0.6, label='Positive divergence (sources)')
        ax.fill_between(positions, divergence, 0, where=(divergence < 0),
                       color='#D62828', alpha=0.6, label='Negative divergence (sinks)')
        
        ax.plot(positions, divergence, color='black', linewidth=1.5, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        
        ax.set_xlabel('Position', fontsize=12)
        ax.set_ylabel('Divergence of E', fontsize=12)
        ax.set_title('Divergence of Electric Field', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


class PoleZeroVisualizer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        poles = data['poles']
        zeros = data['zeros']
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.config.FIGURE_DPI)
        
        if len(poles) > 0:
            ax.scatter(np.real(poles), np.imag(poles), s=150, marker='x',
                      color='#D62828', linewidths=2.5, label='Poles', zorder=5)
        
        if len(zeros) > 0:
            ax.scatter(np.real(zeros), np.imag(zeros), s=150, marker='o',
                      facecolors='none', edgecolors='#06A77D', linewidths=2.5, label='Zeros', zorder=5)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        unit_circle = Circle((0, 0), 1, fill=False, edgecolor='black', linestyle=':', linewidth=1.5, alpha=0.3)
        ax.add_patch(unit_circle)
        
        ax.axvline(x=-self.config.STABILITY_MARGIN, color='#F18F01',
                  linestyle='--', linewidth=2, alpha=0.7, label='Stability margin')
        
        ax.set_xlabel('Real axis', fontsize=12)
        ax.set_ylabel('Imaginary axis', fontsize=12)
        ax.set_title('Pole-Zero Map', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


class BodeVisualizer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        bode_data = data['bode']
        margins = data['margins']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=self.config.FIGURE_DPI)
        
        ax1.semilogx(bode_data['frequency'], bode_data['magnitude_db'], color='#2E86AB', linewidth=2)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel('Magnitude (dB)', fontsize=12)
        ax1.set_title('Bode Diagram', fontsize=14, fontweight='bold')
        ax1.grid(True, which='both', alpha=0.3, linestyle=':')
        
        # Solo agregar línea y leyenda si hay crossover válido
        if margins['gain_crossover_freq'] > self.config.FREQUENCY_MIN:
            ax1.axvline(
                x=margins['gain_crossover_freq'],
                color='#F18F01',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=f"Gain crossover: {margins['gain_crossover_freq']:.2f} rad/s"
            )
            ax1.legend(loc='best', fontsize=9)
        
        ax2.semilogx(bode_data['frequency'], bode_data['phase_deg'], color='#A23B72', linewidth=2)
        ax2.axhline(y=-180, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Frequency (rad/s)', fontsize=12)
        ax2.set_ylabel('Phase (deg)', fontsize=12)
        ax2.grid(True, which='both', alpha=0.3, linestyle=':')
        
        # Solo agregar línea y leyenda si hay crossover válido
        if margins['phase_crossover_freq'] > self.config.FREQUENCY_MIN:
            ax2.axvline(
                x=margins['phase_crossover_freq'],
                color='#F18F01',
                linestyle='--',
                linewidth=1.5,
                alpha=0.7,
                label=f"Phase crossover: {margins['phase_crossover_freq']:.2f} rad/s"
            )
            ax2.legend(loc='best', fontsize=9)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


class NyquistVisualizer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        nyquist_data = data['nyquist']
        
        fig, ax = plt.subplots(figsize=(10, 10), dpi=self.config.FIGURE_DPI)
        
        real = nyquist_data['real']
        imag = nyquist_data['imag']
        
        ax.plot(real, imag, color='#2E86AB', linewidth=2, label='Nyquist plot')
        ax.plot(real, -imag, color='#2E86AB', linewidth=2, linestyle='--', alpha=0.5)
        
        ax.plot(-1, 0, marker='x', markersize=15, color='#D62828',
               markeredgewidth=3, label='Critical point (-1, 0)')
        
        critical_circle = Circle((-1, 0), 0.5, fill=False, edgecolor='#F18F01',
                                linestyle=':', linewidth=2, alpha=0.5)
        ax.add_patch(critical_circle)
        
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        
        ax.set_xlabel('Real axis', fontsize=12)
        ax.set_ylabel('Imaginary axis', fontsize=12)
        ax.set_title('Nyquist Diagram', fontsize=14, fontweight='bold')
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3, linestyle=':')
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


class TimeResponseVisualizer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        step_data = data['step']
        impulse_data = data['impulse']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), dpi=self.config.FIGURE_DPI)
        
        ax1.plot(step_data['time'], step_data['output'], color='#06A77D', linewidth=2, label='Step response')
        ax1.axhline(y=1.0, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Target')
        ax1.set_ylabel('Output', fontsize=12)
        ax1.set_title('Step Response', fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        ax2.plot(impulse_data['time'], impulse_data['output'], color='#A23B72', linewidth=2, label='Impulse response')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel('Output', fontsize=12)
        ax2.set_title('Impulse Response', fontsize=14, fontweight='bold')
        ax2.legend(loc='best', fontsize=10)
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


class CombinedVisualizer:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        charge_density = data['charge_density']
        point_positions = data['point_positions']
        point_charges = data['point_charges']
        electric_field = data['electric_field']
        divergence = data['divergence']
        poles = data['poles']
        zeros = data['zeros']
        
        fig = plt.figure(figsize=(18, 12), dpi=self.config.FIGURE_DPI)
        
        ax1 = plt.subplot(2, 3, 1)
        positions_charge = np.arange(len(charge_density))
        ax1.plot(positions_charge, charge_density, color='#2E86AB', linewidth=1.5)
        ax1.scatter(point_positions, point_charges, color='#A23B72', s=80,
                   marker='o', edgecolors='black', linewidths=1.2, zorder=5)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax1.set_ylabel('Charge density', fontsize=10)
        ax1.set_title('Charge Distribution', fontsize=11, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        ax2 = plt.subplot(2, 3, 2)
        positions_field = np.linspace(0, len(electric_field) - 1, len(electric_field))
        ax2.plot(positions_field, electric_field, color='#F18F01', linewidth=2)
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax2.set_ylabel('Electric field (V/m)', fontsize=10)
        ax2.set_title('Electric Field', fontsize=11, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle=':')
        
        ax3 = plt.subplot(2, 3, 3)
        positions_div = np.linspace(0, len(divergence) - 1, len(divergence))
        ax3.fill_between(positions_div, divergence, 0, where=(divergence >= 0),
                         color='#06A77D', alpha=0.6)
        ax3.fill_between(positions_div, divergence, 0, where=(divergence < 0),
                         color='#D62828', alpha=0.6)
        ax3.plot(positions_div, divergence, color='black', linewidth=1.5, alpha=0.8)
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax3.set_ylabel('Divergence of E', fontsize=10)
        ax3.set_title('Divergence', fontsize=11, fontweight='bold')
        ax3.grid(True, alpha=0.3, linestyle=':')
        
        ax4 = plt.subplot(2, 3, 4)
        if len(poles) > 0:
            ax4.scatter(np.real(poles), np.imag(poles), s=100, marker='x',
                       color='#D62828', linewidths=2, label='Poles')
        if len(zeros) > 0:
            ax4.scatter(np.real(zeros), np.imag(zeros), s=100, marker='o',
                       facecolors='none', edgecolors='#06A77D', linewidths=2, label='Zeros')
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax4.axvline(x=0, color='gray', linestyle='--', linewidth=1, alpha=0.5)
        ax4.set_xlabel('Real', fontsize=10)
        ax4.set_ylabel('Imaginary', fontsize=10)
        ax4.set_title('Pole-Zero Map', fontsize=11, fontweight='bold')
        ax4.legend(loc='best', fontsize=8)
        ax4.grid(True, alpha=0.3)
        ax4.set_aspect('equal')
        
        ax5 = plt.subplot(2, 3, 5)
        weights_u = []
        weights_v = []
        weights_w = []
        for pos in range(min(len(charge_density), 50)):
            if pos < len(charge_density) // 3:
                weights_u.append(charge_density[pos])
            elif pos < 2 * len(charge_density) // 3:
                weights_v.append(charge_density[pos])
            else:
                weights_w.append(charge_density[pos])
        
        parts = [weights_u, weights_v, weights_w]
        labels = ['U weights', 'V weights', 'W weights']
        colors = ['#2E86AB', '#F18F01', '#A23B72']
        
        # Usar tick_labels en lugar de labels para compatibilidad con Matplotlib 3.9+
        bp = ax5.boxplot(parts, tick_labels=labels, patch_artist=True)
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax5.set_ylabel('Weight value', fontsize=10)
        ax5.set_title('Weight Distribution by Layer', fontsize=11, fontweight='bold')
        ax5.grid(True, alpha=0.3, axis='y')
        
        ax6 = plt.subplot(2, 3, 6)
        all_weights = np.concatenate([weights_u, weights_v, weights_w])
        hist, bins = np.histogram(all_weights, bins=30, density=True)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        ax6.bar(bin_centers, hist, width=np.diff(bins), alpha=0.7,
               color='#06A77D', edgecolor='black')
        ax6.set_xlabel('Weight value', fontsize=10)
        ax6.set_ylabel('Density', fontsize=10)
        ax6.set_title('Overall Weight Histogram', fontsize=11, fontweight='bold')
        ax6.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Complete System Analysis', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()



class SystemAnalyzer:
    def __init__(self, checkpoint_path: str, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        self.checkpoint_loader = CheckpointLoader()
        self.checkpoint_migrator = CheckpointMigrator()
        self.charge_extractor = ChargeDistributionExtractor()
        self.dirac_analyzer = DiracDeltaAnalyzer(config)
        self.field_calculator = ElectricFieldCalculator(config)
        self.flux_calculator = ElectricFluxCalculator(config)
        self.divergence_calculator = DivergenceCalculator()
        self.gauss_verifier = GaussLawVerifier(config)
        self.state_extractor = StateSpaceExtractor()
        self.tf_computer = TransferFunctionComputer()
        
        self._load_model()
    
    def _load_model(self):
        checkpoint = self.checkpoint_loader.load(self.checkpoint_path, self.config.DEVICE)
        
        self.model = BilinearModel(
            hidden_dim=self.config.HIDDEN_DIM,
            matrix_size=self.config.MATRIX_SIZE
        ).to(self.config.DEVICE)
        
        migrated_state = self.checkpoint_migrator.migrate(checkpoint)
        if migrated_state is not None:
            self.model.load_state_dict(migrated_state)
        else:
            raise RuntimeError(f"Failed to migrate checkpoint: {self.checkpoint_path}")
        
        self.epoch = checkpoint.get('epoch', 'unknown')
    
    def analyze(self) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        charge_density = self.charge_extractor.extract(self.model)
        dirac_data = self.dirac_analyzer.analyze(charge_density)
        electric_field = self.field_calculator.calculate(dirac_data, None)
        flux_data = self.flux_calculator.calculate(electric_field, None)
        divergence = self.divergence_calculator.calculate(electric_field)
        gauss_verification = self.gauss_verifier.verify(dirac_data, flux_data)
        
        A, B, C, D = self.state_extractor.extract(self.model)
        num, den = self.tf_computer.compute(A, B, C, D)
        
        pz_analyzer = PoleZeroAnalyzer(num, den, self.config)
        stability = pz_analyzer.analyze_stability()
        pole_classification = pz_analyzer.classify_poles()
        damping = pz_analyzer.compute_damping()
        time_constants = pz_analyzer.compute_time_constants()
        
        freq_analyzer = FrequencyResponseAnalyzer(num, den, self.config)
        bode_data = freq_analyzer.compute_bode()
        margins = freq_analyzer.compute_margins()
        nyquist_data = freq_analyzer.compute_nyquist()
        nyquist_stability = freq_analyzer.evaluate_nyquist_stability(nyquist_data)
        
        time_analyzer = TimeResponseAnalyzer(num, den, self.config)
        step_response = time_analyzer.compute_step()
        impulse_response = time_analyzer.compute_impulse()
        step_characteristics = time_analyzer.analyze_step_characteristics(step_response)
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat()
            },
            'dirac_distribution': {
                'num_point_charges': dirac_data['num_point_charges'],
                'delta_strength': dirac_data['delta_strength'],
                'discrete_mass': dirac_data['discrete_mass'],
                'continuous_mass': dirac_data['continuous_mass'],
                'total_mass': dirac_data['total_mass']
            },
            'electric_field': {
                'max_magnitude': float(np.max(np.abs(electric_field))),
                'mean_magnitude': float(np.mean(np.abs(electric_field))),
                'std_magnitude': float(np.std(electric_field))
            },
            'electric_flux': flux_data,
            'gauss_law': gauss_verification,
            'divergence': {
                'max': float(np.max(divergence)),
                'min': float(np.min(divergence)),
                'mean': float(np.mean(divergence)),
                'std': float(np.std(divergence))
            },
            'state_space': {
                'A_shape': A.shape,
                'B_shape': B.shape,
                'C_shape': C.shape,
                'D_shape': D.shape,
                'num_states': A.shape[0]
            },
            'transfer_function': {
                'numerator': num.tolist(),
                'denominator': den.tolist(),
                'order': len(den) - 1
            },
            'poles_zeros': {
                'poles': [complex(p) for p in pz_analyzer.get_poles()],
                'zeros': [complex(z) for z in pz_analyzer.get_zeros()],
                'num_poles': len(pz_analyzer.get_poles()),
                'num_zeros': len(pz_analyzer.get_zeros())
            },
            'stability': stability,
            'pole_classification': pole_classification,
            'damping': damping,
            'time_constants': time_constants,
            'frequency_margins': margins,
            'nyquist_stability': nyquist_stability,
            'step_response_characteristics': step_characteristics
        }
        
        plot_data = {
            'charge_density': dirac_data['charge_density_full'],
            'point_charges': dirac_data['point_charges'],
            'point_positions': dirac_data['point_positions'],
            'electric_field': electric_field,
            'divergence': divergence,
            'poles': pz_analyzer.get_poles(),
            'zeros': pz_analyzer.get_zeros(),
            'bode': bode_data,
            'margins': margins,
            'nyquist': nyquist_data,
            'step': step_response,
            'impulse': impulse_response
        }
        
        self._print_report(results)
        
        return results, plot_data
    
    def _print_report(self, results: Dict):
        print("=" * 70)
        print("SYSTEM ANALYSIS REPORT")
        print("=" * 70)
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {results['metadata']['checkpoint_path']}")
        print(f"  Epoch: {results['metadata']['epoch']}")
        
        print(f"\n[DIRAC DISTRIBUTION]")
        dd = results['dirac_distribution']
        print(f"  Point charges: {dd['num_point_charges']}")
        print(f"  Delta strength: {dd['delta_strength']:.6f}")
        print(f"  Discrete mass: {dd['discrete_mass']:.6e}")
        print(f"  Continuous mass: {dd['continuous_mass']:.6e}")
        print(f"  Total mass: {dd['total_mass']:.6e}")
        
        print(f"\n[ELECTRIC FIELD]")
        ef = results['electric_field']
        print(f"  Max magnitude: {ef['max_magnitude']:.6e} V/m")
        print(f"  Mean magnitude: {ef['mean_magnitude']:.6e} V/m")
        print(f"  Std deviation: {ef['std_magnitude']:.6e} V/m")
        
        print(f"\n[ELECTRIC FLUX]")
        flux = results['electric_flux']
        print(f"  Outward flux: {flux['flux_outward']:.6e} V·m")
        print(f"  Inward flux: {flux['flux_inward']:.6e} V·m")
        print(f"  Net flux: {flux['flux_net']:.6e} V·m")
        print(f"  Enclosed charge: {flux['enclosed_charge']:.6e} C")
        
        print(f"\n[GAUSS LAW VERIFICATION]")
        gauss = results['gauss_law']
        print(f"  Total charge (direct): {gauss['total_charge_direct']:.6e} C")
        print(f"  Enclosed charge (flux): {gauss['enclosed_charge_flux']:.6e} C")
        print(f"  Relative error: {gauss['relative_error']:.6f}")
        print(f"  Gauss consistent: {gauss['is_consistent']}")
        
        print(f"\n[STATE SPACE]")
        ss = results['state_space']
        print(f"  A matrix: {ss['A_shape']}")
        print(f"  B matrix: {ss['B_shape']}")
        print(f"  C matrix: {ss['C_shape']}")
        print(f"  Number of states: {ss['num_states']}")
        
        print(f"\n[POLES AND ZEROS]")
        pz = results['poles_zeros']
        print(f"  Number of poles: {pz['num_poles']}")
        print(f"  Number of zeros: {pz['num_zeros']}")
        
        print(f"\n[STABILITY ANALYSIS]")
        stab = results['stability']
        print(f"  Stable: {stab['is_stable']}")
        print(f"  Stability type: {stab['stability_type']}")
        print(f"  Stability margin: {stab['stability_margin']:.6f}")
        print(f"  Unstable poles: {stab['num_unstable']}")
        if stab['dominant_pole'] is not None:
            dp = stab['dominant_pole']
            print(f"  Dominant pole: {dp.real:.6f} + {dp.imag:.6f}j")
        
        print(f"\n[FREQUENCY DOMAIN]")
        margins = results['frequency_margins']
        print(f"  Gain margin: {margins['gain_margin_db']:.2f} dB")
        print(f"  Phase margin: {margins['phase_margin_deg']:.2f} deg")
        print(f"  Gain crossover: {margins['gain_crossover_freq']:.6f} rad/s")
        print(f"  Phase crossover: {margins['phase_crossover_freq']:.6f} rad/s")
        
        print(f"\n[NYQUIST STABILITY]")
        nyq = results['nyquist_stability']
        print(f"  Encirclements: {nyq['encirclements']}")
        print(f"  Stable: {nyq['is_stable']}")
        print(f"  Distance to critical point: {nyq['distance_to_critical']:.6f}")
        
        print(f"\n[STEP RESPONSE]")
        step = results['step_response_characteristics']
        print(f"  Rise time: {step['rise_time']:.6f} s")
        print(f"  Settling time: {step['settling_time']:.6f} s")
        print(f"  Overshoot: {step['overshoot_percent']:.2f} %")
        print(f"  Peak time: {step['peak_time']:.6f} s")
        print(f"  Steady state value: {step['steady_state_value']:.6f}")
        print(f"  Steady state error: {step['steady_state_error']:.6f}")
        
        print("=" * 70)


class AnalysisPipeline:
    def __init__(self, config: AnalysisConfig = AnalysisConfig()):
        self.config = config
        self.charge_viz = ChargeDistributionVisualizer(config)
        self.field_viz = ElectricFieldVisualizer(config)
        self.divergence_viz = DivergenceVisualizer(config)
        self.pole_zero_viz = PoleZeroVisualizer(config)
        self.bode_viz = BodeVisualizer(config)
        self.nyquist_viz = NyquistVisualizer(config)
        self.time_viz = TimeResponseVisualizer(config)
        self.combined_viz = CombinedVisualizer(config)
    
    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer = SystemAnalyzer(checkpoint_path, self.config)
        results, plot_data = analyzer.analyze()
        
        base_name = Path(checkpoint_path).stem
        
        self.charge_viz.visualize({
            'charge_density': plot_data['charge_density'],
            'point_positions': plot_data['point_positions'],
            'point_charges': plot_data['point_charges']
        }, os.path.join(output_dir, f'{base_name}_charge.{self.config.SAVE_FORMAT}'))
        
        self.field_viz.visualize({
            'electric_field': plot_data['electric_field']
        }, os.path.join(output_dir, f'{base_name}_field.{self.config.SAVE_FORMAT}'))
        
        self.divergence_viz.visualize({
            'divergence': plot_data['divergence']
        }, os.path.join(output_dir, f'{base_name}_divergence.{self.config.SAVE_FORMAT}'))
        
        self.pole_zero_viz.visualize({
            'poles': plot_data['poles'],
            'zeros': plot_data['zeros']
        }, os.path.join(output_dir, f'{base_name}_poles.{self.config.SAVE_FORMAT}'))
        
        self.bode_viz.visualize({
            'bode': plot_data['bode'],
            'margins': plot_data['margins']
        }, os.path.join(output_dir, f'{base_name}_bode.{self.config.SAVE_FORMAT}'))
        
        self.nyquist_viz.visualize({
            'nyquist': plot_data['nyquist']
        }, os.path.join(output_dir, f'{base_name}_nyquist.{self.config.SAVE_FORMAT}'))
        
        self.time_viz.visualize({
            'step': plot_data['step'],
            'impulse': plot_data['impulse']
        }, os.path.join(output_dir, f'{base_name}_time.{self.config.SAVE_FORMAT}'))
        
        self.combined_viz.visualize({
            'charge_density': plot_data['charge_density'],
            'point_positions': plot_data['point_positions'],
            'point_charges': plot_data['point_charges'],
            'electric_field': plot_data['electric_field'],
            'divergence': plot_data['divergence'],
            'poles': plot_data['poles'],
            'zeros': plot_data['zeros']
        }, os.path.join(output_dir, f'{base_name}_combined.{self.config.SAVE_FORMAT}'))
        
        results_path = os.path.join(output_dir, f'{base_name}_results.json')
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
        
        # Si n_latest es None, procesar todos los checkpoints
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
            'aggregate_statistics': self._compute_aggregate_statistics(all_results),
            'individual_results': all_results
        }
        
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Generar reporte de texto legible
        self._generate_text_report(summary, output_dir)
        
        print(f"\nSaved summary: {summary_path}")
    
    def _compute_aggregate_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        # Extraer métricas clave de todos los resultados
        delta_strengths = []
        stability_values = []
        num_unstable_list = []
        stability_margins = []
        gain_margins = []
        phase_margins = []
        delta_strengths_stable = []
        delta_strengths_unstable = []
        
        for r in results:
            # Delta strength
            ds = r.get('dirac_distribution', {}).get('delta_strength', 0)
            delta_strengths.append(ds)
            
            # Estabilidad
            is_stable = r.get('stability', {}).get('is_stable', False)
            stability_values.append(is_stable)
            
            if is_stable:
                delta_strengths_stable.append(ds)
            else:
                delta_strengths_unstable.append(ds)
            
            num_unstable = r.get('stability', {}).get('num_unstable', 0)
            num_unstable_list.append(num_unstable)
            
            margin = r.get('stability', {}).get('stability_margin', 0)
            stability_margins.append(margin)
            
            gm = r.get('frequency_margins', {}).get('gain_margin_db', float('inf'))
            if gm != float('inf'):
                gain_margins.append(gm)
            
            pm = r.get('frequency_margins', {}).get('phase_margin_deg', float('inf'))
            if pm != float('inf'):
                phase_margins.append(pm)
        
        return {
            'delta_strength': {
                'mean': float(np.mean(delta_strengths)) if delta_strengths else 0,
                'std': float(np.std(delta_strengths)) if delta_strengths else 0,
                'min': float(np.min(delta_strengths)) if delta_strengths else 0,
                'max': float(np.max(delta_strengths)) if delta_strengths else 0,
                'mean_stable': float(np.mean(delta_strengths_stable)) if delta_strengths_stable else 0,
                'mean_unstable': float(np.mean(delta_strengths_unstable)) if delta_strengths_unstable else 0
            },
            'stability': {
                'num_stable': sum(stability_values),
                'num_unstable': len(stability_values) - sum(stability_values),
                'stability_rate': sum(stability_values) / len(stability_values) if stability_values else 0
            },
            'unstable_poles': {
                'mean': float(np.mean(num_unstable_list)) if num_unstable_list else 0,
                'max': int(np.max(num_unstable_list)) if num_unstable_list else 0
            },
            'stability_margin': {
                'mean': float(np.mean(stability_margins)) if stability_margins else 0,
                'min': float(np.min(stability_margins)) if stability_margins else 0
            },
            'frequency_margins': {
                'gain_margin_mean': float(np.mean(gain_margins)) if gain_margins else float('inf'),
                'phase_margin_mean': float(np.mean(phase_margins)) if phase_margins else float('inf')
            }
        }
    
    def _generate_text_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        report_path = os.path.join(output_dir, 'summary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("AGGREGATE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total checkpoints analyzed: {summary['total_checkpoints_analyzed']}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            stats = summary['aggregate_statistics']
            
            f.write("-" * 80 + "\n")
            f.write("DELTA STRENGTH STATISTICS\n")
            f.write("-" * 80 + "\n")
            ds = stats['delta_strength']
            f.write(f"  Mean: {ds['mean']:.6f}\n")
            f.write(f"  Std:  {ds['std']:.6f}\n")
            f.write(f"  Min:  {ds['min']:.6f}\n")
            f.write(f"  Max:  {ds['max']:.6f}\n")
            f.write(f"  Mean (stable systems):   {ds['mean_stable']:.6f}\n")
            f.write(f"  Mean (unstable systems): {ds['mean_unstable']:.6f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("STABILITY STATISTICS\n")
            f.write("-" * 80 + "\n")
            stab = stats['stability']
            f.write(f"  Stable systems:   {stab['num_stable']}\n")
            f.write(f"  Unstable systems: {stab['num_unstable']}\n")
            f.write(f"  Stability rate:   {stab['stability_rate']:.2%}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("UNSTABLE POLES STATISTICS\n")
            f.write("-" * 80 + "\n")
            up = stats['unstable_poles']
            f.write(f"  Mean per system: {up['mean']:.2f}\n")
            f.write(f"  Max in any system: {up['max']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("STABILITY MARGIN STATISTICS\n")
            f.write("-" * 80 + "\n")
            sm = stats['stability_margin']
            f.write(f"  Mean: {sm['mean']:.6f}\n")
            f.write(f"  Min:  {sm['min']:.6f}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("FREQUENCY MARGIN STATISTICS\n")
            f.write("-" * 80 + "\n")
            fm = stats['frequency_margins']
            f.write(f"  Mean gain margin:  {fm['gain_margin_mean']:.2f} dB\n")
            f.write(f"  Mean phase margin: {fm['phase_margin_mean']:.2f} deg\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("INDIVIDUAL CHECKPOINT SUMMARIES\n")
            f.write("=" * 80 + "\n\n")
            
            for i, r in enumerate(summary['individual_results'], 1):
                f.write(f"[{i}] {r['metadata']['checkpoint_path']}\n")
                f.write(f"    Epoch: {r['metadata']['epoch']}\n")
                f.write(f"    Delta strength: {r['dirac_distribution']['delta_strength']:.6f}\n")
                f.write(f"    Stable: {r['stability']['is_stable']}\n")
                f.write(f"    Stability type: {r['stability']['stability_type']}\n")
                f.write(f"    Unstable poles: {r['stability']['num_unstable']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved text report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Dirac delta and control theory analysis for complex systems'
    )
    parser.add_argument(
        'checkpoint',
        nargs='?',
        default=None,
        help='Path to specific checkpoint file (optional)'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all checkpoints in directory (default behavior if no checkpoint specified)'
    )
    parser.add_argument(
        '--latest',
        type=int,
        default=None,
        help='Process only N latest checkpoints (overrides --all)'
    )
    parser.add_argument(
        '--dir',
        default='checkpoints',
        help='Checkpoint directory to scan'
    )
    parser.add_argument(
        '--output',
        default='analysis_output',
        help='Output directory for plots and results'
    )
    
    args = parser.parse_args()
    
    config = AnalysisConfig()
    pipeline = AnalysisPipeline(config)
    
    # Caso 1: Checkpoint específico proporcionado
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            pipeline.process_checkpoint(args.checkpoint, args.output)
        else:
            print(f"Error: Checkpoint file not found: {args.checkpoint}")
            return
    
    # Caso 2: Procesar todos los checkpoints
    elif args.all or args.latest is None:
        # Si --latest no está especificado, procesar todos (n_latest=None)
        n_to_process = args.latest if args.latest is not None else None
        results = pipeline.process_directory(args.dir, n_to_process, args.output)
        if results:
            pipeline.generate_summary(results, args.output)
    
    # Caso 3: Procesar N últimos checkpoints
    elif args.latest:
        results = pipeline.process_directory(args.dir, args.latest, args.output)
        if results:
            pipeline.generate_summary(results, args.output)

if __name__ == '__main__':
    main()