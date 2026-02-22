#!/usr/bin/env python3
"""
maxwell_strassen_analysis.py

Electromagnetic Analysis of Strassen Algorithm Crystallization
Applies Maxwell's Equations to distinguish Crystalline (Strassen) phases 
from Amorphous (Glass) phases in neural network weight distributions.

Implements:
- Mapping of weights to 3D Dielectric/Charge lattices.
- Solving Electrostatic Potential (Poisson Equation).
- Calculating EM Scattering (Bragg peaks vs Rayleigh scattering).
- Photonic Entropy and Bandgap analysis.
"""

import torch
import torch.nn as nn
import numpy as np
import json
import os
import argparse
import time
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Protocol, runtime_checkable
from pathlib import Path
from dataclasses import dataclass, field
from scipy.fft import fftn, fftshift
from scipy.stats import entropy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import warnings

warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass(frozen=True)
class MaxwellConfiguration:
    """
    Configuration for Maxwellian analysis of Strassen crystals.
    
    Physics Parameters:
    -------------------
    LATTICE_CONSTANT: Spacing between weight nodes (arbitrary units, e.g., nm scale).
    PERMITTIVITY_VACUUM: ε_0 (scaled for numerical stability).
    PERMITTIVITY_WEIGHT_SCALE: Factor to convert weight magnitude to dielectric contrast.
    
    Simulation Parameters:
    ----------------------
    GRID_DIMENSION: Size of the cubic lattice for weight embedding (N x N x N).
    FREQUENCY_SAMPLES: Number of frequency points for spectral analysis.
    WAVEVECTOR_RANGE: Range of k-vectors for scattering simulation.
    
    Analysis Thresholds:
    --------------------
    CRYSTALLINITY_THRESHOLD: Entropy threshold to classify as Crystal vs Glass.
    BRAGG_PEAK_SHARPNESS: Minimum prominence for a peak to be considered Bragg scattering.
    """
    # Architecture Dimensions (Must match model)
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    INPUT_DIM: int = 4
    EXPANSION_SCALES: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    
    # Electromagnetic Physics Constants
    LATTICE_CONSTANT: float = 1.0  # Unit lattice spacing
    PERMITTIVITY_VACUUM: float = 8.854e-12
    PERMEABILITY_VACUUM: float = 4 * np.pi * 1e-7
    SPEED_OF_LIGHT: float = 3e8
    PERMITTIVITY_WEIGHT_SCALE: float = 1.0  # Scaling for dielectric strength
    
    # Simulation Grid
    GRID_DIMENSION: int = 16  # 3D Lattice size (power of 2 preferred for FFT)
    
    # Spectral Analysis
    FREQUENCY_SAMPLES: int = 100
    WAVEVECTOR_MIN: float = -np.pi
    WAVEVECTOR_MAX: float = np.pi
    
    # Phase Classification Thresholds
    CRYSTALLINITY_ENTROPY_THRESHOLD: float = 2.0
    BRAGG_PEAK_PROMINENCE: float = 0.8
    ANISOTROPY_RATIO_THRESHOLD: float = 5.0
    
    # Discretization (From Percolation/MBL logic)
    DISCRETIZATION_MARGIN: float = 0.1
    ALPHA_THRESHOLD_CRYSTAL: float = 7.0
    ALPHA_THRESHOLD_GLASS: float = 1.0
    GLASS_TEMPERATURE_THRESHOLD: float = 0.1
    CRYSTAL_TEMPERATURE_THRESHOLD: float = 0.01
    
    # Checkpointing
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    METRIC_PRECISION: int = 6
    
    def get_effective_input_dim(self) -> int:
        return self.MATRIX_SIZE * self.MATRIX_SIZE
    
    def get_total_parameters(self) -> int:
        input_dim = self.get_effective_input_dim()
        return (input_dim * self.HIDDEN_DIM * 2) + (self.HIDDEN_DIM * input_dim)


# =============================================================================
# PROTOCOLS (INTERFACES)
# =============================================================================

@runtime_checkable
class IModel(Protocol):
    def get_coefficients(self) -> Dict[str, np.ndarray]: ...


@runtime_checkable
class IGeometryMapper(Protocol):
    def map_weights_to_lattice(self, weights: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class IMaxwellSolver(Protocol):
    def solve_poisson(self, charge_density: np.ndarray, permittivity: np.ndarray) -> np.ndarray: ...
    def compute_scattering(self, permittivity: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class IDielectricAnalyzer(Protocol):
    def analyze_permittivity_tensor(self, permittivity: np.ndarray) -> Dict[str, Any]: ...


@runtime_checkable
class IPhaseClassifier(Protocol):
    def classify(self, metrics: Dict[str, Any]) -> str: ...


# =============================================================================
# MODEL AND CHECKPOINT HANDLING
# =============================================================================

class BilinearStrassenModel(nn.Module):
    def __init__(self, config: MaxwellConfiguration):
        super().__init__()
        self.config = config
        input_dim = config.get_effective_input_dim()
        
        self.U = nn.Linear(input_dim, config.HIDDEN_DIM, bias=False)
        self.V = nn.Linear(input_dim, config.HIDDEN_DIM, bias=False)
        self.W = nn.Linear(config.HIDDEN_DIM, input_dim, bias=False)
        self._initialize_weights()

    def _initialize_weights(self):
        nn.init.xavier_uniform_(self.U.weight)
        self.V.weight.data = self.U.weight.data.clone()
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, a, b):
        return self.W(self.U(a) * self.V(b))

    def get_coefficients(self) -> Dict[str, np.ndarray]:
        return {
            'U': self.U.weight.data.cpu().numpy(),
            'V': self.V.weight.data.cpu().numpy(),
            'W': self.W.weight.data.cpu().numpy()}


class CheckpointMigrator:
    def migrate(self, raw_data: Any, config: MaxwellConfiguration) -> Optional[Dict[str, np.ndarray]]:
        if isinstance(raw_data, dict):
            candidates = ['state_dict', 'model_state_dict']
            for key in candidates:
                if key in raw_data:
                    result = self._migrate_dict(raw_data[key], config)
                    if result is not None:
                        return result
            result = self._migrate_dict(raw_data, config)
            if result is not None:
                return result
        return None

    def _migrate_dict(self, state_dict: Dict[str, Any], config: MaxwellConfiguration) -> Optional[Dict[str, np.ndarray]]:
        if any(k in state_dict for k in ['U', 'V', 'W']):
            return self._migrate_custom(state_dict, config)
        elif 'U_coefs' in state_dict:
            return self._migrate_coefs(state_dict)
        elif any(k.endswith('.weight') for k in state_dict.keys()):
            return self._migrate_standard(state_dict)
        return None

    def _migrate_custom(self, sd: Dict[str, Any], config: MaxwellConfiguration) -> Optional[Dict[str, np.ndarray]]:
        U = sd.get('U', sd.get('U_coefs'))
        V = sd.get('V', sd.get('V_coefs'))
        W = sd.get('W', sd.get('W_coefs'))
        if U is None or V is None or W is None:
            return None
        U, V, W = self._np(U), self._np(V), self._np(W)
        return {'U': U, 'V': V, 'W': W}

    def _migrate_coefs(self, sd: Dict[str, Any]) -> Dict[str, np.ndarray]:
        return {'U': self._np(sd['U_coefs']),
                'V': self._np(sd['V_coefs']),
                'W': self._np(sd['W_coefs'])}

    def _migrate_standard(self, sd: Dict[str, Any]) -> Optional[Dict[str, np.ndarray]]:
        m = {'U.weight': 'U', 'V.weight': 'V', 'W.weight': 'W'}
        r = {}
        for key, name in m.items():
            if key in sd:
                r[name] = self._np(sd[key])
        return r if len(r) == 3 else None

    @staticmethod
    def _np(tensor: Any) -> np.ndarray:
        if hasattr(tensor, 'cpu'):
            return tensor.detach().cpu().numpy() if hasattr(tensor, 'detach') else tensor.cpu().numpy()
        return np.array(tensor)


# =============================================================================
# GEOMETRY MAPPING (WEIGHTS -> PHYSICS SPACE)
# =============================================================================

class StrassenGeometryMapper:
    """
    Maps abstract weight vectors to a 3D Dielectric Lattice.
    
    Strategy:
    1. Flatten U, V, W into a single vector.
    2. Populate a 3D grid (NxNxN) using a space-filling curve (Z-order) 
       or layer-wise assignment.
    3. Compute Effective Charge Density ρ and Permittivity ε.
    
    Crystal Hypothesis:
    - Crystallized weights (discrete values) form ordered arrays.
    - Glass weights (random) form noise.
    """
    def __init__(self, config: MaxwellConfiguration):
        self.config = config

    def map_weights_to_lattice(self, weights: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Returns:
            charge_density: 3D array of charge distribution.
            permittivity: 3D array of dielectric constants.
        """
        N = self.config.GRID_DIMENSION
        total_params = sum(w.size for w in weights.values())
        
        # 1. Flatten and Pad
        flat_weights = np.concatenate([w.flatten() for w in weights.values()])
        volume = N ** 3
        
        # Normalize weights to act as physical properties
        # Weights -> Dielectric Modulation (epsilon = epsilon_0 * (1 + chi))
        # Chi (susceptibility) is derived from weight deviation
        weight_mean = np.mean(flat_weights)
        susceptibility = flat_weights - weight_mean
        
        # 2. Embed into 3D Grid (Z-order curve logic for locality)
        permittivity = np.ones((N, N, N), dtype=np.float32)
        charge_density = np.zeros((N, N, N), dtype=np.float32)
        
        idx = 0
        for x in range(N):
            for y in range(N):
                for z in range(N):
                    if idx < len(flat_weights):
                        # Permittivity: Base + Weight influence
                        # epsilon_r = 1 + alpha * |w| or complex
                        permittivity[x, y, z] = 1.0 + self.config.PERMITTIVITY_WEIGHT_SCALE * flat_weights[idx]
                        
                        # Charge Density: Approximated by divergence of polarization
                        # Simplification: ρ ∝ w
                        charge_density[x, y, z] = flat_weights[idx]
                        idx += 1
                    else:
                        # Padding
                        permittivity[x, y, z] = 1.0
                        charge_density[x, y, z] = 0.0
        
        return charge_density, permittivity


class DielectricTensorAnalyzer:
    """
    Analyzes the anisotropy of the dielectric medium.
    
    Glass: Isotropic (ε ~ scalar)
    Crystal: Anisotropic (ε ~ tensor with specific principal axes)
    """
    def __init__(self, config: MaxwellConfiguration):
        self.config = config

    def analyze_permittivity_tensor(self, permittivity: np.ndarray) -> Dict[str, Any]:
        """
        Analyze the effective permittivity tensor of the medium.
        """
        # Compute covariance structure as a proxy for the dielectric tensor
        # Flatten spatial dimensions
        flat_eps = permittivity.flatten()
        
        # Construct a 3x3 tensor representing anisotropy along axes
        # <E_x E_x>, <E_x E_y>, etc.
        # Using gradient structure
        grad_x = np.gradient(permittivity, axis=0)
        grad_y = np.gradient(permittivity, axis=1)
        grad_z = np.gradient(permittivity, axis=2)
        
        # Structure Tensor (used in image processing for anisotropy)
        # J_ij = sum( grad_i * grad_j )
        J = np.zeros((3, 3))
        J[0, 0] = np.sum(grad_x**2)
        J[1, 1] = np.sum(grad_y**2)
        J[2, 2] = np.sum(grad_z**2)
        J[0, 1] = np.sum(grad_x * grad_y)
        J[0, 2] = np.sum(grad_x * grad_z)
        J[1, 2] = np.sum(grad_y * grad_z)
        J[1, 0] = J[0, 1]
        J[2, 0] = J[0, 2]
        J[2, 1] = J[1, 2]
        
        # Eigenvalues indicate principal dielectric constants
        eigenvalues = np.linalg.eigvalsh(J)
        
        # Coherence measure (anisotropy)
        # (lambda_max - lambda_min) / (lambda_max + lambda_min)
        total = np.sum(eigenvalues)
        if total > 1e-10:
            anisotropy = (eigenvalues[-1] - eigenvalues[0]) / total
        else:
            anisotropy = 0.0
            
        return {
            'dielectric_tensor_eigenvalues': eigenvalues.tolist(),
            'anisotropy_ratio': float(anisotropy),
            'mean_permittivity': float(np.mean(permittivity)),
            'variance_permittivity': float(np.var(permittivity)),
            'is_isotropic': anisotropy < 0.1
        }


# =============================================================================
# MAXWELL SOLVER (SCATTERING AND POTENTIAL)
# =============================================================================

class MaxwellScatteringSolver:
    """
    Solves Maxwell's equations in the frequency domain via FFT.
    
    Key Analysis:
    1. Electrostatics: Solve ∇·(ε∇φ) = -ρ.
    2. Scattering: Calculate Fourier Transform of Dielectric contrast.
       - Crystal: Sharp Bragg Peaks at k-vectors determined by lattice periodicity.
       - Glass: Broad diffuse scattering (Rayleigh).
    """
    def __init__(self, config: MaxwellConfiguration):
        self.config = config

    def solve_poisson(self, charge_density: np.ndarray, permittivity: np.ndarray) -> np.ndarray:
        """
        Solves Poisson equation for Electric Potential φ.
        Using Spectral Method (FFT):
        ∇²φ = -ρ / ε_0
        
        For variable ε (heterogeneous medium), this is approximate.
        We use the convolution theorem for the Green's function.
        """
        N = self.config.GRID_DIMENSION
        
        # FFT of Charge Density
        rho_k = fftn(charge_density)
        
        # Laplacian Operator in Fourier Space: -k²
        kx = np.fft.fftfreq(N) * 2 * np.pi
        ky = np.fft.fftfreq(N) * 2 * np.pi
        kz = np.fft.fftfreq(N) * 2 * np.pi
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing='ij')
        
        k_squared = KX**2 + KY**2 + KZ**2
        k_squared[0, 0, 0] = 1.0  # Avoid division by zero
        
        # Phi(k) = Rho(k) / (ε₀ * k²)
        # Using average permittivity for linear solver
        eps_avg = np.mean(permittivity) * self.config.PERMITTIVITY_VACUUM
        phi_k = rho_k / (eps_avg * k_squared)
        
        # Inverse FFT to get Potential
        phi = np.real(np.fft.ifftn(phi_k))
        
        # Electric Field E = -∇φ
        E_x = -np.gradient(phi, axis=0)
        E_y = -np.gradient(phi, axis=1)
        E_z = -np.gradient(phi, axis=2)
        
        return phi

    def compute_scattering(self, permittivity: np.ndarray) -> Dict[str, Any]:
        """
        Compute the Scattering Amplitude S(k).
        S(k) ∝ |FT(Δε)|²
        
        Δε = ε(r) - ε_avg (Dielectric contrast)
        
        Returns:
            Dict containing scattering intensity map and peak analysis.
        """
        eps_avg = np.mean(permittivity)
        delta_eps = permittivity - eps_avg
        
        # Fourier Transform
        F_eps = fftshift(fftn(delta_eps))
        intensity = np.abs(F_eps)**2
        
        # Normalize
        intensity /= np.max(intensity) if np.max(intensity) > 0 else 1.0
        
        # Central Slice (kx-ky plane at kz=0) for visualization
        N = self.config.GRID_DIMENSION
        center = N // 2
        slice_2d = intensity[:, :, center]
        
        # Analyze Peaks
        peaks = self._find_peaks(intensity)
        
        return {
            'scattering_intensity_3d': intensity,
            'scattering_slice_xy': slice_2d,
            'fourier_coefficients': F_eps,
            'peak_positions': peaks['positions'],
            'peak_count': peaks['count'],
            'is_bragg_pattern': peaks['count'] > 5 and peaks['prominence'] > self.config.BRAGG_PEAK_PROMINENCE
        }

    def _find_peaks(self, intensity: np.ndarray) -> Dict[str, Any]:
        """
        Detect sharp peaks indicative of crystallinity.
        """
        # Simplified peak detection: Count high intensity points in FFT
        threshold = 0.5 * np.max(intensity)
        indices = np.where(intensity > threshold)
        
        # Cluster indices to count distinct peaks
        # Basic count of non-zero voxels above threshold
        count = len(indices[0])
        
        # Prominence estimate
        prominence = 0.0
        if count > 0:
            prominence = np.max(intensity) / np.mean(intensity[intensity > 0.01])
            
        return {
            'positions': list(zip(indices[0].tolist(), indices[1].tolist(), indices[2].tolist())),
            'count': count,
            'prominence': float(prominence)
        }


# =============================================================================
# PHOTONIC ENTROPY AND BANDGAP
# =============================================================================

class PhotonicEntropyCalculator:
    """
    Calculates the entropy of the electromagnetic field distribution.
    S = -∑ p(E) log p(E)
    
    Glass: High Entropy (Disordered field).
    Crystal: Low Entropy (Ordered, localized modes).
    """
    def __init__(self, config: MaxwellConfiguration):
        self.config = config

    def calculate(self, potential: np.ndarray, intensity: np.ndarray) -> Dict[str, float]:
        # Energy Distribution
        energy_density = potential**2
        hist, _ = np.histogram(energy_density.flatten(), bins=50, density=True)
        hist = hist[hist > 0]
        energy_entropy = entropy(hist)
        
        # Photon State Entropy (Density of States analogy)
        # Use the FFT intensity as proxy for density of modes
        dos = intensity.flatten()
        dos_norm = dos / np.sum(dos)
        dos_norm = dos_norm[dos_norm > 0]
        dos_entropy = entropy(dos_norm)
        
        return {
            'field_entropy': float(energy_entropy),
            'mode_entropy': float(dos_entropy),
            'total_photonic_entropy': float(energy_entropy + dos_entropy)
        }


class BandgapAnalyzer:
    """
    Estimates if a photonic bandgap exists.
    A bandgap implies certain frequencies cannot propagate.
    
    We use the Fourier coefficients to estimate the gap.
    """
    def __init__(self, config: MaxwellConfiguration):
        self.config = config

    def analyze(self, fourier_coeffs: np.ndarray) -> Dict[str, Any]:
        # Radial Average of Fourier Spectrum (1D Band Structure approximation)
        N = self.config.GRID_DIMENSION
        center = N // 2
        
        # Calculate distances from center
        z, y, x = np.ogrid[:N, :N, :N]
        r = np.sqrt((x-center)**2 + (y-center)**2 + (z-center)**2)
        r = r.astype(int)
        
        # Radial Profile
        intensity = np.abs(fourier_coeffs)**2
        radial_sum = np.bincount(r.ravel(), intensity.ravel())
        radial_count = np.bincount(r.ravel())
        radial_profile = radial_sum / radial_count
        
        # Gap detection: Find regions of low density in the profile
        # Skip the DC component (center)
        profile_smooth = radial_profile[1:len(radial_profile)//2]
        min_val = np.min(profile_smooth) if len(profile_smooth) > 0 else 1.0
        max_val = np.max(profile_smooth) if len(profile_smooth) > 0 else 1.0
        
        gap_ratio = min_val / (max_val + 1e-10)
        has_gap = gap_ratio < 0.01 # Arbitrary threshold for demonstration
        
        return {
            'radial_profile': radial_profile.tolist(),
            'estimated_bandgap': bool(has_gap),
            'gap_depth': float(gap_ratio)
        }


# =============================================================================
# PHASE CLASSIFICATION
# =============================================================================

class CrystalPhaseClassifier:
    """
    Classifies the material phase (Crystal vs Glass) based on EM metrics.
    """
    def __init__(self, config: MaxwellConfiguration):
        self.config = config

    def classify(self, em_metrics: Dict[str, Any], purity_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """
        Decision Logic:
        1. If Purity Alpha > Threshold AND Discretization is high -> Pre-cursor.
        2. If Anisotropy is High -> Crystal Order.
        3. If Scattering shows Bragg Peaks -> Long Range Order.
        4. If Entropy is Low -> Ordered System.
        """
        is_crystal = False
        is_glass = False
        confidence = 0.0
        
        # Criteria
        high_alpha = purity_metrics['alpha'] > self.config.ALPHA_THRESHOLD_CRYSTAL
        high_anisotropy = em_metrics['anisotropy']['anisotropy_ratio'] > 0.5
        bragg_peaks = em_metrics['scattering']['is_bragg_pattern']
        low_entropy = em_metrics['entropy']['total_photonic_entropy'] < self.config.CRYSTALLINITY_ENTROPY_THRESHOLD
        
        # Scoring
        score = 0
        if high_alpha: score += 1
        if high_anisotropy: score += 1
        if bragg_peaks: score += 2
        if low_entropy: score += 1
        
        if score >= 3:
            is_crystal = True
            confidence = score / 5.0
            phase_name = "Strassen_Crystal"
        elif score >= 1:
            is_glass = True
            confidence = 1.0 - (score / 5.0)
            phase_name = "Amorphous_Glass"
        else:
            phase_name = "Disordered_Gas"
            
        return {
            'phase': phase_name,
            'is_crystal': bool(is_crystal),
            'is_glass': bool(is_glass),
            'confidence': float(confidence),
            'scoring': {
                'high_alpha': high_alpha,
                'high_anisotropy': high_anisotropy,
                'bragg_peaks': bragg_peaks,
                'low_entropy': low_entropy
            }
        }


# =============================================================================
# CHECKPOINT MANAGER
# =============================================================================

class CheckpointManager:
    def __init__(self, config: MaxwellConfiguration):
        self.config = config
        self.last_save_time = time.time()

    def should_save(self) -> bool:
        return (time.time() - self.last_save_time) / 60.0 >= self.config.CHECKPOINT_INTERVAL_MINUTES

    def save(self, data: Dict[str, Any], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'maxwell_analysis_checkpoint_latest.json')
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        self.last_save_time = time.time()


# =============================================================================
# VISUALIZATION
# =============================================================================

class MaxwellVisualizer:
    def __init__(self, config: MaxwellConfiguration):
        self.config = config

    def visualize_lattice(self, permittivity: np.ndarray, output_dir: str, name: str):
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        N = permittivity.shape[0]
        x, y, z = np.indices((N, N, N))
        
        # Plot voxels with high permittivity
        mask = permittivity > np.mean(permittivity) + np.std(permittivity)
        ax.voxels(mask, edgecolor='k', alpha=0.7)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Dielectric Lattice: {name}')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'dielectric_lattice_{name}.{self.config.SAVE_FORMAT}'), dpi=self.config.FIGURE_DPI)
        plt.close()

    def visualize_scattering(self, scattering_slice: np.ndarray, output_dir: str, name: str):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(scattering_slice, cmap='hot', interpolation='nearest')
        ax.set_title(f'EM Scattering (Bragg Pattern): {name}')
        ax.set_xlabel('kx')
        ax.set_ylabel('ky')
        plt.colorbar(ax.imshow(scattering_slice, cmap='hot'), ax=ax, label='Intensity')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'scattering_{name}.{self.config.SAVE_FORMAT}'), dpi=self.config.FIGURE_DPI)
        plt.close()

    def visualize_potential(self, potential: np.ndarray, output_dir: str, name: str):
        fig, ax = plt.subplots(figsize=(10, 8))
        N = potential.shape[0]
        center = N // 2
        slice_2d = potential[center, :, :]
        
        im = ax.imshow(slice_2d, cmap='viridis')
        ax.set_title(f'Electrostatic Potential (Slice): {name}')
        plt.colorbar(im, ax=ax, label='Potential φ')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'potential_{name}.{self.config.SAVE_FORMAT}'), dpi=self.config.FIGURE_DPI)
        plt.close()


# =============================================================================
# MAIN ANALYZER
# =============================================================================

class MaxwellAnalyzer:
    def __init__(self, config: MaxwellConfiguration):
        self.config = config
        self.migrator = CheckpointMigrator()
        self.mapper = StrassenGeometryMapper(config)
        self.dielectric_analyzer = DielectricTensorAnalyzer(config)
        self.maxwell_solver = MaxwellScatteringSolver(config)
        self.entropy_calc = PhotonicEntropyCalculator(config)
        self.bandgap_analyzer = BandgapAnalyzer(config)
        self.classifier = CrystalPhaseClassifier(config)
        self.visualizer = MaxwellVisualizer(config)
        self.checkpoint_manager = CheckpointManager(config)

    def _calculate_purity_metrics(self, weights: Dict[str, np.ndarray]) -> Dict[str, float]:
        flat = np.concatenate([w.flatten() for w in weights.values()])
        rounded = np.round(flat)
        delta = np.max(np.abs(flat - rounded))
        alpha = -np.log(delta + 1e-15)
        return {
            'delta': float(delta),
            'alpha': float(alpha),
            'discretized': bool(delta < self.config.DISCRETIZATION_MARGIN)
        }

    def analyze_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        """
        Executes the full Maxwellian analysis pipeline on a single checkpoint.
        """
        print(f"Processing: {checkpoint_path}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Load
        raw = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        weights = self.migrator.migrate(raw, self.config)
        if weights is None:
            raise ValueError(f"Migration failed for {checkpoint_path}")
        
        # 2. Geometry Mapping
        charge, permittivity = self.mapper.map_weights_to_lattice(weights)
        
        # 3. Dielectric Analysis
        dielectric_props = self.dielectric_analyzer.analyze_permittivity_tensor(permittivity)
        
        # 4. Maxwell Solver
        potential = self.maxwell_solver.solve_poisson(charge, permittivity)
        scattering = self.maxwell_solver.compute_scattering(permittivity)
        
        # 5. Thermodynamics
        entropy = self.entropy_calc.calculate(potential, scattering['scattering_intensity_3d'])
        bandgap = self.bandgap_analyzer.analyze(scattering['fourier_coefficients'])
        
        # 6. Purity (Required for Classification)
        purity = self._calculate_purity_metrics(weights)
        
        # 7. Classification
        classification = self.classifier.classify(
            {'anisotropy': dielectric_props, 'scattering': scattering, 'entropy': entropy},
            purity
        )
        
        results = {
            'checkpoint': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'purity_metrics': purity,
            'dielectric_properties': dielectric_props,
            'electromagnetic_results': {
                'max_potential': float(np.max(np.abs(potential))),
                'scattering_peaks': scattering['peak_count'],
                'is_bragg_pattern': scattering['is_bragg_pattern']
            },
            'thermodynamics': {
                'entropy': entropy,
                'bandgap': bandgap
            },
            'phase_classification': classification
        }
        
        # 8. Visualization
        name = Path(checkpoint_path).stem
        self.visualizer.visualize_scattering(scattering['scattering_slice_xy'], output_dir, name)
        self.visualizer.visualize_potential(potential, output_dir, name)
        
        # 9. Checkpoint
        if self.checkpoint_manager.should_save():
            self.checkpoint_manager.save({'latest_result': results}, output_dir)
            
        return results

    def generate_report(self, results: List[Dict], output_dir: str):
        path = os.path.join(output_dir, 'maxwell_analysis_summary.txt')
        with open(path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("MAXWELLIAN ANALYSIS REPORT: STRASSEN CRYSTALLIZATION\n")
            f.write("=" * 80 + "\n\n")
            
            for res in results:
                f.write(f"Checkpoint: {res['checkpoint']}\n")
                f.write(f"  Phase: {res['phase_classification']['phase']}\n")
                f.write(f"  Confidence: {res['phase_classification']['confidence']:.4f}\n")
                f.write(f"  Purity Alpha: {res['purity_metrics']['alpha']:.4f}\n")
                f.write(f"  Anisotropy Ratio: {res['dielectric_properties']['anisotropy_ratio']:.4f}\n")
                f.write(f"  Bragg Peaks: {res['electromagnetic_results']['scattering_peaks']}\n")
                f.write(f"  Photonic Entropy: {res['thermodynamics']['entropy']['total_photonic_entropy']:.4f}\n")
                f.write("-" * 80 + "\n")
        print(f"Report saved to {path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Maxwell Analysis for Strassen Crystals')
    parser.add_argument('path', nargs='?', default='checkpoints', help='Checkpoint file or directory')
    parser.add_argument('--output', default='maxwell_analysis_output', help='Output directory')
    parser.add_argument('--device', default='cpu', help='Device (cpu/cuda)')
    args = parser.parse_args()
    
    config = MaxwellConfiguration(DEVICE=args.device)
    analyzer = MaxwellAnalyzer(config)
    
    p = Path(args.path)
    results = []
    
    if p.is_file():
        res = analyzer.analyze_checkpoint(str(p), args.output)
        results.append(res)
    elif p.is_dir():
        files = sorted(list(p.glob('*.pt')) + list(p.glob('*.pth')))
        for i, f in enumerate(files):
            print(f"[{i+1}/{len(files)}]")
            try:
                res = analyzer.analyze_checkpoint(str(f), args.output)
                results.append(res)
            except Exception as e:
                print(f"Error processing {f}: {e}")
    else:
        print("Path not found.")
        return
        
    if results:
        analyzer.generate_report(results, args.output)
        with open(os.path.join(args.output, 'full_results.json'), 'w') as f:
            json.dump(results, f, indent=2, default=str)

if __name__ == "__main__":
    main()
