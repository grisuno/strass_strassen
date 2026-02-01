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
from scipy.linalg import eigh
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SchrodingerConfig:
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    BATCH_SIZE: int = 32
    
    HBAR: float = 1.0
    MASS_EFFECTIVE: float = 1.0
    POTENTIAL_WELL_DEPTH: float = 10.0
    
    EIGENVALUE_COUNT: int = 10
    SPATIAL_GRID_POINTS: int = 100
    TIME_STEPS: int = 1000
    TIME_MAX: float = 10.0
    
    ENERGY_TOLERANCE: float = 1e-6
    WAVE_FUNCTION_NORM_TOLERANCE: float = 1e-8
    
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'
    COLORMAP: str = 'viridis'
    
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@runtime_checkable
class IModel(Protocol):
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...


@runtime_checkable
class IWaveFunctionExtractor(Protocol):
    def extract(self, model: IModel) -> torch.Tensor: ...


@runtime_checkable
class IPotentialCalculator(Protocol):
    def calculate(self, weights: torch.Tensor) -> torch.Tensor: ...


@runtime_checkable
class IHamiltonianConstructor(Protocol):
    def construct(self, potential: torch.Tensor, mass: float) -> torch.Tensor: ...


@runtime_checkable
class IEigenvalueSolver(Protocol):
    def solve(self, hamiltonian: torch.Tensor, count: int) -> Tuple[np.ndarray, np.ndarray]: ...


@runtime_checkable
class ITimeEvolver(Protocol):
    def evolve(self, initial_state: torch.Tensor, hamiltonian: torch.Tensor, time_steps: int, dt: float) -> List[torch.Tensor]: ...


@runtime_checkable
class IExpectationValueCalculator(Protocol):
    def calculate(self, wave_function: torch.Tensor, operator: torch.Tensor) -> float: ...


@runtime_checkable
class IUncertaintyCalculator(Protocol):
    def calculate(self, wave_function: torch.Tensor, position_grid: torch.Tensor) -> Dict[str, float]: ...


@runtime_checkable
class ICheckpointLoader(Protocol):
    def load(self, path: str, device: str) -> Any: ...


@runtime_checkable
class ICheckpointMigrator(Protocol):
    def migrate(self, raw_data: Any) -> Optional[Dict[str, torch.Tensor]]: ...


class BilinearModel(nn.Module):
    def __init__(self, hidden_dim: int = SchrodingerConfig.HIDDEN_DIM, matrix_size: int = SchrodingerConfig.MATRIX_SIZE):
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


class WaveFunctionExtractor:
    def extract(self, model: IModel) -> torch.Tensor:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        return all_weights


class PotentialCalculator:
    def __init__(self, config: SchrodingerConfig = SchrodingerConfig()):
        self.config = config
    
    def calculate(self, weights: torch.Tensor) -> torch.Tensor:
        rounded = torch.round(weights)
        distances = torch.abs(weights - rounded)
        
        harmonic_potential = 0.5 * self.config.MASS_EFFECTIVE * (distances ** 2)
        
        well_mask = distances > 0.5
        barrier_height = torch.where(
            well_mask,
            self.config.POTENTIAL_WELL_DEPTH * (distances - 0.5) ** 2,
            torch.zeros_like(distances)
        )
        
        total_potential = harmonic_potential + barrier_height
        
        return total_potential


class HamiltonianConstructor:
    def __init__(self, config: SchrodingerConfig = SchrodingerConfig()):
        self.config = config
    
    def construct(self, potential: torch.Tensor, mass: float) -> torch.Tensor:
        n = len(potential)
        dx = 1.0 / n
        
        kinetic_diagonal = torch.ones(n) * (self.config.HBAR ** 2) / (mass * dx ** 2)
        kinetic_off_diagonal = torch.ones(n - 1) * (-0.5 * self.config.HBAR ** 2) / (mass * dx ** 2)
        
        kinetic = torch.diag(kinetic_diagonal) + torch.diag(kinetic_off_diagonal, 1) + torch.diag(kinetic_off_diagonal, -1)
        
        potential_matrix = torch.diag(potential)
        
        hamiltonian = kinetic + potential_matrix
        
        return hamiltonian


class EigenvalueSolver:
    def __init__(self, config: SchrodingerConfig = SchrodingerConfig()):
        self.config = config
    
    def solve(self, hamiltonian: torch.Tensor, count: int) -> Tuple[np.ndarray, np.ndarray]:
        h_np = hamiltonian.cpu().numpy()
        
        eigenvalues, eigenvectors = eigh(h_np)
        
        idx = np.argsort(eigenvalues)
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        eigenvalues = eigenvalues[:count]
        eigenvectors = eigenvectors[:, :count]
        
        for i in range(eigenvectors.shape[1]):
            norm = np.sqrt(np.sum(np.abs(eigenvectors[:, i]) ** 2))
            if norm > 0:
                eigenvectors[:, i] /= norm
        
        return eigenvalues, eigenvectors


class TimeEvolver:
    def __init__(self, config: SchrodingerConfig = SchrodingerConfig()):
        self.config = config
    
    def evolve(self, initial_state: torch.Tensor, hamiltonian: torch.Tensor, time_steps: int, dt: float) -> List[torch.Tensor]:
        states = [initial_state]
        current_state = initial_state.clone()
        
        h_np = hamiltonian.cpu().numpy()
        eigenvalues, eigenvectors = eigh(h_np)
        
        coefficients = np.dot(eigenvectors.T, current_state.cpu().numpy())
        
        for step in range(time_steps):
            t = (step + 1) * dt
            evolved_coefficients = coefficients * np.exp(-1j * eigenvalues * t / self.config.HBAR)
            evolved_state = np.dot(eigenvectors, evolved_coefficients)
            states.append(torch.tensor(evolved_state, dtype=torch.complex64, device=initial_state.device))
        
        return states


class ExpectationValueCalculator:
    def calculate(self, wave_function: torch.Tensor, operator: torch.Tensor) -> float:
        if wave_function.dtype.is_complex and not operator.dtype.is_complex:
            operator = operator.to(wave_function.dtype)
        
        wf_conj = torch.conj(wave_function)
        expectation = torch.dot(wf_conj, torch.matmul(operator, wave_function))
        return torch.real(expectation).item()


class UncertaintyCalculator:
    def __init__(self, config: SchrodingerConfig = SchrodingerConfig()):
        self.config = config
    
    def calculate(self, wave_function: torch.Tensor, position_grid: torch.Tensor) -> Dict[str, float]:
        wf_np = wave_function.cpu().numpy()
        positions = position_grid.cpu().numpy()
        
        probability = np.abs(wf_np) ** 2
        total_prob = np.sum(probability)
        if total_prob > 0:
            probability /= total_prob
        
        position_expectation = np.sum(positions * probability)
        position_squared_expectation = np.sum((positions ** 2) * probability)
        position_variance = position_squared_expectation - position_expectation ** 2
        position_uncertainty = np.sqrt(max(position_variance, 0))
        
        dx = positions[1] - positions[0] if len(positions) > 1 else 1.0
        gradient = np.gradient(wf_np, dx)
        momentum_expectation = np.real(np.vdot(wf_np, -1j * self.config.HBAR * gradient))
        momentum_squared_expectation = np.real(np.vdot(gradient, gradient)) * (self.config.HBAR ** 2)
        momentum_variance = momentum_squared_expectation - momentum_expectation ** 2
        momentum_uncertainty = np.sqrt(max(momentum_variance, 0))
        
        heisenberg_product = position_uncertainty * momentum_uncertainty
        
        return {
            'position_expectation': float(position_expectation),
            'position_uncertainty': float(position_uncertainty),
            'momentum_expectation': float(momentum_expectation),
            'momentum_uncertainty': float(momentum_uncertainty),
            'heisenberg_product': float(heisenberg_product),
            'heisenberg_satisfied': heisenberg_product >= (0.5 * self.config.HBAR)
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
            u_padded = torch.zeros(SchrodingerConfig.HIDDEN_DIM, 4, device=SchrodingerConfig.DEVICE)
            v_padded = torch.zeros(SchrodingerConfig.HIDDEN_DIM, 4, device=SchrodingerConfig.DEVICE)
            w_padded = torch.zeros(4, SchrodingerConfig.HIDDEN_DIM, device=SchrodingerConfig.DEVICE)
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


class SchrodingerAnalyzer:
    def __init__(self, checkpoint_path: str, config: SchrodingerConfig = SchrodingerConfig()):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        self.wave_extractor = WaveFunctionExtractor()
        self.potential_calculator = PotentialCalculator(config)
        self.hamiltonian_constructor = HamiltonianConstructor(config)
        self.eigenvalue_solver = EigenvalueSolver(config)
        self.time_evolver = TimeEvolver(config)
        self.expectation_calculator = ExpectationValueCalculator()
        self.uncertainty_calculator = UncertaintyCalculator(config)
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        loader = CheckpointLoader()
        migrator = CheckpointMigrator()
        
        try:
            raw_data = loader.load(self.checkpoint_path, self.config.DEVICE)
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        self.model = BilinearModel(
            hidden_dim=self.config.HIDDEN_DIM,
            matrix_size=self.config.MATRIX_SIZE
        ).to(self.config.DEVICE)
        
        migrated_state = migrator.migrate(raw_data)
        if migrated_state is not None:
            self.model.load_state_dict(migrated_state)
        else:
            raise RuntimeError(f"Failed to migrate checkpoint: {self.checkpoint_path}")
        
        self.epoch = raw_data.get('epoch', 'unknown') if isinstance(raw_data, dict) else 'unknown'
    
    def analyze(self) -> Dict[str, Any]:
        wave_function = self.wave_extractor.extract(self.model)
        
        potential = self.potential_calculator.calculate(wave_function)
        
        hamiltonian = self.hamiltonian_constructor.construct(potential, self.config.MASS_EFFECTIVE)
        
        eigenvalues, eigenvectors = self.eigenvalue_solver.solve(
            hamiltonian, 
            self.config.EIGENVALUE_COUNT
        )
        
        ground_state = torch.tensor(eigenvectors[:, 0], dtype=torch.complex64, device=self.config.DEVICE)
        
        dt = self.config.TIME_MAX / self.config.TIME_STEPS
        time_evolution = self.time_evolver.evolve(
            ground_state, 
            hamiltonian, 
            self.config.TIME_STEPS, 
            dt
        )
        
        position_grid = torch.linspace(0, 1, len(wave_function), device=self.config.DEVICE)
        
        position_operator = torch.diag(position_grid).to(ground_state.dtype)
        position_expectation = self.expectation_calculator.calculate(ground_state, position_operator)
        
        uncertainty = self.uncertainty_calculator.calculate(ground_state, position_grid)
        
        energy_gap = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0.0
        
        tunneling_probability = self._calculate_tunneling_probability(
            potential.cpu().numpy(), 
            eigenvectors[:, 0]
        )
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat()
            },
            'wave_function': {
                'dimension': len(wave_function),
                'norm': torch.norm(wave_function).item(),
                'mean': torch.mean(torch.abs(wave_function)).item(),
                'std': torch.std(torch.abs(wave_function)).item()
            },
            'potential': {
                'mean': torch.mean(potential).item(),
                'max': torch.max(potential).item(),
                'min': torch.min(potential).item(),
                'barrier_height': self.config.POTENTIAL_WELL_DEPTH
            },
            'hamiltonian': {
                'dimension': hamiltonian.shape[0],
                'trace': torch.trace(hamiltonian).item(),
                'norm': torch.norm(hamiltonian).item()
            },
            'eigenvalues': {
                'ground_state': float(eigenvalues[0]),
                'first_excited': float(eigenvalues[1]) if len(eigenvalues) > 1 else None,
                'energy_gap': float(energy_gap),
                'all_eigenvalues': eigenvalues.tolist()
            },
            'eigenvectors': {
                'ground_state_shape': eigenvectors[:, 0].shape,
                'ground_state_norm': float(np.linalg.norm(eigenvectors[:, 0]))
            },
            'time_evolution': {
                'time_steps': len(time_evolution),
                'dt': dt,
                'final_state_norm': torch.norm(time_evolution[-1]).item() if time_evolution else 0.0
            },
            'expectation_values': {
                'position': position_expectation,
                'ground_state_energy': float(eigenvalues[0])
            },
            'uncertainty': uncertainty,
            'tunneling': {
                'probability': tunneling_probability,
                'barrier_penetration': tunneling_probability > 0.01
            },
            'quantum_numbers': {
                'principal_quantum_number': 0,
                'energy_level': float(eigenvalues[0]),
                'degeneracy': self._count_degeneracy(eigenvalues)
            }
        }
        
        self._print_report(results)
        
        return results
    
    def _calculate_tunneling_probability(self, potential: np.ndarray, wave_function: np.ndarray) -> float:
        barrier_region = potential > np.mean(potential)
        if not np.any(barrier_region):
            return 0.0
        
        probability_density = np.abs(wave_function) ** 2
        total_prob = np.sum(probability_density)
        if total_prob < 1e-10:
            return 0.0
        
        tunneling_prob = np.sum(probability_density[barrier_region]) / total_prob
        
        return float(tunneling_prob)
    
    def _count_degeneracy(self, eigenvalues: np.ndarray) -> int:
        if len(eigenvalues) < 2:
            return 1
        
        degeneracy = 1
        for i in range(1, len(eigenvalues)):
            if np.abs(eigenvalues[i] - eigenvalues[0]) < self.config.ENERGY_TOLERANCE:
                degeneracy += 1
            else:
                break
        
        return degeneracy
    
    def _print_report(self, results: Dict):
        print("=" * 80)
        print("SCHRODINGER ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {results['metadata']['checkpoint_path']}")
        print(f"  Epoch: {results['metadata']['epoch']}")
        
        print(f"\n[WAVE FUNCTION]")
        wf = results['wave_function']
        print(f"  Dimension: {wf['dimension']}")
        print(f"  Norm: {wf['norm']:.6f}")
        print(f"  Mean amplitude: {wf['mean']:.6f}")
        print(f"  Std amplitude: {wf['std']:.6f}")
        
        print(f"\n[POTENTIAL]")
        pot = results['potential']
        print(f"  Mean: {pot['mean']:.6f}")
        print(f"  Max: {pot['max']:.6f}")
        print(f"  Min: {pot['min']:.6f}")
        print(f"  Barrier height: {pot['barrier_height']:.6f}")
        
        print(f"\n[HAMILTONIAN]")
        ham = results['hamiltonian']
        print(f"  Dimension: {ham['dimension']}x{ham['dimension']}")
        print(f"  Trace: {ham['trace']:.6f}")
        print(f"  Norm: {ham['norm']:.6f}")
        
        print(f"\n[EIGENVALUES]")
        ev = results['eigenvalues']
        print(f"  Ground state: {ev['ground_state']:.6f}")
        if ev['first_excited'] is not None:
            print(f"  First excited: {ev['first_excited']:.6f}")
            print(f"  Energy gap: {ev['energy_gap']:.6e}")
        
        print(f"\n[EXPECTATION VALUES]")
        exp = results['expectation_values']
        print(f"  Position: {exp['position']:.6f}")
        print(f"  Energy: {exp['ground_state_energy']:.6f}")
        
        print(f"\n[UNCERTAINTY PRINCIPLE]")
        unc = results['uncertainty']
        print(f"  Position expectation: {unc['position_expectation']:.6f}")
        print(f"  Position uncertainty: {unc['position_uncertainty']:.6e}")
        print(f"  Momentum expectation: {unc['momentum_expectation']:.6e}")
        print(f"  Momentum uncertainty: {unc['momentum_uncertainty']:.6e}")
        print(f"  Heisenberg product: {unc['heisenberg_product']:.6e}")
        print(f"  Principle satisfied: {unc['heisenberg_satisfied']}")
        
        print(f"\n[TUNNELING]")
        tun = results['tunneling']
        print(f"  Probability: {tun['probability']:.6e}")
        print(f"  Barrier penetration: {tun['barrier_penetration']}")
        
        print(f"\n[QUANTUM NUMBERS]")
        qn = results['quantum_numbers']
        print(f"  Principal quantum number: {qn['principal_quantum_number']}")
        print(f"  Energy level: {qn['energy_level']:.6f}")
        print(f"  Degeneracy: {qn['degeneracy']}")
        
        print("=" * 80)
            

class WaveFunctionVisualizer:
    def __init__(self, config: SchrodingerConfig = SchrodingerConfig()):
        self.config = config
    
    def visualize(self, data: Dict[str, Any], output_path: str) -> None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 12), dpi=self.config.FIGURE_DPI)
        
        eigenvalues = np.array(data['eigenvalues']['all_eigenvalues'])
        eigenvectors = data.get('eigenvectors_data', None)
        
        ax1 = axes[0, 0]
        ax1.bar(range(len(eigenvalues)), eigenvalues, color='#2E86AB', alpha=0.7)
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=1)
        ax1.set_xlabel('Quantum number n', fontsize=12)
        ax1.set_ylabel('Energy E_n', fontsize=12)
        ax1.set_title('Energy Spectrum', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3, linestyle=':')
        
        ax2 = axes[0, 1]
        if eigenvectors is not None:
            x = np.linspace(0, 1, eigenvectors.shape[0])
            for i in range(min(3, eigenvectors.shape[1])):
                psi = eigenvectors[:, i]
                probability = np.abs(psi) ** 2
                ax2.plot(x, probability, linewidth=2, label=f'n={i}, E={eigenvalues[i]:.3f}')
            ax2.set_xlabel('Position x', fontsize=12)
            ax2.set_ylabel('|ψ(x)|²', fontsize=12)
            ax2.set_title('Probability Densities', fontsize=14, fontweight='bold')
            ax2.legend(loc='best', fontsize=10)
            ax2.grid(True, alpha=0.3, linestyle=':')
        
        ax3 = axes[1, 0]
        potential = data.get('potential_data', None)
        if potential is not None:
            x = np.linspace(0, 1, len(potential))
            ax3.fill_between(x, potential, alpha=0.3, color='#D62828')
            ax3.plot(x, potential, color='#D62828', linewidth=2, label='V(x)')
            for i in range(min(3, len(eigenvalues))):
                ax3.axhline(y=eigenvalues[i], color='#06A77D', linestyle='--', alpha=0.5, label=f'E_{i}' if i == 0 else '')
            ax3.set_xlabel('Position x', fontsize=12)
            ax3.set_ylabel('Potential V(x) / Energy E', fontsize=12)
            ax3.set_title('Potential Well & Energy Levels', fontsize=14, fontweight='bold')
            ax3.legend(loc='best', fontsize=10)
            ax3.grid(True, alpha=0.3, linestyle=':')
        
        ax4 = axes[1, 1]
        uncertainty_data = data.get('uncertainty', {})
        metrics = ['Position Δx', 'Momentum Δp', 'Product ΔxΔp']
        values = [
            uncertainty_data.get('position_uncertainty', 0),
            uncertainty_data.get('momentum_uncertainty', 0),
            uncertainty_data.get('heisenberg_product', 0)
        ]
        colors = ['#2E86AB', '#A23B72', '#F18F01']
        bars = ax4.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
        ax4.axhline(y=0.5 * self.config.HBAR, color='red', linestyle='--', linewidth=2, label='ℏ/2 limit')
        ax4.set_ylabel('Uncertainty', fontsize=12)
        ax4.set_title('Heisenberg Uncertainty', fontsize=14, fontweight='bold')
        ax4.legend(loc='best', fontsize=10)
        ax4.grid(True, alpha=0.3, axis='y')
        
        plt.suptitle('Schrödinger Analysis of Neural Network', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(output_path, dpi=self.config.FIGURE_DPI, format=self.config.SAVE_FORMAT)
        plt.close()


class SchrodingerPipeline:
    def __init__(self, config: SchrodingerConfig = SchrodingerConfig()):
        self.config = config
        self.visualizer = WaveFunctionVisualizer(config)
    
    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer = SchrodingerAnalyzer(checkpoint_path, self.config)
        results = analyzer.analyze()
        
        base_name = Path(checkpoint_path).stem
        
        results_path = os.path.join(output_dir, f'{base_name}_schrodinger.json')
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
        
        energy_gaps = []
        heisenberg_products = []
        tunneling_probs = []
        ground_states = []
        
        for r in all_results:
            ev = r.get('eigenvalues', {})
            if ev.get('energy_gap') is not None:
                energy_gaps.append(ev['energy_gap'])
            
            unc = r.get('uncertainty', {})
            if unc.get('heisenberg_product') is not None:
                heisenberg_products.append(unc['heisenberg_product'])
            
            tun = r.get('tunneling', {})
            if tun.get('probability') is not None:
                tunneling_probs.append(tun['probability'])
            
            if ev.get('ground_state') is not None:
                ground_states.append(ev['ground_state'])
        
        summary = {
            'total_checkpoints_analyzed': len(all_results),
            'timestamp': datetime.now().isoformat(),
            'aggregate_statistics': {
                'energy_gap': {
                    'mean': float(np.mean(energy_gaps)) if energy_gaps else 0,
                    'std': float(np.std(energy_gaps)) if energy_gaps else 0,
                    'min': float(np.min(energy_gaps)) if energy_gaps else 0,
                    'max': float(np.max(energy_gaps)) if energy_gaps else 0
                },
                'heisenberg_product': {
                    'mean': float(np.mean(heisenberg_products)) if heisenberg_products else 0,
                    'std': float(np.std(heisenberg_products)) if heisenberg_products else 0,
                    'min': float(np.min(heisenberg_products)) if heisenberg_products else 0,
                    'max': float(np.max(heisenberg_products)) if heisenberg_products else 0,
                    'principle_satisfied_rate': sum(1 for h in heisenberg_products if h >= 0.5 * self.config.HBAR) / len(heisenberg_products) if heisenberg_products else 0
                },
                'tunneling': {
                    'mean_probability': float(np.mean(tunneling_probs)) if tunneling_probs else 0,
                    'penetration_rate': sum(1 for t in tunneling_probs if t > 0.01) / len(tunneling_probs) if tunneling_probs else 0
                },
                'ground_state_energy': {
                    'mean': float(np.mean(ground_states)) if ground_states else 0,
                    'std': float(np.std(ground_states)) if ground_states else 0
                }
            },
            'individual_results': all_results
        }
        
        summary_path = os.path.join(output_dir, 'schrodinger_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._generate_text_report(summary, output_dir)
        
        print(f"\nSaved summary: {summary_path}")
    
    def _generate_text_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        report_path = os.path.join(output_dir, 'schrodinger_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("SCHRODINGER ANALYSIS OF NEURAL NETWORKS\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total checkpoints analyzed: {summary['total_checkpoints_analyzed']}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            stats = summary['aggregate_statistics']
            
            f.write("-" * 80 + "\n")
            f.write("QUANTUM STATISTICS\n")
            f.write("-" * 80 + "\n")
            
            eg = stats['energy_gap']
            f.write(f"\n[ENERGY GAP]\n")
            f.write(f"  Mean: {eg['mean']:.6e}\n")
            f.write(f"  Std:  {eg['std']:.6e}\n")
            f.write(f"  Min:  {eg['min']:.6e}\n")
            f.write(f"  Max:  {eg['max']:.6e}\n")
            
            hp = stats['heisenberg_product']
            f.write(f"\n[HEISENBERG UNCERTAINTY]\n")
            f.write(f"  Mean: {hp['mean']:.6e}\n")
            f.write(f"  Std:  {hp['std']:.6e}\n")
            f.write(f"  Min:  {hp['min']:.6e}\n")
            f.write(f"  Max:  {hp['max']:.6e}\n")
            f.write(f"  Principle satisfied rate: {hp['principle_satisfied_rate']:.2%}\n")
            
            tun = stats['tunneling']
            f.write(f"\n[TUNNELING]\n")
            f.write(f"  Mean probability: {tun['mean_probability']:.6e}\n")
            f.write(f"  Penetration rate: {tun['penetration_rate']:.2%}\n")
            
            gs = stats['ground_state_energy']
            f.write(f"\n[GROUND STATE ENERGY]\n")
            f.write(f"  Mean: {gs['mean']:.6f}\n")
            f.write(f"  Std:  {gs['std']:.6f}\n")
            
            f.write("\n" + "=" * 80 + "\n")
            f.write("INDIVIDUAL CHECKPOINT ANALYSIS\n")
            f.write("=" * 80 + "\n\n")
            
            for i, r in enumerate(summary['individual_results'], 1):
                f.write(f"[{i}] {r['metadata']['checkpoint_path']}\n")
                f.write(f"    Epoch: {r['metadata']['epoch']}\n")
                f.write(f"    Ground state energy: {r['eigenvalues']['ground_state']:.6f}\n")
                f.write(f"    Energy gap: {r['eigenvalues'].get('energy_gap', 'N/A')}\n")
                f.write(f"    Heisenberg product: {r['uncertainty']['heisenberg_product']:.6e}\n")
                f.write(f"    Tunneling probability: {r['tunneling']['probability']:.6e}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved text report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Schrödinger equation analysis for neural network checkpoints'
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
        default='schrodinger_analysis',
        help='Output directory'
    )
    
    args = parser.parse_args()
    
    config = SchrodingerConfig()
    pipeline = SchrodingerPipeline(config)
    
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