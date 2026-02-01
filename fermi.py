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
class FermiConfig:
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    
    NUM_ELECTRONS: int = 7
    NUM_BANDS: int = 10
    
    LATTICE_CONSTANT: float = 1.0
    HBAR: float = 1.054571817e-34
    ELECTRON_MASS: float = 9.10938356e-31
    EV_CONVERSION: float = 6.242e+18
    
    FERMI_TOLERANCE: float = 1e-6
    MAX_OCCUPATION: int = 2
    
    TEMPERATURE_ZERO: float = 0.0
    TEMPERATURE_LOW: float = 0.001
    TEMPERATURE_ROOM: float = 0.025
    
    K_POINTS: int = 20
    K_MIN: float = -np.pi
    K_MAX: float = np.pi
    
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'
    
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@runtime_checkable
class IModel(Protocol):
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...


@runtime_checkable
class IBlochWaveConstructor(Protocol):
    def construct(self, weights: torch.Tensor, k: float) -> np.ndarray: ...


@runtime_checkable
class IBandStructureCalculator(Protocol):
    def calculate(self, model: IModel) -> Dict[str, Any]: ...


@runtime_checkable
class IFermiLevelCalculator(Protocol):
    def calculate(self, eigenvalues: np.ndarray, num_electrons: int) -> Dict[str, float]: ...


@runtime_checkable
class IDensityOfStatesCalculator(Protocol):
    def calculate(self, eigenvalues: np.ndarray, energies: np.ndarray) -> np.ndarray: ...


@runtime_checkable
class IElectronicPropertiesCalculator(Protocol):
    def calculate(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray, fermi_level: float) -> Dict[str, Any]: ...


@runtime_checkable
class IMetalInsulatorClassifier(Protocol):
    def classify(self, band_gap: float, dos_at_fermi: float) -> str: ...


class BilinearModel(nn.Module):
    def __init__(self, hidden_dim: int = FermiConfig.HIDDEN_DIM, matrix_size: int = FermiConfig.MATRIX_SIZE):
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


class BlochWaveConstructor:
    def __init__(self, config: FermiConfig = FermiConfig()):
        self.config = config
    
    def construct(self, weights: torch.Tensor, k: float) -> np.ndarray:
        n = weights.numel()
        hamiltonian = np.zeros((n, n), dtype=complex)
        
        effective_mass = self.config.ELECTRON_MASS * (1 + torch.std(weights).item())
        
        kinetic = (self.config.HBAR ** 2) * (k ** 2) / (2 * effective_mass)
        hamiltonian[np.arange(n), np.arange(n)] = kinetic
        
        potential = weights.cpu().numpy().flatten()
        for i in range(n):
            for j in range(n):
                if i != j:
                    coupling = potential[i] * potential[j] / (abs(i - j) + 1)
                    phase = np.exp(1j * k * (i - j) * self.config.LATTICE_CONSTANT)
                    hamiltonian[i, j] += coupling * phase
        
        return hamiltonian


class BandStructureCalculator:
    def __init__(self, config: FermiConfig = FermiConfig()):
        self.config = config
        self.bloch_constructor = BlochWaveConstructor(config)
    
    def calculate(self, model: IModel) -> Dict[str, Any]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        k_points = np.linspace(self.config.K_MIN, self.config.K_MAX, self.config.K_POINTS)
        
        band_structure = np.zeros((self.config.K_POINTS, self.config.NUM_BANDS))
        
        for i, k in enumerate(k_points):
            hamiltonian = self.bloch_constructor.construct(all_weights, k)
            
            eigenvalues = np.linalg.eigvalsh(hamiltonian[:self.config.NUM_BANDS, :self.config.NUM_BANDS].real)
            eigenvalues = np.sort(eigenvalues)
            
            band_structure[i, :] = eigenvalues[:self.config.NUM_BANDS]
        
        band_gap, valence_idx, conduction_idx = self._calculate_band_gap(band_structure)
        
        effective_masses = self._calculate_effective_masses(k_points, band_structure, valence_idx, conduction_idx)
        
        return {
            'k_points': k_points.tolist(),
            'band_structure': band_structure.tolist(),
            'num_bands': self.config.NUM_BANDS,
            'band_gap': float(band_gap),
            'valence_band_index': valence_idx,
            'conduction_band_index': conduction_idx,
            'effective_masses': effective_masses,
            'is_direct_gap': self._is_direct_gap(band_structure, valence_idx, conduction_idx)
        }
    
    def _calculate_band_gap(self, band_structure: np.ndarray) -> Tuple[float, int, int]:
        num_electrons = self.config.NUM_ELECTRONS
        filled_bands = num_electrons // self.config.MAX_OCCUPATION
        
        valence_idx = filled_bands - 1
        conduction_idx = filled_bands
        
        if conduction_idx >= band_structure.shape[1]:
            return 0.0, valence_idx, valence_idx
        
        valence_max = np.max(band_structure[:, valence_idx])
        conduction_min = np.min(band_structure[:, conduction_idx])
        
        band_gap = conduction_min - valence_max
        
        return float(band_gap), valence_idx, conduction_idx
    
    def _calculate_effective_masses(self, k_points: np.ndarray, band_structure: np.ndarray, 
                                    valence_idx: int, conduction_idx: int) -> Dict[str, float]:
        dk = k_points[1] - k_points[0]
        
        if len(k_points) < 3:
            return {'electron': float('inf'), 'hole': float('inf')}
        
        conduction_curvature = np.gradient(np.gradient(band_structure[:, conduction_idx], dk), dk)
        valence_curvature = np.gradient(np.gradient(band_structure[:, valence_idx], dk), dk)
        
        center_idx = len(k_points) // 2
        
        m_electron = (self.config.HBAR ** 2) / (conduction_curvature[center_idx] + 1e-20)
        m_hole = -(self.config.HBAR ** 2) / (valence_curvature[center_idx] - 1e-20)
        
        return {
            'electron': float(m_electron * self.config.EV_CONVERSION),
            'hole': float(abs(m_hole) * self.config.EV_CONVERSION)
        }
    
    def _is_direct_gap(self, band_structure: np.ndarray, valence_idx: int, conduction_idx: int) -> bool:
        valence_max_idx = np.argmax(band_structure[:, valence_idx])
        conduction_min_idx = np.argmin(band_structure[:, conduction_idx])
        
        return bool(valence_max_idx == conduction_min_idx)


class FermiLevelCalculator:
    def __init__(self, config: FermiConfig = FermiConfig()):
        self.config = config
    
    def calculate(self, eigenvalues: np.ndarray, num_electrons: int) -> Dict[str, float]:
        sorted_eigenvalues = np.sort(eigenvalues.flatten())
        
        num_states = len(sorted_eigenvalues)
        
        fermi_idx = min(num_electrons - 1, num_states - 1)
        fermi_energy = sorted_eigenvalues[fermi_idx] if fermi_idx >= 0 else sorted_eigenvalues[0]
        
        chemical_potential = self._calculate_chemical_potential(sorted_eigenvalues, num_electrons)
        
        work_function = sorted_eigenvalues[-1] - fermi_energy if len(sorted_eigenvalues) > 0 else 0.0
        
        return {
            'fermi_energy': float(fermi_energy),
            'fermi_level_ev': float(fermi_energy * self.config.EV_CONVERSION),
            'chemical_potential': float(chemical_potential),
            'work_function': float(work_function * self.config.EV_CONVERSION),
            'occupation_at_fermi': self._fermi_dirac(fermi_energy, fermi_energy, self.config.TEMPERATURE_ROOM),
            'num_states_below_fermi': int(np.sum(sorted_eigenvalues <= fermi_energy))
        }
    
    def _calculate_chemical_potential(self, eigenvalues: np.ndarray, num_electrons: int) -> float:
        temperatures = [self.config.TEMPERATURE_ZERO, self.config.TEMPERATURE_LOW, self.config.TEMPERATURE_ROOM]
        
        chemical_potentials = []
        for T in temperatures:
            if T == 0:
                mu = eigenvalues[min(num_electrons - 1, len(eigenvalues) - 1)] if num_electrons > 0 else eigenvalues[0]
            else:
                mu = self._find_chemical_potential_iterative(eigenvalues, num_electrons, T)
            chemical_potentials.append(mu)
        
        return float(np.mean(chemical_potentials))
    
    def _find_chemical_potential_iterative(self, eigenvalues: np.ndarray, num_electrons: int, 
                                           temperature: float, max_iter: int = 100) -> float:
        mu_min = np.min(eigenvalues) - 1.0
        mu_max = np.max(eigenvalues) + 1.0
        
        for _ in range(max_iter):
            mu = (mu_min + mu_max) / 2.0
            
            total_occupation = np.sum(self._fermi_dirac(eigenvalues, mu, temperature))
            
            if abs(total_occupation - num_electrons) < self.config.FERMI_TOLERANCE:
                return mu
            
            if total_occupation < num_electrons:
                mu_min = mu
            else:
                mu_max = mu
        
        return (mu_min + mu_max) / 2.0
    
    def _fermi_dirac(self, energy: np.ndarray, mu: float, temperature: float) -> np.ndarray:
        if temperature == 0:
            return (energy <= mu).astype(float)
        
        kT = temperature
        exponent = (energy - mu) / kT
        
        exponent = np.clip(exponent, -500, 500)
        
        return 1.0 / (np.exp(exponent) + 1.0)


class DensityOfStatesCalculator:
    def __init__(self, config: FermiConfig = FermiConfig()):
        self.config = config
    
    def calculate(self, eigenvalues: np.ndarray, energies: np.ndarray) -> np.ndarray:
        dos = np.zeros_like(energies)
        
        sigma = (np.max(energies) - np.min(energies)) / len(energies) if len(energies) > 1 else 0.1
        
        for i, E in enumerate(energies):
            dos[i] = np.sum(self._gaussian(eigenvalues, E, sigma))
        
        if np.sum(dos) > 0:
            dos /= np.sum(dos)
        
        return dos
    
    def _gaussian(self, x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
        return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


class ElectronicPropertiesCalculator:
    def __init__(self, config: FermiConfig = FermiConfig()):
        self.config = config
    
    def calculate(self, eigenvalues: np.ndarray, eigenvectors: np.ndarray, fermi_level: float) -> Dict[str, Any]:
        occupied_mask = eigenvalues <= fermi_level
        
        num_occupied = np.sum(occupied_mask)
        num_unoccupied = len(eigenvalues) - num_occupied
        
        total_energy = np.sum(eigenvalues[occupied_mask])
        
        kinetic_energy = self._calculate_kinetic_energy(eigenvectors[:, occupied_mask])
        potential_energy = total_energy - kinetic_energy
        
        pressure = self._calculate_electronic_pressure(eigenvalues, fermi_level)
        
        compressibility = self._calculate_compressibility(eigenvalues, fermi_level)
        
        return {
            'num_occupied_states': int(num_occupied),
            'num_unoccupied_states': int(num_unoccupied),
            'total_energy': float(total_energy * self.config.EV_CONVERSION),
            'kinetic_energy': float(kinetic_energy * self.config.EV_CONVERSION),
            'potential_energy': float(potential_energy * self.config.EV_CONVERSION),
            'energy_per_electron': float(total_energy / max(num_occupied, 1) * self.config.EV_CONVERSION),
            'electronic_pressure': float(pressure),
            'compressibility': float(compressibility),
            'susceptibility': float(compressibility * num_occupied)
        }
    
    def _calculate_kinetic_energy(self, occupied_states: np.ndarray) -> float:
        if occupied_states.size == 0:
            return 0.0
        
        momentum_expectation = np.sum(np.abs(np.gradient(occupied_states, axis=0)) ** 2)
        
        kinetic = (self.config.HBAR ** 2) * momentum_expectation / (2 * self.config.ELECTRON_MASS)
        
        return float(kinetic)
    
    def _calculate_electronic_pressure(self, eigenvalues: np.ndarray, fermi_level: float) -> float:
        occupied = eigenvalues[eigenvalues <= fermi_level]
        
        if len(occupied) == 0:
            return 0.0
        
        degeneracy_pressure = (3 * np.pi ** 2) ** (2/3) * (self.config.HBAR ** 2) * (len(occupied) ** (5/3)) / (5 * self.config.ELECTRON_MASS)
        
        return float(degeneracy_pressure * self.config.EV_CONVERSION)
    
    def _calculate_compressibility(self, eigenvalues: np.ndarray, fermi_level: float) -> float:
        dos_at_fermi = np.sum(np.abs(eigenvalues - fermi_level) < 0.01)
        
        if dos_at_fermi == 0:
            return 0.0
        
        compressibility = dos_at_fermi / (fermi_level + 1e-10)
        
        return float(compressibility)


class MetalInsulatorClassifier:
    def __init__(self, config: FermiConfig = FermiConfig()):
        self.config = config
    
    def classify(self, band_gap: float, dos_at_fermi: float) -> str:
        gap_threshold = 0.1 * self.config.EV_CONVERSION
        
        if band_gap > gap_threshold and dos_at_fermi < 0.01:
            return 'insulator'
        elif band_gap > gap_threshold * 0.1 and dos_at_fermi < 0.1:
            return 'semiconductor'
        elif band_gap < gap_threshold and dos_at_fermi > 0.5:
            return 'metal'
        elif dos_at_fermi > 0.1:
            return 'semimetal'
        else:
            return 'disordered_metal'
    
    def classify_transport(self, effective_masses: Dict[str, float], band_gap: float) -> Dict[str, str]:
        electron_mass = effective_masses.get('electron', float('inf'))
        hole_mass = effective_masses.get('hole', float('inf'))
        
        mobility_type = 'high_mobility' if electron_mass < 0.1 * self.config.EV_CONVERSION else 'low_mobility'
        
        if band_gap < 0.01 * self.config.EV_CONVERSION:
            conduction_type = 'metallic'
        elif electron_mass < hole_mass:
            conduction_type = 'n_type'
        elif hole_mass < electron_mass:
            conduction_type = 'p_type'
        else:
            conduction_type = 'intrinsic'
        
        return {
            'mobility_type': mobility_type,
            'conduction_type': conduction_type,
            'dominant_carrier': 'electrons' if electron_mass < hole_mass else 'holes'
        }


class CheckpointMigrator:
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


class FermiLevelAnalyzer:
    def __init__(self, checkpoint_path: str, config: FermiConfig = FermiConfig()):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        self.band_calculator = BandStructureCalculator(config)
        self.fermi_calculator = FermiLevelCalculator(config)
        self.dos_calculator = DensityOfStatesCalculator(config)
        self.electronic_calculator = ElectronicPropertiesCalculator(config)
        self.classifier = MetalInsulatorClassifier(config)
        self.migrator = CheckpointMigrator()
        
        self._load_checkpoint()
    
    def _load_checkpoint(self):
        try:
            self.checkpoint = torch.load(
                self.checkpoint_path,
                map_location=self.config.DEVICE,
                weights_only=False
            )
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint: {e}")
        
        self.model = BilinearModel(
            hidden_dim=self.config.HIDDEN_DIM,
            matrix_size=self.config.MATRIX_SIZE
        ).to(self.config.DEVICE)
        
        state_dict = self.migrator.migrate(self.checkpoint, self.config.DEVICE)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f"Failed to migrate checkpoint: {self.checkpoint_path}")
        
        self.epoch = self.checkpoint.get('epoch', 'unknown')
    
    def analyze(self) -> Dict[str, Any]:
        band_structure = self.band_calculator.calculate(self.model)
        
        all_eigenvalues = np.array(band_structure['band_structure']).flatten()
        
        fermi_level = self.fermi_calculator.calculate(all_eigenvalues, self.config.NUM_ELECTRONS)
        
        energy_range = np.linspace(np.min(all_eigenvalues), np.max(all_eigenvalues), 100)
        dos = self.dos_calculator.calculate(all_eigenvalues, energy_range)
        
        dos_at_fermi = np.interp(fermi_level['fermi_energy'], energy_range, dos)
        
        electronic_props = self.electronic_calculator.calculate(
            all_eigenvalues, 
            np.eye(len(all_eigenvalues)), 
            fermi_level['fermi_energy']
        )
        
        material_type = self.classifier.classify(band_structure['band_gap'], dos_at_fermi)
        
        transport = self.classifier.classify_transport(band_structure['effective_masses'], band_structure['band_gap'])
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat()
            },
            'band_structure': band_structure,
            'fermi_level': fermi_level,
            'density_of_states': {
                'energies': energy_range.tolist(),
                'dos': dos.tolist(),
                'dos_at_fermi': float(dos_at_fermi),
                'peak_dos': float(np.max(dos)),
                'peak_dos_energy': float(energy_range[np.argmax(dos)])
            },
            'electronic_properties': electronic_props,
            'material_classification': {
                'type': material_type,
                'is_metal': material_type in ['metal', 'semimetal', 'disordered_metal'],
                'is_insulator': material_type == 'insulator',
                'transport_properties': transport
            },
            'quantum_numbers': {
                'num_electrons': self.config.NUM_ELECTRONS,
                'num_bands': self.config.NUM_BANDS,
                'filled_bands': self.config.NUM_ELECTRONS // self.config.MAX_OCCUPATION
            }
        }
        
        self._print_report(results)
        
        return results
    
    def _print_report(self, results: Dict):
        print("=" * 80)
        print("FERMI LEVEL ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {results['metadata']['checkpoint_path']}")
        print(f"  Epoch: {results['metadata']['epoch']}")
        
        print(f"\n[BAND STRUCTURE]")
        bs = results['band_structure']
        print(f"  Number of bands: {bs['num_bands']}")
        print(f"  Band gap: {bs['band_gap']:.6e} eV")
        print(f"  Is direct gap: {bs['is_direct_gap']}")
        print(f"  Valence band index: {bs['valence_band_index']}")
        print(f"  Conduction band index: {bs['conduction_band_index']}")
        
        print(f"\n[EFFECTIVE MASSES]")
        em = bs['effective_masses']
        print(f"  Electron mass: {em['electron']:.6e} eV")
        print(f"  Hole mass: {em['hole']:.6e} eV")
        
        print(f"\n[FERMI LEVEL]")
        fl = results['fermi_level']
        print(f"  Fermi energy: {fl['fermi_energy']:.6e}")
        print(f"  Fermi level (eV): {fl['fermi_level_ev']:.6f} eV")
        print(f"  Chemical potential: {fl['chemical_potential']:.6e}")
        print(f"  Work function: {fl['work_function']:.6f} eV")
        print(f"  Occupation at Fermi: {fl['occupation_at_fermi']:.6f}")
        print(f"  States below Fermi: {fl['num_states_below_fermi']}")
        
        print(f"\n[DENSITY OF STATES]")
        dos = results['density_of_states']
        print(f"  DOS at Fermi: {dos['dos_at_fermi']:.6e}")
        print(f"  Peak DOS: {dos['peak_dos']:.6e}")
        print(f"  Peak DOS energy: {dos['peak_dos_energy']:.6e}")
        
        print(f"\n[ELECTRONIC PROPERTIES]")
        ep = results['electronic_properties']
        print(f"  Total energy: {ep['total_energy']:.6f} eV")
        print(f"  Kinetic energy: {ep['kinetic_energy']:.6f} eV")
        print(f"  Potential energy: {ep['potential_energy']:.6f} eV")
        print(f"  Energy per electron: {ep['energy_per_electron']:.6f} eV")
        print(f"  Electronic pressure: {ep['electronic_pressure']:.6e}")
        print(f"  Compressibility: {ep['compressibility']:.6e}")
        
        print(f"\n[MATERIAL CLASSIFICATION]")
        mc = results['material_classification']
        print(f"  Type: {mc['type']}")
        print(f"  Is metal: {mc['is_metal']}")
        print(f"  Is insulator: {mc['is_insulator']}")
        print(f"  Mobility type: {mc['transport_properties']['mobility_type']}")
        print(f"  Conduction type: {mc['transport_properties']['conduction_type']}")
        print(f"  Dominant carrier: {mc['transport_properties']['dominant_carrier']}")
        
        print("=" * 80)


class FermiPipeline:
    def __init__(self, config: FermiConfig = FermiConfig()):
        self.config = config
    
    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer = FermiLevelAnalyzer(checkpoint_path, self.config)
        results = analyzer.analyze()
        
        base_name = Path(checkpoint_path).stem
        
        results_path = os.path.join(output_dir, f'{base_name}_fermi.json')
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
        
        print(f"\nProcessing {len(checkpoints)} checkpoints for Fermi level analysis...\n")
        
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
        
        metal_count = sum(1 for r in all_results if r['material_classification']['is_metal'])
        insulator_count = sum(1 for r in all_results if r['material_classification']['is_insulator'])
        
        band_gaps = [r['band_structure']['band_gap'] for r in all_results]
        fermi_levels = [r['fermi_level']['fermi_level_ev'] for r in all_results]
        dos_at_fermi = [r['density_of_states']['dos_at_fermi'] for r in all_results]
        
        summary = {
            'total_checkpoints_analyzed': len(all_results),
            'metal_count': metal_count,
            'insulator_count': insulator_count,
            'semiconductor_count': len(all_results) - metal_count - insulator_count,
            'mean_band_gap_ev': float(np.mean(band_gaps)) if band_gaps else 0,
            'mean_fermi_level_ev': float(np.mean(fermi_levels)) if fermi_levels else 0,
            'mean_dos_at_fermi': float(np.mean(dos_at_fermi)) if dos_at_fermi else 0,
            'band_gap_range_ev': [float(np.min(band_gaps)), float(np.max(band_gaps))] if band_gaps else [0, 0],
            'timestamp': datetime.now().isoformat(),
            'individual_results': all_results
        }
        
        summary_path = os.path.join(output_dir, 'fermi_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._generate_text_report(summary, output_dir)
        
        print(f"\nSaved Fermi summary: {summary_path}")
    
    def _generate_text_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        report_path = os.path.join(output_dir, 'fermi_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FERMI LEVEL ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total checkpoints analyzed: {summary['total_checkpoints_analyzed']}\n")
            f.write(f"Metals: {summary['metal_count']}\n")
            f.write(f"Insulators: {summary['insulator_count']}\n")
            f.write(f"Semiconductors/Semimetals: {summary['semiconductor_count']}\n")
            f.write(f"Mean band gap: {summary['mean_band_gap_ev']:.6f} eV\n")
            f.write(f"Mean Fermi level: {summary['mean_fermi_level_ev']:.6f} eV\n")
            f.write(f"Mean DOS at Fermi: {summary['mean_dos_at_fermi']:.6e}\n")
            f.write(f"Band gap range: [{summary['band_gap_range_ev'][0]:.6f}, {summary['band_gap_range_ev'][1]:.6f}] eV\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("INDIVIDUAL CHECKPOINT ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            for i, r in enumerate(summary['individual_results'], 1):
                f.write(f"[{i}] {r['metadata']['checkpoint_path']}\n")
                f.write(f"    Epoch: {r['metadata']['epoch']}\n")
                f.write(f"    Material type: {r['material_classification']['type']}\n")
                f.write(f"    Band gap: {r['band_structure']['band_gap']:.6e} eV\n")
                f.write(f"    Fermi level: {r['fermi_level']['fermi_level_ev']:.6f} eV\n")
                f.write(f"    DOS at Fermi: {r['density_of_states']['dos_at_fermi']:.6e}\n")
                f.write(f"    Is metal: {r['material_classification']['is_metal']}\n")
                f.write(f"    Conduction type: {r['material_classification']['transport_properties']['conduction_type']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved text report: {report_path}")
    
    def plot_band_structures(self, all_results: List[Dict[str, Any]], output_dir: str) -> None:
        """Generate comparison plots of band structures across checkpoints."""
        if not all_results:
            return
        
        n_plots = min(len(all_results), 9)
        n_cols = 3
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_plots == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        for idx, (ax, result) in enumerate(zip(axes[:n_plots], all_results[:n_plots])):
            bs = result['band_structure']
            k_points = np.array(bs['k_points'])
            bands = np.array(bs['band_structure'])
            
            for band_idx in range(min(bands.shape[1], 5)):
                ax.plot(k_points, bands[:, band_idx], 'b-', alpha=0.7, linewidth=1)
            
            fermi = result['fermi_level']['fermi_energy']
            ax.axhline(y=fermi, color='r', linestyle='--', label=f'Fermi: {fermi:.2e}')
            
            ax.set_xlabel('k')
            ax.set_ylabel('Energy')
            ax.set_title(f"Epoch {result['metadata']['epoch']}\n{result['material_classification']['type']}")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        for idx in range(n_plots, len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        plot_path = os.path.join(output_dir, f'band_structures_comparison.{self.config.SAVE_FORMAT}')
        plt.savefig(plot_path, dpi=self.config.FIGURE_DPI)
        plt.close()
        
        print(f"Saved band structure plots: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Fermi level analysis for neural network weights as electronic system'
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
        default='fermi_analysis',
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
        help='Matrix size'
    )
    parser.add_argument(
        '--num-electrons',
        type=int,
        default=7,
        help='Number of electrons (7 for Strassen)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate band structure plots'
    )
    
    args = parser.parse_args()
    
    config = FermiConfig(
        HIDDEN_DIM=args.hidden_dim,
        MATRIX_SIZE=args.matrix_size,
        NUM_ELECTRONS=args.num_electrons
    )
    
    pipeline = FermiPipeline(config)
    
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            pipeline.process_checkpoint(args.checkpoint, args.output)
        else:
            print(f"Error: Checkpoint not found: {args.checkpoint}")
    elif args.all or args.latest is None:
        n_to_process = args.latest if args.latest is not None else None
        results = pipeline.process_directory(args.dir, n_to_process, args.output)
        if results:
            pipeline.generate_summary(results, args.output)
            if args.plot:
                pipeline.plot_band_structures(results, args.output)
    elif args.latest:
        results = pipeline.process_directory(args.dir, args.latest, args.output)
        if results:
            pipeline.generate_summary(results, args.output)
            if args.plot:
                pipeline.plot_band_structures(results, args.output)


if __name__ == '__main__':
    main()
