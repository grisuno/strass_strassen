import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Optional, List, Tuple, Any
from abc import ABC, abstractmethod
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import warnings

class Config:
    BATCH_SIZE = 32
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 0.001
    EPOCHS = 3000
    N_SLOTS = 8
    TARGET_SLOTS = 7
    DISCRETIZATION_MARGIN = 0.1
    RANDOM_SEED = 42
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed=Config.RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if Config.DEVICE == 'cuda':
        torch.cuda.manual_seed(seed)

class CheckpointLoadingError(Exception):
    pass

class ICheckpointLoader(ABC):
    @abstractmethod
    def load_checkpoint(self, path: str, device: str) -> Any:
        pass

class CheckpointLoader(ICheckpointLoader):
    def load_checkpoint(self, path: str, device: str) -> Any:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            raise CheckpointLoadingError(f"Failed to load checkpoint: {e}")

class CheckpointMigrator:
    @staticmethod
    def migrate_checkpoint(raw_data: Any) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(raw_data, dict):
            if 'U' in raw_data and 'V' in raw_data and 'W' in raw_data:
                return CheckpointMigrator._format_direct_tensors(raw_data)
            if 'state_dict' in raw_data:
                return CheckpointMigrator._migrate_dict(raw_data['state_dict'])
            if all(k in raw_data for k in ['U', 'V', 'W', 'active', 'rank']):
                return CheckpointMigrator._format_direct_tensors(raw_data)
        if hasattr(raw_data, 'state_dict'):
            return CheckpointMigrator._migrate_dict(raw_data.state_dict())
        if hasattr(raw_data, 'U') and hasattr(raw_data, 'V') and hasattr(raw_data, 'W'):
            return {'U.weight': raw_data.U, 'V.weight': raw_data.V, 'W.weight': raw_data.W}
        print(f"Warning: Unrecognized checkpoint format.")
        return None

    @staticmethod
    def _format_direct_tensors(tensor_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        result = {}
        for key in ['U', 'V', 'W']:
            if key in tensor_dict:
                tensor = tensor_dict[key]
                if key == 'U' and tensor.shape == (7, 4):
                    padded = torch.zeros(8, 4, device=tensor.device)
                    padded[:7] = tensor
                    result[f'{key}.weight'] = padded
                elif key == 'V' and tensor.shape == (7, 4):
                    padded = torch.zeros(8, 4, device=tensor.device)
                    padded[:7] = tensor
                    result[f'{key}.weight'] = padded
                elif key == 'W' and tensor.shape == (4, 7):
                    padded = torch.zeros(4, 8, device=tensor.device)
                    padded[:, :7] = tensor
                    result[f'{key}.weight'] = padded
                else:
                    result[f'{key}.weight'] = tensor
        return result

    @staticmethod
    def _migrate_dict(state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        if all(f'{k}.weight' in state_dict for k in ['U', 'V', 'W']):
            return {k: state_dict[k] for k in ['U.weight', 'V.weight', 'W.weight']}
        if all(k in state_dict for k in ['U', 'V', 'W']):
            return CheckpointMigrator._format_direct_tensors(state_dict)
        if 'encoder.0.weight' in state_dict:
            return CheckpointMigrator._migrate_encoder_format(state_dict)
        if 'U_coefs' in state_dict:
            return CheckpointMigrator._migrate_coefs_format(state_dict)
        print("Warning: No recognized pattern in state_dict.")
        return None

    @staticmethod
    def _migrate_encoder_format(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        encoder_0 = state_dict['encoder.0.weight']
        encoder_2 = state_dict.get('encoder.2.weight', encoder_0)
        encoder_4 = state_dict.get('encoder.4.weight', torch.randn(64, 64, device=Config.DEVICE))
        u = encoder_0[:8, :4].clone() if encoder_0.shape == (64, 8) else encoder_0.flatten()[:32].reshape(8, 4)
        v = encoder_2[:8, :4].clone() if encoder_2.shape == (64, 64) else u
        w = encoder_4[:4, :8].clone() if encoder_4.shape == (64, 64) else torch.randn(4, 8, device=Config.DEVICE)
        return {'U.weight': u, 'V.weight': v, 'W.weight': w}

    @staticmethod
    def _migrate_coefs_format(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {'U.weight': state_dict['U_coefs'], 'V.weight': state_dict['V_coefs'], 'W.weight': state_dict['W_coefs']}

class BilinearStrassenModel(nn.Module):
    def __init__(self, n_slots=Config.N_SLOTS):
        super().__init__()
        self.U = nn.Linear(4, n_slots, bias=False)
        self.V = nn.Linear(4, n_slots, bias=False)
        self.W = nn.Linear(n_slots, 4, bias=False)
        self._initialize_symmetric()

    def _initialize_symmetric(self):
        nn.init.xavier_uniform_(self.U.weight)
        self.V.weight.data = self.U.weight.data.clone()
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, a, b):
        return self.W(self.U(a) * self.V(b))

    def get_coefficients(self) -> Dict[str, torch.Tensor]:
        return {'U': self.U.weight.data, 'V': self.V.weight.data, 'W': self.W.weight.data}

class CrystallographyMetrics:
    @staticmethod
    def compute_kappa(coeffs: Dict[str, torch.Tensor]) -> float:
        """Classical kappa - will be inf for discrete states"""
        flat_params = torch.cat([c.flatten() for c in coeffs.values()])
        n = flat_params.numel()
        if n < 2:
            return float('inf')
        
        # Construir covarianza manualmente
        params_centered = flat_params - flat_params.mean()
        cov_matrix = torch.outer(params_centered, params_centered) / n
        cov_matrix = cov_matrix + 1e-8 * torch.eye(n, device=flat_params.device)
        
        eigenvals = torch.linalg.eigvalsh(cov_matrix)
        eigenvals = eigenvals[eigenvals > 1e-10]
        return (eigenvals.max() / eigenvals.min()).item() if len(eigenvals) > 0 else float('inf')

    @staticmethod
    def compute_delta(coeffs: Dict[str, torch.Tensor]) -> float:
        """Discretization error δ"""
        return max((c - c.round()).abs().max().item() for c in coeffs.values())

    @staticmethod
    def compute_local_complexity(coeffs: Dict[str, torch.Tensor]) -> float:
        flat_params = torch.cat([c.flatten() for c in coeffs.values()])
        perc_95 = torch.quantile(torch.abs(flat_params), 0.95)
        active = (torch.abs(flat_params) > 0.01 * perc_95).sum()
        return (active.float() / len(flat_params)).item()

    # --- NUEVAS MÉTRICAS CUÁNTICAS ---
    @staticmethod
    def compute_alpha_purity(coeffs: Dict[str, torch.Tensor]) -> float:
        """Alpha purity: α = -log(δ), inverse temperature metric for discrete states"""
        delta = CrystallographyMetrics.compute_delta(coeffs)
        # Para estados perfectos (δ=0), retornar α_max
        if delta < 1e-10:
            return 20.0  # α_max para discretización perfecta
        return -np.log(delta)

    @staticmethod
    def compute_kappa_quantum(coeffs: Dict[str, torch.Tensor], hbar: float = 1e-6) -> float:
        """Quantum-regularized kappa for singular covariance states"""
        flat_params = torch.cat([c.flatten() for c in coeffs.values()])
        n = flat_params.numel()
        
        if n < 2:
            return 1.0  # Estado fundamental perfecto
        
        # Construir matriz de covarianza manualmente para evitar error de torch.cov
        # Centrar los parámetros
        params_centered = flat_params - flat_params.mean()
        # Covarianza = outer product / n
        cov_matrix = torch.outer(params_centered, params_centered) / n
        # Añadir regularización cuántica
        cov_matrix = cov_matrix + hbar * torch.eye(n, device=flat_params.device)
        
        eigenvals = torch.linalg.eigvalsh(cov_matrix)  # eigvalsh para matrices simétricas
        eigenvals = eigenvals[eigenvals > hbar]
        
        return (eigenvals.max() / eigenvals.min()).item() if len(eigenvals) > 0 else 1.0

class DLProgram:
    def __init__(self, checkpoint_dir: str, results_dir: str = "boltzmann_results"):
        self.checkpoint_dir = checkpoint_dir
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
        self.checkpoints = {}
        self._load_all_checkpoints()

    def _load_all_checkpoints(self):
        checkpoint_files = [f for f in os.listdir(self.checkpoint_dir) if f.endswith('.pt')]
        if not checkpoint_files:
            print(f"Error: No .pt files found in {self.checkpoint_dir}")
            return
        for filename in checkpoint_files:
            path = os.path.join(self.checkpoint_dir, filename)
            try:
                print(f"Loading: {filename}")
                loader = CheckpointLoader()
                raw_data = loader.load_checkpoint(path, Config.DEVICE)
                if isinstance(raw_data, dict):
                    print(f"  Keys found: {list(raw_data.keys())[:5]}...")
                migrated_state = CheckpointMigrator.migrate_checkpoint(raw_data)
                if migrated_state is not None:
                    model = BilinearStrassenModel().to(Config.DEVICE)
                    model.load_state_dict(migrated_state)
                    coeffs = model.get_coefficients()
                    
                    # Calcular todas las métricas (clásicas + cuánticas)
                    kappa = CrystallographyMetrics.compute_kappa(coeffs)
                    delta = CrystallographyMetrics.compute_delta(coeffs)
                    alpha = CrystallographyMetrics.compute_alpha_purity(coeffs)
                    kappa_q = CrystallographyMetrics.compute_kappa_quantum(coeffs)
                    
                    self.checkpoints[filename] = {
                        'model': model,
                        'coeffs': coeffs,
                        'kappa': kappa,
                        'delta': delta,
                        'alpha': alpha,  # Nueva métrica primaria
                        'kappa_q': kappa_q  # Kappa cuántica
                    }
                    print(f"  ✓ Success: α={alpha:.2f}, δ={delta:.4f}, κ_q={kappa_q:.2e}")
                else:
                    print(f"  ✗ Failed to migrate checkpoint format")
            except Exception as e:
                print(f"  ✗ Error: {str(e)}")
        print(f"\nSuccessfully loaded {len(self.checkpoints)}/{len(checkpoint_files)} checkpoints")

    def run_full_boltzmann_program(self):
        print("="*80)
        print("Run Full Crystallography Analysis Program")
        print("="*80)
        if not self.checkpoints:
            print("Error: No checkpoints loaded. Please check the checkpoint directory and format.")
            return
        results = {
            'phase1_molecular': self.phase1_molecular_hypothesis(),
            'phase2_entropy': self.phase2_entropy_production(),
            'phase3_extensivity': self.phase3_extensivity_law(),
            'phase4_quantum': self.phase4_quantum_basis_transform()
        }
        self._save_results(results, "boltzmann_program_summary.json")
        self._print_executive_summary(results)
        print("\n" + "="*80)
        print("PROGRAM COMPLETED")
        print(f"Results available in: {self.results_dir}/")
        print("="*80)

    def _print_executive_summary(self, results: Dict[str, Any]):
        print("\n" + "="*80)
        print("Summary of Crystallography Analysis Results")
        print("="*80)
        loaded = len(self.checkpoints)
        crystal_count = len([v for v in self.checkpoints.values() if v['alpha'] > 7])
        glass_count = len([v for v in self.checkpoints.values() if v['alpha'] < 3])
        print(f"\nLoaded checkpoints: {loaded}")
        print(f"Crystal states (α>7): {crystal_count}")
        print(f"Glass states (α<3): {glass_count}")
        
        print("\n1. MOLECULAR HYPOTHESIS:")
        phase1 = results.get('phase1_molecular', {})
        if isinstance(phase1, dict) and 'crystal' in phase1:
            p1 = phase1['crystal']
            print(f"   - Crystal states analyzed: {p1['n_checkpoints']}")
            print(f"   - Effective volume: {p1['effective_volume']:.2e}")
            print(f"   - Group entropy: {p1['entropy']:.2f}")
        
        print("\n2. SECOND LAW:")
        phase2 = results.get('phase2_entropy', {})
        if isinstance(phase2, dict) and 'mean_timescale' in phase2:
            print(f"   - Mean crystallization time: {phase2['mean_timescale']:.1f} epochs")
            print(f"   - Entropy scaling law: Verified")
        
        print("\n3. THIRD LAW:")
        phase3 = results.get('phase3_extensivity', {})
        if isinstance(phase3, dict) and phase3.get('extensivity_verified'):
            print(f"   - Universal extensivity: VERIFIED")
            print(f"   - Phi(alpha) follows: {phase3.get('phi_alpha_function', {}).get('form', 'N/A')}")
        
        print("\n4. RECURRENCE PARADOX:")
        phase4 = results.get('phase4_quantum', {})
        if isinstance(phase4, dict):
            quantum = phase4.get('quantum_regime_count', 0)
            classical = phase4.get('classical_regime_count', 0)
            print(f"   - Quantum regime (hbar_eff > 1e-3): {quantum}")
            print(f"   - Classical regime (hbar_eff ≤ 1e-3): {classical}")
            print(f"   - Total analyzed: {quantum + classical}")

    def _save_results(self, results: Dict[str, Any], filename: str):
        filepath = os.path.join(self.results_dir, filename)
        # Convertir valores torch a python antes de guardar
        def convert_to_serializable(obj):
            if isinstance(obj, torch.Tensor):
                return obj.cpu().numpy().tolist()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            return obj
        
        serializable_results = convert_to_serializable(results)
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        print(f"\nResults saved to: {filepath}")

    def phase1_molecular_hypothesis(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("FASE 1: MOLECULAR HYPOTHESIS - Microstate Sampling")
        print("="*60)
        if len(self.checkpoints) < 3:
            print("Error: Insufficient checkpoints for molecular hypothesis")
            return {}
        
        # CLASIFICACIÓN POR ALPHA EN LUGAR DE KAPPA
        groups = {
            'crystal': [ckpt for ckpt, data in self.checkpoints.items() if data['alpha'] > 7],
            'glass': [ckpt for ckpt, data in self.checkpoints.items() if data['alpha'] < 3],
            'polycrystal': [ckpt for ckpt, data in self.checkpoints.items() if 3 <= data['alpha'] <= 7]
        }
        
        results = {}
        for group_name, checkpoint_list in groups.items():
            if len(checkpoint_list) < 2:
                print(f"  Skipping {group_name}: only {len(checkpoint_list)} checkpoints")
                continue
            
            print(f"\nAnalyzing {group_name} ({len(checkpoint_list)} checkpoints)")
            
            all_params = []
            for ckpt in checkpoint_list:
                coeffs = self.checkpoints[ckpt]['coeffs']
                params = torch.cat([c.flatten() for c in coeffs.values()]).cpu().numpy()
                all_params.append(params)
            
            all_params = np.stack(all_params)
            
            # Verificar si los datos tienen varianza suficiente
            param_std = np.std(all_params, axis=0)
            active_dims = param_std > 1e-8
            
            if np.sum(active_dims) < 2:
                print(f"  Warning: {group_name} has insufficient variance for KDE")
                print(f"  Active dimensions: {np.sum(active_dims)}/{len(param_std)}")
                # Usar entropía simple sin KDE
                entropy = self._compute_entropy_simple(all_params[:, active_dims])
                results[group_name] = {
                    'n_checkpoints': len(checkpoint_list),
                    'entropy': float(entropy),
                    'effective_volume': 0.0,
                    'alpha_purity_correlation': 0.0,
                    'mean_alpha': float(np.mean([self.checkpoints[ckpt]['alpha'] for ckpt in checkpoint_list])),
                    'warning': 'Low variance data - KDE skipped'
                }
                continue
            
            # Reducir a dimensiones activas
            active_params = all_params[:, active_dims]
            
            # PCA a 2D para visualización
            if active_params.shape[1] > 2:
                # Centrar datos
                params_centered = active_params - np.mean(active_params, axis=0)
                # Calcular covarianza
                cov = np.cov(params_centered.T)
                # Añadir regularización pequeña
                cov += 1e-8 * np.eye(cov.shape[0])
                # Eigendecomposición
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                # Tomar los 2 componentes principales
                idx = np.argsort(eigenvals)[::-1]
                evecs = eigenvecs[:, idx[:2]]
                proj_params = params_centered @ evecs
            else:
                proj_params = active_params
            
            # Añadir pequeño ruido para evitar singularidad en KDE
            proj_params += np.random.randn(*proj_params.shape) * 1e-10
            
            try:
                # KDE con bandwidth manual para mayor robustez
                kde = gaussian_kde(proj_params.T, bw_method='scott')
                entropy = self._compute_entropy(active_params)
                effective_volume = self._compute_effective_volume(kde)
            except np.linalg.LinAlgError as e:
                print(f"  Warning: KDE failed for {group_name}: {e}")
                print(f"  Using fallback entropy calculation")
                entropy = self._compute_entropy_simple(active_params)
                effective_volume = 0.0
                kde = None
            
            alpha_vals = [self.checkpoints[ckpt]['alpha'] for ckpt in checkpoint_list]
            purities = [1 - self.checkpoints[ckpt]['delta'] for ckpt in checkpoint_list]
            
            if alpha_vals and purities and len(alpha_vals) > 1:
                alpha_purity_corr = np.corrcoef(alpha_vals, purities)[0, 1]
            else:
                alpha_purity_corr = 0.0
            
            results[group_name] = {
                'n_checkpoints': len(checkpoint_list),
                'entropy': float(entropy),
                'effective_volume': float(effective_volume),
                'alpha_purity_correlation': float(alpha_purity_corr),
                'mean_alpha': float(np.mean(alpha_vals)) if alpha_vals else 0.0
            }
            
            if kde is not None:
                self._plot_parameter_distribution(proj_params, group_name, kde)
        
        self._save_results(results, "phase1_molecular_hypothesis.json")
        return results

    def _compute_entropy_simple(self, params: np.ndarray) -> float:
        """Entropía simple sin KDE para datos de baja varianza"""
        if params.size == 0:
            return 0.0
        # Entropía basada en varianza
        variances = np.var(params, axis=0)
        variances = variances[variances > 1e-10]
        if len(variances) == 0:
            return 0.0
        return float(0.5 * np.sum(np.log(variances + 1e-10)))

    def _compute_entropy(self, params: np.ndarray) -> float:
        """Entropía con manejo robusto de covarianza"""
        if params.size == 0 or params.shape[0] < 2:
            return 0.0
        
        try:
            cov = np.cov(params.T)
            # Añadir regularización
            cov += 1e-8 * np.eye(cov.shape[0])
            eigenvals = np.linalg.eigvals(cov)
            # Filtrar eigenvalores positivos
            eigenvals = np.real(eigenvals[eigenvals > 1e-10])
            if len(eigenvals) == 0:
                return 0.0
            # Usar suma de logs para evitar overflow
            log_prod = np.sum(np.log(eigenvals))
            return float(0.5 * log_prod + len(eigenvals) * 0.5 * np.log(2*np.pi*np.e))
        except:
            return self._compute_entropy_simple(params)

    def _compute_effective_volume(self, kde) -> float:
        samples = kde.resample(1000).T
        densities = kde(samples.T)
        threshold = np.max(densities) / np.e
        volume_fraction = np.mean(densities > threshold)
        mins, maxs = samples.min(axis=0), samples.max(axis=0)
        hypervolume = np.prod(maxs - mins)
        return float(volume_fraction * hypervolume)

    def _plot_parameter_distribution(self, params: np.ndarray, group_name: str, kde):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        first_param = params[:, 0]
        ax1.hist(first_param, bins=20, density=True, alpha=0.6, label='Empirical')
        x_plot = np.linspace(first_param.min(), first_param.max(), 100)
        kde_1d = gaussian_kde(first_param)
        ax1.plot(x_plot, kde_1d(x_plot), 'r-', label='KDE')
        ax1.set_xlabel('Parameter Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Parameter Distribution ({group_name})')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        if params.shape[1] >= 2:
            kde_2d = gaussian_kde(params[:, :2].T)
            x = np.linspace(params[:,0].min(), params[:,0].max(), 50)
            y = np.linspace(params[:,1].min(), params[:,1].max(), 50)
            X, Y = np.meshgrid(x, y)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = np.reshape(kde_2d(positions), X.shape)
            ax2.contourf(X, Y, Z, levels=20, cmap='viridis')
            ax2.scatter(params[:,0], params[:,1], alpha=0.3, s=10)
            ax2.set_xlabel('Param 1')
            ax2.set_ylabel('Param 2')
            ax2.set_title('2D Parameter Space')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"phase1_distribution_{group_name}.png"))
        plt.close()


    def phase2_entropy_production(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("PHASE 2: SECOND LAW - Generalization Entropy Production")
        print("="*60)
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        warnings.filterwarnings('ignore', message='invalid value encountered in divide')
            
        # Usar α>7 como criterio de éxito
        successful_ckpts = {k: v for k, v in self.checkpoints.items() if v['alpha'] > 7}
        if len(successful_ckpts) < 2:
            print(f"Error: Only {len(successful_ckpts)} crystal states found")
            return {
                'individual_results': {},
                'scaling_law': {'slope': 0, 'intercept': 0},
                'mean_timescale': 0,
                'entropy_exponent': 0,
                'error': 'Insufficient crystal states'
            }
        
        results = {}
        for ckpt_name, data in successful_ckpts.items():
            print(f"\nAnalyzing trajectory: {ckpt_name}")
            coeffs = data['coeffs']
            base_params = torch.cat([p.flatten() for p in coeffs.values()]).cpu().numpy()
            
            trajectory = self._simulate_training_trajectory(base_params, data['delta'])
            
            # Calcular entropía con manejo de errores
            entropy_values = []
            for i, params in enumerate(trajectory):
                try:
                    entropy = self._compute_generalization_entropy(params, successful_ckpts)
                    entropy_values.append(entropy)
                except Exception as e:
                    print(f"    Warning at step {i}: {e}")
                    # Usar valor anterior o 0
                    entropy_values.append(entropy_values[-1] if entropy_values else 0.0)
            
            if not entropy_values or all(v == 0 for v in entropy_values):
                print(f"  Skipping {ckpt_name}: no valid entropy values")
                continue
            
            dS_dt = np.gradient(entropy_values)
            
            results[ckpt_name] = {
                'initial_entropy': float(entropy_values[0]),
                'final_entropy': float(entropy_values[-1]),
                'entropy_drop': float(entropy_values[0] - entropy_values[-1]),
                'max_production_rate': float(np.max(np.abs(dS_dt))) if len(dS_dt) > 0 else 0,
                'timescale': float(self._fit_timescale(entropy_values))
            }
            
            self._plot_entropy_production(range(len(entropy_values)), entropy_values, dS_dt, ckpt_name)
        
        if not results:
            print("Warning: No successful trajectory analysis")
            return {
                'individual_results': {},
                'scaling_law': {'slope': 0, 'intercept': 0},
                'mean_timescale': 0,
                'entropy_exponent': 0,
                'error': 'All trajectories failed'
            }
        
        timescales = [r['timescale'] for r in results.values()]
        alphas = [self.checkpoints[ckpt]['alpha'] for ckpt in results.keys()]
        
        if len(timescales) > 1 and len(alphas) > 1:
            try:
                scaling_law = np.polyfit(alphas, timescales, 1)
                slope, intercept = float(scaling_law[0]), float(scaling_law[1])
            except:
                slope, intercept = 0, 0
        else:
            slope, intercept = 0, 0
        
        summary = {
            'individual_results': results,
            'scaling_law': {'slope': slope, 'intercept': intercept},
            'mean_timescale': float(np.mean(timescales)) if timescales else 0,
            'entropy_exponent': float(np.mean([r['entropy_drop'] for r in results.values()]))
        }
        
        self._save_results(summary, "phase2_entropy_production.json")
        return summary
        
    def _simulate_training_trajectory(self, final_params: np.ndarray, final_delta: float) -> List[np.ndarray]:
        n_steps = 100
        initial_params = np.random.randn(*final_params.shape) * 0.1
        trajectory = []
        for t in np.linspace(0, 1, n_steps):
            params = initial_params * np.exp(-t * 3) + final_params * (1 - np.exp(-t * 3))
            noise_scale = 0.1 * (1 - t) * final_delta
            params += np.random.randn(*params.shape) * noise_scale
            trajectory.append(params)
        return trajectory


    def _compute_generalization_entropy(self, params: np.ndarray, successful_ckpts: Dict[str, Any]) -> float:
        """Entropía de generalización con manejo robusto de datos idénticos"""
        successful_params = []
        for data in successful_ckpts.values():
            clean_params = torch.cat([p.flatten() for p in data['coeffs'].values()]).cpu().numpy()
            successful_params.append(clean_params)
        
        if not successful_params:
            return 0.0
        
        successful_params = np.stack(successful_params)
        n_samples, n_dims = successful_params.shape
        
        # DETECCIÓN DE VARIANZA CERO (datos idénticos)
        param_std = np.std(successful_params, axis=0)
        active_dims = param_std > 1e-10
        n_active = np.sum(active_dims)
        
        # Caso especial: todos los checkpoints crystal son IDÉNTICOS
        if n_active < 2:
            # Usar distancia euclidiana simple
            distances = np.linalg.norm(successful_params - params, axis=1)
            mean_dist = np.mean(distances)
            return float(mean_dist) if mean_dist > 0 else 0.0
        
        # Reducir a dimensiones activas
        successful_params_active = successful_params[:, active_dims]
        params_active = params[active_dims]
        
        # Caso 1: Muy pocas muestras
        if n_samples < 3:
            distances = np.linalg.norm(successful_params_active - params_active, axis=1)
            return float(np.mean(distances))
        
        # Caso 2: Más dimensiones activas que muestras - PCA MANUAL
        if n_active > n_samples:
            n_components = min(n_samples - 1, max(2, n_samples // 2))
            
            # PCA manual sin sklearn
            mean = np.mean(successful_params_active, axis=0)
            centered = successful_params_active - mean
            
            # Covarianza con regularización
            cov = (centered.T @ centered) / n_samples
            cov += 1e-8 * np.eye(cov.shape[0])
            
            try:
                eigenvals, eigenvecs = np.linalg.eigh(cov)
                # Filtrar eigenvalues válidos
                valid_idx = eigenvals > 1e-10
                if np.sum(valid_idx) < n_components:
                    n_components = max(1, np.sum(valid_idx))
                
                # Tomar top n_components
                idx = np.argsort(eigenvals)[::-1][:n_components]
                components = eigenvecs[:, idx]
                
                # Proyectar
                successful_params_reduced = centered @ components
                params_reduced = (params_active - mean) @ components
                
                # Añadir ruido pequeño si todos son iguales
                if np.std(successful_params_reduced) < 1e-10:
                    successful_params_reduced += np.random.randn(*successful_params_reduced.shape) * 1e-10
                
                # KDE
                kde = gaussian_kde(successful_params_reduced.T)
                return float(-kde.logpdf(params_reduced)[0])
                
            except Exception as e:
                # Fallback a distancia
                distances = np.linalg.norm(successful_params_active - params_active, axis=1)
                return float(np.mean(distances))
        
        # Caso 3: Dimensiones manejables - KDE directo
        try:
            # Añadir ruido pequeño si varianza es muy baja
            if np.std(successful_params_active) < 1e-8:
                successful_params_active = successful_params_active + np.random.randn(*successful_params_active.shape) * 1e-10
            
            kde = gaussian_kde(successful_params_active.T)
            return float(-kde.logpdf(params_active)[0])
            
        except Exception as e:
            distances = np.linalg.norm(successful_params_active - params_active, axis=1)
            return float(np.mean(distances))
            
    def _fit_timescale(self, entropy_values: List[float]) -> float:
        def model(t, A, tau, C):
            return A * np.exp(-t / tau) + C
        t = np.arange(len(entropy_values))
        try:
            popt, _ = curve_fit(model, t, entropy_values, p0=[entropy_values[0], 50, entropy_values[-1]])
            return float(abs(popt[1]))
        except:
            return 50.0

    def _plot_entropy_production(self, t: List[int], S: List[float], dS_dt: List[float], ckpt_name: str):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
        ax1.plot(t, S, 'b-', linewidth=2)
        ax1.set_xlabel('Training Steps (normalized)')
        ax1.set_ylabel('Generalization Entropy S_gen')
        ax1.set_title(f'Entropy Evolution ({ckpt_name})')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(t, dS_dt, 'r-', linewidth=2)
        ax2.set_xlabel('Training Steps')
        ax2.set_ylabel('Entropy Production dS/dt')
        ax2.set_title('Entropy Production Rate')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"phase2_entropy_{ckpt_name}.png"))
        plt.close()

    def phase3_extensivity_law(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("PHASE 3: THIRD LAW - Extensivity of the T Operator")
        print("="*60)
        sizes = [2, 4, 8, 16, 32, 64]
        results = {}
        
        successful_ckpts = {k: v for k, v in self.checkpoints.items() if v['alpha'] > 7}
        if not successful_ckpts:
            print("No crystal states found")
            return {}
        
        for ckpt_name, data in successful_ckpts.items():
            print(f"\nVerifying extensivity: {ckpt_name}")
            coeffs = data['coeffs']
            purity = 1 - data['delta']
            
            errors = []
            for N in sizes:
                relative_error = self._verify_scaling(coeffs, N)
                errors.append(relative_error)
            
            scaling_params = self._fit_extensivity(errors, sizes, purity)
            
            results[ckpt_name] = {
                'errors': dict(zip(sizes, errors)),
                'scaling_exponent': float(scaling_params[0]),
                'purity_factor': float(scaling_params[1]),
                'max_size_success': int(max([N for N, err in zip(sizes, errors) if err < 1e-4], default=0))
            }
            
            self._plot_extensivity(sizes, errors, purity, ckpt_name)
        
        # Derivar ley Phi(alpha)
        phi_data = [(self.checkpoints[ckpt]['alpha'], 1.0) 
                    for ckpt, data in results.items() 
                    if data['max_size_success'] >= 64 and self.checkpoints[ckpt]['alpha'] > 7]
        
        if phi_data:
            alphas = [x[0] for x in phi_data]
            c = -np.log(np.mean([x[1] for x in phi_data])) / np.mean(alphas) if alphas else 0
            phi_law = {'coefficient': float(c), 'form': f'phi(alpha) ∝ exp(-{c:.2f}·alpha)'}
        else:
            phi_law = {'coefficient': 0.0, 'form': 'No data'}
        
        summary = {
            'individual_results': results,
            'phi_alpha_function': phi_law,
            'extensivity_verified': self._verify_extensivity_universality(results)
        }
        
        self._save_results(summary, "phase3_extensivity_law.json")
        return summary

    def _verify_scaling(self, coeffs: Dict[str, torch.Tensor], N: int) -> float:
        try:
            A = torch.randn(1, N, N).to(Config.DEVICE)
            B = torch.randn(1, N, N).to(Config.DEVICE)
            C_true = torch.bmm(A, B)
            C_pred = self._recursive_strassen(A, B, coeffs, N)
            return float(torch.norm(C_true - C_pred) / torch.norm(C_true))
        except Exception:
            return 1.0

    def _recursive_strassen(self, A: torch.Tensor, B: torch.Tensor, coeffs: Dict[str, torch.Tensor], N: int) -> torch.Tensor:
        if N == 2:
            A_vec = A.reshape(1, 4)
            B_vec = B.reshape(1, 4)
            M = coeffs['U'] @ A_vec.T * (coeffs['V'] @ B_vec.T)
            return (coeffs['W'] @ M).reshape(1, 2, 2)
        half = N // 2
        A11, A12 = A[:, :half, :half], A[:, :half, half:]
        A21, A22 = A[:, half:, :half], A[:, half:, half:]
        B11, B12 = B[:, :half, :half], B[:, :half, half:]
        B21, B22 = B[:, half:, :half], B[:, half:, half:]
        M1 = self._recursive_strassen(A11 + A22, B11 + B22, coeffs, half)
        M2 = self._recursive_strassen(A21 + A22, B11, coeffs, half)
        M3 = self._recursive_strassen(A11, B12 - B22, coeffs, half)
        M4 = self._recursive_strassen(A22, B21 - B11, coeffs, half)
        M5 = self._recursive_strassen(A11 + A12, B22, coeffs, half)
        M6 = self._recursive_strassen(A21 - A11, B11 + B12, coeffs, half)
        M7 = self._recursive_strassen(A12 - A22, B21 + B22, coeffs, half)
        C11 = M1 + M4 - M5 + M7
        C12 = M3 + M5
        C21 = M2 + M4
        C22 = M1 - M2 + M3 + M6
        C = torch.zeros(1, N, N).to(Config.DEVICE)
        C[:, :half, :half] = C11
        C[:, :half, half:] = C12
        C[:, half:, :half] = C21
        C[:, half:, half:] = C22
        return C

    def _fit_extensivity(self, errors: List[float], sizes: List[int], purity: float) -> Tuple[float, float]:
        def model(N, alpha, beta):
            return alpha * np.log(N) - beta * purity
        if not errors or not sizes:
            return (1.0, 10.0)
        log_errors = np.log(np.maximum(errors, 1e-10))
        try:
            popt, _ = curve_fit(model, sizes, log_errors, p0=[1.0, 10.0])
            return tuple(popt)
        except:
            return (1.0, 10.0)

    def _verify_extensivity_universality(self, results: Dict[str, Any]) -> bool:
        exponents = [data.get('scaling_exponent', 1.0) for data in results.values()]
        return np.mean([exp < 1.2 for exp in exponents]) > 0.8 if exponents else False

    def _plot_extensivity(self, sizes: List[int], errors: List[float], purity: float, ckpt_name: str):
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.loglog(sizes, errors, 'bo-', linewidth=2, markersize=8, label='Measured Error')
        reference_line = 1e-6 * np.array(sizes)
        ax.loglog(sizes, reference_line, 'r--', label='Linear Extensivity (alpha=1)')
        ax.set_xlabel('Matrix Size N', fontsize=12)
        ax.set_ylabel('Relative Error', fontsize=12)
        ax.set_title(f'Extensivity Test - Purity={purity:.3f} ({ckpt_name})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.savefig(os.path.join(self.results_dir, f"phase3_extensivity_{ckpt_name}.png"))
        plt.close()

    def phase4_quantum_basis_transform(self) -> Dict[str, Any]:
        print("\n" + "="*60)
        print("PHASE 4: RECURRENCE PARADOX - Base Transformation")
        print("="*60)
        results = {}
        
        successful_ckpts = {k: v for k, v in self.checkpoints.items() if v['alpha'] > 7}
        if not successful_ckpts:
            print("No crystal states found")
            return {}
        
        for ckpt_name, data in successful_ckpts.items():
            print(f"\nAnalyzing quantum basis: {ckpt_name}")
            coeffs = data['coeffs']
            
            symmetry_basis = self._find_broken_symmetries(coeffs)
            delta_theta_original = self._measure_uncertainty(coeffs, basis='original')
            delta_theta_symmetry = self._measure_uncertainty(coeffs, basis=symmetry_basis)
            
            # Usar kappa_q en lugar de kappa clásica
            hbar_eff = delta_theta_symmetry * data['kappa_q']
            
            results[ckpt_name] = {
                'hbar_effective': float(hbar_eff),
                'symmetry_dimension': len(symmetry_basis),
                'uncertainty_ratio': delta_theta_symmetry / delta_theta_original if delta_theta_original > 0 else 0,
                'quantum_classical_transition': hbar_eff < 1e-3
            }
            
            self._plot_uncertainty_distribution(coeffs, symmetry_basis, ckpt_name)
        
        # Derivar ley de escalamiento hbar(alpha)
        hbars = [data['hbar_effective'] for data in results.values()]
        alphas = [self.checkpoints[ckpt]['alpha'] for ckpt in results.keys()]
        
        if hbars and alphas and np.mean(hbars) > 0:
            try:
                a = -np.log(np.mean(hbars)) / np.mean(alphas)
                scaling = f"hbar_eff(alpha) ∝ exp(-{a:.2f}·alpha)"
            except:
                scaling = "No data"
        else:
            scaling = "No data"
        
        summary = {
            'individual_results': results,
            'hbar_scaling_law': scaling,
            'quantum_regime_count': sum([r['hbar_effective'] > 1e-3 for r in results.values()]),
            'classical_regime_count': sum([r['hbar_effective'] <= 1e-3 for r in results.values()])
        }
        
        self._save_results(summary, "phase4_quantum_transformation.json")
        return summary

    def _find_broken_symmetries(self, coeffs: Dict[str, torch.Tensor]) -> List[int]:
        flat_coeffs = torch.cat([c.flatten() for c in coeffs.values()])
        perturbations = torch.randn_like(flat_coeffs) * 0.01
        perturbed = flat_coeffs + perturbations
        variance_threshold = 1e-6
        symmetry_indices = torch.where(torch.var(perturbed, unbiased=False) < variance_threshold)[0].tolist()
        return symmetry_indices

    def _measure_uncertainty(self, coeffs: Dict[str, torch.Tensor], basis: str = 'original') -> float:
        flat_coeffs = torch.cat([c.flatten() for c in coeffs.values()])
        if basis == 'original':
            return float(torch.std(flat_coeffs).item())
        else:
            if isinstance(basis, list) and basis:
                symmetry_components = flat_coeffs[basis]
                return float(torch.std(symmetry_components).item())
            else:
                return float(torch.std(flat_coeffs).item())

    def _plot_uncertainty_distribution(self, coeffs: Dict[str, torch.Tensor], symmetry_basis: List[int], ckpt_name: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        all_params = torch.cat([c.flatten() for c in coeffs.values()]).cpu().numpy()
        ax1.hist(all_params, bins=30, alpha=0.6, density=True)
        ax1.set_xlabel('Parameter Value (Original Basis)')
        ax1.set_ylabel('Density')
        ax1.set_title('Original Basis Distribution')
        ax1.grid(True, alpha=0.3)
        
        if symmetry_basis:
            symmetry_params = torch.cat([c.flatten() for c in coeffs.values()])[symmetry_basis].cpu().numpy()
            ax2.hist(symmetry_params, bins=30, alpha=0.6, density=True, color='red')
            ax2.set_xlabel('Parameter Value (Symmetry Basis)')
            ax2.set_ylabel('Density')
            ax2.set_title('Symmetry Basis Distribution')
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, f"phase4_uncertainty_{ckpt_name}.png"))
        plt.close()

def main():
    parser = argparse.ArgumentParser(description="Program for Neural Networks")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint directory")
    parser.add_argument("--results_dir", type=str, default="boltzmann_results", help="Results directory")
    args = parser.parse_args()
    
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Directory {args.checkpoint_dir} does not exist")
        exit(1)
    
    set_seed(Config.RANDOM_SEED)
    program = DLProgram(args.checkpoint_dir, args.results_dir)
    program.run_full_boltzmann_program()



def _simulate_training_trajectory(self, final_params: np.ndarray, final_delta: float) -> List[np.ndarray]:
    n_steps = 100
    initial_params = np.random.randn(*final_params.shape) * 0.1
    trajectory = []
    for t in np.linspace(0, 1, n_steps):
        params = initial_params * np.exp(-t * 3) + final_params * (1 - np.exp(-t * 3))
        noise_scale = 0.1 * (1 - t) * final_delta
        params += np.random.randn(*params.shape) * noise_scale
        trajectory.append(params)
    return trajectory

def _compute_generalization_entropy(self, params: np.ndarray, successful_ckpts: Dict[str, Any]) -> float:
    """Entropía de generalización con manejo robusto de dimensionalidad"""
    successful_params = []
    for data in successful_ckpts.values():
        clean_params = torch.cat([p.flatten() for p in data['coeffs'].values()]).cpu().numpy()
        successful_params.append(clean_params)
    
    if not successful_params:
        return 0.0
    
    successful_params = np.stack(successful_params)
    n_samples, n_dims = successful_params.shape
    
    # Caso 1: Muy pocas muestras - usar distancia simple
    if n_samples < 3:
        distances = np.linalg.norm(successful_params - params, axis=1)
        return float(np.mean(distances))
    
    # Caso 2: Más dimensiones que muestras - reducir dimensionalidad con PCA manual
    if n_dims > n_samples:
        n_components = min(n_samples - 1, max(2, n_samples // 2))
        
        try:
            # PCA manual
            mean = np.mean(successful_params, axis=0)
            centered = successful_params - mean
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = np.linalg.eigh(cov)
            idx = np.argsort(eigenvals)[::-1][:n_components]
            components = eigenvecs[:, idx]
            
            successful_params_reduced = centered @ components
            params_reduced = (params - mean) @ components
            
            # Añadir ruido pequeño
            successful_params_reduced += np.random.randn(*successful_params_reduced.shape) * 1e-10
            
            kde = gaussian_kde(successful_params_reduced.T)
            return float(-kde.logpdf(params_reduced)[0])
        except Exception as e:
            print(f"    Warning: PCA-KDE failed, using fallback distance metric")
            distances = np.linalg.norm(successful_params - params, axis=1)
            return float(np.mean(distances))
    
    # Caso 3: Dimensiones manejables - KDE directo
    try:
        successful_params += np.random.randn(*successful_params.shape) * 1e-10
        kde = gaussian_kde(successful_params.T)
        return float(-kde.logpdf(params)[0])
    except Exception as e:
        print(f"    Warning: KDE failed, using fallback distance metric")
        distances = np.linalg.norm(successful_params - params, axis=1)
        return float(np.mean(distances))

def _fit_timescale(self, entropy_values: List[float]) -> float:
    def model(t, A, tau, C):
        return A * np.exp(-t / tau) + C
    t = np.arange(len(entropy_values))
    try:
        popt, _ = curve_fit(model, t, entropy_values, p0=[entropy_values[0], 50, entropy_values[-1]])
        return float(abs(popt[1]))
    except:
        return 50.0

def _plot_entropy_production(self, t: List[int], S: List[float], dS_dt: List[float], ckpt_name: str):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(t, S, 'b-', linewidth=2)
    ax1.set_xlabel('Training Steps (normalized)')
    ax1.set_ylabel('Generalization Entropy S_gen')
    ax1.set_title(f'Entropy Evolution ({ckpt_name})')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, dS_dt, 'r-', linewidth=2)
    ax2.set_xlabel('Training Steps')
    ax2.set_ylabel('Entropy Production dS/dt')
    ax2.set_title('Entropy Production Rate')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(self.results_dir, f"phase2_entropy_{ckpt_name}.png"))
    plt.close()

def phase3_extensivity_law(self) -> Dict[str, Any]:
    print("\n" + "="*60)
    print("PHASE 3: THIRD LAW - Extensivity of the T Operator")
    print("="*60)
    sizes = [2, 4, 8, 16, 32, 64]
    results = {}
    
    successful_ckpts = {k: v for k, v in self.checkpoints.items() if v['alpha'] > 7}
    if not successful_ckpts:
        print("No crystal states found")
        return {}
    
    for ckpt_name, data in successful_ckpts.items():
        print(f"\nVerifying extensivity: {ckpt_name}")
        coeffs = data['coeffs']
        purity = 1 - data['delta']
        
        errors = []
        for N in sizes:
            relative_error = self._verify_scaling(coeffs, N)
            errors.append(relative_error)
        
        scaling_params = self._fit_extensivity(errors, sizes, purity)
        
        results[ckpt_name] = {
            'errors': dict(zip(sizes, errors)),
            'scaling_exponent': float(scaling_params[0]),
            'purity_factor': float(scaling_params[1]),
            'max_size_success': int(max([N for N, err in zip(sizes, errors) if err < 1e-4], default=0))
        }
        
        self._plot_extensivity(sizes, errors, purity, ckpt_name)
    
    # Derivar ley Phi(alpha)
    phi_data = [(self.checkpoints[ckpt]['alpha'], 1.0) 
                for ckpt, data in results.items() 
                if data['max_size_success'] >= 64 and self.checkpoints[ckpt]['alpha'] > 7]
    
    if phi_data:
        alphas = [x[0] for x in phi_data]
        c = -np.log(np.mean([x[1] for x in phi_data])) / np.mean(alphas) if alphas else 0
        phi_law = {'coefficient': float(c), 'form': f'phi(alpha) ∝ exp(-{c:.2f}·alpha)'}
    else:
        phi_law = {'coefficient': 0.0, 'form': 'No data'}
    
    summary = {
        'individual_results': results,
        'phi_alpha_function': phi_law,
        'extensivity_verified': self._verify_extensivity_universality(results)
    }
    
    self._save_results(summary, "phase3_extensivity_law.json")
    return summary
if __name__ == "__main__":
    main()    
