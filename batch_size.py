#!/usr/bin/env python3
"""
StrassenPlanckCalculator - Extended with Batch Size Thermodynamics

Calculates effective Planck constant (h_bar) from Strassen model checkpoints and
derives optimal batch size from quantum-thermodynamic principles.

Physics implemented:
- Batch size as inverse temperature: Var(∇L) ∝ 1/B
- Quantum noise quantization: ⟨(δ∇L)²⟩ ∼ ℏ_eff/B
- Saturation condition: B_sat ≈ ℏ_eff/Δ_struct
- Optimal batch size: B_opt ≈ ℏ_eff/‖∇L‖²_det
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
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path


@dataclass(frozen=True)
class Configuration:
    BATCH_SIZE: int = 32
    HIDDEN_SLOTS: int = 8
    TARGET_SLOTS: int = 7
    MATRIX_SIZE: int = 2
    INPUT_DIM: int = 4
    OUTPUT_DIM: int = 4
    WEIGHT_DECAY: float = 1e-4
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 3000
    RANDOM_SEED: int = 42
    DISCRETIZATION_MARGIN: float = 0.1
    OPTIMAL_DELTA_THRESHOLD: float = 0.01
    INDUSTRIAL_DELTA_THRESHOLD: float = 0.1
    PLANCK_SI: float = 1.054571817e-34
    SPEED_OF_LIGHT: float = 299792458
    GRAVITATIONAL_CONSTANT: float = 6.67430e-11
    SOLAR_MASS: float = 1.98847e30
    BOLTZMANN_CONSTANT: float = 1.380649e-23
    BATCH_SIZE_RANGE: List[int] = field(default_factory=lambda: [8, 16, 24, 32, 64, 96, 128, 256, 512])
    GRADIENT_ESTIMATION_BATCHES: int = 10
    STRUCTURAL_GAP_EPSILON: float = 1e-6
    NOISE_LEVELS: List[float] = field(default_factory=lambda: [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])
    RESILIENCE_TRIALS: int = 5
    RESILIENCE_EPOCHS: int = 30
    GAUGE_SAMPLES: int = 50
    KAPPA_BATCHES: int = 5
    LC_PERCENTILE: float = 0.95
    ULTRA_STRONG_WEIGHTS: Tuple[float, float, float, float] = (0.6, 0.25, 0.1, 0.05)
    STRONG_WEIGHTS: Tuple[float, float, float, float] = (0.5, 0.3, 0.15, 0.05)
    WEAK_WEIGHTS: Tuple[float, float, float, float] = (0.25, 0.25, 0.25, 0.25)
    ULTRA_STRONG_LAMBDA: float = 1e30
    STRONG_LAMBDA: float = 1e10
    EFFECTIVE_MODES: int = 31
    OUTPUT_DIRECTORY: str = "planck_analysis_reports"


CONFIG = Configuration()


def set_random_seed(seed: int = CONFIG.RANDOM_SEED) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


class BilinearStrassenModel(nn.Module):
    def __init__(self, config: Configuration = CONFIG):
        super().__init__()
        self.config = config
        self.U = nn.Linear(config.INPUT_DIM, config.HIDDEN_SLOTS, bias=False)
        self.V = nn.Linear(config.INPUT_DIM, config.HIDDEN_SLOTS, bias=False)
        self.W = nn.Linear(config.HIDDEN_SLOTS, config.OUTPUT_DIM, bias=False)
        nn.init.xavier_uniform_(self.U.weight)
        self.V.weight.data = self.U.weight.data.clone()
        nn.init.xavier_uniform_(self.W.weight)

    def forward(self, a, b):
        return self.W(self.U(a) * self.V(b))

    def get_coefficients(self):
        return {'U': self.U.weight.data, 'V': self.V.weight.data, 'W': self.W.weight.data}

    def compute_lambda_effective(self):
        with torch.no_grad():
            total = torch.norm(self.U.weight) + torch.norm(self.V.weight) + torch.norm(self.W.weight)
            return 1.0 / (total.item() ** 2 + 1e-10)


class CheckpointMigrator(ABC):
    @abstractmethod
    def can_migrate(self, state_dict):
        pass

    @abstractmethod
    def migrate(self, state_dict):
        pass


class CustomFormatMigrator(CheckpointMigrator):
    def can_migrate(self, state_dict):
        return 'U' in state_dict and isinstance(state_dict['U'], torch.Tensor)

    def migrate(self, state_dict):
        if state_dict['U'].shape == (7, 4):
            u, v, w = torch.zeros(8, 4), torch.zeros(8, 4), torch.zeros(4, 8)
            u[:7], v[:7], w[:, :7] = state_dict['U'], state_dict['V'], state_dict['W']
            return {'U.weight': u, 'V.weight': v, 'W.weight': w}
        return {'U.weight': state_dict['U'], 'V.weight': state_dict['V'], 'W.weight': state_dict['W']}


class StandardFormatMigrator(CheckpointMigrator):
    def can_migrate(self, state_dict):
        return any(k.endswith('.weight') for k in state_dict.keys())

    def migrate(self, state_dict):
        result = {k: state_dict[k] for k in ['U.weight', 'V.weight', 'W.weight'] if k in state_dict}
        return result if len(result) == 3 else None


class CheckpointMigrationManager:
    def __init__(self):
        self.strategies = [CustomFormatMigrator(), StandardFormatMigrator()]

    def migrate_checkpoint(self, path, device='cpu'):
        try:
            data = torch.load(path, map_location=device, weights_only=False)
            state_dict = data.get('state_dict', data.get('model_state_dict', data)) if isinstance(data, dict) else (data.state_dict() if hasattr(data, 'state_dict') else data)
            for strategy in self.strategies:
                if strategy.can_migrate(state_dict):
                    return strategy.migrate(state_dict)
        except Exception as e:
            print(f"Migration error: {e}")
        return None


class StrassenDataGenerator:
    @staticmethod
    def generate_batch(batch_size=CONFIG.BATCH_SIZE, config=CONFIG):
        A = torch.randn(batch_size, config.MATRIX_SIZE, config.MATRIX_SIZE)
        B = torch.randn(batch_size, config.MATRIX_SIZE, config.MATRIX_SIZE)
        C = torch.bmm(A, B)
        return A.reshape(batch_size, config.INPUT_DIM), B.reshape(batch_size, config.INPUT_DIM), C.reshape(batch_size, config.OUTPUT_DIM)


class CrystallographyMetrics:
    @staticmethod
    def compute_kappa(model, num_batches=CONFIG.KAPPA_BATCHES, config=CONFIG):
        model.eval()
        grads = []
        for _ in range(num_batches):
            A, B, C = StrassenDataGenerator.generate_batch(config.BATCH_SIZE, config)
            loss = nn.functional.mse_loss(model(A, B), C)
            grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            grads.append(torch.cat([g.flatten() for g in grad]))
        if len(grads) < 2:
            return float('inf')
        try:
            return torch.linalg.cond(torch.cov(torch.stack(grads).T)).item()
        except:
            return float('inf')

    @staticmethod
    def compute_discretization_margin(coeffs):
        return max((t - t.round()).abs().max().item() for t in coeffs.values())

    @staticmethod
    def compute_local_complexity(model, config=CONFIG):
        params = torch.cat([p.flatten() for p in model.parameters()])
        with torch.no_grad():
            perc = torch.quantile(torch.abs(params), config.LC_PERCENTILE)
            active = (torch.abs(params) > 0.01 * perc).sum()
            return (active.float() / len(params)).item()


class PlanckConstantCalculator:
    def __init__(self, metrics, training_metrics, config=CONFIG):
        self.metrics = metrics
        self.training = training_metrics
        self.config = config
        self.lambda_val = metrics.get('lambda_effective', 0.5)
        self.delta = metrics.get('delta', 1.0)
        self.mse = training_metrics.get('mse', 1.0)
        self.val_acc = training_metrics.get('val_acc', 0.0)

    def calculate_all(self):
        h_bar_unc = 2.0 * self.delta ** 2 * self.lambda_val
        omega = np.sqrt(self.lambda_val) if self.lambda_val > 0 else 1.0
        period = 2.0 * np.pi / omega
        T, V = self.mse, self.lambda_val * self.delta ** 2
        h_bar_act = abs(T - V) * period
        h_bar_cond = (1.0 / (self.val_acc / self.mse)) if self.mse > 0 and self.val_acc > 0 else 0.0
        h_bar_inf = ((T + V) / np.log2(self.config.EFFECTIVE_MODES)) * period

        weights = (self.config.ULTRA_STRONG_WEIGHTS if self.lambda_val > self.config.ULTRA_STRONG_LAMBDA 
                   else self.config.STRONG_WEIGHTS if self.lambda_val > self.config.STRONG_LAMBDA 
                   else self.config.WEAK_WEIGHTS)

        h_bar = sum(w * v for w, v in zip(weights, [h_bar_unc, h_bar_act, h_bar_cond, h_bar_inf])) / sum(weights)

        return {
            'h_bar': {'value': float(h_bar), 'methods': {'uncertainty': h_bar_unc, 'action': h_bar_act}},
            'inputs': {'lambda': self.lambda_val, 'delta': self.delta, 'mse': self.mse}
        }


class BatchSizeThermodynamics:
    def __init__(self, model, h_bar, delta_struct, config=CONFIG):
        self.model = model
        self.h_bar = h_bar
        self.delta_struct = max(delta_struct, config.STRUCTURAL_GAP_EPSILON)
        self.config = config

    def analyze_batch_size_spectrum(self):
        stats = {}
        for bs in self.config.BATCH_SIZE_RANGE:
            stats[bs] = self._measure_gradients(bs)

        B_sat = self.h_bar / self.delta_struct
        largest = max(self.config.BATCH_SIZE_RANGE)
        grad_det_sq = stats[largest]['grad_rms'] ** 2 if largest in stats else 1e-6
        B_opt = self.h_bar / grad_det_sq if grad_det_sq > 0 else float('inf')

        return {
            'gradient_statistics': stats,
            'theoretical_predictions': {
                'B_saturation': B_sat,
                'B_opt_practical': B_opt,
                'h_bar': self.h_bar,
                'delta_struct': self.delta_struct
            },
            'empirical_optimal': {'optimal_gradient_batch': min(stats.keys(), key=lambda k: stats[k]['grad_variance'])},
            'consistency_check': {'regime': 'PARTIAL_GROKKING' if 0.01 <= self.delta_struct < 0.1 else 'AMORPHOUS' if self.delta_struct >= 0.1 else 'PURE_CRYSTAL'}
        }

    def _measure_gradients(self, batch_size):
        self.model.eval()
        grads = []
        for _ in range(self.config.GRADIENT_ESTIMATION_BATCHES):
            A, B, C = StrassenDataGenerator.generate_batch(batch_size, self.config)
            loss = nn.functional.mse_loss(self.model(A, B), C)
            grad = torch.autograd.grad(loss, self.model.parameters(), create_graph=False)
            grads.append(torch.cat([g.flatten() for g in grad]))

        grad_tensor = torch.stack(grads)
        return {
            'grad_variance': grad_tensor.var(dim=0).mean().item(),
            'grad_rms': torch.sqrt((grad_tensor.mean(dim=0) ** 2).mean()).item()
        }


class StrassenCheckpointLoader:
    def __init__(self, config=CONFIG):
        self.config = config
        self.migrator = CheckpointMigrationManager()

    def load(self, path, device='cpu'):
        model = BilinearStrassenModel(self.config).to(device)
        try:
            data = torch.load(path, map_location=device, weights_only=False)
            state_dict = data.get('state_dict', data.get('model_state_dict', data)) if isinstance(data, dict) else data
            model.load_state_dict(state_dict)
            return model
        except:
            migrated = self.migrator.migrate_checkpoint(path, device)
            if migrated:
                model.load_state_dict(migrated)
                return model
        return None

    def extract_training_metrics(self, path):
        try:
            data = torch.load(path, map_location='cpu', weights_only=False)
            if isinstance(data, dict):
                m = data.get('metrics', {})
                return {'mse': m.get('val_mse', m.get('mse', 1.0)), 'val_acc': data.get('val_acc', m.get('val_acc', 0.0))}
        except:
            pass
        return {'mse': 1.0, 'val_acc': 0.0}


class StrassenPlanckAnalyzer:
    def __init__(self, config=CONFIG):
        self.config = config
        self.loader = StrassenCheckpointLoader(config)

    def analyze_checkpoint(self, path, device='cpu'):
        print(f"Analyzing: {os.path.basename(path)}")
        model = self.loader.load(path, device)
        if not model:
            raise ValueError(f"Failed to load: {path}")

        training = self.loader.extract_training_metrics(path)
        cryst = {
            'kappa': CrystallographyMetrics.compute_kappa(model, config=self.config),
            'delta': CrystallographyMetrics.compute_discretization_margin(model.get_coefficients()),
            'lambda_effective': model.compute_lambda_effective()
        }

        planck = PlanckConstantCalculator(cryst, training, self.config).calculate_all()
        h_bar = planck['h_bar']['value']

        batch = BatchSizeThermodynamics(model, h_bar, cryst['delta'], self.config).analyze_batch_size_spectrum()

        return {
            'metadata': {'checkpoint': path, 'timestamp': datetime.now().isoformat()},
            'crystallography': {'metrics': cryst},
            'planck_physics': planck,
            'batch_thermodynamics': batch,
            'training_info': training
        }

    def analyze_directory(self, directory, device='cpu', pattern='*.pt'):
        path = Path(directory)
        if not path.exists():
            raise ValueError(f"Directory does not exist: {directory}")

        checkpoints = sorted(path.glob(pattern))
        results = []
        for ckpt in checkpoints:
            try:
                results.append(self.analyze_checkpoint(str(ckpt), device))
                r = results[-1]
                print(f"  h_bar: {r['planck_physics']['h_bar']['value']:.6e}")
                print(f"  B_opt: {r['batch_thermodynamics']['theoretical_predictions']['B_opt_practical']:.1f}")
            except Exception as e:
                print(f"  Error: {e}")
        return results


def main():
    parser = argparse.ArgumentParser(description='Calculate Planck constant and optimal batch size from Strassen checkpoints')
    parser.add_argument('path', nargs='?', default="checkpoints", help='Path to checkpoint or directory')
    parser.add_argument('-o', '--output', default="output_batch_size_results", help='Output directory')
    parser.add_argument('-d', '--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device')
    args = parser.parse_args()

    set_random_seed()
    config = Configuration(OUTPUT_DIRECTORY=args.output)
    analyzer = StrassenPlanckAnalyzer(config)

    path = Path(args.path)
    if path.is_file():
        report = analyzer.analyze_checkpoint(str(path), args.device)
        print(json.dumps(report, indent=2, default=str))
    elif path.is_dir():
        results = analyzer.analyze_directory(str(path), args.device)
        print(f"\nProcessed {len(results)} checkpoints")

        # Save aggregate
        os.makedirs(args.output, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        with open(f"{args.output}/aggregate_{timestamp}.json", 'w') as f:
            json.dump({
                'count': len(results),
                'h_bar_mean': np.mean([r['planck_physics']['h_bar']['value'] for r in results]),
                'B_opt_mean': np.mean([r['batch_thermodynamics']['theoretical_predictions']['B_opt_practical'] for r in results])
            }, f, indent=2)


if __name__ == "__main__":
    exit(main())