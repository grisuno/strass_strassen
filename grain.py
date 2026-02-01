#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import json
import os
import argparse
import time
import signal
import sys
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Protocol, runtime_checkable, Callable
from pathlib import Path
import glob
from dataclasses import dataclass, field
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches


@dataclass(frozen=True)
class StrassenConfig:
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    TARGET_SLOTS: int = 7
    
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 1e-3
    WEIGHT_DECAY: float = 1e-4
    
    MAX_EPOCHS: int = 10000
    PATIENCE: int = 500
    
    DISCRETIZATION_MARGIN: float = 0.1
    PRUNING_THRESHOLD: float = 0.5
    
    CHECKPOINT_INTERVAL_MINUTES: float = 5.0
    CHECKPOINT_DIR: str = 'checkpoints'
    OUTPUT_DIR: str = 'grain_boundary_analysis'
    
    GRAIN_BOUNDARY_LAYERS: int = 3
    
    ENTROPY_BINS: int = 50
    TEMPERATURE_WINDOW: int = 100
    
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'
    
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    RANDOM_SEED: int = 42
    
    VERBOSE: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


@runtime_checkable
class IModel(Protocol):
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor: ...
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...


@runtime_checkable
class IGrainBoundaryDetector(Protocol):
    def detect(self, model: IModel, pruning_level: float) -> Dict[str, Any]: ...


@runtime_checkable
class ILayerAnalyzer(Protocol):
    def analyze_layer(self, weights: torch.Tensor, layer_name: str) -> Dict[str, float]: ...


@runtime_checkable
class IDislocationCalculator(Protocol):
    def calculate(self, layer_deltas: Dict[str, float]) -> Dict[str, Any]: ...


@runtime_checkable
class IDomainFragmentationAnalyzer(Protocol):
    def analyze(self, model: IModel, pruning_level: float) -> Dict[str, Any]: ...


@runtime_checkable
class ICheckpointManager(Protocol):
    def save(self, model: IModel, epoch: int, metrics: Dict[str, Any], path: str) -> None: ...
    def load(self, path: str) -> Dict[str, Any]: ...


@runtime_checkable
class ITrainingMonitor(Protocol):
    def update(self, epoch: int, metrics: Dict[str, Any]) -> None: ...
    def should_checkpoint(self) -> bool: ...


class BilinearStrassenModel(nn.Module):
    def __init__(self, hidden_dim: int = StrassenConfig.HIDDEN_DIM, matrix_size: int = StrassenConfig.MATRIX_SIZE):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.matrix_size = matrix_size
        input_dim = matrix_size * matrix_size
        
        self.U = nn.Linear(input_dim, hidden_dim, bias=False)
        self.V = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, input_dim, bias=False)
        self._initialize_symmetric()
    
    def _initialize_symmetric(self):
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


class LayerAnalyzer:
    def analyze_layer(self, weights: torch.Tensor, layer_name: str) -> Dict[str, float]:
        rounded = torch.round(weights)
        deltas = torch.abs(weights - rounded)
        
        delta_max = torch.max(deltas).item()
        delta_mean = torch.mean(deltas).item()
        delta_std = torch.std(deltas).item()
        delta_var = torch.var(deltas).item()
        
        integer_ratio = torch.mean((deltas < 0.1).float()).item()
        
        return {
            'layer_name': layer_name,
            'delta_max': delta_max,
            'delta_mean': delta_mean,
            'delta_std': delta_std,
            'delta_var': delta_var,
            'integer_ratio': integer_ratio,
            'num_parameters': weights.numel()
        }


class GrainBoundaryDetector:
    def __init__(self, config: StrassenConfig = StrassenConfig()):
        self.config = config
        self.layer_analyzer = LayerAnalyzer()
    
    def detect(self, model: IModel, pruning_level: float) -> Dict[str, Any]:
        original_state = {name: param.clone() for name, param in model.named_parameters()}
        
        self._prune_model(model, pruning_level)
        
        coeffs = model.get_coefficients()
        
        layer_analysis = {}
        for name, weights in coeffs.items():
            layer_analysis[name] = self.layer_analyzer.analyze_layer(weights, name)
        
        deltas = {name: analysis['delta_mean'] for name, analysis in layer_analysis.items()}
        dislocation = self._calculate_dislocation(deltas)
        
        fragmentation = self._analyze_fragmentation(layer_analysis)
        
        model.load_state_dict(original_state)
        
        return {
            'pruning_level': pruning_level,
            'layer_analysis': layer_analysis,
            'dislocation': dislocation,
            'fragmentation': fragmentation,
            'is_fragmented': dislocation['variance_across_layers'] > dislocation['mean_delta'] * 0.5,
            'grain_boundary_severity': dislocation['grain_boundary_metric']
        }
    
    def _prune_model(self, model: IModel, sparsity: float):
        with torch.no_grad():
            for param in model.parameters():
                flat = param.flatten()
                k = int(sparsity * flat.numel())
                if k > 0:
                    threshold = torch.topk(torch.abs(flat), k, largest=False).values[-1]
                    param[torch.abs(param) < threshold] = 0
    
    def _calculate_dislocation(self, layer_deltas: Dict[str, float]) -> Dict[str, float]:
        delta_values = list(layer_deltas.values())
        
        if len(delta_values) == 0:
            return {
                'mean_delta': 0.0,
                'variance_across_layers': 0.0,
                'std_across_layers': 0.0,
                'max_delta': 0.0,
                'min_delta': 0.0,
                'delta_range': 0.0,
                'grain_boundary_metric': 0.0
            }
        
        mean_delta = np.mean(delta_values)
        variance_delta = np.var(delta_values)
        std_delta = np.std(delta_values)
        max_delta = np.max(delta_values)
        min_delta = np.min(delta_values)
        delta_range = max_delta - min_delta
        
        grain_boundary_metric = variance_delta / (mean_delta + 1e-10) if mean_delta > 0 else float('inf')
        
        return {
            'mean_delta': float(mean_delta),
            'variance_across_layers': float(variance_delta),
            'std_across_layers': float(std_delta),
            'max_delta': float(max_delta),
            'min_delta': float(min_delta),
            'delta_range': float(delta_range),
            'grain_boundary_metric': float(grain_boundary_metric)
        }
    
    def _analyze_fragmentation(self, layer_analysis: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        variances = [analysis['delta_var'] for analysis in layer_analysis.values()]
        means = [analysis['delta_mean'] for analysis in layer_analysis.values()]
        
        coefficient_of_variation = np.std(variances) / (np.mean(variances) + 1e-10) if np.mean(variances) > 0 else 0
        
        fragmentation_score = np.std(means) / (np.mean(means) + 1e-10) if np.mean(means) > 0 else 0
        
        return {
            'coefficient_of_variation': float(coefficient_of_variation),
            'fragmentation_score': float(fragmentation_score),
            'layer_variances': {name: analysis['delta_var'] for name, analysis in layer_analysis.items()},
            'layer_means': {name: analysis['delta_mean'] for name, analysis in layer_analysis.items()}
        }


class DomainFragmentationAnalyzer:
    def __init__(self, config: StrassenConfig = StrassenConfig()):
        self.config = config
        self.detector = GrainBoundaryDetector(config)
    
    def analyze(self, model: IModel, pruning_level: float) -> Dict[str, Any]:
        grain_result = self.detector.detect(model, pruning_level)
        
        layer_analysis = grain_result['layer_analysis']
        
        coordination_loss = self._calculate_coordination_loss(layer_analysis)
        
        domain_count = self._estimate_domain_count(layer_analysis)
        
        coherence_length = self._calculate_coherence_length(layer_analysis)
        
        return {
            'pruning_level': pruning_level,
            'grain_boundary': grain_result,
            'coordination_loss': coordination_loss,
            'estimated_domains': domain_count,
            'coherence_length': coherence_length,
            'is_policrystal': domain_count > 1,
            'global_order_destroyed': coordination_loss > 0.5
        }
    
    def _calculate_coordination_loss(self, layer_analysis: Dict[str, Dict[str, float]]) -> float:
        integer_ratios = [analysis['integer_ratio'] for analysis in layer_analysis.values()]
        
        if len(integer_ratios) < 2:
            return 0.0
        
        max_ratio = max(integer_ratios)
        min_ratio = min(integer_ratios)
        
        coordination_loss = max_ratio - min_ratio
        
        return float(coordination_loss)
    
    def _estimate_domain_count(self, layer_analysis: Dict[str, Dict[str, float]]) -> int:
        deltas = [analysis['delta_mean'] for analysis in layer_analysis.values()]
        
        if len(deltas) < 2:
            return 1
        
        threshold = np.mean(deltas) * 0.2
        
        clusters = 1
        for i in range(1, len(deltas)):
            if abs(deltas[i] - deltas[i-1]) > threshold:
                clusters += 1
        
        return clusters
    
    def _calculate_coherence_length(self, layer_analysis: Dict[str, Dict[str, float]]) -> float:
        num_params = sum(analysis['num_parameters'] for analysis in layer_analysis.values())
        total_delta_var = sum(analysis['delta_var'] for analysis in layer_analysis.values())
        
        if total_delta_var < 1e-10:
            return float(num_params)
        
        coherence = num_params / (1 + total_delta_var)
        
        return float(coherence)


class CheckpointManager:
    def __init__(self, config: StrassenConfig = StrassenConfig()):
        self.config = config
        self.last_checkpoint_time = time.time()
    
    def save(self, model: IModel, epoch: int, metrics: Dict[str, Any], path: str):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'metrics': metrics,
            'timestamp': datetime.now().isoformat(),
            'config': self.config.to_dict()
        }
        
        torch.save(checkpoint, path)
        
        latest_path = os.path.join(os.path.dirname(path), 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        self.last_checkpoint_time = time.time()
    
    def load(self, path: str) -> Dict[str, Any]:
        return torch.load(path, map_location=self.config.DEVICE, weights_only=False)
    
    def should_save(self) -> bool:
        elapsed = time.time() - self.last_checkpoint_time
        return elapsed >= (self.config.CHECKPOINT_INTERVAL_MINUTES * 60)


class TrainingMetricsTracker:
    def __init__(self, config: StrassenConfig = StrassenConfig()):
        self.config = config
        self.metrics_history = {
            'epochs': [],
            'loss': [],
            'accuracy': [],
            'delta': [],
            'kappa': [],
            'entropy': [],
            'specific_heat': [],
            'grain_boundary_metric': [],
            'fragmentation_score': [],
            'coordination_loss': []
        }
        self.gradient_history = deque(maxlen=config.TEMPERATURE_WINDOW)
        self.loss_history = deque(maxlen=config.TEMPERATURE_WINDOW)
    
    def update(self, epoch: int, loss: float, accuracy: float, model: IModel, grain_result: Optional[Dict] = None):
        self.metrics_history['epochs'].append(epoch)
        self.metrics_history['loss'].append(loss)
        self.metrics_history['accuracy'].append(accuracy)
        
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        
        delta = torch.max(torch.abs(all_weights - torch.round(all_weights))).item()
        self.metrics_history['delta'].append(delta)
        
        self.loss_history.append(loss)
        
        if grain_result:
            self.metrics_history['grain_boundary_metric'].append(
                grain_result.get('dislocation', {}).get('grain_boundary_metric', 0)
            )
            self.metrics_history['fragmentation_score'].append(
                grain_result.get('fragmentation', {}).get('fragmentation_score', 0)
            )
            self.metrics_history['coordination_loss'].append(
                grain_result.get('coordination_loss', 0)
            )
    
    def get_current_metrics(self) -> Dict[str, float]:
        return {
            'loss': self.metrics_history['loss'][-1] if self.metrics_history['loss'] else 0,
            'delta': self.metrics_history['delta'][-1] if self.metrics_history['delta'] else 1.0,
            'grain_boundary': self.metrics_history['grain_boundary_metric'][-1] if self.metrics_history['grain_boundary_metric'] else 0
        }
    
    def get_training_bar_string(self, epoch: int, total_epochs: int) -> str:
        metrics = self.get_current_metrics()
        
        bar_length = 40
        progress = epoch / total_epochs
        filled = int(bar_length * progress)
        bar = '=' * filled + '>' + '.' * (bar_length - filled - 1)
        
        status = (
            f"\rEpoch [{epoch}/{total_epochs}] [{bar}] "
            f"Loss: {metrics['loss']:.6f} | "
            f"Delta: {metrics['delta']:.6f} | "
            f"Grain: {metrics['grain_boundary']:.4f}"
        )
        
        return status


class StrassenTrainer:
    def __init__(self, config: StrassenConfig = StrassenConfig(), seed: int = None):
        self.config = config
        if seed is not None:
            self.config = dataclass.replace(config, RANDOM_SEED=seed)
        
        torch.manual_seed(self.config.RANDOM_SEED)
        np.random.seed(self.config.RANDOM_SEED)
        
        self.model = BilinearStrassenModel(
            hidden_dim=self.config.HIDDEN_DIM,
            matrix_size=self.config.MATRIX_SIZE
        ).to(self.config.DEVICE)
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=self.config.LEARNING_RATE,
            weight_decay=self.config.WEIGHT_DECAY
        )
        
        self.checkpoint_manager = CheckpointManager(self.config)
        self.metrics_tracker = TrainingMetricsTracker(self.config)
        self.grain_detector = GrainBoundaryDetector(self.config)
        self.fragmentation_analyzer = DomainFragmentationAnalyzer(self.config)
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.current_epoch = 0
        
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        print("\nInterrupted. Saving checkpoint...")
        self._save_checkpoint(interrupted=True)
        sys.exit(0)
    
    def _generate_batch(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = torch.randn(
            self.config.BATCH_SIZE,
            self.config.MATRIX_SIZE,
            self.config.MATRIX_SIZE,
            device=self.config.DEVICE
        )
        B = torch.randn(
            self.config.BATCH_SIZE,
            self.config.MATRIX_SIZE,
            self.config.MATRIX_SIZE,
            device=self.config.DEVICE
        )
        C = torch.bmm(A, B)
        
        return (
            A.reshape(self.config.BATCH_SIZE, self.config.MATRIX_SIZE * self.config.MATRIX_SIZE),
            B.reshape(self.config.BATCH_SIZE, self.config.MATRIX_SIZE * self.config.MATRIX_SIZE),
            C.reshape(self.config.BATCH_SIZE, self.config.MATRIX_SIZE * self.config.MATRIX_SIZE)
        )
    
    def _compute_accuracy(self, pred: torch.Tensor, target: torch.Tensor) -> float:
        with torch.no_grad():
            mse = torch.mean((pred - target) ** 2).item()
            return 1.0 / (1.0 + mse)
    
    def _save_checkpoint(self, interrupted: bool = False):
        metrics = self.metrics_tracker.get_current_metrics()
        metrics['interrupted'] = interrupted
        metrics['best_loss'] = self.best_loss
        metrics['current_epoch'] = self.current_epoch
        
        path = os.path.join(
            self.config.CHECKPOINT_DIR,
            f'bs{self.config.BATCH_SIZE}_seed{self.config.RANDOM_SEED}.pt'
        )
        
        self.checkpoint_manager.save(self.model, self.current_epoch, metrics, path)
    
    def train(self):
        print(f"Training Strassen model with seed {self.config.RANDOM_SEED}")
        print(f"Hidden dim: {self.config.HIDDEN_DIM}, Matrix size: {self.config.MATRIX_SIZE}")
        print(f"Device: {self.config.DEVICE}")
        print("=" * 80)
        
        for epoch in range(1, self.config.MAX_EPOCHS + 1):
            self.current_epoch = epoch
            
            self.model.train()
            self.optimizer.zero_grad()
            
            a, b, c_target = self._generate_batch()
            c_pred = self.model(a, b)
            
            loss = torch.mean((c_pred - c_target) ** 2)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            with torch.no_grad():
                accuracy = self._compute_accuracy(c_pred, c_target)
            
            grain_result = None
            if epoch % 100 == 0:
                grain_result = self.grain_detector.detect(self.model, self.config.PRUNING_THRESHOLD)
            
            self.metrics_tracker.update(epoch, loss.item(), accuracy, self.model, grain_result)
            
            if self.config.VERBOSE and epoch % 10 == 0:
                status = self.metrics_tracker.get_training_bar_string(epoch, self.config.MAX_EPOCHS)
                print(status, end='', flush=True)
            
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            if self.checkpoint_manager.should_save():
                self._save_checkpoint()
                if self.config.VERBOSE:
                    print(f"\nCheckpoint saved at epoch {epoch}")
            
            if self.patience_counter >= self.config.PATIENCE:
                if self.config.VERBOSE:
                    print(f"\nEarly stopping at epoch {epoch}")
                break
            
            if loss.item() < 1e-6 and grain_result and grain_result.get('is_fragmented', False) is False:
                if self.config.VERBOSE:
                    print(f"\nConverged at epoch {epoch}")
                break
        
        self._save_checkpoint()
        print("\nTraining complete.")
        
        return self.metrics_tracker.metrics_history


class GrainBoundaryAnalyzer:
    def __init__(self, checkpoint_path: str, config: StrassenConfig = StrassenConfig()):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        self.detector = GrainBoundaryDetector(config)
        self.fragmentation_analyzer = DomainFragmentationAnalyzer(config)
        
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
        
        self.model = BilinearStrassenModel(
            hidden_dim=self.config.HIDDEN_DIM,
            matrix_size=self.config.MATRIX_SIZE
        ).to(self.config.DEVICE)
        
        state_dict = self._migrate_checkpoint(self.checkpoint)
        if state_dict is not None:
            self.model.load_state_dict(state_dict)
        else:
            raise RuntimeError(f"Failed to migrate checkpoint: {self.checkpoint_path}")
        
        self.epoch = self.checkpoint.get('epoch', 'unknown')
    
    def _migrate_checkpoint(self, raw_data: Any) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(raw_data, dict):
            if 'state_dict' in raw_data:
                return self._migrate_dict(raw_data['state_dict'])
            elif 'model_state_dict' in raw_data:
                return self._migrate_dict(raw_data['model_state_dict'])
            else:
                return self._migrate_dict(raw_data)
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
    
    def analyze(self) -> Dict[str, Any]:
        pruning_levels = [0.0, 0.3, 0.5, 0.7, 0.9]
        
        grain_results = {}
        for level in pruning_levels:
            grain_results[level] = self.detector.detect(self.model, level)
        
        fragmentation_50 = self.fragmentation_analyzer.analyze(self.model, 0.5)
        
        dislocation_evolution = self._analyze_dislocation_evolution(grain_results)
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat()
            },
            'pruning_analysis': grain_results,
            'fragmentation_at_50': fragmentation_50,
            'dislocation_evolution': dislocation_evolution,
            'grain_boundary_detected': any(
                r.get('is_fragmented', False) for r in grain_results.values()
            ),
            'critical_pruning_level': self._find_critical_pruning_level(grain_results)
        }
        
        self._print_report(results)
        
        return results
    
    def _analyze_dislocation_evolution(self, grain_results: Dict[float, Dict]) -> Dict[str, Any]:
        pruning_levels = sorted(grain_results.keys())
        variances = [grain_results[p]['dislocation']['variance_across_layers'] for p in pruning_levels]
        means = [grain_results[p]['dislocation']['mean_delta'] for p in pruning_levels]
        
        variance_increase = variances[-1] - variances[0] if len(variances) > 1 else 0
        mean_change = means[-1] - means[0] if len(means) > 1 else 0
        
        critical_point = None
        for i in range(1, len(pruning_levels)):
            if variances[i] > means[i] * 0.5:
                critical_point = pruning_levels[i]
                break
        
        return {
            'variance_increase': float(variance_increase),
            'mean_change': float(mean_change),
            'variance_trajectory': [(p, v) for p, v in zip(pruning_levels, variances)],
            'mean_trajectory': [(p, m) for p, m in zip(pruning_levels, means)],
            'critical_pruning_point': critical_point,
            'dislocation_sharpness': variance_increase / (mean_change + 1e-10) if mean_change != 0 else float('inf')
        }
    
    def _find_critical_pruning_level(self, grain_results: Dict[float, Dict]) -> Optional[float]:
        for level in sorted(grain_results.keys()):
            if grain_results[level].get('is_fragmented', False):
                return level
        return None
    
    def _print_report(self, results: Dict):
        print("=" * 80)
        print("GRAIN BOUNDARY ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {results['metadata']['checkpoint_path']}")
        print(f"  Epoch: {results['metadata']['epoch']}")
        
        print(f"\n[DISLOCATION EVOLUTION]")
        de = results['dislocation_evolution']
        print(f"  Variance increase: {de['variance_increase']:.6f}")
        print(f"  Mean change: {de['mean_change']:.6f}")
        print(f"  Critical pruning point: {de['critical_pruning_point']}")
        print(f"  Dislocation sharpness: {de['dislocation_sharpness']:.6f}")
        
        print(f"\n[PRUNING ANALYSIS]")
        for level, analysis in sorted(results['pruning_analysis'].items()):
            print(f"  Pruning {level*100:.0f}%:")
            print(f"    Mean delta: {analysis['dislocation']['mean_delta']:.6f}")
            print(f"    Variance: {analysis['dislocation']['variance_across_layers']:.6e}")
            print(f"    Grain boundary metric: {analysis['dislocation']['grain_boundary_metric']:.6f}")
            print(f"    Is fragmented: {analysis['is_fragmented']}")
        
        print(f"\n[FRAGMENTATION AT 50%]")
        f50 = results['fragmentation_at_50']
        print(f"  Coordination loss: {f50['coordination_loss']:.6f}")
        print(f"  Estimated domains: {f50['estimated_domains']}")
        print(f"  Coherence length: {f50['coherence_length']:.6f}")
        print(f"  Is policrystal: {f50['is_policrystal']}")
        print(f"  Global order destroyed: {f50['global_order_destroyed']}")
        
        print(f"\n[CONCLUSION]")
        print(f"  Grain boundary detected: {results['grain_boundary_detected']}")
        print(f"  Critical pruning level: {results['critical_pruning_level']}")
        
        print("=" * 80)


class GrainBoundaryPipeline:
    def __init__(self, config: StrassenConfig = StrassenConfig()):
        self.config = config
    
    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer = GrainBoundaryAnalyzer(checkpoint_path, self.config)
        results = analyzer.analyze()
        
        base_name = Path(checkpoint_path).stem
        
        results_path = os.path.join(output_dir, f'{base_name}_grain.json')
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
        
        fragmented_count = sum(1 for r in all_results if r.get('grain_boundary_detected', False))
        
        critical_levels = [r.get('critical_pruning_level') for r in all_results if r.get('critical_pruning_level') is not None]
        
        summary = {
            'total_checkpoints_analyzed': len(all_results),
            'fragmented_checkpoints': fragmented_count,
            'fragmentation_rate': fragmented_count / len(all_results) if all_results else 0,
            'mean_critical_pruning_level': float(np.mean(critical_levels)) if critical_levels else None,
            'timestamp': datetime.now().isoformat(),
            'individual_results': all_results
        }
        
        summary_path = os.path.join(output_dir, 'grain_boundary_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._generate_text_report(summary, output_dir)
        
        print(f"\nSaved summary: {summary_path}")
    
    def _generate_text_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        report_path = os.path.join(output_dir, 'grain_boundary_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("GRAIN BOUNDARY ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total checkpoints analyzed: {summary['total_checkpoints_analyzed']}\n")
            f.write(f"Fragmented checkpoints: {summary['fragmented_checkpoints']}\n")
            f.write(f"Fragmentation rate: {summary['fragmentation_rate']:.2%}\n")
            if summary['mean_critical_pruning_level'] is not None:
                f.write(f"Mean critical pruning level: {summary['mean_critical_pruning_level']:.2%}\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("INDIVIDUAL CHECKPOINT ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            for i, r in enumerate(summary['individual_results'], 1):
                f.write(f"[{i}] {r['metadata']['checkpoint_path']}\n")
                f.write(f"    Epoch: {r['metadata']['epoch']}\n")
                f.write(f"    Grain boundary detected: {r['grain_boundary_detected']}\n")
                f.write(f"    Critical pruning level: {r.get('critical_pruning_level', 'N/A')}\n")
                
                de = r.get('dislocation_evolution', {})
                f.write(f"    Variance increase: {de.get('variance_increase', 0):.6e}\n")
                f.write(f"    Dislocation sharpness: {de.get('dislocation_sharpness', 0):.6f}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved text report: {report_path}")


def run_training(seed: int, config: StrassenConfig):
    trainer = StrassenTrainer(config, seed=seed)
    return trainer.train()


def run_analysis(checkpoint_dir: str, output_dir: str, n_latest: Optional[int], config: StrassenConfig):
    pipeline = GrainBoundaryPipeline(config)
    results = pipeline.process_directory(checkpoint_dir, n_latest, output_dir)
    if results:
        pipeline.generate_summary(results, output_dir)


def main():
    parser = argparse.ArgumentParser(
        description='Grain boundary analysis for Strassen algorithm crystallization'
    )
    parser.add_argument(
        '--mode',
        choices=['train', 'analyze', 'both'],
        default='analyze',
        help='Mode: train models, analyze checkpoints, or both'
    )
    parser.add_argument(
        '--checkpoint',
        nargs='?',
        default=None,
        help='Path to specific checkpoint file'
    )
    parser.add_argument(
        '--all',
        action='store_true',
        help='Process all checkpoints'
    )
    parser.add_argument(
        '--latest',
        type=int,
        default=None,
        help='Process N latest checkpoints'
    )
    parser.add_argument(
        '--dir',
        default='checkpoints',
        help='Checkpoint directory'
    )
    parser.add_argument(
        '--output',
        default='grain_boundary_analysis',
        help='Output directory'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for training'
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
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--max-epochs',
        type=int,
        default=10000,
        help='Maximum epochs'
    )
    parser.add_argument(
        '--pruning-threshold',
        type=float,
        default=0.5,
        help='Pruning threshold for grain boundary detection'
    )
    
    args = parser.parse_args()
    
    config = StrassenConfig(
        HIDDEN_DIM=args.hidden_dim,
        MATRIX_SIZE=args.matrix_size,
        BATCH_SIZE=args.batch_size,
        MAX_EPOCHS=args.max_epochs,
        PRUNING_THRESHOLD=args.pruning_threshold
    )
    
    if args.mode == 'train':
        run_training(args.seed if args.seed else config.RANDOM_SEED, config)
    elif args.mode == 'analyze':
        if args.checkpoint:
            if os.path.isfile(args.checkpoint):
                pipeline = GrainBoundaryPipeline(config)
                pipeline.process_checkpoint(args.checkpoint, args.output)
            else:
                print(f"Error: Checkpoint not found: {args.checkpoint}")
        else:
            n_to_process = args.latest if args.latest is not None else (None if args.all else 1)
            run_analysis(args.dir, args.output, n_to_process, config)
    elif args.mode == 'both':
        run_training(args.seed if args.seed else config.RANDOM_SEED, config)
        run_analysis(config.CHECKPOINT_DIR, args.output, None, config)


if __name__ == '__main__':
    main()
