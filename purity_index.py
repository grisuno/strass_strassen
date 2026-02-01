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
from scipy.stats import entropy
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class PurityConfig:
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    
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
    
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'
    
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'


@runtime_checkable
class IModel(Protocol):
    def get_coefficients(self) -> Dict[str, torch.Tensor]: ...


@runtime_checkable
class IPurityIndexCalculator(Protocol):
    def calculate(self, model: IModel) -> Dict[str, float]: ...


@runtime_checkable
class IEffectiveTemperatureCalculator(Protocol):
    def calculate(self, loss_history: List[float]) -> Dict[str, float]: ...


@runtime_checkable
class IPhaseClassifier(Protocol):
    def classify(self, alpha: float, temperature: float) -> str: ...


@runtime_checkable
class IPolycrystalAnalyzer(Protocol):
    def analyze_polycrystal(self, model: IModel, pruning_level: float) -> Dict[str, Any]: ...


@runtime_checkable
class IPurityComparator(Protocol):
    def compare(self, original: Dict[str, float], polycrystal: Dict[str, float]) -> Dict[str, Any]: ...


class BilinearModel(nn.Module):
    def __init__(self, hidden_dim: int = PurityConfig.HIDDEN_DIM, matrix_size: int = PurityConfig.MATRIX_SIZE):
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


class PurityIndexCalculator:
    def __init__(self, config: PurityConfig = PurityConfig()):
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
        if delta < 1e-10:
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
    def __init__(self, config: PurityConfig = PurityConfig()):
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
    def __init__(self, config: PurityConfig = PurityConfig()):
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
    
    def classify_polycrystal_state(self, original_alpha: float, original_temp: float, 
                                   poly_alpha: float, poly_temp: float) -> str:
        alpha_retention = poly_alpha / original_alpha if original_alpha > 0 else 0
        temp_ratio = poly_temp / original_temp if original_temp > 0 else float('inf')
        
        if alpha_retention > 0.8 and temp_ratio > 2.0:
            return 'polycrystal_with_residual_heat'
        elif alpha_retention > 0.5 and temp_ratio > 1.5:
            return 'fragmented_but_recognizable'
        elif alpha_retention > 0.5:
            return 'cold_fragmentation'
        elif temp_ratio > 2.0:
            return 'thermal_amorphization'
        else:
            return 'complete_amorphization'


class PolycrystalAnalyzer:
    def __init__(self, config: PurityConfig = PurityConfig()):
        self.config = config
        self.purity_calculator = PurityIndexCalculator(config)
        self.temperature_calculator = EffectiveTemperatureCalculator(config)
        self.phase_classifier = PhaseClassifier(config)
    
    def analyze_polycrystal(self, model: IModel, pruning_level: float, loss_history: List[float] = None) -> Dict[str, Any]:
        original_state = {name: param.clone() for name, param in model.named_parameters()}
        
        self._prune_model(model, pruning_level)
        
        purity = self.purity_calculator.calculate(model)
        
        temperature = self.temperature_calculator.calculate(loss_history if loss_history else [])
        
        phase = self.phase_classifier.classify(purity['global_alpha'], temperature['temperature'])
        
        model.load_state_dict(original_state)
        
        return {
            'pruning_level': pruning_level,
            'purity': purity,
            'temperature': temperature,
            'phase': phase,
            'is_polycrystal': 'polycrystal' in phase or 'fragmented' in phase
        }
    
    def _prune_model(self, model: IModel, sparsity: float):
        with torch.no_grad():
            for param in model.parameters():
                flat = param.flatten()
                k = int(sparsity * flat.numel())
                if k > 0:
                    threshold = torch.topk(torch.abs(flat), k, largest=False).values[-1]
                    param[torch.abs(param) < threshold] = 0


class PurityComparator:
    def __init__(self, config: PurityConfig = PurityConfig()):
        self.config = config
        self.phase_classifier = PhaseClassifier(config)
    
    def compare(self, original: Dict[str, float], polycrystal: Dict[str, float]) -> Dict[str, Any]:
        alpha_ratio = polycrystal.get('alpha', 0) / original.get('alpha', 1e-10)
        temp_ratio = polycrystal.get('temperature', 0) / (original.get('temperature', 1e-10) + 1e-10)
        
        alpha_retention = min(alpha_ratio, self.config.ALPHA_SATURATION)
        
        thermal_excess = max(0, temp_ratio - 1.0)
        
        intermediate_phase = self.phase_classifier.classify_polycrystal_state(
            original.get('alpha', 0),
            original.get('temperature', 0),
            polycrystal.get('alpha', 0),
            polycrystal.get('temperature', 0)
        )
        
        is_intermediate_phase = (
            self.config.ALPHA_THRESHOLD_GLASS < polycrystal.get('alpha', 0) < self.config.ALPHA_THRESHOLD_CRYSTAL
            or intermediate_phase in ['fragmented_but_recognizable', 'polycrystal_with_residual_heat']
        )
        
        return {
            'alpha_ratio': float(alpha_ratio),
            'alpha_retention': float(alpha_retention),
            'temperature_ratio': float(temp_ratio),
            'thermal_excess': float(thermal_excess),
            'intermediate_phase_detected': is_intermediate_phase,
            'intermediate_phase_type': intermediate_phase,
            'structural_memory_preserved': alpha_retention > 0.5,
            'thermal_damage': thermal_excess > 1.0
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


class PurityAnalyzer:
    def __init__(self, checkpoint_path: str, config: PurityConfig = PurityConfig()):
        self.config = config
        self.checkpoint_path = checkpoint_path
        
        self.purity_calculator = PurityIndexCalculator(config)
        self.temperature_calculator = EffectiveTemperatureCalculator(config)
        self.phase_classifier = PhaseClassifier(config)
        self.polycrystal_analyzer = PolycrystalAnalyzer(config)
        self.comparator = PurityComparator(config)
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
        self.loss_history = self.checkpoint.get('loss_history', [])
    
    def analyze(self) -> Dict[str, Any]:
        original_purity = self.purity_calculator.calculate(self.model)
        original_temperature = self.temperature_calculator.calculate(self.loss_history)
        original_phase = self.phase_classifier.classify(
            original_purity['global_alpha'],
            original_temperature['temperature']
        )
        
        polycrystal_analysis = {}
        for level in self.config.PRUNING_LEVELS:
            polycrystal_analysis[level] = self.polycrystal_analyzer.analyze_polycrystal(
                self.model, level, self.loss_history
            )
        
        level_50 = polycrystal_analysis.get(0.5, {})
        comparison = self.comparator.compare(
            {'alpha': original_purity['global_alpha'], 'temperature': original_temperature['temperature']},
            {'alpha': level_50.get('purity', {}).get('global_alpha', 0), 
             'temperature': level_50.get('temperature', {}).get('temperature', 0)}
        )
        
        phase_transition_detected = any(
            pa['phase'] != original_phase for pa in polycrystal_analysis.values()
        )
        
        results = {
            'metadata': {
                'checkpoint_path': self.checkpoint_path,
                'epoch': self.epoch,
                'timestamp': datetime.now().isoformat()
            },
            'original': {
                'purity': original_purity,
                'temperature': original_temperature,
                'phase': original_phase
            },
            'polycrystal_analysis': polycrystal_analysis,
            'comparison': comparison,
            'phase_transition_detected': phase_transition_detected,
            'intermediate_phase_exists': comparison['intermediate_phase_detected']
        }
        
        self._print_report(results)
        
        return results
    
    def _print_report(self, results: Dict):
        print("=" * 80)
        print("PURITY INDEX ANALYSIS REPORT")
        print("=" * 80)
        
        print(f"\n[METADATA]")
        print(f"  Checkpoint: {results['metadata']['checkpoint_path']}")
        print(f"  Epoch: {results['metadata']['epoch']}")
        
        print(f"\n[ORIGINAL STATE]")
        orig = results['original']
        print(f"  Alpha: {orig['purity']['global_alpha']:.6f}")
        print(f"  Delta: {orig['purity']['global_delta']:.6f}")
        print(f"  Temperature: {orig['temperature']['temperature']:.6e}")
        print(f"  Phase: {orig['phase']}")
        print(f"  Purity quality: {orig['purity']['purity_quality']}")
        print(f"  Is homogeneous: {orig['purity']['is_homogeneous']}")
        
        print(f"\n[LAYER ALPHAS]")
        for name, alpha in orig['purity']['layer_alphas'].items():
            print(f"  {name}: {alpha:.6f}")
        
        print(f"\n[POLYCRYSTAL ANALYSIS]")
        for level, analysis in sorted(results['polycrystal_analysis'].items()):
            print(f"  Pruning {level*100:.0f}%:")
            print(f"    Alpha: {analysis['purity']['global_alpha']:.6f}")
            print(f"    Temperature: {analysis['temperature']['temperature']:.6e}")
            print(f"    Phase: {analysis['phase']}")
            print(f"    Is polycrystal: {analysis['is_polycrystal']}")
        
        print(f"\n[COMPARISON]")
        comp = results['comparison']
        print(f"  Alpha retention: {comp['alpha_retention']:.2%}")
        print(f"  Temperature ratio: {comp['temperature_ratio']:.2f}x")
        print(f"  Thermal excess: {comp['thermal_excess']:.2f}")
        print(f"  Intermediate phase: {comp['intermediate_phase_detected']}")
        print(f"  Phase type: {comp['intermediate_phase_type']}")
        print(f"  Structural memory preserved: {comp['structural_memory_preserved']}")
        print(f"  Thermal damage: {comp['thermal_damage']}")
        
        print(f"\n[CONCLUSION]")
        print(f"  Phase transition detected: {results['phase_transition_detected']}")
        print(f"  Intermediate phase exists: {results['intermediate_phase_exists']}")
        
        print("=" * 80)


class PurityPipeline:
    def __init__(self, config: PurityConfig = PurityConfig()):
        self.config = config
    
    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        
        analyzer = PurityAnalyzer(checkpoint_path, self.config)
        results = analyzer.analyze()
        
        base_name = Path(checkpoint_path).stem
        
        results_path = os.path.join(output_dir, f'{base_name}_purity.json')
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
        
        intermediate_count = sum(1 for r in all_results if r.get('intermediate_phase_exists', False))
        transition_count = sum(1 for r in all_results if r.get('phase_transition_detected', False))
        
        alpha_retentions = []
        temp_ratios = []
        
        for r in all_results:
            comp = r.get('comparison', {})
            if 'alpha_retention' in comp:
                alpha_retentions.append(comp['alpha_retention'])
            if 'temperature_ratio' in comp:
                temp_ratios.append(comp['temperature_ratio'])
        
        summary = {
            'total_checkpoints_analyzed': len(all_results),
            'intermediate_phase_count': intermediate_count,
            'phase_transition_count': transition_count,
            'intermediate_phase_rate': intermediate_count / len(all_results) if all_results else 0,
            'mean_alpha_retention': float(np.mean(alpha_retentions)) if alpha_retentions else 0,
            'mean_temperature_ratio': float(np.mean(temp_ratios)) if temp_ratios else 0,
            'timestamp': datetime.now().isoformat(),
            'individual_results': all_results
        }
        
        summary_path = os.path.join(output_dir, 'purity_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        self._generate_text_report(summary, output_dir)
        
        print(f"\nSaved summary: {summary_path}")
    
    def _generate_text_report(self, summary: Dict[str, Any], output_dir: str) -> None:
        report_path = os.path.join(output_dir, 'purity_report.txt')
        
        with open(report_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("PURITY INDEX ANALYSIS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Total checkpoints analyzed: {summary['total_checkpoints_analyzed']}\n")
            f.write(f"Intermediate phase count: {summary['intermediate_phase_count']}\n")
            f.write(f"Phase transition count: {summary['phase_transition_count']}\n")
            f.write(f"Intermediate phase rate: {summary['intermediate_phase_rate']:.2%}\n")
            f.write(f"Mean alpha retention: {summary['mean_alpha_retention']:.2%}\n")
            f.write(f"Mean temperature ratio: {summary['mean_temperature_ratio']:.2f}x\n")
            f.write(f"Timestamp: {summary['timestamp']}\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("INDIVIDUAL CHECKPOINT ANALYSIS\n")
            f.write("-" * 80 + "\n\n")
            
            for i, r in enumerate(summary['individual_results'], 1):
                f.write(f"[{i}] {r['metadata']['checkpoint_path']}\n")
                f.write(f"    Epoch: {r['metadata']['epoch']}\n")
                f.write(f"    Original alpha: {r['original']['purity']['global_alpha']:.6f}\n")
                f.write(f"    Original phase: {r['original']['phase']}\n")
                f.write(f"    Alpha retention: {r['comparison']['alpha_retention']:.2%}\n")
                f.write(f"    Temperature ratio: {r['comparison']['temperature_ratio']:.2f}x\n")
                f.write(f"    Intermediate phase: {r['intermediate_phase_exists']}\n")
                f.write(f"    Phase type: {r['comparison']['intermediate_phase_type']}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("END OF REPORT\n")
            f.write("=" * 80 + "\n")
        
        print(f"Saved text report: {report_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Purity index analysis for Strassen algorithm crystallization'
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
        default='purity_analysis',
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
    
    args = parser.parse_args()
    
    config = PurityConfig(
        HIDDEN_DIM=args.hidden_dim,
        MATRIX_SIZE=args.matrix_size
    )
    
    pipeline = PurityPipeline(config)
    
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
    elif args.latest:
        results = pipeline.process_directory(args.dir, args.latest, args.output)
        if results:
            pipeline.generate_summary(results, args.output)


if __name__ == '__main__':
    main()
