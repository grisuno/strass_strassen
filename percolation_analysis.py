import numpy as np
import json
import os
import argparse
import time
import glob
import warnings
from datetime import datetime
from typing import Dict, Any, List, Tuple, Optional, Protocol, runtime_checkable, Union
from pathlib import Path
from dataclasses import dataclass, field
from scipy.stats import entropy
from scipy.linalg import eigh
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

warnings.filterwarnings('ignore')


@dataclass(frozen=True)
class PercolationConfiguration:
    HIDDEN_DIM: int = 8
    MATRIX_SIZE: int = 2
    INPUT_DIM: int = 4
    EXPANSION_SCALES: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)
    PERCOLATION_NUM_THRESHOLDS: int = 50
    PERCOLATION_THRESHOLD_MIN: float = 0.01
    PERCOLATION_THRESHOLD_MAX: float = 0.99
    PRUNING_LEVELS: Tuple[float, ...] = (
        0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35,
        0.40, 0.42, 0.44, 0.46, 0.48, 0.49, 0.50, 0.51,
        0.52, 0.54, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80,
        0.85, 0.90, 0.95)
    DISCRETIZATION_MARGIN: float = 0.1
    ALPHA_SATURATION: float = 20.0
    ALPHA_THRESHOLD_CRYSTAL: float = 7.0
    ALPHA_THRESHOLD_GLASS: float = 1.0
    GLASS_TEMPERATURE_THRESHOLD: float = 0.1
    CRYSTAL_TEMPERATURE_THRESHOLD: float = 0.01
    TEMPERATURE_WINDOW: int = 100
    SPECIFIC_HEAT_WINDOW: int = 50
    LEVEL_SPACING_WIGNER_DYSON: float = 0.5307
    LEVEL_SPACING_POISSON: float = 0.3863
    LEVEL_SPACING_TOLERANCE: float = 0.05
    PR_LOCALIZATION_THRESHOLD: float = 0.8
    PR_DELIMITED_THRESHOLD: float = 0.1
    PR_RENYI_INDEX: int = 2
    HBAR_ENERGY_GAP_SCALE: float = 1.0
    HBAR_NUMERICAL_NOISE_FLOOR: float = 1e-7
    DISCRETIZATION_NOISE_LEVELS: Tuple[float, ...] = (0.0, 0.001, 0.005, 0.01, 0.05, 0.1)
    DISCRETIZATION_GAP_COLLAPSE_THRESHOLD: float = 0.5
    PERCOLATION_2D_SITE_PC: float = 0.592746
    PERCOLATION_2D_BOND_PC: float = 0.5
    PERCOLATION_3D_SITE_PC: float = 0.3116
    PERCOLATION_BETA_2D: float = 5.0 / 36.0
    PERCOLATION_GAMMA_2D: float = 43.0 / 18.0
    PERCOLATION_NU_2D: float = 4.0 / 3.0
    PERCOLATION_BETA_MF: float = 1.0
    PERCOLATION_GAMMA_MF: float = 1.0
    PERCOLATION_NU_MF: float = 0.5
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    FIGURE_DPI: int = 150
    SAVE_FORMAT: str = 'png'
    FIGURE_WIDTH: float = 14.0
    FIGURE_HEIGHT: float = 10.0
    DEVICE: str = 'cpu'
    METRIC_PRECISION: int = 6
    ENTROPY_BINS: int = 50
    STRASSEN_RANK: int = 7
    STRASSEN_DISCRETE_VALUES: Tuple[int, ...] = (-1, 0, 1)

    def get_effective_input_dim(self) -> int:
        return self.MATRIX_SIZE * self.MATRIX_SIZE

    def get_total_parameters(self) -> int:
        input_dim = self.get_effective_input_dim()
        return (input_dim * self.HIDDEN_DIM * 2) + (self.HIDDEN_DIM * input_dim)

    def get_percolation_thresholds(self) -> np.ndarray:
        return np.linspace(
            self.PERCOLATION_THRESHOLD_MIN,
            self.PERCOLATION_THRESHOLD_MAX,
            self.PERCOLATION_NUM_THRESHOLDS)


@runtime_checkable
class IModel(Protocol):
    def get_coefficients(self) -> Dict[str, np.ndarray]: ...


class NumpyModelWrapper:
    def __init__(self, weights: Dict[str, np.ndarray]):
        self._weights = weights

    def get_coefficients(self) -> Dict[str, np.ndarray]:
        return self._weights

    def get_flat_parameters(self) -> np.ndarray:
        parts = []
        for name in ['U', 'V', 'W']:
            if name in self._weights:
                parts.append(self._weights[name].flatten())
        return np.concatenate(parts)


if TORCH_AVAILABLE:
    class BilinearStrassenModel(nn.Module):
        def __init__(self, config: PercolationConfiguration):
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

        def get_flat_parameters(self) -> np.ndarray:
            params = []
            for param in self.parameters():
                params.append(param.detach().cpu().numpy().flatten())
            return np.concatenate(params)


class _DummyObject:
    """
    Transparent stand-in for any unpicklable class found inside a checkpoint.
    Stores all keyword and positional constructor arguments as attributes so
    that downstream dict-key lookups still work on objects that behave like
    namespaces (e.g. UnifiedConfig, TrainingConfig, etc.).
    """
    def __init__(self, *args, **kwargs):
        for i, a in enumerate(args):
            setattr(self, f'_arg{i}', a)
        for k, v in kwargs.items():
            setattr(self, k, v)

    def __repr__(self):
        return f'<_DummyObject attrs={list(self.__dict__.keys())}>'


def _safe_torch_load(path: str) -> Any:
    """
    Load a torch checkpoint that may contain unknown serialized classes
    (UnifiedConfig, TrainingConfig, custom dataclasses, etc.).

    Strategy:
      1. Try normal torch.load.  If it works, return immediately.
      2. On AttributeError / ModuleNotFoundError, extract the missing
         class names from the exception, inject _DummyObject into
         sys.modules / __main__ under those names, and retry.
      3. Repeat up to MAX_RETRIES times (each retry may reveal a
         new missing class that was nested deeper in the pickle stream).
      4. Clean up injected names after loading.

    This keeps torch's own deserialization path (including correct
    persistent_load for tensor storage) fully intact.
    """
    import torch as _torch
    import sys
    import re

    _MAX_PATCH_RETRIES = 20
    _patched_names: List[Tuple[str, str]] = []

    def _patch_missing(exc: Exception) -> bool:
        msg = str(exc)
        patched = False
        for pattern in [
            r"Can't get attribute '(\w+)' on <module '([^']+)'",
            r"cannot find attribute '(\w+)' in <module '([^']+)'",
            r"attribute '(\w+)' of module '([^']+)'"
        ]:
            match = re.search(pattern, msg)
            if match:
                attr_name = match.group(1)
                module_name = match.group(2)
                mod = sys.modules.get(module_name)
                if mod is None:
                    import types
                    mod = types.ModuleType(module_name)
                    sys.modules[module_name] = mod
                setattr(mod, attr_name, _DummyObject)
                _patched_names.append((module_name, attr_name))
                patched = True
                break
        if not patched:
            match2 = re.search(r"No module named '([^']+)'", msg)
            if match2:
                module_name = match2.group(1)
                import types
                parts = module_name.split('.')
                for i in range(len(parts)):
                    partial = '.'.join(parts[:i + 1])
                    if partial not in sys.modules:
                        dummy_mod = types.ModuleType(partial)
                        sys.modules[partial] = dummy_mod
                        _patched_names.append((partial, ''))
                        if i > 0:
                            parent = '.'.join(parts[:i])
                            parent_mod = sys.modules.get(parent)
                            if parent_mod is not None:
                                setattr(parent_mod, parts[i], dummy_mod)
                patched = True
        return patched

    def _cleanup():
        for module_name, attr_name in _patched_names:
            if attr_name:
                mod = sys.modules.get(module_name)
                if mod is not None and hasattr(mod, attr_name):
                    try:
                        delattr(mod, attr_name)
                    except Exception:
                        pass

    last_exc = None
    for _ in range(_MAX_PATCH_RETRIES):
        try:
            result = _torch.load(path, map_location='cpu', weights_only=False)
            _cleanup()
            return result
        except (AttributeError, ModuleNotFoundError, ImportError) as exc:
            last_exc = exc
            if not _patch_missing(exc):
                break
        except Exception as exc:
            _cleanup()
            raise

    _cleanup()
    raise RuntimeError(
        f"Could not load checkpoint {path} after {_MAX_PATCH_RETRIES} "
        f"patching attempts. Last error: {last_exc}. "
        f"Patched names: {_patched_names}. "
        f"This may indicate the checkpoint contains deeply nested custom "
        f"classes. Try loading with the original training script and "
        f"re-saving: torch.save({{'model_state_dict': model.state_dict()}}, path)")


class CheckpointMigrator:
    def migrate(self, raw_data: Any, config: PercolationConfiguration) -> Optional[Dict[str, np.ndarray]]:
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
            for key, val in raw_data.items():
                if isinstance(val, dict):
                    result = self._migrate_dict(val, config)
                    if result is not None:
                        return result
        if hasattr(raw_data, '__dict__'):
            return self.migrate(vars(raw_data), config)
        return None

    def _migrate_dict(self, state_dict: Dict[str, Any],
                      config: PercolationConfiguration) -> Optional[Dict[str, np.ndarray]]:
        if any(k in state_dict for k in ['U', 'V', 'W']):
            return self._migrate_custom(state_dict, config)
        elif 'U_coefs' in state_dict:
            return self._migrate_coefs(state_dict)
        elif any(k.endswith('.weight') for k in state_dict.keys()):
            return self._migrate_standard(state_dict)
        for val in state_dict.values():
            if isinstance(val, dict):
                result = self._try_migrate_nested(val, config)
                if result is not None:
                    return result
        return None

    def _try_migrate_nested(self, candidate: Dict[str, Any],
                            config: PercolationConfiguration) -> Optional[Dict[str, np.ndarray]]:
        if any(k in candidate for k in ['U', 'V', 'W']):
            return self._migrate_custom(candidate, config)
        if 'U_coefs' in candidate:
            return self._migrate_coefs(candidate)
        if any(k.endswith('.weight') for k in candidate.keys()):
            return self._migrate_standard(candidate)
        return None

    def _migrate_custom(self, sd: Dict[str, Any],
                        config: PercolationConfiguration) -> Optional[Dict[str, np.ndarray]]:
        U = sd.get('U', sd.get('U_coefs'))
        V = sd.get('V', sd.get('V_coefs'))
        W = sd.get('W', sd.get('W_coefs'))
        if U is None or V is None or W is None:
            return None
        U, V, W = self._np(U), self._np(V), self._np(W)
        if U.shape == (7, 4):
            up = np.zeros((config.HIDDEN_DIM, config.INPUT_DIM))
            vp = np.zeros((config.HIDDEN_DIM, config.INPUT_DIM))
            wp = np.zeros((config.INPUT_DIM, config.HIDDEN_DIM))
            up[:7] = U; vp[:7] = V; wp[:, :7] = W
            return {'U': up, 'V': vp, 'W': wp}
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
        if TORCH_AVAILABLE and hasattr(tensor, 'cpu'):
            if hasattr(tensor, 'detach'):
                return tensor.detach().cpu().numpy()
            return tensor.cpu().numpy()
        if isinstance(tensor, np.ndarray):
            return tensor
        return np.array(tensor)


class WeightGraphConstructor:
    def __init__(self, config: PercolationConfiguration):
        self.config = config

    def construct_adjacency_from_weights(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        U, V, W = weights['U'], weights['V'], weights['W']
        hidden = U.shape[0]
        input_dim = U.shape[1]
        n_nodes = input_dim + hidden + input_dim
        adj = np.zeros((n_nodes, n_nodes))
        for h in range(hidden):
            for i in range(input_dim):
                w_u = np.abs(U[h, i])
                w_v = np.abs(V[h, i])
                combined = max(w_u, w_v)
                adj[i, input_dim + h] = combined
                adj[input_dim + h, i] = combined
        offset = input_dim + hidden
        for o in range(input_dim):
            for h in range(hidden):
                w_w = np.abs(W[o, h])
                adj[input_dim + h, offset + o] = w_w
                adj[offset + o, input_dim + h] = w_w
        return adj

    def construct_weight_correlation_graph(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        parts = []
        for name in ['U', 'V', 'W']:
            parts.append(weights[name].flatten())
        wv = np.concatenate(parts)
        outer = np.outer(np.abs(wv), np.abs(wv))
        mx = np.max(outer) if np.max(outer) > 1e-15 else 1.0
        adj = outer / mx
        np.fill_diagonal(adj, 0.0)
        return adj

    def construct_slot_interaction_graph(self, weights: Dict[str, np.ndarray]) -> np.ndarray:
        U, V, W = weights['U'], weights['V'], weights['W']
        hidden = U.shape[0]
        vecs = []
        for h in range(hidden):
            vec = np.concatenate([U[h, :], V[h, :], W[:, h]])
            vecs.append(vec)
        vecs = np.array(vecs)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms = np.where(norms < 1e-15, 1.0, norms)
        normed = vecs / norms
        adj = np.abs(normed @ normed.T)
        np.fill_diagonal(adj, 0.0)
        return adj


class BondPercolationAnalyzer:
    def __init__(self, config: PercolationConfiguration):
        self.config = config

    def analyze(self, adjacency: np.ndarray,
                thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if thresholds is None:
            thresholds = self.config.get_percolation_thresholds()
        n = adjacency.shape[0]
        res = {
            'thresholds': thresholds.tolist(),
            'largest_component_fraction': [],
            'num_clusters': [],
            'susceptibility': [],
            'second_largest_component': [],
            'average_cluster_size': [],
            'cluster_size_distributions': [],
            'percolation_threshold': None,
            'critical_exponents': {},
            'order_parameter_curve': [],
            'graph_dimension': n}
        mx = np.max(adjacency) if np.max(adjacency) > 1e-15 else 1.0
        for thr in thresholds:
            cutoff = thr * mx
            binary = (adjacency >= cutoff).astype(int)
            np.fill_diagonal(binary, 0)
            sparse = csr_matrix(binary)
            nc, labels = connected_components(sparse, directed=False)
            sizes = sorted([int(np.sum(labels == c)) for c in range(nc)], reverse=True)
            largest = sizes[0] if sizes else 0
            second = sizes[1] if len(sizes) > 1 else 0
            frac = largest / n if n > 0 else 0.0
            chi = self._susceptibility(sizes, n)
            avg = float(np.mean(sizes)) if sizes else 0.0
            res['largest_component_fraction'].append(float(frac))
            res['num_clusters'].append(int(nc))
            res['susceptibility'].append(float(chi))
            res['second_largest_component'].append(int(second))
            res['average_cluster_size'].append(avg)
            res['order_parameter_curve'].append(float(frac))
            res['cluster_size_distributions'].append(sizes[:10])
        pc = self._find_pc(thresholds, res)
        res['percolation_threshold'] = pc
        if pc is not None:
            res['critical_exponents'] = self._exponents(thresholds, res, pc)
        return res

    def _susceptibility(self, sizes: List[int], n: int) -> float:
        if n == 0 or not sizes:
            return 0.0
        arr = np.array(sizes, dtype=float)
        exc = np.delete(arr, np.argmax(arr))
        return float(np.sum(exc ** 2) / n) if len(exc) > 0 else 0.0

    def _find_pc(self, thresholds: np.ndarray, res: Dict) -> Optional[float]:
        chi = np.array(res['susceptibility'])
        if len(chi) == 0 or np.max(chi) < 1e-15:
            return None
        return float(thresholds[np.argmax(chi)])

    def _exponents(self, thresholds: np.ndarray,
                   res: Dict, pc: float) -> Dict[str, Any]:
        op = np.array(res['largest_component_fraction'])
        chi = np.array(res['susceptibility'])
        above = thresholds <= pc
        beta = self._fit_pl(pc - thresholds[above], op[above])
        gamma = self._fit_pl(np.abs(thresholds - pc) + 1e-15, chi)
        return {
            'beta': float(beta) if beta is not None else None,
            'gamma': float(gamma) if gamma is not None else None,
            'beta_2d_reference': self.config.PERCOLATION_BETA_2D,
            'gamma_2d_reference': self.config.PERCOLATION_GAMMA_2D,
            'beta_mean_field_reference': self.config.PERCOLATION_BETA_MF,
            'gamma_mean_field_reference': self.config.PERCOLATION_GAMMA_MF}

    @staticmethod
    def _fit_pl(x: np.ndarray, y: np.ndarray) -> Optional[float]:
        mask = (x > 1e-10) & (y > 1e-10)
        if np.sum(mask) < 3:
            return None
        try:
            return float(np.polyfit(np.log(x[mask]), np.log(y[mask]), 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return None


class SitePercolationAnalyzer:
    def __init__(self, config: PercolationConfiguration):
        self.config = config

    def analyze(self, weights: Dict[str, np.ndarray],
                thresholds: Optional[np.ndarray] = None) -> Dict[str, Any]:
        if thresholds is None:
            thresholds = self.config.get_percolation_thresholds()
        parts = []
        bounds = {}
        off = 0
        for name in ['U', 'V', 'W']:
            flat = weights[name].flatten()
            bounds[name] = (off, off + len(flat))
            parts.append(flat)
            off += len(flat)
        wv = np.concatenate(parts)
        aw = np.abs(wv)
        mx = np.max(aw) if np.max(aw) > 1e-15 else 1.0
        hidden = self.config.HIDDEN_DIM
        input_dim = self.config.get_effective_input_dim()
        res = {
            'thresholds': thresholds.tolist(),
            'occupation_fraction': [],
            'active_parameters': [],
            'layer_occupation': {n: [] for n in ['U', 'V', 'W']},
            'site_percolation_threshold': None,
            'active_slots_per_threshold': [],
            'structural_integrity': []}
        for thr in thresholds:
            cutoff = thr * mx
            active = aw >= cutoff
            res['occupation_fraction'].append(float(np.mean(active)))
            res['active_parameters'].append(int(np.sum(active)))
            for name in ['U', 'V', 'W']:
                s, e = bounds[name]
                res['layer_occupation'][name].append(float(np.mean(active[s:e])))
            pw = wv.copy()
            pw[~active] = 0.0
            rounded = np.round(pw)
            delta = float(np.max(np.abs(pw - rounded)))
            U_p = pw[bounds['U'][0]:bounds['U'][1]].reshape(hidden, input_dim)
            slots = int(np.sum(np.linalg.norm(U_p, axis=1) > 1e-8))
            res['active_slots_per_threshold'].append(slots)
            ok = 1.0 if (delta < self.config.DISCRETIZATION_MARGIN and
                         slots >= self.config.STRASSEN_RANK) else 0.0
            res['structural_integrity'].append(ok)
        ia = np.array(res['structural_integrity'])
        if np.any(ia < 1.0) and np.any(ia >= 1.0):
            idx = np.where(np.diff(ia) < 0)[0]
            res['site_percolation_threshold'] = float(thresholds[idx[0]]) if len(idx) > 0 else None
        return res


class PruningPercolationAnalyzer:
    def __init__(self, config: PercolationConfiguration):
        self.config = config

    def analyze(self, weights: Dict[str, np.ndarray]) -> Dict[str, Any]:
        U, V, W = weights['U'], weights['V'], weights['W']
        all_flat = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        abs_flat = np.abs(all_flat)
        sorted_mag = np.sort(abs_flat)
        hidden = self.config.HIDDEN_DIM
        input_dim = self.config.get_effective_input_dim()
        u_sz, v_sz = U.size, V.size
        res = {
            'pruning_levels': list(self.config.PRUNING_LEVELS),
            'discretization_delta': [], 'discretization_alpha': [],
            'active_slots': [], 'structural_integrity': [],
            'occupation_fraction': [], 'kappa_proxy': [],
            'hbar_proxy': [], 'effective_temperature_proxy': [],
            'local_complexity_proxy': [], 'purity_alpha': [],
            'entropy_proxy': [], 'ipr_proxy': [],
            'level_spacing_ratio': [], 'fractal_dimension_proxy': [],
            'coherence_length_proxy': [], 'yield_strength': None,
            'collapse_sparsity': None, 'elastic_regime_end': None,
            'phase_per_level': []}
        for sparsity in self.config.PRUNING_LEVELS:
            n_prune = int(len(all_flat) * sparsity)
            mask = np.ones(len(all_flat), dtype=bool)
            if 0 < n_prune < len(all_flat):
                thr_val = sorted_mag[n_prune]
                mask = abs_flat >= thr_val
            elif n_prune >= len(all_flat):
                mask[:] = False
            pruned = all_flat.copy()
            pruned[~mask] = 0.0
            U_p = pruned[:u_sz].reshape(U.shape)
            V_p = pruned[u_sz:u_sz + v_sz].reshape(V.shape)
            rounded = np.round(pruned)
            delta = float(np.max(np.abs(pruned - rounded)))
            alpha = self._d2a(delta)
            sn_u = np.linalg.norm(U_p, axis=1)
            sn_v = np.linalg.norm(V_p, axis=1)
            active_slots = int(np.sum((sn_u + sn_v) > 1e-8))
            occ = float(np.mean(mask))
            integrity = 1.0 if (delta < self.config.DISCRETIZATION_MARGIN and
                                active_slots >= self.config.STRASSEN_RANK) else 0.0
            kp = self._kappa(pruned)
            hp = self._hbar(pruned)
            tp = self._teff(pruned)
            lc = self._lc(pruned, delta)
            ep = self._entropy(pruned)
            ipr = self._ipr(pruned)
            lsr = self._lsr(pruned)
            fd = self._fractal(ipr, len(pruned))
            cl = 1.0 / np.sqrt(max(ipr, 1e-15))
            phase = self._phase(alpha, tp, delta)
            res['discretization_delta'].append(delta)
            res['discretization_alpha'].append(alpha)
            res['active_slots'].append(active_slots)
            res['structural_integrity'].append(integrity)
            res['occupation_fraction'].append(occ)
            res['kappa_proxy'].append(kp)
            res['hbar_proxy'].append(hp)
            res['effective_temperature_proxy'].append(tp)
            res['local_complexity_proxy'].append(lc)
            res['purity_alpha'].append(alpha)
            res['entropy_proxy'].append(ep)
            res['ipr_proxy'].append(ipr)
            res['level_spacing_ratio'].append(lsr)
            res['fractal_dimension_proxy'].append(fd)
            res['coherence_length_proxy'].append(cl)
            res['phase_per_level'].append(phase)
        ia = np.array(res['structural_integrity'])
        cidx = np.where(ia < 1.0)[0]
        if len(cidx) > 0:
            ci = cidx[0]
            res['collapse_sparsity'] = float(self.config.PRUNING_LEVELS[ci])
            if ci > 0:
                res['yield_strength'] = float(self.config.PRUNING_LEVELS[ci - 1])
                for idx in range(ci):
                    if res['discretization_delta'][idx] > self.config.DISCRETIZATION_MARGIN * 0.5:
                        res['elastic_regime_end'] = float(self.config.PRUNING_LEVELS[idx])
                        break
        return res

    def _d2a(self, delta: float) -> float:
        if delta < 1e-15:
            return self.config.ALPHA_SATURATION
        return -np.log(delta + 1e-15)

    def _kappa(self, wv: np.ndarray) -> float:
        active = wv[np.abs(wv) > 1e-15]
        if len(active) < 2:
            return 999999.0
        cov = np.cov(active.reshape(1, -1)) if len(active) > 1 else np.array([[1.0]])
        if np.isscalar(cov):
            return 1.0 if cov > 1e-15 else 999999.0
        ev = np.linalg.eigvalsh(np.atleast_2d(cov))
        ev = ev[ev > 1e-15]
        if len(ev) < 1:
            return 999999.0
        return float(np.max(ev) / np.min(ev))

    def _hbar(self, wv: np.ndarray) -> float:
        active = np.abs(wv[np.abs(wv) > 1e-15])
        if len(active) < 2:
            return self.config.HBAR_NUMERICAL_NOISE_FLOOR
        gaps = np.diff(np.sort(active))
        mg = np.min(gaps) if len(gaps) > 0 and np.min(gaps) > 1e-15 else 1e-15
        ipr = self._ipr(wv)
        if ipr < 1e-15 or mg < 1e-15:
            return self.config.HBAR_NUMERICAL_NOISE_FLOOR
        h = 1.0 / np.sqrt(ipr * mg * self.config.HBAR_ENERGY_GAP_SCALE)
        return max(float(h), self.config.HBAR_NUMERICAL_NOISE_FLOOR)

    def _teff(self, wv: np.ndarray) -> float:
        return float(np.var(wv - np.round(wv)))

    def _lc(self, wv: np.ndarray, delta: float) -> float:
        if delta < 1e-10:
            return 0.0
        active = wv[np.abs(wv) > 1e-15]
        if len(active) < 2:
            return 0.0
        d = np.abs(active - np.round(active))
        vol = np.prod(d + 1e-15)
        return float(np.log(vol + 1e-300)) if vol > 0 else 0.0

    def _entropy(self, wv: np.ndarray) -> float:
        active = wv[np.abs(wv) > 1e-15]
        if len(active) < 2:
            return 0.0
        h, _ = np.histogram(active, bins=self.config.ENTROPY_BINS, density=True)
        h = h[h > 0]
        if len(h) == 0:
            return 0.0
        h = h / np.sum(h)
        return float(entropy(h))

    def _ipr(self, wv: np.ndarray) -> float:
        norm = np.sum(np.abs(wv) ** 2)
        if norm < 1e-15:
            return 0.0
        n = wv / np.sqrt(norm)
        return float(np.sum(np.abs(n) ** 4))

    def _lsr(self, wv: np.ndarray) -> float:
        active = np.sort(np.abs(wv[np.abs(wv) > 1e-15]))
        if len(active) < 3:
            return 0.0
        sp = np.diff(active)
        ratios = []
        for i in range(len(sp) - 1):
            d = max(sp[i], sp[i + 1])
            if d > 1e-15:
                ratios.append(min(sp[i], sp[i + 1]) / d)
        return float(np.mean(ratios)) if ratios else 0.0

    def _fractal(self, ipr: float, n: int) -> float:
        if n <= 1 or ipr <= 0:
            return 0.0
        return -np.log(max(ipr, 1e-300)) / np.log(n)

    def _phase(self, alpha: float, temp: float, delta: float) -> str:
        if (alpha > self.config.ALPHA_THRESHOLD_CRYSTAL and
                temp < self.config.CRYSTAL_TEMPERATURE_THRESHOLD and
                delta < self.config.DISCRETIZATION_MARGIN):
            return 'perfect_crystal'
        elif (alpha > self.config.ALPHA_THRESHOLD_CRYSTAL and
              temp < self.config.GLASS_TEMPERATURE_THRESHOLD):
            return 'crystal_with_thermal_fluctuations'
        elif alpha > self.config.ALPHA_THRESHOLD_CRYSTAL:
            return 'hot_crystal'
        elif (alpha > self.config.ALPHA_THRESHOLD_GLASS and
              temp < self.config.CRYSTAL_TEMPERATURE_THRESHOLD):
            return 'cold_polycrystal'
        elif alpha > self.config.ALPHA_THRESHOLD_GLASS:
            return 'warm_polycrystal'
        elif temp < self.config.CRYSTAL_TEMPERATURE_THRESHOLD:
            return 'cold_glass'
        else:
            return 'hot_glass'


class ClusterSizeDistributionAnalyzer:
    def __init__(self, config: PercolationConfiguration):
        self.config = config

    def analyze_at_threshold(self, adjacency: np.ndarray,
                             threshold: float) -> Dict[str, Any]:
        n = adjacency.shape[0]
        mx = np.max(adjacency) if np.max(adjacency) > 1e-15 else 1.0
        binary = (adjacency >= threshold * mx).astype(int)
        np.fill_diagonal(binary, 0)
        nc, labels = connected_components(csr_matrix(binary), directed=False)
        sizes = sorted([int(np.sum(labels == c)) for c in range(nc)], reverse=True)
        largest = sizes[0] if sizes else 0
        tau = self._tau(sizes)
        dist = {}
        for s in sizes:
            dist[s] = dist.get(s, 0) + 1
        return {
            'threshold': float(threshold),
            'num_clusters': nc,
            'largest_cluster': largest,
            'largest_cluster_fraction': float(largest / n) if n > 0 else 0.0,
            'cluster_sizes': sizes,
            'size_distribution': dist,
            'tau_estimate': tau,
            'tau_2d_reference': 187.0 / 91.0,
            'is_critical': self._critical(sizes, n)}

    def _tau(self, sizes: List[int]) -> Optional[float]:
        s = np.array([x for x in sizes if x > 1])
        if len(s) < 5:
            return None
        u, c = np.unique(s, return_counts=True)
        m = (u > 0) & (c > 0)
        if np.sum(m) < 3:
            return None
        try:
            return float(-np.polyfit(np.log(u[m].astype(float)),
                                     np.log(c[m].astype(float)), 1)[0])
        except (np.linalg.LinAlgError, ValueError):
            return None

    def _critical(self, sizes: List[int], n: int) -> bool:
        if not sizes or n == 0 or len(sizes) < 2:
            return False
        ss = sorted(sizes, reverse=True)
        f1, f2 = ss[0] / n, ss[1] / n
        return f1 < 0.8 and f1 > 0.1 and f2 > 0.01


class PercolationUniversalityAnalyzer:
    def __init__(self, config: PercolationConfiguration):
        self.config = config

    def classify_universality(self, measured: Dict[str, Optional[float]]) -> Dict[str, Any]:
        classes = {
            '2D_percolation': {
                'beta': self.config.PERCOLATION_BETA_2D,
                'gamma': self.config.PERCOLATION_GAMMA_2D},
            'mean_field': {
                'beta': self.config.PERCOLATION_BETA_MF,
                'gamma': self.config.PERCOLATION_GAMMA_MF}}
        beta = measured.get('beta')
        gamma = measured.get('gamma')
        distances = {}
        for cn, refs in classes.items():
            d, cnt = 0.0, 0
            if beta is not None:
                d += (beta - refs['beta']) ** 2; cnt += 1
            if gamma is not None:
                d += (gamma - refs['gamma']) ** 2; cnt += 1
            distances[cn] = np.sqrt(d / cnt) if cnt > 0 else float('inf')
        best = min(distances, key=distances.get) if distances else 'unknown'
        md = min(distances.values()) if distances else float('inf')
        conf = 'high' if md < 0.1 else ('moderate' if md < 0.5 else ('low' if md < 1.0 else 'no_match'))
        return {
            'best_match': best,
            'distances': {k: float(v) for k, v in distances.items()},
            'measured_exponents': {k: float(v) if v is not None else None for k, v in measured.items()},
            'reference_classes': classes,
            'confidence': conf}


class PercolationCheckpointManager:
    def __init__(self, config: PercolationConfiguration):
        self.config = config
        self.last_time = time.time()
        self.counter = 0

    def should_save(self) -> bool:
        return (time.time() - self.last_time) / 60.0 >= self.config.CHECKPOINT_INTERVAL_MINUTES

    def save(self, results: Dict[str, Any], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'percolation_checkpoint_latest.json')
        results['_checkpoint_metadata'] = {
            'timestamp': datetime.now().isoformat(),
            'counter': self.counter}
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.last_time = time.time()
        self.counter += 1
        return path

    def load(self, output_dir: str) -> Optional[Dict[str, Any]]:
        path = os.path.join(output_dir, 'percolation_checkpoint_latest.json')
        if os.path.exists(path):
            with open(path, 'r') as f:
                return json.load(f)
        return None


class PercolationVisualizationEngine:
    def __init__(self, config: PercolationConfiguration):
        self.config = config

    def generate_all_figures(self, results: Dict[str, Any],
                            output_dir: str) -> List[str]:
        os.makedirs(output_dir, exist_ok=True)
        generated = []
        if 'bond_percolation' in results:
            generated.append(self._plot_bond(results['bond_percolation'], output_dir))
        if 'pruning_percolation' in results:
            generated.append(self._plot_pruning(results['pruning_percolation'], output_dir))
            generated.append(self._plot_dashboard(results['pruning_percolation'], output_dir))
        if 'site_percolation' in results:
            generated.append(self._plot_site(results['site_percolation'], output_dir))
        if 'cluster_analysis' in results:
            generated.append(self._plot_cluster(results['cluster_analysis'], output_dir))
        return generated

    def _plot_bond(self, data: Dict, out: str) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(self.config.FIGURE_WIDTH, self.config.FIGURE_HEIGHT))
        fig.suptitle('Bond Percolation Analysis: Weight Graph Connectivity', fontsize=14, fontweight='bold')
        t = data['thresholds']
        ax = axes[0, 0]
        ax.plot(t, data['largest_component_fraction'], 'b-', linewidth=2, label='P_inf')
        if data['percolation_threshold'] is not None:
            ax.axvline(x=data['percolation_threshold'], color='r', linestyle='--',
                       label=f'p_c = {data["percolation_threshold"]:.4f}')
        ax.set_xlabel('Threshold (p)'); ax.set_ylabel('Largest Component Fraction')
        ax.set_title('Order Parameter P_inf(p)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        ax.plot(t, data['susceptibility'], 'g-', linewidth=2)
        if data['percolation_threshold'] is not None:
            ax.axvline(x=data['percolation_threshold'], color='r', linestyle='--',
                       label=f'p_c = {data["percolation_threshold"]:.4f}')
        ax.set_xlabel('Threshold (p)'); ax.set_ylabel('Susceptibility chi(p)')
        ax.set_title('Susceptibility (Diverges at p_c)'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        ax.plot(t, data['num_clusters'], 'm-', linewidth=2)
        ax.set_xlabel('Threshold (p)'); ax.set_ylabel('Number of Clusters')
        ax.set_title('Cluster Count'); ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        ax.plot(t, data['second_largest_component'], 'orange', linewidth=2)
        if data['percolation_threshold'] is not None:
            ax.axvline(x=data['percolation_threshold'], color='r', linestyle='--')
        ax.set_xlabel('Threshold (p)'); ax.set_ylabel('Second Largest Component')
        ax.set_title('Second Largest (Peaks at p_c)'); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out, f'bond_percolation.{self.config.SAVE_FORMAT}')
        fig.savefig(path, dpi=self.config.FIGURE_DPI, bbox_inches='tight'); plt.close(fig)
        return path

    def _plot_pruning(self, data: Dict, out: str) -> str:
        fig, axes = plt.subplots(3, 3, figsize=(self.config.FIGURE_WIDTH + 4, self.config.FIGURE_HEIGHT + 6))
        fig.suptitle('Pruning Percolation: Structural Collapse\n'
                     'Metrics: kappa, delta, T_eff, hbar_eff, LC, SP, h_bar, alpha, IPR, r',
                     fontsize=13, fontweight='bold')
        lv = data['pruning_levels']
        ax = axes[0, 0]
        ax.plot(lv, data['discretization_delta'], 'b-o', markersize=3, linewidth=1.5)
        ax.axhline(y=self.config.DISCRETIZATION_MARGIN, color='r', linestyle='--',
                    label=f'Margin={self.config.DISCRETIZATION_MARGIN}')
        if data['yield_strength'] is not None:
            ax.axvline(x=data['yield_strength'], color='green', linestyle=':', label=f'Yield={data["yield_strength"]:.2f}')
        if data['collapse_sparsity'] is not None:
            ax.axvline(x=data['collapse_sparsity'], color='red', linestyle=':', label=f'Collapse={data["collapse_sparsity"]:.2f}')
        ax.set_xlabel('Sparsity'); ax.set_ylabel('delta'); ax.set_title('delta vs Pruning')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax = axes[0, 1]
        ax.plot(lv, data['kappa_proxy'], 'r-o', markersize=3, linewidth=1.5)
        ax.set_xlabel('Sparsity'); ax.set_ylabel('kappa'); ax.set_title('kappa vs Pruning')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)
        ax = axes[0, 2]
        ax.plot(lv, data['active_slots'], 'g-o', markersize=3, linewidth=1.5)
        ax.axhline(y=self.config.STRASSEN_RANK, color='r', linestyle='--', label=f'Rank={self.config.STRASSEN_RANK}')
        ax.set_xlabel('Sparsity'); ax.set_ylabel('Active Slots'); ax.set_title('Slots vs Pruning')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax = axes[1, 0]
        ax.plot(lv, data['hbar_proxy'], 'purple', linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('Sparsity'); ax.set_ylabel('hbar_eff'); ax.set_title('hbar_eff vs Pruning')
        ax.set_yscale('log'); ax.grid(True, alpha=0.3)
        ax = axes[1, 1]
        ax.plot(lv, data['effective_temperature_proxy'], 'darkorange', linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('Sparsity'); ax.set_ylabel('T_eff'); ax.set_title('T_eff vs Pruning')
        ax.set_yscale('symlog', linthresh=1e-15); ax.grid(True, alpha=0.3)
        ax = axes[1, 2]
        ax.plot(lv, data['local_complexity_proxy'], 'teal', linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('Sparsity'); ax.set_ylabel('LC'); ax.set_title('Local Complexity vs Pruning')
        ax.grid(True, alpha=0.3)
        ax = axes[2, 0]
        ax.plot(lv, data['purity_alpha'], 'navy', linewidth=1.5, marker='o', markersize=3)
        ax.set_xlabel('Sparsity'); ax.set_ylabel('alpha'); ax.set_title('Purity vs Pruning')
        ax.grid(True, alpha=0.3)
        ax = axes[2, 1]
        ax.plot(lv, data['ipr_proxy'], 'brown', linewidth=1.5, marker='o', markersize=3, label='IPR')
        ax.plot(lv, data['level_spacing_ratio'], 'olive', linewidth=1.5, marker='s', markersize=3, label='r')
        ax.axhline(y=self.config.LEVEL_SPACING_WIGNER_DYSON, color='blue', linestyle='--', alpha=0.5, label='WD')
        ax.axhline(y=self.config.LEVEL_SPACING_POISSON, color='green', linestyle='--', alpha=0.5, label='Poisson')
        ax.set_xlabel('Sparsity'); ax.set_ylabel('Value'); ax.set_title('IPR + Level Spacing')
        ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
        ax = axes[2, 2]
        ax.plot(lv, data['entropy_proxy'], 'darkred', linewidth=1.5, marker='o', markersize=3, label='h_bar')
        ax2 = ax.twinx()
        ax2.plot(lv, data['fractal_dimension_proxy'], 'darkblue', linewidth=1.5, marker='s', markersize=3, label='D_q')
        ax.set_xlabel('Sparsity'); ax.set_ylabel('Entropy', color='darkred')
        ax2.set_ylabel('Fractal Dim', color='darkblue')
        ax.set_title('Entropy + Fractal Dim')
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1 + l2, lb1 + lb2, fontsize=7); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out, f'pruning_percolation.{self.config.SAVE_FORMAT}')
        fig.savefig(path, dpi=self.config.FIGURE_DPI, bbox_inches='tight'); plt.close(fig)
        return path

    def _plot_dashboard(self, data: Dict, out: str) -> str:
        fig, axes = plt.subplots(4, 3, figsize=(self.config.FIGURE_WIDTH + 4, self.config.FIGURE_HEIGHT + 8))
        fig.suptitle('Complete Percolation Metrics Dashboard\n'
                     'kappa, delta, T_eff, hbar_eff, LC, SP(psi), h_bar, alpha, IPR, r, D_q, xi',
                     fontsize=13, fontweight='bold')
        lv = data['pruning_levels']
        phases = data['phase_per_level']
        pcol = {'perfect_crystal': 'blue', 'crystal_with_thermal_fluctuations': 'cyan',
                'hot_crystal': 'deepskyblue', 'cold_polycrystal': 'green',
                'warm_polycrystal': 'yellowgreen', 'cold_glass': 'orange', 'hot_glass': 'red'}
        colors = [pcol.get(p, 'gray') for p in phases]
        metrics = [
            (axes[0, 0], data['discretization_delta'], 'delta', 'delta (Discretization)', False),
            (axes[0, 1], data['kappa_proxy'], 'kappa', 'kappa (Gradient Cov)', True),
            (axes[0, 2], data['effective_temperature_proxy'], 'T_eff', 'T_eff (Temperature)', 'symlog'),
            (axes[1, 0], data['hbar_proxy'], 'hbar_eff', 'hbar_eff (Planck)', True),
            (axes[1, 1], data['local_complexity_proxy'], 'LC', 'LC (Local Complexity)', False),
            (axes[1, 2], data['purity_alpha'], 'alpha', 'SP(psi) / alpha', False),
            (axes[2, 0], data['entropy_proxy'], 'h_bar', 'h_bar (Entropy)', False),
            (axes[2, 1], data['ipr_proxy'], 'IPR', 'IPR (Participation)', False),
            (axes[2, 2], data['level_spacing_ratio'], 'r', 'r (Level Spacing)', False),
            (axes[3, 0], data['fractal_dimension_proxy'], 'D_q', 'Fractal Dimension', False),
            (axes[3, 1], data['coherence_length_proxy'], 'xi', 'Coherence Length', False)]
        for ax, vals, ylabel, title, yscale in metrics:
            ax.scatter(lv, vals, c=colors, s=30, zorder=5)
            ax.plot(lv, vals, 'k-', alpha=0.3, linewidth=0.5)
            ax.set_xlabel('Sparsity'); ax.set_ylabel(ylabel); ax.set_title(title)
            if yscale == True:
                ax.set_yscale('log')
            elif yscale == 'symlog':
                ax.set_yscale('symlog', linthresh=1e-15)
            ax.grid(True, alpha=0.3)
        ax = axes[3, 2]
        for pn in set(phases):
            ax.scatter([], [], c=pcol.get(pn, 'gray'), s=60, label=pn)
        ax.legend(loc='center', fontsize=8); ax.set_title('Phase Legend'); ax.axis('off')
        plt.tight_layout()
        path = os.path.join(out, f'full_metrics_dashboard.{self.config.SAVE_FORMAT}')
        fig.savefig(path, dpi=self.config.FIGURE_DPI, bbox_inches='tight'); plt.close(fig)
        return path

    def _plot_site(self, data: Dict, out: str) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(self.config.FIGURE_WIDTH, self.config.FIGURE_HEIGHT))
        fig.suptitle('Site Percolation Analysis', fontsize=14, fontweight='bold')
        t = data['thresholds']
        axes[0, 0].plot(t, data['occupation_fraction'], 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Threshold'); axes[0, 0].set_ylabel('Occupation'); axes[0, 0].set_title('Occupation')
        axes[0, 0].grid(True, alpha=0.3)
        for n in ['U', 'V', 'W']:
            axes[0, 1].plot(t, data['layer_occupation'][n], linewidth=1.5, label=n)
        axes[0, 1].set_xlabel('Threshold'); axes[0, 1].set_ylabel('Layer Occ'); axes[0, 1].set_title('Layer-wise')
        axes[0, 1].legend(); axes[0, 1].grid(True, alpha=0.3)
        axes[1, 0].plot(t, data['active_slots_per_threshold'], 'g-', linewidth=2)
        axes[1, 0].axhline(y=self.config.STRASSEN_RANK, color='r', linestyle='--', label=f'Rank={self.config.STRASSEN_RANK}')
        axes[1, 0].set_xlabel('Threshold'); axes[1, 0].set_ylabel('Slots'); axes[1, 0].set_title('Active Slots')
        axes[1, 0].legend(); axes[1, 0].grid(True, alpha=0.3)
        axes[1, 1].plot(t, data['structural_integrity'], 'r-', linewidth=2)
        if data['site_percolation_threshold'] is not None:
            axes[1, 1].axvline(x=data['site_percolation_threshold'], color='blue', linestyle='--',
                               label=f'p_c={data["site_percolation_threshold"]:.4f}')
        axes[1, 1].set_xlabel('Threshold'); axes[1, 1].set_ylabel('Integrity'); axes[1, 1].set_title('Structural Integrity')
        axes[1, 1].legend(); axes[1, 1].grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out, f'site_percolation.{self.config.SAVE_FORMAT}')
        fig.savefig(path, dpi=self.config.FIGURE_DPI, bbox_inches='tight'); plt.close(fig)
        return path

    def _plot_cluster(self, data: Dict, out: str) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(self.config.FIGURE_WIDTH, self.config.FIGURE_HEIGHT / 2))
        fig.suptitle('Cluster Size Distribution at Criticality', fontsize=14, fontweight='bold')
        sizes = data.get('cluster_sizes', [])
        if sizes:
            axes[0].hist(sizes, bins=min(30, max(len(set(sizes)), 1)), color='steelblue', edgecolor='black', alpha=0.7)
        axes[0].set_xlabel('Cluster Size'); axes[0].set_ylabel('Count'); axes[0].set_title('Histogram')
        axes[0].grid(True, alpha=0.3)
        dist = data.get('size_distribution', {})
        if dist:
            sv = sorted([int(k) for k in dist.keys()])
            nv = [dist.get(s, dist.get(str(s), 0)) for s in sv]
            sa, na = np.array(sv, dtype=float), np.array(nv, dtype=float)
            m = (sa > 0) & (na > 0)
            if np.sum(m) > 1:
                axes[1].loglog(sa[m], na[m], 'ko', markersize=4)
                if data.get('tau_estimate') is not None:
                    tau = data['tau_estimate']
                    xf = np.linspace(np.min(sa[m]), np.max(sa[m]), 100)
                    yf = xf ** (-tau) * (na[m][0] / (sa[m][0] ** (-tau)))
                    axes[1].loglog(xf, yf, 'r--', label=f'tau={tau:.3f} (2D:{187/91:.3f})')
                    axes[1].legend(fontsize=8)
        axes[1].set_xlabel('s'); axes[1].set_ylabel('n(s)'); axes[1].set_title('n(s)~s^{-tau}')
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        path = os.path.join(out, f'cluster_distribution.{self.config.SAVE_FORMAT}')
        fig.savefig(path, dpi=self.config.FIGURE_DPI, bbox_inches='tight'); plt.close(fig)
        return path


class PercolationReportGenerator:
    def __init__(self, config: PercolationConfiguration):
        self.config = config

    def generate_text_report(self, results: Dict[str, Any], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'percolation_report.txt')
        with open(path, 'w') as f:
            f.write("=" * 90 + "\n")
            f.write("PERCOLATION ANALYSIS REPORT\n")
            f.write("Strassen Algorithm Crystallization: Structural Connectivity Study\n")
            f.write("=" * 90 + "\n\n")
            f.write(f"Timestamp: {results.get('timestamp', 'N/A')}\n")
            f.write(f"Checkpoint: {results.get('checkpoint_path', 'N/A')}\n")
            f.write(f"Config: HIDDEN_DIM={self.config.HIDDEN_DIM}, MATRIX_SIZE={self.config.MATRIX_SIZE}\n")
            f.write(f"Total Parameters: {self.config.get_total_parameters()}\n\n")
            if 'bond_percolation' in results:
                bp = results['bond_percolation']
                f.write("-" * 90 + "\n")
                f.write("BOND PERCOLATION (Weight Graph Connectivity)\n")
                f.write("-" * 90 + "\n")
                f.write(f"  Percolation Threshold (p_c): {bp['percolation_threshold']}\n")
                f.write(f"  Graph Dimension: {bp['graph_dimension']}\n")
                if bp.get('critical_exponents'):
                    ce = bp['critical_exponents']
                    f.write(f"  beta: {ce.get('beta')} (2D ref: {ce.get('beta_2d_reference')}, MF: {ce.get('beta_mean_field_reference')})\n")
                    f.write(f"  gamma: {ce.get('gamma')} (2D ref: {ce.get('gamma_2d_reference')}, MF: {ce.get('gamma_mean_field_reference')})\n")
                f.write("\n")
            if 'pruning_percolation' in results:
                pp = results['pruning_percolation']
                f.write("-" * 90 + "\n")
                f.write("PRUNING PERCOLATION (Structural Yield Strength)\n")
                f.write("-" * 90 + "\n")
                f.write(f"  Yield Strength: {pp['yield_strength']}\n")
                f.write(f"  Collapse Sparsity: {pp['collapse_sparsity']}\n")
                f.write(f"  Elastic Regime End: {pp['elastic_regime_end']}\n")
                f.write(f"  Paper Reference: 50% stable, 51% collapse\n\n")
                hdr = (f"  {'Sparsity':>10} {'delta':>12} {'kappa':>12} {'Slots':>6} "
                       f"{'T_eff':>14} {'hbar':>14} {'LC':>10} {'alpha':>10} "
                       f"{'IPR':>10} {'r':>10} {'Phase':>30}\n")
                f.write(hdr)
                f.write("  " + "-" * 158 + "\n")
                for i in range(len(pp['pruning_levels'])):
                    f.write(f"  {pp['pruning_levels'][i]:>10.4f} "
                            f"{pp['discretization_delta'][i]:>12.6f} "
                            f"{pp['kappa_proxy'][i]:>12.4f} "
                            f"{pp['active_slots'][i]:>6d} "
                            f"{pp['effective_temperature_proxy'][i]:>14.6e} "
                            f"{pp['hbar_proxy'][i]:>14.6e} "
                            f"{pp['local_complexity_proxy'][i]:>10.4f} "
                            f"{pp['purity_alpha'][i]:>10.4f} "
                            f"{pp['ipr_proxy'][i]:>10.6f} "
                            f"{pp['level_spacing_ratio'][i]:>10.6f} "
                            f"{pp['phase_per_level'][i]:>30}\n")
                f.write("\n")
            if 'site_percolation' in results:
                sp = results['site_percolation']
                f.write("-" * 90 + "\n")
                f.write("SITE PERCOLATION\n")
                f.write("-" * 90 + "\n")
                f.write(f"  Site Percolation Threshold: {sp['site_percolation_threshold']}\n\n")
            if 'cluster_analysis' in results:
                ca = results['cluster_analysis']
                f.write("-" * 90 + "\n")
                f.write("CLUSTER DISTRIBUTION AT CRITICALITY\n")
                f.write("-" * 90 + "\n")
                f.write(f"  Threshold: {ca.get('threshold')}\n")
                f.write(f"  Clusters: {ca.get('num_clusters')}\n")
                f.write(f"  Largest: {ca.get('largest_cluster')} ({ca.get('largest_cluster_fraction'):.4f})\n")
                f.write(f"  tau: {ca.get('tau_estimate')} (2D ref: {ca.get('tau_2d_reference'):.4f})\n")
                f.write(f"  Critical: {ca.get('is_critical')}\n\n")
            if 'universality' in results:
                u = results['universality']
                f.write("-" * 90 + "\n")
                f.write("UNIVERSALITY CLASS\n")
                f.write("-" * 90 + "\n")
                f.write(f"  Best Match: {u.get('best_match')}\n")
                f.write(f"  Confidence: {u.get('confidence')}\n")
                f.write(f"  Distances: {u.get('distances')}\n\n")
            f.write("=" * 90 + "\nEND OF REPORT\n" + "=" * 90 + "\n")
        return path

    def generate_json_report(self, results: Dict[str, Any], output_dir: str) -> str:
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, 'percolation_results.json')
        with open(path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        return path


class PercolationAnalysisPipeline:
    def __init__(self, config: PercolationConfiguration):
        self.config = config
        self.graph_ctor = WeightGraphConstructor(config)
        self.bond_az = BondPercolationAnalyzer(config)
        self.site_az = SitePercolationAnalyzer(config)
        self.prune_az = PruningPercolationAnalyzer(config)
        self.cluster_az = ClusterSizeDistributionAnalyzer(config)
        self.univ_az = PercolationUniversalityAnalyzer(config)
        self.ckpt_mgr = PercolationCheckpointManager(config)
        self.viz = PercolationVisualizationEngine(config)
        self.report = PercolationReportGenerator(config)
        self.migrator = CheckpointMigrator()

    def _load_weights(self, checkpoint_path: str) -> Dict[str, np.ndarray]:
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required to load .pt checkpoints.")
        ckpt = _safe_torch_load(checkpoint_path)
        weights = self.migrator.migrate(ckpt, self.config)
        if weights is None:
            raise RuntimeError(
                f"Failed to extract U/V/W tensors from checkpoint: {checkpoint_path}. "
                f"Keys found: {list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt).__name__}")
        return weights

    def process_checkpoint(self, checkpoint_path: str, output_dir: str) -> Dict[str, Any]:
        os.makedirs(output_dir, exist_ok=True)
        print(f"Loading checkpoint: {checkpoint_path}")
        weights = self._load_weights(checkpoint_path)
        results = {
            'checkpoint_path': checkpoint_path,
            'timestamp': datetime.now().isoformat(),
            'configuration': {
                'hidden_dim': self.config.HIDDEN_DIM,
                'matrix_size': self.config.MATRIX_SIZE,
                'total_parameters': self.config.get_total_parameters(),
                'expansion_scales': list(self.config.EXPANSION_SCALES)}}
        print("  [1/6] Constructing weight graphs...")
        adj = self.graph_ctor.construct_adjacency_from_weights(weights)
        corr = self.graph_ctor.construct_weight_correlation_graph(weights)
        slot = self.graph_ctor.construct_slot_interaction_graph(weights)
        self._maybe_save(results, output_dir)
        print("  [2/6] Bond percolation analysis...")
        thr = self.config.get_percolation_thresholds()
        results['bond_percolation'] = self.bond_az.analyze(adj, thr)
        results['bond_percolation_correlation'] = self.bond_az.analyze(corr, thr)
        results['bond_percolation_slots'] = self.bond_az.analyze(slot, thr)
        self._maybe_save(results, output_dir)
        print("  [3/6] Site percolation analysis...")
        results['site_percolation'] = self.site_az.analyze(weights, thr)
        self._maybe_save(results, output_dir)
        print("  [4/6] Pruning percolation (full metric sweep)...")
        results['pruning_percolation'] = self.prune_az.analyze(weights)
        self._maybe_save(results, output_dir)
        print("  [5/6] Cluster distribution at criticality...")
        pc = results['bond_percolation'].get('percolation_threshold')
        if pc is not None:
            results['cluster_analysis'] = self.cluster_az.analyze_at_threshold(adj, pc)
        else:
            results['cluster_analysis'] = self.cluster_az.analyze_at_threshold(adj, self.config.PERCOLATION_2D_BOND_PC)
        exp = results['bond_percolation'].get('critical_exponents', {})
        results['universality'] = self.univ_az.classify_universality(exp)
        self._maybe_save(results, output_dir)
        print("  [6/6] Generating visualizations and reports...")
        results['figure_paths'] = self.viz.generate_all_figures(results, output_dir)
        results['report_path'] = self.report.generate_text_report(results, output_dir)
        results['json_path'] = self.report.generate_json_report(results, output_dir)
        self.ckpt_mgr.save(results, output_dir)
        print(f"\n  Complete. Output: {output_dir}")
        print(f"  Report: {results['report_path']}")
        print(f"  Figures: {len(results['figure_paths'])}")
        return results

    def process_directory(self, checkpoint_dir: str, n_latest: Optional[int],
                          output_dir: str) -> List[Dict[str, Any]]:
        cps = sorted(glob.glob(os.path.join(checkpoint_dir, '*.pt')),
                     key=os.path.getmtime, reverse=True)
        if not cps:
            print(f"No checkpoints in {checkpoint_dir}"); return []
        if n_latest is not None:
            cps = cps[:n_latest]
        print(f"\nProcessing {len(cps)} checkpoints...\n")
        all_res = []
        for i, cp in enumerate(cps):
            print(f"\n[{i + 1}/{len(cps)}] {cp}")
            try:
                r = self.process_checkpoint(cp, os.path.join(output_dir, Path(cp).stem))
                all_res.append(r)
            except Exception as e:
                print(f"  Error: {e}")
                import traceback; traceback.print_exc()
        if all_res:
            self._comparative_summary(all_res, output_dir)
        return all_res

    def _maybe_save(self, results: Dict, output_dir: str):
        if self.ckpt_mgr.should_save():
            self.ckpt_mgr.save(results, output_dir)

    def _comparative_summary(self, all_res: List[Dict], output_dir: str):
        os.makedirs(output_dir, exist_ok=True)
        summary = {
            'total_checkpoints': len(all_res),
            'timestamp': datetime.now().isoformat(),
            'percolation_thresholds': [r.get('bond_percolation', {}).get('percolation_threshold') for r in all_res],
            'yield_strengths': [r.get('pruning_percolation', {}).get('yield_strength') for r in all_res],
            'collapse_sparsities': [r.get('pruning_percolation', {}).get('collapse_sparsity') for r in all_res],
            'universality_matches': [r.get('universality', {}).get('best_match') for r in all_res]}
        valid_pc = [x for x in summary['percolation_thresholds'] if x is not None]
        valid_ys = [x for x in summary['yield_strengths'] if x is not None]
        valid_cs = [x for x in summary['collapse_sparsities'] if x is not None]
        summary['mean_percolation_threshold'] = float(np.mean(valid_pc)) if valid_pc else None
        summary['mean_yield_strength'] = float(np.mean(valid_ys)) if valid_ys else None
        summary['mean_collapse_sparsity'] = float(np.mean(valid_cs)) if valid_cs else None
        with open(os.path.join(output_dir, 'percolation_summary.json'), 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        print(f"\n  Comparative summary saved.")
        if valid_pc:
            print(f"  Mean p_c: {summary['mean_percolation_threshold']:.4f}")
        if valid_ys:
            print(f"  Mean yield strength: {summary['mean_yield_strength']:.4f}")
        if valid_cs:
            print(f"  Mean collapse sparsity: {summary['mean_collapse_sparsity']:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description='Percolation Analysis for Strassen Algorithm Crystallization Checkpoints')
    parser.add_argument('checkpoint', nargs='?', default=None,
                        help='Path to checkpoint file (.pt)')
    parser.add_argument('--all', action='store_true',
                        help='Process all checkpoints in directory')
    parser.add_argument('--latest', type=int, default=None,
                        help='Process N latest checkpoints')
    parser.add_argument('--dir', default='checkpoints',
                        help='Checkpoint directory')
    parser.add_argument('--output', default='percolation_analysis',
                        help='Output directory')
    parser.add_argument('--hidden-dim', type=int, default=8,
                        help='Hidden dimension of model')
    parser.add_argument('--matrix-size', type=int, default=2,
                        help='Matrix size (2 for 2x2 Strassen)')
    args = parser.parse_args()
    config = PercolationConfiguration(
        HIDDEN_DIM=args.hidden_dim,
        MATRIX_SIZE=args.matrix_size,
        INPUT_DIM=args.matrix_size * args.matrix_size)
    pipeline = PercolationAnalysisPipeline(config)
    if args.checkpoint:
        if os.path.isfile(args.checkpoint):
            pipeline.process_checkpoint(args.checkpoint, args.output)
        else:
            print(f"Error: Not found: {args.checkpoint}")
    elif args.all or args.latest is not None:
        pipeline.process_directory(args.dir, args.latest, args.output)
    else:
        print("No action specified. Use --help for usage.")


if __name__ == '__main__':
    main()
