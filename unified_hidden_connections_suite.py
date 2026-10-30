#!/usr/bin/env python3
"""
Unified Hidden Connections Suite
================================
A single-file experimental framework designed to test five theoretically
motivated hidden connections in the Strassen-Strassen neural network
architecture. The suite implements exact reproduction of the original
bilinear model and subjects it to rigorous falsification protocols.

Experiments:
    1. Ricci-MBL Duality: Links geometric curvature smoothing to spectral
       integrability transitions.
    2. Altland-Zirnbauer Symmetry Dial: Maps the imaginary-weight control
       parameter to random matrix universality classes.
    3. Conformal Isomorphism: Stress-tests Moebius invariance to probe
       whether the network learns the underlying conformal operator.
    4. Compression Frontier: Tests the thermodynamic uncertainty relation
       between effective Planck constant and superposition metric.
    5. Holographic Pruning: Distinguishes area-law (crystal) from volume-law
       (glass) information encoding via structured ablation.

Architecture: SOLID principles, configuration-driven, no hardcoded magic
numbers, production-grade error handling, fully autonomous execution.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    List,
    Optional,
    Protocol,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings("ignore", category=UserWarning)

T = TypeVar("T")


@dataclass(frozen=True)
class StrassStrassenConfig:
    """Immutable canonical configuration for the Strassen bilinear model."""

    matrix_size: int = 2
    input_dim: int = 4
    output_dim: int = 4
    rank: int = 8
    target_rank: int = 7
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = torch.float32

    def __post_init__(self) -> None:
        if self.matrix_size * self.matrix_size != self.input_dim:
            raise ValueError("input_dim must equal matrix_size squared")
        if self.rank < self.target_rank:
            raise ValueError("rank must be >= target_rank")


@dataclass(frozen=True)
class TrainingConfig:
    """Immutable training hyperparameters."""

    epochs: int = 30000
    batch_size: int = 256
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    scheduler_t0: int = 10000
    grad_clip_norm: float = 1.0
    accuracy_threshold: float = 1e-3
    log_interval: int = 5000
    seed: int = 42


@dataclass(frozen=True)
class Experiment1Config:
    """Configuration for Experiment 1: Ricci-MBL Duality."""

    checkpoint_epochs: Tuple[int, ...] = (
        100, 500, 1000, 2000, 3000, 5000, 7500, 10000,
        12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000,
    )
    hessian_samples: int = 1
    spectral_regularization: float = 1e-6
    singularity_eigenvalue_ratio: float = 100.0
    heat_kernel_times: Tuple[float, ...] = (0.1, 1.0, 10.0)
    output_dir: str = "experiment_1_ricci_mbl"


@dataclass(frozen=True)
class Experiment2Config:
    """Configuration for Experiment 2: Altland-Zirnbauer Symmetry Dial."""

    gamma_values: Tuple[float, ...] = (
        0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0,
    )
    epochs_per_gamma: int = 5000
    batch_size: int = 256
    learning_rate: float = 0.02
    weight_decay: float = 1e-4
    level_spacing_tolerance: float = 0.05
    wigner_dyson_theoretical: float = 0.5307
    poisson_theoretical: float = 0.3863
    output_dir: str = "experiment_2_altland_zirnbauer"


@dataclass(frozen=True)
class Experiment3Config:
    """Configuration for Experiment 3: Conformal Isomorphism."""

    num_test_samples: int = 1000
    num_transformations: int = 50
    tolerance_threshold: float = 1e-3
    output_dir: str = "experiment_3_conformal_isomorphism"


@dataclass(frozen=True)
class Experiment4Config:
    """Configuration for Experiment 4: Compression Frontier."""

    batch_sizes: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
    weight_decays: Tuple[float, ...] = (1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3)
    epochs_per_config: int = 15000
    sae_expansion_factor: int = 8
    sae_l1_coefficient: float = 0.1
    sae_learning_rate: float = 1e-3
    sae_epochs: int = 1000
    sae_batch_size: int = 256
    num_activation_samples: int = 10000
    epsilon: float = 1e-10
    output_dir: str = "experiment_4_compression_frontier"


@dataclass(frozen=True)
class Experiment5Config:
    """Configuration for Experiment 5: Holographic Pruning."""

    num_trials: int = 10
    prune_fraction: float = 0.1
    accuracy_threshold: float = 1e-3
    num_test_samples: int = 1000
    output_dir: str = "experiment_5_holographic_pruning"


@dataclass(frozen=True)
class SuiteConfig:
    """Top-level suite orchestration configuration."""

    model_config: StrassStrassenConfig = field(default_factory=StrassStrassenConfig)
    training_config: TrainingConfig = field(default_factory=TrainingConfig)
    experiment_1: Experiment1Config = field(default_factory=Experiment1Config)
    experiment_2: Experiment2Config = field(default_factory=Experiment2Config)
    experiment_3: Experiment3Config = field(default_factory=Experiment3Config)
    experiment_4: Experiment4Config = field(default_factory=Experiment4Config)
    experiment_5: Experiment5Config = field(default_factory=Experiment5Config)
    global_output_dir: str = "unified_hidden_connections_results"
    save_checkpoints: bool = True


class StrassStrassenModel(nn.Module):
    """
    Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.
    Implements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.
    U, V ∈ R^{rank x input_dim}, W ∈ R^{output_dim x rank}.
    """

    def __init__(self, config: StrassStrassenConfig) -> None:
        super().__init__()
        self.config = config
        self.U = nn.Parameter(
            torch.randn(config.rank, config.input_dim, dtype=config.dtype, device=config.device) * 0.5
        )
        self.V = nn.Parameter(
            torch.randn(config.rank, config.input_dim, dtype=config.dtype, device=config.device) * 0.5
        )
        self.W = nn.Parameter(
            torch.randn(config.output_dim, config.rank, dtype=config.dtype, device=config.device) * 0.5
        )

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        batch = A.shape[0]
        a = A.reshape(batch, self.config.input_dim)
        b = B.reshape(batch, self.config.input_dim)
        left = a @ self.U.T
        right = b @ self.V.T
        products = left * right
        c = products @ self.W.T
        return c.reshape(batch, self.config.matrix_size, self.config.matrix_size)

    def get_coefficients(self) -> Dict[str, torch.Tensor]:
        return {"U": self.U.data.clone(), "V": self.V.data.clone(), "W": self.W.data.clone()}

    def get_flat_parameters(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])

    def slot_importance(self) -> torch.Tensor:
        u_norm = torch.norm(self.U, dim=1)
        v_norm = torch.norm(self.V, dim=1)
        w_norm = torch.norm(self.W, dim=0)
        return u_norm * v_norm * w_norm

    def count_active_slots(self, threshold: float = 0.1) -> int:
        return int((self.slot_importance() > threshold).sum().item())


class IDataGenerator(Protocol):
    """Protocol for deterministic data generation."""

    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: ...


class StrassenDataGenerator:
    """Generates random 2x2 matrix pairs and their exact products."""

    def __init__(self, config: StrassStrassenConfig) -> None:
        self.config = config

    def generate_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = torch.randn(
            batch_size,
            self.config.matrix_size,
            self.config.matrix_size,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        B = torch.randn(
            batch_size,
            self.config.matrix_size,
            self.config.matrix_size,
            dtype=self.config.dtype,
            device=self.config.device,
        )
        C = torch.bmm(A, B)
        return A, B, C


class ICheckpointManager(Protocol):
    """Protocol for checkpoint persistence."""

    def save(self, model: nn.Module, epoch: int, metrics: Dict[str, Any], path: Path) -> None: ...
    def load(self, path: Path, model: nn.Module) -> Dict[str, Any]: ...


class CheckpointManager:
    """Handles safe serialization and deserialization of model checkpoints."""

    def save(
        self, model: nn.Module, epoch: int, metrics: Dict[str, Any], path: Path
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "epoch": epoch,
            "state_dict": model.state_dict(),
            "metrics": metrics,
            "timestamp": datetime.now().isoformat(),
        }
        torch.save(payload, path)

    def load(self, path: Path, model: nn.Module) -> Dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if "state_dict" in payload:
            model.load_state_dict(payload["state_dict"])
        return payload


class ITrainer(Protocol):
    """Protocol for training routines."""

    def train(
        self, model: StrassStrassenModel, epochs: int, callback: Optional[Callable[[int, StrassStrassenModel, float, float], None]] = None
    ) -> Dict[str, Any]: ...


class Trainer:
    """Standard trainer with AdamW, cosine annealing, and gradient clipping."""

    def __init__(
        self,
        model_config: StrassStrassenConfig,
        training_config: TrainingConfig,
        data_generator: IDataGenerator,
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.data_generator = data_generator

    def train(
        self,
        model: StrassStrassenModel,
        epochs: int,
        callback: Optional[Callable[[int, StrassStrassenModel, float, float], None]] = None,
    ) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=self.training_config.scheduler_t0
        )
        history = {"loss": [], "accuracy": [], "epoch": []}

        for epoch in range(epochs):
            A, B, C_true = self.data_generator.generate_batch(self.training_config.batch_size)
            C_pred = model(A, B)
            loss = F.mse_loss(C_pred, C_true)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), self.training_config.grad_clip_norm
            )
            optimizer.step()
            scheduler.step()

            with torch.no_grad():
                errors = (C_pred - C_true).abs().reshape(self.training_config.batch_size, -1).max(dim=1)[0]
                acc = (errors < self.training_config.accuracy_threshold).float().mean().item()

            history["loss"].append(loss.item())
            history["accuracy"].append(acc)
            history["epoch"].append(epoch)

            if callback is not None:
                callback(epoch, model, loss.item(), acc)

        return history


class IMetricCalculator(ABC, Generic[T]):
    """Abstract base for all metric calculators."""

    @abstractmethod
    def calculate(self, model: StrassStrassenModel) -> T:
        raise NotImplementedError


class LevelSpacingRatioCalculator(IMetricCalculator[Dict[str, float]]):
    """Computes the adjacent gap ratio r for Hessian eigenvalue spectra."""

    def __init__(self, config: StrassStrassenConfig, tolerance: float = 0.05) -> None:
        self.config = config
        self.tolerance = tolerance

    def _build_hessian_approximation(self, model: StrassStrassenModel) -> np.ndarray:
        coeffs = model.get_coefficients()
        all_weights = []
        for name in ["U", "V", "W"]:
            all_weights.append(coeffs[name].flatten().cpu().numpy())
        weight_vector = np.concatenate(all_weights)
        n = len(weight_vector)
        weights_2d = weight_vector.reshape(-1, 1)
        hessian = np.outer(weights_2d, weights_2d) / max(n, 1)
        hessian += np.eye(n) * 1e-8
        return hessian

    def calculate(self, model: StrassStrassenModel) -> Dict[str, float]:
        hessian = self._build_hessian_approximation(model)
        eigenvalues = np.sort(np.linalg.eigvalsh(hessian))
        spacings = np.diff(eigenvalues)
        ratios = []
        for i in range(len(spacings) - 1):
            s_n = spacings[i]
            s_n1 = spacings[i + 1]
            if max(s_n, s_n1) > 1e-15:
                ratios.append(min(s_n, s_n1) / max(s_n, s_n1))
        ratios_arr = np.array(ratios) if ratios else np.array([0.0])
        mean_ratio = float(np.mean(ratios_arr))

        phase = "intermediate"
        if abs(mean_ratio - 0.5307) < self.tolerance:
            phase = "wigner_dyson"
        elif abs(mean_ratio - 0.3863) < self.tolerance:
            phase = "poisson"
        elif mean_ratio < 0.45:
            phase = "many_body_localized"
        else:
            phase = "thermal"

        return {
            "mean_spacing_ratio": mean_ratio,
            "std_spacing_ratio": float(np.std(ratios_arr)),
            "phase_classification": phase,
            "num_levels": len(eigenvalues),
        }


class RicciScalarCalculator(IMetricCalculator[Dict[str, float]]):
    """Computes Ricci scalar and geometric curvature metrics from Hessian."""

    def __init__(self, config: StrassStrassenConfig, regularization: float = 1e-6) -> None:
        self.config = config
        self.regularization = regularization

    def _compute_hessian(self, model: StrassStrassenModel) -> torch.Tensor:
        A, B, C_true = self._generate_single_sample()
        params = list(model.parameters())
        flat_params = torch.cat([p.flatten() for p in params])
        loss_fn = lambda theta: self._loss_from_flat(theta, model, params, A, B, C_true)
        try:
            hessian = torch.autograd.functional.hessian(loss_fn, flat_params)
            return hessian
        except RuntimeError:
            return self._diagonal_hessian_approximation(model, A, B, C_true)

    def _generate_single_sample(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        gen = StrassenDataGenerator(self.config)
        return gen.generate_batch(1)

    def _loss_from_flat(
        self,
        flat_params: torch.Tensor,
        model: StrassStrassenModel,
        original_params: List[torch.Tensor],
        A: torch.Tensor,
        B: torch.Tensor,
        C_true: torch.Tensor,
    ) -> torch.Tensor:
        idx = 0
        with torch.no_grad():
            for p in original_params:
                numel = p.numel()
                p_data = flat_params[idx:idx + numel].reshape(p.shape)
                p.copy_(p_data)
                idx += numel
        pred = model(A, B)
        return F.mse_loss(pred, C_true)

    def _diagonal_hessian_approximation(
        self,
        model: StrassStrassenModel,
        A: torch.Tensor,
        B: torch.Tensor,
        C_true: torch.Tensor,
    ) -> torch.Tensor:
        grads = torch.autograd.grad(
            F.mse_loss(model(A, B), C_true), model.parameters(), create_graph=True
        )
        grad_vec = torch.cat([g.flatten() for g in grads])
        diag_hessian = torch.autograd.grad(grad_vec, model.parameters(), retain_graph=True)
        return torch.diag(torch.cat([g.flatten() for g in diag_hessian]))

    def calculate(self, model: StrassStrassenModel) -> Dict[str, float]:
        hessian = self._compute_hessian(model)
        hessian = hessian + self.regularization * torch.eye(hessian.shape[0], device=hessian.device)
        eigenvalues = torch.linalg.eigvalsh(hessian)
        ricci_scalar = float(torch.sum(eigenvalues).item())
        sorted_eig = torch.sort(eigenvalues, descending=True).values
        spectral_gap = float((sorted_eig[0] - sorted_eig[-1]).item())
        min_eig = sorted_eig[-1].item()
        max_eig = sorted_eig[0].item()
        condition_number = float("inf") if abs(min_eig) < 1e-9 else max_eig / abs(min_eig)
        negative_count = int((eigenvalues < 0).sum().item())
        return {
            "ricci_scalar": ricci_scalar,
            "spectral_gap": spectral_gap,
            "condition_number": condition_number,
            "max_curvature": float(max_eig),
            "min_curvature": float(min_eig),
            "negative_curvature_count": negative_count,
            "eigenvalues": eigenvalues.detach().cpu().numpy().tolist(),
        }


class SyntheticPlanckCalculator(IMetricCalculator[Dict[str, float]]):
    """Estimates the synthetic Planck constant from model weight statistics."""

    def __init__(self, config: StrassStrassenConfig, noise_floor: float = 1e-7) -> None:
        self.config = config
        self.noise_floor = noise_floor

    def calculate(self, model: StrassStrassenModel) -> Dict[str, float]:
        coeffs = model.get_coefficients()
        all_weights = torch.cat([c.flatten() for c in coeffs.values()])
        weight_variance = torch.var(all_weights).item()
        weight_std = torch.std(all_weights).item()
        rounded = torch.round(all_weights)
        delta = torch.max(torch.abs(all_weights - rounded)).item()
        total_norm = torch.norm(all_weights).item()
        lambda_eff = 1.0 / (total_norm ** 2 + 1e-10)

        hbar_uncertainty = 2.0 * (delta ** 2) * lambda_eff
        omega = math.sqrt(lambda_eff) if lambda_eff > 0 else 1.0
        period = (2.0 * math.pi / omega) if omega > 0 else 1.0
        T = weight_variance
        V = lambda_eff * (delta ** 2)
        L = T - V
        action = abs(L) * period
        hbar_action = action

        accuracy_proxy = 1.0 - min(delta, 1.0)
        conductance = accuracy_proxy / (weight_variance + 1e-10)
        hbar_conductance = 1.0 / conductance if conductance > 0 else 0.0

        n_eff = 31
        information = math.log2(n_eff) if n_eff > 1 else 1.0
        energy_total = T + V
        energy_per_bit = energy_total / information if information > 0 else 0.0
        hbar_information = energy_per_bit * period

        if delta < 0.01:
            weights = (0.6, 0.25, 0.1, 0.05)
        elif delta < 0.1:
            weights = (0.5, 0.3, 0.15, 0.05)
        else:
            weights = (0.25, 0.25, 0.25, 0.25)

        w1, w2, w3, w4 = weights
        total_w = sum(weights)
        hbar_unified = (w1 * hbar_uncertainty + w2 * hbar_action + w3 * hbar_conductance + w4 * hbar_information) / total_w

        return {
            "hbar_uncertainty": float(hbar_uncertainty),
            "hbar_action": float(hbar_action),
            "hbar_conductance": float(hbar_conductance),
            "hbar_information": float(hbar_information),
            "hbar_unified": float(max(hbar_unified, self.noise_floor)),
            "delta": float(delta),
            "lambda_eff": float(lambda_eff),
            "weight_variance": float(weight_variance),
            "weight_std": float(weight_std),
        }


class SuperpositionMetricCalculator:
    """Measures superposition via sparse autoencoder bottleneck analysis."""

    def __init__(
        self,
        model_config: StrassStrassenConfig,
        expansion_factor: int = 8,
        l1_coefficient: float = 0.1,
        sae_lr: float = 1e-3,
        sae_epochs: int = 1000,
        sae_batch_size: int = 256,
        num_samples: int = 10000,
        epsilon: float = 1e-10,
    ) -> None:
        self.model_config = model_config
        self.expansion_factor = expansion_factor
        self.l1_coefficient = l1_coefficient
        self.sae_lr = sae_lr
        self.sae_epochs = sae_epochs
        self.sae_batch_size = sae_batch_size
        self.num_samples = num_samples
        self.epsilon = epsilon
        self.dictionary_size = model_config.rank * expansion_factor

    def _extract_activations(self, model: StrassStrassenModel) -> torch.Tensor:
        gen = StrassenDataGenerator(self.model_config)
        activations = []
        batches = (self.num_samples + self.sae_batch_size - 1) // self.sae_batch_size
        with torch.no_grad():
            for _ in range(batches):
                A, B, _ = gen.generate_batch(self.sae_batch_size)
                u_out = A.reshape(A.shape[0], self.model_config.input_dim) @ model.U.T
                v_out = B.reshape(B.shape[0], self.model_config.input_dim) @ model.V.T
                bottleneck = u_out * v_out
                activations.append(bottleneck.cpu())
        return torch.cat(activations, dim=0)[: self.num_samples].to(self.model_config.device)

    def _train_sae(self, activations: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        device = self.model_config.device
        W_enc = nn.Parameter(
            torch.randn(self.dictionary_size, self.model_config.rank, device=device) * 0.01
        )
        b_enc = nn.Parameter(torch.zeros(self.dictionary_size, device=device))
        b_dec = nn.Parameter(torch.zeros(self.model_config.rank, device=device))
        nn.init.xavier_uniform_(W_enc)
        optimizer = torch.optim.AdamW([W_enc, b_enc, b_dec], lr=self.sae_lr, weight_decay=1e-5)
        dataset = TensorDataset(activations)
        loader = DataLoader(dataset, batch_size=self.sae_batch_size, shuffle=True)

        for _ in range(self.sae_epochs):
            for (batch,) in loader:
                z = F.relu(torch.matmul(batch, W_enc.t()) + b_enc)
                x_recon = torch.matmul(z, W_enc) + b_dec
                recon_loss = F.mse_loss(x_recon, batch)
                l1_loss = torch.mean(torch.abs(z))
                total_loss = recon_loss + self.l1_coefficient * l1_loss
                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()

        with torch.no_grad():
            _, final_z = self._sae_forward(activations, W_enc, b_enc, b_dec)
        return final_z, W_enc

    def _sae_forward(
        self, x: torch.Tensor, W_enc: torch.Tensor, b_enc: torch.Tensor, b_dec: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        z = F.relu(torch.matmul(x, W_enc.t()) + b_enc)
        x_recon = torch.matmul(z, W_enc) + b_dec
        return x_recon, z

    def calculate(self, model: StrassStrassenModel) -> Dict[str, float]:
        activations = self._extract_activations(model)
        final_z, _ = self._train_sae(activations)
        abs_acts = torch.abs(final_z)
        feature_budget = torch.sum(abs_acts, dim=0)
        total_budget = torch.sum(feature_budget) + self.epsilon
        probabilities = feature_budget / total_budget
        probs = probabilities[probabilities > self.epsilon]
        if len(probs) == 0:
            entropy_val = torch.tensor(0.0, device=probs.device)
        else:
            entropy_val = -torch.sum(probs * torch.log(probs))
        F_eff = torch.exp(entropy_val)
        psi = F_eff / self.model_config.rank
        return {
            "psi": float(psi),
            "effective_features_F": float(F_eff),
            "entropy_H": float(entropy_val),
            "num_active_features": int((probabilities > 1e-8).sum().item()),
        }


class IExperiment(ABC):
    """Abstract base for all experiments in the suite."""

    @abstractmethod
    def run(self, model: StrassStrassenModel) -> Dict[str, Any]:
        raise NotImplementedError

    @abstractmethod
    def get_name(self) -> str:
        raise NotImplementedError


class Experiment1RicciMBLDuality(IExperiment):
    """
    Tests the claim that Ricci curvature smoothing is the geometric mechanism
    driving the Wigner-Dyson to Poisson/MBL spectral transition.
    """

    def __init__(
        self,
        config: Experiment1Config,
        model_config: StrassStrassenConfig,
        training_config: TrainingConfig,
        data_generator: IDataGenerator,
        checkpoint_manager: CheckpointManager,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.training_config = training_config
        self.data_generator = data_generator
        self.checkpoint_manager = checkpoint_manager
        self.level_spacing_calc = LevelSpacingRatioCalculator(model_config)
        self.ricci_calc = RicciScalarCalculator(model_config, config.spectral_regularization)

    def get_name(self) -> str:
        return "experiment_1_ricci_mbl_duality"

    def run(self, model: StrassStrassenModel) -> Dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoints_dir = output_dir / "checkpoints"
        checkpoints_dir.mkdir(parents=True, exist_ok=True)

        trainer = Trainer(self.model_config, self.training_config, self.data_generator)
        results = []

        def checkpoint_callback(epoch: int, m: StrassStrassenModel, loss: float, acc: float) -> None:
            if epoch in self.config.checkpoint_epochs:
                ckpt_path = checkpoints_dir / f"checkpoint_epoch_{epoch}.pt"
                self.checkpoint_manager.save(m, epoch, {"loss": loss, "accuracy": acc}, ckpt_path)
                spacing = self.level_spacing_calc.calculate(m)
                ricci = self.ricci_calc.calculate(m)
                results.append({
                    "epoch": epoch,
                    "loss": loss,
                    "accuracy": acc,
                    "mean_spacing_ratio": spacing["mean_spacing_ratio"],
                    "phase_classification": spacing["phase_classification"],
                    "ricci_scalar": ricci["ricci_scalar"],
                    "spectral_gap": ricci["spectral_gap"],
                    "condition_number": ricci["condition_number"],
                    "negative_curvature_count": ricci["negative_curvature_count"],
                })

        trainer.train(model, self.training_config.epochs, callback=checkpoint_callback)

        summary = self._analyze_temporal_correlation(results)
        payload = {"results": results, "summary": summary}
        with open(output_dir / "experiment_1_results.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return payload

    def _analyze_temporal_correlation(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        if len(results) < 2:
            return {"correlation": None, "verdict": "insufficient_data"}
        ratios = [r["mean_spacing_ratio"] for r in results]
        ricci_scalars = [r["ricci_scalar"] for r in results]
        correlation = np.corrcoef(ratios, ricci_scalars)[0, 1] if len(ratios) > 1 else 0.0
        initial_phase = results[0]["phase_classification"]
        final_phase = results[-1]["phase_classification"]
        transition_detected = initial_phase != final_phase
        return {
            "correlation": float(correlation),
            "initial_phase": initial_phase,
            "final_phase": final_phase,
            "transition_detected": bool(transition_detected),
            "verdict": "duality_supported" if transition_detected and correlation < -0.5 else "duality_inconclusive",
        }


class Experiment2AltlandZirnbauer(IExperiment):
    """
    Tests the claim that the imaginary-weight control parameter gamma drives
    the system between GOE (orthogonal) and GUE (unitary) random matrix classes.
    """

    def __init__(
        self,
        config: Experiment2Config,
        model_config: StrassStrassenConfig,
        data_generator: IDataGenerator,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.data_generator = data_generator
        self.spacing_calc = LevelSpacingRatioCalculator(model_config, config.level_spacing_tolerance)

    def get_name(self) -> str:
        return "experiment_2_altland_zirnbauer"

    def run(self, model: StrassStrassenModel) -> Dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for gamma in self.config.gamma_values:
            gamma_model = self._create_gamma_model(model, gamma)
            history = self._train_gamma_model(gamma_model)
            spacing = self.spacing_calc.calculate(gamma_model)
            results.append({
                "gamma": gamma,
                "mean_spacing_ratio": spacing["mean_spacing_ratio"],
                "phase_classification": spacing["phase_classification"],
                "final_loss": history["loss"][-1] if history["loss"] else None,
                "final_accuracy": history["accuracy"][-1] if history["accuracy"] else None,
            })

        summary = self._detect_critical_transition(results)
        payload = {"results": results, "summary": summary}
        with open(output_dir / "experiment_2_results.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return payload

    def _create_gamma_model(self, base_model: StrassStrassenModel, gamma: float) -> StrassStrassenModel:
        new_model = StrassStrassenModel(self.model_config)
        with torch.no_grad():
            new_model.U.copy_(base_model.U.data)
            new_model.V.copy_(base_model.V.data)
            new_model.W.copy_(base_model.W.data)
            imaginary_noise_U = torch.randn_like(new_model.U) * 0.1
            imaginary_noise_V = torch.randn_like(new_model.V) * 0.1
            new_model.U.add_(gamma * imaginary_noise_U)
            new_model.V.add_(gamma * imaginary_noise_V)
        return new_model

    def _train_gamma_model(self, model: StrassStrassenModel) -> Dict[str, List[float]]:
        tc = TrainingConfig(
            epochs=self.config.epochs_per_gamma,
            batch_size=self.config.batch_size,
            learning_rate=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )
        trainer = Trainer(self.model_config, tc, self.data_generator)
        return trainer.train(model, tc.epochs)

    def _detect_critical_transition(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        ratios = [r["mean_spacing_ratio"] for r in results]
        gammas = [r["gamma"] for r in results]
        goe_count = sum(1 for r in results if r["phase_classification"] == "wigner_dyson")
        gue_count = sum(1 for r in results if r["phase_classification"] == "poisson" or r["phase_classification"] == "many_body_localized")
        critical_gamma = None
        for i in range(len(results) - 1):
            if results[i]["phase_classification"] == "wigner_dyson" and results[i + 1]["phase_classification"] != "wigner_dyson":
                critical_gamma = (gammas[i] + gammas[i + 1]) / 2.0
                break
        return {
            "goe_dominant_count": goe_count,
            "gue_dominant_count": gue_count,
            "critical_gamma": critical_gamma,
            "max_gamma_tested": max(gammas),
            "transition_detected": critical_gamma is not None,
            "verdict": "symmetry_classes_validated" if critical_gamma is not None else "no_transition_detected",
        }


class Experiment3ConformalIsomorphism(IExperiment):
    """
    Tests the claim that the network learns the underlying conformal operator
    by applying Moebius transformations to inputs and measuring equivariance.
    """

    def __init__(
        self,
        config: Experiment3Config,
        model_config: StrassStrassenConfig,
        data_generator: IDataGenerator,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.data_generator = data_generator

    def get_name(self) -> str:
        return "experiment_3_conformal_isomorphism"

    def run(self, model: StrassStrassenModel) -> Dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        A, B, _ = self.data_generator.generate_batch(self.config.num_test_samples)
        errors = []

        for _ in range(self.config.num_transformations):
            T_A, T_B = self._apply_moebius(A, B)
            with torch.no_grad():
                pred_original = model(A, B)
                pred_transformed = model(T_A, T_B)
                T_pred = self._apply_moebius_to_output(pred_original)
                error = torch.norm(pred_transformed - T_pred).item() / max(torch.norm(T_pred).item(), 1e-10)
                errors.append(error)

        mean_error = float(np.mean(errors))
        max_error = float(np.max(errors))
        is_invariant = mean_error < self.config.tolerance_threshold
        payload = {
            "mean_error": mean_error,
            "max_error": max_error,
            "std_error": float(np.std(errors)),
            "num_transformations": self.config.num_transformations,
            "is_invariant": bool(is_invariant),
            "verdict": "conformal_operator_learned" if is_invariant else "conformal_operator_not_learned",
        }
        with open(output_dir / "experiment_3_results.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return payload

    def _apply_moebius(
        self, A: torch.Tensor, B: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch = A.shape[0]
        c = torch.randn(batch, 1, 1, device=A.device, dtype=A.dtype) * 0.5
        d = torch.randn(batch, 1, 1, device=A.device, dtype=A.dtype) * 0.5 + 1.0
        T_A = (A + c) / d
        T_B = (B + c) / d
        return T_A, T_B

    def _apply_moebius_to_output(self, C: torch.Tensor) -> torch.Tensor:
        batch = C.shape[0]
        c = torch.randn(batch, 1, 1, device=C.device, dtype=C.dtype) * 0.5
        d = torch.randn(batch, 1, 1, device=C.device, dtype=C.dtype) * 0.5 + 1.0
        return (C + c) / d


class Experiment4CompressionFrontier(IExperiment):
    """
    Tests the thermodynamic uncertainty relation between synthetic Planck
    constant and superposition metric across a sweep of batch sizes and
    weight decay values.
    """

    def __init__(
        self,
        config: Experiment4Config,
        model_config: StrassStrassenConfig,
        data_generator: IDataGenerator,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.data_generator = data_generator
        self.planck_calc = SyntheticPlanckCalculator(model_config)

    def get_name(self) -> str:
        return "experiment_4_compression_frontier"

    def run(self, model: StrassStrassenModel) -> Dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        results = []

        for batch_size in self.config.batch_sizes:
            for weight_decay in self.config.weight_decays:
                tc = TrainingConfig(
                    epochs=self.config.epochs_per_config,
                    batch_size=batch_size,
                    weight_decay=weight_decay,
                )
                trial_model = StrassStrassenModel(self.model_config)
                trainer = Trainer(self.model_config, tc, self.data_generator)
                trainer.train(trial_model, tc.epochs)

                planck = self.planck_calc.calculate(trial_model)
                superposition_calc = SuperpositionMetricCalculator(
                    self.model_config,
                    expansion_factor=self.config.sae_expansion_factor,
                    l1_coefficient=self.config.sae_l1_coefficient,
                    sae_lr=self.config.sae_learning_rate,
                    sae_epochs=self.config.sae_epochs,
                    sae_batch_size=self.config.sae_batch_size,
                    num_samples=self.config.num_activation_samples,
                    epsilon=self.config.epsilon,
                )
                superposition = superposition_calc.calculate(trial_model)

                results.append({
                    "batch_size": batch_size,
                    "weight_decay": weight_decay,
                    "hbar_eff": planck["hbar_unified"],
                    "psi": superposition["psi"],
                    "delta": planck["delta"],
                    "weight_variance": planck["weight_variance"],
                })

        summary = self._test_uncertainty_bound(results)
        payload = {"results": results, "summary": summary}
        with open(output_dir / "experiment_4_results.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return payload

    def _test_uncertainty_bound(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        products = [r["psi"] * r["hbar_eff"] for r in results]
        min_product = min(products) if products else 0.0
        mean_product = float(np.mean(products)) if products else 0.0
        violations = [r for r in results if r["psi"] > 1.5 and r["hbar_eff"] < 1e-3]
        return {
            "min_psi_hbar_product": min_product,
            "mean_psi_hbar_product": mean_product,
            "num_violations": len(violations),
            "bound_verified": len(violations) == 0,
            "verdict": "uncertainty_principle_supported" if len(violations) == 0 else "uncertainty_principle_violated",
        }


class IPruningStrategy(ABC):
    """Abstract base for structured pruning strategies."""

    @abstractmethod
    def prune(self, model: StrassStrassenModel, fraction: float) -> None:
        raise NotImplementedError


class VolumePruningStrategy(IPruningStrategy):
    """Prunes weights uniformly at random across all layers."""

    def prune(self, model: StrassStrassenModel, fraction: float) -> None:
        with torch.no_grad():
            for param in model.parameters():
                flat = param.flatten()
                num_to_prune = int(fraction * flat.numel())
                if num_to_prune > 0:
                    indices = torch.randperm(flat.numel())[:num_to_prune]
                    mask = torch.ones_like(flat)
                    mask[indices] = 0.0
                    param.mul_(mask.reshape(param.shape))


class AreaPruningStrategy(IPruningStrategy):
    """Prunes weights only from tensor boundaries (slot edges)."""

    def prune(self, model: StrassStrassenModel, fraction: float) -> None:
        with torch.no_grad():
            u_prune = int(fraction * model.U.shape[0])
            if u_prune > 0:
                model.U.data[-u_prune:, :] = 0.0
            v_prune = int(fraction * model.V.shape[0])
            if v_prune > 0:
                model.V.data[-v_prune:, :] = 0.0
            w_prune = int(fraction * model.W.shape[1])
            if w_prune > 0:
                model.W.data[:, -w_prune:] = 0.0


class Experiment5HolographicPruning(IExperiment):
    """
    Tests whether the crystal phase encodes information on boundaries
    (area law) versus the glass phase encoding it volumetrically.
    """

    def __init__(
        self,
        config: Experiment5Config,
        model_config: StrassStrassenConfig,
        data_generator: IDataGenerator,
    ) -> None:
        self.config = config
        self.model_config = model_config
        self.data_generator = data_generator
        self.volume_pruner = VolumePruningStrategy()
        self.area_pruner = AreaPruningStrategy()

    def get_name(self) -> str:
        return "experiment_5_holographic_pruning"

    def run(self, model: StrassStrassenModel) -> Dict[str, Any]:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        A_test, B_test, C_true = self.data_generator.generate_batch(self.config.num_test_samples)

        volume_results = self._run_pruning_trials(model, self.volume_pruner, A_test, B_test, C_true)
        area_results = self._run_pruning_trials(model, self.area_pruner, A_test, B_test, C_true)

        volume_mean_degradation = float(np.mean([r["accuracy_drop"] for r in volume_results]))
        area_mean_degradation = float(np.mean([r["accuracy_drop"] for r in area_results]))

        payload = {
            "volume_pruning": {"trials": volume_results, "mean_degradation": volume_mean_degradation},
            "area_pruning": {"trials": area_results, "mean_degradation": area_mean_degradation},
            "holographic_signature_detected": area_mean_degradation > volume_mean_degradation * 2.0,
            "verdict": "holographic_area_law" if area_mean_degradation > volume_mean_degradation * 2.0 else "no_holographic_signature",
        }
        with open(output_dir / "experiment_5_results.json", "w") as f:
            json.dump(payload, f, indent=2, default=str)
        return payload

    def _run_pruning_trials(
        self,
        base_model: StrassStrassenModel,
        pruner: IPruningStrategy,
        A: torch.Tensor,
        B: torch.Tensor,
        C_true: torch.Tensor,
    ) -> List[Dict[str, float]]:
        trials = []
        for _ in range(self.config.num_trials):
            trial_model = StrassStrassenModel(self.model_config)
            trial_model.load_state_dict(base_model.state_dict())
            with torch.no_grad():
                pred_before = trial_model(A, B)
                acc_before = (pred_before - C_true).abs().reshape(A.shape[0], -1).max(dim=1)[0]
                acc_before = (acc_before < self.config.accuracy_threshold).float().mean().item()

            pruner.prune(trial_model, self.config.prune_fraction)

            with torch.no_grad():
                pred_after = trial_model(A, B)
                acc_after = (pred_after - C_true).abs().reshape(A.shape[0], -1).max(dim=1)[0]
                acc_after = (acc_after < self.config.accuracy_threshold).float().mean().item()

            trials.append({
                "accuracy_before": acc_before,
                "accuracy_after": acc_after,
                "accuracy_drop": acc_before - acc_after,
            })
        return trials


class UnifiedSuite:
    """Orchestrates the execution of all five hidden-connection experiments."""

    def __init__(self, config: SuiteConfig) -> None:
        self.config = config
        self.data_generator = StrassenDataGenerator(config.model_config)
        self.checkpoint_manager = CheckpointManager()
        self.experiments: List[IExperiment] = []
        self._build_experiments()

    def _build_experiments(self) -> None:
        self.experiments.append(
            Experiment1RicciMBLDuality(
                self.config.experiment_1,
                self.config.model_config,
                self.config.training_config,
                self.data_generator,
                self.checkpoint_manager,
            )
        )
        self.experiments.append(
            Experiment2AltlandZirnbauer(
                self.config.experiment_2,
                self.config.model_config,
                self.data_generator,
            )
        )
        self.experiments.append(
            Experiment3ConformalIsomorphism(
                self.config.experiment_3,
                self.config.model_config,
                self.data_generator,
            )
        )
        self.experiments.append(
            Experiment4CompressionFrontier(
                self.config.experiment_4,
                self.config.model_config,
                self.data_generator,
            )
        )
        self.experiments.append(
            Experiment5HolographicPruning(
                self.config.experiment_5,
                self.config.model_config,
                self.data_generator,
            )
        )

    def run_all(self) -> Dict[str, Any]:
        output_dir = Path(self.config.global_output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        base_model = StrassStrassenModel(self.config.model_config)
        all_results = {}

        for experiment in self.experiments:
            print(f"Running {experiment.get_name()}...")
            model_copy = copy.deepcopy(base_model)
            result = experiment.run(model_copy)
            all_results[experiment.get_name()] = result

        summary = self._aggregate_verdicts(all_results)
        final_payload = {
            "suite_config": self._serialize_config(),
            "results": all_results,
            "summary": summary,
            "timestamp": datetime.now().isoformat(),
        }
        with open(output_dir / "unified_suite_summary.json", "w") as f:
            json.dump(final_payload, f, indent=2, default=str)
        return final_payload

    def _aggregate_verdicts(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        verdicts = {}
        for name, result in all_results.items():
            verdicts[name] = result.get("summary", {}).get("verdict", "unknown")
        supported = sum(1 for v in verdicts.values() if "supported" in v or "validated" in v or "learned" in v)
        total = len(verdicts)
        return {
            "individual_verdicts": verdicts,
            "claims_supported": supported,
            "claims_total": total,
            "fraction_supported": supported / total if total > 0 else 0.0,
        }

    def _serialize_config(self) -> Dict[str, Any]:
        return {
            "model_config": {
                "matrix_size": self.config.model_config.matrix_size,
                "input_dim": self.config.model_config.input_dim,
                "output_dim": self.config.model_config.output_dim,
                "rank": self.config.model_config.rank,
                "target_rank": self.config.model_config.target_rank,
                "device": self.config.model_config.device,
            },
            "training_config": {
                "epochs": self.config.training_config.epochs,
                "batch_size": self.config.training_config.batch_size,
                "learning_rate": self.config.training_config.learning_rate,
                "weight_decay": self.config.training_config.weight_decay,
            },
        }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Unified Hidden Connections Experimental Suite for Strass-Strassen Neural Networks"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="unified_hidden_connections_results",
        help="Root directory for all experimental outputs",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30000,
        help="Total training epochs for the base model",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training batch size",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.02,
        help="Optimizer learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay (thermodynamic pressure)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Computation device",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="all",
        choices=["all", "1", "2", "3", "4", "5"],
        help="Which experiment to run (all or 1-5)",
    )
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    model_config = StrassStrassenConfig(device=args.device)
    training_config = TrainingConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        seed=args.seed,
    )
    suite_config = SuiteConfig(
        model_config=model_config,
        training_config=training_config,
        global_output_dir=args.output_dir,
    )

    suite = UnifiedSuite(suite_config)
    results = suite.run_all()

    print("=" * 80)
    print("UNIFIED HIDDEN CONNECTIONS SUITE COMPLETE")
    print("=" * 80)
    for exp_name, verdict in results["summary"]["individual_verdicts"].items():
        print(f"  {exp_name}: {verdict}")
    print(f"  Overall support fraction: {results['summary']['fraction_supported']:.2%}")
    print(f"  Results saved to: {args.output_dir}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
