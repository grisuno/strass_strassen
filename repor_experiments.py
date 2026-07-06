#!/usr/bin/env python3
"""
repor_experiments.py — Suite de experimentos corregido
=======================================================
Valida los 5 claims de repor.md usando la metodología del README
(protocolo de dos fases: entrenar, podar a 7 slots, discretizar
a {-1, 0, 1}, verificar zero-shot hasta 64x64).

El batch_size default es 32 (dentro del rango validado [24, 128])
y TODOS los experimentos aplican la Fase 2 de poda+discretización.

Modos:
  --mode train        Entrena modelos desde cero (experimentos 1-5)
  --mode checkpoints  Analiza checkpoints existentes (cristal vs vidrio)

Autor: grisun0
"""

from __future__ import annotations

import argparse
import copy
import csv
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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------
STRASSEN_U = torch.tensor([
    [1, 0, 0, 1],
    [0, 0, 1, 1],
    [1, 0, 0, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [-1, 0, 1, 0],
    [0, 1, 0, -1],
], dtype=torch.float32)

STRASSEN_V = torch.tensor([
    [1, 0, 0, 1],
    [1, 0, 0, 0],
    [0, 1, 0, -1],
    [-1, 0, 1, 0],
    [0, 0, 0, 1],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
], dtype=torch.float32)

STRASSEN_W = torch.zeros(4, 7)
STRASSEN_W[0, 0] = 1; STRASSEN_W[0, 3] = 1; STRASSEN_W[0, 4] = -1; STRASSEN_W[0, 6] = 1
STRASSEN_W[1, 2] = 1; STRASSEN_W[1, 4] = 1
STRASSEN_W[2, 1] = 1; STRASSEN_W[2, 3] = 1
STRASSEN_W[3, 0] = 1; STRASSEN_W[3, 1] = -1; STRASSEN_W[3, 2] = 1; STRASSEN_W[3, 5] = 1

CANONICAL = (STRASSEN_U, STRASSEN_V, STRASSEN_W)

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------
@dataclass
class ModelConfig:
    rank: int = 8
    target_rank: int = 7
    input_dim: int = 4
    output_dim: int = 4
    matrix_size: int = 2

@dataclass
class TrainConfig:
    epochs: int = 30000
    batch_size: int = 32
    lr: float = 0.02
    weight_decay: float = 1e-4
    scheduler_t0: int = 10000
    grad_clip: float = 1.0
    acc_threshold: float = 1e-3

@dataclass
class Exp1Config:
    checkpoint_epochs: Tuple[int, ...] = (
        100, 500, 1000, 2000, 3000, 5000, 7500, 10000,
        12500, 15000, 17500, 20000, 22500, 25000, 27500, 30000,
    )

@dataclass
class Exp2Config:
    gamma_values: Tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0)
    epochs_per_gamma: int = 5000

@dataclass
class Exp3Config:
    num_samples: int = 1000
    num_transforms: int = 50
    tolerance: float = 1e-3

@dataclass
class Exp4Config:
    batch_sizes: Tuple[int, ...] = (8, 16, 32, 64, 128, 256, 512)
    weight_decays: Tuple[float, ...] = (1e-5, 1e-4, 1e-3)
    epochs_per_config: int = 15000

@dataclass
class Exp5Config:
    num_trials: int = 10
    prune_fraction: float = 0.1
    fractions: Tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25)

@dataclass
class SuiteConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    exp1: Exp1Config = field(default_factory=Exp1Config)
    exp2: Exp2Config = field(default_factory=Exp2Config)
    exp3: Exp3Config = field(default_factory=Exp3Config)
    exp4: Exp4Config = field(default_factory=Exp4Config)
    exp5: Exp5Config = field(default_factory=Exp5Config)
    output_dir: str = "repor_results"
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    seed: int = 42

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
class BilinearModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.U = nn.Linear(cfg.input_dim, cfg.rank, bias=False)
        self.V = nn.Linear(cfg.input_dim, cfg.rank, bias=False)
        self.W = nn.Linear(cfg.rank, cfg.output_dim, bias=False)

    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        batch = A.shape[0]
        a = A.reshape(batch, self.cfg.input_dim)
        b = B.reshape(batch, self.cfg.input_dim)
        c = self.W(self.U(a) * self.V(b))
        return c.reshape(batch, self.cfg.matrix_size, self.cfg.matrix_size)

    def slot_importance(self) -> torch.Tensor:
        Uw = self.U.weight
        Vw = self.V.weight
        Ww = self.W.weight
        return (torch.norm(Uw, dim=1) * torch.norm(Vw, dim=1)
                * torch.norm(Ww, dim=0))

    def get_weights(self) -> Dict[str, torch.Tensor]:
        return {"U": self.U.weight.data.clone(), "V": self.V.weight.data.clone(),
                "W": self.W.weight.data.clone()}

    def get_flat(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])

# ---------------------------------------------------------------------------
# Fase 2: poda + discretización + verificación
# ---------------------------------------------------------------------------
def discretize_q(w: torch.Tensor) -> torch.Tensor:
    return torch.round(w).clamp(-1, 1)

def compute_delta(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor) -> float:
    weights = torch.cat([U.flatten(), V.flatten(), W.flatten()])
    q = discretize_q(weights)
    return float(torch.max(torch.abs(weights - q)))

def phase2(model: BilinearModel) -> Optional[Dict[str, torch.Tensor]]:
    """Poda a target_rank slots, discretiza a {-1,0,1}, verifica."""
    cfg = model.cfg
    imp = model.slot_importance()
    if imp.numel() <= cfg.target_rank:
        keep = set(range(imp.numel()))
    else:
        _, idx = torch.topk(imp, cfg.target_rank)
        keep = set(idx.tolist())
    U = model.U.weight.data.clone()
    V = model.V.weight.data.clone()
    W = model.W.weight.data.clone()
    for i in range(U.shape[0]):
        if i not in keep:
            U[i] = 0
            V[i] = 0
            W[:, i] = 0
    Uq, Vq, Wq = discretize_q(U), discretize_q(V), discretize_q(W)
    err = _verify_2x2(Uq, Vq, Wq)
    if err > 1e-5:
        return None
    return {"U": Uq, "V": Vq, "W": Wq}

def _verify_2x2(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor,
                n: int = 100) -> float:
    device = U.device
    A = torch.randn(n, 2, 2, device=device)
    B = torch.randn(n, 2, 2, device=device)
    a, b = A.reshape(n, 4), B.reshape(n, 4)
    C_pred = ((a @ U.T) * (b @ V.T)) @ W.T
    C_true = torch.bmm(A, B)
    return float(torch.norm(C_pred.reshape(n, 4) - C_true.reshape(n, 4), dim=1).max().item())

def zero_shot_verify(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor,
                     sizes: Tuple[int, ...] = (2, 4, 8, 16, 32, 64)) -> Dict[int, float]:
    results = {}
    for n in sizes:
        err = _recursive_strassen(U, V, W, n)
        results[n] = err
    return results

def _recursive_strassen(U: torch.Tensor, V: torch.Tensor, W: torch.Tensor,
                        n: int, trials: int = 3) -> float:
    device = U.device
    errs = []
    for _ in range(trials):
        A = torch.randn(1, n, n, device=device)
        B = torch.randn(1, n, n, device=device)
        C_true = torch.bmm(A, B)
        C_pred = _strassen_rec(A, B, U, V, W, n)
        errs.append(float(torch.norm(C_pred - C_true) / (torch.norm(C_true) + 1e-10)))
    return float(np.mean(errs))

def _strassen_rec(A: torch.Tensor, B: torch.Tensor,
                  U: torch.Tensor, V: torch.Tensor, W: torch.Tensor,
                  n: int) -> torch.Tensor:
    if n == 2:
        a, b = A.reshape(1, 4), B.reshape(1, 4)
        return (((a @ U.T) * (b @ V.T)) @ W.T).reshape(1, 2, 2)
    h = n // 2
    A11, A12 = A[:, :h, :h], A[:, :h, h:]
    A21, A22 = A[:, h:, :h], A[:, h:, h:]
    B11, B12 = B[:, :h, :h], B[:, :h, h:]
    B21, B22 = B[:, h:, :h], B[:, h:, h:]
    M1 = _strassen_rec(A11 + A22, B11 + B22, U, V, W, h)
    M2 = _strassen_rec(A21 + A22, B11, U, V, W, h)
    M3 = _strassen_rec(A11, B12 - B22, U, V, W, h)
    M4 = _strassen_rec(A22, B21 - B11, U, V, W, h)
    M5 = _strassen_rec(A11 + A12, B22, U, V, W, h)
    M6 = _strassen_rec(A21 - A11, B11 + B12, U, V, W, h)
    M7 = _strassen_rec(A12 - A22, B21 + B22, U, V, W, h)
    C = torch.zeros(1, n, n, device=A.device)
    C[:, :h, :h] = M1 + M4 - M5 + M7
    C[:, :h, h:] = M3 + M5
    C[:, h:, :h] = M2 + M4
    C[:, h:, h:] = M1 - M2 + M3 + M6
    return C

# ---------------------------------------------------------------------------
# Métricas del README
# ---------------------------------------------------------------------------
def compute_kappa(model: BilinearModel, num_batches: int = 20,
                  bs: int = 32) -> float:
    grads = []
    for _ in range(num_batches):
        A = torch.randn(bs, 2, 2, device=next(model.parameters()).device)
        B = torch.randn(bs, 2, 2, device=next(model.parameters()).device)
        loss = F.mse_loss(model(A, B), torch.bmm(A, B))
        g = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        grads.append(torch.cat([gi.flatten() for gi in g]))
    G = torch.stack(grads)
    cov = torch.cov(G.T)
    eig = torch.linalg.eigvalsh(cov)
    eig = eig[eig > 1e-12]
    if len(eig) < 2:
        return float("inf")
    return float(eig.max() / eig.min())

def compute_alpha(delta: float) -> float:
    if delta < 1e-10:
        return 20.0
    return float(-np.log(delta))

def compute_teff(model: BilinearModel, num_batches: int = 20,
                  bs: int = 32) -> float:
    total = 0.0
    count = 0
    for _ in range(num_batches):
        A = torch.randn(bs, 2, 2, device=next(model.parameters()).device)
        B = torch.randn(bs, 2, 2, device=next(model.parameters()).device)
        loss = F.mse_loss(model(A, B), torch.bmm(A, B))
        g = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        gv = torch.cat([gi.flatten() for gi in g])
        total += float(torch.sum(gv ** 2))
        count += gv.numel()
    return total / count if count > 0 else 0.0

def classify_phase(delta: float) -> str:
    if delta < 0.01:
        return "crystal"
    elif delta < 0.3:
        return "polycrystal"
    else:
        return "glass"

# ---------------------------------------------------------------------------
# Carga de checkpoints
# ---------------------------------------------------------------------------
def load_checkpoint(path: str, device: str = "cpu") -> Optional[BilinearModel]:
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(raw, dict):
        state = _extract_state(raw)
    elif hasattr(raw, "state_dict"):
        state = _extract_state(raw.state_dict())
    else:
        return None
    if state is None:
        return None
    state = {k: v.to(device) for k, v in state.items()}
    hidden = state["U.weight"].shape[0]
    mc = ModelConfig(rank=hidden, target_rank=min(7, hidden))
    model = BilinearModel(mc).to(device)
    try:
        model.load_state_dict(state)
    except Exception:
        return None
    return model

def _extract_state(d: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
    # Exact match: llaves con .weight
    if all(f"{k}.weight" in d for k in ["U", "V", "W"]):
        return {k: d[k] for k in ["U.weight", "V.weight", "W.weight"]}
    # Llaves directas U, V, W
    if all(k in d for k in ["U", "V", "W"]):
        u, v, w = d["U"], d["V"], d["W"]
        if u.dim() == 2 and v.dim() == 2 and w.dim() == 2:
            return {"U.weight": u, "V.weight": v, "W.weight": w}
    # model_state_dict anidado
    if "model_state_dict" in d:
        return _extract_state(d["model_state_dict"])
    if "state_dict" in d:
        return _extract_state(d["state_dict"])
    # _coefs
    if all(k in d for k in ["U_coefs", "V_coefs", "W_coefs"]):
        return {"U.weight": d["U_coefs"], "V.weight": d["V_coefs"], "W.weight": d["W_coefs"]}
    return None

# ---------------------------------------------------------------------------
# Entrenamiento (Fase 1)
# ---------------------------------------------------------------------------
def train_model(cfg: SuiteConfig, model: BilinearModel,
                epochs: int, bs: int, wd: float, lr: float,
                callback=None) -> Dict[str, list]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, T_0=cfg.train.scheduler_t0)
    history = {"loss": [], "acc": [], "epoch": []}
    for ep in range(epochs):
        A = torch.randn(bs, 2, 2, device=cfg.device)
        B = torch.randn(bs, 2, 2, device=cfg.device)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        loss = F.mse_loss(C_pred, C_true)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        sched.step()
        with torch.no_grad():
            errs = (C_pred - C_true).abs().reshape(bs, -1).max(dim=1)[0]
            acc = (errs < cfg.train.acc_threshold).float().mean().item()
        history["loss"].append(loss.item())
        history["acc"].append(acc)
        history["epoch"].append(ep)
        if callback:
            callback(ep, model, loss.item(), acc)
    return history

def analyze_checkpoint(path: str, device: str = "cpu") -> Dict[str, Any]:
    model = load_checkpoint(path, device)
    if model is None:
        return {"error": f"cannot load {path}"}
    with torch.no_grad():
        weights = model.get_weights()
        delta = compute_delta(weights["U"], weights["V"], weights["W"])
        alpha = compute_alpha(delta)
        phase = classify_phase(delta)
        flat = model.get_flat()
        n = flat.numel()
        gram = torch.outer(flat, flat) + 1e-8 * torch.eye(n, device=flat.device)
        eig = torch.linalg.eigvalsh(gram)
        eig = eig[eig > 1e-12]
        spacings = torch.diff(eig)
        ratios = []
        for i in range(len(spacings) - 1):
            s_n, s_n1 = spacings[i], spacings[i + 1]
            mx = max(s_n, s_n1)
            if mx > 1e-15:
                ratios.append(float(min(s_n, s_n1) / mx))
        r_mean = float(np.mean(ratios)) if ratios else 0.0
        cond = float(eig.max() / eig.min()) if len(eig) >= 2 and eig.min() > 1e-12 else float("inf")
        imp = model.slot_importance()
        active = int((imp > 0.1).sum().item())
    result = {
        "checkpoint": Path(path).name,
        "delta": delta,
        "alpha": alpha,
        "phase": phase,
        "mean_gap_ratio_r": r_mean,
        "condition_number": cond,
        "active_slots": active,
        "rank": model.cfg.rank,
    }
    # Si es cristal, verificar zero-shot
    if phase == "crystal":
        phase2_result = phase2(model)
        if phase2_result is not None:
            zs = zero_shot_verify(phase2_result["U"], phase2_result["V"],
                                  phase2_result["W"])
            result["zero_shot"] = zs
            result["phase2_ok"] = True
        else:
            result["phase2_ok"] = False
    return result

# ---------------------------------------------------------------------------
# EXPERIMENTO 1: Curvature-Spectral (usando κ, no Hessiano roto)
# ---------------------------------------------------------------------------
def experiment1(cfg: SuiteConfig) -> Dict[str, Any]:
    device = cfg.device
    mc = cfg.model
    tc = cfg.train
    model = BilinearModel(mc).to(device)
    results = []
    def cb(ep: int, m: BilinearModel, loss: float, acc: float):
        if ep in cfg.exp1.checkpoint_epochs:
            delta = compute_delta(m.U.weight.data, m.V.weight.data, m.W.weight.data)
            kappa = compute_kappa(m)
            weights = m.get_weights()
            flat = torch.cat([w.flatten() for w in weights.values()])
            n = flat.numel()
            gram = torch.outer(flat, flat) + 1e-8 * torch.eye(n, device=device)
            eig = torch.linalg.eigvalsh(gram)
            eig = eig[eig > 1e-12]
            spacings = torch.diff(eig)
            ratios = []
            for i in range(len(spacings) - 1):
                s_n, s_n1 = spacings[i], spacings[i + 1]
                mx = max(s_n, s_n1)
                if mx > 1e-15:
                    ratios.append(float(min(s_n, s_n1) / mx))
            r_mean = float(np.mean(ratios)) if ratios else 0.0
            results.append({
                "epoch": ep, "loss": loss, "acc": acc,
                "delta": delta, "kappa": kappa, "mean_gap_ratio_r": r_mean,
                "phase": classify_phase(delta),
            })
    train_model(cfg, model, tc.epochs, tc.batch_size, tc.weight_decay, tc.lr, cb)
    # Fase 2 al final
    phase2_result = phase2(model)
    final = {
        "checkpoints": results,
        "phase2_success": phase2_result is not None,
    }
    if phase2_result is not None:
        zs = zero_shot_verify(phase2_result["U"], phase2_result["V"],
                              phase2_result["W"])
        final["zero_shot"] = zs
        final["final_delta"] = compute_delta(
            phase2_result["U"], phase2_result["V"], phase2_result["W"])
    return final

# ---------------------------------------------------------------------------
# EXPERIMENTO 2: Symmetry Dial (γ, con batch size correcto)
# ---------------------------------------------------------------------------
def experiment2(cfg: SuiteConfig) -> Dict[str, Any]:
    device = cfg.device
    mc = cfg.model
    results = []
    for gamma in cfg.exp2.gamma_values:
        model = BilinearModel(mc).to(device)
        if gamma > 0:
            with torch.no_grad():
                noise_u = torch.randn_like(model.U.weight) * 0.1
                noise_v = torch.randn_like(model.V.weight) * 0.1
                model.U.weight.add_(gamma * noise_u)
                model.V.weight.add_(gamma * noise_v)
        train_model(cfg, model, cfg.exp2.epochs_per_gamma,
                    cfg.train.batch_size, cfg.train.weight_decay, cfg.train.lr)
        p2 = phase2(model)
        delta = compute_delta(model.U.weight.data, model.V.weight.data, model.W.weight.data)
        alpha = compute_alpha(delta)
        kappa = compute_kappa(model)
        entry = {
            "gamma": gamma, "delta": delta, "alpha": alpha, "kappa": kappa,
            "phase": classify_phase(delta),
            "phase2_success": p2 is not None,
        }
        if p2 is not None:
            zs = zero_shot_verify(p2["U"], p2["V"], p2["W"])
            entry["zero_shot"] = zs
        results.append(entry)
    return {"results": results}

# ---------------------------------------------------------------------------
# EXPERIMENTO 3: Scale-Equivariance (Moebius corregido + homogeneidad)
# ---------------------------------------------------------------------------
def experiment3(cfg: SuiteConfig) -> Dict[str, Any]:
    device = cfg.device
    mc = cfg.model
    model = BilinearModel(mc).to(device)
    train_model(cfg, model, cfg.train.epochs, cfg.train.batch_size,
                cfg.train.weight_decay, cfg.train.lr)
    p2 = phase2(model)
    if p2 is None:
        return {"error": "phase2 failed — model no cristalizó"}
    U, V, W = p2["U"].to(device), p2["V"].to(device), p2["W"].to(device)
    bilinear_errs = []
    moebius_errs = []
    scales = [0.5, 2.0, 3.0]
    for _ in range(cfg.exp3.num_samples):
        A = torch.randn(1, 2, 2, device=device)
        B = torch.randn(1, 2, 2, device=device)
        alpha = float(random.choice(scales))
        beta = float(random.choice(scales))
        a, b = A.reshape(1, 4), B.reshape(1, 4)
        pred_ref = ((a @ U.T) * (b @ V.T)) @ W.T
        pred_scaled = ((alpha * a @ U.T) * (beta * b @ V.T)) @ W.T
        expected = (alpha * beta) * pred_ref
        err = float(torch.norm(pred_scaled - expected) / (torch.norm(expected) + 1e-10))
        bilinear_errs.append(err)
    for _ in range(cfg.exp3.num_transforms):
        A = torch.randn(1, 2, 2, device=device)
        B = torch.randn(1, 2, 2, device=device)
        # MISMO c,d para input y output
        c = float(torch.randn(1) * 0.5)
        d = float(torch.randn(1) * 0.5 + 1.0)
        T_input = lambda x: (x + c) / d
        a, b = A.reshape(1, 4), B.reshape(1, 4)
        pred_orig = ((a @ U.T) * (b @ V.T)) @ W.T
        a_t, b_t = T_input(A).reshape(1, 4), T_input(B).reshape(1, 4)
        pred_trans = ((a_t @ U.T) * (b_t @ V.T)) @ W.T
        T_output = lambda x: (x + c) / d
        expected = T_output(pred_orig)
        err = float(torch.norm(pred_trans - expected) / (torch.norm(expected) + 1e-10))
        moebius_errs.append(err)
    return {
        "phase2_ok": True,
        "bilinear_homogeneity": {
            "mean_error": float(np.mean(bilinear_errs)),
            "max_error": float(np.max(bilinear_errs)),
            "holds": float(np.mean(bilinear_errs)) < cfg.exp3.tolerance,
        },
        "moebius_equivariance": {
            "mean_error": float(np.mean(moebius_errs)),
            "max_error": float(np.max(moebius_errs)),
            "holds": float(np.mean(moebius_errs)) < cfg.exp3.tolerance,
        },
    }

# ---------------------------------------------------------------------------
# EXPERIMENTO 4: Resolution Bound (con δ del README)
# ---------------------------------------------------------------------------
def experiment4(cfg: SuiteConfig) -> Dict[str, Any]:
    device = cfg.device
    mc = cfg.model
    results = []
    for bs in cfg.exp4.batch_sizes:
        for wd in cfg.exp4.weight_decays:
            model = BilinearModel(mc).to(device)
            train_model(cfg, model, cfg.exp4.epochs_per_config, bs, wd,
                        cfg.train.lr)
            U, V, W = model.U.weight.data, model.V.weight.data, model.W.weight.data
            delta_raw = compute_delta(U, V, W)
            p2 = phase2(model)
            if p2 is not None:
                delta = 0.0
                psi = 1.0 / 7.0
            else:
                delta = delta_raw
                flat = model.get_flat()
                abs_w = torch.abs(flat)
                probs = abs_w / (abs_w.sum() + 1e-12)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12))
                psi = float(torch.exp(entropy) / 7.0)
            weight_var = float(torch.var(model.get_flat()))
            hbar_eff = float(weight_var * delta + 1e-12)
            kappa = compute_kappa(model)
            results.append({
                "batch_size": bs, "weight_decay": wd,
                "delta": delta, "psi": psi, "hbar_eff": hbar_eff,
                "product": psi * hbar_eff, "kappa": kappa,
                "phase2_ok": p2 is not None,
            })
    products = [r["product"] for r in results]
    return {
        "sweep_results": results,
        "min_product": min(products) if products else None,
        "mean_product": float(np.mean(products)) if products else None,
        "max_product": max(products) if products else None,
        "violations": sum(1 for r in results if r["psi"] > 1.5 and r["hbar_eff"] < 1e-3),
        "n_crystals": sum(1 for r in results if r["phase2_ok"]),
    }

# ---------------------------------------------------------------------------
# EXPERIMENTO 5: Boundary Pruning
# ---------------------------------------------------------------------------
def experiment5(cfg: SuiteConfig) -> Dict[str, Any]:
    device = cfg.device
    mc = cfg.model
    model = BilinearModel(mc).to(device)
    train_model(cfg, model, cfg.train.epochs, cfg.train.batch_size,
                cfg.train.weight_decay, cfg.train.lr)
    p2 = phase2(model)
    base_acc = _test_accuracy(model, device=device)
    results = {"base_accuracy": base_acc, "phase2_ok": p2 is not None}
    for label, pruner in [("random", _random_prune), ("boundary", _boundary_prune)]:
        frac_results = {}
        for frac in cfg.exp5.fractions:
            accs = []
            for _ in range(cfg.exp5.num_trials):
                m = BilinearModel(mc).to(device)
                m.load_state_dict(model.state_dict())
                pruner(m, frac)
                accs.append(_test_accuracy(m, device=device))
            frac_results[str(frac)] = {
                "mean_accuracy": float(np.mean(accs)),
                "std_accuracy": float(np.std(accs)),
                "degradation": base_acc - float(np.mean(accs)),
            }
        results[label] = frac_results
    return results

def _test_accuracy(model: BilinearModel, n: int = 500,
                   device: str = "cpu") -> float:
    model.eval()
    with torch.no_grad():
        A = torch.randn(n, 2, 2, device=device)
        B = torch.randn(n, 2, 2, device=device)
        C_pred = model(A, B)
        C_true = torch.bmm(A, B)
        errs = (C_pred - C_true).abs().reshape(n, -1).max(dim=1)[0]
        return float((errs < 1e-3).float().mean().item())

def _random_prune(model: BilinearModel, fraction: float):
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            k = int(n * fraction)
            if k > 0:
                idx = torch.randperm(n)[:k]
                flat = p.flatten()
                flat[idx] = 0.0
                p.data = flat.reshape(p.shape)

def _boundary_prune(model: BilinearModel, fraction: float):
    with torch.no_grad():
        k = max(1, int(model.cfg.rank * fraction))
        model.U.weight.data[-k:, :] = 0.0
        model.V.weight.data[-k:, :] = 0.0
        model.W.weight.data[:, -k:] = 0.0

# ---------------------------------------------------------------------------
# Modo checkpoints
# ---------------------------------------------------------------------------
def analyze_checkpoints(ckpt_dir: str, device: str = "cpu") -> Dict[str, Any]:
    ckpt_dir = Path(ckpt_dir)
    results = {"crystals": [], "glass": [], "polycrystal": []}
    for fname in sorted(os.listdir(ckpt_dir)):
        if not fname.endswith(".pt"):
            continue
        fpath = str(ckpt_dir / fname)
        r = analyze_checkpoint(fpath, device)
        if "error" in r:
            continue
        phase = r["phase"]
        results.setdefault(phase, []).append(r)
    summary = {}
    for phase in ["crystal", "glass", "polycrystal"]:
        entries = results.get(phase, [])
        if entries:
            deltas = [e["delta"] for e in entries]
            rs = [e["mean_gap_ratio_r"] for e in entries]
            summary[phase] = {
                "count": len(entries),
                "mean_delta": float(np.mean(deltas)),
                "mean_r": float(np.mean(rs)),
                "std_r": float(np.std(rs)),
                "checkpoints": [e["checkpoint"] for e in entries],
            }
    # t-test cristal vs vidrio
    crystals = results.get("crystal", [])
    glass = results.get("glass", [])
    if crystals and glass:
        from scipy.stats import ttest_ind
        r_c = [e["mean_gap_ratio_r"] for e in crystals]
        r_g = [e["mean_gap_ratio_r"] for e in glass]
        stat, pval = ttest_ind(r_c, r_g, equal_var=False)
        summary["crystal_vs_glass_ttest"] = {
            "t_statistic": float(stat), "p_value": float(pval)}
    return summary

# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "checkpoints"], default="train")
    parser.add_argument("--output-dir", default="repor_results")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30000)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--ckpt-dir", default="checkpoints")
    parser.add_argument("--experiment", default="all",
                        choices=["all", "1", "2", "3", "4", "5"])
    parser.add_argument("--gamma", type=float, default=None,
                        help="Gamma override for exp2")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    cfg = SuiteConfig(
        model=ModelConfig(),
        train=TrainConfig(
            epochs=args.epochs, batch_size=args.batch_size,
            weight_decay=args.weight_decay, lr=args.lr,
        ),
        output_dir=args.output_dir,
        device=device, seed=args.seed,
    )

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    if args.mode == "checkpoints":
        print(f"Analizando checkpoints en {args.ckpt_dir}...")
        result = analyze_checkpoints(args.ckpt_dir, device)
        _save(result, out / "checkpoint_analysis.json")
        print(json.dumps(result, indent=2, default=str))
        return

    # Modo train
    experiments = {
        "1": ("Curvature-Spectral (κ)", lambda: experiment1(cfg)),
        "2": ("Symmetry Dial (γ)", lambda: experiment2(cfg)),
        "3": ("Scale-Equivariance", lambda: experiment3(cfg)),
        "4": ("Resolution Bound", lambda: experiment4(cfg)),
        "5": ("Boundary Pruning", lambda: experiment5(cfg)),
    }

    to_run = list(experiments.keys()) if args.experiment == "all" else [args.experiment]

    all_results = {"config": {
        "batch_size": args.batch_size, "epochs": args.epochs,
        "weight_decay": args.weight_decay, "lr": args.lr,
        "device": device, "seed": args.seed,
    }}
    for key in to_run:
        name, fn = experiments[key]
        print(f"\n--- Experimento {key}: {name} ---")
        try:
            result = fn()
            all_results[f"exp{key}"] = result
            _save(result, out / f"exp{key}_results.json")
            print(f"  OK -> {out / f'exp{key}_results.json'}")
        except Exception as e:
            print(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results[f"exp{key}"] = {"error": str(e)}

    all_results["timestamp"] = datetime.now().isoformat()
    _save(all_results, out / "all_results.json")
    print(f"\nResultados completos en {out / 'all_results.json'}")

def _save(data: Any, path: Path):
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)

if __name__ == "__main__":
    main()
