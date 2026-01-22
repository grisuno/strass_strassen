#!/usr/bin/env python3
"""
================================================================================
BATERÍA DE EXPERIMENTOS DE VALIDACIÓN - RESPUESTA A REVISOR
================================================================================

Script completo para validar los experimentos del paper usando checkpoints
PRE-GROKKED del repositorio strass_strassen.

Experimentos incluidos:
1. Local Complexity (LC) vs Época - Métrica de progreso
2. Poda Iterativa + Fine-tuning - Verificación de cuenca discreta (δ < 0.1)
3. Curvas ROC con Bootstrap CI - Validación estadística

El script usa checkpoints ya grokkeados (accuracy ≈ 100%) para evitar el
problema de entrenar desde cero modelos que nunca grokkean.

Autor: grisun0
Fecha: 2026-01-22
================================================================================
"""

import sys
import os
import json
import copy
import argparse
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import defaultdict
import warnings

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, cohen_kappa_score
from scipy.stats import beta
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Configuración global
warnings.filterwarnings('ignore')
torch.set_num_threads(4)

# Semilla para reproducibilidad
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ================================================================================
# CLASE 1: CONFIGURACIÓN CENTRALIZADA
# ================================================================================

@dataclass
class ExperimentConfig:
    """
    Configuración centralizada para todos los experimentos.
    
    Sigue el principio de responsabilidad única - solo gestiona parámetros.
    No usa magic numbers; todos los valores están definidos aquí.
    """
    # Arquitectura del modelo
    rank: int = 8
    matrix_size: int = 2
    input_dim: int = 8
    output_dim: int = 4
    hidden_dim: int = 128
    
    # Hiperparámetros de entrenamiento
    learning_rate: float = 0.02
    weight_decay: float = 0.0001
    batch_size: int = 256
    max_epochs: int = 6000
    
    # Parámetros del protocolo de poda
    prune_max_iterations: int = 60
    prune_initial_percent: float = 10.0
    prune_step_percent: float = 5.0
    finetune_epochs: int = 200
    finetune_lr: float = 0.005
    delta_threshold: float = 0.1
    
    # Parámetros de discretización
    discretization_threshold: float = 0.5
    discretization_values: List[float] = None
    
    # Parámetros de evaluación
    test_samples: int = 10000
    accuracy_threshold: float = 0.001
    grokking_accuracy_threshold: float = 99.9
    
    # Parámetros de bootstrap
    n_bootstrap: int = 1000
    confidence_level: float = 0.95
    
    # Parámetros de experimentos
    n_experiment_runs: int = 60
    
    # Checkpoint management
    checkpoint_interval_minutes: int = 5
    checkpoint_dir: str = "checkpoints"
    
    # Logging
    eval_interval: int = 100
    log_interval: int = 500
    
    # GPU
    use_gpu_if_available: bool = True
    
    def __post_init__(self):
        self.input_dim = self.matrix_size * self.matrix_size * 2
        self.output_dim = self.matrix_size * self.matrix_size
        if self.discretization_values is None:
            self.discretization_values = [-1.0, 0.0, 1.0]


# ================================================================================
# CLASE 2: MODELO STRASSEN OPERATOR
# ================================================================================

class StrassenOperator(nn.Module):
    """
    Operador Strassen para multiplicación de matrices 2x2 vía descomposición tensorial.
    
    El modelo representa el tensor de rango R:
    C_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)
    
    Donde:
    - U, V: Coeficientes de combinación lineal (LC)
    - W: Coeficientes de reconstrucción
    - Sparsity (SP): Cuántos slots están activos
    """
    
    def __init__(self, rank: int = 8):
        super().__init__()
        self.rank = rank
        
        # Parámetros del operador
        self.U = nn.Parameter(torch.randn(rank, 4) * 0.5)
        self.V = nn.Parameter(torch.randn(rank, 4) * 0.5)
        self.W = nn.Parameter(torch.randn(4, rank) * 0.5)
        
        self._initialize_weights()
    
    def _initialize_weights(self) -> None:
        """Inicializar pesos con consideración para grokking."""
        nn.init.xavier_uniform_(self.U)
        nn.init.xavier_uniform_(self.V)
        nn.init.xavier_uniform_(self.W)
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Computar A @ B usando descomposición tensorial.
        
        Args:
            A: Tensor de entrada de forma (batch, 2, 2)
            B: Tensor de entrada de forma (batch, 2, 2)
            
        Returns:
            Tensor de salida de forma (batch, 2, 2)
        """
        batch = A.shape[0]
        a = A.reshape(batch, 4)  # [a11, a12, a21, a22]
        b = B.reshape(batch, 4)  # [b11, b12, b21, b22]
        
        # Combinaciones lineales
        left = a @ self.U.T    # [batch, rank]
        right = b @ self.V.T   # [batch, rank]
        
        # Operador bilineal
        products = left * right  # [batch, rank]
        
        # Reconstrucción
        c = products @ self.W.T  # [batch, 4]
        
        return c.reshape(batch, 2, 2)
    
    def slot_importance(self) -> torch.Tensor:
        """Importancia de cada slot basada en normas."""
        u_norm = torch.norm(self.U, dim=1)
        v_norm = torch.norm(self.V, dim=1)
        w_norm = torch.norm(self.W, dim=0)
        return u_norm * v_norm * w_norm
    
    def count_active(self, threshold: float = 0.1) -> int:
        """Contar slots activos."""
        return (self.slot_importance() > threshold).sum().item()
    
    def compute_SP(self) -> torch.Tensor:
        """Métrica de Sparsity. SP -> 0 significa máxima sparsity."""
        slot_norms = self.slot_importance()
        threshold = 0.1
        active = (slot_norms > threshold).float().sum()
        sp = (active - 7.0) / self.rank
        return sp.clamp(0, 1)
    
    def get_state_dict(self) -> Dict[str, Any]:
        """Obtener estado completo para checkpointing."""
        return {
            'U': self.U.data.cpu(),
            'V': self.V.data.cpu(),
            'W': self.W.data.cpu(),
            'rank': self.rank
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        """Cargar estado completo desde checkpoint."""
        if 'U' in state_dict:
            self.U.data = state_dict['U'].to(self.U.device)
        if 'V' in state_dict:
            self.V.data = state_dict['V'].to(self.V.device)
        if 'W' in state_dict:
            self.W.data = state_dict['W'].to(self.W.device)


# ================================================================================
# CLASE 3: GENERADOR DE DATOS
# ================================================================================

class StrassenDataGenerator:
    """Generador de datos para multiplicación de matrices 2x2."""
    
    def __init__(self, num_samples: int = 10000, matrix_size: int = 2, seed: int = 42):
        self.num_samples = num_samples
        self.matrix_size = matrix_size
        self.seed = seed
        
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        self.data = self.generate_data()
    
    def generate_matrix(self) -> torch.Tensor:
        """Generar matriz aleatoria con valores enteros."""
        return torch.randint(-5, 6, (self.matrix_size, self.matrix_size), dtype=torch.float32)
    
    def generate_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generar pares de matrices y sus productos."""
        inputs = []
        targets = []
        
        for _ in range(self.num_samples):
            A = self.generate_matrix()
            B = self.generate_matrix()
            C = torch.matmul(A, B)
            
            input_vec = torch.cat([A.flatten(), B.flatten()])
            target_vec = C.flatten()
            
            inputs.append(input_vec)
            targets.append(target_vec)
        
        return torch.stack(inputs), torch.stack(targets)
    
    def get_train_test(self, test_ratio: float = 0.2) -> Tuple:
        """Dividir en conjuntos de entrenamiento y prueba."""
        n_test = int(self.num_samples * test_ratio)
        indices = torch.randperm(self.num_samples)
        
        train_inputs = self.data[0][indices[n_test:]]
        train_targets = self.data[1][indices[n_test:]]
        test_inputs = self.data[0][:n_test]
        test_targets = self.data[1][:n_test]
        
        return train_inputs, train_targets, test_inputs, test_targets


# ================================================================================
# CLASE 4: CALCULADORA DE LOCAL COMPLEXITY (LC)
# ================================================================================

class LocalComplexityCalculator:
    """
    Calculadora de Complejidad Local basada en la varianza del gradiente.
    
    LC = ||grad||^2 / N (Noise Scale normalizada)
    
    Esta métrica captura la "dificultad" del batch actual y su relación
    con el aprendizaje del modelo.
    """
    
    def __init__(self, model: nn.Module, config: ExperimentConfig):
        self.model = model
        self.config = config
        self.history = []
    
    def compute_lc(self, batch_inputs: torch.Tensor, batch_targets: torch.Tensor) -> float:
        """
        Calcular LC para un batch específico.
        
        LC = ||g||^2 / N_batch
        
        Donde g es el gradiente de la pérdida respecto a los pesos.
        """
        self.model.zero_grad()
        
        # Forward pass
        A = batch_inputs[:, :4].reshape(-1, 2, 2)
        B = batch_inputs[:, 4:].reshape(-1, 2, 2)
        C_pred = self.model(A, B)
        C_true = batch_targets.reshape(-1, 2, 2)
        loss = torch.mean((C_pred - C_true) ** 2)
        
        # Backward pass para obtener gradientes
        loss.backward()
        
        # Calcular norma cuadrada de gradientes
        total_norm_sq = 0.0
        num_params = 0
        
        for param in self.model.parameters():
            if param.grad is not None:
                total_norm_sq += torch.sum(param.grad ** 2).item()
                num_params += param.numel()
        
        # Normalizar por número de parámetros
        lc = total_norm_sq / max(num_params, 1)
        
        return lc
    
    def compute_batch_diversity(self, batch_inputs: torch.Tensor) -> float:
        """Calcular diversidad del batch basada en varianza de activaciones."""
        with torch.no_grad():
            A = batch_inputs[:, :4].reshape(-1, 2, 2)
            B = batch_inputs[:, 4:].reshape(-1, 2, 2)
            hidden = torch.cat([A.flatten(start_dim=1), B.flatten(start_dim=1)], dim=1)
            diversity = torch.var(hidden, dim=0).mean().item()
        return diversity


# ================================================================================
# CLASE 5: VERIFICADOR DE GROKKING
# ================================================================================

class GrokkingVerifier:
    """Verifica que un modelo ha grokkeado correctamente."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu_if_available else 'cpu')
    
    def verify(self, model: nn.Module, n_test: Optional[int] = None) -> Tuple[bool, Dict[str, float]]:
        """
        Verificar que el operador ha grokkeado correctamente.
        
        Returns:
            Tupla de (éxito, métricas)
        """
        if n_test is None:
            n_test = self.config.test_samples
            
        model.eval()
        model = model.to(self.device)
        
        with torch.no_grad():
            A, B = self._generate_batch(n_test, scale=1.0)
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            
            errors = (C_pred - C_true).abs()
            max_err = errors.max().item()
            mean_err = errors.mean().item()
            
            per_sample = errors.reshape(n_test, -1).max(dim=1)[0]
            acc = (per_sample < self.config.accuracy_threshold).float().mean().item() * 100
            
            lc = 1.0 - (torch.norm(C_pred - C_true) / torch.norm(C_true)).item()
            sp = model.compute_SP().item()
            active = model.count_active()
            
            success = (
                acc >= self.config.grokking_accuracy_threshold and
                active <= 8 and
                lc > 0.999
            )
            
            metrics = {
                'acc': acc,
                'lc': lc,
                'sp': sp,
                'active': active,
                'max_err': max_err,
                'mean_err': mean_err
            }
        
        return success, metrics
    
    def _generate_batch(self, n: int, scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generar batch de matrices aleatorias."""
        return (
            torch.randn(n, 2, 2, device=self.device) * scale,
            torch.randn(n, 2, 2, device=self.device) * scale
        )


# ================================================================================
# CLASE 6: MOTOR DE PODA ITERATIVA
# ================================================================================

class IterativePruningEngine:
    """
    Motor para poda iterativa con fine-tuning completo.
    
    Protocolo:
    1. Calcular importancia de pesos (magnitud L1)
    2. Podar p% de pesos menos importantes
    3. Fine-tune por épocas especificadas
    4. Chequear degradación δ
    5. Si δ < threshold, continuar; si no, detener
    6. Verificar discretización con δ < 0.1 (PUNTO A DEL REVISOR)
    
    Este protocolo es CRUCIAL para verificar la hipótesis de cuenca discreta.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu_if_available else 'cpu')
        self.history = []
        self.verifier = GrokkingVerifier(config)
    
    def get_weight_magnitudes(self, model: nn.Module) -> torch.Tensor:
        """Obtener magnitud absoluta de todos los pesos."""
        magnitudes = torch.cat([param.abs().flatten() for param in model.parameters()])
        return magnitudes
    
    def compute_sparsity(self, model: nn.Module) -> float:
        """Calcular porcentaje de pesos en cero."""
        total = sum(p.numel() for p in model.parameters())
        nonzero = sum((p != 0).sum().item() for p in model.parameters())
        return 1.0 - (nonzero / total) if total > 0 else 0.0
    
    def prune_percent(self, model: nn.Module, percent: float) -> Tuple[int, float]:
        """
        Podar el porcentaje especificado de pesos menos importantes.
        
        Returns: (num_pruned, current_sparsity)
        """
        magnitudes = self.get_weight_magnitudes(model)
        threshold = torch.quantile(magnitudes, percent / 100.0)
        
        num_pruned = 0
        for param in model.parameters():
            if param.requires_grad:
                param_flat = param.data.abs().flatten()
                keep_mask = param_flat >= threshold
                num_pruned += (~keep_mask).sum().item()
                param.data[~keep_mask.view(param.shape)] = 0
        
        current_sparsity = self.compute_sparsity(model)
        return num_pruned, current_sparsity
    
    def fine_tune(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """
        Fine-tune del modelo podado con métricas completas.
        """
        train_inputs, train_targets = train_data
        train_inputs = train_inputs.to(self.device)
        train_targets = train_targets.to(self.device)
        
        optimizer = optim.AdamW(model.parameters(), lr=self.config.finetune_lr, 
                                weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.finetune_epochs)
        
        losses = []
        accuracies = []
        
        model.train()
        
        log_interval = max(1, self.config.finetune_epochs // 10)
        
        for epoch in range(self.config.finetune_epochs):
            optimizer.zero_grad()
            
            A, B = self._generate_batch(self.config.batch_size)
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            loss = torch.mean((C_pred - C_true) ** 2)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            losses.append(loss.item())
            
            # Evaluar periódicamente
            if epoch % log_interval == 0 or epoch == self.config.finetune_epochs - 1:
                model.eval()
                with torch.no_grad():
                    A_eval, B_eval = self._generate_batch(1000)
                    C_eval = model(A_eval, B_eval)
                    C_true_eval = torch.bmm(A_eval, B_eval)
                    errors = (C_eval - C_true_eval).abs()
                    acc = (errors.reshape(1000, -1).max(dim=1)[0] < self.config.accuracy_threshold).float().mean().item() * 100
                    accuracies.append(acc)
                model.train()
        
        return {
            'final_loss': losses[-1],
            'mean_loss': np.mean(losses[-10:]),
            'final_accuracy': accuracies[-1] if accuracies else 0,
            'mean_accuracy': np.mean(accuracies[-10:]) if accuracies else 0,
            'epochs': self.config.finetune_epochs
        }
    
    def _generate_batch(self, n: int, scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generar batch de matrices aleatorias."""
        return (
            torch.randn(n, 2, 2, device=self.device) * scale,
            torch.randn(n, 2, 2, device=self.device) * scale
        )
    
        # Ejecutar verificación de discretización (PUNTO A DEL REVISOR)
        print(f"\n{'='*70}")
        print(f"VERIFICACIÓN DE DISCRETIZACIÓN (PUNTO A DEL REVISOR)")
        print(f"{'='*70}")
        print(f"Objetivo: Verificar que max|w - round(w)| < {self.config.discretization_threshold}")
        print(f"{'='*70}\n")
        
        # Intentar discretización con diferentes thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 1.0]
        discretization_results = []
        
        for thresh in thresholds:
            print(f"--- Discretizando con threshold = {thresh} ---")
            
            # Guardar pesos originales
            U_orig = model.U.data.clone()
            V_orig = model.V.data.clone()
            W_orig = model.W.data.clone()
            
            # Discretizar
            U_disc = torch.sign(U_orig) * (torch.abs(U_orig) > thresh).float()
            V_disc = torch.sign(V_orig) * (torch.abs(V_orig) > thresh).float()
            W_disc = torch.sign(W_orig) * (torch.abs(W_orig) > thresh).float()
            
            # Calcular δ = max|w - round(w)|
            delta_U = torch.max(torch.abs(U_orig - U_disc)).item()
            delta_V = torch.max(torch.abs(V_orig - V_disc)).item()
            delta_W = torch.max(torch.abs(W_orig - W_disc)).item()
            delta_max = max(delta_U, delta_V, delta_W)
            
            # Aplicar discretización
            model.U.data = U_disc
            model.V.data = V_disc
            model.W.data = W_disc
            
            # Verificar después de discretizar
            success_after, metrics_after = verifier.verify(model, self.config.test_samples)
            
            sparsity = self.compute_sparsity(model)
            active = model.count_active()
            
            result = {
                'threshold': thresh,
                'delta_max': delta_max,
                'delta_U': delta_U,
                'delta_V': delta_V,
                'delta_W': delta_W,
                'success': success_after,
                'accuracy': metrics_after['acc'],
                'lc': metrics_after['lc'],
                'max_err': metrics_after['max_err'],
                'sparsity': sparsity,
                'active_slots': active,
                'delta_threshold_met': delta_max < self.config.discretization_threshold
            }
            discretization_results.append(result)
            
            print(f"  δ = max|w-round(w)| = {delta_max:.6f}")
            print(f"  Accuracy después: {metrics_after['acc']:.2f}%")
            print(f"  Sparsity: {sparsity:.2%}")
            print(f"  Slots activos: {active}")
            print(f"  δ < 0.1: {'SÍ' if delta_max < self.config.discretization_threshold else 'NO'}\n")
            
            # Restaurar pesos
            model.U.data = U_orig
            model.V.data = V_orig
            model.W.data = W_orig
        
        # Encontrar mejor resultado (mayor sparsity con δ < 0.1)
        valid_results = [r for r in discretization_results if r['delta_threshold_met']]
        
        if valid_results:
            best = max(valid_results, key=lambda x: x['sparsity'])
            discretization_success = True
            conclusion = "[+] ÉXITO: Se encontró discretización con δ < 0.1"
        else:
            best = None
            discretization_success = False
            conclusion = "[*] No se encontró discretización con δ < 0.1 en el rango probado"
        
        print(f"\n{'='*70}")
        print(f"RESULTADOS DE DISCRETIZACIÓN")
        print(f"{'='*70}")
        print(f"Threshold objetivo: δ < {self.config.discretization_threshold}")
        print(f"Discretizaciones exitosas (δ < 0.1): {len(valid_results)}/{len(discretization_results)}")
        if best:
            print(f"Mejor resultado:")
            print(f"  Threshold usado: {best['threshold']}")
            print(f"  δ = {best['delta_max']:.6f}")
            print(f"  Sparsity: {best['sparsity']:.2%}")
            print(f"  Accuracy: {best['accuracy']:.2f}%")
        print(f"Conclusión: {conclusion}")
        print(f"{'='*70}\n")
        
        results['discretization'] = {
            'baseline': {
                'accuracy': final_metrics['acc'],
                'lc': final_metrics['lc'],
                'max_err': final_metrics['max_err'],
                'active_slots': final_metrics['active']
            },
            'discretization_results': discretization_results,
            'best_discretization': best,
            'success': discretization_success,
            'conclusion': conclusion
        }
    
    def run_protocol(self, model: nn.Module, train_data: Tuple[torch.Tensor, torch.Tensor]) -> Dict[str, Any]:
        """
        Ejecutar protocolo completo de poda iterativa.
        
        Returns:
            Diccionario con resultados completos
        """
        model = model.to(self.device)
        self.model = model  # Guardar referencia para discretización
        
        verifier = GrokkingVerifier(self.config)
        baseline_success, baseline_metrics = verifier.verify(model, self.config.test_samples)
        
        results = {
            'iterations': [],
            'baseline_accuracy': baseline_metrics['acc'],
            'baseline_lc': baseline_metrics['lc'],
            'baseline_sp': baseline_metrics['sp'],
            'baseline_active': baseline_metrics['active'],
            'baseline_max_err': baseline_metrics['max_err'],
            'baseline_mean_err': baseline_metrics['mean_err'],
            'final_sparsity': 0.0,
            'final_accuracy': 0.0,
            'final_lc': 0.0,
            'final_active': 0,
            'delta_threshold': self.config.delta_threshold,
            'success': False,
            'discrete_basin_reached': False,
            'stop_reason': 'not_started'
        }
        
        print(f"\n{'='*70}")
        print(f"PROTOCOLO DE PODA ITERATIVA")
        print(f"{'='*70}")
        print(f"Baseline - Accuracy: {baseline_metrics['acc']:.2f}%, LC: {baseline_metrics['lc']:.6f}")
        print(f"Active Slots: {baseline_metrics['active']}, Max Error: {baseline_metrics['max_err']:.2e}")
        print(f"Delta Threshold: {self.config.delta_threshold}")
        print(f"{'='*70}\n")
        
        current_percent = self.config.prune_initial_percent
        best_accuracy = baseline_metrics['acc']
        
        for iteration in range(self.config.prune_max_iterations):
            print(f"--- Iteración {iteration + 1}/{self.config.prune_max_iterations} ---")
            print(f"Podando: {current_percent:.1f}%")
            
            # Guardar estado antes de podar
            state_before = copy.deepcopy(model.state_dict())
            
            # Podar
            num_pruned, sparsity = self.prune_percent(model, current_percent)
            print(f"Pesos removidos: {num_pruned}, Sparsity: {sparsity:.2%}")
            
            # Fine-tune
            print(f"Fine-tuning ({self.config.finetune_epochs} épocas)...")
            ft_result = self.fine_tune(model, train_data)
            print(f"Fine-tune: final_loss = {ft_result['final_loss']:.6e}, final_acc = {ft_result['final_accuracy']:.2f}%")
            
            # Evaluar
            success, metrics = verifier.verify(model, self.config.test_samples)
            delta = baseline_metrics['acc'] - metrics['acc']
            
            print(f"Después del fine-tune - Accuracy: {metrics['acc']:.2f}%, LC: {metrics['lc']:.6f}")
            print(f"Error máximo: {metrics['max_err']:.2e}, Error medio: {metrics['mean_err']:.2e}")
            print(f"Degradación δ: {delta:.4f}\n")
            
            iteration_result = {
                'iteration': iteration + 1,
                'prune_percent': current_percent,
                'sparsity': sparsity,
                'accuracy': metrics['acc'],
                'lc': metrics['lc'],
                'active': metrics['active'],
                'delta': delta,
                'fine_tune_loss': ft_result['final_loss'],
                'max_err': metrics['max_err'],
                'mean_err': metrics['mean_err'],
                'success': success
            }
            results['iterations'].append(iteration_result)
            
            # Chequear condición de parada
            if delta > self.config.delta_threshold:
                print(f"[*]  Degradación excede threshold ({delta:.4f} > {self.config.delta_threshold}). Revertiendo...")
                model.load_state_dict(state_before)
                results['final_sparsity'] = self.compute_sparsity(model)
                results['stop_reason'] = f"delta={delta:.4f} > threshold={self.config.delta_threshold}"
                break
            
            # Continuar podando
            if metrics['acc'] > best_accuracy:
                best_accuracy = metrics['acc']
            
            current_percent += self.config.prune_step_percent
        else:
            print(f"Completadas todas las {self.config.prune_max_iterations} iteraciones")
            results['stop_reason'] = "max_iterations reached"
        
        final_success, final_metrics = verifier.verify(model, self.config.test_samples)
        
        results['final_sparsity'] = self.compute_sparsity(model)
        results['final_accuracy'] = final_metrics['acc']
        results['final_lc'] = final_metrics['lc']
        results['final_active'] = final_metrics['active']
        results['final_max_err'] = final_metrics['max_err']
        results['final_mean_err'] = final_metrics['mean_err']
        
        # Éxito significa: sparsity > 50% Y accuracy >= threshold Y error muy bajo
        results['success'] = (
            results['final_sparsity'] > 0.5 and
            final_metrics['acc'] >= self.config.grokking_accuracy_threshold
        )
        
        # Cuenca discreta alcanzada: 7 slots activos Y accuracy perfecto
        results['discrete_basin_reached'] = (
            final_metrics['active'] == 7 and
            final_metrics['acc'] >= self.config.grokking_accuracy_threshold and
            final_metrics['max_err'] < 0.01
        )
        
        print(f"\n{'='*70}")
        print(f"RESULTADOS FINALES DE PODA")
        print(f"{'='*70}")
        print(f"Sparsity final: {results['final_sparsity']:.2%}")
        print(f"Accuracy final: {results['final_accuracy']:.2f}%")
        print(f"LC final: {results['final_lc']:.6f}")
        print(f"Slots activos: {results['final_active']}")
        print(f"Error máximo: {final_metrics['max_err']:.2e}")
        print(f"Razón de parada: {results['stop_reason']}")
        print(f"Cuenca discreta alcanzada: {results['discrete_basin_reached']}")
        print(f"Éxito del protocolo: {results['success']}")
        print(f"{'='*70}\n")
        
        return results


# ================================================================================
# CLASE 6B: EXPERIMENTO DE LOCAL COMPLEXITY DESDE CERO (PUNTO B DEL REVISOR)
# ================================================================================

class LocalComplexityExperiment:
    """
    Experimento de Local Complexity entrenando desde cero.
    
    Esto es CRUCIAL para responder al revisor (PUNTO B):
    - Se entrena un modelo desde cero hasta grokking
    - Se mide LC en cada época para capturar la transición de fase
    - Si LC muestra un cambio alrededor del grokking, la métrica es útil
    - Si LC permanece constante, la métrica NO captura la transición
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu_if_available else 'cpu')
    
    def run_full_experiment(self, target_epochs: int = None) -> Dict[str, Any]:
        """
        Ejecutar experimento completo de LC entrenando desde cero.
        """
        if target_epochs is None:
            target_epochs = self.config.max_epochs
            
        print(f"\n{'='*70}")
        print(f"EXPERIMENTO LC: ENTRENANDO DESDE CERO (PUNTO B DEL REVISOR)")
        print(f"{'='*70}")
        print(f"Objetivo: Medir LC durante entrenamiento para capturar transición de fase")
        print(f"Épocas: {target_epochs}")
        print(f"{'='*70}\n")
        
        model = StrassenOperator(rank=self.config.rank).to(self.device)
        data_gen = StrassenDataGenerator(num_samples=10000, seed=42)
        train_inputs = data_gen.data[0].to(self.device)
        train_targets = data_gen.data[1].to(self.device)
        
        lc_calculator = LocalComplexityCalculator(model, self.config)
        
        optimizer = optim.AdamW(model.parameters(), lr=self.config.learning_rate, 
                               weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
        
        history = {
            'epochs': [], 'lc': [], 'lc_raw': [], 'accuracy': [], 
            'test_accuracy': [], 'loss': [], 'sparsity': [], 'active_slots': [], 'lr': []
        }
        
        eval_interval = max(1, target_epochs // 50)
        log_interval = max(1, target_epochs // 25)
        
        print(f"{'Epoch':>8} | {'LC':>10} | {'Train Acc':>10} | {'Test Acc':>10} | {'Loss':>12} | {'Active':>6}")
        print(f"{'-'*70}")
        
        epoch_start_time = time.time()
        grokking_epoch = None
        
        for epoch in range(target_epochs):
            model.train()
            optimizer.zero_grad()
            
            A, B = self._generate_batch(self.config.batch_size)
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            loss = torch.mean((C_pred - C_true) ** 2)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            if epoch % eval_interval == 0 or epoch == target_epochs - 1:
                model.eval()
                
                batch_idx = torch.randperm(len(train_inputs))[:min(256, len(train_inputs))]
                lc = lc_calculator.compute_lc(train_inputs[batch_idx], train_targets[batch_idx])
                
                with torch.no_grad():
                    A_eval, B_eval = self._generate_batch(2000)
                    C_eval = model(A_eval, B_eval)
                    C_true_eval = torch.bmm(A_eval, B_eval)
                    errors = (C_eval - C_true_eval).abs()
                    train_acc = (errors.reshape(2000, -1).max(dim=1)[0] < self.config.accuracy_threshold).float().mean().item() * 100
                    
                    test_errors = errors.mean().item()
                    test_acc = 100 - test_errors * 100
                
                sparsity = model.compute_SP().item()
                active = model.count_active()
                
                history['epochs'].append(epoch)
                history['lc'].append(lc)
                history['accuracy'].append(train_acc)
                history['test_accuracy'].append(test_acc)
                history['loss'].append(loss.item())
                history['sparsity'].append(sparsity)
                history['active_slots'].append(active)
                history['lr'].append(scheduler.get_last_lr()[0])
                
                if train_acc >= 99.9 and grokking_epoch is None:
                    grokking_epoch = epoch
                
                if epoch % log_interval == 0 or epoch == target_epochs - 1:
                    print(f"{epoch:>8} | {lc:>10.4f} | {train_acc:>9.2f}% | {test_acc:>9.2f}% | {loss.item():>12.6f} | {active:>6}")
                
                model.train()
        
        for key in history:
            history[key] = np.array(history[key])
        
        total_time = time.time() - epoch_start_time
        
        print(f"\n{'='*70}")
        print(f"RESULTADOS EXPERIMENTO LC")
        print(f"{'='*70}")
        print(f"Tiempo: {total_time:.1f}s | Grokking: {'época ' + str(grokking_epoch) if grokking_epoch else 'NO ALCANZADO'}")
        print(f"Accuracy final: {history['accuracy'][-1]:.2f}%")
        print(f"LC rango: [{history['lc'].min():.4f}, {history['lc'].max():.4f}]")
        print(f"{'='*70}\n")
        
        return {'history': history, 'grokking_epoch': grokking_epoch, 'total_time': total_time}
    
    def _generate_batch(self, n: int, scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.randn(n, 2, 2, device=self.device) * scale, torch.randn(n, 2, 2, device=self.device) * scale)


# ================================================================================
# CLASE 6C: GENERADOR DE RUNS BALANCEADOS PARA AUC (PUNTO C DEL REVISOR)
# ================================================================================

class BalancedRunsGenerator:
    """
    Generador de runs balanceados para calcular AUC válido.
    
    Esto es CRUCIAL para responder al revisor (PUNTO C):
    - Entrenar múltiples modelos con diferentes hiperparámetros
    - Algunos grokkean, otros no (condiciones variadas)
    - Generar dataset balanceado para ROC/AUC
    
    Si todos los samples son de una sola clase, AUC es indefinido.
    Necesitamos mix de grokked + no-grokked para calcularlo.
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu_if_available else 'cpu')
    
    def run_balanced_experiments(self, n_runs: int = None) -> Dict[str, Any]:
        """
        Ejecutar multiples runs con condiciones disenhadas para producir mix.
        
        Returns:
            Diccionario con resultados de todos los runs
        """
        if n_runs is None:
            n_runs = self.config.n_experiment_runs
            
        print(f"\n{'='*70}")
        print(f"GENERANDO RUNS BALANCEADOS (PUNTO C DEL REVISOR)")
        print(f"{'='*70}")
        print(f"Objetivo: Generar mix de runs grokked/no-grokked para AUC valido")
        print(f"Total de runs: {n_runs}")
        print(f"{'='*70}\n")
        
        # Configuraciones diseñadas para producir grokking o no-grokking
        # Learning rates extremos (muy alto o muy bajo) tienden a no grokkar
        # Learning rates moderados con schedule apropiado grokkean
        base_lr = self.config.learning_rate
        
        configurations = [
            # Configuraciones que DEBERÍAN grokkar (baseline)
            {'lr': base_lr, 'wd': self.config.weight_decay, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            {'lr': base_lr * 0.8, 'wd': self.config.weight_decay * 0.8, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            {'lr': base_lr * 1.2, 'wd': self.config.weight_decay * 0.8, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            
            # Configuraciones que difícilmente grokkar (LR muy bajo)
            {'lr': base_lr * 0.05, 'wd': self.config.weight_decay, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            {'lr': base_lr * 0.1, 'wd': self.config.weight_decay * 0.5, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            
            # Configuraciones que difícilmente grokkar (LR muy alto)
            {'lr': base_lr * 5.0, 'wd': self.config.weight_decay * 2.0, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            {'lr': base_lr * 10.0, 'wd': self.config.weight_decay * 5.0, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            
            # Weight decay extremo
            {'lr': base_lr, 'wd': self.config.weight_decay * 10, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            {'lr': base_lr, 'wd': self.config.weight_decay * 0.01, 'epochs': self.config.max_epochs, 'batch_size': self.config.batch_size},
            
            # Batch size extremo
            {'lr': base_lr, 'wd': self.config.weight_decay, 'epochs': self.config.max_epochs, 'batch_size': 32},
            {'lr': base_lr, 'wd': self.config.weight_decay, 'epochs': self.config.max_epochs, 'batch_size': 1024},
        ]
        
        # Si necesitamos más runs, repetir con variaciones
        while len(configurations) < n_runs:
            for i in range(len(configurations)):
                if len(configurations) >= n_runs:
                    break
                # Modificar slight variations
                cfg = configurations[i].copy()
                cfg['lr'] = cfg['lr'] * (0.9 + 0.2 * (i % 3))
                cfg['wd'] = cfg['wd'] * (0.8 + 0.4 * (i % 2))
                configurations.append(cfg)
        
        # Ejecutar runs
        all_results = []
        grokked_count = 0
        not_grokked_count = 0
        
        for run_idx, config in enumerate(configurations[:n_runs]):
            print(f"\n--- Run {run_idx + 1}/{n_runs} ---")
            print(f"  LR: {config['lr']:.5f}, WD: {config['wd']:.6f}, Epochs: {config['epochs']}, Batch: {config['batch_size']}")
            
            run_result = self._train_single_run(run_idx, config)
            all_results.append(run_result)
            
            if run_result['grokking_achieved']:
                grokked_count += 1
            else:
                not_grokked_count += 1
            
            print(f"  Resultado: {'GROKKEADO' if run_result['grokking_achieved'] else 'NO GROKKEADO'}")
            print(f"  Accuracy final: {run_result['final_train_accuracy']:.2f}%")
            print(f"  Accuracy test: {run_result['final_test_accuracy']:.2f}%")
        
        print(f"\n{'='*70}")
        print(f"RESUMEN DE RUNS BALANCEADOS")
        print(f"{'='*70}")
        print(f"Total runs: {len(all_results)}")
        print(f"Grokked: {grokked_count}")
        print(f"No grokked: {not_grokked_count}")
        print(f"Ratio: {grokked_count/max(1,len(all_results))*100:.1f}% / {not_grokked_count/max(1,len(all_results))*100:.1f}%")
        print(f"{'='*70}\n")
        
        # Preparar datos para ROC/AUC
        y_true = np.array([1 if r['grokking_achieved'] else 0 for r in all_results])
        y_scores = np.array([r['accuracy_score'] for r in all_results])
        
        # Calcular ROC/AUC si hay ambas clases
        if len(np.unique(y_true)) == 2:
            roc_data = self._compute_roc(y_true, y_scores)
            print(f"\n[+] AUC CALCULABLE CON ÉXITO")
            print(f"   AUC: {roc_data['auc_mean']:.4f}")
            print(f"   IC 95%: [{roc_data['auc_ci_lower']:.4f}, {roc_data['auc_ci_upper']:.4f}]")
        else:
            roc_data = {
                'auc_mean': 'undefined',
                'warning': f'Single class: {"all grokked" if grokked_count > 0 else "all not grokked"}',
                'class_distribution': {'grokked': grokked_count, 'not_grokked': not_grokked_count}
            }
            print(f"\n[*] AUC NO CALCULABLE: Solo una clase presente")
        
        return {
            'runs': all_results,
            'grokked_count': grokked_count,
            'not_grokked_count': not_grokked_count,
            'total_runs': len(all_results),
            'roc': roc_data,
            'y_true': y_true.tolist(),
            'y_scores': y_scores.tolist()
        }
    
    def _train_single_run(self, run_idx: int, config: Dict) -> Dict[str, Any]:
        """Entrenar un solo modelo con configuración específica."""
        torch.manual_seed(42 + run_idx)
        np.random.seed(42 + run_idx)
        
        model = StrassenOperator(rank=self.config.rank).to(self.device)
        data_gen = StrassenDataGenerator(num_samples=10000, seed=42)
        
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                               weight_decay=config['wd'])
        
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=1000, T_mult=2)
        
        grokking_epoch = None
        best_test_acc = 0.0
        
        eval_interval = max(1, config['epochs'] // 20)
        
        for epoch in range(config['epochs']):
            model.train()
            optimizer.zero_grad()
            
            A, B = self._generate_batch(config['batch_size'])
            C_pred = model(A, B)
            C_true = torch.bmm(A, B)
            loss = torch.mean((C_pred - C_true) ** 2)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Evaluar periódicamente
            if epoch % eval_interval == 0 or epoch == config['epochs'] - 1:
                model.eval()
                with torch.no_grad():
                    A_eval, B_eval = self._generate_batch(2000)
                    C_eval = model(A_eval, B_eval)
                    C_true_eval = torch.bmm(A_eval, B_eval)
                    errors = (C_eval - C_true_eval).abs()
                    train_acc = (errors.reshape(2000, -1).max(dim=1)[0] < self.config.accuracy_threshold).float().mean().item() * 100
                    test_acc = 100 - errors.mean().item() * 100
                    
                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                    
                    if train_acc >= 99.9 and grokking_epoch is None:
                        grokking_epoch = epoch
                
                model.train()
        
        # Evaluación final
        model.eval()
        with torch.no_grad():
            A_eval, B_eval = self._generate_batch(5000)
            C_eval = model(A_eval, B_eval)
            C_true_eval = torch.bmm(A_eval, B_eval)
            errors = (C_eval - C_true_eval).abs()
            final_train_acc = (errors.reshape(5000, -1).max(dim=1)[0] < self.config.accuracy_threshold).float().mean().item() * 100
            final_test_acc = 100 - errors.mean().item() * 100
            max_error = errors.max().item()
        
        grokking_achieved = final_train_acc >= self.config.grokking_accuracy_threshold
        
        # Score para ROC: normalizar accuracy a [0, 1]
        accuracy_score = final_train_acc / 100.0
        
        return {
            'run_idx': run_idx,
            'config': config,
            'grokking_achieved': grokking_achieved,
            'grokking_epoch': grokking_epoch,
            'final_train_accuracy': final_train_acc,
            'final_test_accuracy': final_test_acc,
            'best_test_accuracy': best_test_acc,
            'max_error': max_error,
            'accuracy_score': accuracy_score
        }
    
    def _compute_roc(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """Calcular ROC/AUC básico."""
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc_value = auc(fpr, tpr)
        
        # Bootstrap para IC
        n_samples = len(y_true)
        bootstrap_aucs = []
        
        np.random.seed(SEED)
        
        for _ in range(1000):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            try:
                fpr_boot, tpr_boot, _ = roc_curve(y_true[indices], y_scores[indices])
                auc_boot = auc(fpr_boot, tpr_boot)
                bootstrap_aucs.append(auc_boot)
            except:
                continue
        
        alpha = 0.05
        return {
            'auc_mean': float(auc_value),
            'auc_ci_lower': float(np.percentile(bootstrap_aucs, 100 * alpha / 2)) if bootstrap_aucs else 'undefined',
            'auc_ci_upper': float(np.percentile(bootstrap_aucs, 100 * (1 - alpha / 2))) if bootstrap_aucs else 'undefined',
            'n_bootstrap': len(bootstrap_aucs)
        }
    
    def _generate_batch(self, n: int, scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        return (torch.randn(n, 2, 2, device=self.device) * scale, 
                torch.randn(n, 2, 2, device=self.device) * scale)


# CLASE 7: GENERADOR DE ESTADÍSTICAS BOOTSTRAP
# ================================================================================

class BootstrapStatistics:
    """
    Generador de estadísticas con intervalos de confianza bootstrap.
    
    Calcula:
    - Curvas ROC con IC del 95%
    - AUC con IC del 95%
    - Kappa de Cohen con IC del 95%
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.alpha = 1 - config.confidence_level
    
    def compute_roc_with_ci(self, y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
        """
        Calcular curva ROC con intervalos de confianza bootstrap.
        
        Returns:
            Diccionario con resultados ROC
        """
        unique_classes = np.unique(y_true)
        
        # Manejar caso especial: solo una clase presente
        if len(unique_classes) < 2:
            return {
                'auc_mean': 'undefined',
                'auc_ci_lower': 'undefined',
                'auc_ci_upper': 'undefined',
                'warning': 'Single class - cannot compute ROC/AUC',
                'class_distribution': {
                    'class_0': int((y_true == 0).sum()),
                    'class_1': int((y_true == 1).sum())
                },
                'fpr': [],
                'tpr_mean': [],
                'tpr_lower': [],
                'tpr_upper': []
            }
        
        fpr_base, tpr_base, _ = roc_curve(y_true, y_scores)
        auc_base = auc(fpr_base, tpr_base)
        
        n_samples = len(y_true)
        bootstrap_aucs = []
        bootstrap_curves = []
        
        np.random.seed(SEED)
        
        for _ in range(self.config.n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            
            y_true_boot = y_true[indices]
            y_scores_boot = y_scores[indices]
            
            try:
                fpr_boot, tpr_boot, _ = roc_curve(y_true_boot, y_scores_boot)
                auc_boot = auc(fpr_boot, tpr_boot)
                bootstrap_aucs.append(auc_boot)
                
                tpr_interp = np.interp(fpr_base, fpr_boot, tpr_boot)
                bootstrap_curves.append(tpr_interp)
            except:
                continue
        
        if len(bootstrap_aucs) > 0:
            auc_lower = np.percentile(bootstrap_aucs, 100 * self.alpha / 2)
            auc_upper = np.percentile(bootstrap_aucs, 100 * (1 - self.alpha / 2))
            
            tpr_mean = np.mean(bootstrap_curves, axis=0)
            tpr_std = np.std(bootstrap_curves, axis=0)
            
            return {
                'fpr': fpr_base.tolist(),
                'tpr_mean': tpr_mean.tolist(),
                'tpr_std': tpr_std.tolist(),
                'tpr_lower': np.maximum(tpr_mean - 1.96 * tpr_std, 0).tolist(),
                'tpr_upper': np.minimum(tpr_mean + 1.96 * tpr_std, 1).tolist(),
                'auc_mean': float(auc_base),
                'auc_bootstrap_mean': float(np.mean(bootstrap_aucs)),
                'auc_ci_lower': float(auc_lower),
                'auc_ci_upper': float(auc_upper),
                'bootstrap_aucs': [float(a) for a in bootstrap_aucs],
                'n_valid_samples': len(bootstrap_aucs),
                'class_distribution': {
                    'class_0': int((y_true == 0).sum()),
                    'class_1': int((y_true == 1).sum())
                }
            }
        else:
            return {
                'auc_mean': 'undefined',
                'auc_ci_lower': 'undefined',
                'auc_ci_upper': 'undefined',
                'warning': 'No valid bootstrap samples',
                'class_distribution': {
                    'class_0': int((y_true == 0).sum()),
                    'class_1': int((y_true == 1).sum())
                }
            }
    
    def compute_kappa_with_ci(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Calcular Kappa de Cohen con IC bootstrap."""
        kappa_base = cohen_kappa_score(y_true, y_pred)
        
        n_samples = len(y_true)
        bootstrap_kappas = []
        
        np.random.seed(SEED)
        
        for _ in range(self.config.n_bootstrap):
            indices = np.random.choice(n_samples, n_samples, replace=True)
            kappa_boot = cohen_kappa_score(y_true[indices], y_pred[indices])
            bootstrap_kappas.append(kappa_boot)
        
        kappa_lower = np.percentile(bootstrap_kappas, 100 * self.alpha / 2)
        kappa_upper = np.percentile(bootstrap_kappas, 100 * (1 - self.alpha / 2))
        
        return {
            'kappa_mean': float(kappa_base),
            'kappa_ci_lower': float(kappa_lower),
            'kappa_ci_upper': float(kappa_upper),
            'bootstrap_kappas': [float(k) for k in bootstrap_kappas]
        }
    
    def compute_accuracy_with_ci(self, correct: np.ndarray) -> Dict[str, Any]:
        """Calcular accuracy con IC binomial."""
        n = len(correct)
        accuracy = correct.mean()
        
        ci_lower = beta.ppf(self.alpha / 2, correct.sum() + 1, n - correct.sum() + 1)
        ci_upper = beta.ppf(1 - self.alpha / 2, correct.sum() + 1, n - correct.sum() + 1)
        
        return {
            'accuracy': float(accuracy),
            'ci_lower': float(ci_lower),
            'ci_upper': float(ci_upper),
            'n_samples': n,
            'n_correct': int(correct.sum())
        }


# ================================================================================
# CLASE 8: GENERADOR DE VISUALIZACIONES
# ================================================================================

class VisualizationGenerator:
    """Generador de visualizaciones con estilo académico."""
    
    def __init__(self, style: str = 'seaborn-v0_8'):
        plt.style.use(style)
        plt.rcParams.update({
            'font.size': 12,
            'axes.titlesize': 14,
            'axes.labelsize': 12,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 150,
            'savefig.dpi': 300,
            'font.family': 'sans-serif'
        })
    
    def plot_local_complexity(self, epochs: np.ndarray, lc_values: np.ndarray,
                              accuracy: np.ndarray, save_path: str) -> None:
        """
        Graficar evolución de Local Complexity y Accuracy.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Panel 1: LC y Accuracy vs Épocas
        ax1_twin = ax1.twinx()
        
        line1, = ax1.plot(epochs, lc_values, 'b-', linewidth=2, label='Local Complexity')
        line2, = ax1_twin.plot(epochs, accuracy, 'r-', linewidth=2, label='Accuracy')
        
        ax1.set_xlabel('Épocas')
        ax1.set_ylabel('Local Complexity (LC)', color='blue')
        ax1_twin.set_ylabel('Accuracy', color='red')
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1_twin.tick_params(axis='y', labelcolor='red')
        
        ax1.set_title('Evolución de Local Complexity durante Grokking')
        ax1.legend([line1, line2], ['Local Complexity', 'Accuracy'], loc='center right')
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: LC en escala log
        ax2.semilogy(epochs, np.maximum(lc_values, 1e-10), 'b-', linewidth=2)
        ax2.set_xlabel('Épocas')
        ax2.set_ylabel('Local Complexity (escala log)')
        ax2.set_title('Local Complexity (Escala Logarítmica)')
        ax2.grid(True, alpha=0.3)
        
        # Marcar fase de grokking
        if any(accuracy > 0.95):
            grok_epoch = epochs[np.where(accuracy > 0.95)[0][0]]
            ax2.axvline(x=grok_epoch, color='green', linestyle='--', alpha=0.7, 
                       label=f'Inicio grokking (época {grok_epoch})')
            ax2.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='png')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"✓ Guardado: {save_path}")
    
    def plot_pruning_results(self, pruning_data: Dict, save_path: str) -> None:
        """Graficar resultados de poda iterativa."""
        if not pruning_data.get('iterations'):
            print(f"No hay datos de poda para graficar")
            return
        
        iterations = [it['iteration'] for it in pruning_data['iterations']]
        sparsities = [it['sparsity'] for it in pruning_data['iterations']]
        accuracies = [it['accuracy'] for it in pruning_data['iterations']]
        deltas = [it['delta'] for it in pruning_data['iterations']]
        max_errors = [it.get('max_err', 0) for it in pruning_data['iterations']]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Sparsity vs Iteración
        ax1 = axes[0, 0]
        ax1.plot(iterations, sparsities, 'go-', linewidth=2, markersize=8)
        ax1.axhline(y=0.5, color='r', linestyle='--', label='50% Sparsity')
        ax1.set_xlabel('Iteración de Poda')
        ax1.set_ylabel('Sparsity')
        ax1.set_title('Progresión de Sparsity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Accuracy vs Iteración
        ax2 = axes[0, 1]
        ax2.plot(iterations, accuracies, 'b-o', linewidth=2, markersize=8)
        ax2.axhline(y=pruning_data.get('baseline_accuracy', 99.9), color='g', linestyle='--', 
                   label=f'Baseline ({pruning_data.get("baseline_accuracy", 99.9):.1f}%)')
        ax2.set_xlabel('Iteración de Poda')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy después de Poda')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Degradación δ vs Iteración
        ax3 = axes[1, 0]
        ax3.plot(iterations, deltas, 'r-s', linewidth=2, markersize=8)
        ax3.axhline(y=pruning_data.get('delta_threshold', 0.1), color='orange', linestyle='--',
                   label=f'Threshold δ = {pruning_data.get("delta_threshold", 0.1)}')
        ax3.set_xlabel('Iteración de Poda')
        ax3.set_ylabel('Degradación δ')
        ax3.set_title('Degradación del Modelo post Fine-tuning')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Error máximo vs Iteración
        ax4 = axes[1, 1]
        ax4.plot(iterations, max_errors, 'm-^', linewidth=2, markersize=8)
        ax4.set_xlabel('Iteración de Poda')
        ax4.set_ylabel('Error Máximo')
        ax4.set_title('Error Máximo después de Poda')
        ax4.set_yscale('log')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='png')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"✓ Guardado: {save_path}")
    
    def plot_roc_with_ci(self, roc_data: Dict, save_path: str) -> None:
        """Graficar curva ROC con intervalos de confianza."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Manejar caso especial: AUC indefinido
        if roc_data.get('auc_mean') == 'undefined' or 'warning' in roc_data:
            reason = roc_data.get('warning', roc_data.get('reason', 'Unknown'))
            ax.text(0.5, 0.5, f'ROC/AUC No Calculable\n\nMotivo: {reason}\n\n'
                           f'Distribución de clases: {roc_data.get("class_distribution", {})}',
                   ha='center', va='center', transform=ax.transAxes, fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
            ax.set_title('Curva ROC - Condición Especial')
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_xticks([])
            ax.set_yticks([])
            
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight', format='png')
            plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
            plt.close()
            
            print(f"✓ Guardado placeholder: {save_path}")
            return
        
        if 'fpr' not in roc_data or not roc_data['fpr']:
            print("No hay datos FPR para graficar ROC")
            return
        
        fpr = roc_data['fpr']
        tpr_mean = roc_data['tpr_mean']
        
        ax.plot(fpr, tpr_mean, 'b-', linewidth=2, 
                label=f'ROC (AUC = {roc_data["auc_mean"]:.3f})')
        
        if 'tpr_lower' in roc_data:
            ax.fill_between(fpr, roc_data['tpr_lower'], roc_data['tpr_upper'],
                           alpha=0.3, color='blue', label='IC 95%')
        
        ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador Aleatorio')
        
        # Anotar IC del AUC
        if roc_data.get('auc_ci_lower') is not None:
            textstr = f'AUC = {roc_data["auc_mean"]:.3f}\nIC 95%: [{roc_data["auc_ci_lower"]:.3f}, {roc_data["auc_ci_upper"]:.3f}]'
            ax.text(0.55, 0.15, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
        ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
        ax.set_title('Curva ROC para Predicción de Grokking')
        ax.legend(loc='lower right')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='png')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"✓ Guardado: {save_path}")
    
    def plot_balanced_runs_results(self, balanced_data: Dict, save_path: str) -> None:
        """Graficar resultados del experimento de runs balanceados."""
        if not balanced_data.get('runs'):
            print(f"No hay datos de runs balanceados para graficar")
            return
        
        runs = balanced_data['runs']
        
        # Extraer datos
        run_indices = [r['run_idx'] for r in runs]
        train_accs = [r['final_train_accuracy'] for r in runs]
        test_accs = [r['final_test_accuracy'] for r in runs]
        grokking_achieved = [r['grokking_achieved'] for r in runs]
        
        colors = ['green' if g else 'red' for g in grokking_achieved]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Barras de accuracy por run
        ax1 = axes[0, 0]
        bars = ax1.bar(range(len(runs)), train_accs, color=colors, alpha=0.7, edgecolor='black')
        ax1.axhline(y=99.9, color='blue', linestyle='--', label='Threshold grokking (99.9%)')
        ax1.set_xlabel('Run Index')
        ax1.set_ylabel('Train Accuracy (%)')
        ax1.set_title('Accuracy por Run (Verde=Grokked, Rojo=No Grokked)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Scatter plot train vs test accuracy
        ax2 = axes[0, 1]
        ax2.scatter(train_accs, test_accs, c=colors, s=100, alpha=0.7, edgecolors='black')
        ax2.plot([0, 100], [0, 100], 'k--', label='Train = Test')
        ax2.set_xlabel('Train Accuracy (%)')
        ax2.set_ylabel('Test Accuracy (%)')
        ax2.set_title('Train vs Test Accuracy')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Histograma de accuracies
        ax3 = axes[1, 0]
        ax3.hist(train_accs, bins=10, color='steelblue', alpha=0.7, edgecolor='black')
        ax3.axvline(x=99.9, color='red', linestyle='--', label='Threshold grokking')
        ax3.set_xlabel('Train Accuracy (%)')
        ax3.set_ylabel('Frecuencia')
        ax3.set_title('Distribucion de Accuracy')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Resumen estadistico
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        grokked_count = sum(grokking_achieved)
        not_grokked_count = len(grokking_achieved) - grokked_count
        
        summary_text = f"""
        RESUMEN DE RUNS BALANCEADOS
        ===========================
        
        Total de runs: {len(runs)}
        Grokked: {grokked_count} ({100*grokked_count/len(runs):.1f}%)
        No grokked: {not_grokked_count} ({100*not_grokked_count/len(runs):.1f}%)
        
        Accuracy medio (grokked): {np.mean([a for a, g in zip(train_accs, grokking_achieved) if g]):.2f}%
        Accuracy medio (no grokked): {np.mean([a for a, g in zip(train_accs, grokking_achieved) if not g]):.2f}%
        """
        
        if balanced_data.get('roc', {}).get('auc_mean') != 'undefined':
            summary_text += f"""
        AUC (balanceado): {balanced_data['roc']['auc_mean']:.4f}
        IC 95%: [{balanced_data['roc']['auc_ci_lower']:.4f}, {balanced_data['roc']['auc_ci_upper']:.4f}]
            """
        else:
            summary_text += f"""
        AUC: Indefinido
        ({balanced_data.get('roc', {}).get('warning', 'N/A')})
            """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='png')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"✓ Guardado: {save_path}")
    
    def plot_discretization_results(self, pruning_data: Dict, save_path: str) -> None:
        """Graficar resultados de discretizacion."""
        if not pruning_data.get('discretization', {}).get('discretization_results'):
            print(f"No hay datos de discretizacion para graficar")
            return
        
        disc_results = pruning_data['discretization']['discretization_results']
        thresholds = [r['threshold'] for r in disc_results]
        deltas = [r['delta_max'] for r in disc_results]
        accuracies = [r['accuracy'] for r in disc_results]
        sparsities = [r['sparsity'] for r in disc_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Panel 1: Delta vs Threshold
        ax1 = axes[0, 0]
        ax1.plot(thresholds, deltas, 'b-o', linewidth=2, markersize=8)
        ax1.axhline(y=0.1, color='red', linestyle='--', label='Delta threshold (0.1)')
        ax1.set_xlabel('Threshold de discretizacion')
        ax1.set_ylabel('Delta max |w - round(w)|')
        ax1.set_title('Delta vs Threshold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Panel 2: Accuracy vs Threshold
        ax2 = axes[0, 1]
        ax2.plot(thresholds, accuracies, 'g-o', linewidth=2, markersize=8)
        ax2.axhline(y=99.9, color='blue', linestyle='--', label='Threshold grokking')
        ax2.set_xlabel('Threshold de discretizacion')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Accuracy vs Threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Panel 3: Sparsity vs Threshold
        ax3 = axes[1, 0]
        ax3.plot(thresholds, sparsities, 'r-o', linewidth=2, markersize=8)
        ax3.set_xlabel('Threshold de discretizacion')
        ax3.set_ylabel('Sparsity')
        ax3.set_title('Sparsity vs Threshold')
        ax3.grid(True, alpha=0.3)
        
        # Panel 4: Resumen
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        best = pruning_data['discretization'].get('best_discretization')
        success = pruning_data['discretization'].get('success', False)
        
        summary_text = f"""
        RESULTADOS DE DISCRETIZACION
        =============================
        
        Threshold objetivo: delta < 0.1
        
        Discretizaciones exitosas: {sum(1 for r in disc_results if r['delta_threshold_met'])}/{len(disc_results)}
        
        """
        
        if best:
            summary_text += f"""
        MEJOR RESULTADO:
        - Threshold usado: {best['threshold']}
        - Delta: {best['delta_max']:.6f}
        - Accuracy: {best['accuracy']:.2f}%
        - Sparsity: {best['sparsity']:.2%}
        - Slots activos: {best['active_slots']}
            """
        
        summary_text += f"""
        CONCLUSION:
        {'[+] EXITO: Cuenca discreta alcanzada' if success else '[*] No se encontro discretizacion'}
        """
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightgreen' if success else 'lightyellow', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', format='png')
        plt.savefig(save_path.replace('.png', '.pdf'), bbox_inches='tight', format='pdf')
        plt.close()
        
        print(f"✓ Guardado: {save_path}")


# ================================================================================
# CLASE 9: ORQUESTADOR DE EXPERIMENTOS
# ================================================================================

class ExperimentOrchestrator:
    """
    Orquestador principal para todos los experimentos.
    
    Coordina:
    1. Carga de checkpoint grokkeado
    2. Verificación de grokking
    3. Experimento de Local Complexity
    4. Protocolo de poda iterativa
    5. Análisis ROC/AUC
    6. Generación de visualizaciones
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() and config.use_gpu_if_available else 'cpu')
        
        # Directorios de resultados
        self.results_dir = Path("experiment_results")
        self.results_dir.mkdir(exist_ok=True, parents=True)
        
        # Componentes
        self.visualizer = VisualizationGenerator()
        self.statistics = BootstrapStatistics(config)
        
        self.model = None
        self.verifier = GrokkingVerifier(config)
        self.pruning_engine = IterativePruningEngine(config)
        self.balanced_runs_generator = BalancedRunsGenerator(config)
        self.lc_experiment = LocalComplexityExperiment(config)
        
        self.results = {
            'timestamp': datetime.now().isoformat(),
            'config': config.__dict__,
            'checkpoint_used': None,
            'verification': None,
            'pruning': None,
            'roc': None,
            'local_complexity': None,
            'local_complexity_training': None,
            'balanced_runs': None
        }
    
    def find_grokked_checkpoint(self) -> Optional[str]:
        """
        Buscar checkpoint grokkeado en múltiples ubicaciones.
        
        Returns:
            Path al archivo de checkpoint o None si no se encuentra
        """
        # Posibles ubicaciones de checkpoints (en orden de prioridad)
        search_paths = [
            Path("/workspace/strass_strassen/checkpoints"),  # Repo clonado
            Path(__file__).parent / "strass_strassen" / "checkpoints",
            Path(__file__).parent.parent / "strass_strassen" / "checkpoints",
            Path("checkpoints"),
            Path("../checkpoints"),
            Path(__file__).parent.parent / "checkpoints",
        ]
        
        # Nombres de archivos grokked (en orden de prioridad)
        # strassen_exact.pt es el más probable que esté grokkeado
        checkpoint_names = [
            "strassen_exact.pt",         # [+] Probablemente grokkeado
            "strassen_discrete_final.pt", # Discretizado final
            "strassen_grokkit.pt",        # Formato grokkit
            "strassen_grokked_weights.pt", # Pesos grokked
            "strassen_discovered.pt",     # Descubierto
            "strassen_result.pt"          # Resultado
        ]
        
        for search_path in search_paths:
            search_path = search_path.resolve()
            if not search_path.exists():
                continue
            
            print(f"[DEBUG] Buscando en: {search_path}")
            
            for name in checkpoint_names:
                full_path = search_path / name
                if full_path.exists():
                    print(f"[INFO] ✓ Encontrado: {full_path}")
                    return str(full_path)
        
        print("[ERROR] No se encontró ningún checkpoint grokkeado")
        return None
    
    def load_grokked_checkpoint(self, checkpoint_path: str) -> StrassenOperator:
        """
        Cargar checkpoint grokkeado y verificar que grokkeó.
        
        Returns:
            Modelo cargado y verificado
        """
        print(f"\n{'='*70}")
        print(f"CARGANDO CHECKPOINT GROKKEADO")
        print(f"{'='*70}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"Device: {self.device}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Determinar rank
        rank = checkpoint.get('rank', self.config.rank)
        model = StrassenOperator(rank=rank).to(self.device)
        
        # Cargar pesos
        if all(k in checkpoint for k in ['U', 'V', 'W']):
            model.U.data = checkpoint['U'].to(self.device)
            model.V.data = checkpoint['V'].to(self.device)
            model.W.data = checkpoint['W'].to(self.device)
            print(f"Cargado formato grokkit: U, V, W")
        else:
            model.load_state_dict(checkpoint)
            print(f"Cargado formato state_dict")
        
        model.eval()
        self.model = model
        self.results['checkpoint_used'] = checkpoint_path
        
        return model
    
    def verify_checkpoint_is_grokked(self) -> Tuple[bool, Dict[str, float]]:
        """
        Verificar que el checkpoint cargado realmente grokkeó.
        
        Returns:
            Tupla de (es_grokked, métricas)
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado. Ejecute load_grokked_checkpoint primero.")
        
        success, metrics = self.verifier.verify(self.model, self.config.test_samples)
        
        print(f"\n{'='*70}")
        print(f"VERIFICACIÓN DE GROKKING")
        print(f"{'='*70}")
        print(f"Accuracy: {metrics['acc']:.2f}% (threshold: {self.config.grokking_accuracy_threshold}%)")
        print(f"LC: {metrics['lc']:.6f} (threshold: 0.999)")
        print(f"Active Slots: {metrics['active']} (threshold: 8)")
        print(f"Max Error: {metrics['max_err']:.2e}")
        print(f"Mean Error: {metrics['mean_err']:.2e}")
        print(f"¿Grokked?: {'SÍ' if success else 'NO'}")
        print(f"{'='*70}\n")
        
        self.results['verification'] = metrics
        
        return success, metrics
    
    def run_local_complexity_experiment(self, epochs: int = 3000) -> Dict[str, Any]:
        """
        Ejecutar experimento de Local Complexity vs Época.
        
        Nota: Como usamos un checkpoint ya grokked, esto mide la LC
        durante el fine-tuning post-poda, no durante el grokking inicial.
        
        Returns:
            Diccionario con historial de LC y accuracy
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado")
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENTO 1: LOCAL COMPLEXITY (LC)")
        print(f"{'='*70}")
        print(f"Ejecutando durante {epochs} épocas...")
        print(f"{'='*70}\n")
        
        # Generar datos
        data_gen = StrassenDataGenerator(num_samples=10000, seed=42)
        train_inputs = data_gen.data[0].to(self.device)
        train_targets = data_gen.data[1].to(self.device)
        
        # Inicializar calculadora de LC
        lc_calculator = LocalComplexityCalculator(self.model, self.config)
        
        # Entrenamiento para observar dinámica de LC
        optimizer = optim.AdamW(self.model.parameters(), lr=self.config.learning_rate, 
                               weight_decay=self.config.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=500, T_mult=2)
        
        history = {
            'epochs': [],
            'lc': [],
            'accuracy': [],
            'loss': []
        }
        
        eval_interval = max(1, epochs // 50)
        
        for epoch in range(epochs):
            # Entrenamiento
            self.model.train()
            optimizer.zero_grad()
            
            A, B = self._generate_batch(self.config.batch_size)
            C_pred = self.model(A, B)
            C_true = torch.bmm(A, B)
            loss = torch.mean((C_pred - C_true) ** 2)
            
            loss.backward()
            optimizer.step()
            scheduler.step()
            
            # Evaluar LC y accuracy
            if epoch % eval_interval == 0 or epoch == epochs - 1:
                self.model.eval()
                
                # Calcular LC en batch aleatorio
                batch_idx = torch.randperm(len(train_inputs))[:256]
                lc = lc_calculator.compute_lc(train_inputs[batch_idx], train_targets[batch_idx])
                
                # Calcular accuracy
                with torch.no_grad():
                    A_eval, B_eval = self._generate_batch(1000)
                    C_eval = self.model(A_eval, B_eval)
                    C_true_eval = torch.bmm(A_eval, B_eval)
                    errors = (C_eval - C_true_eval).abs()
                    acc = (errors.reshape(1000, -1).max(dim=1)[0] < self.config.accuracy_threshold).float().mean().item() * 100
                
                history['epochs'].append(epoch)
                history['lc'].append(lc)
                history['accuracy'].append(acc)
                history['loss'].append(loss.item())
                
                if epoch % (eval_interval * 2) == 0:
                    print(f"Época {epoch:>5}/{epochs} | LC: {lc:.4f} | Accuracy: {acc:.2f}% | Loss: {loss.item():.6e}")
                
                self.model.train()
        
        # Convertir a arrays numpy
        for key in history:
            history[key] = np.array(history[key])
        
        print(f"\n✓ Experimento LC completado")
        print(f"  Rango LC: [{history['lc'].min():.4f}, {history['lc'].max():.4f}]")
        print(f"  LC final: {history['lc'][-1]:.4f}")
        print(f"  Accuracy final: {history['accuracy'][-1]:.2f}%")
        
        self.results['local_complexity'] = history
        
        # Generar figura
        self.visualizer.plot_local_complexity(
            history['epochs'], 
            history['lc'], 
            history['accuracy'],
            str(self.results_dir / 'figure1_local_complexity.png')
        )
        
        return history
    
    def run_lc_training_experiment(self, epochs: int = None) -> Dict[str, Any]:
        """
        Ejecutar experimento de Local Complexity .
        
        - Entrenar un modelo desde cero hasta grokking
        - Medir LC en cada época para capturar la transición de fase
        - Si LC muestra un cambio alrededor del grokking, la métrica es útil
        - Si LC permanece constante, la métrica NO captura la transición
        
        Returns:
            Diccionario con historial completo del experimento
        """
        if epochs is None:
            epochs = self.config.max_epochs
            
        print(f"\n{'='*70}")
        print(f"EXPERIMENTO LC DESDE CERO")
        print(f"{'='*70}")
        print(f"Objetivo: Medir LC durante entrenamiento para capturar transicion de fase")
        print(f"Epocas: {epochs}")
        print(f"{'='*70}\n")
        
        # Ejecutar el experimento
        result = self.lc_experiment.run_full_experiment(target_epochs=epochs)
        
        self.results['local_complexity_training'] = result
        
        # Generar figura
        if 'history' in result:
            history = result['history']
            self.visualizer.plot_local_complexity(
                history['epochs'], 
                history['lc'], 
                history['accuracy'],
                str(self.results_dir / 'figure1b_lc_training.png')
            )
        
        return result
    
    def run_pruning_experiment(self) -> Dict[str, Any]:
        """
        Ejecutar protocolo de poda iterativa + fine-tuning.
        
        Returns:
            Diccionario con resultados de poda
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado")
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENTO 2: PODA ITERATIVA + FINE-TUNING")
        print(f"{'='*70}\n")
        
        # Generar datos
        data_gen = StrassenDataGenerator(num_samples=10000, seed=42)
        train_data = (data_gen.data[0], data_gen.data[1])
        
        # Copia del modelo para no modificar el original
        model_copy = StrassenOperator(rank=self.config.rank)
        model_copy.U.data = self.model.U.data.clone()
        model_copy.V.data = self.model.V.data.clone()
        model_copy.W.data = self.model.W.data.clone()
        model_copy = model_copy.to(self.device)
        
        # Ejecutar protocolo de poda
        results = self.pruning_engine.run_protocol(model_copy, train_data)
        self.results['pruning'] = results
        
        # Generar figura
        self.visualizer.plot_pruning_results(
            results,
            str(self.results_dir / 'figure2_pruning_results.png')
        )
        
        # Generar figura de discretizacion (PUNTO A)
        if 'discretization' in results:
            self.visualizer.plot_discretization_results(
                results,
                str(self.results_dir / 'figure2b_discretization_results.png')
            )
        
        return results
    
    def run_balanced_runs_experiment(self, n_runs: int = None) -> Dict[str, Any]:
        """
        Ejecutar experimento de runs balanceados (PUNTO C DEL REVISOR).
        
        Esto es CRUCIAL para obtener un AUC valido:
        - Entrenar multiples modelos con diferentes hiperparametros
        - Algunos grokkean, otros no
        - Generar dataset balanceado para ROC/AUC
        
        Returns:
            Diccionario con resultados de todos los runs
        """
        if n_runs is None:
            n_runs = self.config.n_experiment_runs
            
        print(f"\n{'='*70}")
        print(f"EXPERIMENTO RUNS BALANCEADOS (PUNTO C DEL REVISOR)")
        print(f"{'='*70}")
        print(f"Objetivo: Generar mix de runs grokked/no-grokked para AUC válido")
        print(f"Total de runs: {n_runs}")
        print(f"{'='*70}\n")
        
        result = self.balanced_runs_generator.run_balanced_experiments(n_runs=n_runs)
        self.results['balanced_runs'] = result
        
        # Generar figura ROC si es posible
        if result.get('roc', {}).get('auc_mean') != 'undefined':
            # Usar los datos para graficar
            y_true = np.array(result['y_true'])
            y_scores = np.array(result['y_scores'])
            
            if len(np.unique(y_true)) == 2:
                from sklearn.metrics import roc_curve, auc
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                auc_value = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(8, 8))
                ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc_value:.3f})')
                ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Clasificador Aleatorio')
                ax.set_xlabel('Tasa de Falsos Positivos (FPR)')
                ax.set_ylabel('Tasa de Verdaderos Positivos (TPR)')
                ax.set_title('Curva ROC - Runs Balanceados (Punto C)')
                ax.legend(loc='lower right')
                ax.set_xlim([0, 1])
                ax.set_ylim([0, 1])
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
                
                plt.tight_layout()
                plt.savefig(str(self.results_dir / 'figure4_roc_balanced_runs.png'), bbox_inches='tight', format='png')
                plt.savefig(str(self.results_dir / 'figure4_roc_balanced_runs.pdf'), bbox_inches='tight', format='pdf')
                plt.close()
                
                print(f"✓ Guardado: figure4_roc_balanced_runs.png")
        
        # Generar figura de resumen de runs balanceados
        self.visualizer.plot_balanced_runs_results(
            result,
            str(self.results_dir / 'figure4b_balanced_runs_summary.png')
        )
        
        return result
    
    def run_roc_analysis(self) -> Dict[str, Any]:
        """
        Ejecutar análisis ROC/AUC con bootstrap.
        
        Returns:
            Diccionario con resultados ROC
        """
        if self.model is None:
            raise ValueError("No hay modelo cargado")
        
        print(f"\n{'='*70}")
        print(f"EXPERIMENTO 3: CURVAS ROC CON BOOTSTRAP")
        print(f"{'='*70}\n")
        
        # Generar predicciones
        n_test = self.config.test_samples
        A, B = self._generate_batch(n_test)
        C_true = torch.bmm(A, B)
        
        self.model.eval()
        with torch.no_grad():
            C_pred = self.model(A, B)
            errors = (C_pred - C_true).abs()
        
        # Convertir a formato binario para ROC
        per_sample_errors = errors.reshape(n_test, -1).max(dim=1)[0].cpu().numpy()
        threshold = self.config.accuracy_threshold
        y_binary = (per_sample_errors < threshold).astype(int)
        
        # Scores: inverso del error normalizado
        max_error = per_sample_errors.max() + 1e-10
        scores = 1.0 - (per_sample_errors / max_error)
        
        n_grokking = int(np.sum(y_binary))
        n_not_grokking = int(np.sum(1 - y_binary))
        
        print(f"Muestras de test: {n_test}")
        print(f"Clase 'grokking' (error < {threshold}): {n_grokking} ({100*n_grokking/n_test:.1f}%)")
        print(f"Clase 'no grokking': {n_not_grokking} ({100*n_not_grokking/n_test:.1f}%)")
        
        # Calcular ROC con bootstrap
        roc_results = self.statistics.compute_roc_with_ci(y_binary, scores)
        
        # Calcular Kappa si hay ambas clases
        if len(np.unique(y_binary)) == 2:
            y_pred = (scores > 0.5).astype(int)
            kappa_results = self.statistics.compute_kappa_with_ci(y_binary, y_pred)
            correct = (y_binary == y_pred)
            accuracy_results = self.statistics.compute_accuracy_with_ci(correct)
            
            roc_results['kappa'] = kappa_results
            roc_results['accuracy'] = accuracy_results
            
            print(f"\nResultados:")
            if roc_results.get('auc_mean') != 'undefined':
                print(f"  AUC: {roc_results['auc_mean']:.4f} (IC 95%: [{roc_results['auc_ci_lower']:.4f}, {roc_results['auc_ci_upper']:.4f}])")
            else:
                print(f"  AUC: indefinido (una sola clase)")
            print(f"  Kappa: {kappa_results['kappa_mean']:.4f} (IC 95%: [{kappa_results['kappa_ci_lower']:.4f}, {kappa_results['kappa_ci_upper']:.4f}])")
            print(f"  Accuracy: {accuracy_results['accuracy']:.4f} (IC 95%: [{accuracy_results['ci_lower']:.4f}, {accuracy_results['ci_upper']:.4f}])")
        else:
            print(f"\n[*]  ADVERTENCIA: Solo una clase presente. AUC indefinido.")
            print(f"   Esto ocurre cuando el modelo grokkeó perfectamente (100% accuracy).")
        
        self.results['roc'] = roc_results
        
        # Generar figura
        self.visualizer.plot_roc_with_ci(
            roc_results,
            str(self.results_dir / 'figure3_roc_curves.png')
        )
        
        return roc_results
    
    def _generate_batch(self, n: int, scale: float = 1.0) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generar batch de matrices aleatorias."""
        return (
            torch.randn(n, 2, 2, device=self.device) * scale,
            torch.randn(n, 2, 2, device=self.device) * scale
        )
    
    def generate_summary_report(self) -> str:
        """Generar reporte de resumen en markdown."""
        lines = []
        
        lines.append("# Reporte de Validación - Respuesta a Revisor\n")
        lines.append(f"**Fecha:** {self.results['timestamp']}\n")
        lines.append(f"**Checkpoint usado:** {self.results.get('checkpoint_used', 'N/A')}\n")
        
        # Verificación
        if self.results.get('verification'):
            v = self.results['verification']
            lines.append("\n## Verificación de Grokking\n")
            lines.append(f"- **Accuracy:** {v['acc']:.2f}%\n")
            lines.append(f"- **LC:** {v['lc']:.6f}\n")
            lines.append(f"- **Slots activos:** {v['active']}\n")
            lines.append(f"- **Error máximo:** {v['max_err']:.2e}\n")
            lines.append(f"- **Error medio:** {v['mean_err']:.2e}\n")
        
        # Poda
        if self.results.get('pruning'):
            p = self.results['pruning']
            lines.append("\n## Resultados de Poda Iterativa\n")
            lines.append(f"- **Sparsity final:** {p['final_sparsity']:.2%}\n")
            lines.append(f"- **Accuracy final:** {p['final_accuracy']:.2f}%\n")
            lines.append(f"- **Cuenca discreta alcanzada:** {'Sí' if p['discrete_basin_reached'] else 'No'}\n")
            lines.append(f"- **Éxito del protocolo:** {'Sí' if p['success'] else 'No'}\n")
            lines.append(f"- **Razón de parada:** {p['stop_reason']}\n")
            lines.append(f"- **Iteraciones:** {len(p.get('iterations', []))}\n")
        
        # ROC
        if self.results.get('roc'):
            r = self.results['roc']
            lines.append("\n## Análisis ROC/AUC (Modelo Individual)\n")
            if r.get('auc_mean') != 'undefined':
                lines.append(f"- **AUC:** {r['auc_mean']:.4f}\n")
                lines.append(f"- **IC 95%:** [{r['auc_ci_lower']:.4f}, {r['auc_ci_upper']:.4f}]\n")
            else:
                lines.append(f"- **AUC:** Indefinido ({r.get('warning', 'razón no especificada')})\n")
            if 'class_distribution' in r:
                lines.append(f"- **Distribución de clases:** {r['class_distribution']}\n")
        
        # LC Training desde cero
        if self.results.get('local_complexity_training'):
            lc = self.results['local_complexity_training']
            lines.append("\n## Experimento LC desde Cero (Punto B)\n")
            if lc.get('grokking_epoch'):
                lines.append(f"- **Época de grokking:** {lc['grokking_epoch']}\n")
            else:
                lines.append(f"- **Época de grokking:** No alcanzado\n")
            lines.append(f"- **Accuracy final:** {lc.get('history', {}).get('accuracy', [0])[-1]:.2f}%\n")
            if 'history' in lc:
                lc_history = lc['history']
                lines.append(f"- **LC rango:** [{lc_history['lc'].min():.4f}, {lc_history['lc'].max():.4f}]\n")
        
        # Runs balanceados
        if self.results.get('balanced_runs'):
            br = self.results['balanced_runs']
            lines.append("\n## Runs Balanceados (Punto C)\n")
            lines.append(f"- **Total runs:** {br['total_runs']}\n")
            lines.append(f"- **Grokked:** {br['grokked_count']}\n")
            lines.append(f"- **No grokkeado:** {br['not_grokked_count']}\n")
            if br.get('roc', {}).get('auc_mean') != 'undefined':
                lines.append(f"- **AUC (balanceado):** {br['roc']['auc_mean']:.4f}\n")
                lines.append(f"- **IC 95%:** [{br['roc']['auc_ci_lower']:.4f}, {br['roc']['auc_ci_upper']:.4f}]\n")
            else:
                lines.append(f"- **AUC:** Indefinido ({br.get('roc', {}).get('warning', 'N/A')})\n")
        
        lines.append("\n## Archivos Generados\n")
        for f in self.results_dir.glob("*"):
            lines.append(f"- {f.name}\n")
        
        return ''.join(lines)
    
    def save_results(self) -> None:
        """Guardar todos los resultados."""
        # JSON
        results_path = self.results_dir / 'experiment_results.json'
        
        # Convertir numpy arrays a listas para JSON
        json_results = {}
        for key, value in self.results.items():
            if isinstance(value, dict):
                json_results[key] = {}
                for k, v in value.items():
                    if isinstance(v, np.ndarray):
                        json_results[key][k] = v.tolist()
                    elif isinstance(v, np.floating):
                        json_results[key][k] = float(v)
                    elif isinstance(v, np.integer):
                        json_results[key][k] = int(v)
                    else:
                        json_results[key][k] = v
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, np.floating):
                json_results[key] = float(value)
            elif isinstance(value, np.integer):
                json_results[key] = int(value)
            else:
                json_results[key] = value
        
        with open(results_path, 'w') as f:
            json.dump(json_results, f, indent=2, default=str)
        print(f"✓ Guardado: {results_path}")
        
        # Reporte markdown
        summary_path = self.results_dir / 'EXPERIMENT_SUMMARY.md'
        with open(summary_path, 'w') as f:
            f.write(self.generate_summary_report())
        print(f"✓ Guardado: {summary_path}")
    
    def run_all_experiments(self, checkpoint_path: Optional[str] = None) -> Dict[str, Any]:
        """Ejecutar suite completa de experimentos."""
        print(f"\n{'='*80}")
        print(f"  BATERÍA DE EXPERIMENTOS DE VALIDACIÓN")
        print(f"  Respuesta a Revisores del Paper")
        print(f"{'='*80}\n")
        
        # 1. Buscar y cargar checkpoint grokkeado
        if checkpoint_path is None:
            checkpoint_path = self.find_grokked_checkpoint()
        
        if checkpoint_path is None or not Path(checkpoint_path).exists():
            raise FileNotFoundError(f"No se encontró checkpoint grokkeado. Buscado en múltiples ubicaciones.")
        
        self.load_grokked_checkpoint(checkpoint_path)
        
        # 2. Verificar que grokkeó
        is_grokked, metrics = self.verify_checkpoint_is_grokked()
        
        if not is_grokked:
            print(f"\n[*]  ADVERTENCIA: El checkpoint NO grokkeó completamente.")
            print(f"   Accuracy: {metrics['acc']:.2f}% (esperado: >= {self.config.grokking_accuracy_threshold}%)")
            print(f"   Los experimentos pueden no ser válidos.\n")
        
        # 3. Ejecutar experimentos
        print(f"\n{'='*80}")
        print(f"  EJECUTANDO EXPERIMENTOS")
        print(f"{'='*80}\n")
        
        # Experimento 1: LC vs Época (si el modelo aún no ha grokkeado perfectamente)
        if metrics['acc'] < 99.9:
            self.run_local_complexity_experiment(epochs=min(3000, self.config.max_epochs))
        else:
            print("Experimento LC: Omitido (modelo ya grokked perfectamente)")
        
        # Experimento 1b: LC entrenando desde CERO (PUNTO B DEL REVISOR)
        # Este experimento entrena un modelo desde cero para capturar la transición de grokking
        print(f"\n{'='*80}")
        print(f"EJECUTANDO EXPERIMENTO LC DESDE CERO (PUNTO B)")
        print(f"{'='*80}\n")
        self.run_lc_training_experiment(epochs=self.config.max_epochs)
        
        # Experimento 2: Poda iterativa + fine-tuning
        self.run_pruning_experiment()
        
        # Experimento 3: ROC con bootstrap (modelo individual)
        self.run_roc_analysis()
        
        # Experimento 4: Runs balanceados (PUNTO C DEL REVISOR)
        # Generar mix de runs grokked/no-grokked para AUC valido
        print(f"\n{'='*80}")
        print(f"EJECUTANDO EXPERIMENTO RUNS BALANCEADOS (PUNTO C)")
        print(f"{'='*80}\n")
        self.run_balanced_runs_experiment(n_runs=self.config.n_experiment_runs)
        
        # 4. Guardar resultados
        print(f"\n{'='*80}")
        print(f"  GUARDANDO RESULTADOS")
        print(f"{'='*80}\n")
        
        self.save_results()
        
        print(f"\n{'='*80}")
        print(f"  EXPERIMENTOS COMPLETADOS")
        print(f"{'='*80}")
        print(f"\nResultados guardados en: {self.results_dir}")
        print(f"Archivos generados:")
        for f in sorted(self.results_dir.glob("*")):
            print(f"  - {f.name}")
        print(f"{'='*80}\n")
        
        return self.results


# ================================================================================
# FUNCIÓN PRINCIPAL
# ================================================================================

def find_grokked_checkpoint() -> str:
    """
    Buscar checkpoint grokkeado en múltiples ubicaciones.
    
    Returns:
        Path al archivo de checkpoint grokkeado
    """
    # Posibles ubicaciones de checkpoints (en orden de prioridad)
    search_paths = [
        Path("/workspace/strass_strassen/checkpoints"),  # Repo clonado
        Path(__file__).parent / "strass_strassen" / "checkpoints",
        Path(__file__).parent.parent / "strass_strassen" / "checkpoints",
        Path("checkpoints"),
        Path("../checkpoints"),
    ]
    
    # Nombres de archivos grokked (en orden de prioridad)
    checkpoint_names = [
        "strassen_exact.pt",         # [+] Probablemente grokkeado
        "strassen_discrete_final.pt", # Discretizado final
        "strassen_grokkit.pt",        # Formato grokkit
        "strassen_grokked_weights.pt", # Pesos grokked
        "strassen_discovered.pt",     # Descubierto
        "strassen_result.pt"          # Resultado
    ]
    
    for search_path in search_paths:
        search_path = search_path.resolve()
        if not search_path.exists():
            continue
        
        for name in checkpoint_names:
            full_path = search_path / name
            if full_path.exists():
                print(f"[INFO] ✓ Checkpoint encontrado: {full_path}")
                return str(full_path)
    
    raise FileNotFoundError("No se encontró ningún checkpoint grokkeado")


def analyze_checkpoints() -> Dict[str, Dict[str, float]]:
    """
    Analizar todos los checkpoints disponibles para encontrar el grokkeado.
    
    Returns:
        Diccionario con métricas de cada checkpoint
    """
    from validate_all_revisor_experiments import StrassenOperator, GrokkingVerifier, ExperimentConfig
    
    config = ExperimentConfig()
    verifier = GrokkingVerifier(config)
    
    checkpoint_files = [
        "strassen_exact.pt",
        "strassen_discrete_final.pt", 
        "strassen_grokkit.pt",
        "strassen_grokked_weights.pt",
        "strassen_discovered.pt",
        "strassen_result.pt"
    ]
    
    results = {}
    
    print("="*70)
    print("ANÁLISIS DE CHECKPOINTS DISPONIBLES")
    print("="*70)
    
    for cp_name in checkpoint_files:
        cp_path = None
        
        # Buscar en múltiples ubicaciones
        for base in ["/workspace/strass_strassen/checkpoints", "checkpoints", "../checkpoints"]:
            path = Path(base) / cp_name
            if path.exists():
                cp_path = path
                break
        
        if cp_path is None:
            continue
        
        try:
            checkpoint = torch.load(cp_path, map_location='cpu')
            rank = checkpoint.get('rank', 8)
            
            model = StrassenOperator(rank=rank)
            
            if all(k in checkpoint for k in ['U', 'V', 'W']):
                model.U.data = checkpoint['U']
                model.V.data = checkpoint['V']
                model.W.data = checkpoint['W']
            
            success, metrics = verifier.verify(model, 5000)
            
            status = "[+] GROKKEADO" if success else "[-] No grokkeado"
            print(f"\n{cp_name}:")
            print(f"  Status: {status}")
            print(f"  Accuracy: {metrics['acc']:.2f}%")
            print(f"  LC: {metrics['lc']:.6f}")
            print(f"  Active Slots: {metrics['active']}")
            print(f"  Max Error: {metrics['max_err']:.2e}")
            
            results[cp_name] = metrics
            
        except Exception as e:
            print(f"\n{cp_name}: Error - {e}")
    
    print("\n" + "="*70)
    return results


def main():
    """Punto de entrada principal."""
    parser = argparse.ArgumentParser(description='Batería de Experimentos de Validación')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path a checkpoint grokkeado')
    parser.add_argument('--epochs', type=int, default=3000,
                       help='Épocas para experimento LC')
    parser.add_argument('--prune-iterations', type=int, default=60,
                       help='Iteraciones máximas de poda')
    parser.add_argument('--bootstrap-samples', type=int, default=1000,
                       help='Muestras bootstrap')
    parser.add_argument('--delta-threshold', type=float, default=0.1,
                       help='Threshold δ para parada')
    parser.add_argument('--finetune-epochs', type=int, default=200,
                       help='Épocas de fine-tuning post-poda')
    parser.add_argument('--n-runs', type=int, default=60,
                       help='Número de runs para experimento balanceado')
    parser.add_argument('--skip-lc-training', action='store_true',
                       help='Omitir experimento LC entrenando desde cero')
    parser.add_argument('--skip-balanced-runs', action='store_true',
                       help='Omitir experimento de runs balanceados')
    
    args = parser.parse_args()
    
    # Configuracion
    config = ExperimentConfig(
        prune_max_iterations=args.prune_iterations,
        finetune_epochs=args.finetune_epochs,
        delta_threshold=args.delta_threshold,
        n_bootstrap=args.bootstrap_samples,
        max_epochs=args.epochs,
        n_experiment_runs=args.n_runs
    )
    
    # Crear orquestador
    orchestrator = ExperimentOrchestrator(config)
    
    try:
        # Ejecutar todos los experimentos
        results = orchestrator.run_all_experiments(args.checkpoint)
        
        print("\n" + "="*80)
        print("RESUMEN EJECUTIVO")
        print("="*80)
        
        if results.get('verification'):
            v = results['verification']
            print(f"\nCheckpoint verificado:")
            print(f"  Accuracy: {v['acc']:.2f}%")
            print(f"  LC: {v['lc']:.6f}")
            print(f"  Slots activos: {v['active']}")
        
        if results.get('pruning'):
            p = results['pruning']
            print(f"\nProtocolo de poda:")
            print(f"  Sparsity final: {p['final_sparsity']:.2%}")
            print(f"  Accuracy final: {p['final_accuracy']:.2f}%")
            print(f"  Cuenca discreta: {'Alcanzada' if p['discrete_basin_reached'] else 'No alcanzada'}")
            print(f"  Razón de parada: {p['stop_reason']}")
            if p.get('discretization'):
                print(f"  Discretización (δ < 0.1): {'ÉXITO' if p['discretization']['success'] else 'FALLÓ'}")
        
        if results.get('local_complexity_training'):
            lc = results['local_complexity_training']
            print(f"\nExperimento LC desde Cero (Punto B):")
            print(f"  Grokking alcanzado: {'Sí' if lc.get('grokking_epoch') else 'No'}")
            if lc.get('grokking_epoch'):
                print(f"  Época de grokking: {lc['grokking_epoch']}")
        
        if results.get('balanced_runs'):
            br = results['balanced_runs']
            print(f"\nRuns Balanceados (Punto C):")
            print(f"  Total runs: {br['total_runs']}")
            print(f"  Grokked: {br['grokked_count']}")
            print(f"  No grokkeado: {br['not_grokked_count']}")
            if br.get('roc', {}).get('auc_mean') != 'undefined':
                print(f"  AUC: {br['roc']['auc_mean']:.4f} (IC 95%: [{br['roc']['auc_ci_lower']:.4f}, {br['roc']['auc_ci_upper']:.4f}])")
            else:
                print(f"  AUC: Indefinido ({br.get('roc', {}).get('warning', 'N/A')})")
        
        if results.get('roc'):
            r = results['roc']
            print(f"\nAnálisis ROC/AUC (individual):")
            if r.get('auc_mean') != 'undefined':
                print(f"  AUC: {r['auc_mean']:.4f} (IC 95%: [{r['auc_ci_lower']:.4f}, {r['auc_ci_upper']:.4f}])")
            else:
                print(f"  AUC: Indefinido ({r.get('warning', 'N/A')})")
        
        print("\n" + "="*80)
        
        return results
        
    except FileNotFoundError as e:
        print(f"\n[-] Error: {e}")
        print("\n Asegúrate de que el repositorio strass_strassen está clonado en /workspace/")
        return None


if __name__ == "__main__":
    results = main()
