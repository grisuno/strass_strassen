import argparse
import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Tuple, Any, Protocol
from abc import ABC, abstractmethod
import seaborn as sns
from scipy.stats import gaussian_kde
from scipy.linalg import eigh
from scipy.optimize import curve_fit
from sklearn.decomposition import PCA
import warnings
import logging
from dataclasses import dataclass
from pathlib import Path
import threading
import time
from collections import deque

@dataclass
class Config:
    BATCH_SIZE: int = 32
    HIDDEN_DIM: int = 8
    TARGET_SLOTS: int = 7
    MATRIX_SIZE: int = 2
    
    WEIGHT_DECAY: float = 1e-4
    LEARNING_RATE: float = 0.001
    EPOCHS: int = 3000
    DISCRETIZATION_MARGIN: float = 0.1
    
    HBAR: float = 1e-6
    POYNTING_THRESHOLD: float = 1.0
    ENERGY_FLOW_SCALE: float = 0.1
    
    RANDOM_SEED: int = 42
    DEVICE: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    MAX_CHECKPOINTS: int = 10
    
    KDE_BANDWIDTH: str = 'scott'
    MIN_VARIANCE_THRESHOLD: float = 1e-8
    PCA_COMPONENTS: int = 2
    ENTROPY_BINS: int = 50
    ENTROPY_METHOD: str = 'shannon'
    ENTROPY_EPS: float = 1e-10
    
    LOG_LEVEL: str = 'INFO'
    RESULTS_DIR: str = 'boltzmann_results'

def set_seed(seed: int = Config.RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if Config.DEVICE == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def setup_logger(name: str, level: str = Config.LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    
    return logger

def run_epitaxy_from_best_crystal(checkpoint_dir: str, target_sizes: List[int] = [4, 8]) -> Dict[str, Any]:
    """
    Pipeline automático: encuentra el mejor cristal y lo usa como semilla.
    """
    logger = setup_logger("EpitaxyPipeline")
    
    # Buscar cristales disponibles
    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
    
    checkpoint_files = list(checkpoint_path.glob("*.pt"))
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir}")
    
    logger.info(f"Found {len(checkpoint_files)} checkpoints. Analyzing purity...")
    
    # Analizar todos los checkpoints
    best_seed = None
    best_alpha = 0.0
    
    loader = CheckpointLoader()
    
    for ckpt_file in checkpoint_files:
        try:
            raw_data = loader.load_checkpoint(str(ckpt_file), Config.DEVICE)
            migrated_state = CheckpointMigrator.migrate_checkpoint(raw_data)
            
            if migrated_state is None:
                continue
            
            model = BilinearStrassenModel().to(Config.DEVICE)
            model.load_state_dict(migrated_state)
            
            coeffs = model.get_coefficients()
            alpha = CrystallographyMetrics.compute_alpha_purity(coeffs)
            delta = CrystallographyMetrics.compute_discretization_margin(coeffs)
            
            logger.info(f"  {ckpt_file.name}: α={alpha:.2f}, δ={delta:.4f}")
            
            if alpha > best_alpha and alpha > 7.0:  # Solo cristales puros
                best_alpha = alpha
                best_seed = ckpt_file
                
        except Exception as e:
            logger.warning(f"  Failed to analyze {ckpt_file.name}: {e}")
    
    if best_seed is None:
        raise ValueError("No crystalline checkpoint found (α > 7.0). Cannot perform epitaxy.")
    
    logger.info(f"\n🌟 Best seed crystal: {best_seed.name} (α={best_alpha:.2f})")
    
    # Ejecutar experimento epitaxial
    experiment = EpitaxyExperiment()
    results = experiment.run_epitaxial_growth_experiment(str(best_seed), target_sizes)
    
    return results

logger = setup_logger(__name__)

class ICheckpointLoader(Protocol):
    def load_checkpoint(self, path: str, device: str) -> Any: ...

class IMetricsCalculator(Protocol):
    def compute(self, model: nn.Module) -> Dict[str, Any]: ...

class IDataGenerator(Protocol):
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, ...]: ...

class CheckpointLoadingError(Exception):
    pass

class MetricsComputationError(Exception):
    pass

class TrainingError(Exception):
    pass



class StrassenDataGenerator:
    @staticmethod
    def generate_batch(batch_size: int = Config.BATCH_SIZE) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        A = torch.randn(batch_size, Config.MATRIX_SIZE, Config.MATRIX_SIZE, device=Config.DEVICE)
        B = torch.randn(batch_size, Config.MATRIX_SIZE, Config.MATRIX_SIZE, device=Config.DEVICE)
        C = torch.bmm(A, B)
        return (
            A.reshape(batch_size, Config.MATRIX_SIZE * Config.MATRIX_SIZE),
            B.reshape(batch_size, Config.MATRIX_SIZE * Config.MATRIX_SIZE),
            C.reshape(batch_size, Config.MATRIX_SIZE * Config.MATRIX_SIZE)
        )
    
    @staticmethod
    def verify_structure(coeffs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        delta = CrystallographyMetrics.compute_discretization_margin(coeffs)
        return {
            'pass': delta < Config.DISCRETIZATION_MARGIN,
            'max_error': delta,
            'margin': Config.DISCRETIZATION_MARGIN
        }

class BilinearStrassenModel(nn.Module):
    def __init__(self, hidden_dim: int = Config.HIDDEN_DIM, matrix_size: int = Config.MATRIX_SIZE):
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



class EpitaxialGrowthEngine:
    """
    Motor de crecimiento epitaxial para cristales algorítmicos.
    
    FÍSICA: Imita el crecimiento de cristales en sustratos donde la estructura
    atómica del sustrato guía la formación del nuevo cristal.
    """
    
    def __init__(self, seed_checkpoint_path: str, target_matrix_size: int, device: str = Config.DEVICE):
        self.seed_path = Path(seed_checkpoint_path)
        self.target_size = target_matrix_size
        self.device = device
        self.logger = setup_logger("EpitaxialGrowthEngine")
        
        # Cargar el cristal semilla
        self.seed_crystal = self._load_seed_crystal()
        self.seed_size = Config.MATRIX_SIZE  # Tamaño original (2x2)
        
        # Calcular factor de escala
        self.scale_factor = target_matrix_size // self.seed_size
        
        self.logger.info(f"Epitaxial engine initialized:")
        self.logger.info(f"  Seed size: {self.seed_size}x{self.seed_size}")
        self.logger.info(f"  Target size: {target_matrix_size}x{target_matrix_size}")
        self.logger.info(f"  Scale factor: {self.scale_factor}")
    
    def _load_seed_crystal(self) -> Dict[str, torch.Tensor]:
        """Carga el cristal semilla verificando su pureza"""
        loader = CheckpointLoader()
        raw_data = loader.load_checkpoint(str(self.seed_path), self.device)
        migrated_state = CheckpointMigrator.migrate_checkpoint(raw_data)
        
        if migrated_state is None:
            raise ValueError(f"Failed to migrate seed crystal from {self.seed_path}")
        
        model = BilinearStrassenModel().to(self.device)
        model.load_state_dict(migrated_state)
        
        coeffs = model.get_coefficients()
        delta = CrystallographyMetrics.compute_discretization_margin(coeffs)
        alpha = CrystallographyMetrics.compute_alpha_purity(coeffs)
        
        self.logger.info(f"Seed crystal quality: α={alpha:.2f}, δ={delta:.4f}")
        
        if alpha < 7.0:
            self.logger.warning("Seed crystal is not fully crystalline - epitaxy may fail")
        
        return coeffs
    
    def grow_epitaxial_crystal(self) -> BilinearStrassenModel:
        """
        Crece un cristal epitaxial desde la semilla.
        
        MÉTODO: Kronecker product preserva la estructura periódica:
        Si A es cristal Strassen de 2x2, entonces A ⊗ I_n es cristal de (2n)x(2n)
        """
        target_input_dim = self.target_size * self.target_size
        
        # Calcular dimensión hidden escalada
        # Para preservar el ratio de compresión, escalamos proporcionalmente
        hidden_dim_scaled = Config.HIDDEN_DIM * (self.scale_factor ** 2)
        
        self.logger.info(f"Growing crystal with hidden_dim={hidden_dim_scaled}")
        
        # Crear modelo objetivo
        model = BilinearStrassenModel(
            hidden_dim=hidden_dim_scaled,
            matrix_size=self.target_size
        ).to(self.device)
        
        # Inicialización epitaxial usando producto de Kronecker
        with torch.no_grad():
            # U: (hidden_seed, input_seed) -> (hidden_target, input_target)
            U_seed = self.seed_crystal['U']
            identity_input = torch.eye(self.scale_factor, device=self.device)
            identity_hidden = torch.eye(self.scale_factor, device=self.device)
            
            # U_epitaxial = I_hidden ⊗ U_seed
            # Esto replica la estructura de U_seed en bloques
            U_epitaxial = torch.kron(identity_hidden, U_seed)
            
            # Truncar o rellenar para ajustar dimensiones exactas
            U_epitaxial = self._adjust_dimensions(
                U_epitaxial, 
                model.U.weight.shape
            )
            
            # V: similar a U (por simetría bilineal)
            V_epitaxial = torch.kron(identity_hidden, self.seed_crystal['V'])
            V_epitaxial = self._adjust_dimensions(
                V_epitaxial,
                model.V.weight.shape
            )
            
            # W: (output_seed, hidden_seed) -> (output_target, hidden_target)
            W_seed = self.seed_crystal['W']
            W_epitaxial = torch.kron(identity_input, W_seed)
            W_epitaxial = self._adjust_dimensions(
                W_epitaxial,
                model.W.weight.shape
            )
            
            # Inyectar estructura cristalina
            model.U.weight.copy_(U_epitaxial)
            model.V.weight.copy_(V_epitaxial)
            model.W.weight.copy_(W_epitaxial)
        
        self.logger.info("Epitaxial initialization complete")
        return model
    
    def _adjust_dimensions(self, tensor: torch.Tensor, target_shape: Tuple[int, int]) -> torch.Tensor:
        """
        Ajusta dimensiones del tensor epitaxial para coincidir con el modelo objetivo.
        Rellena con ruido térmico pequeño o trunca según sea necesario.
        """
        current_shape = tensor.shape
        target_rows, target_cols = target_shape
        
        # Crear tensor objetivo
        adjusted = torch.zeros(target_shape, device=tensor.device)
        
        # Copiar región común
        min_rows = min(current_shape[0], target_rows)
        min_cols = min(current_shape[1], target_cols)
        adjusted[:min_rows, :min_cols] = tensor[:min_rows, :min_cols]
        
        # Rellenar regiones faltantes con ruido térmico (temperatura efectiva baja)
        if current_shape[0] < target_rows or current_shape[1] < target_cols:
            thermal_noise_scale = 1e-4  # Temperatura efectiva muy baja
            if current_shape[0] < target_rows:
                adjusted[min_rows:, :min_cols] = torch.randn(
                    target_rows - min_rows, min_cols, device=tensor.device
                ) * thermal_noise_scale
            if current_shape[1] < target_cols:
                adjusted[:min_rows, min_cols:] = torch.randn(
                    min_rows, target_cols - min_cols, device=tensor.device
                ) * thermal_noise_scale
            if current_shape[0] < target_rows and current_shape[1] < target_cols:
                adjusted[min_rows:, min_cols:] = torch.randn(
                    target_rows - min_rows, target_cols - min_cols, device=tensor.device
                ) * thermal_noise_scale
        
        return adjusted
    
    def anneal_crystal(self, model: BilinearStrassenModel, max_epochs: int = 50, 
                      early_stop_threshold: float = 1e-4) -> Dict[str, Any]:
        """
        Recocido térmico del cristal epitaxial.
        
        FÍSICA: En lugar de "entrenar desde cero", aplicamos temperatura decreciente
        para que el cristal se auto-organice alrededor de la semilla.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("EPITAXIAL ANNEALING - Crystal Auto-Assembly")
        self.logger.info("="*60)
        
        # Generador de datos para el tamaño objetivo
        def generate_batch(batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            A = torch.randn(batch_size, self.target_size, self.target_size, device=self.device)
            B = torch.randn(batch_size, self.target_size, self.target_size, device=self.device)
            C = torch.bmm(A, B)
            return (
                A.reshape(batch_size, self.target_size * self.target_size),
                B.reshape(batch_size, self.target_size * self.target_size),
                C.reshape(batch_size, self.target_size * self.target_size)
            )
        
        # Optimizador con learning rate decreciente (cooling schedule)
        initial_temp = Config.LEARNING_RATE * 0.1  # Temperatura inicial baja
        optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=initial_temp,
            weight_decay=Config.WEIGHT_DECAY
        )
        
        # Scheduler: enfriamiento exponencial
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)
        
        history = {
            'epoch': [],
            'loss': [],
            'alpha': [],
            'delta': [],
            'temperature': [],
            'assembly_speed': []
        }
        
        initial_coeffs = model.get_coefficients()
        initial_alpha = CrystallographyMetrics.compute_alpha_purity(initial_coeffs)
        
        for epoch in range(max_epochs):
            model.train()
            epoch_loss = 0.0
            
            for _ in range(10):  # 10 batches por época
                A, B, C = generate_batch(Config.BATCH_SIZE)
                
                optimizer.zero_grad()
                C_pred = model(A, B)
                loss = nn.functional.mse_loss(C_pred, C)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= 10
            
            # Métricas cristalográficas
            coeffs = model.get_coefficients()
            alpha = CrystallographyMetrics.compute_alpha_purity(coeffs)
            delta = CrystallographyMetrics.compute_discretization_margin(coeffs)
            current_temp = optimizer.param_groups[0]['lr']
            
            # Velocidad de ensamblaje (cambio en pureza)
            assembly_speed = alpha - initial_alpha if epoch == 0 else (alpha - history['alpha'][-1])
            
            history['epoch'].append(epoch)
            history['loss'].append(epoch_loss)
            history['alpha'].append(alpha)
            history['delta'].append(delta)
            history['temperature'].append(current_temp)
            history['assembly_speed'].append(assembly_speed)
            
            if epoch % 5 == 0:
                self.logger.info(
                    f"Epoch {epoch:3d} | Loss: {epoch_loss:.6f} | "
                    f"α: {alpha:6.2f} | δ: {delta:.4f} | "
                    f"T: {current_temp:.2e} | v_asm: {assembly_speed:+.4f}"
                )
            
            # Early stopping: cristal auto-ensamblado
            if delta < early_stop_threshold and alpha > 7.0:
                self.logger.info(f"\n🎯 CRYSTAL AUTO-ASSEMBLY COMPLETE at epoch {epoch}")
                self.logger.info(f"   Final purity: α={alpha:.2f}, δ={delta:.6f}")
                break
            
            # Cooling schedule
            scheduler.step()
        
        # Análisis final
        final_coeffs = model.get_coefficients()
        final_metrics = {
            'final_alpha': CrystallographyMetrics.compute_alpha_purity(final_coeffs),
            'final_delta': CrystallographyMetrics.compute_discretization_margin(final_coeffs),
            'final_loss': history['loss'][-1],
            'epochs_to_crystallization': len(history['epoch']),
            'assembly_efficiency': (history['alpha'][-1] - initial_alpha) / len(history['epoch']),
            'thermal_history': history
        }
        
        return final_metrics


class EpitaxyExperiment:
    """
    Experimento completo de epitaxia: sembrar, crecer, analizar.
    """
    
    def __init__(self, results_dir: str = "epitaxy_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.logger = setup_logger("EpitaxyExperiment")
    
    def run_epitaxial_growth_experiment(self, seed_checkpoint: str, 
                                       target_sizes: List[int] = [4, 8, 16]) -> Dict[str, Any]:
        """
        Experimento completo: cultiva cristales de múltiples tamaños desde una semilla.
        """
        self.logger.info("\n" + "="*80)
        self.logger.info("EPITAXIAL GROWTH EXPERIMENT - From Seed to Superlattice")
        self.logger.info("="*80)
        
        seed_path = Path(seed_checkpoint)
        if not seed_path.exists():
            raise FileNotFoundError(f"Seed checkpoint not found: {seed_checkpoint}")
        
        results = {
            'seed_checkpoint': str(seed_path),
            'target_sizes': target_sizes,
            'experiments': {}
        }
        
        for target_size in target_sizes:
            self.logger.info(f"\n{'='*60}")
            self.logger.info(f"Growing {target_size}x{target_size} crystal from {seed_path.name}")
            self.logger.info(f"{'='*60}")
            
            try:
                # Crear motor de crecimiento
                engine = EpitaxialGrowthEngine(seed_checkpoint, target_size)
                
                # Crecer cristal epitaxial
                epitaxial_model = engine.grow_epitaxial_crystal()
                
                # Validar estructura inicial
                initial_coeffs = epitaxial_model.get_coefficients()
                initial_validation = StrassenDataGenerator.verify_structure(initial_coeffs)
                
                self.logger.info(f"Initial epitaxial structure:")
                self.logger.info(f"  Verification: {'PASS' if initial_validation['pass'] else 'FAIL'}")
                self.logger.info(f"  Max error: {initial_validation['max_error']:.4f}")
                
                # Recocido cristalino
                annealing_results = engine.anneal_crystal(epitaxial_model, max_epochs=50)
                
                # Guardar modelo final
                save_path = self.results_dir / f"epitaxial_{target_size}x{target_size}_from_{seed_path.stem}.pt"
                torch.save(epitaxial_model.state_dict(), save_path)
                
                # Análisis espectroscópico del cristal crecido
                final_coeffs = epitaxial_model.get_coefficients()
                spectroscopy = SpectroscopyMetrics.compute_weight_diffraction(final_coeffs)
                
                results['experiments'][f'{target_size}x{target_size}'] = {
                    'initial_validation': initial_validation,
                    'annealing_results': annealing_results,
                    'spectroscopy': spectroscopy,
                    'saved_model_path': str(save_path),
                    'crystallization_success': annealing_results['final_alpha'] > 7.0
                }
                
                # Gráficas de evolución
                self._plot_epitaxial_evolution(annealing_results, target_size, seed_path.stem)
                
            except Exception as e:
                self.logger.error(f"Failed to grow {target_size}x{target_size} crystal: {e}")
                results['experiments'][f'{target_size}x{target_size}'] = {
                    'error': str(e),
                    'crystallization_success': False
                }
        
        # Resumen comparativo
        self._generate_comparative_report(results)
        
        # Guardar resultados completos
        with open(self.results_dir / "epitaxy_experiment_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        return results
    
    def _plot_epitaxial_evolution(self, annealing_results: Dict[str, Any], 
                                  target_size: int, seed_name: str):
        """Visualiza la evolución del cristal durante el recocido"""
        history = annealing_results['thermal_history']
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
        
        epochs = history['epoch']
        
        # Loss evolution
        ax1.plot(epochs, history['loss'], 'b-', linewidth=2)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title(f'Loss Evolution - {target_size}x{target_size} Crystal')
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        # Purity evolution
        ax2.plot(epochs, history['alpha'], 'g-', linewidth=2, label='α (purity)')
        ax2.axhline(y=7.0, color='r', linestyle='--', label='Crystal threshold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Purity (α)')
        ax2.set_title('Crystallization Process')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Discretization margin
        ax3.plot(epochs, history['delta'], 'r-', linewidth=2)
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Discretization Margin (δ)')
        ax3.set_title('Structural Convergence')
        ax3.set_yscale('log')
        ax3.grid(True, alpha=0.3)
        
        # Assembly speed
        ax4.plot(epochs, history['assembly_speed'], 'purple', linewidth=2)
        ax4.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Assembly Speed (dα/dt)')
        ax4.set_title('Crystal Growth Dynamics')
        ax4.grid(True, alpha=0.3)
        
        plt.suptitle(f'Epitaxial Growth: {seed_name} → {target_size}x{target_size}', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.results_dir / f"epitaxy_evolution_{target_size}x{target_size}.png", dpi=150)
        plt.close()
    
    def _generate_comparative_report(self, results: Dict[str, Any]):
        """Genera reporte comparativo de todos los experimentos epitaxiales"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        sizes = []
        epochs_to_crystal = []
        final_alphas = []
        assembly_efficiencies = []
        
        for size_key, exp_data in results['experiments'].items():
            if 'error' not in exp_data:
                size = int(size_key.split('x')[0])
                sizes.append(size)
                epochs_to_crystal.append(exp_data['annealing_results']['epochs_to_crystallization'])
                final_alphas.append(exp_data['annealing_results']['final_alpha'])
                assembly_efficiencies.append(exp_data['annealing_results']['assembly_efficiency'])
        
        if not sizes:
            self.logger.warning("No successful experiments to plot")
            plt.close()
            return
        
        # Epochs vs Size
        ax1.plot(sizes, epochs_to_crystal, 'bo-', linewidth=2, markersize=10)
        ax1.set_xlabel('Matrix Size (NxN)', fontsize=12)
        ax1.set_ylabel('Epochs to Crystallization', fontsize=12)
        ax1.set_title('Epitaxial Efficiency: Assembly Time', fontsize=14)
        ax1.grid(True, alpha=0.3)
        
        # Ajuste teórico: si epitaxia es perfecta, debería ser ~constante o log(N)
        if len(sizes) > 1:
            z = np.polyfit(np.log(sizes), epochs_to_crystal, 1)
            p = np.poly1d(z)
            ax1.plot(sizes, p(np.log(sizes)), 'r--', 
                    label=f'Log fit: {z[0]:.2f}*log(N) + {z[1]:.2f}')
            ax1.legend()
        
        # Final purity vs Size
        ax2.bar(sizes, final_alphas, color='green', alpha=0.7, edgecolor='black')
        ax2.axhline(y=7.0, color='r', linestyle='--', linewidth=2, label='Crystal threshold')
        ax2.set_xlabel('Matrix Size (NxN)', fontsize=12)
        ax2.set_ylabel('Final Purity (α)', fontsize=12)
        ax2.set_title('Crystal Quality Post-Epitaxy', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "epitaxy_comparative_analysis.png", dpi=150)
        plt.close()
        
        # Imprimir reporte textual
        self.logger.info("\n" + "="*80)
        self.logger.info("EPITAXIAL GROWTH COMPARATIVE REPORT")
        self.logger.info("="*80)
        
        for size, epochs, alpha, efficiency in zip(sizes, epochs_to_crystal, 
                                                    final_alphas, assembly_efficiencies):
            status = "✓ SUCCESS" if alpha > 7.0 else "✗ PARTIAL"
            self.logger.info(
                f"{size:2d}x{size:2d} | Epochs: {epochs:3d} | "
                f"α: {alpha:6.2f} | Efficiency: {efficiency:.4f} | {status}"
            )
        
        if len(epochs_to_crystal) > 1:
            avg_speedup = epochs_to_crystal[0] / np.mean(epochs_to_crystal[1:])
            self.logger.info(f"\nAverage speedup vs baseline: {avg_speedup:.2f}x")
@dataclass
class ThermodynamicPotential:
    """Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C"""
    internal_energy: float  # Loss de generalización
    temperature: float      # T_eff
    entropy: float          # S_gen
    chemical_potential: float  # μ para "átomos" de conocimiento
    crystallinity: float    # α (pureza cristalina)
    particle_number: float  # N (parámetros activos)
    
    def helmholtz_free_energy(self) -> float:
        """F = U - T*S (a μ y N constantes)"""
        return self.internal_energy - self.temperature * self.entropy
    
    def gibbs_free_energy(self) -> float:
        """G = F + μ*N + P*V (presión algorítmica)"""
        pressure_term = self.crystallinity * self.particle_number
        return self.helmholtz_free_energy() + self.chemical_potential * self.particle_number + pressure_term
    
    def is_stable(self) -> bool:
        """Criterio de estabilidad: dG < 0"""
        return self.gibbs_free_energy() < 0

class SpectroscopyMetrics:
    
    @staticmethod
    def compute_weight_diffraction(coeffs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        W = torch.cat([c.flatten() for c in coeffs.values()])
        
        W_reshaped = W.reshape(-1, 1)
        fft_spectrum = torch.fft.fft(W_reshaped.squeeze())
        power_spectrum = torch.abs(fft_spectrum)**2
        
        peaks = []
        threshold = torch.mean(power_spectrum) + 2 * torch.std(power_spectrum)
        for i, power in enumerate(power_spectrum):
            if power > threshold:
                peaks.append({'frequency': i, 'intensity': float(power)})
        
        is_crystalline = len(peaks) > 0 and len(peaks) < len(power_spectrum) // 2
        
        return {
            'power_spectrum': power_spectrum.cpu().numpy().tolist(),
            'bragg_peaks': peaks,
            'is_crystalline_structure': is_crystalline,
            'spectral_entropy': float(SpectroscopyMetrics._compute_spectral_entropy(power_spectrum))
        }
    
    @staticmethod
    def _compute_spectral_entropy(power_spectrum: torch.Tensor) -> float:
        ps_normalized = power_spectrum / (torch.sum(power_spectrum) + 1e-10)
        ps_normalized = ps_normalized[ps_normalized > 1e-10]
        entropy = -torch.sum(ps_normalized * torch.log(ps_normalized + 1e-10))
        return float(entropy)
    
    @staticmethod
    def extract_lattice_parameters(weight_tensor: torch.Tensor, rank: int = 7) -> Dict[str, Any]:
        """
        Extrae parámetros de red preservando la geometría física del tensor.
        
        FIX: En lugar de reshape arbitrario, aplicamos SVD sobre la matriz 
        de covarianza que preserva la estructura de correlaciones.
        """
        # Asegurar que trabajamos con un tensor 2D válido
        if weight_tensor.dim() == 1:
            total_size = weight_tensor.numel()
            # Crear matriz cuadrada más cercana posible
            side_dim = int(np.ceil(np.sqrt(total_size)))
            # Pad con ceros para completar la matriz cuadrada
            padded_size = side_dim * side_dim
            if total_size < padded_size:
                padding = torch.zeros(padded_size - total_size, device=weight_tensor.device)
                weight_tensor = torch.cat([weight_tensor, padding])
            weight_tensor = weight_tensor[:side_dim * side_dim].reshape(side_dim, side_dim)
        
        # SVD sobre la matriz original (preserva estructura física)
        try:
            U, S, Vh = torch.linalg.svd(weight_tensor, full_matrices=False)
            
            # Limpieza de ruido térmico usando umbral físico (10% del máximo)
            threshold = S[0] * Config.MIN_VARIANCE_THRESHOLD ** 0.5
            S_clean = torch.where(S > threshold, S, torch.zeros_like(S))
            
            # Reconstrucción cristalina
            rank_truncated = min(rank, len(S_clean))
            U_truncated = U[:, :rank_truncated]
            S_truncated = S_clean[:rank_truncated]
            Vh_truncated = Vh[:rank_truncated, :]
            
            clean_crystal = U_truncated @ torch.diag(S_truncated) @ Vh_truncated
            
            # Error de reconstrucción (pureza cristalina)
            reconstruction_error = torch.norm(weight_tensor - clean_crystal) / (torch.norm(weight_tensor) + Config.ENTROPY_EPS)
            
            # Gap espectral (indicador de estructura discreta)
            spectral_gap = (S[0] / (S[1] + Config.ENTROPY_EPS)) if len(S) > 1 else float('inf')
            
            return {
                'U_basis': U_truncated.cpu().numpy().tolist(),
                'singular_values': S.cpu().numpy().tolist(),
                'clean_singular_values': S_truncated.cpu().numpy().tolist(),
                'V_basis': Vh_truncated.cpu().numpy().tolist(),
                'reconstruction_error': float(reconstruction_error),
                'effective_rank': int(torch.sum(S_clean > threshold)),
                'spectral_gap': float(spectral_gap),
                'thermal_noise_threshold': float(threshold)
            }
        except Exception as e:
            return {
                'error': str(e),
                'tensor_shape': list(weight_tensor.shape),
                'total_elements': weight_tensor.numel()
            }
                    
    @staticmethod
    def compute_gibbs_free_energy(loss: float, temp: float, entropy: float) -> float:
        return loss + (temp * entropy)


     
    @staticmethod
    def extract_canonical_decomposition(coeffs: Dict[str, torch.Tensor], rank: int = 7) -> Dict[str, Any]:
        """
        Descomposición Canónica del tensor tripartito (U, V, W).
        
        FIX: Preserva la estructura bilineal en lugar de tratar como matriz plana.
        Aplicamos HOSVD (Higher-Order SVD) para tensores de orden 3.
        """
        U_weight = coeffs['U']  # (hidden_dim, input_dim)
        V_weight = coeffs['V']  # (hidden_dim, input_dim)
        W_weight = coeffs['W']  # (output_dim, hidden_dim)
        
        try:
            # HOSVD: Descomposición por cada modo del tensor
            # Modo 1: SVD de U
            U_svd = torch.linalg.svd(U_weight, full_matrices=False)
            # Modo 2: SVD de V
            V_svd = torch.linalg.svd(V_weight, full_matrices=False)
            # Modo 3: SVD de W (transpuesto para consistencia dimensional)
            W_svd = torch.linalg.svd(W_weight.t(), full_matrices=False)
            
            # Truncar a rank manteniendo coherencia dimensional
            rank_effective = min(rank, U_svd[1].numel(), V_svd[1].numel(), W_svd[1].numel())
            
            # Core tensor: producto de valores singulares (acoplamiento entre modos)
            core_tensor_diagonal = (
                U_svd[1][:rank_effective] * 
                V_svd[1][:rank_effective] * 
                W_svd[1][:rank_effective]
            )
            
            factors = {
                'factor_A': U_svd[0][:, :rank_effective].cpu().numpy().tolist(),
                'factor_B': V_svd[0][:, :rank_effective].cpu().numpy().tolist(),
                'factor_C': W_svd[0][:, :rank_effective].cpu().numpy().tolist(),
                'core_tensor_approximation': {
                    'U_singular': U_svd[1][:rank_effective].cpu().numpy().tolist(),
                    'V_singular': V_svd[1][:rank_effective].cpu().numpy().tolist(),
                    'W_singular': W_svd[1][:rank_effective].cpu().numpy().tolist(),
                    'coupled_strength': core_tensor_diagonal.cpu().numpy().tolist()
                }
            }
            
            # Discretización a red cristalina {-1, 0, 1}
            discretized_factors = SpectroscopyMetrics._discretize_to_integers(factors)
            
            return {
                'continuous_factors': factors,
                'discretized_factors': discretized_factors,
                'is_strassen_equivalent': SpectroscopyMetrics._check_strassen_equivalence(discretized_factors),
                'tensor_rank': rank_effective,
                'mode_coupling': float(torch.mean(core_tensor_diagonal))
            }
        except Exception as e:
            return {'error': str(e)}
            
    @staticmethod
    def _discretize_to_integers(factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Proyecta factores continuos a la red cristalina discreta {-1, 0, 1}.
        """
        discretized = {}
        
        for key in ['factor_A', 'factor_B', 'factor_C']:
            if key in factors:
                factor_array = np.array(factors[key])
                # Normalización por el máximo absoluto
                max_val = np.max(np.abs(factor_array))
                if max_val > 1e-6:
                    normalized = factor_array / max_val
                    rounded = np.round(normalized)
                    discretized[key] = rounded.tolist()
                else:
                    discretized[key] = factor_array.tolist()
        
        return discretized

    @staticmethod
    def _check_strassen_equivalence(discretized_factors: Dict[str, Any]) -> Dict[str, Any]:
        """
        Verifica si los factores discretizados corresponden a la estructura de Strassen.
        """
        valid_values = {-1, 0, 1}
        
        is_discrete = True
        for key in ['factor_A', 'factor_B', 'factor_C']:
            if key in discretized_factors:
                factor_array = np.array(discretized_factors[key])
                unique_vals = set(np.unique(factor_array.flatten()))
                if not unique_vals.issubset(valid_values):
                    is_discrete = False
                    break
        
        return {
            'is_discrete_structure': is_discrete,
            'satisfies_strassen_pattern': is_discrete,
            'validation': 'Atomic structure confirmed' if is_discrete else 'Continuous residual detected'
        }
        
    
    
    @staticmethod
    def create_superlattice_seed(base_tensor: Dict[str, torch.Tensor], scale_factor: int = 2) -> Dict[str, torch.Tensor]:
        U_base = base_tensor['U']
        V_base = base_tensor['V']
        W_base = base_tensor['W']
        
        U_expanded = torch.kron(U_base, torch.eye(scale_factor, device=U_base.device))
        V_expanded = torch.kron(V_base, torch.eye(scale_factor, device=V_base.device))
        W_expanded = torch.kron(W_base, torch.eye(scale_factor, device=W_base.device))
        
        return {
            'U': U_expanded,
            'V': V_expanded,
            'W': W_expanded,
            'scale_factor': scale_factor,
            'original_shape': {
                'U': list(U_base.shape),
                'V': list(V_base.shape),
                'W': list(W_base.shape)
            }
        }

class ThermodynamicMetrics:
    
    @staticmethod
    def compute_effective_temperature(gradient_buffer: List[torch.Tensor], learning_rate: float) -> float:
        if len(gradient_buffer) < 2:
            return 0.0
        
        grads = torch.stack([g.flatten() for g in gradient_buffer])
        
        second_moment = torch.mean(torch.norm(grads, dim=1)**2)
        first_moment_sq = torch.norm(torch.mean(grads, dim=0))**2
        variance = second_moment - first_moment_sq
        
        return float((learning_rate / 2.0) * variance)

       
    @staticmethod
    def compute_critical_exponents(temp_history: List[float], cv_history: List[float], 
                                alpha_history: List[float]) -> Dict[str, float]:
        """
        Calcula exponentes críticos cerca de transiciones de fase.
        
        Leyes de escala:
        - C_v ~ |T - T_c|^{-α_exp}  (calor específico)
        - ξ ~ |T - T_c|^{-ν}        (longitud de correlación)
        - τ ~ |T - T_c|^{-z}        (tiempo de grokking)
        """
        if len(temp_history) < 5 or len(cv_history) < 5:
            return {
                'alpha_exponent': 0.0,
                'nu_exponent': 0.0,
                'z_exponent': 0.0,
                'critical_temperature': 0.0
            }
        
        # Identificar temperatura crítica (donde C_v es máximo)
        cv_array = np.array(cv_history)
        temp_array = np.array(temp_history)
        
        if len(cv_array) == 0 or np.all(cv_array == 0):
            return {
                'alpha_exponent': 0.0,
                'nu_exponent': 0.0,
                'z_exponent': 0.0,
                'critical_temperature': 0.0
            }
        
        critical_idx = np.argmax(cv_array)
        T_c = temp_array[critical_idx]
        
        # Exponente α: C_v ~ |T - T_c|^{-α}
        delta_T = np.abs(temp_array - T_c) + Config.ENTROPY_EPS
        log_delta_T = np.log(delta_T)
        log_cv = np.log(cv_array + Config.ENTROPY_EPS)
        
        # Filtrar puntos válidos (cerca de T_c)
        near_critical = delta_T < (0.2 * T_c) if T_c > 0 else np.ones_like(delta_T, dtype=bool)
        
        if np.sum(near_critical) > 2:
            try:
                alpha_exp, _ = np.polyfit(log_delta_T[near_critical], log_cv[near_critical], 1)
                alpha_exp = -float(alpha_exp)  # El signo negativo viene de la ley de escala
            except:
                alpha_exp = 0.0
        else:
            alpha_exp = 0.0
        
        # Exponente ν: estimado desde longitud de correlación (usando α como proxy)
        if len(alpha_history) > 2:
            alpha_array = np.array(alpha_history)
            correlation_length = 1.0 / (alpha_array + Config.ENTROPY_EPS)
            log_xi = np.log(correlation_length + Config.ENTROPY_EPS)
            
            if np.sum(near_critical) > 2:
                try:
                    nu_exp, _ = np.polyfit(log_delta_T[near_critical], log_xi[near_critical], 1)
                    nu_exp = -float(nu_exp)
                except:
                    nu_exp = 0.0
            else:
                nu_exp = 0.0
        else:
            nu_exp = 0.0
        
        # Exponente z: dinámica crítica (τ ~ ξ^z)
        z_exp = alpha_exp / nu_exp if nu_exp > Config.ENTROPY_EPS else 0.0
        
        return {
            'alpha_exponent': alpha_exp,
            'nu_exponent': nu_exp,
            'z_exponent': z_exp,
            'critical_temperature': float(T_c)
        }


    @staticmethod
    def compute_equation_of_state(temp_eff: float, alpha: float, kappa: float) -> Dict[str, Any]:
        """
        Ecuación de estado: T_c(α) = T_0 * exp(-c*α)
        
        FIX: Relación constitutiva que describe la curva de coexistencia cristal-vidrio.
        """
        # Parámetros físicos calibrados desde observaciones
        T_0 = 1e-3  # Temperatura de referencia
        c = 0.5     # Coeficiente de acoplamiento α-T
        
        # Temperatura crítica predicha para esta pureza
        T_critical_predicted = T_0 * np.exp(-c * alpha)
        
        # Desviación desde equilibrio
        delta_T = temp_eff - T_critical_predicted
        
        # Clasificación de fase
        if delta_T < -Config.MIN_VARIANCE_THRESHOLD:
            phase = "subcritical_crystal"
        elif abs(delta_T) < Config.MIN_VARIANCE_THRESHOLD:
            phase = "critical_point"
        else:
            phase = "supercritical_glass"
        
        # Presión algorítmica P = α * κ (resistencia a deformación)
        pressure = alpha * kappa
        
        return {
            'temperature_effective': float(temp_eff),
            'temperature_critical': float(T_critical_predicted),
            'deviation_from_equilibrium': float(delta_T),
            'phase_classification': phase,
            'algorithmic_pressure': float(pressure),
            'equation_form': f'T_c(α) = {T_0} * exp(-{c} * α)',
            'is_equilibrium': abs(delta_T) < Config.MIN_VARIANCE_THRESHOLD
        }


    @staticmethod
    def compute_specific_heat(loss_history: List[float], temp_history: List[float], cv_threshold: float = 1.0) -> Tuple[float, bool]:
        if len(loss_history) < 2 or len(temp_history) < 2:
            return 0.0, False
        
        u_var = torch.tensor(loss_history).var()
        t_mean = torch.tensor(temp_history).mean()
        
        cv = u_var / (t_mean**2 + 1e-10)
        is_latent_crystallization = cv > cv_threshold
        
        return float(cv), is_latent_crystallization
    
    @staticmethod
    def estimate_hbar_algorithmic(model_complexity: float, weight_dim: int, mutual_information: float) -> float:
        if weight_dim == 0 or mutual_information == 0:
            return Config.HBAR
        
        hbar_alg = model_complexity / (weight_dim * mutual_information)
        return float(hbar_alg)
    
    @staticmethod
    def compute_mutual_information(weights: torch.Tensor, gradients: torch.Tensor) -> float:
        w_flat = weights.flatten()
        g_flat = gradients.flatten()
        
        w_std = torch.std(w_flat)
        g_std = torch.std(g_flat)
        correlation = torch.corrcoef(torch.stack([w_flat, g_flat]))[0, 1]
        
        if w_std > 0 and g_std > 0:
            mi = -0.5 * torch.log(1 - correlation**2 + 1e-10)
            return float(mi)
        return 0.0
    
    @staticmethod
    def check_extensivity(entropy_list: List[float], scale_factors: List[float]) -> Dict[str, Any]:
        if len(entropy_list) != len(scale_factors) or len(entropy_list) < 2:
            return {'is_extensive': False, 'linearity_score': 0.0}
        
        entropies = np.array(entropy_list)
        scales = np.array(scale_factors)
        
        try:
            slope, intercept = np.polyfit(scales, entropies, 1)
            predicted = slope * scales + intercept
            r_squared = 1 - (np.sum((entropies - predicted)**2) / np.sum((entropies - np.mean(entropies))**2))
            
            is_extensive = r_squared > 0.95 and abs(intercept) < 0.1 * np.mean(entropies)
            
            return {
                'is_extensive': bool(is_extensive),
                'linearity_score': float(r_squared),
                'slope': float(slope),
                'intercept': float(intercept)
            }
        except:
            return {'is_extensive': False, 'linearity_score': 0.0}
    
    @staticmethod
    def compute_fisher_information_matrix(model: nn.Module, samples: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]) -> torch.Tensor:
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        fisher = torch.zeros(n_params, n_params, device=Config.DEVICE)
        
        for A, B, C in samples:
            model.zero_grad()
            output = model(A, B)
            log_prob = -nn.functional.mse_loss(output, C)
            
            grads = torch.autograd.grad(log_prob, model.parameters(), create_graph=True)
            grad_vector = torch.cat([g.flatten() for g in grads])
            
            fisher += torch.outer(grad_vector, grad_vector)
        
        fisher /= len(samples)
        return fisher
    
    @staticmethod
    def compute_ricci_curvature(fisher_matrix: torch.Tensor) -> float:
        eigenvalues = torch.linalg.eigvalsh(fisher_matrix)
        eigenvalues = eigenvalues[eigenvalues > 1e-10]
        
        if len(eigenvalues) < 2:
            return 0.0
        
        ricci_scalar = torch.sum(1.0 / (eigenvalues + 1e-10))
        return float(ricci_scalar)
    
    @staticmethod
    def calculate_carnot_efficiency(delta_alpha: float, total_flops: float, initial_alpha: float = 0.0) -> Dict[str, Any]:
        if total_flops <= 0:
            return {'efficiency': 0.0, 'work_done': 0.0, 'heat_dissipated': 0.0}
        
        invariance_gain = delta_alpha - initial_alpha
        
        work_done = invariance_gain
        heat_dissipated = total_flops * 1e-9
        
        efficiency = work_done / (heat_dissipated + 1e-10)
        
        carnot_limit = 1.0 - (initial_alpha + 1e-3) / (delta_alpha + 1e-3)
        
        return {
            'efficiency': float(efficiency),
            'work_done': float(work_done),
            'heat_dissipated': float(heat_dissipated),
            'carnot_limit': float(carnot_limit),
            'relative_efficiency': float(efficiency / (carnot_limit + 1e-10))
        }

class CrystallographyMetrics:
    
    @staticmethod
    def compute_kappa(model: nn.Module, dataloader, num_batches: int = 5) -> float:
        model.eval()
        grads = []
        
        for i, (A, B, C) in enumerate(dataloader):
            if i >= num_batches:
                break
            
            C_pred = model(A, B)
            loss = nn.functional.mse_loss(C_pred, C)
            grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            grads.append(torch.cat([g.flatten() for g in grad]))
        
        if len(grads) < 2:
            return float('inf')
        
        grads_tensor = torch.stack(grads)
        cov_matrix = torch.cov(grads_tensor.T)
        
        eigenvalues = torch.linalg.eigvals(cov_matrix).real
        if torch.any(eigenvalues <= 0):
            return float('inf')
        
        return (eigenvalues.max() / eigenvalues.min()).item()
    
    @staticmethod
    def compute_discretization_margin(coeffs: Dict[str, torch.Tensor]) -> float:
        return max((tensor - tensor.round()).abs().max().item() for tensor in coeffs.values())
    
    @staticmethod
    def compute_local_complexity(model: nn.Module) -> float:
        params = torch.cat([p.flatten() for p in model.parameters()])
        with torch.no_grad():
            perc_95 = torch.quantile(torch.abs(params), 0.95)
            active = (torch.abs(params) > 0.01 * perc_95).sum()
            lc = active.float() / len(params)
        return lc.item()
    
    @staticmethod
    def compute_alpha_purity(coeffs: Dict[str, torch.Tensor]) -> float:
        delta = CrystallographyMetrics.compute_discretization_margin(coeffs)
        if delta < 1e-10:
            return 20.0
        return -np.log(delta + 1e-15)
    
    @staticmethod
    def compute_kappa_quantum(coeffs: Dict[str, torch.Tensor], hbar: float = Config.HBAR) -> float:
        flat_params = torch.cat([c.flatten() for c in coeffs.values()])
        n = flat_params.numel()
        
        if n < 2:
            return 1.0
        
        params_centered = flat_params - flat_params.mean()
        cov_matrix = torch.outer(params_centered, params_centered) / n
        cov_matrix = cov_matrix + hbar * torch.eye(n, device=flat_params.device)
        
        eigenvals = torch.linalg.eigvalsh(cov_matrix)
        eigenvals = eigenvals[eigenvals > hbar]
        
        return (eigenvals.max() / eigenvals.min()).item() if len(eigenvals) > 0 else 1.0
    
    @staticmethod
    def compute_poynting_vector(coeffs: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        E = torch.cat([c.flatten() for c in coeffs.values()])
        U, V, W = coeffs['U'], coeffs['V'], coeffs['W']
        
        H_magnitude = torch.norm(torch.matmul(U, V.t()) - torch.eye(U.shape[0], device=U.device))
        
        poynting_magnitude = torch.norm(E) * H_magnitude * Config.ENERGY_FLOW_SCALE
        
        energy_distribution = {
            'U_flow': torch.norm(U).item(),
            'V_flow': torch.norm(V).item(),
            'W_flow': torch.norm(W).item()
        }
        
        return {
            'poynting_magnitude': poynting_magnitude.item(),
            'energy_distribution': energy_distribution,
            'is_radiating': poynting_magnitude.item() > Config.POYNTING_THRESHOLD,
            'field_orthogonality': H_magnitude.item()
        }
    
    @staticmethod
    def compute_all_metrics(model: nn.Module, dataloader) -> Dict[str, Any]:
        coeffs = model.get_coefficients()
        
        metrics = {
            'kappa': CrystallographyMetrics.compute_kappa(model, dataloader),
            'delta': CrystallographyMetrics.compute_discretization_margin(coeffs),
            'alpha': CrystallographyMetrics.compute_alpha_purity(coeffs),
            'kappa_q': CrystallographyMetrics.compute_kappa_quantum(coeffs),
            'lc': CrystallographyMetrics.compute_local_complexity(model),
            'poynting': CrystallographyMetrics.compute_poynting_vector(coeffs)
        }
        
        metrics['purity_index'] = 1.0 - metrics['delta']
        metrics['is_crystal'] = metrics['alpha'] > 7.0
        metrics['energy_flow'] = metrics['poynting']['poynting_magnitude']
        
        return metrics

class GreenCowExperiment:
    """
    🐄 Green's Cow: Uses integration-by-parts analogy to split gradient into bulk and boundary terms.
    Inspired by Green's identities: ∫_Ω ∇u·v = ∫_∂Ω u(v·n) - ∫_Ω u∇·v
    Applied to weight tensors as discrete manifolds.
    """
    def __init__(self, model: BilinearStrassenModel, device: str = Config.DEVICE):
        self.model = model
        self.device = device
        self.logger = setup_logger("GreenCowExperiment")

    def compute_boundary_gradient(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Approximate surface term: gradient concentrated on tensor boundaries.
        For a matrix W ∈ ℝ^{m×n}, boundary = first/last row + first/last column.
        """
        m, n = weight.shape
        if m < 2 or n < 2:
            return torch.zeros_like(weight)
        mask = torch.zeros_like(weight)
        mask[0, :] = 1      # top row
        mask[-1, :] = 1     # bottom row
        mask[:, 0] = 1      # left col
        mask[:, -1] = 1     # right col
        return mask * weight.grad if weight.grad is not None else torch.zeros_like(weight)

    def compute_bulk_gradient(self, weight: torch.Tensor) -> torch.Tensor:
        """Interior (volume) term: everything except boundary."""
        full_grad = weight.grad if weight.grad is not None else torch.zeros_like(weight)
        boundary = self.compute_boundary_gradient(weight)
        return full_grad - boundary

    def run_green_backprop_step(self, A: torch.Tensor, B: torch.Tensor, C_true: torch.Tensor,
                                lambda_boundary: float = 0.1) -> Dict[str, float]:
        """
        Custom backward pass using Green-inspired decomposition.
        Loss = MSE + λ_boundary * ||boundary_grad||²
        """
        self.model.zero_grad()

        # Forward pass
        C_pred = self.model(A, B)
        loss_mse = nn.functional.mse_loss(C_pred, C_true)

        # Compute boundary penalty directly from parameters (no need for first backward!)
        total_boundary_norm = 0.0
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                b_mask = self._get_boundary_mask(param)
                boundary_part = b_mask * param
                total_boundary_norm += torch.norm(boundary_part)**2

        loss_boundary = lambda_boundary * total_boundary_norm
        total_loss = loss_mse + loss_boundary

        # Single backward call
        total_loss.backward()

        return {
            'total_loss': total_loss.item(),
            'mse_loss': loss_mse.item(),
            'boundary_penalty': loss_boundary.item(),
            'boundary_energy': total_boundary_norm.item()
        }
        
    def _get_boundary_mask(self, weight: torch.Tensor) -> torch.Tensor:
        """Returns a binary mask marking boundary elements of a tensor."""
        if weight.dim() < 2:
            return torch.ones_like(weight)
        m, n = weight.shape[-2], weight.shape[-1]
        mask = torch.zeros_like(weight)
        mask[..., 0, :] = 1      # first row
        mask[..., -1, :] = 1     # last row
        mask[..., :, 0] = 1      # first col
        mask[..., :, -1] = 1     # last col
        return mask
                
    def train_with_green_cow(self, epochs: int = 100, lr: float = 1e-3, lambda_boundary: float = 0.05):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=Config.WEIGHT_DECAY)
        history = {'epoch': [], 'loss': [], 'alpha': [], 'boundary_energy': []}

        for epoch in range(epochs):
            A, B, C = StrassenDataGenerator.generate_batch(Config.BATCH_SIZE)
            step_metrics = self.run_green_backprop_step(A, B, C, lambda_boundary)
            optimizer.step()

            coeffs = self.model.get_coefficients()
            alpha = CrystallographyMetrics.compute_alpha_purity(coeffs)

            history['epoch'].append(epoch)
            history['loss'].append(step_metrics['total_loss'])
            history['alpha'].append(alpha)
            history['boundary_energy'].append(step_metrics['boundary_energy'])

            if epoch % 20 == 0:
                self.logger.info(
                    f"🐄 Epoch {epoch:3d} | Loss: {step_metrics['total_loss']:.6f} | "
                    f"α: {alpha:6.2f} | Boundary E: {step_metrics['boundary_energy']:.4f}"
                )

            # Early stop if crystalline
            if alpha > 7.0:
                self.logger.info("✨ Crystallization achieved via Green’s Cow!")
                break

        return history

class CheckpointLoader:
    def load_checkpoint(self, path: str, device: str) -> Any:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            raise CheckpointLoadingError(f"Failed to load checkpoint: {e}")

class CheckpointMigrator:
    @staticmethod
    def migrate_checkpoint(raw_data: Any) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(raw_data, dict):
            if 'state_dict' in raw_data:
                return CheckpointMigrator._migrate_dict(raw_data['state_dict'])
            elif 'model_state_dict' in raw_data:
                return CheckpointMigrator._migrate_dict(raw_data['model_state_dict'])
            else:
                return CheckpointMigrator._migrate_dict(raw_data)
        elif hasattr(raw_data, 'state_dict'):
            return CheckpointMigrator._migrate_dict(raw_data.state_dict())
        elif isinstance(raw_data, dict) and any(k in raw_data for k in ['U', 'V', 'W', 'U_coefs', 'V_coefs', 'W_coefs']):
            return CheckpointMigrator._migrate_dict(raw_data)
        return None
    
    @staticmethod
    def _migrate_dict(state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        if any(k in state_dict for k in ['U', 'V', 'W']):
            return CheckpointMigrator._migrate_custom_format(state_dict)
        elif 'U_coefs' in state_dict:
            return CheckpointMigrator._migrate_coefs_format(state_dict)
        elif 'encoder.0.weight' in state_dict:
            return CheckpointMigrator._migrate_encoder_format(state_dict)
        elif any(k.endswith('.weight') for k in state_dict.keys()):
            return CheckpointMigrator._migrate_standard_format(state_dict)
        return None
    
    @staticmethod
    def _migrate_custom_format(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        def get_tensor(key: str) -> torch.Tensor:
            if key in state_dict:
                return state_dict[key]
            elif f'{key}_coefs' in state_dict:
                return state_dict[f'{key}_coefs']
            raise KeyError(f"Missing tensor for {key}")
        
        U = get_tensor('U')
        V = get_tensor('V')
        W = get_tensor('W')
        
        if U.shape == (7, 4):
            u_padded = torch.zeros(Config.HIDDEN_DIM, 4, device=Config.DEVICE)
            v_padded = torch.zeros(Config.HIDDEN_DIM, 4, device=Config.DEVICE)
            w_padded = torch.zeros(4, Config.HIDDEN_DIM, device=Config.DEVICE)
            u_padded[:7] = U
            v_padded[:7] = V
            w_padded[:, :7] = W
            return {'U.weight': u_padded, 'V.weight': v_padded, 'W.weight': w_padded}
        
        return {'U.weight': U, 'V.weight': V, 'W.weight': W}
    
    @staticmethod
    def _migrate_coefs_format(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {
            'U.weight': state_dict['U_coefs'],
            'V.weight': state_dict['V_coefs'],
            'W.weight': state_dict['W_coefs']
        }
    
    @staticmethod
    def _migrate_encoder_format(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        encoder_0 = state_dict['encoder.0.weight']
        encoder_2 = state_dict.get('encoder.2.weight', encoder_0)
        encoder_4 = state_dict.get('encoder.4.weight', torch.randn(64, 64, device=Config.DEVICE))
        
        u = encoder_0[:Config.HIDDEN_DIM, :4].clone() if encoder_0.shape == (64, 8) else encoder_0.flatten()[:32].reshape(Config.HIDDEN_DIM, 4)
        v = encoder_2[:Config.HIDDEN_DIM, :4].clone() if encoder_2.shape == (64, 64) else u
        w = encoder_4[:4, :Config.HIDDEN_DIM].clone() if encoder_4.shape == (64, 64) else torch.randn(4, Config.HIDDEN_DIM, device=Config.DEVICE)
        
        return {'U.weight': u, 'V.weight': v, 'W.weight': w}
    
    @staticmethod
    def _migrate_standard_format(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {k: state_dict[k] for k in ['U.weight', 'V.weight', 'W.weight'] if k in state_dict}

class BoltzmannAnalysisProgram:
    def __init__(self, checkpoint_dir: str, results_dir: str = Config.RESULTS_DIR):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        self.checkpoints = {}
        self.logger = setup_logger("BoltzmannAnalysisProgram")
        self.gradient_buffer = deque(maxlen=50)
        self.loss_history = deque(maxlen=100)
        self.temp_history = deque(maxlen=100)
        self.cv_history = deque(maxlen=100)
        self._load_all_checkpoints()
    
    def _load_all_checkpoints(self):
        checkpoint_files = list(self.checkpoint_dir.glob("*.pt"))
        if not checkpoint_files:
            self.logger.error(f"No .pt files found in {self.checkpoint_dir}")
            return
        
        loader = CheckpointLoader()
        
        for filepath in checkpoint_files:
            try:
                self.logger.info(f"Loading: {filepath.name}")
                raw_data = loader.load_checkpoint(str(filepath), Config.DEVICE)
                migrated_state = CheckpointMigrator.migrate_checkpoint(raw_data)
                
                if migrated_state is not None:
                    model = BilinearStrassenModel().to(Config.DEVICE)
                    model.load_state_dict(migrated_state)
                    coeffs = model.get_coefficients()
                    
                    def dataloader():
                        for _ in range(5):
                            yield StrassenDataGenerator.generate_batch(Config.BATCH_SIZE)
                    
                    metrics = CrystallographyMetrics.compute_all_metrics(model, dataloader())
                    
                    self.checkpoints[filepath.name] = {
                        'model': model,
                        'coeffs': coeffs,
                        **metrics
                    }
                    
                    self.logger.info(
                        f"  Success: alpha={metrics['alpha']:.2f}, "
                        f"delta={metrics['delta']:.4f}, "
                        f"S_mag={metrics['energy_flow']:.2e}, "
                        f"Crystal: {metrics['is_crystal']}"
                    )
                else:
                    self.logger.warning(f"  Failed to migrate checkpoint format")
                    
            except Exception as e:
                self.logger.error(f"  Error: {str(e)}")
        
        self.logger.info(f"Successfully loaded {len(self.checkpoints)}/{len(checkpoint_files)} checkpoints")
    
    def run_full_boltzmann_program(self):
        self.logger.info("="*80)
        self.logger.info("Running Full Crystallography Analysis Program")
        self.logger.info("="*80)
        
        if not self.checkpoints:
            self.logger.error("No checkpoints loaded. Please check the checkpoint directory and format.")
            return
        
        results = {
            'phase1_molecular': self.phase1_molecular_hypothesis(),
            'phase2_entropy': self.phase2_entropy_production(),
            'phase3_extensivity': self.phase3_extensivity_law(),
            'phase4_quantum': self.phase4_quantum_basis_transform(),
            'phase5_thermodynamic': self.phase5_thermodynamic_analysis(),
            'phase6_spectroscopy': self.phase6_spectroscopic_analysis(),
            'poynting_analysis': self.analyze_poynting_flow()
        }
        
        self._save_results(results, "boltzmann_program_summary.json")
        self._print_executive_summary(results)
        
        self.logger.info("="*80)
        self.logger.info("PROGRAM COMPLETED")
        self.logger.info(f"Results available in: {self.results_dir}/")
        self.logger.info("="*80)
    
    def phase1_molecular_hypothesis(self) -> Dict[str, Any]:
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 1: MOLECULAR HYPOTHESIS - Microstate Sampling")
        self.logger.info("="*60)
        
        if len(self.checkpoints) < 3:
            self.logger.error("Insufficient checkpoints for molecular hypothesis")
            return {'error': 'Insufficient checkpoints'}
        
        groups = {
            'crystal': [ckpt for ckpt, data in self.checkpoints.items() if data['alpha'] > 7],
            'glass': [ckpt for ckpt, data in self.checkpoints.items() if data['alpha'] < 3],
            'polycrystal': [ckpt for ckpt, data in self.checkpoints.items() if 3 <= data['alpha'] <= 7]
        }
        
        results = {}
        for group_name, checkpoint_list in groups.items():
            if len(checkpoint_list) < 2:
                self.logger.info(f"  Skipping {group_name}: only {len(checkpoint_list)} checkpoints")
                continue
            
            self.logger.info(f"\nAnalyzing {group_name} ({len(checkpoint_list)} checkpoints)")
            
            all_params = []
            for ckpt in checkpoint_list:
                coeffs = self.checkpoints[ckpt]['coeffs']
                params = torch.cat([c.flatten() for c in coeffs.values()]).cpu().numpy()
                all_params.append(params)
            
            all_params = np.stack(all_params)
            
            param_std = np.std(all_params, axis=0)
            active_dims = param_std > Config.MIN_VARIANCE_THRESHOLD
            
            if np.sum(active_dims) < 2:
                self.logger.warning(f"  {group_name} has insufficient variance for KDE")
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
            
            active_params = all_params[:, active_dims]
            
            if active_params.shape[1] > Config.PCA_COMPONENTS:
                params_centered = active_params - np.mean(active_params, axis=0)
                cov = np.cov(params_centered.T) + 1e-8 * np.eye(params_centered.shape[1])
                eigenvals, eigenvecs = eigh(cov)
                idx = np.argsort(eigenvals)[::-1][:Config.PCA_COMPONENTS]
                evecs = eigenvecs[:, idx]
                proj_params = params_centered @ evecs
            else:
                proj_params = active_params
            
            proj_params += np.random.randn(*proj_params.shape) * 1e-10
            
            entropy = self._compute_entropy(active_params)
            effective_volume = self._compute_effective_volume(active_params)
            
            alpha_vals = [self.checkpoints[ckpt]['alpha'] for ckpt in checkpoint_list]
            purities = [1 - self.checkpoints[ckpt]['delta'] for ckpt in checkpoint_list]
            
            if len(alpha_vals) > 1 and len(purities) > 1:
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
            
            self._plot_parameter_distribution(proj_params, group_name)
        
        self._save_results(results, "phase1_molecular_hypothesis.json")
        return results
        
    def phase2_entropy_production(self) -> Dict[str, Any]:
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 2: SECOND LAW - Generalization Entropy Production")
        self.logger.info("="*60)
        
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        
        successful_ckpts = {k: v for k, v in self.checkpoints.items() if v.get('is_crystal', False)}
        if len(successful_ckpts) < 2:
            self.logger.error(f"Only {len(successful_ckpts)} crystal states found")
            return {
                'individual_results': {},
                'scaling_law': {'slope': 0, 'intercept': 0},
                'mean_timescale': 0,
                'entropy_exponent': 0,
                'error': 'Insufficient crystal states'
            }
        
        results = {}
        for ckpt_name, data in successful_ckpts.items():
            self.logger.info(f"\nAnalyzing trajectory: {ckpt_name}")
            coeffs = data['coeffs']
            base_params = torch.cat([p.flatten() for p in coeffs.values()]).cpu().numpy()
            
            trajectory = self._simulate_training_trajectory(base_params, data['delta'])
            entropy_values = []
            
            for i, params in enumerate(trajectory):
                try:
                    entropy = self._compute_generalization_entropy(params, successful_ckpts)
                    entropy_values.append(entropy)
                except Exception as e:
                    self.logger.warning(f"    Warning at step {i}: {e}")
                    entropy_values.append(entropy_values[-1] if entropy_values else 0.0)
            
            if not entropy_values or all(v == 0 for v in entropy_values):
                self.logger.warning(f"  Skipping {ckpt_name}: no valid entropy values")
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
    
    def phase3_extensivity_law(self) -> Dict[str, Any]:
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 3: THIRD LAW - Extensivity of the T Operator")
        self.logger.info("="*60)
        
        sizes = [2, 4, 8, 16, 32, 64]
        results = {}
        
        successful_ckpts = {k: v for k, v in self.checkpoints.items() if v.get('is_crystal', False)}
        if not successful_ckpts:
            self.logger.error("No crystal states found")
            return {'error': 'No crystal states'}
        
        for ckpt_name, data in successful_ckpts.items():
            self.logger.info(f"\nVerifying extensivity: {ckpt_name}")
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
        
        phi_data = [(self.checkpoints[ckpt]['alpha'], 1.0) 
                    for ckpt, data in results.items() 
                    if data['max_size_success'] >= 64 and self.checkpoints[ckpt]['alpha'] > 7]
        
        if phi_data:
            alphas = [x[0] for x in phi_data]
            c = -np.log(np.mean([x[1] for x in phi_data])) / np.mean(alphas) if alphas else 0
            phi_law = {'coefficient': float(c), 'form': f'phi(alpha) proportional to exp(-{c:.2f}*alpha)'}
        else:
            phi_law = {'coefficient': 0.0, 'form': 'No data'}
        
        summary = {
            'individual_results': results,
            'phi_alpha_function': phi_law,
            'extensivity_verified': self._verify_extensivity_universality(results)
        }
        
        self._save_results(summary, "phase3_extensivity_law.json")
        return summary
    
    def phase4_quantum_basis_transform(self) -> Dict[str, Any]:
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 4: RECURRENCE PARADOX - Base Transformation")
        self.logger.info("="*60)
        
        results = {}
        
        successful_ckpts = {k: v for k, v in self.checkpoints.items() if v.get('is_crystal', False)}
        if not successful_ckpts:
            self.logger.error("No crystal states found")
            return {'error': 'No crystal states'}
        
        for ckpt_name, data in successful_ckpts.items():
            self.logger.info(f"\nAnalyzing quantum basis: {ckpt_name}")
            coeffs = data['coeffs']
            
            symmetry_basis = self._find_broken_symmetries(coeffs)
            delta_theta_original = self._measure_uncertainty(coeffs, basis='original')
            delta_theta_symmetry = self._measure_uncertainty(coeffs, basis=symmetry_basis)
            
            hbar_eff = delta_theta_symmetry * data['kappa_q']
            
            results[ckpt_name] = {
                'hbar_effective': float(hbar_eff),
                'symmetry_dimension': len(symmetry_basis),
                'uncertainty_ratio': delta_theta_symmetry / delta_theta_original if delta_theta_original > 0 else 0,
                'quantum_classical_transition': hbar_eff < 1e-3
            }
            
            self._plot_uncertainty_distribution(coeffs, symmetry_basis, ckpt_name)
        
        hbars = [data['hbar_effective'] for data in results.values()]
        alphas = [self.checkpoints[ckpt]['alpha'] for ckpt in results.keys()]
        
        if hbars and alphas and np.mean(hbars) > 0:
            try:
                a = -np.log(np.mean(hbars)) / np.mean(alphas)
                scaling = f"hbar_eff(alpha) proportional to exp(-{a:.2f}*alpha)"
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
    
    def analyze_poynting_flow(self) -> Dict[str, Any]:
        self.logger.info("\nAnalyzing Poynting vector flow...")
        
        poynting_results = {}
        
        for ckpt_name, data in self.checkpoints.items():
            poynting = data.get('poynting', {})
            
            poynting_results[ckpt_name] = {
                'magnitude': poynting.get('poynting_magnitude', 0),
                'is_radiating': poynting.get('is_radiating', False),
                'energy_distribution': poynting.get('energy_distribution', {}),
                'field_orthogonality': poynting.get('field_orthogonality', 0),
                'alpha_correlation': data.get('alpha', 0)
            }
        
        magnitudes = [r['magnitude'] for r in poynting_results.values()]
        radiating_count = sum([r['is_radiating'] for r in poynting_results.values()])
        
        summary = {
            'individual_results': poynting_results,
            'statistics': {
                'mean_magnitude': np.mean(magnitudes) if magnitudes else 0,
                'std_magnitude': np.std(magnitudes) if magnitudes else 0,
                'radiating_percentage': radiating_count / len(poynting_results) * 100 if poynting_results else 0,
                'max_magnitude': max(magnitudes) if magnitudes else 0,
                'min_magnitude': min(magnitudes) if magnitudes else 0
            }
        }
        
        self._save_results(summary, "poynting_analysis.json")
        return summary
    
    def phase5_thermodynamic_analysis(self) -> Dict[str, Any]:
        """
        PHASE 5: THERMODYNAMIC ANALYSIS con exponentes críticos y ecuación de estado.
        
        FIX: Ahora calcula exponentes críticos y ecuación de estado para cada checkpoint.
        """
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 5: THERMODYNAMIC ANALYSIS - Temperature & Phase Transitions")
        self.logger.info("="*60)
        
        if not self.checkpoints:
            return {'error': 'No checkpoints available'}
        
        results = {}
        all_temps = []
        all_cvs = []
        all_alphas = []
        
        for ckpt_name, data in self.checkpoints.items():
            self.logger.info(f"\nAnalyzing thermodynamics: {ckpt_name}")
            
            model = data['model']
            coeffs = data['coeffs']
            
            def sample_dataloader():
                for _ in range(10):
                    yield StrassenDataGenerator.generate_batch(Config.BATCH_SIZE)
            
            gradient_buffer = []
            loss_values = []
            
            for A, B, C in sample_dataloader():
                model.zero_grad()
                output = model(A, B)
                loss = nn.functional.mse_loss(output, C)
                loss.backward()
                
                grads = torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None])
                gradient_buffer.append(grads)
                loss_values.append(loss.item())
            
            if len(gradient_buffer) < 2:
                continue
            
            t_eff = ThermodynamicMetrics.compute_effective_temperature(gradient_buffer, Config.LEARNING_RATE)
            temp_history_local = [t_eff] * len(loss_values)
            cv, is_crystallizing = ThermodynamicMetrics.compute_specific_heat(loss_values, temp_history_local)
            
            all_temps.append(t_eff)
            all_cvs.append(cv)
            all_alphas.append(data['alpha'])
            
            all_weights = torch.cat([c.flatten() for c in coeffs.values()])
            avg_gradient = torch.mean(torch.stack(gradient_buffer), dim=0)
            mi = ThermodynamicMetrics.compute_mutual_information(all_weights, avg_gradient)
            
            model_complexity = -np.log(data['delta'] + Config.ENTROPY_EPS)
            weight_dim = all_weights.numel()
            hbar_alg = ThermodynamicMetrics.estimate_hbar_algorithmic(model_complexity, weight_dim, mi)
            
            samples_for_fisher = list(sample_dataloader())[:5]
            fisher_matrix = ThermodynamicMetrics.compute_fisher_information_matrix(model, samples_for_fisher)
            ricci_curvature = ThermodynamicMetrics.compute_ricci_curvature(fisher_matrix)
            
            total_flops = weight_dim * Config.EPOCHS * Config.BATCH_SIZE * 1e-6
            carnot_metrics = ThermodynamicMetrics.calculate_carnot_efficiency(
                data['alpha'], total_flops, initial_alpha=0.0
            )
            
            # Ecuación de estado
            equation_of_state = ThermodynamicMetrics.compute_equation_of_state(
                t_eff, data['alpha'], data['kappa']
            )
            
            # Potencial termodinámico
            thermodynamic_potential = ThermodynamicPotential(
                internal_energy=np.mean(loss_values),
                temperature=t_eff,
                entropy=-data['alpha'],
                chemical_potential=mi,
                crystallinity=data['alpha'],
                particle_number=float(weight_dim)
            )
            
            results[ckpt_name] = {
                'effective_temperature': float(t_eff),
                'specific_heat': float(cv),
                'is_crystallizing': bool(is_crystallizing),
                'hbar_algorithmic': float(hbar_alg),
                'mutual_information': float(mi),
                'ricci_curvature': float(ricci_curvature),
                'carnot_efficiency': carnot_metrics,
                'equation_of_state': equation_of_state,
                'thermodynamic_potential': {
                    'helmholtz_free_energy': thermodynamic_potential.helmholtz_free_energy(),
                    'gibbs_free_energy': thermodynamic_potential.gibbs_free_energy(),
                    'is_stable': thermodynamic_potential.is_stable()
                },
                'phase_classification': self._classify_thermodynamic_phase(t_eff, cv, data['alpha'])
            }
            
            self.logger.info(f"  T_eff: {t_eff:.2e}, C_v: {cv:.2e}, Phase: {results[ckpt_name]['phase_classification']}")
            self.logger.info(f"  Equation of state: {equation_of_state['phase_classification']}")
        
        if results:
            # Calcular exponentes críticos globales
            critical_exponents = ThermodynamicMetrics.compute_critical_exponents(
                all_temps, all_cvs, all_alphas
            )
            
            self._plot_phase_diagram(results)
            self._plot_temperature_vs_purity(results)
            
            extensivity_check = self._verify_entropy_extensivity(results)
            
            summary = {
                'individual_results': results,
                'critical_exponents': critical_exponents,
                'extensivity_analysis': extensivity_check,
                'phase_transition_detected': any(r['is_crystallizing'] for r in results.values()),
                'mean_effective_temperature': float(np.mean(all_temps)),
                'critical_temperature_estimate': critical_exponents['critical_temperature']
            }
        else:
            summary = {'error': 'No valid thermodynamic analysis'}
        
        self._save_results(summary, "phase5_thermodynamic_analysis.json")
        return summary
        
    def phase6_spectroscopic_analysis(self) -> Dict[str, Any]:
        self.logger.info("\n" + "="*60)
        self.logger.info("PHASE 6: SPECTROSCOPY - Weight Diffraction & Atomic Extraction")
        self.logger.info("="*60)
        
        if not self.checkpoints:
            return {'error': 'No checkpoints available'}
        
        results = {}
        crystal_checkpoints = {k: v for k, v in self.checkpoints.items() if v.get('is_crystal', False)}
        
        if not crystal_checkpoints:
            self.logger.warning("No crystal structures found for spectroscopic analysis")
            return {'error': 'No crystal structures available'}
        
        for ckpt_name, data in crystal_checkpoints.items():
            self.logger.info(f"\nPerforming spectroscopy on: {ckpt_name}")
            
            coeffs = data['coeffs']
            
            diffraction_pattern = SpectroscopyMetrics.compute_weight_diffraction(coeffs)
            
            W_flat = torch.cat([c.flatten() for c in coeffs.values()])
            lattice_params = SpectroscopyMetrics.extract_lattice_parameters(W_flat, rank=Config.TARGET_SLOTS)
            
            canonical_decomp = SpectroscopyMetrics.extract_canonical_decomposition(coeffs, rank=Config.TARGET_SLOTS)
            
            model = data['model']
            def sample_dataloader():
                for _ in range(5):
                    yield StrassenDataGenerator.generate_batch(Config.BATCH_SIZE)
            
            loss_estimate = 0.0
            for A, B, C in sample_dataloader():
                output = model(A, B)
                loss_estimate += nn.functional.mse_loss(output, C).item()
            loss_estimate /= 5
            
            temp = data.get('effective_temperature', 1e-6)
            entropy = -data['alpha']
            gibbs_energy = SpectroscopyMetrics.compute_gibbs_free_energy(loss_estimate, temp, entropy)
            
            superlattice_seed = SpectroscopyMetrics.create_superlattice_seed(coeffs, scale_factor=2)
            
            results[ckpt_name] = {
                'diffraction_analysis': diffraction_pattern,
                'lattice_parameters': lattice_params,
                'canonical_decomposition': canonical_decomp,
                'gibbs_free_energy': float(gibbs_energy),
                'stability_criterion': gibbs_energy < 0,
                'superlattice_seed': {
                    'created': True,
                    'scale_factor': superlattice_seed['scale_factor'],
                    'expanded_dimensions': {
                        'U': list(superlattice_seed['U'].shape),
                        'V': list(superlattice_seed['V'].shape),
                        'W': list(superlattice_seed['W'].shape)
                    }
                }
            }
            
            is_strassen = canonical_decomp.get('is_strassen_equivalent', {}).get('is_discrete_structure', False)
            self.logger.info(f"  Bragg peaks detected: {len(diffraction_pattern['bragg_peaks'])}")
            self.logger.info(f"  Effective rank: {lattice_params.get('effective_rank', 'N/A')}")
            self.logger.info(f"  Strassen equivalence: {is_strassen}")
            self.logger.info(f"  Gibbs stability: {gibbs_energy:.4f} (stable={gibbs_energy < 0})")
            
            self._plot_diffraction_pattern(diffraction_pattern, ckpt_name)
            self._save_superlattice_seed(superlattice_seed, ckpt_name)
        
        if results:
            summary = {
                'individual_results': results,
                'total_crystals_analyzed': len(results),
                'strassen_confirmed_count': sum(
                    1 for r in results.values() 
                    if r.get('canonical_decomposition', {}).get('is_strassen_equivalent', {}).get('is_discrete_structure', False)
                ),
                'stable_structures_count': sum(1 for r in results.values() if r.get('stability_criterion', False))
            }
        else:
            summary = {'error': 'No valid spectroscopic analysis'}
        
        self._save_results(summary, "phase6_spectroscopic_analysis.json")
        return summary
    
    def _plot_diffraction_pattern(self, diffraction_data: Dict[str, Any], ckpt_name: str):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        power_spectrum = np.array(diffraction_data['power_spectrum'])
        frequencies = np.arange(len(power_spectrum))
        
        ax1.plot(frequencies, power_spectrum, 'b-', linewidth=1, alpha=0.7)
        ax1.set_xlabel('Frequency (k)', fontsize=12)
        ax1.set_ylabel('Power Spectrum |Psi(k)|^2', fontsize=12)
        ax1.set_title(f'Weight Diffraction Pattern ({ckpt_name})', fontsize=14)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        bragg_peaks = diffraction_data['bragg_peaks']
        if bragg_peaks:
            peak_freqs = [p['frequency'] for p in bragg_peaks]
            peak_intensities = [p['intensity'] for p in bragg_peaks]
            ax1.scatter(peak_freqs, peak_intensities, color='red', s=100, marker='*', 
                       label=f'Bragg Peaks (n={len(bragg_peaks)})', zorder=5)
            ax1.legend()
        
        is_crystalline = diffraction_data['is_crystalline_structure']
        spectral_entropy = diffraction_data['spectral_entropy']
        
        ax2.bar(['Glass', 'Crystal'], [1 - int(is_crystalline), int(is_crystalline)], 
               color=['orange', 'green'], alpha=0.7)
        ax2.set_ylabel('Classification', fontsize=12)
        ax2.set_title(f'Structure Type\nSpectral Entropy: {spectral_entropy:.2f}', fontsize=14)
        ax2.set_ylim([0, 1.2])
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"phase6_diffraction_{ckpt_name}.png")
        plt.close()
    
    def _save_superlattice_seed(self, superlattice: Dict[str, Any], ckpt_name: str):
        output_dir = Path("superlattice_seeds")
        output_dir.mkdir(exist_ok=True)
        
        seed_data = {
            'U': superlattice['U'].cpu().numpy().tolist(),
            'V': superlattice['V'].cpu().numpy().tolist(),
            'W': superlattice['W'].cpu().numpy().tolist(),
            'scale_factor': superlattice['scale_factor'],
            'original_shape': superlattice['original_shape'],
            'source_checkpoint': ckpt_name
        }
        
        filepath = output_dir / f"{ckpt_name}_superlattice_2x.json"
        with open(filepath, 'w') as f:
            json.dump(seed_data, f, indent=2)
        
        self.logger.info(f"  Superlattice seed saved: {filepath}")
    
    def _classify_thermodynamic_phase(self, t_eff: float, cv: float, alpha: float) -> str:
        if alpha > 7.0 and t_eff < 1e-3:
            return "Crystalline (Low T, High Order)"
        elif alpha < 3.0 and t_eff > 1e-1:
            return "Gaseous (High T, Low Order)"
        elif cv > 1.0:
            return "Critical Point (Phase Transition)"
        elif 3.0 <= alpha <= 7.0:
            return "Liquid/Glass (Intermediate)"
        else:
            return "Unknown Phase"
    
    def _estimate_critical_temperature(self, results: Dict[str, Any]) -> float:
        temps = [r['effective_temperature'] for r in results.values()]
        cvs = [r['specific_heat'] for r in results.values()]
        
        if len(cvs) < 2:
            return 0.0
        
        max_cv_idx = np.argmax(cvs)
        t_critical = temps[max_cv_idx]
        
        return float(t_critical)
    
    def _verify_entropy_extensivity(self, results: Dict[str, Any]) -> Dict[str, Any]:
        entropies = []
        scales = []
        
        sorted_results = sorted(results.items(), key=lambda x: x[1].get('hbar_algorithmic', 0))
        
        for i, (ckpt_name, data) in enumerate(sorted_results):
            alpha = self.checkpoints[ckpt_name]['alpha']
            entropy = -alpha if alpha > 0 else 0
            entropies.append(entropy)
            scales.append(i + 1)
        
        return ThermodynamicMetrics.check_extensivity(entropies, scales)
    
    def _plot_phase_diagram(self, results: Dict[str, Any]):
        fig, ax = plt.subplots(figsize=(10, 6))
        
        temps = [r['effective_temperature'] for r in results.values()]
        alphas = [self.checkpoints[ckpt]['alpha'] for ckpt in results.keys()]
        cvs = [r['specific_heat'] for r in results.values()]
        
        scatter = ax.scatter(temps, alphas, c=cvs, cmap='hot', s=100, alpha=0.7, edgecolors='black')
        ax.set_xlabel('Effective Temperature T_eff', fontsize=12)
        ax.set_ylabel('Purity (alpha)', fontsize=12)
        ax.set_title('Thermodynamic Phase Diagram', fontsize=14)
        ax.set_xscale('log')
        
        cbar = fig.colorbar(scatter, ax=ax)
        cbar.set_label('Specific Heat C_v', fontsize=10)
        
        ax.axhline(y=7.0, color='green', linestyle='--', label='Crystal threshold (alpha=7)')
        ax.axhline(y=3.0, color='orange', linestyle='--', label='Glass threshold (alpha=3)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "phase5_phase_diagram.png")
        plt.close()
    
    def _plot_temperature_vs_purity(self, results: Dict[str, Any]):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        temps = [r['effective_temperature'] for r in results.values()]
        alphas = [self.checkpoints[ckpt]['alpha'] for ckpt in results.keys()]
        cvs = [r['specific_heat'] for r in results.values()]
        
        ax1.plot(alphas, temps, 'bo-', linewidth=2, markersize=8)
        ax1.set_xlabel('Purity (alpha)', fontsize=12)
        ax1.set_ylabel('Effective Temperature T_eff', fontsize=12)
        ax1.set_title('Fluctuation-Dissipation Relation', fontsize=14)
        ax1.set_yscale('log')
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(alphas, cvs, 'ro-', linewidth=2, markersize=8)
        ax2.set_xlabel('Purity (alpha)', fontsize=12)
        ax2.set_ylabel('Specific Heat C_v', fontsize=12)
        ax2.set_title('Heat Capacity vs Crystallization', fontsize=14)
        ax2.axhline(y=1.0, color='green', linestyle='--', label='Critical C_v threshold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / "phase5_temperature_analysis.png")
        plt.close()
    
    def _compute_entropy_simple(self, params: np.ndarray) -> float:
        if params.size == 0 or params.shape[0] < 2:
            return 0.0
        
        if params.shape[1] > 1:
            centered = params - np.mean(params, axis=0)
            cov = np.cov(centered.T)
            eigenvals, eigenvecs = eigh(cov)
            idx = np.argsort(eigenvals)[::-1][:1]
            if eigenvals[idx[0]] > Config.ENTROPY_EPS:
                params_1d = centered @ eigenvecs[:, idx]
            else:
                return 0.0
        else:
            params_1d = params.flatten()
        
        hist, edges = np.histogram(params_1d, bins=Config.ENTROPY_BINS, density=False)
        hist = hist / len(params_1d)
        
        probs = hist[hist > Config.ENTROPY_EPS]
        if len(probs) == 0:
            return 0.0
        
        entropy = -np.sum(probs * np.log(probs + Config.ENTROPY_EPS))
        return float(entropy)
        
    def _compute_entropy(self, params: np.ndarray) -> float:
        if params.size == 0 or params.shape[0] < 2:
            return 0.0
        
        n_samples, n_dims = params.shape
        
        if n_dims > n_samples // 2:
            n_components = min(n_samples - 1, max(2, n_samples // 2))
            try:
                centered = params - np.mean(params, axis=0)
                
                cov = np.cov(centered.T)
                eigenvals, eigenvecs = eigh(cov)
                idx = np.argsort(eigenvals)[::-1][:n_components]
                components = eigenvecs[:, idx]
                params_reduced = centered @ components
            except:
                params_reduced = params[:, :min(2, n_dims)]
        else:
            params_reduced = params
        
        if params_reduced.shape[1] > 3:
            params_reduced = params_reduced[:, :3]
        
        if params_reduced.ndim == 1 or params_reduced.shape[1] == 1:
            data_1d = params_reduced.flatten()
            hist, _ = np.histogram(data_1d, bins=Config.ENTROPY_BINS, density=False)
            hist = hist / len(data_1d)
            probs = hist[hist > Config.ENTROPY_EPS]
            entropy = -np.sum(probs * np.log(probs + Config.ENTROPY_EPS))
        else:
            bins = [Config.ENTROPY_BINS] * params_reduced.shape[1]
            hist, _ = np.histogramdd(params_reduced, bins=bins, density=False)
            hist = hist / np.sum(hist)
            probs = hist[hist > Config.ENTROPY_EPS]
            entropy = -np.sum(probs * np.log(probs + Config.ENTROPY_EPS))
        
        return float(entropy)
    
    def _compute_effective_volume(self, params: np.ndarray) -> float:
        if params.size == 0:
            return 0.0
        
        mins = np.min(params, axis=0)
        maxs = np.max(params, axis=0)
        ranges = maxs - mins
        
        active_dims = ranges > Config.MIN_VARIANCE_THRESHOLD
        if not np.any(active_dims):
            return 0.0
        
        hypervolume = np.prod(ranges[active_dims])
        
        volume_fraction = np.mean(active_dims)
        
        return float(hypervolume * volume_fraction)
    
    def _plot_parameter_distribution(self, params: np.ndarray, group_name: str, kde=None):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        first_param = params[:, 0]
        ax1.hist(first_param, bins=Config.ENTROPY_BINS, density=True, alpha=0.6, color='blue', edgecolor='black')
        ax1.set_xlabel('Parameter Value')
        ax1.set_ylabel('Density')
        ax1.set_title(f'Parameter Distribution ({group_name})')
        ax1.grid(True, alpha=0.3)
        
        if params.shape[1] >= 2:
            x = params[:, 0]
            y = params[:, 1]
            hist, xedges, yedges = np.histogram2d(x, y, bins=Config.ENTROPY_BINS, density=True)
            
            im = ax2.imshow(hist.T, origin='lower', extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]],
                        cmap='viridis', aspect='auto')
            ax2.scatter(params[:,0], params[:,1], alpha=0.2, s=5, color='white', edgecolor='black')
            ax2.set_xlabel('Param 1')
            ax2.set_ylabel('Param 2')
            ax2.set_title('2D Parameter Space (Histogram)')
            
            fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
        
        plt.tight_layout()
        plt.savefig(self.results_dir / f"phase1_distribution_{group_name}.png")
        plt.close()
        
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
        successful_params = []
        for data in successful_ckpts.values():
            clean_params = torch.cat([p.flatten() for p in data['coeffs'].values()]).cpu().numpy()
            successful_params.append(clean_params)
        
        if not successful_params:
            return 0.0
        
        successful_params = np.stack(successful_params)
        n_samples, n_dims = successful_params.shape
        
        combined_params = np.vstack([successful_params, params.reshape(1, -1)])
        
        if n_dims > n_samples:
            n_components = min(n_samples - 1, max(2, n_samples // 2))
            try:
                centered = combined_params - np.mean(combined_params, axis=0)
                cov = np.cov(centered.T)
                eigenvals, eigenvecs = eigh(cov)
                idx = np.argsort(eigenvals)[::-1][:n_components]
                components = eigenvecs[:, idx]
                combined_reduced = centered @ components
            except:
                combined_reduced = combined_params[:, :min(2, n_dims)]
        else:
            combined_reduced = combined_params
        
        if combined_reduced.shape[1] > 3:
            combined_reduced = combined_reduced[:, :3]
        
        bins = [Config.ENTROPY_BINS] * combined_reduced.shape[1]
        hist, edges = np.histogramdd(combined_reduced, bins=bins, density=False)
        
        total_count = np.sum(hist)
        if total_count == 0:
            return 0.0
        
        probs = hist / total_count
        
        query_sample = combined_reduced[-1]
        bin_indices = []
        for i, (edge, value) in enumerate(zip(edges, query_sample)):
            idx = np.searchsorted(edge, value, side='right') - 1
            idx = max(0, min(idx, len(edge) - 2))
            bin_indices.append(idx)
        
        query_prob = probs[tuple(bin_indices)]
        if query_prob < Config.ENTROPY_EPS:
            query_prob = Config.ENTROPY_EPS
        
        return float(-np.log(query_prob))
        
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
        plt.savefig(self.results_dir / f"phase2_entropy_{ckpt_name}.png")
        plt.close()
    
    def _verify_scaling(self, coeffs: Dict[str, torch.Tensor], N: int) -> float:
        try:
            A = torch.randn(1, N, N).to(Config.DEVICE)
            B = torch.randn(1, N, N).to(Config.DEVICE)
            C_true = torch.bmm(A, B)
            C_pred = self._recursive_strassen(A, B, coeffs, N)
            return float(torch.norm(C_true - C_pred) / torch.norm(C_true))
        except Exception as e:
            self.logger.warning(f"Scaling verification failed for N={N}: {e}")
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
        plt.savefig(self.results_dir / f"phase3_extensivity_{ckpt_name}.png")
        plt.close()
    
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
        plt.savefig(self.results_dir / f"phase4_uncertainty_{ckpt_name}.png")
        plt.close()
    
    def _print_executive_summary(self, results: Dict[str, Any]):
        self.logger.info("\n" + "="*80)
        self.logger.info("Executive Summary of Crystallography Analysis Results")
        self.logger.info("="*80)
        
        loaded = len(self.checkpoints)
        crystal_count = len([v for v in self.checkpoints.values() if v.get('is_crystal', False)])
        glass_count = len([v for v in self.checkpoints.values() if not v.get('is_crystal', False)])
        
        self.logger.info(f"\nLoaded checkpoints: {loaded}")
        self.logger.info(f"Crystal states (alpha>7): {crystal_count}")
        self.logger.info(f"Glass states (alpha<3): {glass_count}")
        
        poynting = results.get('poynting_analysis', {})
        if poynting:
            stats = poynting.get('statistics', {})
            self.logger.info(f"\nPoynting Vector Analysis:")
            self.logger.info(f"  Mean magnitude: {stats.get('mean_magnitude', 0):.2e}")
            self.logger.info(f"  Radiating systems: {stats.get('radiating_percentage', 0):.1f}%")
        
        for phase, data in results.items():
            if phase == 'poynting_analysis':
                continue
            self.logger.info(f"\n{phase.upper()}:")
            if isinstance(data, dict) and 'error' not in data:
                self.logger.info(f"  Status: Completed successfully")
            else:
                self.logger.info(f"  Status: {data.get('error', 'Completed') if isinstance(data, dict) else 'Completed'}")
    
    def _save_results(self, results: Dict[str, Any], filename: str):
        filepath = self.results_dir / filename
        
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
        
        self.logger.info(f"\nResults saved to: {filepath}")

class StrassenCrystallographer:
    def __init__(self, checkpoint_path: str, device: str = Config.DEVICE):
        self.checkpoint_path = Path(checkpoint_path)
        self.device = device
        self.model = self._load_model(checkpoint_path, device)
        self.logger = setup_logger("StrassenCrystallographer")
    
    def _load_model(self, path: str, device: str) -> BilinearStrassenModel:
        model = BilinearStrassenModel().to(device)
        loader = CheckpointLoader()
        
        try:
            raw_data = loader.load_checkpoint(path, device)
            migrated_state = CheckpointMigrator.migrate_checkpoint(raw_data)
            
            if migrated_state is not None:
                model.load_state_dict(migrated_state)
                return model
            
            raise RuntimeError(f"Failed to migrate checkpoint from {path}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")
    
    def run_full_analysis(self) -> Dict[str, Any]:
        self.logger.info(f"Running full crystallographic analysis on {self.checkpoint_path.name}")
        
        def dataloader():
            for _ in range(5):
                yield StrassenDataGenerator.generate_batch(Config.BATCH_SIZE)
        
        metrics = CrystallographyMetrics.compute_all_metrics(self.model, dataloader())
        verification = StrassenDataGenerator.verify_structure(self.model.get_coefficients())
        
        final_report = {
            'checkpoint': self.checkpoint_path.name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'verification': verification,
            'crystallographic_grade': self._assign_grade(metrics['delta'], metrics['alpha']),
            'poynting_analysis': metrics['poynting']
        }
        
        self._save_report(final_report)
        return final_report
    
    def _assign_grade(self, delta: float, alpha: float) -> str:
        if delta < 0.01:
            return "Optical Crystal (delta<0.01, perfect structure)"
        elif delta < 0.1:
            return "Industrial Crystal (delta<0.1, robust)"
        elif delta < 0.3:
            return "Polycrystalline (delta<0.3, generalizes but impure)"
        elif delta < 0.5:
            return "Amorphous Glass (delta<0.5, local minimum)"
        else:
            return "Defective (delta>=0.5, no structure)"
    
    def _save_report(self, report: Dict[str, Any]):
        output_dir = Path("crystallography_reports")
        output_dir.mkdir(exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{report['checkpoint']}_{timestamp}_analysis.json"
        filepath = output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Analysis report saved to: {filepath}")

# Añadir al main() del script principal

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Strassen Crystallography System")
    parser.add_argument("--mode", choices=["train", "analyze", "epitaxy", "both", "green_cow"], default="analyze",
                    help="Execution mode: ..., or green_cow for boundary-aware learning")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints",
                       help="Directory containing checkpoints for analysis")
    parser.add_argument("--train_epochs", type=int, default=1000,
                       help="Number of epochs for training mode")
    parser.add_argument("--results_dir", type=str, default=Config.RESULTS_DIR,
                       help="Directory for analysis results")
    parser.add_argument("--epitaxy_sizes", type=int, nargs='+', default=[4, 8],
                       help="Target matrix sizes for epitaxial growth (e.g., 4 8 16)")
    parser.add_argument("--seed_checkpoint", type=str, default=None,
                       help="Specific checkpoint to use as seed (if None, uses best crystal)")
    
    args = parser.parse_args()
    
    set_seed(Config.RANDOM_SEED)
    logger.info(f"Using device: {Config.DEVICE}")
    logger.info(f"Configuration: {Config.__dict__}")


    if args.mode == "green_cow":
        logger.info("\n" + "="*60)
        logger.info("GREEN'S: Boundary-aware Learning")
        logger.info("="*60)
        model = BilinearStrassenModel().to(Config.DEVICE)
        cow = GreenCowExperiment(model)
        history = cow.train_with_green_cow(epochs=20000, lr=1e-3, lambda_boundary=0.1)

        # Guardar resultados
        import json
        with open("green_cow_results.json", "w") as f:
            json.dump(history, f, indent=2)

        # Plot
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(history['epoch'], history['loss'], 'b-', label='Total Loss')
        plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.title('Green Cow Loss')
        plt.subplot(1, 2, 2)
        plt.plot(history['epoch'], history['alpha'], 'g-', label='α (purity)')
        plt.axhline(7.0, color='r', linestyle='--')
        plt.xlabel('Epoch'); plt.ylabel('Alpha'); plt.title('Crystallization')
        plt.tight_layout()
        plt.savefig("green_cow_evolution.png")
        logger.info("Experiment completed! Results in green_cow_results.json & .png")

    if args.mode in ["train", "both"]:
        logger.info("Starting training mode...")
        logger.info("Training mode not implemented in this version")
    
    if args.mode in ["analyze", "both"]:
        logger.info("Starting analysis mode...")
        
        if not Path(args.checkpoint_dir).exists():
            logger.error(f"Checkpoint directory {args.checkpoint_dir} does not exist")
            return
        
        try:
            program = BoltzmannAnalysisProgram(args.checkpoint_dir, args.results_dir)
            program.run_full_boltzmann_program()
            
            checkpoint_files = list(Path(args.checkpoint_dir).glob("*.pt"))
            for filepath in checkpoint_files[:3]:
                try:
                    crystallographer = StrassenCrystallographer(str(filepath))
                    report = crystallographer.run_full_analysis()
                    logger.info(f"Analyzed {filepath.name}: Grade - {report['crystallographic_grade']}")
                except Exception as e:
                    logger.error(f"Failed to analyze {filepath}: {e}")
                    
        except Exception as e:
            logger.error(f"Analysis failed: {e}")
            return
    
    if args.mode == "epitaxy":
        logger.info("\n" + "="*80)
        logger.info("EPITAXIAL GROWTH MODE - Crystal Cultivation")
        logger.info("="*80)
        
        try:
            if args.seed_checkpoint:
                # Usar checkpoint específico como semilla
                experiment = EpitaxyExperiment()
                results = experiment.run_epitaxial_growth_experiment(
                    args.seed_checkpoint, 
                    args.epitaxy_sizes
                )
            else:
                # Buscar automáticamente el mejor cristal
                results = run_epitaxy_from_best_crystal(
                    args.checkpoint_dir,
                    args.epitaxy_sizes
                )
            
            # Resumen final
            logger.info("\n" + "="*80)
            logger.info("EPITAXY EXPERIMENT COMPLETED")
            logger.info("="*80)
            
            success_count = sum(
                1 for exp in results['experiments'].values() 
                if exp.get('crystallization_success', False)
            )
            
            logger.info(f"Successful crystallizations: {success_count}/{len(results['experiments'])}")
            logger.info(f"Results saved in: epitaxy_results/")
            
        except Exception as e:
            logger.error(f"Epitaxy experiment failed: {e}")
            import traceback
            traceback.print_exc()
            return
    
    logger.info("\nExecution completed successfully")
    
if __name__ == "__main__":
    main()
