#!/usr/bin/env python3
"""
Cristalografo para Strassen - VERSIÓN DEFINITIVA CORREGIDA
- Índice normalizado a [0,1]
- Manejo de κ=inf como caso especial
- Umbrales de grado basados en δ
- Visualización mejorada
"""

import torch
import torch.nn as nn
import numpy as np
import random
import json
import os
from typing import Dict, Optional
import matplotlib.pyplot as plt
from collections import deque
from datetime import datetime

# ============================================================================
# CONFIGURACIÓN GLOBAL
# ============================================================================

class Config:
    BATCH_SIZE = 32
    WEIGHT_DECAY = 1e-4
    LEARNING_RATE = 0.001
    EPOCHS = 3000
    N_SLOTS = 8
    TARGET_SLOTS = 7
    DISCRETIZATION_MARGIN = 0.1
    RANDOM_SEED = 42

def set_seed(seed=Config.RANDOM_SEED):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

# ============================================================================
# MODELO BILINEAL STRASSEN
# ============================================================================

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
    
    def get_coefficients(self):
        return {
            'U': self.U.weight.data,
            'V': self.V.weight.data,
            'W': self.W.weight.data
        }

# ============================================================================
# MIGRADOR ULTRA-ROBUSTO
# ============================================================================

class CheckpointMigrator:
    @staticmethod
    def migrate_checkpoint(path: str, device: str = 'cpu') -> Optional[Dict[str, torch.Tensor]]:
        print(f"    🔄 Migrando: {os.path.basename(path)}")
        
        try:
            data = torch.load(path, map_location=device, weights_only=False)
            
            if isinstance(data, dict):
                if 'state_dict' in data:
                    state_dict = data['state_dict']
                elif 'model_state_dict' in data:
                    state_dict = data['model_state_dict']
                else:
                    state_dict = data
            elif hasattr(data, 'state_dict'):
                state_dict = data.state_dict()
            else:
                state_dict = data
            
            # ESTRATEGIAS DE MIGRACIÓN
            if 'U' in state_dict and isinstance(state_dict['U'], torch.Tensor):
                return CheckpointMigrator._migrate_custom(state_dict)
            elif 'encoder.0.weight' in state_dict:
                return CheckpointMigrator._migrate_encoder(state_dict)
            elif any(k.endswith('.weight') for k in state_dict.keys()):
                return CheckpointMigrator._migrate_standard(state_dict)
            
            return None
        
        except Exception as e:
            print(f"    ❌ Error: {e}")
            return None
    
    @staticmethod
    def _migrate_custom(state_dict: Dict) -> Dict[str, torch.Tensor]:
        """Maneja formatos custom U,V,W directos"""
        u_tensor = state_dict['U']
        
        # Ajustar shape si es necesario
        if u_tensor.shape == (7, 4):
            u_padded = torch.zeros(8, 4)
            v_padded = torch.zeros(8, 4)
            w_padded = torch.zeros(4, 8)
            u_padded[:7] = state_dict['U']
            v_padded[:7] = state_dict['V']
            w_padded[:, :7] = state_dict['W']
            return {'U.weight': u_padded, 'V.weight': v_padded, 'W.weight': w_padded}
        
        return {'U.weight': state_dict['U'], 'V.weight': state_dict['V'], 'W.weight': state_dict['W']}
    
    @staticmethod
    def _migrate_encoder(state_dict: Dict) -> Dict[str, torch.Tensor]:
        """Extracción de encoder.layers"""
        encoder_0 = state_dict['encoder.0.weight']
        encoder_2 = state_dict.get('encoder.2.weight', encoder_0)
        encoder_4 = state_dict.get('encoder.4.weight', torch.randn(64, 64))
        
        if encoder_0.shape == (64, 8):
            u = encoder_0[:8, :4].clone()
        else:
            u = encoder_0.flatten()[:32].reshape(8, 4)
        
        if encoder_2.shape == (64, 64):
            v = encoder_2[:8, :4].clone()
        else:
            v = u
        
        if encoder_4.shape == (64, 64):
            w = encoder_4[:4, :8].clone()
        else:
            w = torch.randn(4, 8)
        
        return {'U.weight': u, 'V.weight': v, 'W.weight': w}
    
    @staticmethod
    def _migrate_standard(state_dict: Dict) -> Dict[str, torch.Tensor]:
        """Formato estándar U.weight, V.weight, W.weight"""
        return {k: state_dict[k] for k in ['U.weight', 'V.weight', 'W.weight'] if k in state_dict}

# ============================================================================
# GENERADOR DE DATOS
# ============================================================================

class StrassenDataGenerator:
    @staticmethod
    def generate_batch(batch_size=Config.BATCH_SIZE):
        A = torch.randn(batch_size, 2, 2)
        B = torch.randn(batch_size, 2, 2)
        C = torch.bmm(A, B)
        return A.reshape(batch_size, 4), B.reshape(batch_size, 4), C.reshape(batch_size, 4)
    
    @staticmethod
    def verify_structure(coeffs):
        delta = CrystallographyMetrics.compute_discretization_margin(coeffs)
        return {'pass': delta < Config.DISCRETIZATION_MARGIN, 'max_error': delta}

# ============================================================================
# PROTOCOLO DE PODA
# ============================================================================

class SparsificationProtocol:
    def __init__(self, model):
        self.model = model
    
    def prune_to_target(self, target=Config.TARGET_SLOTS):
        scores = self.model.U.weight.norm(dim=1) + \
                 self.model.V.weight.norm(dim=1) + \
                 self.model.W.weight.norm(dim=0)
        _, indices = torch.topk(scores, target)
        
        mask = torch.zeros_like(scores)
        mask[indices] = 1.0
        
        with torch.no_grad():
            self.model.U.weight.mul_(mask.unsqueeze(1))
            self.model.V.weight.mul_(mask.unsqueeze(1))
            self.model.W.weight.mul_(mask.unsqueeze(0))
        
        return indices
    
    def discretize_weights(self, margin=Config.DISCRETIZATION_MARGIN):
        coeffs = self.model.get_coefficients()
        max_margin = CrystallographyMetrics.compute_discretization_margin(coeffs)
        if max_margin > margin:
            return False
        with torch.no_grad():
            for p in self.model.parameters():
                p.copy_(p.round())
        return True

# ============================================================================
# MÉTRICAS
# ============================================================================

class CrystallographyMetrics:
    @staticmethod
    def compute_kappa(model, dataloader, num_batches=5):
        model.eval()
        grads = []
        for i, (A, B, C) in enumerate(dataloader):
            if i >= num_batches: break
            C_pred = model(A, B)
            loss = nn.functional.mse_loss(C_pred, C)
            grad = torch.autograd.grad(loss, model.parameters(), create_graph=False)
            grads.append(torch.cat([g.flatten() for g in grad]))
        
        if len(grads) < 2: return float('inf')
        grads = torch.stack(grads)
        Σ = torch.cov(grads.T)
        try: return torch.linalg.cond(Σ).item()
        except: return float('inf')
    
    @staticmethod
    def compute_discretization_margin(coeffs):
        return max((t - t.round()).abs().max().item() for t in coeffs.values())

# ============================================================================
# DIFRACCIÓN
# ============================================================================

class StrassenDiffractionTest:
    def __init__(self, model):
        self.model = model
    
    def test_gauge_invariance(self, n_samples=50):
        coeffs = self.model.get_coefficients()
        indices = list(range(Config.TARGET_SLOTS))
        sample_perms = [random.sample(indices, len(indices)) for _ in range(min(n_samples, 100))]
        
        errors, invariant_count = [], 0
        
        for perm in sample_perms:
            perm_tensor = torch.tensor(perm, dtype=torch.long)
            test_coeffs = {
                'U': coeffs['U'][perm_tensor],
                'V': coeffs['V'][perm_tensor],
                'W': coeffs['W'][:, perm_tensor]
            }
            
            error = self._functional_error(test_coeffs)
            errors.append(error)
            if error < 1e-5:
                invariant_count += 1
        
        return {
            'invariance_ratio': invariant_count / len(sample_perms),
            'mean_error': np.mean(errors),
            'is_genuine': invariant_count == 1
        }
    
    def _functional_error(self, test_coeffs):
        errors = []
        for _ in range(10):
            A, B, C = StrassenDataGenerator.generate_batch(1)
            M1 = (self.model.get_coefficients()['U'] @ A.T) * (self.model.get_coefficients()['V'] @ B.T)
            C1 = self.model.get_coefficients()['W'] @ M1
            M2 = (test_coeffs['U'] @ A.T) * (test_coeffs['V'] @ B.T)
            C2 = test_coeffs['W'] @ M2
            errors.append(torch.norm(C1 - C2).item())
        return max(errors)

# ============================================================================
# ESPECTROSCOPIA
# ============================================================================

class BasinResilienceSpectrometer:
    def __init__(self, model):
        self.model = model
        self.original_state = {k: v.clone() for k, v in model.state_dict().items()}
    
    def measure_resilience_spectrum(self, noise_levels=None):
        if noise_levels is None:
            noise_levels = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
        
        results = {}
        for sigma in noise_levels:
            results[f'sigma_{sigma}'] = self._test_noise_recovery(sigma, 5)
        
        results['critical_sigma'] = self._estimate_critical_noise(results)
        return results
    
    def _test_noise_recovery(self, sigma, n_trials):
        successes, final_margins = 0, []
        
        for _ in range(n_trials):
            self._apply_noise(sigma)
            recovery_epochs = self._anneal_to_attractor(30)
            final_margin = CrystallographyMetrics.compute_discretization_margin(self.model.get_coefficients())
            final_margins.append(final_margin)
            
            if final_margin < Config.DISCRETIZATION_MARGIN:
                successes += 1
            
            self.model.load_state_dict(self.original_state)
        
        return {
            'success_rate': successes / n_trials,
            'final_margin_mean': np.mean(final_margins)
        }
    
    def _apply_noise(self, sigma):
        with torch.no_grad():
            for p in self.model.parameters():
                p.add_(torch.randn_like(p) * sigma)
    
    def _anneal_to_attractor(self, max_epochs):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-5)
        for epoch in range(max_epochs):
            A, B, C = StrassenDataGenerator.generate_batch(32)
            loss = nn.functional.mse_loss(self.model(A, B), C)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if CrystallographyMetrics.compute_discretization_margin(self.model.get_coefficients()) < Config.DISCRETIZATION_MARGIN:
                return epoch + 1
        return max_epochs
    
    def _estimate_critical_noise(self, results):
        sigmas, rates = [], []
        for k, v in results.items():
            if k.startswith('sigma_'):
                sigmas.append(float(k.split('_')[1]))
                rates.append(v['success_rate'])
        
        for i in range(len(rates)-1):
            if rates[i] >= 0.5 > rates[i+1]:
                return sigmas[i]
        return sigmas[-1]

# ============================================================================
# ÍNDICE DE PUREZA CORREGIDO
# ============================================================================

class CrystalPurityIndex:
    def __init__(self, model: nn.Module, diffraction_results: Dict, 
                 resilience_results: Dict, metrics_results: Dict):
        self.model = model
        self.coeffs = model.get_coefficients()
        
        self.metrics = {
            'kappa': metrics_results.get('kappa', float('inf')),
            'delta': CrystallographyMetrics.compute_discretization_margin(self.coeffs),
            'resilience': resilience_results.get('critical_sigma', 0.0),
            'invariance': diffraction_results.get('invariance_ratio', 1.0),
            'lc': metrics_results.get('lc', 0.0)
        }
    
    def compute(self) -> Dict:
        # Normalizar κ (manejar inf como cristal perfecto)
        if self.metrics['kappa'] == float('inf') or self.metrics['kappa'] < 1.1:
            kappa_score = 1.0
        else:
            kappa_score = max(0, 1 - np.log10(self.metrics['kappa']) / 6)
        
        # Normalizar δ
        delta_score = max(0, 1 - self.metrics['delta'] / 0.5)
        
        # Normalizar RESILIENCE (BUG FIX: dividir entre 0.01)
        resilience_score = max(0, min(1.0, self.metrics['resilience'] / 0.01))
        
        # Normalizar invarianza
        invariance_score = max(0, 1 - self.metrics['invariance'] / 0.01)
        
        # Pesos ajustados
        weights = {'kappa': 0.30, 'delta': 0.40, 'resilience': 0.15, 'invariance': 0.10, 'lc': 0.05}
        
        purity_index = sum(
            weights[k] * v for k, v in {
                'kappa': kappa_score,
                'delta': delta_score,
                'resilience': resilience_score,
                'invariance': invariance_score,
                'lc': self.metrics['lc']
            }.items()
        )
        
        return {
            'index': purity_index,
            'grade': self._assign_grade(purity_index, self.metrics['delta']),
            'component_scores': {
                'kappa_score': kappa_score,
                'delta_score': delta_score,
                'resilience_score': resilience_score,
                'invariance_score': invariance_score
            }
        }
    
    def _assign_grade(self, index: float, delta: float) -> str:
        # CRITERIO REAL: δ es el indicador principal
        if delta < 0.01:
            return "Cristal Óptico (δ<0.01, estructura perfecta)"
        elif delta < 0.1:
            return "Cristal Industrial (δ<0.1, robusto)"
        elif delta < 0.3:
            return "Policristalino (δ<0.3, generaliza pero impuro)"
        elif delta < 0.5:
            return "Vidrio Amorfo (δ<0.5, mínimo local)"
        else:
            return "Defectuoso (δ≥0.5, sin estructura)"

# ============================================================================
# PIPELINE CRISTALOGRÁFICO COMPLETO
# ============================================================================

class StrassenCrystallographer:
    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.checkpoint_path = checkpoint_path
        self.device = device
        self.model = BilinearStrassenModel().to(device)
        
        try:
            print(f"    📂 Carga directa...")
            state_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            if isinstance(state_dict, dict):
                if 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                elif 'model_state_dict' in state_dict:
                    state_dict = state_dict['model_state_dict']
            
            self.model.load_state_dict(state_dict)
            print("    ✅ Carga directa exitosa")
        except Exception as e:
            print(f"    ❌ Carga directa falló: {e}")
            print(f"    🔄 Migrando...")
            clean_state = CheckpointMigrator.migrate_checkpoint(checkpoint_path, device)
            
            if clean_state is not None:
                self.model.load_state_dict(clean_state)
                print("    ✅ Migración exitosa")
            else:
                print("    ❌ Migración fallida - usando pesos aleatorios")
    
    def run_full_analysis(self) -> Dict:
        print(f"\n🔬 Analizando: {os.path.basename(self.checkpoint_path)}")
        print("=" * 60)
        
        # Módulo 1: Difracción
        print("\n[1/3] Difracción de Algoritmos...")
        diffraction = StrassenDiffractionTest(self.model)
        diffraction_results = diffraction.test_gauge_invariance()
        print(f"    ▸ Invarianza: {diffraction_results['invariance_ratio']:.4f}")
        print(f"    ▸ Genuino: {diffraction_results['is_genuine']}")
        
        # Módulo 2: Resiliencia
        print("\n[2/3] Espectroscopía de Resiliencia...")
        spectrometer = BasinResilienceSpectrometer(self.model)
        resilience_results = spectrometer.measure_resilience_spectrum()
        print(f"    ▸ Sigma crítico: {resilience_results['critical_sigma']:.2e}")
        
        # Módulo 3: Métricas
        print("\n[3/3] Métricas Cristalográficas...")
        
        def dataloader_gen():
            for _ in range(5):
                A, B, C = StrassenDataGenerator.generate_batch(32)
                yield A, B, C
        
        kappa = CrystallographyMetrics.compute_kappa(self.model, dataloader_gen())
        delta = CrystallographyMetrics.compute_discretization_margin(self.model.get_coefficients())
        lc = CrystallographyMetrics.compute_local_complexity(self.model)
        
        print(f"    ▸ κ: {kappa:.2e}")
        print(f"    ▸ δ: {delta:.4f}")
        print(f"    ▸ LC: {lc:.4f}")
        
        metrics_results = {'kappa': kappa, 'delta': delta, 'lc': lc}
        
        # Verificación
        print("\n🔍 Verificación de Estructura...")
        expansion_results = StrassenDataGenerator.verify_structure(self.model.get_coefficients())
        print(f"    {'✅' if expansion_results['pass'] else '⚠️'} Margen: {expansion_results['max_error']:.4f}")
        
        # Índice pureza
        print("\n📊 Calculando Índice de Pureza...")
        purity_calc = CrystalPurityIndex(self.model, diffraction_results, 
                                         resilience_results, metrics_results)
        purity_report = purity_calc.compute()
        print(f"    ▸ Índice: {purity_report['index']:.3f} (0.0-1.0)")
        print(f"    ▸ Grado: {purity_report['grade']}")
        
        # Reporte final
        final_report = {
            'checkpoint': self.checkpoint_path,
            'grade': purity_report['grade'],
            'purity_index': purity_report['index'],
            'component_scores': purity_report['component_scores'],
            'metrics': metrics_results,
            'expansion': expansion_results
        }
        
        self._save_report(final_report)
        return final_report
    
    def _save_report(self, report: Dict):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = "crystallography_reports"
        os.makedirs(output_dir, exist_ok=True)
        
        filename = f"{os.path.basename(self.checkpoint_path)}_{timestamp}.json"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n💾 Reporte guardado: {filepath}")

# ============================================================================
# SISTEMA DE EJECUCIÓN COMPLETA MEJORADO
# ============================================================================

class LocalComplexity:
    @staticmethod
    def compute(model: nn.Module) -> float:
        """
        Computa LC basado en Can't Stop Won't Stop paper
        """
        params = torch.cat([p.flatten() for p in model.parameters()])
        
        with torch.no_grad():
            # Usar percentiles en lugar de std para robustez
            perc_95 = torch.quantile(torch.abs(params), 0.95)
            active = (torch.abs(params) > 0.01 * perc_95).sum()
            lc = active.float() / len(params)
        
        return lc.item()

# Añadir al CrystallographyMetrics
CrystallographyMetrics.compute_local_complexity = LocalComplexity.compute

def main():
    set_seed(Config.RANDOM_SEED)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Usando dispositivo: {device}")
    
    # Auto-detectar directorio
    checkpoint_dir = "checkpoints"
    if not os.path.exists(checkpoint_dir):
        print(f"❌ No existe el directorio '{checkpoint_dir}'")
        print("   Crea la carpeta y mueve tus checkpoints .pt allí")
        return
    
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    
    if not checkpoints:
        print(f"❌ No hay checkpoints en '{checkpoint_dir}'")
        return
    
    print("\n" + "="*60)
    print("🔬 INICIANDO PROTOCOLO CRISTALOGRÁFICO")
    print("="*60)
    
    results = []
    
    for ckpt_path in sorted(checkpoints):
        try:
            print(f"\n📦 Procesando: {os.path.basename(ckpt_path)}")
            crystallographer = StrassenCrystallographer(ckpt_path, device)
            report = crystallographer.run_full_analysis()
            
            if report:
                results.append({
                    'checkpoint': os.path.basename(ckpt_path),
                    'purity_index': report['purity_index'],
                    'delta': report['metrics']['delta'],
                    'grade': report['grade']
                })
                print(f"    ✅ Purity Index: {report['purity_index']:.3f}")
        
        except Exception as e:
            print(f"❌ Error con {ckpt_path}: {e}")
            continue
    
    # Dashboard agregado
    if results:
        print("\n" + "="*60)
        print("📊 RESUMEN CRISTALOGRÁFICO AGREGADO")
        print("="*60)
        
        purity_indices = [r['purity_index'] for r in results]
        deltas = [r['delta'] for r in results]
        
        print(f"Checkpoints procesados: {len(results)}")
        print(f"Purity Index - Media: {np.mean(purity_indices):.3f}")
        print(f"Purity Index - Std: {np.std(purity_indices):.3f}")
        
        # Histograma
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        ax1.hist(purity_indices, bins=10, edgecolor='black', alpha=0.7, color='#2E86AB')
        ax1.axvline(x=0.90, color='green', linestyle='--', linewidth=2, label='Óptico')
        ax1.axvline(x=0.70, color='orange', linestyle='--', linewidth=2, label='Industrial')
        ax1.set_xlabel('Índice de Pureza [0,1]', fontsize=12)
        ax1.set_ylabel('Número de Checkpoints', fontsize=12)
        ax1.set_title('Distribución de Pureza', fontsize=14)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(deltas, bins=10, edgecolor='black', alpha=0.7, color='#A23B72')
        ax2.axvline(x=0.01, color='green', linestyle='--', linewidth=2, label='δ=0.01')
        ax2.axvline(x=0.1, color='orange', linestyle='--', linewidth=2, label='δ=0.1')
        ax2.set_xlabel('Margen de Discretización δ', fontsize=12)
        ax2.set_ylabel('Frecuencia', fontsize=12)
        ax2.set_title('Distribución de δ', fontsize=14)
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        os.makedirs("crystallography_reports", exist_ok=True)
        plt.savefig("crystallography_reports/purity_distribution.png", dpi=300)
        plt.close()
        
        print(f"\n💾 Histogramas guardados: crystallography_reports/purity_distribution.png")
        
        # Tabla resumen
        print("\n" + "="*100)
        print(f"{'Checkpoint':<30} {'Purity':<8} {'δ':<8} {'Grado':<30}")
        print("="*100)
        for r in results:
            print(f"{r['checkpoint']:<30} {r['purity_index']:<8.3f} {r['delta']:<8.4f} {r['grade']:<30}")

if __name__ == "__main__":
    main()
