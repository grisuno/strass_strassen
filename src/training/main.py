#!/usr/bin/env python3
"""
STRASSEN DISCOVERY: Descubrimiento de algoritmo de 7 multiplicaciones
======================================================================
Framework GOGF para descubrir algoritmos equivalentes a Strassen mediante
enmascaramiento progresivo de slots.

Uso:
    python main.py

Resultados:
    - 7 slots activos (1 eliminado)
    - Loss ~0 (precisión perfecta)
    - 100% accuracy
"""

import os
import sys
import time
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

# Directorio base (donde está este script)
BASE_DIR = Path(__file__).parent.resolve()

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Configuración del experimento."""
    SEED: int = 42
    DEVICE: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")
    
    # Arquitectura
    NUM_SLOTS: int = 8      # Slots iniciales
    TARGET_SLOTS: int = 7   # Strassen usa 7 multiplicaciones
    
    # Training - Fase 1 (Fitting con 8 slots)
    PHASE1_EPOCHS: int = 2000
    PHASE1_LR: float = 0.01
    PHASE1_WD: float = 0.001
    
    # Training - Fase 2 (Refinamiento con 7 slots)
    PHASE2_EPOCHS: int = 5000
    PHASE2_LR: float = 0.005
    PHASE2_WD: float = 0.01
    
    # Datos
    BATCH_SIZE: int = 256
    TRAIN_SAMPLES: int = 2000
    TEST_SAMPLES: int = 500
    
    # Métricas
    ACCURACY_THRESHOLD: float = 0.1
    
    # Logging
    LOG_INTERVAL: int = 200
    
    # Paths (relativos al directorio base)
    OUTPUT_DIR: Path = field(default_factory=lambda: BASE_DIR / "output")
    MODEL_PATH: Path = field(default_factory=lambda: BASE_DIR / "output" / "strassen_discovered.pt")


def set_seed(seed: int):
    """Fijar semilla para reproducibilidad."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class StrassenDiscovery(nn.Module):
    """
    Modelo para descubrir Strassen mediante coeficientes aprendibles.
    
    Arquitectura:
    - U_coefs[i]: Combinación de bloques de A para producto M_i
    - V_coefs[i]: Combinación de bloques de B para producto M_i  
    - W_coefs[j,i]: Contribución de M_i al bloque C_j del resultado
    """
    def __init__(self, num_slots: int = 8):
        super().__init__()
        self.num_slots = num_slots
        
        # Coeficientes aprendibles
        self.U_coefs = nn.Parameter(torch.randn(num_slots, 4) * 0.5)
        self.V_coefs = nn.Parameter(torch.randn(num_slots, 4) * 0.5)
        self.W_coefs = nn.Parameter(torch.randn(4, num_slots) * 0.5)
        
        # Máscara de slots
        self.register_buffer('slot_mask', torch.ones(num_slots))
    
    def forward(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
        """
        Forward pass con multiplicación matemática pura.
        A, B: (batch, 4, 4) -> C: (batch, 4, 4)
        """
        batch = A.size(0)
        device = A.device
        
        # Extraer bloques 2x2
        A11 = A[:, :2, :2].reshape(batch, 4)
        A12 = A[:, :2, 2:].reshape(batch, 4)
        A21 = A[:, 2:, :2].reshape(batch, 4)
        A22 = A[:, 2:, 2:].reshape(batch, 4)
        
        B11 = B[:, :2, :2].reshape(batch, 4)
        B12 = B[:, :2, 2:].reshape(batch, 4)
        B21 = B[:, 2:, :2].reshape(batch, 4)
        B22 = B[:, 2:, 2:].reshape(batch, 4)
        
        A_blocks = torch.stack([A11, A12, A21, A22], dim=1)
        B_blocks = torch.stack([B11, B12, B21, B22], dim=1)
        
        # Multiplicaciones con máscara
        mult_results = []
        for i in range(self.num_slots):
            if self.slot_mask[i] > 0.5:
                sub_A = torch.einsum('j,bjd->bd', self.U_coefs[i], A_blocks)
                sub_B = torch.einsum('j,bjd->bd', self.V_coefs[i], B_blocks)
                M_i = torch.bmm(sub_A.view(batch, 2, 2), sub_B.view(batch, 2, 2)).view(batch, 4)
            else:
                M_i = torch.zeros(batch, 4, device=device)
            mult_results.append(M_i)
        
        # Recombinar
        M_stack = torch.stack(mult_results, dim=1)
        C_blocks = torch.einsum('ji,bid->bjd', self.W_coefs, M_stack)
        
        # Reorganizar
        C11 = C_blocks[:, 0, :].view(batch, 2, 2)
        C12 = C_blocks[:, 1, :].view(batch, 2, 2)
        C21 = C_blocks[:, 2, :].view(batch, 2, 2)
        C22 = C_blocks[:, 3, :].view(batch, 2, 2)
        
        C_top = torch.cat([C11, C12], dim=2)
        C_bot = torch.cat([C21, C22], dim=2)
        return torch.cat([C_top, C_bot], dim=1)
    
    def get_slot_norms(self) -> List[float]:
        """Norma promedio de cada slot."""
        return [(self.U_coefs[i].norm() + self.V_coefs[i].norm() + 
                self.W_coefs[:, i].norm()).item() / 3 for i in range(self.num_slots)]
    
    def get_active_slots(self) -> int:
        """Número de slots activos."""
        return int(self.slot_mask.sum().item())
    
    def mask_slot(self, slot_idx: int):
        """Desactiva un slot."""
        self.slot_mask[slot_idx] = 0.0
        logger.info(f"Slot {slot_idx} enmascarado. Activos: {self.get_active_slots()}")
    
    def get_weakest_slot(self) -> int:
        """Slot con menor norma entre los activos."""
        norms = self.get_slot_norms()
        active = [(i, n) for i, n in enumerate(norms) if self.slot_mask[i] > 0.5]
        return min(active, key=lambda x: x[1])[0]
    
    def print_coefficients(self):
        """Muestra coeficientes descubiertos."""
        print("\n" + "="*60)
        print("COEFICIENTES DESCUBIERTOS")
        print("="*60)
        
        for i in range(self.num_slots):
            if self.slot_mask[i] > 0.5:
                u = self.U_coefs[i].detach().cpu().numpy()
                v = self.V_coefs[i].detach().cpu().numpy()
                w = self.W_coefs[:, i].detach().cpu().numpy()
                
                u_r = np.round(u * 2) / 2
                v_r = np.round(v * 2) / 2
                w_r = np.round(w * 2) / 2
                
                print(f"\nSlot {i}:")
                print(f"  U[{i}] (bloques A): {u_r}")
                print(f"  V[{i}] (bloques B): {v_r}")
                print(f"  W[:,{i}] (salida):  {w_r}")
        
        print("\n" + "="*60)


class Matrix4x4Dataset(Dataset):
    """Dataset de multiplicación de matrices 4x4."""
    
    def __init__(self, num_samples: int, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.samples = [(
            torch.tensor(self.rng.uniform(-1, 1, (4, 4)).astype(np.float32)),
            torch.tensor(self.rng.uniform(-1, 1, (4, 4)).astype(np.float32))
        ) for _ in range(num_samples)]
        self.samples = [(A, B, A @ B) for A, B in self.samples]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


class Trainer:
    """Entrenador con enmascaramiento progresivo."""
    
    def __init__(self, config: Config):
        self.config = config
        self.device = torch.device(config.DEVICE)
        
        # Crear directorio de salida
        config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Modelo
        self.model = StrassenDiscovery(config.NUM_SLOTS).to(self.device)
        
        # Datos
        train_ds = Matrix4x4Dataset(config.TRAIN_SAMPLES, config.SEED)
        test_ds = Matrix4x4Dataset(config.TEST_SAMPLES, config.SEED + 1000)
        self.train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, shuffle=True)
        self.test_loader = DataLoader(test_ds, batch_size=config.BATCH_SIZE)
        
        self.criterion = nn.MSELoss()
        self.start_time = time.time()
        
        logger.info(f"Modelo: {sum(p.numel() for p in self.model.parameters()):,} params")
        logger.info(f"Device: {self.device}")
    
    def accuracy(self, pred, target):
        return ((pred - target).abs() < self.config.ACCURACY_THRESHOLD).float().mean().item()
    
    def train_epoch(self, optimizer):
        self.model.train()
        total_loss, total_acc, n = 0, 0, 0
        
        for A, B, C in self.train_loader:
            A, B, C = A.to(self.device), B.to(self.device), C.to(self.device)
            
            optimizer.zero_grad()
            pred = self.model(A, B)
            loss = self.criterion(pred, C)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item() * A.size(0)
            total_acc += self.accuracy(pred, C) * A.size(0)
            n += A.size(0)
        
        return {'loss': total_loss/n, 'acc': total_acc/n}
    
    def evaluate(self):
        self.model.eval()
        total_loss, total_acc, n = 0, 0, 0
        
        with torch.no_grad():
            for A, B, C in self.test_loader:
                A, B, C = A.to(self.device), B.to(self.device), C.to(self.device)
                pred = self.model(A, B)
                loss = self.criterion(pred, C)
                total_loss += loss.item() * A.size(0)
                total_acc += self.accuracy(pred, C) * A.size(0)
                n += A.size(0)
        
        return {'loss': total_loss/n, 'acc': total_acc/n}
    
    def train(self):
        logger.info("="*70)
        logger.info("STRASSEN DISCOVERY: Enmascaramiento Progresivo")
        logger.info("="*70)
        
        # FASE 1: 8 slots
        logger.info("\n[FASE 1] Entrenando con 8 slots...")
        opt = optim.AdamW(self.model.parameters(), lr=self.config.PHASE1_LR, weight_decay=self.config.PHASE1_WD)
        
        for epoch in range(1, self.config.PHASE1_EPOCHS + 1):
            m = self.train_epoch(opt)
            if epoch % self.config.LOG_INTERVAL == 0 or epoch == 1:
                t = self.evaluate()
                logger.info(f"E {epoch:4d} | Loss:{m['loss']:.8f} | Acc:{m['acc']*100:.1f}% | Test:{t['acc']*100:.1f}%")
            if m['acc'] > 0.999 and epoch > 500:
                logger.info(f"  100% alcanzado en epoch {epoch}")
                break
        
        logger.info(f"  Fase 1: {self.evaluate()['acc']*100:.1f}% accuracy")
        
        # FASE 2: Enmascarar y reentrenar
        logger.info("\n[FASE 2] Enmascarando slot más débil...")
        weakest = self.model.get_weakest_slot()
        self.model.mask_slot(weakest)
        
        opt = optim.AdamW(self.model.parameters(), lr=self.config.PHASE2_LR, weight_decay=self.config.PHASE2_WD)
        
        for epoch in range(1, self.config.PHASE2_EPOCHS + 1):
            m = self.train_epoch(opt)
            if epoch % self.config.LOG_INTERVAL == 0 or epoch == 1:
                t = self.evaluate()
                logger.info(f"E {epoch:4d} | Loss:{m['loss']:.8f} | Acc:{m['acc']*100:.1f}% | Test:{t['acc']*100:.1f}% | Slots:7")
            if m['acc'] > 0.999 and epoch > 1000:
                logger.info(f"  100% recuperado en epoch {epoch}")
                break
        
        # Resultado
        final = self.evaluate()
        logger.info("\n" + "="*70)
        logger.info("RESULTADO FINAL")
        logger.info("="*70)
        logger.info(f"  Slots activos: {self.model.get_active_slots()}/8")
        logger.info(f"  Test Loss: {final['loss']:.10f}")
        logger.info(f"  Test Accuracy: {final['acc']*100:.2f}%")
        logger.info(f"  Tiempo: {(time.time()-self.start_time)/60:.1f} min")
        
        if self.model.get_active_slots() == 7 and final['acc'] > 0.99:
            logger.info("\n  EXITO: Algoritmo de 7 multiplicaciones descubierto")
        
        self.model.print_coefficients()
        
        # Guardar
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'active_slots': self.model.get_active_slots(),
            'slot_mask': self.model.slot_mask.cpu().numpy(),
            'slot_norms': self.model.get_slot_norms(),
            'test_accuracy': final['acc'],
            'test_loss': final['loss']
        }, self.config.MODEL_PATH)
        logger.info(f"\nModelo guardado: {self.config.MODEL_PATH}")
        
        return self.model


def main():
    config = Config()
    set_seed(config.SEED)
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
