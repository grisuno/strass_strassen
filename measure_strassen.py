import argparse
import torch
import torch.nn as nn
import numpy as np
import json
from pathlib import Path
from typing import Dict, Any, Tuple, List, Optional
import logging
from dataclasses import dataclass

@dataclass
class Config:
    DEVICE: str = 'cpu'
    MATRIX_SIZE: int = 2
    HIDDEN_DIM: int = 8
    TARGET_SLOTS: int = 7
    DISCRETIZATION_THRESHOLD: float = 0.1
    PRUNING_SPARSITY_LEVELS: List[float] = None
    ZERO_SHOT_SIZES: List[int] = None
    LOG_LEVEL: str = 'INFO'

    def __post_init__(self):
        if self.PRUNING_SPARSITY_LEVELS is None:
            self.PRUNING_SPARSITY_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]
        if self.ZERO_SHOT_SIZES is None:
            self.ZERO_SHOT_SIZES = [4, 8, 16, 32, 64]

def setup_logger(name: str, level: str = Config.LOG_LEVEL) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger

class BilinearStrassenModel(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.U = nn.Linear(input_dim, hidden_dim, bias=False)
        self.V = nn.Linear(input_dim, hidden_dim, bias=False)
        self.W = nn.Linear(hidden_dim, input_dim, bias=False)

    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return self.W(self.U(a) * self.V(b))

    def get_all_weights(self) -> torch.Tensor:
        return torch.cat([p.flatten() for p in self.parameters()])

    def prune_to_sparsity(self, sparsity: float) -> 'BilinearStrassenModel':
        model_copy = BilinearStrassenModel(
            input_dim=self.U.in_features,
            hidden_dim=self.U.out_features
        ).to(next(self.parameters()).device)
        model_copy.load_state_dict(self.state_dict())
        all_weights = torch.cat([p.data.flatten() for p in model_copy.parameters()])
        threshold = torch.quantile(torch.abs(all_weights), sparsity)
        with torch.no_grad():
            for param in model_copy.parameters():
                mask = torch.abs(param) >= threshold
                param.mul_(mask.float())
        return model_copy

# === CHECKPOINT LOADING FROM YOUR SYSTEM ===
class CheckpointLoadingError(Exception):
    pass

class CheckpointLoader:
    def load_checkpoint(self, path: str, device: str) -> Any:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            raise CheckpointLoading Error(f"Failed to load checkpoint: {e}")

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
        elif isinstance(raw_data, dict) and any(k in raw_data for k in ['U', 'V', 'W']):
            return CheckpointMigrator._migrate_dict(raw_data)
        return None

    @staticmethod
    def _migrate_dict(state_dict: Dict[str, Any]) -> Optional[Dict[str, torch.Tensor]]:
        if any(k in state_dict for k in ['U', 'V', 'W']):
            return CheckpointMigrator._migrate_custom_format(state_dict)
        elif 'U_coefs' in state_dict:
            return CheckpointMigrator._migrate_coefs_format(state_dict)
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
    def _migrate_standard_format(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {k: state_dict[k] for k in ['U.weight', 'V.weight', 'W.weight'] if k in state_dict}

def load_strassen_checkpoint(checkpoint_path: str, device: str = 'cpu') -> BilinearStrassenModel:
    loader = CheckpointLoader()
    raw_data = loader.load_checkpoint(checkpoint_path, device)
    migrated_state = CheckpointMigrator.migrate_checkpoint(raw_data)
    if migrated_state is None:
        raise ValueError("Could not migrate checkpoint to standard format")
    input_dim = Config.MATRIX_SIZE ** 2
    # Infer hidden_dim from U.weight
    hidden_dim = migrated_state['U.weight'].shape[0]
    model = BilinearStrassenModel(input_dim=input_dim, hidden_dim=hidden_dim).to(device)
    model.load_state_dict(migrated_state)
    return model

# === METRICS (IGUALES QUE ANTES) ===
def compute_delta(model: BilinearStrassenModel) -> float:
    weights = model.get_all_weights()
    return (weights - weights.round()).abs().max().item()

def compute_kappa(model: BilinearStrassenModel, num_samples: int = 10, batch_size: int = 32) -> float:
    grads = []
    for _ in range(num_samples):
        A = torch.randn(batch_size, Config.MATRIX_SIZE, Config.MATRIX_SIZE, device=Config.DEVICE)
        B = torch.randn(batch_size, Config.MATRIX_SIZE, Config.MATRIX_SIZE, device=Config.DEVICE)
        C = torch.bmm(A, B)
        A = A.reshape(batch_size, -1)
        B = B.reshape(batch_size, -1)
        C = C.reshape(batch_size, -1)
        C_pred = model(A, B)
        loss = nn.functional.mse_loss(C_pred, C)
        grad = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        grads.append(torch.cat([g.flatten() for g in grad]))
    grads = torch.stack(grads)
    cov = torch.cov(grads.T)
    eigvals = torch.linalg.eigvalsh(cov)
    eigvals = eigvals[eigvals > 1e-12]
    if len(eigvals) < 2:
        return float('inf')
    return (eigvals.max() / eigvals.min()).item()

def compute_local_complexity(model: BilinearStrassenModel) -> float:
    params = model.get_all_weights()
    perc_95 = torch.quantile(torch.abs(params), 0.95)
    active = (torch.abs(params) > 0.01 * perc_95).sum().float()
    return (active / len(params)).item()

def evaluate_pruning_robustness(model: BilinearStrassenModel, batch_size: int = 64) -> Dict[float, float]:
    results = {}
    A_test = torch.randn(batch_size, Config.MATRIX_SIZE, Config.MATRIX_SIZE, device=Config.DEVICE)
    B_test = torch.randn(batch_size, Config.MATRIX_SIZE, Config.MATRIX_SIZE, device=Config.DEVICE)
    C_true = torch.bmm(A_test, B_test)
    A_test = A_test.reshape(batch_size, -1)
    B_test = B_test.reshape(batch_size, -1)
    C_true = C_true.reshape(batch_size, -1)
    with torch.no_grad():
        C_orig = model(A_test, B_test)
        baseline_acc = (torch.abs(C_orig - C_true) < 1e-4).float().mean().item()
    for sparsity in Config.PRUNING_SPARSITY_LEVELS:
        pruned_model = model.prune_to_sparsity(sparsity)
        with torch.no_grad():
            C_pruned = pruned_model(A_test, B_test)
            acc = (torch.abs(C_pruned - C_true) < 1e-4).float().mean().item()
        results[sparsity] = acc
    return results

def recursive_strassen(A: torch.Tensor, B: torch.Tensor, coeffs: Dict[str, torch.Tensor], N: int) -> torch.Tensor:
    if N == 2:
        A_vec = A.reshape(-1, 4)
        B_vec = B.reshape(-1, 4)
        M = coeffs['U'] @ A_vec.T * (coeffs['V'] @ B_vec.T)
        C_flat = coeffs['W'] @ M
        return C_flat.T.reshape(-1, 2, 2)
    half = N // 2
    A11, A12 = A[:, :half, :half], A[:, :half, half:]
    A21, A22 = A[:, half:, :half], A[:, half:, half:]
    B11, B12 = B[:, :half, :half], B[:, :half, half:]
    B21, B22 = B[:, half:, :half], B[:, half:, half:]
    M1 = recursive_strassen(A11 + A22, B11 + B22, coeffs, half)
    M2 = recursive_strassen(A21 + A22, B11, coeffs, half)
    M3 = recursive_strassen(A11, B12 - B22, coeffs, half)
    M4 = recursive_strassen(A22, B21 - B11, coeffs, half)
    M5 = recursive_strassen(A11 + A12, B22, coeffs, half)
    M6 = recursive_strassen(A21 - A11, B11 + B12, coeffs, half)
    M7 = recursive_strassen(A12 - A22, B21 + B22, coeffs, half)
    C11 = M1 + M4 - M5 + M7
    C12 = M3 + M5
    C21 = M2 + M4
    C22 = M1 - M2 + M3 + M6
    C = torch.zeros(A.shape[0], N, N, device=A.device)
    C[:, :half, :half] = C11
    C[:, :half, half:] = C12
    C[:, half:, :half] = C21
    C[:, half:, half:] = C22
    return C

def evaluate_zero_shot_transfer(model: BilinearStrassenModel, batch_size: int = 16) -> Dict[int, float]:
    results = {}
    coeffs = {
        'U': model.U.weight.data.clone(),
        'V': model.V.weight.data.clone(),
        'W': model.W.weight.data.clone()
    }
    for N in Config.ZERO_SHOT_SIZES:
        try:
            A = torch.randn(batch_size, N, N, device=Config.DEVICE)
            B = torch.randn(batch_size, N, N, device=Config.DEVICE)
            C_true = torch.bmm(A, B)
            C_pred = recursive_strassen(A, B, coeffs, N)
            mse = torch.mean((C_true - C_pred) ** 2).item()
            results[N] = mse
        except Exception:
            results[N] = float('inf')
    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="strassen_metrics.json")
    args = parser.parse_args()

    logger = setup_logger("StrassenMeasurement")
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    model = load_strassen_checkpoint(args.checkpoint, Config.DEVICE)

    metrics = {}
    delta = compute_delta(model)
    kappa = compute_kappa(model)
    lc = compute_local_complexity(model)
    pruning = evaluate_pruning_robustness(model)
    zero_shot = evaluate_zero_shot_transfer(model)

    metrics.update({
        'delta': delta,
        'is_crystal': delta < Config.DISCRETIZATION_THRESHOLD,
        'kappa': kappa,
        'local_complexity': lc,
        'pruning_robustness': pruning,
        'maintains_100_until_50p_sparsity': all(acc >= 0.99 for s, acc in pruning.items() if s <= 0.5),
        'zero_shot_transfer_mse': zero_shot,
        'zero_shot_success': all(mse < 1e-6 for mse in zero_shot.values())
    })

    logger.info("✅ Strassen metrics computed:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v}")

    with open(args.output, 'w') as f:
        json.dump(metrics, f, indent=2, default=str)

    logger.info(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()