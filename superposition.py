import argparse
import json
import logging
import time
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Protocol, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import pearsonr


@dataclass(frozen=True)
class Config:
    """Configuration for Superposition Analysis on Strassen Checkpoints."""
    
    # Strassen Model Dimensions (from your existing checkpoints)
    MATRIX_SIZE: int = 2
    HIDDEN_DIM: int = 8
    INPUT_DIM: int = 4  # MATRIX_SIZE * MATRIX_SIZE
    
    # SAE Parameters for Superposition Measurement
    SAE_EXPANSION_FACTOR: int = 8  # Dictionary size = HIDDEN_DIM * 8
    SAE_L1_COEFFICIENT: float = 0.1
    SAE_LEARNING_RATE: float = 1e-3
    SAE_EPOCHS: int = 1000
    SAE_BATCH_SIZE: int = 256
    
    # Data Generation for Activation Extraction
    NUM_ACTIVATION_SAMPLES: int = 10000
    BATCH_SIZE: int = 32
    
    # Checkpointing and Logging
    CHECKPOINT_INTERVAL_MINUTES: int = 5
    CHECKPOINT_DIR: str = "checkpoints_analysis"
    RESULTS_DIR: str = "superposition_results"
    
    # Numerical Stability
    EPSILON: float = 1e-10
    MIN_VARIANCE_THRESHOLD: float = 1e-8
    
    # Hardware
    DEVICE: str = field(default_factory=lambda: 'cuda' if torch.cuda.is_available() else 'cpu')
    
    def __post_init__(self):
        object.__setattr__(self, 'DEVICE', torch.device(self.DEVICE))
        object.__setattr__(self, 'SAE_DICTIONARY_SIZE', self.HIDDEN_DIM * self.SAE_EXPANSION_FACTOR)


class ICheckpointLoader(Protocol):
    def load_checkpoint(self, path: str, device: str) -> Any: ...

class IMetricsCalculator(ABC):
    @abstractmethod
    def compute(self, **kwargs) -> Dict[str, float]: ...

class IAnalyzer(ABC):
    @abstractmethod
    def analyze_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]: ...


class CheckpointLoadingError(Exception):
    pass


class CheckpointLoader:
    """Loads raw checkpoint files."""
    
    def load_checkpoint(self, path: str, device: str) -> Any:
        try:
            return torch.load(path, map_location=device, weights_only=False)
        except Exception as e:
            raise CheckpointLoadingError(f"Failed to load checkpoint from {path}: {e}")


class CheckpointMigrator:
    """Migrates various checkpoint formats to standard state_dict."""
    
    @staticmethod
    def detect_hidden_dim(raw_data: Any) -> Optional[int]:
        """
        Detect hidden dimension from checkpoint data structure by inspecting
        tensor shapes in various known formats.
        Returns None if cannot be determined unambiguously.
        """
        if not isinstance(raw_data, dict):
            if hasattr(raw_data, 'state_dict'):
                return CheckpointMigrator.detect_hidden_dim(raw_data.state_dict())
            return None
        
        # Check direct tensor keys that indicate hidden dimension (output features of U/V)
        for key in ['U', 'U_coefs', 'U.weight']:
            if key in raw_data:
                tensor = raw_data[key]
                if isinstance(tensor, torch.Tensor):
                    return tensor.shape[0]
        
        # Check nested standard PyTorch checkpoint formats
        for nested_key in ['state_dict', 'model_state_dict', 'model']:
            if nested_key in raw_data and isinstance(raw_data[nested_key], dict):
                dim = CheckpointMigrator.detect_hidden_dim(raw_data[nested_key])
                if dim is not None:
                    return dim
        
        # For encoder format, dimension cannot be determined without inspection of tensor contents
        # Return None to allow post-migration detection from extracted tensors
        if 'encoder.0.weight' in raw_data:
            return None
            
        return None
    
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
        """
        Handle encoder-based format from specific experimental architectures.
        Extracts U, V, W from sequential encoder layers assuming specific indexing.
        """
        encoder_0 = state_dict['encoder.0.weight']
        encoder_2 = state_dict.get('encoder.2.weight', encoder_0)
        encoder_4 = state_dict.get('encoder.4.weight', 
                                   torch.randn(64, 64, device=encoder_0.device))
        
        # Infer input dimension from first encoder layer (typically 4 for 2x2 matrices)
        input_dim = encoder_0.shape[1] if encoder_0.shape[1] <= 16 else 4
        
        # Infer hidden dimension: if tensor is oversized (e.g., 64x8), extract meaningful submatrix
        # Otherwise use full dimension
        if encoder_0.shape[0] > 16:
            # Heuristic: for oversized tensors, assume standard hidden_dim is in first 8 or 7 rows
            # This handles both superposition (8) and exact Strassen (7) models
            hidden_dim = 8 if encoder_0.shape[0] >= 8 else encoder_0.shape[0]
        else:
            hidden_dim = encoder_0.shape[0]
        
        # Extract U: [hidden_dim, input_dim]
        u = encoder_0[:hidden_dim, :input_dim].clone()
        
        # Extract V: [hidden_dim, input_dim]
        v = encoder_2[:hidden_dim, :input_dim].clone() if encoder_2.shape[0] >= hidden_dim else \
            encoder_2.clone()
        
        # Extract W: [input_dim, hidden_dim]
        # W maps from hidden space back to input space, may need transpose depending on storage
        if encoder_4.shape[0] >= input_dim and encoder_4.shape[1] >= hidden_dim:
            w = encoder_4[:input_dim, :hidden_dim].clone()
        elif encoder_4.shape[0] >= hidden_dim and encoder_4.shape[1] >= input_dim:
            w = encoder_4[:hidden_dim, :input_dim].clone().t()
        else:
            # Fallback: extract available submatrix and pad if necessary
            w = torch.zeros(input_dim, hidden_dim, device=encoder_0.device)
            min_rows = min(input_dim, encoder_4.shape[0])
            min_cols = min(hidden_dim, encoder_4.shape[1])
            w[:min_rows, :min_cols] = encoder_4[:min_rows, :min_cols]
        
        return {'U.weight': u, 'V.weight': v, 'W.weight': w}
    
    @staticmethod
    def _migrate_standard_format(state_dict: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        return {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}



class StrassenDataGenerator:
    """Generates matrix multiplication data for activation extraction."""
    
    def __init__(self, config: Config):
        self.config = config
    
    def generate_batch(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate batch of matrix pairs and their product."""
        A = torch.randn(batch_size, self.config.MATRIX_SIZE, self.config.MATRIX_SIZE, 
                       device=self.config.DEVICE)
        B = torch.randn(batch_size, self.config.MATRIX_SIZE, self.config.MATRIX_SIZE, 
                       device=self.config.DEVICE)
        C = torch.bmm(A, B)
        
        A_flat = A.reshape(batch_size, self.config.INPUT_DIM)
        B_flat = B.reshape(batch_size, self.config.INPUT_DIM)
        C_flat = C.reshape(batch_size, self.config.INPUT_DIM)
        
        return A_flat, B_flat, C_flat
    
    def generate_dataset(self, num_samples: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate full dataset."""
        A_list, B_list, C_list = [], [], []
        batches = (num_samples + self.config.BATCH_SIZE - 1) // self.config.BATCH_SIZE
        
        for _ in range(batches):
            A, B, C = self.generate_batch(self.config.BATCH_SIZE)
            A_list.append(A)
            B_list.append(B)
            C_list.append(C)
        
        A_all = torch.cat(A_list, dim=0)[:num_samples]
        B_all = torch.cat(B_list, dim=0)[:num_samples]
        C_all = torch.cat(C_list, dim=0)[:num_samples]
        
        return A_all, B_all, C_all


class BilinearStrassenModel(nn.Module):
    """Your existing Strassen model architecture."""
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.U = nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM, bias=False)
        self.V = nn.Linear(config.INPUT_DIM, config.HIDDEN_DIM, bias=False)
        self.W = nn.Linear(config.HIDDEN_DIM, config.INPUT_DIM, bias=False)
        
        self._initialize_symmetric()
    
    def _initialize_symmetric(self):
        nn.init.xavier_uniform_(self.U.weight)
        self.V.weight.data = self.U.weight.data.clone()
        nn.init.xavier_uniform_(self.W.weight)
    
    def forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        """Forward pass returning output and bottleneck activations."""
        u_out = self.U(a)  # [batch, hidden]
        v_out = self.V(b)  # [batch, hidden]
        bottleneck = u_out * v_out  # Element-wise product [batch, hidden]
        output = self.W(bottleneck)
        return output, bottleneck
    
    def get_coefficients(self) -> Dict[str, torch.Tensor]:
        return {
            'U': self.U.weight.data,
            'V': self.V.weight.data,
            'W': self.W.weight.data
        }


class SparseAutoencoder(nn.Module):
    """
    SAE with tied weights (W_dec = W_enc^T).
    Corrected dimensions: W_enc: [D, N], encode uses W_enc^T, decode uses W_enc.
    """
    
    def __init__(self, config: Config):
        super().__init__()
        self.config = config
        
        self.W_enc = nn.Parameter(
            torch.randn(config.SAE_DICTIONARY_SIZE, config.HIDDEN_DIM, device=config.DEVICE) * 0.01
        )
        self.b_enc = nn.Parameter(torch.zeros(config.SAE_DICTIONARY_SIZE, device=config.DEVICE))
        self.b_dec = nn.Parameter(torch.zeros(config.HIDDEN_DIM, device=config.DEVICE))
        
        nn.init.xavier_uniform_(self.W_enc)
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode bottleneck activations to sparse features.
        x: [batch, N]
        W_enc: [D, N]
        Returns: [batch, D]
        """
        return F.relu(torch.matmul(x, self.W_enc.t()) + self.b_enc)
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode sparse features back to bottleneck.
        z: [batch, D]
        W_enc: [D, N]
        Returns: [batch, N]
        """
        return torch.matmul(z, self.W_enc) + self.b_dec
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_recon = self.decode(z)
        return x_recon, z


class SuperpositionMetrics(IMetricsCalculator):
    """
    Calculates superposition metrics from Section 4 of the paper.
    """
    
    def __init__(self, config: Config):
        self.config = config
    
    def compute_feature_probabilities(self, sae_activations: torch.Tensor) -> torch.Tensor:
        """
        Calculate feature probabilities from SAE activations.
        p_i = Σ_s |z_i,s| / Σ_j Σ_s |z_j,s|
        """
        abs_acts = torch.abs(sae_activations)  # [batch, D]
        feature_budget = torch.sum(abs_acts, dim=0)  # [D]
        total_budget = torch.sum(feature_budget) + self.config.EPSILON
        return feature_budget / total_budget
    
    def compute_entropy(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Shannon entropy H(p) = -Σ p_i log p_i."""
        probs = probabilities[probabilities > self.config.EPSILON]
        if len(probs) == 0:
            return torch.tensor(0.0, device=probs.device)
        return -torch.sum(probs * torch.log(probs))
    
    def compute_superposition(self, sae_activations: torch.Tensor) -> Dict[str, float]:
        """
        Main metric: ψ = F/N where F = e^{H(p)}.
        """
        p = self.compute_feature_probabilities(sae_activations)
        H = self.compute_entropy(p)
        F = torch.exp(H)
        psi = F / self.config.HIDDEN_DIM
        
        return {
            'psi': float(psi),
            'effective_features_F': float(F),
            'entropy_H': float(H),
            'max_probability': float(p.max()),
            'min_probability': float(p[p > self.config.EPSILON].min()) if (p > 0).any() else 0.0,
            'num_active_features': int((p > self.config.MIN_VARIANCE_THRESHOLD).sum())
        }
    
    def compute_frobenius_metric(self, weight_matrix: torch.Tensor) -> float:
        """
        Baseline from Eq 2: ψ_Frob = ||W||_F^2 / N.
        Applied to the bottleneck transformation (W matrix of Strassen).
        """
        frob_norm_sq = torch.norm(weight_matrix, p='fro') ** 2
        return float(frob_norm_sq / weight_matrix.shape[0])
    
    def compute_interference_matrix(self, weight_matrix: torch.Tensor) -> torch.Tensor:
        """Compute W^T @ W to analyze interference patterns."""
        return torch.matmul(weight_matrix.t(), weight_matrix)
    
    def compute(self, sae_activations: Optional[torch.Tensor] = None, 
                weight_matrix: Optional[torch.Tensor] = None) -> Dict[str, float]:
        """Unified interface."""
        results = {}
        
        if sae_activations is not None:
            results.update(self.compute_superposition(sae_activations))
        
        if weight_matrix is not None:
            results['psi_frob'] = self.compute_frobenius_metric(weight_matrix)
            interference = self.compute_interference_matrix(weight_matrix)
            results['interference_diagonal_mean'] = float(torch.diag(interference).mean())
            off_diag = interference - torch.diag(torch.diag(interference))
            results['interference_off_diagonal_mean'] = float(off_diag.mean())
        
        return results


class SAETrainer:
    """Trains SAE on bottleneck activations extracted from Strassen model."""
    
    def __init__(self, sae: SparseAutoencoder, config: Config):
        self.sae = sae
        self.config = config
        self.optimizer = torch.optim.AdamW(
            sae.parameters(),
            lr=config.SAE_LEARNING_RATE,
            weight_decay=1e-5
        )
        self.metrics = SuperpositionMetrics(config)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def train(self, bottleneck_activations: torch.Tensor) -> Dict[str, List[float]]:
        """Train SAE on extracted activations."""
        dataset = TensorDataset(bottleneck_activations)
        loader = DataLoader(dataset, batch_size=self.config.SAE_BATCH_SIZE, shuffle=True)
        
        history = {
            'epoch': [],
            'recon_loss': [],
            'l1_loss': [],
            'total_loss': [],
            'psi': [],
            'effective_features': []
        }
        
        self.sae.train()
        num_epochs = self.config.SAE_EPOCHS
        
        for epoch in tqdm(range(num_epochs), desc="Training SAE"):
            epoch_recon = 0.0
            epoch_l1 = 0.0
            num_batches = 0
            all_activations = []
            
            for (batch,) in loader:
                x_recon, z = self.sae(batch)
                
                recon_loss = F.mse_loss(x_recon, batch)
                l1_loss = torch.mean(torch.abs(z))
                total_loss = recon_loss + self.config.SAE_L1_COEFFICIENT * l1_loss
                
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                
                epoch_recon += recon_loss.item()
                epoch_l1 += l1_loss.item()
                num_batches += 1
                all_activations.append(z.detach())
            
            # Calculate metrics
            all_z = torch.cat(all_activations, dim=0)
            metrics = self.metrics.compute(sae_activations=all_z)
            
            history['epoch'].append(epoch)
            history['recon_loss'].append(epoch_recon / num_batches)
            history['l1_loss'].append(epoch_l1 / num_batches)
            history['total_loss'].append((epoch_recon / num_batches) + 
                                          self.config.SAE_L1_COEFFICIENT * (epoch_l1 / num_batches))
            history['psi'].append(metrics['psi'])
            history['effective_features'].append(metrics['effective_features_F'])
        
        return history


class StrassenCheckpointAnalyzer(IAnalyzer):
    """
    Analyzes existing Strassen checkpoints for superposition metrics.
    """
    
    def __init__(self, config: Config):
        self.config = config
        self.loader = CheckpointLoader()
        self.migrator = CheckpointMigrator()
        self.data_generator = StrassenDataGenerator(config)
        self.metrics = SuperpositionMetrics(config)
        self.logger = logging.getLogger(self.__class__.__name__)
        
        self.results_dir = Path(config.RESULTS_DIR)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        self.checkpoint_dir = Path(config.CHECKPOINT_DIR)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.last_checkpoint_time = datetime.now()
    
    def load_model(self, checkpoint_path: Path) -> Optional[Tuple[BilinearStrassenModel, Config]]:
        """
        Load and migrate checkpoint to model.
        Returns tuple of (model, effective_config) where effective_config has 
        the correct HIDDEN_DIM for this specific checkpoint to avoid dimension mismatches.
        """
        try:
            raw_data = self.loader.load_checkpoint(str(checkpoint_path), self.config.DEVICE)
            
            # Detect hidden dimension from checkpoint before creating model
            detected_dim = self.migrator.detect_hidden_dim(raw_data)
            target_dim = detected_dim if detected_dim is not None else self.config.HIDDEN_DIM
            
            migrated_state = self.migrator.migrate_checkpoint(raw_data)
            
            if migrated_state is None:
                self.logger.warning(f"Could not migrate checkpoint: {checkpoint_path}")
                return None
            
            # Re-detect from migrated state if previous detection failed (e.g., encoder format)
            # This ensures we get the actual dimension from the extracted tensors
            if 'U.weight' in migrated_state:
                actual_dim = migrated_state['U.weight'].shape[0]
                if actual_dim != target_dim:
                    target_dim = actual_dim
            
            # Create effective configuration with detected dimensions for this specific checkpoint
            if target_dim != self.config.HIDDEN_DIM:
                effective_config = Config(
                    MATRIX_SIZE=self.config.MATRIX_SIZE,
                    HIDDEN_DIM=target_dim,
                    INPUT_DIM=self.config.INPUT_DIM,
                    SAE_EXPANSION_FACTOR=self.config.SAE_EXPANSION_FACTOR,
                    SAE_L1_COEFFICIENT=self.config.SAE_L1_COEFFICIENT,
                    SAE_LEARNING_RATE=self.config.SAE_LEARNING_RATE,
                    SAE_EPOCHS=self.config.SAE_EPOCHS,
                    SAE_BATCH_SIZE=self.config.SAE_BATCH_SIZE,
                    NUM_ACTIVATION_SAMPLES=self.config.NUM_ACTIVATION_SAMPLES,
                    BATCH_SIZE=self.config.BATCH_SIZE,
                    CHECKPOINT_INTERVAL_MINUTES=self.config.CHECKPOINT_INTERVAL_MINUTES,
                    CHECKPOINT_DIR=self.config.CHECKPOINT_DIR,
                    RESULTS_DIR=self.config.RESULTS_DIR,
                    EPSILON=self.config.EPSILON,
                    MIN_VARIANCE_THRESHOLD=self.config.MIN_VARIANCE_THRESHOLD,
                    DEVICE=self.config.DEVICE,
                )
            else:
                effective_config = self.config
            
            model = BilinearStrassenModel(effective_config).to(self.config.DEVICE)
            model.load_state_dict(migrated_state, strict=False)
            
            return model, effective_config
            
        except Exception as e:
            self.logger.error(f"Error loading checkpoint {checkpoint_path}: {e}")
            return None
            
    def extract_bottleneck_activations(self, model: BilinearStrassenModel) -> torch.Tensor:
        """Extract bottleneck activations (U(a) * V(b)) from model."""
        A, B, _ = self.data_generator.generate_dataset(self.config.NUM_ACTIVATION_SAMPLES)
        
        activations = []
        batch_size = self.config.BATCH_SIZE
        
        with torch.no_grad():
            for i in range(0, len(A), batch_size):
                A_batch = A[i:i+batch_size]
                B_batch = B[i:i+batch_size]
                
                _, bottleneck = model(A_batch, B_batch)
                activations.append(bottleneck.cpu())
        
        return torch.cat(activations, dim=0).to(self.config.DEVICE)
    
    def analyze_checkpoint(self, checkpoint_path: Path) -> Dict[str, Any]:
        """
        Full analysis pipeline for a single checkpoint:
        1. Load model with correct dimensions (detecting hidden_dim from checkpoint)
        2. Extract bottleneck activations
        3. Train SAE with matching dimensions (using effective_config)
        4. Calculate superposition metrics with correct normalization (N=hidden_dim)
        5. Calculate baseline Frobenius metric on W weights
        """
        self.logger.info(f"Analyzing checkpoint: {checkpoint_path.name}")
        
        result = self.load_model(checkpoint_path)
        if result is None:
            return {'error': 'Failed to load model', 'checkpoint': str(checkpoint_path)}
        
        model, effective_config = result
        
        # Create metrics calculator with effective configuration to ensure correct psi calculation (F/N)
        metrics = SuperpositionMetrics(effective_config)
        
        # Get weight-based metrics (baseline)
        coeffs = model.get_coefficients()
        W_matrix = coeffs['W']  # [output_dim, hidden_dim]
        
        weight_metrics = metrics.compute(weight_matrix=W_matrix)
        
        # Extract activations using the model with correct dimensions
        self.logger.info("  Extracting bottleneck activations...")
        bottleneck_acts = self.extract_bottleneck_activations(model)
        
        # Train SAE with dimensions matching the loaded model (critical for correct analysis)
        self.logger.info(f"  Training SAE on {len(bottleneck_acts)} samples...")
        sae = SparseAutoencoder(effective_config).to(self.config.DEVICE)
        trainer = SAETrainer(sae, effective_config)
        sae_history = trainer.train(bottleneck_acts)
        
        # Final metrics on trained SAE
        with torch.no_grad():
            _, final_z = sae(bottleneck_acts)
            final_metrics = metrics.compute(sae_activations=final_z)
        
        # Combine results
        results = {
            'checkpoint_name': checkpoint_path.name,
            'checkpoint_path': str(checkpoint_path),
            'timestamp': datetime.now().isoformat(),
            'detected_hidden_dim': effective_config.HIDDEN_DIM,
            'weight_metrics': weight_metrics,
            'sae_metrics': final_metrics,
            'training_history': sae_history,
            'model_coefficients': {
                'U_shape': list(coeffs['U'].shape),
                'V_shape': list(coeffs['V'].shape),
                'W_shape': list(coeffs['W'].shape)
            }
        }
        
        # Save intermediate result
        self._save_intermediate_result(results, checkpoint_path.stem)
        
        return results
        
    def _save_intermediate_result(self, result: Dict[str, Any], name: str):
        """Save result for individual checkpoint."""
        filepath = self.results_dir / f"analysis_{name}.json"
        with open(filepath, 'w') as f:
            # Convert tensors to lists for JSON serialization
            json.dump(result, f, indent=2, default=str)
        self.logger.info(f"  Saved analysis to: {filepath}")
    
    def analyze_directory(self, checkpoint_dir: str) -> List[Dict[str, Any]]:
        """Analyze all checkpoints in directory."""
        checkpoint_path = Path(checkpoint_dir)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir}")
        
        checkpoint_files = list(checkpoint_path.glob("*.pt"))
        if not checkpoint_files:
            self.logger.warning(f"No .pt files found in {checkpoint_dir}")
            return []
        
        self.logger.info(f"Found {len(checkpoint_files)} checkpoints to analyze")
        
        all_results = []
        
        for ckpt_file in tqdm(checkpoint_files, desc="Analyzing checkpoints"):
            try:
                result = self.analyze_checkpoint(ckpt_file)
                all_results.append(result)
                
                # Checkpointing analysis progress every 5 minutes
                elapsed = datetime.now() - self.last_checkpoint_time
                if elapsed >= timedelta(minutes=self.config.CHECKPOINT_INTERVAL_MINUTES):
                    self._save_progress_checkpoint(all_results)
                    self.last_checkpoint_time = datetime.now()
                    
            except Exception as e:
                self.logger.error(f"Failed to analyze {ckpt_file.name}: {e}")
                all_results.append({
                    'error': str(e),
                    'checkpoint_name': ckpt_file.name
                })
        
        # Final save
        self._save_final_results(all_results)
        self._generate_comparison_plots(all_results)
        
        return all_results
    
    def _save_progress_checkpoint(self, results: List[Dict[str, Any]]):
        """Save intermediate progress."""
        filepath = self.checkpoint_dir / f"progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Progress checkpoint saved: {filepath}")
    
    def _save_final_results(self, results: List[Dict[str, Any]]):
        """Save complete results."""
        filepath = self.results_dir / "complete_analysis.json"
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        self.logger.info(f"Complete analysis saved: {filepath}")
    
    def _generate_comparison_plots(self, results: List[Dict[str, Any]]):
        """Generate comparison plots across checkpoints."""
        valid_results = [r for r in results if 'error' not in r and 'sae_metrics' in r]
        
        if len(valid_results) < 2:
            self.logger.warning("Insufficient valid results for comparison plots")
            return
        
        names = [r['checkpoint_name'] for r in valid_results]
        psi_values = [r['sae_metrics']['psi'] for r in valid_results]
        frob_values = [r['weight_metrics']['psi_frob'] for r in valid_results]
        F_values = [r['sae_metrics']['effective_features_F'] for r in valid_results]
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Psi comparison
        axes[0, 0].bar(range(len(names)), psi_values, alpha=0.7, color='blue')
        axes[0, 0].set_xticks(range(len(names)))
        axes[0, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 0].set_ylabel('Superposition ψ')
        axes[0, 0].set_title('Superposition Metric Across Checkpoints')
        axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Lossless boundary')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Frobenius baseline
        axes[0, 1].bar(range(len(names)), frob_values, alpha=0.7, color='green')
        axes[0, 1].set_xticks(range(len(names)))
        axes[0, 1].set_xticklabels(names, rotation=45, ha='right')
        axes[0, 1].set_ylabel('ψ_Frob')
        axes[0, 1].set_title('Frobenius Baseline Metric')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Effective features F
        axes[1, 0].bar(range(len(names)), F_values, alpha=0.7, color='purple')
        axes[1, 0].set_xticks(range(len(names)))
        axes[1, 0].set_xticklabels(names, rotation=45, ha='right')
        axes[1, 0].set_ylabel('Effective Features F')
        axes[1, 0].axhline(y=self.config.HIDDEN_DIM, color='r', linestyle='--', 
                          label=f'Physical dim (N={self.config.HIDDEN_DIM})')
        axes[1, 0].set_title('Effective vs Physical Dimensions')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Correlation scatter
        if len(psi_values) > 1:
            axes[1, 1].scatter(frob_values, psi_values, s=100, alpha=0.7)
            for i, name in enumerate(names):
                axes[1, 1].annotate(name[:10], (frob_values[i], psi_values[i]), 
                                   fontsize=8, alpha=0.7)
            axes[1, 1].set_xlabel('ψ_Frob (Weight-based)')
            axes[1, 1].set_ylabel('ψ (Activation-based)')
            axes[1, 1].set_title('Metric Correlation')
            
            # Fit line
            if len(psi_values) > 2:
                z = np.polyfit(frob_values, psi_values, 1)
                p = np.poly1d(z)
                x_line = np.linspace(min(frob_values), max(frob_values), 100)
                axes[1, 1].plot(x_line, p(x_line), "r--", alpha=0.5, 
                               label=f'Fit: slope={z[0]:.2f}')
                axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = self.results_dir / "comparison_analysis.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        self.logger.info(f"Comparison plot saved: {plot_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Superposition Analysis for Strassen Checkpoints"
    )
    parser.add_argument("--checkpoint_dir", type=str, required=True,
                       help="Directory containing Strassen checkpoint files (.pt)")
    parser.add_argument("--hidden_dim", type=int, default=8,
                       help="Hidden dimension of Strassen model (N)")
    parser.add_argument("--matrix_size", type=int, default=2,
                       help="Matrix size (2 for 2x2)")
    parser.add_argument("--sae_expansion", type=int, default=8,
                       help="SAE dictionary expansion factor (D = N * expansion)")
    parser.add_argument("--sae_epochs", type=int, default=1000,
                       help="Training epochs for SAE on each checkpoint")
    parser.add_argument("--results_dir", type=str, default="superposition_results",
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    config = Config(
        HIDDEN_DIM=args.hidden_dim,
        MATRIX_SIZE=args.matrix_size,
        INPUT_DIM=args.matrix_size * args.matrix_size,
        SAE_EXPANSION_FACTOR=args.sae_expansion,
        SAE_EPOCHS=args.sae_epochs,
        RESULTS_DIR=args.results_dir
    )
    
    analyzer = StrassenCheckpointAnalyzer(config)
    results = analyzer.analyze_directory(args.checkpoint_dir)
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)
    print(f"Analyzed {len(results)} checkpoints")
    print(f"Results saved in: {args.results_dir}/")
    
    # Print summary table
    valid_results = [r for r in results if 'error' not in r]
    if valid_results:
        print(f"\n{'Checkpoint':<30} {'ψ':<10} {'F':<10} {'ψ_Frob':<10}")
        print("-" * 60)
        for r in valid_results:
            name = r['checkpoint_name'][:28]
            psi = r['sae_metrics']['psi']
            F = r['sae_metrics']['effective_features_F']
            frob = r['weight_metrics']['psi_frob']
            print(f"{name:<30} {psi:<10.3f} {F:<10.1f} {frob:<10.3f}")


if __name__ == "__main__":
    main()
