# train_batch_sweep.py
import torch
import torch.nn as nn
import os
from pathlib import Path
from vector8 import (
    Config, BilinearStrassenModel, StrassenDataGenerator,
    set_seed, setup_logger
)

def train_for_batch_size(B: int, seed: int, output_dir: str):
    set_seed(seed)
    logger = setup_logger(f"Train_B{B}_seed{seed}")
    device = Config.DEVICE

    # Configuración local
    class LocalConfig:
        BATCH_SIZE = B
        HIDDEN_DIM = 8
        MATRIX_SIZE = 2
        WEIGHT_DECAY = 1e-4
        LEARNING_RATE = 1e-3
        EPOCHS = 3000

    model = BilinearStrassenModel(
        hidden_dim=LocalConfig.HIDDEN_DIM,
        matrix_size=LocalConfig.MATRIX_SIZE
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LocalConfig.LEARNING_RATE,
        weight_decay=LocalConfig.WEIGHT_DECAY
    )

    for epoch in range(LocalConfig.EPOCHS):
        A, B_batch, C = StrassenDataGenerator.generate_batch(batch_size=B)
        optimizer.zero_grad()
        C_pred = model(A, B_batch)
        loss = nn.functional.mse_loss(C_pred, C)
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            logger.info(f"B={B}, seed={seed}, epoch={epoch}, loss={loss.item():.2e}")

    # Guardar checkpoint con metadatos
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    save_path = Path(output_dir) / f"bs{B}_seed{seed}.pt"
    torch.save({
        'state_dict': model.state_dict(),
        'batch_size': B,
        'seed': seed,
        'epochs': LocalConfig.EPOCHS,
        'config': {
            'weight_decay': LocalConfig.WEIGHT_DECAY,
            'lr': LocalConfig.LEARNING_RATE
        }
    }, save_path)
    logger.info(f"Saved to {save_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="sweep_checkpoints")
    args = parser.parse_args()

    batch_sizes = [8, 16, 32, 64, 128, 256, 512]
    seeds = list(range(10))

    for B in batch_sizes:
        for seed in seeds:
            train_for_batch_size(B, seed, args.output_dir)