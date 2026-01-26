# compute_gns_by_batch.py
import torch
import torch.nn as nn
from pathlib import Path
from collections import defaultdict
from vector8 import (
    Config, BilinearStrassenModel, StrassenDataGenerator,
    CheckpointLoader, CheckpointMigrator
)

def estimate_gns(model, batch_size, num_batches=20):
    model.eval()
    grads = []
    losses = []
    for _ in range(num_batches):
        A, B, C = StrassenDataGenerator.generate_batch(batch_size)
        C_pred = model(A, B)
        loss = nn.functional.mse_loss(C_pred, C)
        grad = torch.autograd.grad(loss, model.parameters(), retain_graph=False)
        grad_vec = torch.cat([g.flatten() for g in grad])
        grads.append(grad_vec)
        losses.append(loss.item())
    
    grads_tensor = torch.stack(grads)  # (B, 21)
    mean_grad = grads_tensor.mean(dim=0)
    centered = grads_tensor - mean_grad
    cov = centered.T @ centered / (len(grads) - 1)
    cov += 1e-12 * torch.eye(cov.shape[0], device=cov.device)
    
    tr_Sigma = torch.trace(cov).item()
    grad_norm_sq = torch.norm(mean_grad).item() ** 2
    
    if grad_norm_sq < 1e-15:
        return float('inf')  # cristal exacto
    return tr_Sigma / grad_norm_sq

def main():
    loader = CheckpointLoader()
    results = defaultdict(list)
    
    for f in Path("sweep_checkpoints").glob("*.pt"):
        name = f.stem
        if "bs" in name:
            B = int(name.split("bs")[1].split("_")[0])
        else:
            continue
        
        raw = loader.load_checkpoint(str(f), Config.DEVICE)
        state = CheckpointMigrator.migrate_checkpoint(raw)
        model = BilinearStrassenModel().to(Config.DEVICE)
        model.load_state_dict(state)
        
        gns = estimate_gns(model, B)
        results[B].append(gns)
        print(f"{name}: B={B}, GNS={gns:.3e}")
    
    # Resumen
    for B in sorted(results.keys()):
        vals = [v for v in results[B] if v != float('inf')]
        if vals:
            mean_gns = sum(vals) / len(vals)
            print(f"B={B}: GNS_mean = {mean_gns:.3e} (N={len(vals)})")
        else:
            print(f"B={B}: todos son cristales exactos (GNS indefinido)")

if __name__ == "__main__":
    main()