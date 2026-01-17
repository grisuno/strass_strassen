"""
Corrected cache analysis including model parameters and optimizer state.
"""

import json

def cache_analysis():
    """
    Full memory analysis for training.
    
    Model: Bilinear with rank-8 (target 7 active)
    - U: 8 x 4 = 32 floats
    - V: 8 x 4 = 32 floats  
    - W: 4 x 8 = 32 floats
    Total params: 96 floats = 384 bytes
    
    AdamW optimizer state (per param):
    - m (momentum): 96 floats
    - v (variance): 96 floats
    Total optimizer: 192 floats = 768 bytes
    
    Per-batch memory:
    - Input batch: B x 8 floats (A, B flattened)
    - Hidden: B x 8 floats (intermediate M)
    - Output: B x 4 floats (C)
    - Gradients: same as forward
    Total per-sample: ~40 floats x 2 (forward+backward) = 80 floats = 320 bytes
    
    Total training memory = model + optimizer + batch
    """
    
    sizeof_float = 4
    model_memory = 96 * sizeof_float  # 384 bytes
    optimizer_memory = 192 * sizeof_float  # 768 bytes
    per_sample_memory = 320  # bytes
    
    fixed_overhead = model_memory + optimizer_memory  # 1152 bytes
    
    batch_sizes = [8, 16, 24, 32, 48, 64, 96, 128, 256, 512, 1024]
    
    # L3 cache sizes
    l3_sizes = {
        "Intel Xeon E5 (tested)": 35 * 1024 * 1024,
        "Intel i7-8700": 12 * 1024 * 1024,
        "Apple M1/M2": 12 * 1024 * 1024,
        "Intel i5 (laptop)": 6 * 1024 * 1024,
        "Raspberry Pi 4": 1 * 1024 * 1024
    }
    
    print("CACHE COHERENCE ANALYSIS")
    print("=" * 60)
    print(f"Model parameters: {model_memory} bytes")
    print(f"Optimizer state: {optimizer_memory} bytes")
    print(f"Fixed overhead: {fixed_overhead} bytes")
    print(f"Per-sample: {per_sample_memory} bytes")
    print()
    
    results = {}
    
    for hw, l3 in l3_sizes.items():
        l3_mb = l3 / 1024 / 1024
        # Use 50% L3 as working set threshold (conservative)
        threshold = l3 * 0.5
        
        print(f"{hw} (L3={l3_mb:.0f}MB, threshold={threshold/1024:.0f}KB):")
        coherent = []
        
        for B in batch_sizes:
            total_memory = fixed_overhead + B * per_sample_memory
            fits = total_memory < threshold
            
            if fits:
                coherent.append(B)
                status = "COHERENT"
            else:
                status = "EXCEEDS"
            
            print(f"  B={B:4d}: {total_memory/1024:6.1f}KB [{status}]")
        
        if coherent:
            results[hw] = {"range": [min(coherent), max(coherent)], "l3_mb": l3_mb}
        else:
            results[hw] = {"range": None, "l3_mb": l3_mb}
        print()
    
    # The real insight: for this tiny model, cache is NOT the bottleneck
    print("=" * 60)
    print("KEY INSIGHT:")
    print("For this 2x2 model, L3 cache is NOT the constraint.")
    print("The observed [24, 128] optimal range is likely due to:")
    print("  1. Gradient noise (too small B = noisy, too large B = poor generalization)")
    print("  2. Training dynamics (batch size affects grokking transition)")
    print("  3. Learning rate / batch size coupling (linear scaling rule)")
    print()
    print("The cache hypothesis was incorrect for this model size.")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    results = cache_analysis()
    with open("cache_analysis_results.json", "w") as f:
        json.dump(results, f, indent=2)
