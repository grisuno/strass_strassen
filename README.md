# Engineering Algorithmic Structure in Neural Networks: A Materials Science Perspective

<a herf="https://zenodo.org/records/18293019" target="_blank"><img width="191" height="20" alt="image" src="https://github.com/user-attachments/assets/8cb513de-461d-4122-827c-e3f9528df058" /></a> [![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**Author:** grisun0

---

## Abstract

This work presents what I learned from attempting to induce Strassen matrix multiplication structure in neural networks, and why I now view this work as materials engineering rather than theory.

I demonstrate through Strassen matrix multiplication that by controlling batch size, training duration, and regularization, I can induce discrete algorithmic structure that transfers zero-shot from 2x2 to 64x64 matrices. Under controlled conditions, 68% of runs crystallize into verifiable Strassen structure.

What I initially framed as a theory did not hold up to scrutiny. Post-hoc analysis revealed that κ (the condition number I proposed) correlates with success but does not predict it prospectively. The hypothesis was backwards: successful models have κ≈1, but models with κ≈1 are not guaranteed to succeed.

What remains valid is the engineering protocol itself. Here is what actually works: train with batch sizes in [24, 128], use weight decay ≥1e-4, run for 1000+ epochs, prune to 7 slots, round weights to integers. Do this, and you will induce Strassen structure with 68% probability.

Key results:
- **Engineering Protocol:** Working recipe with explicit tolerance windows
- **Success Rate:** 68% (133/195 runs) achieve verifiable Strassen structure
- **Statistical Validation:** N=195 observations, F(4,190)=15.34, p<0.001, eta-squared=0.244
- **Zero-Shot Transfer:** Learned coefficients work for 2x2 to 64x64 matrices
- **Strassen Rediscovery:** 1.95x speedup over OpenBLAS on 8192x8192 matrices (single-threaded)

**What I claimed vs what I demonstrated:**
- κ causes success: NOT DEMONSTRATED (58.3% prediction accuracy is chance)
- κ predicts outcome prospectively: NOT DEMONSTRATED
- Batch size explained by κ: NOT DEMONSTRATED (mechanism remains open)
- Fragility confirms narrow basin: DEMONSTRATED (0% success for σ≥0.001)
- Engineering protocol works: DEMONSTRATED (68% success rate)

---

## Repository Structure

```
.
├── paper/                          # Publication materials
│   ├── algorithmic_invariance.md   # Full paper (Markdown)
│   └── algorithmic_invariance.pdf  # Full paper (PDF)
│
├── src/                            # Source code
│   ├── training/                   # Training implementations
│   │   ├── main.py                 # Main training script
│   │   ├── strassen_grokkit.py     # Strassen grokking implementation
│   │   ├── convergence_theory.py   # Convergence analysis tools
│   │   └── train_strassen.py       # Strassen-specific training
│   │
│   ├── discovery/                  # T operator discovery
│   │   └── auto_T_discovery.py     # Automatic expansion operator discovery
│   │
│   ├── benchmarks/                 # Performance benchmarks
│   │   ├── benchmark_strassen.py   # Strassen vs BLAS benchmarks
│   │   ├── benchmark_scientific.py # Scientific validation benchmarks
│   │   └── strassen_numpy.py       # NumPy reference implementation
│   │
│   └── native/                     # Native C implementations
│       ├── strassen_c.c            # Basic C implementation
│       ├── strassen_optimal.c      # Cache-optimized implementation
│       └── strassen_turbo.c        # SIMD-accelerated implementation
│
├── checkpoints/                    # Pre-trained model weights
│   ├── strassen_discrete_final.pt  # Final discrete Strassen weights
│   ├── strassen_exact.pt           # Exact coefficient weights
│   ├── strassen_robust.pt          # Robust training weights
│   └── strassen_discovered.pt      # Auto-discovered weights
│
├── experiments/                    # Experimental results
│   ├── ablation/                   # Ablation studies
│   │   ├── ablation_study.py       # Main ablation script
│   │   └── ABLATION_STUDY.md       # Ablation results documentation
│   │
│   ├── statistics/                 # Statistical analysis
│   │   ├── rigorous_experiment.py  # ANOVA and hypothesis testing
│   │   ├── coherence_analysis.py   # Cache coherence analysis
│   │   └── scientific_data.json    # Raw experimental data
│   │
│   └── validation/                 # Validation experiments
│       └── planck_benchmark.py     # Hardware validation benchmarks
│
├── configs/                        # Configuration files
│   └── config.toml                 # Default hyperparameters
│
├── scripts/                        # Utility scripts
│   └── reproduce_experiments.py    # Reproduction script
│
├── requirements.txt                # Python dependencies
└── LICENSE                         # AGPL v3 License
```

---

## Installation

```bash
git clone https://github.com/grisuno/agi.git
cd agi
pip install -r requirements.txt
```

### Requirements

- Python >= 3.10
- PyTorch >= 2.1.0
- NumPy >= 1.24.0
- SciPy >= 1.10.0

---

## Quick Start

### 1. Train a Strassen Model

```bash
python src/training/main.py --batch_size 32 --epochs 1000
```

### 2. Verify Convergence

```python
from src.discovery.auto_T_discovery import discover_expansion_operator

# Load trained weights
weights = torch.load('checkpoints/strassen_discrete_final.pt')

# Discover expansion operator
T, metrics = discover_expansion_operator(weights)
print(f"Effective rank: {metrics['rank']}")
print(f"Discrete values: {metrics['discrete_values']}")
```

### 3. Benchmark Performance

```bash
python src/benchmarks/benchmark_strassen.py --sizes 2048 4096 8192
```

---

## Reproducing Paper Results

To reproduce all experiments from the paper:

```bash
python scripts/reproduce_experiments.py --batch_sizes 24 32 64 --seeds 5 --replicas 3
```

This will:
1. Run Protocol A (N=75): Discretization error across batch sizes
2. Run Protocol B (N=120): Expansion success across batch sizes
3. Compute ANOVA statistics
4. Generate Table 3 from the paper

Expected output:
```
Combined ANOVA (N=195):
  F(4, 189) = 15.34
  p < 0.0001
  eta-squared = 0.244
```

---

## Key Results

### Strassen Algorithm Rediscovery

The network learns the exact Strassen formulas:

| M_k | Formula |
|-----|---------|
| M1 | (A11 + A22)(B11 + B22) |
| M2 | (A21 + A22)B11 |
| M3 | A11(B12 - B22) |
| M4 | A22(B21 - B11) |
| M5 | (A11 + A12)B22 |
| M6 | (A21 - A11)(B11 + B12) |
| M7 | (A12 - A22)(B21 + B22) |

### Performance vs OpenBLAS

| Matrix Size | Strassen Time | BLAS Time | Speedup |
|-------------|---------------|-----------|---------|
| 2048 | 0.42s | 0.41s | 0.98x |
| 4096 | 19.8s | 22.4s | 1.13x |
| 8192 | 142s | 277s | 1.95x |

### Optimal Batch Size

The optimal batch size is a **range** empirically determined:
- B in [24, 128] correlates with successful discretization
- B outside this range: success drops to near zero
- **Mechanism remains open:** Initial cache coherence hypothesis was ruled out (B=1024 fits in L3)
- **κ correlation:** Post-hoc analysis shows κ≈1 for successful models, but κ does not enable prediction

---

## Limitations (What Did Not Work)

1. **κ as predictor:** Early-epoch κ does not predict final success (58.3% accuracy = chance)
2. **Batch size mechanism:** No validated explanation for why [24, 128] works
3. **Generalization:** Attempts on 3x3 matrices (Laderman's algorithm) failed
4. **Theory:** Gradient covariance hypothesis is post-hoc correlation, not causal theory

---

## Pre-trained Checkpoints

| Checkpoint | Description | Effective Rank |
|------------|-------------|----------------|
| `strassen_discrete_final.pt` | Final model with discrete {-1, 0, 1} coefficients | 7 |
| `strassen_exact.pt` | Exact Strassen coefficients | 7 |
| `strassen_robust.pt` | Robust training with noise injection | 7 |
| `strassen_grokkit.pt` | Grokking-optimized training | 7 |

---

## Citation

If you use this code, please cite:

```bibtex
@software{grisun0_2026_algorithmic_invariance,
  author       = {grisun0},
  title        = {Algorithmic Invariance and Zero-Shot Structural Scaling in Neural Networks},
  year         = {2026},
  publisher    = {Zenodo},
  doi          = {10.5281/zenodo.18263654},
  url          = {https://doi.org/10.5281/zenodo.18263654}
}
```

---

## License

This project is licensed under the GNU Affero General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

---

## Contact

- **Author:** grisun0
- **Repository:** https://github.com/grisuno/agi
- **DOI:** [https://zenodo.org/records/18293019](https://zenodo.org/records/18293019)
- **WIKI:** [https://deepwiki.com/grisuno/strass_strassen](https://deepwiki.com/grisuno/strass_strassen)

---

*Manuscript prepared: January 2026*


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
<a herf="https://zenodo.org/records/18293019" target="_blank"><img width="191" height="20" alt="image" src="https://github.com/user-attachments/assets/8cb513de-461d-4122-827c-e3f9528df058" /></a> 

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
