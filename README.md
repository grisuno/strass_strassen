# Algorithmic Invariance and Zero-Shot Structural Scaling in Neural Networks

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18263654.svg)](https://zenodo.org/records/18277664)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

**Author:** grisun0

---

## Abstract

We study a class of training dynamics in neural networks where, after sufficient optimization, the learned solution exhibits strong structural invariance. Once this regime is reached, the model can be deterministically expanded to significantly larger input dimensions without retraining, while preserving perfect or near-perfect generalization.

Key results:
- **Convergence Theorem:** Conditions under which SGD converges to algorithmically invariant solutions
- **Uniqueness of T:** The expansion operator is unique up to permutation symmetry
- **Automatic T Discovery:** Algorithm for constructing T from converged weights
- **Statistical Validation:** N=195 observations, F(4,190)=15.34, p<0.001, eta-squared=0.244
- **Strassen Rediscovery:** 1.95x speedup over OpenBLAS on 8192x8192 matrices

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

The optimal batch size is a **range** determined by cache coherence:
- L1 cache (32KB): B* in [32, 64]
- L2 cache (256KB): B* in [64, 128]

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
- **DOI:** [https://zenodo.org/records/18277664](https://zenodo.org/records/18277664)
- **WIKI:** [https://deepwiki.com/grisuno/strass_strassen](https://deepwiki.com/grisuno/strass_strassen)

---

*Manuscript prepared: January 2026*


![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18072859.svg)](https://zenodo.org/records/18277664)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
