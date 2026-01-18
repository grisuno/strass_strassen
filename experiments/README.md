# Experiments

This directory contains experimental code, validation scripts, and post-hoc analysis for the engineering protocol.

## Directory Structure

```
experiments/
├── ablation/                   # Ablation studies
│   ├── ablation_study.py       # Main ablation script
│   └── ABLATION_STUDY.md       # Results documentation
│
├── statistics/                 # Statistical analysis
│   ├── rigorous_experiment.py  # ANOVA and hypothesis testing
│   ├── coherence_analysis.py   # Cache coherence analysis
│   └── scientific_data.json    # Raw experimental data
│
├── validation/                 # Validation experiments
│   └── benchmark.py            # Hardware validation benchmarks
│
├── reviewer_experiments/       # Post-hoc analysis (reviewer feedback)
│   ├── run_all_experiments.py  # Main runner for all experiments
│   ├── exp1-5_*.py             # Individual experiment scripts
│   └── all_experiments_results.json
│
└── verification/               # Checkpoint verification
    └── verify_checkpoints.py   # Verify trained models
```

## Key Findings

### What Was Demonstrated

1. **Engineering Protocol**: 68% success rate for inducing Strassen structure
2. **κ Correlation**: κ≈1 for discretized models, κ>>1 for non-discretized
3. **Fragility**: 0% success for noise σ≥0.001 (narrow basin)
4. **Zero-Shot Transfer**: Coefficients work 2x2 to 64x64

### What Was NOT Demonstrated

1. **κ as Predictor**: Prospective prediction accuracy = 58.3% (chance)
2. **Batch Size Mechanism**: Effect is real but unexplained
3. **Generalization**: 3x3 matrices (Laderman) failed to converge

## Running Experiments

```bash
# Run all reviewer experiments
python experiments/reviewer_experiments/run_all_experiments.py

# Verify checkpoints
python experiments/verify_checkpoints.py

# Run ablation study
python experiments/ablation/ablation_study.py
```

## Post-Hoc κ Analysis (Appendix H)

Following reviewer feedback, we conducted post-hoc experiments on 12 checkpoints to validate the gradient covariance hypothesis. Results:

- κ correlates with discretization status (κ≈1 for discretized)
- κ does not enable prospective prediction
- The mechanism remains open

See `paper/algorithmic_invariance.md` Section 7.6 for full analysis.
