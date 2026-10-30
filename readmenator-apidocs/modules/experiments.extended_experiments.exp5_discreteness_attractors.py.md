# `experiments.extended_experiments.exp5_discreteness_attractors`

> Source: [experiments/extended_experiments/exp5_discreteness_attractors.py:1](../../experiments/extended_experiments/exp5_discreteness_attractors.py#L1) | language `py` | coverage 100.0% | [API index](../index.md)

EXPERIMENT 5: Discreteness Attractors Measurement =================================================== Characterizes whether weights converge toward discrete basins.

Protocol: 1. Post-training analysis on successful runs: - Compute Hessian at final solution - Perturb along eigenvectors - Measure how discretization margin changes  2. Basin volume measurement: - Maximum sphere around θ where δ < 0.1 - Compare basin size vs batch size  3. Discrete structure stability: - Fine-tuning from discrete weights - Do they return to discrete or diverge?  Author: MiniMax Agent
