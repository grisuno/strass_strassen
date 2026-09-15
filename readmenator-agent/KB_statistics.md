# Subsystem: statistics

## experiments/statistics/coherence_analysis.py
- Layer: utility
- Language: py
- Symbols:
  - `strassen_numpy` (function, line 15) `def strassen_numpy(A, B, threshold)`
  - `run_coherence_analysis` (function, line 42) `def run_coherence_analysis()`

## experiments/statistics/rigorous_experiment.py
- Layer: utility
- Language: py
- Symbols:
  - `ExperimentConfig` (class, line 58) `class ExperimentConfig`
  - `ExperimentResult` (class, line 84) `class ExperimentResult`
  - `StrassenModel` (class, line 106) `class StrassenModel(Module)`
  - `generate_data` (method, line 131) `def generate_data(n_samples, seed)`
  - `compute_discretization_error` (method, line 143) `def compute_discretization_error(model, values)`
  - `compute_spectral_gap` (method, line 157) `def compute_spectral_gap(model)`
  - `run_single_experiment` (method, line 169) `def run_single_experiment(batch_size, seed, run_id, config)`
  - `run_full_experiment` (method, line 269) `def run_full_experiment(batch_sizes, n_seeds, n_runs_per_seed)`
  - `perform_anova` (method, line 306) `def perform_anova(results)`
  - `print_anova_table` (method, line 401) `def print_anova_table(anova)`
  - `fit_noise_model` (method, line 448) `def fit_noise_model(results)`
  - `find_optimal_B` (method, line 519) `def find_optimal_B(results, n_bootstrap)`
  - `generate_report` (method, line 555) `def generate_report(results, config)`
  - `__init__` (method, line 108) `def __init__(self, config)`
  - `forward` (method, line 124) `def forward(self, x)`
  - `cache_miss_proxy` (method, line 467) `def cache_miss_proxy(B)`
  - `full_model` (method, line 472) `def full_model(B, alpha, beta, gamma)`
  - `null_model` (method, line 476) `def null_model(B, alpha, gamma)`
  - `get_mean_error` (method, line 526) `def get_mean_error(data, B)`
