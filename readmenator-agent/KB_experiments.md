# Subsystem: experiments

## experiments/apendix_experiments.py
- Layer: infrastructure
- Language: py
- Symbols:
  - `setup_matplotlib` (function, line 34) `def setup_matplotlib()`
  - `StrassenOperator` (class, line 51) `class StrassenOperator(Module)`
  - `generate_batch` (method, line 87) `def generate_batch(n, device)`
  - `generate_test_set` (method, line 93) `def generate_test_set(n, device)`
  - `compute_delta` (method, line 100) `def compute_delta(model)`
  - `verify_strassen_structure` (method, line 117) `def verify_strassen_structure(U_disc, V_disc, W_disc, tolerance)`
  - `compute_S_theta` (method, line 138) `def compute_S_theta(model)`
  - `compute_gradient_covariance` (method, line 154) `def compute_gradient_covariance(model, batch_size, n_samples)`
  - `train_with_logging` (method, line 187) `def train_with_logging(batch_size, total_epochs, lr, wd, symmetric_init, seed, log_interval)`
  - `sparsify_and_discretize` (method, line 274) `def sparsify_and_discretize(model, batch_size)`
  - `run_phase_diagram` (method, line 327) `def run_phase_diagram()`
  - `run_batch_size_effect` (method, line 429) `def run_batch_size_effect()`
  - `main` (method, line 526) `def main()`
  - `__init__` (method, line 53) `def __init__(self, rank, symmetric_init)`
  - `forward` (method, line 67) `def forward(self, A, B)`
  - `slot_importance` (method, line 77) `def slot_importance(self)`
  - `count_active` (method, line 83) `def count_active(self, threshold)`

## experiments/cache_analysis_v2.py
- Layer: infrastructure
- Language: py
- Symbols:
  - `cache_analysis` (function, line 7) `def cache_analysis()`

## experiments/generate_figures.py
- Layer: utility
- Language: py
- Symbols:
  - `setup_matplotlib_for_plotting` (function, line 15) `def setup_matplotlib_for_plotting()`
  - `generate_benchmark_figure` (function, line 63) `def generate_benchmark_figure()`
  - `generate_ablation_figure` (function, line 124) `def generate_ablation_figure()`
  - `load_checkpoint_weights` (function, line 219) `def load_checkpoint_weights()`
  - `generate_weight_geometry_figure` (function, line 258) `def generate_weight_geometry_figure()`
  - `generate_phase_transition_figure` (function, line 354) `def generate_phase_transition_figure()`
  - `generate_coherence_figure` (function, line 465) `def generate_coherence_figure()`
  - `generate_crystallization_figure` (function, line 534) `def generate_crystallization_figure()`
  - `main` (function, line 623) `def main()`

## experiments/validation_experiments.py
- Layer: utility
- Language: py
- Symbols:
  - `strassen_2x2` (function, line 47) `def strassen_2x2(A, B, U, V, W)`
  - `strassen_recursive` (function, line 56) `def strassen_recursive(A, B, U, V, W, threshold)`
  - `test_uniqueness_via_permutation` (function, line 85) `def test_uniqueness_via_permutation()`
  - `test_noise_stability` (function, line 125) `def test_noise_stability()`
  - `test_expansion_sizes` (function, line 162) `def test_expansion_sizes()`
  - `simulate_grokking_dynamics` (function, line 184) `def simulate_grokking_dynamics()`
  - `compute_cache_math` (function, line 250) `def compute_cache_math()`
  - `main` (function, line 294) `def main()`
  - `convert_types` (function, line 313) `def convert_types(obj)`

## experiments/verify_checkpoints.py
- Layer: utility
- Language: py
- Symbols:
  - `StrassenBilinear` (class, line 25) `class StrassenBilinear(Module)`
  - `compute_delta` (method, line 51) `def compute_delta(model)`
  - `verify_2x2` (method, line 68) `def verify_2x2(U, V, W, n_test)`
  - `strassen_expand` (method, line 89) `def strassen_expand(A, B, U, V, W)`
  - `verify_expansion` (method, line 126) `def verify_expansion(U, V, W, sizes)`
  - `compute_S_theta` (method, line 152) `def compute_S_theta(model)`
  - `load_checkpoint` (method, line 166) `def load_checkpoint(path)`
  - `verify_checkpoint` (method, line 183) `def verify_checkpoint(checkpoint_path)`
  - `run_noise_stability_test` (method, line 226) `def run_noise_stability_test(checkpoint_path, noise_levels)`
  - `main` (method, line 249) `def main()`
  - `__init__` (method, line 27) `def __init__(self, rank)`
  - `forward` (method, line 34) `def forward(self, A, B)`
  - `get_discrete_coefficients` (method, line 44) `def get_discrete_coefficients(self)`
