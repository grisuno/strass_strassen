# Subsystem: misc

## experiments/validation/benchmark.py
- Layer: utility
- Language: py
- Symbols:
  - `strassen_numpy` (function, line 15) `def strassen_numpy(A, B, threshold)`
  - `measure_single_sgemm` (function, line 49) `def measure_single_sgemm(n, threads)`
  - `run_planck_analysis` (function, line 59) `def run_planck_analysis()`

## src/discovery/auto_T_discovery.py
- Layer: infrastructure
- Language: py
- Symbols:
  - `SymmetryStructure` (class, line 22) `class SymmetryStructure`
  - `AutoTDiscovery` (class, line 32) `class AutoTDiscovery`
  - `verify_strassen_T` (method, line 306) `def verify_strassen_T(model_path, target_sizes)`
  - `verify_expanded_correctness` (method, line 353) `def verify_expanded_correctness(U, V, W, target_size, expanded)`
  - `recursive_strassen_multiply` (method, line 389) `def recursive_strassen_multiply(A, B, U, V, W, base_size)`
  - `__init__` (method, line 42) `def __init__(self, tolerance, verbose)`
  - `analyze_structure` (method, line 46) `def analyze_structure(self, W)`
  - `_detect_discrete_values` (method, line 89) `def _detect_discrete_values(self, W_flat)`
  - `_detect_block_structure` (method, line 105) `def _detect_block_structure(self, W)`
  - `_block_repetition_score` (method, line 127) `def _block_repetition_score(self, W, bm, bn)`
  - `_detect_symmetry_type` (method, line 144) `def _detect_symmetry_type(self, W)`
  - `_is_permutation_symmetric` (method, line 165) `def _is_permutation_symmetric(self, W)`
  - `_is_cyclic` (method, line 171) `def _is_cyclic(self, W)`
  - `_invariant_subspace_dim` (method, line 184) `def _invariant_subspace_dim(self, U, S, rank)`
  - `_discretization_error` (method, line 198) `def _discretization_error(self, W, values)`
  - `_print_analysis` (method, line 213) `def _print_analysis(self, W, S, structure)`
  - `construct_T` (method, line 229) `def construct_T(self, W_dict, target_size)`
  - `_validate_expansion` (method, line 290) `def _validate_expansion(self, expanded, structure)`
