# Subsystem: training

## src/training/convergence_theory.py
- Layer: utility
- Language: py
- Symbols:
  - `ConvergenceMetrics` (class, line 21) `class ConvergenceMetrics`
  - `HutchinsonTraceEstimator` (class, line 31) `class HutchinsonTraceEstimator`
  - `HardwareNoiseEstimator` (class, line 136) `class HardwareNoiseEstimator`
  - `convergence_theorem` (method, line 193) `def convergence_theorem()`
  - `verify_convergence_conditions` (method, line 265) `def verify_convergence_conditions(model, loss_fn, train_data, noise_threshold)`
  - `SimpleStrassenModel` (class, line 346) `class SimpleStrassenModel(Module)`
  - `__init__` (method, line 40) `def __init__(self, model, loss_fn, n_samples, device)`
  - `estimate_trace` (method, line 47) `def estimate_trace(self, data)`
  - `_rademacher_vector` (method, line 73) `def _rademacher_vector(self)`
  - `_hessian_vector_product` (method, line 90) `def _hessian_vector_product(self, x, y, v)`
  - `compute_kappa_eff` (method, line 118) `def compute_kappa_eff(self, data)`
  - `__init__` (method, line 144) `def __init__(self, model, loss_fn)`
  - `estimate_noise` (method, line 148) `def estimate_noise(self, data_loader, n_batches, n_threads)`
  - `__init__` (method, line 348) `def __init__(self, rank)`
  - `forward` (method, line 354) `def forward(self, x)`

## src/training/grokkit_physics.py
- Layer: utility
- Language: py
- Symbols:
  - `strassen_multiply` (function, line 49) `def strassen_multiply(A, B)`
  - `measure_physics` (function, line 64) `def measure_physics(N, num_samples)`
  - `detect_phase_transition` (function, line 126) `def detect_phase_transition(results)`
  - `main` (function, line 148) `def main()`

## src/training/main.py
- Layer: utility
- Language: py
- Symbols:
  - `Config` (class, line 44) `class Config`
  - `set_seed` (method, line 79) `def set_seed(seed)`
  - `StrassenDiscovery` (class, line 87) `class StrassenDiscovery(Module)`
  - `Matrix4x4Dataset` (class, line 199) `class Matrix4x4Dataset(Dataset)`
  - `Trainer` (class, line 217) `class Trainer`
  - `main` (method, line 345) `def main()`
  - `__init__` (method, line 96) `def __init__(self, num_slots)`
  - `forward` (method, line 108) `def forward(self, A, B)`
  - `get_slot_norms` (method, line 155) `def get_slot_norms(self)`
  - `get_active_slots` (method, line 160) `def get_active_slots(self)`
  - `mask_slot` (method, line 164) `def mask_slot(self, slot_idx)`
  - `get_weakest_slot` (method, line 169) `def get_weakest_slot(self)`
  - `print_coefficients` (method, line 175) `def print_coefficients(self)`
  - `__init__` (method, line 202) `def __init__(self, num_samples, seed)`
  - `__len__` (method, line 210) `def __len__(self)`
  - `__getitem__` (method, line 213) `def __getitem__(self, idx)`
  - `__init__` (method, line 220) `def __init__(self, config)`
  - `accuracy` (method, line 242) `def accuracy(self, pred, target)`
  - `train_epoch` (method, line 245) `def train_epoch(self, optimizer)`
  - `evaluate` (method, line 265) `def evaluate(self)`
  - `train` (method, line 280) `def train(self)`

## src/training/main_pure_math.py
- Layer: utility
- Language: py
- Symbols:
  - `StrassenModel` (class, line 22) `class StrassenModel(Module)`
  - `gen_data` (method, line 58) `def gen_data(n, scale)`
  - `train` (method, line 64) `def train(model, epochs, lr, l1, batch, verbose)`
  - `verify` (method, line 88) `def verify(model, n)`
  - `hard_prune` (method, line 106) `def hard_prune(model, keep)`
  - `refine_pruned` (method, line 122) `def refine_pruned(model, active, epochs, lr)`
  - `show_coeffs` (method, line 159) `def show_coeffs(model, active)`
  - `main` (method, line 186) `def main()`
  - `__init__` (method, line 28) `def __init__(self, rank)`
  - `forward` (method, line 35) `def forward(self, A, B)`
  - `slot_norms` (method, line 47) `def slot_norms(self)`
  - `active_count` (method, line 54) `def active_count(self, thresh)`

## src/training/strassen_core.py
- Layer: utility
- Language: py
- Symbols:
  - `_load_weights` (function, line 14) `def _load_weights()`
  - `strassen_2x2` (function, line 21) `def strassen_2x2(A, B)`
  - `strassen` (function, line 44) `def strassen(X, Y)`
  - `get_coefficients` (function, line 77) `def get_coefficients()`
  - `multiplication_count` (function, line 82) `def multiplication_count(n)`

## src/training/strassen_grokkit.py
- Layer: utility
- Language: py
- Symbols:
  - `StrassenOperator` (class, line 27) `class StrassenOperator(Module)`
  - `generate_batch` (method, line 113) `def generate_batch(n, scale)`
  - `train_grokkit` (method, line 120) `def train_grokkit(epochs, batch_size, lr, wd)`
  - `verify_grokking` (method, line 203) `def verify_grokking(model, n_test)`
  - `progressive_sparsification` (method, line 254) `def progressive_sparsification(model, target_slots)`
  - `main` (method, line 351) `def main()`
  - `__init__` (method, line 40) `def __init__(self, rank)`
  - `forward` (method, line 49) `def forward(self, A, B)`
  - `compute_LC` (method, line 67) `def compute_LC(self)`
  - `compute_SP` (method, line 83) `def compute_SP(self)`
  - `slot_importance` (method, line 101) `def slot_importance(self)`
  - `count_active` (method, line 108) `def count_active(self, threshold)`

## src/training/train_strassen.py
- Layer: utility
- Language: py
- Symbols:
  - `StrassenOperator` (class, line 26) `class StrassenOperator(Module)`
  - `generate_batch` (method, line 60) `def generate_batch(n, scale)`
  - `train_phase1` (method, line 66) `def train_phase1(epochs, batch_size, lr, wd)`
  - `sparsify` (method, line 104) `def sparsify(model, target_slots)`
  - `discretize` (method, line 183) `def discretize(model, slots_to_prune)`
  - `get_canonical_strassen` (method, line 211) `def get_canonical_strassen()`
  - `verify` (method, line 260) `def verify(U, V, W, n_test)`
  - `main` (method, line 299) `def main()`
  - `__init__` (method, line 33) `def __init__(self, rank)`
  - `forward` (method, line 40) `def forward(self, A, B)`
  - `slot_importance` (method, line 50) `def slot_importance(self)`
  - `count_active` (method, line 56) `def count_active(self, threshold)`
