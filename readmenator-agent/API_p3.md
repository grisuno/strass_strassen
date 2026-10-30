# API (page 3 of 3)
Previous: [API_p2.md](API_p2.md)

## src/benchmarks/benchmark_scientific.py
- `strassen_multiply` (function) `src/benchmarks/benchmark_scientific.py:38` `def strassen_multiply(A, B)`
- `standard_avx512_multiply` (function) `src/benchmarks/benchmark_scientific.py:51` `def standard_avx512_multiply(A, B)`
- `numpy_multiply` (function) `src/benchmarks/benchmark_scientific.py:64` `def numpy_multiply(A, B)`
- `benchmark_function` (function) `src/benchmarks/benchmark_scientific.py:67` `def benchmark_function(func, A, B, runs, warmup)` -- Benchmark with statistical analysis
- `main` (function) `src/benchmarks/benchmark_scientific.py:91` `def main()`

## src/benchmarks/benchmark_strassen.py
- `load_config` (function) `src/benchmarks/benchmark_strassen.py:61` `def load_config(config_path)` -- Load configuration from TOML file.
- `get_dtype` (function) `src/benchmarks/benchmark_strassen.py:93` `def get_dtype(dtype_str)` -- Convert dtype string to torch dtype.
- `estimate_memory_mb` (function) `src/benchmarks/benchmark_strassen.py:104` `def estimate_memory_mb(n, dtype, batch_size)` -- Estimate memory usage for matrix multiplication.
- `benchmark_resolution` (function) `src/benchmarks/benchmark_strassen.py:114` `def benchmark_resolution(n, cfg, dtype)` -- Benchmark Strassen vs standard matmul for given resolution.
- `run_benchmark` (function) `src/benchmarks/benchmark_strassen.py:218` `def run_benchmark(cfg)` -- Run full benchmark suite.
- `save_results` (function) `src/benchmarks/benchmark_strassen.py:310` `def save_results(results, filepath)` -- Save benchmark results to JSON.
- `main` (function) `src/benchmarks/benchmark_strassen.py:322` `def main()`

## src/benchmarks/strassen_numpy.py
- `strassen_2x2_numpy` (function) `src/benchmarks/strassen_numpy.py:29` `def strassen_2x2_numpy(A, B)` -- Strassen 2x2 using grokked coefficients.
- `strassen_numpy` (function) `src/benchmarks/strassen_numpy.py:44` `def strassen_numpy(A, B)` -- Recursive Strassen using NumPy.
- `strassen_hybrid` (function) `src/benchmarks/strassen_numpy.py:78` `def strassen_hybrid(A, B, threshold)` -- Hybrid Strassen: use Strassen for large matrices, NumPy for small.
- `multiplication_count` (function) `src/benchmarks/strassen_numpy.py:112` `def multiplication_count(n)` -- Count multiplications used by Strassen.

## src/discovery/auto_T_discovery.py
- `AutoTDiscovery.__init__` (method) `src/discovery/auto_T_discovery.py:42` `def __init__(self, tolerance, verbose)`
- `AutoTDiscovery.analyze_structure` (method) `src/discovery/auto_T_discovery.py:46` `def analyze_structure(self, W)` -- Phase 1 & 2: Analyze weight matrix structure
- `AutoTDiscovery.construct_T` (method) `src/discovery/auto_T_discovery.py:229` `def construct_T(self, W_dict, target_size)` -- Phase 3: Construct expansion operator T
- `verify_strassen_T` (function) `src/discovery/auto_T_discovery.py:306` `def verify_strassen_T(model_path, target_sizes)` -- Verify T discovery on Strassen model
- `verify_expanded_correctness` (function) `src/discovery/auto_T_discovery.py:353` `def verify_expanded_correctness(U, V, W, target_size, expanded)` -- Verify that expanded operator correctly computes matrix multiplication
- `recursive_strassen_multiply` (function) `src/discovery/auto_T_discovery.py:389` `def recursive_strassen_multiply(A, B, U, V, W, base_size)` -- Recursively apply learned Strassen decomposition

## src/native/strassen_c.c
- `alloc_matrix` (function) `src/native/strassen_c.c:15` `static float* alloc_matrix(int n)` -- Strassen Matrix Multiplication - C Implementation Author: grisun0  Compila: gcc -O3 -ffast-math -march=native...
- `matmul_standard` (function) `src/native/strassen_c.c:20` `static void matmul_standard(float* C, float* A, float* B, int n)` -- #include <stdlib.h> #include <string.h> #include <stdio.h> #define THRESHOLD 64 /* Allocate matrix static float*...
- `mat_add` (function) `src/native/strassen_c.c:33` `static void mat_add(float* C, float* A, float* B, int n)` -- /* Standard matrix multiplication for small matrices static void matmul_standard(float* C, float* A, float* B, int...
- `mat_sub` (function) `src/native/strassen_c.c:41` `static void mat_sub(float* C, float* A, float* B, int n)` -- } } } } /* Add matrices: C = A + B static void mat_add(float* C, float* A, float* B, int n) { int nn = n * n; for...
- `extract_quadrant` (function) `src/native/strassen_c.c:49` `static void extract_quadrant(float* Q, float* M, int n, int row, int col)` -- for (int i = 0; i < nn; i++) { C[i] = A[i] + B[i]; } } /* Subtract matrices: C = A - B static void mat_sub(float* C...
- `insert_quadrant` (function) `src/native/strassen_c.c:57` `static void insert_quadrant(float* M, float* Q, int n, int row, int col)` -- for (int i = 0; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Extract quadrant from matrix static void...
- `strassen_recursive` (function) `src/native/strassen_c.c:65` `void strassen_recursive(float* C, float* A, float* B, int n)` -- for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant...
- `strassen_multiply` (function) `src/native/strassen_c.c:164` `void strassen_multiply(float* C, float* A, float* B, int n)` -- insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0)...
- `standard_multiply` (function) `src/native/strassen_c.c:169` `void standard_multiply(float* C, float* A, float* B, int n)` -- /* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1)...

## src/native/strassen_optimal.c
- `strassen_level` (function) `src/native/strassen_optimal.c:18` `static void strassen_level(float* C, float* A, float* B, int n, 
                           float...` -- Uses in-place operations where possible and only applies Strassen for very large matrices where the asymptotic...
- `strassen_optimal` (function) `src/native/strassen_optimal.c:131` `void strassen_optimal(float* C, float* A, float* B, int n)`

## src/native/strassen_turbo.c
- `alloc_matrix` (function) `src/native/strassen_turbo.c:25` `static inline float* alloc_matrix(int n)` -- Compile: gcc -O3 -ffast-math -march=native -fopenmp -mavx2 -shared -fPIC -o libstrassen_turbo.so strassen_turbo.c...
- `mat_add_avx` (function) `src/native/strassen_turbo.c:30` `static void mat_add_avx(float* __restrict C, const float* __restrict A, 
                        ...` -- #include <stdio.h> #include <omp.h> #include <immintrin.h> #define THRESHOLD 128 #define BLOCK_SIZE 32 #define ALIGN...
- `mat_sub_avx` (function) `src/native/strassen_turbo.c:50` `static void mat_sub_avx(float* __restrict C, const float* __restrict A, 
                        ...` -- for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc =...
- `matmul_blocked_avx` (function) `src/native/strassen_turbo.c:68` `static void matmul_blocked_avx(float* __restrict C, const float* __restrict A, 
                 ...` -- for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc =...
- `extract_quadrant` (function) `src/native/strassen_turbo.c:104` `static void extract_quadrant(float* __restrict Q, const float* __restrict M, 
                   ...` -- _mm256_storeu_ps(&C[i * n + j], vc); } for (; j < j_end; j++) { C[i * n + j] += a_ik * B[k * n + j]; } } } } } } }...
- `insert_quadrant` (function) `src/native/strassen_turbo.c:114` `static void insert_quadrant(float* __restrict M, const float* __restrict Q, 
                    ...` -- } } /* Extract quadrant static void extract_quadrant(float* __restrict Q, const float* __restrict M, int n, int row...
- `strassen_turbo_recursive` (function) `src/native/strassen_turbo.c:124` `void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)` -- } } /* Insert quadrant static void insert_quadrant(float* __restrict M, const float* __restrict Q, int n, int row...
- `strassen_turbo` (function) `src/native/strassen_turbo.c:261` `void strassen_turbo(float* C, float* A, float* B, int n)` -- insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0)...
- `get_num_threads` (function) `src/native/strassen_turbo.c:267` `int get_num_threads(void)` -- free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2)...

## src/training/convergence_theory.py
- `HutchinsonTraceEstimator.__init__` (method) `src/training/convergence_theory.py:40` `def __init__(self, model, loss_fn, n_samples, device)`
- `HutchinsonTraceEstimator.estimate_trace` (method) `src/training/convergence_theory.py:47` `def estimate_trace(self, data)` -- Estimate tr(H) using Hutchinson's stochastic trace estimator.
- `HutchinsonTraceEstimator.compute_kappa_eff` (method) `src/training/convergence_theory.py:118` `def compute_kappa_eff(self, data)` -- Compute κ_eff = -tr(H) / N
- `HardwareNoiseEstimator.__init__` (method) `src/training/convergence_theory.py:144` `def __init__(self, model, loss_fn)`
- `HardwareNoiseEstimator.estimate_noise` (method) `src/training/convergence_theory.py:148` `def estimate_noise(self, data_loader, n_batches, n_threads)` -- Estimate hardware noise by measuring gradient variance across batches
- `convergence_theorem` (function) `src/training/convergence_theory.py:193` `def convergence_theorem()` -- THEOREM (Convergence to Algorithmic Invariance)
- `verify_convergence_conditions` (function) `src/training/convergence_theory.py:265` `def verify_convergence_conditions(model, loss_fn, train_data, noise_threshold)` -- Verify that convergence conditions are satisfied for a trained model.
- `SimpleStrassenModel.__init__` (method) `src/training/convergence_theory.py:348` `def __init__(self, rank)`
- `SimpleStrassenModel.forward` (method) `src/training/convergence_theory.py:354` `def forward(self, x)`

## src/training/grokkit_physics.py
- `strassen_multiply` (function) `src/training/grokkit_physics.py:49` `def strassen_multiply(A, B)` -- Wrapper for Strassen multiplication (uses float32).
- `measure_physics` (function) `src/training/grokkit_physics.py:64` `def measure_physics(N, num_samples)` -- Measure the 'physical quantities' for a given matrix size.
- `detect_phase_transition` (function) `src/training/grokkit_physics.py:126` `def detect_phase_transition(results)` -- Find the critical size N_c where the phase transition occurs.
- `main` (function) `src/training/grokkit_physics.py:148` `def main()`

## src/training/main.py
- `set_seed` (function) `src/training/main.py:79` `def set_seed(seed)` -- Fijar semilla para reproducibilidad.
- `StrassenDiscovery.__init__` (method) `src/training/main.py:96` `def __init__(self, num_slots)`
- `StrassenDiscovery.forward` (method) `src/training/main.py:108` `def forward(self, A, B)` -- Forward pass con multiplicación matemática pura.
- `StrassenDiscovery.get_slot_norms` (method) `src/training/main.py:155` `def get_slot_norms(self)` -- Norma promedio de cada slot.
- `StrassenDiscovery.get_active_slots` (method) `src/training/main.py:160` `def get_active_slots(self)` -- Número de slots activos.
- `StrassenDiscovery.mask_slot` (method) `src/training/main.py:164` `def mask_slot(self, slot_idx)` -- Desactiva un slot.
- `StrassenDiscovery.get_weakest_slot` (method) `src/training/main.py:169` `def get_weakest_slot(self)` -- Slot con menor norma entre los activos.
- `StrassenDiscovery.print_coefficients` (method) `src/training/main.py:175` `def print_coefficients(self)` -- Muestra coeficientes descubiertos.
- `Matrix4x4Dataset.__init__` (method) `src/training/main.py:202` `def __init__(self, num_samples, seed)`
- `Trainer.__init__` (method) `src/training/main.py:220` `def __init__(self, config)`
- `Trainer.accuracy` (method) `src/training/main.py:242` `def accuracy(self, pred, target)`
- `Trainer.train_epoch` (method) `src/training/main.py:245` `def train_epoch(self, optimizer)`
- `Trainer.evaluate` (method) `src/training/main.py:265` `def evaluate(self)`
- `Trainer.train` (method) `src/training/main.py:280` `def train(self)`
- `main` (function) `src/training/main.py:345` `def main()`

## src/training/main_pure_math.py
- `StrassenModel.__init__` (method) `src/training/main_pure_math.py:28` `def __init__(self, rank)`
- `StrassenModel.forward` (method) `src/training/main_pure_math.py:35` `def forward(self, A, B)`
- `StrassenModel.slot_norms` (method) `src/training/main_pure_math.py:47` `def slot_norms(self)` -- Norma combinada de cada slot.
- `StrassenModel.active_count` (method) `src/training/main_pure_math.py:54` `def active_count(self, thresh)`
- `gen_data` (function) `src/training/main_pure_math.py:58` `def gen_data(n, scale)`
- `train` (function) `src/training/main_pure_math.py:64` `def train(model, epochs, lr, l1, batch, verbose)`
- `verify` (function) `src/training/main_pure_math.py:88` `def verify(model, n)`
- `hard_prune` (function) `src/training/main_pure_math.py:106` `def hard_prune(model, keep)` -- Poda los slots más débiles, mantiene top-k.
- `refine_pruned` (function) `src/training/main_pure_math.py:122` `def refine_pruned(model, active, epochs, lr)` -- Refina manteniendo slots podados en cero.
- `show_coeffs` (function) `src/training/main_pure_math.py:159` `def show_coeffs(model, active)`
- `main` (function) `src/training/main_pure_math.py:186` `def main()`

## src/training/strassen_core.py
- `strassen_2x2` (function) `src/training/strassen_core.py:21` `def strassen_2x2(A, B)`
- `strassen` (function) `src/training/strassen_core.py:44` `def strassen(X, Y)`
- `get_coefficients` (function) `src/training/strassen_core.py:77` `def get_coefficients()`
- `multiplication_count` (function) `src/training/strassen_core.py:82` `def multiplication_count(n)`

## src/training/strassen_grokkit.py
- `StrassenOperator.__init__` (method) `src/training/strassen_grokkit.py:40` `def __init__(self, rank)`
- `StrassenOperator.forward` (method) `src/training/strassen_grokkit.py:49` `def forward(self, A, B)` -- Computa A @ B usando la descomposición tensorial.
- `StrassenOperator.compute_LC` (method) `src/training/strassen_grokkit.py:67` `def compute_LC(self)` -- Linear Combination metric.
- `StrassenOperator.compute_SP` (method) `src/training/strassen_grokkit.py:83` `def compute_SP(self)` -- Sparsity metric.
- `StrassenOperator.slot_importance` (method) `src/training/strassen_grokkit.py:101` `def slot_importance(self)` -- Importancia de cada slot basada en normas.
- `StrassenOperator.count_active` (method) `src/training/strassen_grokkit.py:108` `def count_active(self, threshold)` -- Cuenta slots activos.
- `generate_batch` (function) `src/training/strassen_grokkit.py:113` `def generate_batch(n, scale)` -- Genera batch de matrices aleatorias.
- `train_grokkit` (function) `src/training/strassen_grokkit.py:120` `def train_grokkit(epochs, batch_size, lr, wd)` -- Entrena usando el framework Grokkit.
- `verify_grokking` (function) `src/training/strassen_grokkit.py:203` `def verify_grokking(model, n_test)` -- Verifica que el operador ha grokkeado correctamente.
- `progressive_sparsification` (function) `src/training/strassen_grokkit.py:254` `def progressive_sparsification(model, target_slots)` -- Fase 2: Esparsificación progresiva.
- `main` (function) `src/training/strassen_grokkit.py:351` `def main()` -- Pipeline principal Grokkit para Strassen.

## src/training/train_strassen.py
- `StrassenOperator.__init__` (method) `src/training/train_strassen.py:33` `def __init__(self, rank)`
- `StrassenOperator.forward` (method) `src/training/train_strassen.py:40` `def forward(self, A, B)`
- `StrassenOperator.slot_importance` (method) `src/training/train_strassen.py:50` `def slot_importance(self)`
- `StrassenOperator.count_active` (method) `src/training/train_strassen.py:56` `def count_active(self, threshold)`
- `generate_batch` (function) `src/training/train_strassen.py:60` `def generate_batch(n, scale)`
- `train_phase1` (function) `src/training/train_strassen.py:66` `def train_phase1(epochs, batch_size, lr, wd)` -- Phase 1: Grokking with Weight Decay as thermodynamic pressure.
- `sparsify` (function) `src/training/train_strassen.py:104` `def sparsify(model, target_slots)` -- Phase 2: Progressive sparsification to target rank.
- `discretize` (function) `src/training/train_strassen.py:183` `def discretize(model, slots_to_prune)` -- Phase 3: Discretize coefficients to {-1, 0, 1}.
- `get_canonical_strassen` (function) `src/training/train_strassen.py:211` `def get_canonical_strassen()` -- Returns the canonical Strassen coefficients.
- `verify` (function) `src/training/train_strassen.py:260` `def verify(U, V, W, n_test)` -- Verify the discretized operator.
- `main` (function) `src/training/train_strassen.py:299` `def main()` -- Main training pipeline.

## superposition.py
- `ICheckpointLoader.load_checkpoint` (method) `superposition.py:60` `def load_checkpoint(self, path, device)`
- `IMetricsCalculator.compute` (method) `superposition.py:64` `def compute(self)`
- `IAnalyzer.analyze_checkpoint` (method) `superposition.py:68` `def analyze_checkpoint(self, checkpoint_path)`
- `CheckpointLoader.load_checkpoint` (method) `superposition.py:78` `def load_checkpoint(self, path, device)`
- `CheckpointMigrator.detect_hidden_dim` (method) `superposition.py:89` `def detect_hidden_dim(raw_data)` -- Detect hidden dimension from checkpoint data structure by inspecting tensor shapes in various known formats.
- `CheckpointMigrator.migrate_checkpoint` (method) `superposition.py:122` `def migrate_checkpoint(raw_data)`
- `get_tensor` (function) `superposition.py:148` `def get_tensor(key)`
- `StrassenDataGenerator.__init__` (method) `superposition.py:223` `def __init__(self, config)`
- `StrassenDataGenerator.generate_batch` (method) `superposition.py:226` `def generate_batch(self, batch_size)` -- Generate batch of matrix pairs and their product.
- `StrassenDataGenerator.generate_dataset` (method) `superposition.py:240` `def generate_dataset(self, num_samples)` -- Generate full dataset.
- `BilinearStrassenModel.__init__` (method) `superposition.py:261` `def __init__(self, config)`
- `BilinearStrassenModel.forward` (method) `superposition.py:276` `def forward(self, a, b)` -- Forward pass returning output and bottleneck activations.
- `BilinearStrassenModel.get_coefficients` (method) `superposition.py:284` `def get_coefficients(self)`
- `SparseAutoencoder.__init__` (method) `superposition.py:298` `def __init__(self, config)`
- `SparseAutoencoder.encode` (method) `superposition.py:310` `def encode(self, x)` -- Encode bottleneck activations to sparse features. x: [batch, N] W_enc: [D, N] Returns: [batch, D]
- `SparseAutoencoder.decode` (method) `superposition.py:319` `def decode(self, z)` -- Decode sparse features back to bottleneck. z: [batch, D] W_enc: [D, N] Returns: [batch, N]
- `SparseAutoencoder.forward` (method) `superposition.py:328` `def forward(self, x)`
- `SuperpositionMetrics.__init__` (method) `superposition.py:339` `def __init__(self, config)`
- `SuperpositionMetrics.compute_feature_probabilities` (method) `superposition.py:342` `def compute_feature_probabilities(self, sae_activations)` -- Calculate feature probabilities from SAE activations.
- `SuperpositionMetrics.compute_entropy` (method) `superposition.py:352` `def compute_entropy(self, probabilities)` -- Shannon entropy H(p) = -Σ p_i log p_i.
- `SuperpositionMetrics.compute_superposition` (method) `superposition.py:359` `def compute_superposition(self, sae_activations)` -- Main metric: ψ = F/N where F = e^{H(p)}.
- `SuperpositionMetrics.compute_frobenius_metric` (method) `superposition.py:377` `def compute_frobenius_metric(self, weight_matrix)` -- Baseline from Eq 2: ψ_Frob = ||W||_F^2 / N.
- `SuperpositionMetrics.compute_interference_matrix` (method) `superposition.py:385` `def compute_interference_matrix(self, weight_matrix)` -- Compute W^T @ W to analyze interference patterns.
- `SuperpositionMetrics.compute` (method) `superposition.py:389` `def compute(self, sae_activations, weight_matrix)` -- Unified interface.
- `SAETrainer.__init__` (method) `superposition.py:410` `def __init__(self, sae, config)`
- `SAETrainer.train` (method) `superposition.py:421` `def train(self, bottleneck_activations)` -- Train SAE on extracted activations.
- `StrassenCheckpointAnalyzer.__init__` (method) `superposition.py:480` `def __init__(self, config)`
- `StrassenCheckpointAnalyzer.load_model` (method) `superposition.py:496` `def load_model(self, checkpoint_path)` -- Load and migrate checkpoint to model.
- `StrassenCheckpointAnalyzer.extract_bottleneck_activations` (method) `superposition.py:554` `def extract_bottleneck_activations(self, model)` -- Extract bottleneck activations (U(a) * V(b)) from model.
- `StrassenCheckpointAnalyzer.analyze_checkpoint` (method) `superposition.py:571` `def analyze_checkpoint(self, checkpoint_path)` -- Full analysis pipeline for a single checkpoint: 1.
- `StrassenCheckpointAnalyzer.analyze_directory` (method) `superposition.py:641` `def analyze_directory(self, checkpoint_dir)` -- Analyze all checkpoints in directory.
- `main` (function) `superposition.py:765` `def main()`

## train_batch_sweep.py
- `train_for_batch_size` (function) `train_batch_sweep.py:11` `def train_for_batch_size(B, seed, output_dir)`

## unified_hidden_connections_suite.py
- `StrassStrassenModel.__init__` (method) `unified_hidden_connections_suite.py:190` `def __init__(self, config)`
- `StrassStrassenModel.forward` (method) `unified_hidden_connections_suite.py:203` `def forward(self, A, B)`
- `StrassStrassenModel.get_coefficients` (method) `unified_hidden_connections_suite.py:213` `def get_coefficients(self)`
- `StrassStrassenModel.get_flat_parameters` (method) `unified_hidden_connections_suite.py:216` `def get_flat_parameters(self)`
- `StrassStrassenModel.slot_importance` (method) `unified_hidden_connections_suite.py:219` `def slot_importance(self)`
- `StrassStrassenModel.count_active_slots` (method) `unified_hidden_connections_suite.py:225` `def count_active_slots(self, threshold)`
- `IDataGenerator.generate_batch` (method) `unified_hidden_connections_suite.py:232` `def generate_batch(self, batch_size)`
- `StrassenDataGenerator.__init__` (method) `unified_hidden_connections_suite.py:238` `def __init__(self, config)`
- `StrassenDataGenerator.generate_batch` (method) `unified_hidden_connections_suite.py:241` `def generate_batch(self, batch_size)`
- `ICheckpointManager.save` (method) `unified_hidden_connections_suite.py:265` `def save(self, model, epoch, metrics, path)`
- `ICheckpointManager.load` (method) `unified_hidden_connections_suite.py:266` `def load(self, path, model)`
- `CheckpointManager.save` (method) `unified_hidden_connections_suite.py:272` `def save(self, model, epoch, metrics, path)`
- `CheckpointManager.load` (method) `unified_hidden_connections_suite.py:284` `def load(self, path, model)`
- `ITrainer.train` (method) `unified_hidden_connections_suite.py:296` `def train(self, model, epochs, callback)`
- `Trainer.__init__` (method) `unified_hidden_connections_suite.py:304` `def __init__(self, model_config, training_config, data_generator)`
- `Trainer.train` (method) `unified_hidden_connections_suite.py:314` `def train(self, model, epochs, callback)`
- `IMetricCalculator.calculate` (method) `unified_hidden_connections_suite.py:361` `def calculate(self, model)`
- `LevelSpacingRatioCalculator.__init__` (method) `unified_hidden_connections_suite.py:368` `def __init__(self, config, tolerance)`
- `LevelSpacingRatioCalculator.calculate` (method) `unified_hidden_connections_suite.py:384` `def calculate(self, model)`
- `RicciScalarCalculator.__init__` (method) `unified_hidden_connections_suite.py:418` `def __init__(self, config, regularization)`
- `RicciScalarCalculator.calculate` (method) `unified_hidden_connections_suite.py:470` `def calculate(self, model)`
- `SyntheticPlanckCalculator.__init__` (method) `unified_hidden_connections_suite.py:495` `def __init__(self, config, noise_floor)`
- `SyntheticPlanckCalculator.calculate` (method) `unified_hidden_connections_suite.py:499` `def calculate(self, model)`
- `SuperpositionMetricCalculator.__init__` (method) `unified_hidden_connections_suite.py:555` `def __init__(self, model_config, expansion_factor, l1_coefficient, sae_lr, sae_epochs, sae_batch_size, num_samples...`
- `SuperpositionMetricCalculator.calculate` (method) `unified_hidden_connections_suite.py:623` `def calculate(self, model)`
- `IExperiment.run` (method) `unified_hidden_connections_suite.py:649` `def run(self, model)`
- `IExperiment.get_name` (method) `unified_hidden_connections_suite.py:653` `def get_name(self)`
- `Experiment1RicciMBLDuality.__init__` (method) `unified_hidden_connections_suite.py:663` `def __init__(self, config, model_config, training_config, data_generator, checkpoint_manager)`
- `Experiment1RicciMBLDuality.get_name` (method) `unified_hidden_connections_suite.py:679` `def get_name(self)`
- `Experiment1RicciMBLDuality.run` (method) `unified_hidden_connections_suite.py:682` `def run(self, model)`
- `checkpoint_callback` (function) `unified_hidden_connections_suite.py:691` `def checkpoint_callback(epoch, m, loss, acc)`
- `Experiment2AltlandZirnbauer.__init__` (method) `unified_hidden_connections_suite.py:741` `def __init__(self, config, model_config, data_generator)`
- `Experiment2AltlandZirnbauer.get_name` (method) `unified_hidden_connections_suite.py:752` `def get_name(self)`
- `Experiment2AltlandZirnbauer.run` (method) `unified_hidden_connections_suite.py:755` `def run(self, model)`
- `Experiment3ConformalIsomorphism.__init__` (method) `unified_hidden_connections_suite.py:826` `def __init__(self, config, model_config, data_generator)`
- `Experiment3ConformalIsomorphism.get_name` (method) `unified_hidden_connections_suite.py:836` `def get_name(self)`
- `Experiment3ConformalIsomorphism.run` (method) `unified_hidden_connections_suite.py:839` `def run(self, model)`
- `Experiment4CompressionFrontier.__init__` (method) `unified_hidden_connections_suite.py:893` `def __init__(self, config, model_config, data_generator)`
- `Experiment4CompressionFrontier.get_name` (method) `unified_hidden_connections_suite.py:904` `def get_name(self)`
- `Experiment4CompressionFrontier.run` (method) `unified_hidden_connections_suite.py:907` `def run(self, model)`
- `IPruningStrategy.prune` (method) `unified_hidden_connections_suite.py:969` `def prune(self, model, fraction)`
- `VolumePruningStrategy.prune` (method) `unified_hidden_connections_suite.py:976` `def prune(self, model, fraction)`
- `AreaPruningStrategy.prune` (method) `unified_hidden_connections_suite.py:991` `def prune(self, model, fraction)`
- `Experiment5HolographicPruning.__init__` (method) `unified_hidden_connections_suite.py:1010` `def __init__(self, config, model_config, data_generator)`
- `Experiment5HolographicPruning.get_name` (method) `unified_hidden_connections_suite.py:1022` `def get_name(self)`
- `Experiment5HolographicPruning.run` (method) `unified_hidden_connections_suite.py:1025` `def run(self, model)`
- `UnifiedSuite.__init__` (method) `unified_hidden_connections_suite.py:1081` `def __init__(self, config)`
- `UnifiedSuite.run_all` (method) `unified_hidden_connections_suite.py:1127` `def run_all(self)`
- `main` (function) `unified_hidden_connections_suite.py:1182` `def main()`

## xray_tensor_diffractometer.py
- `set_seed` (function) `xray_tensor_diffractometer.py:64` `def set_seed(seed)`
- `setup_logger` (function) `xray_tensor_diffractometer.py:72` `def setup_logger(name, level)`
- `run_epitaxy_from_best_crystal` (function) `xray_tensor_diffractometer.py:86` `def run_epitaxy_from_best_crystal(checkpoint_dir, target_sizes)` -- Pipeline automático: encuentra el mejor cristal y lo usa como semilla.
- `ICheckpointLoader.load_checkpoint` (method) `xray_tensor_diffractometer.py:147` `def load_checkpoint(self, path, device)`
- `IMetricsCalculator.compute` (method) `xray_tensor_diffractometer.py:150` `def compute(self, model)`
- `IDataGenerator.generate_batch` (method) `xray_tensor_diffractometer.py:153` `def generate_batch(self, batch_size)`
- `StrassenDataGenerator.generate_batch` (method) `xray_tensor_diffractometer.py:168` `def generate_batch(batch_size)`
- `StrassenDataGenerator.verify_structure` (method) `xray_tensor_diffractometer.py:179` `def verify_structure(coeffs)`
- `BilinearStrassenModel.__init__` (method) `xray_tensor_diffractometer.py:188` `def __init__(self, hidden_dim, matrix_size)`
- `BilinearStrassenModel.forward` (method) `xray_tensor_diffractometer.py:204` `def forward(self, a, b)`
- `BilinearStrassenModel.get_coefficients` (method) `xray_tensor_diffractometer.py:207` `def get_coefficients(self)`
- `EpitaxialGrowthEngine.__init__` (method) `xray_tensor_diffractometer.py:224` `def __init__(self, seed_checkpoint_path, target_matrix_size, device)`
- `EpitaxialGrowthEngine.grow_epitaxial_crystal` (method) `xray_tensor_diffractometer.py:265` `def grow_epitaxial_crystal(self)` -- Crece un cristal epitaxial desde la semilla.
- `EpitaxialGrowthEngine.anneal_crystal` (method) `xray_tensor_diffractometer.py:360` `def anneal_crystal(self, model, max_epochs, early_stop_threshold)` -- Recocido térmico del cristal epitaxial.
- `generate_batch` (function) `xray_tensor_diffractometer.py:373` `def generate_batch(batch_size)`
- `EpitaxyExperiment.__init__` (method) `xray_tensor_diffractometer.py:474` `def __init__(self, results_dir)`
- `EpitaxyExperiment.run_epitaxial_growth_experiment` (method) `xray_tensor_diffractometer.py:479` `def run_epitaxial_growth_experiment(self, seed_checkpoint, target_sizes)` -- Experimento completo: cultiva cristales de múltiples tamaños desde una semilla.
- `ThermodynamicPotential.helmholtz_free_energy` (method) `xray_tensor_diffractometer.py:680` `def helmholtz_free_energy(self)` -- F = U - T*S (a μ y N constantes)
- `ThermodynamicPotential.gibbs_free_energy` (method) `xray_tensor_diffractometer.py:684` `def gibbs_free_energy(self)` -- G = F + μ*N + P*V (presión algorítmica)
- `ThermodynamicPotential.is_stable` (method) `xray_tensor_diffractometer.py:689` `def is_stable(self)` -- Criterio de estabilidad: dG < 0
- `SpectroscopyMetrics.compute_weight_diffraction` (method) `xray_tensor_diffractometer.py:696` `def compute_weight_diffraction(coeffs)`
- `SpectroscopyMetrics.extract_lattice_parameters` (method) `xray_tensor_diffractometer.py:726` `def extract_lattice_parameters(weight_tensor, rank)` -- Extrae parámetros de red preservando la geometría física del tensor.
- `SpectroscopyMetrics.compute_gibbs_free_energy` (method) `xray_tensor_diffractometer.py:785` `def compute_gibbs_free_energy(loss, temp, entropy)`
- `SpectroscopyMetrics.extract_canonical_decomposition` (method) `xray_tensor_diffractometer.py:791` `def extract_canonical_decomposition(coeffs, rank)` -- Descomposición Canónica del tensor tripartito (U, V, W).
- `SpectroscopyMetrics.create_superlattice_seed` (method) `xray_tensor_diffractometer.py:892` `def create_superlattice_seed(base_tensor, scale_factor)`
- `ThermodynamicMetrics.compute_effective_temperature` (method) `xray_tensor_diffractometer.py:916` `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `ThermodynamicMetrics.compute_critical_exponents` (method) `xray_tensor_diffractometer.py:930` `def compute_critical_exponents(temp_history, cv_history, alpha_history)` -- Calcula exponentes críticos cerca de transiciones de fase.
- `ThermodynamicMetrics.compute_equation_of_state` (method) `xray_tensor_diffractometer.py:1009` `def compute_equation_of_state(temp_eff, alpha, kappa)` -- Ecuación de estado: T_c(α) = T_0 * exp(-c*α)
- `ThermodynamicMetrics.compute_specific_heat` (method) `xray_tensor_diffractometer.py:1048` `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `ThermodynamicMetrics.estimate_hbar_algorithmic` (method) `xray_tensor_diffractometer.py:1061` `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)`
- `ThermodynamicMetrics.compute_mutual_information` (method) `xray_tensor_diffractometer.py:1069` `def compute_mutual_information(weights, gradients)`
- `ThermodynamicMetrics.check_extensivity` (method) `xray_tensor_diffractometer.py:1083` `def check_extensivity(entropy_list, scale_factors)`
- `ThermodynamicMetrics.compute_fisher_information_matrix` (method) `xray_tensor_diffractometer.py:1107` `def compute_fisher_information_matrix(model, samples)`
- `ThermodynamicMetrics.compute_ricci_curvature` (method) `xray_tensor_diffractometer.py:1126` `def compute_ricci_curvature(fisher_matrix)`
- `ThermodynamicMetrics.calculate_carnot_efficiency` (method) `xray_tensor_diffractometer.py:1137` `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)`
- `CrystallographyMetrics.compute_kappa` (method) `xray_tensor_diffractometer.py:1161` `def compute_kappa(model, dataloader, num_batches)`
- `CrystallographyMetrics.compute_discretization_margin` (method) `xray_tensor_diffractometer.py:1187` `def compute_discretization_margin(coeffs)`
- `CrystallographyMetrics.compute_local_complexity` (method) `xray_tensor_diffractometer.py:1191` `def compute_local_complexity(model)`
- `CrystallographyMetrics.compute_alpha_purity` (method) `xray_tensor_diffractometer.py:1200` `def compute_alpha_purity(coeffs)`
- `CrystallographyMetrics.compute_kappa_quantum` (method) `xray_tensor_diffractometer.py:1207` `def compute_kappa_quantum(coeffs, hbar)`
- `CrystallographyMetrics.compute_poynting_vector` (method) `xray_tensor_diffractometer.py:1224` `def compute_poynting_vector(coeffs)`
- `CrystallographyMetrics.compute_all_metrics` (method) `xray_tensor_diffractometer.py:1246` `def compute_all_metrics(model, dataloader)`
- `GreenCowExperiment.__init__` (method) `xray_tensor_diffractometer.py:1270` `def __init__(self, model, device)`
- `GreenCowExperiment.compute_boundary_gradient` (method) `xray_tensor_diffractometer.py:1275` `def compute_boundary_gradient(self, weight)` -- Approximate surface term: gradient concentrated on tensor boundaries.
- `GreenCowExperiment.compute_bulk_gradient` (method) `xray_tensor_diffractometer.py:1290` `def compute_bulk_gradient(self, weight)` -- Interior (volume) term: everything except boundary.
- `GreenCowExperiment.run_green_backprop_step` (method) `xray_tensor_diffractometer.py:1296` `def run_green_backprop_step(self, A, B, C_true, lambda_boundary)` -- Custom backward pass using Green-inspired decomposition.
- `GreenCowExperiment.train_with_green_cow` (method) `xray_tensor_diffractometer.py:1341` `def train_with_green_cow(self, epochs, lr, lambda_boundary)`
- `CheckpointLoader.load_checkpoint` (method) `xray_tensor_diffractometer.py:1372` `def load_checkpoint(self, path, device)` -- Load checkpoint with robust deserialization handling.
- `CheckpointMigrator.migrate_checkpoint` (method) `xray_tensor_diffractometer.py:1407` `def migrate_checkpoint(raw_data)` -- Migrate checkpoint to standard format, extracting config if present.
- `get_tensor` (function) `xray_tensor_diffractometer.py:1444` `def get_tensor(key)`
- `BoltzmannAnalysisProgram.__init__` (method) `xray_tensor_diffractometer.py:1491` `def __init__(self, checkpoint_dir, results_dir)`
- `dataloader` (function) `xray_tensor_diffractometer.py:1522` `def dataloader()`
- `BoltzmannAnalysisProgram.run_full_boltzmann_program` (method) `xray_tensor_diffractometer.py:1548` `def run_full_boltzmann_program(self)`
- `BoltzmannAnalysisProgram.phase1_molecular_hypothesis` (method) `xray_tensor_diffractometer.py:1575` `def phase1_molecular_hypothesis(self)`
- `BoltzmannAnalysisProgram.phase2_entropy_production` (method) `xray_tensor_diffractometer.py:1660` `def phase2_entropy_production(self)`
- `BoltzmannAnalysisProgram.phase3_extensivity_law` (method) `xray_tensor_diffractometer.py:1742` `def phase3_extensivity_law(self)`
- `BoltzmannAnalysisProgram.phase4_quantum_basis_transform` (method) `xray_tensor_diffractometer.py:1796` `def phase4_quantum_basis_transform(self)`
- `BoltzmannAnalysisProgram.analyze_poynting_flow` (method) `xray_tensor_diffractometer.py:1849` `def analyze_poynting_flow(self)`
- `BoltzmannAnalysisProgram.phase5_thermodynamic_analysis` (method) `xray_tensor_diffractometer.py:1882` `def phase5_thermodynamic_analysis(self)` -- PHASE 5: THERMODYNAMIC ANALYSIS con exponentes críticos y ecuación de estado.
- `sample_dataloader` (function) `xray_tensor_diffractometer.py:1906` `def sample_dataloader()`
- `BoltzmannAnalysisProgram.phase6_spectroscopic_analysis` (method) `xray_tensor_diffractometer.py:2011` `def phase6_spectroscopic_analysis(self)`
- `sample_dataloader` (function) `xray_tensor_diffractometer.py:2039` `def sample_dataloader()`
- `model` (function) `xray_tensor_diffractometer.py:2415` `def model(t, A, tau, C)`
- `model` (function) `xray_tensor_diffractometer.py:2488` `def model(N, alpha, beta)`
- `convert_to_serializable` (function) `xray_tensor_diffractometer.py:2590` `def convert_to_serializable(obj)`
- `StrassenCrystallographer.__init__` (method) `xray_tensor_diffractometer.py:2611` `def __init__(self, checkpoint_path, device)`
- `StrassenCrystallographer.run_full_analysis` (method) `xray_tensor_diffractometer.py:2634` `def run_full_analysis(self)`
- `dataloader` (function) `xray_tensor_diffractometer.py:2637` `def dataloader()`
- `main` (function) `xray_tensor_diffractometer.py:2683` `def main()`

