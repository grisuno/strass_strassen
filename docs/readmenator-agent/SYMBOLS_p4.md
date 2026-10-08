# Symbols (page 4 of 5)
Previous: [SYMBOLS_p3.md](SYMBOLS_p3.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `PurityAnalyzer` | class | `purity_index.py:361` | `class PurityAnalyzer` |
| `PurityComparator` | class | `purity_index.py:274` | `class PurityComparator` |
| `PurityConfig` | class | `purity_index.py:19` | `class PurityConfig` |
| `PurityIndexCalculator` | class | `purity_index.py:101` | `class PurityIndexCalculator` |
| `PurityPipeline` | class | `purity_index.py:492` | `class PurityPipeline` |
| `__init__` | method | `purity_index.py:74` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `purity_index.py:102` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:157` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:200` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:237` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:275` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:362` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `purity_index.py:493` | `def __init__(self, config)` |
| `_assess_purity_quality` | method | `purity_index.py:145` | `def _assess_purity_quality(self, alpha, variance)` |
| `_compute_layer_purity` | method | `purity_index.py:134` | `def _compute_layer_purity(self, weights)` |
| `_delta_to_alpha` | method | `purity_index.py:140` | `def _delta_to_alpha(self, delta)` |
| `_generate_text_report` | method | `purity_index.py:574` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize` | method | `purity_index.py:85` | `def _initialize(self)` |
| `_load_checkpoint` | method | `purity_index.py:375` | `def _load_checkpoint(self)` |
| `_migrate_coefs_format` | method | `purity_index.py:350` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `purity_index.py:331` | `def _migrate_custom_format(self, state_dict, device)` |
| `_migrate_dict` | method | `purity_index.py:322` | `def _migrate_dict(self, state_dict, device)` |
| `_migrate_standard_format` | method | `purity_index.py:357` | `def _migrate_standard_format(self, state_dict)` |
| `_print_report` | method | `purity_index.py:445` | `def _print_report(self, results)` |
| `_prune_model` | method | `purity_index.py:264` | `def _prune_model(self, model, sparsity)` |
| `analyze` | method | `purity_index.py:399` | `def analyze(self)` |
| `analyze_polycrystal` | method | `purity_index.py:65` | `def analyze_polycrystal(self, model, pruning_level)` |
| `analyze_polycrystal` | method | `purity_index.py:243` | `def analyze_polycrystal(self, model, pruning_level, loss_history)` |
| `calculate` | method | `purity_index.py:50` | `def calculate(self, model)` |
| `calculate` | method | `purity_index.py:55` | `def calculate(self, loss_history)` |
| `calculate` | method | `purity_index.py:105` | `def calculate(self, model)` |
| `calculate` | method | `purity_index.py:160` | `def calculate(self, loss_history)` |
| `classify` | method | `purity_index.py:60` | `def classify(self, alpha, temperature)` |
| `classify` | method | `purity_index.py:203` | `def classify(self, alpha, temperature)` |
| `classify_polycrystal_state` | method | `purity_index.py:219` | `def classify_polycrystal_state(self, original_alpha, original_temp, poly_alpha, poly_temp)` |
| `compare` | method | `purity_index.py:70` | `def compare(self, original, polycrystal)` |
| `compare` | method | `purity_index.py:279` | `def compare(self, original, polycrystal)` |
| `forward` | method | `purity_index.py:90` | `def forward(self, a, b)` |
| `generate_summary` | method | `purity_index.py:537` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `purity_index.py:45` | `def get_coefficients(self)` |
| `get_coefficients` | method | `purity_index.py:93` | `def get_coefficients(self)` |
| `main` | method | `purity_index.py:611` | `def main()` |
| `migrate` | method | `purity_index.py:312` | `def migrate(self, raw_data, device)` |
| `process_checkpoint` | method | `purity_index.py:496` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `purity_index.py:510` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `BilinearModel` | class | `repor_experiments.py:141` | `class BilinearModel(Module)` |
| `Exp1Config` | class | `repor_experiments.py:96` | `class Exp1Config` |
| `Exp2Config` | class | `repor_experiments.py:103` | `class Exp2Config` |
| `Exp3Config` | class | `repor_experiments.py:108` | `class Exp3Config` |
| `Exp4Config` | class | `repor_experiments.py:114` | `class Exp4Config` |
| `Exp5Config` | class | `repor_experiments.py:120` | `class Exp5Config` |
| `ModelConfig` | class | `repor_experiments.py:78` | `class ModelConfig` |
| `SuiteConfig` | class | `repor_experiments.py:126` | `class SuiteConfig` |
| `TrainConfig` | class | `repor_experiments.py:86` | `class TrainConfig` |
| `__init__` | method | `repor_experiments.py:142` | `def __init__(self, cfg)` |
| `_boundary_prune` | method | `repor_experiments.py:655` | `def _boundary_prune(model, fraction)` |
| `_extract_state` | method | `repor_experiments.py:329` | `def _extract_state(d)` |
| `_random_prune` | method | `repor_experiments.py:644` | `def _random_prune(model, fraction)` |
| `_recursive_strassen` | method | `repor_experiments.py:222` | `def _recursive_strassen(U, V, W, n, trials)` |
| `_save` | method | `repor_experiments.py:783` | `def _save(data, path)` |
| `_strassen_rec` | method | `repor_experiments.py:234` | `def _strassen_rec(A, B, U, V, W, n)` |
| `_test_accuracy` | method | `repor_experiments.py:633` | `def _test_accuracy(model, n, device)` |
| `_verify_2x2` | method | `repor_experiments.py:204` | `def _verify_2x2(U, V, W, n)` |
| `analyze_checkpoint` | method | `repor_experiments.py:378` | `def analyze_checkpoint(path, device)` |
| `analyze_checkpoints` | method | `repor_experiments.py:665` | `def analyze_checkpoints(ckpt_dir, device)` |
| `cb` | method | `repor_experiments.py:434` | `def cb(ep, m, loss, acc)` |
| `classify_phase` | method | `repor_experiments.py:298` | `def classify_phase(delta)` |
| `compute_alpha` | method | `repor_experiments.py:279` | `def compute_alpha(delta)` |
| `compute_delta` | method | `repor_experiments.py:176` | `def compute_delta(U, V, W)` |
| `compute_kappa` | method | `repor_experiments.py:262` | `def compute_kappa(model, num_batches, bs)` |
| `compute_teff` | method | `repor_experiments.py:284` | `def compute_teff(model, num_batches, bs)` |
| `discretize_q` | method | `repor_experiments.py:173` | `def discretize_q(w)` |
| `experiment1` | method | `repor_experiments.py:428` | `def experiment1(cfg)` |
| `experiment2` | method | `repor_experiments.py:475` | `def experiment2(cfg)` |
| `experiment3` | method | `repor_experiments.py:507` | `def experiment3(cfg)` |
| `experiment4` | method | `repor_experiments.py:563` | `def experiment4(cfg)` |
| `experiment5` | method | `repor_experiments.py:607` | `def experiment5(cfg)` |
| `forward` | method | `repor_experiments.py:149` | `def forward(self, A, B)` |
| `get_flat` | method | `repor_experiments.py:167` | `def get_flat(self)` |
| `get_weights` | method | `repor_experiments.py:163` | `def get_weights(self)` |
| `load_checkpoint` | method | `repor_experiments.py:309` | `def load_checkpoint(path, device)` |
| `main` | method | `repor_experiments.py:705` | `def main()` |
| `phase2` | method | `repor_experiments.py:181` | `def phase2(model)` |
| `slot_importance` | method | `repor_experiments.py:156` | `def slot_importance(self)` |
| `train_model` | method | `repor_experiments.py:351` | `def train_model(cfg, model, epochs, bs, wd, lr, callback)` |
| `zero_shot_verify` | method | `repor_experiments.py:214` | `def zero_shot_verify(U, V, W, sizes)` |
| `BilinearModel` | class | `scrodingger.py:93` | `class BilinearModel(Module)` |
| `CheckpointLoader` | class | `scrodingger.py:263` | `class CheckpointLoader` |
| `CheckpointMigrator` | class | `scrodingger.py:271` | `class CheckpointMigrator` |
| `EigenvalueSolver` | class | `scrodingger.py:170` | `class EigenvalueSolver` |
| `ExpectationValueCalculator` | class | `scrodingger.py:216` | `class ExpectationValueCalculator` |
| `HamiltonianConstructor` | class | `scrodingger.py:150` | `class HamiltonianConstructor` |
| `ICheckpointLoader` | class | `scrodingger.py:84` | `class ICheckpointLoader(Protocol)` |
| `ICheckpointMigrator` | class | `scrodingger.py:89` | `class ICheckpointMigrator(Protocol)` |
| `IEigenvalueSolver` | class | `scrodingger.py:64` | `class IEigenvalueSolver(Protocol)` |
| `IExpectationValueCalculator` | class | `scrodingger.py:74` | `class IExpectationValueCalculator(Protocol)` |
| `IHamiltonianConstructor` | class | `scrodingger.py:59` | `class IHamiltonianConstructor(Protocol)` |
| `IModel` | class | `scrodingger.py:44` | `class IModel(Protocol)` |
| `IPotentialCalculator` | class | `scrodingger.py:54` | `class IPotentialCalculator(Protocol)` |
| `ITimeEvolver` | class | `scrodingger.py:69` | `class ITimeEvolver(Protocol)` |
| `IUncertaintyCalculator` | class | `scrodingger.py:79` | `class IUncertaintyCalculator(Protocol)` |
| `IWaveFunctionExtractor` | class | `scrodingger.py:49` | `class IWaveFunctionExtractor(Protocol)` |
| `PotentialCalculator` | class | `scrodingger.py:128` | `class PotentialCalculator` |
| `SchrodingerAnalyzer` | class | `scrodingger.py:320` | `class SchrodingerAnalyzer` |
| `SchrodingerConfig` | class | `scrodingger.py:19` | `class SchrodingerConfig` |
| `SchrodingerPipeline` | class | `scrodingger.py:609` | `class SchrodingerPipeline` |
| `TimeEvolver` | class | `scrodingger.py:194` | `class TimeEvolver` |
| `UncertaintyCalculator` | class | `scrodingger.py:226` | `class UncertaintyCalculator` |
| `WaveFunctionExtractor` | class | `scrodingger.py:121` | `class WaveFunctionExtractor` |
| `WaveFunctionVisualizer` | class | `scrodingger.py:542` | `class WaveFunctionVisualizer` |
| `__init__` | method | `scrodingger.py:94` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `scrodingger.py:129` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:151` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:171` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:195` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:227` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:321` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `scrodingger.py:543` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:610` | `def __init__(self, config)` |
| `_calculate_tunneling_probability` | method | `scrodingger.py:451` | `def _calculate_tunneling_probability(self, potential, wave_function)` |
| `_count_degeneracy` | method | `scrodingger.py:465` | `def _count_degeneracy(self, eigenvalues)` |
| `_generate_text_report` | method | `scrodingger.py:718` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize` | method | `scrodingger.py:105` | `def _initialize(self)` |
| `_load_checkpoint` | method | `scrodingger.py:335` | `def _load_checkpoint(self)` |
| `_migrate_coefs_format` | method | `scrodingger.py:309` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `scrodingger.py:293` | `def _migrate_custom_format(self, state_dict)` |
| `_migrate_dict` | method | `scrodingger.py:284` | `def _migrate_dict(self, state_dict)` |
| `_migrate_standard_format` | method | `scrodingger.py:316` | `def _migrate_standard_format(self, state_dict)` |
| `_print_report` | method | `scrodingger.py:478` | `def _print_report(self, results)` |
| `analyze` | method | `scrodingger.py:357` | `def analyze(self)` |
| `calculate` | method | `scrodingger.py:55` | `def calculate(self, weights)` |
| `calculate` | method | `scrodingger.py:75` | `def calculate(self, wave_function, operator)` |
| `calculate` | method | `scrodingger.py:80` | `def calculate(self, wave_function, position_grid)` |
| `calculate` | method | `scrodingger.py:132` | `def calculate(self, weights)` |
| `calculate` | method | `scrodingger.py:217` | `def calculate(self, wave_function, operator)` |
| `calculate` | method | `scrodingger.py:230` | `def calculate(self, wave_function, position_grid)` |
| `construct` | method | `scrodingger.py:60` | `def construct(self, potential, mass)` |
| `construct` | method | `scrodingger.py:154` | `def construct(self, potential, mass)` |
| `evolve` | method | `scrodingger.py:70` | `def evolve(self, initial_state, hamiltonian, time_steps, dt)` |
| `evolve` | method | `scrodingger.py:198` | `def evolve(self, initial_state, hamiltonian, time_steps, dt)` |
| `extract` | method | `scrodingger.py:50` | `def extract(self, model)` |
| `extract` | method | `scrodingger.py:122` | `def extract(self, model)` |
| `forward` | method | `scrodingger.py:110` | `def forward(self, a, b)` |
| `generate_summary` | method | `scrodingger.py:655` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `scrodingger.py:45` | `def get_coefficients(self)` |
| `get_coefficients` | method | `scrodingger.py:113` | `def get_coefficients(self)` |
| `load` | method | `scrodingger.py:85` | `def load(self, path, device)` |
| `load` | method | `scrodingger.py:264` | `def load(self, path, device)` |
| `main` | method | `scrodingger.py:779` | `def main()` |
| `migrate` | method | `scrodingger.py:90` | `def migrate(self, raw_data)` |
| `migrate` | method | `scrodingger.py:272` | `def migrate(self, raw_data)` |
| `process_checkpoint` | method | `scrodingger.py:614` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `scrodingger.py:628` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `solve` | method | `scrodingger.py:65` | `def solve(self, hamiltonian, count)` |
| `solve` | method | `scrodingger.py:174` | `def solve(self, hamiltonian, count)` |
| `visualize` | method | `scrodingger.py:546` | `def visualize(self, data, output_path)` |
| `benchmark` | function | `src/benchmarks/benchmark_final.py:64` | `def benchmark(func, A, B, warmup, runs)` |
| `main` | function | `src/benchmarks/benchmark_final.py:80` | `def main()` |
| `numpy_multiply` | function | `src/benchmarks/benchmark_final.py:60` | `def numpy_multiply(A, B)` |
| `strassen_hybrid_multiply` | function | `src/benchmarks/benchmark_final.py:45` | `def strassen_hybrid_multiply(A, B)` |
| `benchmark_function` | function | `src/benchmarks/benchmark_scientific.py:67` | `def benchmark_function(func, A, B, runs, warmup)` |
| `main` | function | `src/benchmarks/benchmark_scientific.py:91` | `def main()` |
| `numpy_multiply` | function | `src/benchmarks/benchmark_scientific.py:64` | `def numpy_multiply(A, B)` |
| `standard_avx512_multiply` | function | `src/benchmarks/benchmark_scientific.py:51` | `def standard_avx512_multiply(A, B)` |
| `strassen_multiply` | function | `src/benchmarks/benchmark_scientific.py:38` | `def strassen_multiply(A, B)` |
| `BenchmarkConfig` | class | `src/benchmarks/benchmark_strassen.py:45` | `class BenchmarkConfig` |
| `BenchmarkResult` | class | `src/benchmarks/benchmark_strassen.py:29` | `class BenchmarkResult` |
| `benchmark_resolution` | method | `src/benchmarks/benchmark_strassen.py:114` | `def benchmark_resolution(n, cfg, dtype)` |
| `estimate_memory_mb` | method | `src/benchmarks/benchmark_strassen.py:104` | `def estimate_memory_mb(n, dtype, batch_size)` |
| `get_dtype` | method | `src/benchmarks/benchmark_strassen.py:93` | `def get_dtype(dtype_str)` |
| `load_config` | method | `src/benchmarks/benchmark_strassen.py:61` | `def load_config(config_path)` |
| `main` | method | `src/benchmarks/benchmark_strassen.py:322` | `def main()` |
| `run_benchmark` | method | `src/benchmarks/benchmark_strassen.py:218` | `def run_benchmark(cfg)` |
| `save_results` | method | `src/benchmarks/benchmark_strassen.py:310` | `def save_results(results, filepath)` |
| `_load_weights` | function | `src/benchmarks/strassen_numpy.py:19` | `def _load_weights()` |
| `multiplication_count` | function | `src/benchmarks/strassen_numpy.py:112` | `def multiplication_count(n)` |
| `strassen_2x2_numpy` | function | `src/benchmarks/strassen_numpy.py:29` | `def strassen_2x2_numpy(A, B)` |
| `strassen_hybrid` | function | `src/benchmarks/strassen_numpy.py:78` | `def strassen_hybrid(A, B, threshold)` |
| `strassen_numpy` | function | `src/benchmarks/strassen_numpy.py:44` | `def strassen_numpy(A, B)` |
| `AutoTDiscovery` | class | `src/discovery/auto_T_discovery.py:32` | `class AutoTDiscovery` |
| `SymmetryStructure` | class | `src/discovery/auto_T_discovery.py:22` | `class SymmetryStructure` |
| `__init__` | method | `src/discovery/auto_T_discovery.py:42` | `def __init__(self, tolerance, verbose)` |
| `_block_repetition_score` | method | `src/discovery/auto_T_discovery.py:127` | `def _block_repetition_score(self, W, bm, bn)` |
| `_detect_block_structure` | method | `src/discovery/auto_T_discovery.py:105` | `def _detect_block_structure(self, W)` |
| `_detect_discrete_values` | method | `src/discovery/auto_T_discovery.py:89` | `def _detect_discrete_values(self, W_flat)` |
| `_detect_symmetry_type` | method | `src/discovery/auto_T_discovery.py:144` | `def _detect_symmetry_type(self, W)` |
| `_discretization_error` | method | `src/discovery/auto_T_discovery.py:198` | `def _discretization_error(self, W, values)` |
| `_invariant_subspace_dim` | method | `src/discovery/auto_T_discovery.py:184` | `def _invariant_subspace_dim(self, U, S, rank)` |
| `_is_cyclic` | method | `src/discovery/auto_T_discovery.py:171` | `def _is_cyclic(self, W)` |
| `_is_permutation_symmetric` | method | `src/discovery/auto_T_discovery.py:165` | `def _is_permutation_symmetric(self, W)` |
| `_print_analysis` | method | `src/discovery/auto_T_discovery.py:213` | `def _print_analysis(self, W, S, structure)` |
| `_validate_expansion` | method | `src/discovery/auto_T_discovery.py:290` | `def _validate_expansion(self, expanded, structure)` |
| `analyze_structure` | method | `src/discovery/auto_T_discovery.py:46` | `def analyze_structure(self, W)` |
| `construct_T` | method | `src/discovery/auto_T_discovery.py:229` | `def construct_T(self, W_dict, target_size)` |
| `recursive_strassen_multiply` | method | `src/discovery/auto_T_discovery.py:389` | `def recursive_strassen_multiply(A, B, U, V, W, base_size)` |
| `verify_expanded_correctness` | method | `src/discovery/auto_T_discovery.py:353` | `def verify_expanded_correctness(U, V, W, target_size, expanded)` |
| `verify_strassen_T` | method | `src/discovery/auto_T_discovery.py:306` | `def verify_strassen_T(model_path, target_sizes)` |
| `THRESHOLD` | macro | `src/native/strassen_c.c:12` | `#define THRESHOLD` |
| `alloc_matrix` | function | `src/native/strassen_c.c:15` | `static float* alloc_matrix(int n)` |
| `extract_quadrant` | function | `src/native/strassen_c.c:49` | `static void extract_quadrant(float* Q, float* M, int n, int row, int col)` |
| `insert_quadrant` | function | `src/native/strassen_c.c:57` | `static void insert_quadrant(float* M, float* Q, int n, int row, int col)` |
| `mat_add` | function | `src/native/strassen_c.c:33` | `static void mat_add(float* C, float* A, float* B, int n)` |
| `mat_sub` | function | `src/native/strassen_c.c:41` | `static void mat_sub(float* C, float* A, float* B, int n)` |
| `matmul_standard` | function | `src/native/strassen_c.c:20` | `static void matmul_standard(float* C, float* A, float* B, int n)` |
| `standard_multiply` | function | `src/native/strassen_c.c:169` | `void standard_multiply(float* C, float* A, float* B, int n)` |
| `strassen_multiply` | function | `src/native/strassen_c.c:164` | `void strassen_multiply(float* C, float* A, float* B, int n)` |
| `strassen_recursive` | function | `src/native/strassen_c.c:65` | `void strassen_recursive(float* C, float* A, float* B, int n)` |
| `STRASSEN_THRESHOLD` | macro | `src/native/strassen_optimal.c:15` | `#define STRASSEN_THRESHOLD` |
| `strassen_level` | function | `src/native/strassen_optimal.c:18` | `static void strassen_level(float* C, float* A, float* B, int n,                             float...` |
| `strassen_optimal` | function | `src/native/strassen_optimal.c:131` | `void strassen_optimal(float* C, float* A, float* B, int n)` |
| `ALIGN` | macro | `src/native/strassen_turbo.c:22` | `#define ALIGN` |
| `BLOCK_SIZE` | macro | `src/native/strassen_turbo.c:21` | `#define BLOCK_SIZE` |
| `THRESHOLD` | macro | `src/native/strassen_turbo.c:20` | `#define THRESHOLD` |
| `alloc_matrix` | function | `src/native/strassen_turbo.c:25` | `static inline float* alloc_matrix(int n)` |
| `extract_quadrant` | function | `src/native/strassen_turbo.c:104` | `static void extract_quadrant(float* __restrict Q, const float* __restrict M,                     ...` |
| `get_num_threads` | function | `src/native/strassen_turbo.c:267` | `int get_num_threads(void)` |
| `insert_quadrant` | function | `src/native/strassen_turbo.c:114` | `static void insert_quadrant(float* __restrict M, const float* __restrict Q,                      ...` |
| `mat_add_avx` | function | `src/native/strassen_turbo.c:30` | `static void mat_add_avx(float* __restrict C, const float* __restrict A,                          ...` |
| `mat_sub_avx` | function | `src/native/strassen_turbo.c:50` | `static void mat_sub_avx(float* __restrict C, const float* __restrict A,                          ...` |
| `matmul_blocked_avx` | function | `src/native/strassen_turbo.c:68` | `static void matmul_blocked_avx(float* __restrict C, const float* __restrict A,                   ...` |
| `strassen_turbo` | function | `src/native/strassen_turbo.c:261` | `void strassen_turbo(float* C, float* A, float* B, int n)` |
| `strassen_turbo_recursive` | function | `src/native/strassen_turbo.c:124` | `void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)` |
| `ConvergenceMetrics` | class | `src/training/convergence_theory.py:21` | `class ConvergenceMetrics` |
| `HardwareNoiseEstimator` | class | `src/training/convergence_theory.py:136` | `class HardwareNoiseEstimator` |
| `HutchinsonTraceEstimator` | class | `src/training/convergence_theory.py:31` | `class HutchinsonTraceEstimator` |
| `SimpleStrassenModel` | class | `src/training/convergence_theory.py:346` | `class SimpleStrassenModel(Module)` |
| `__init__` | method | `src/training/convergence_theory.py:40` | `def __init__(self, model, loss_fn, n_samples, device)` |
| `__init__` | method | `src/training/convergence_theory.py:144` | `def __init__(self, model, loss_fn)` |
| `__init__` | method | `src/training/convergence_theory.py:348` | `def __init__(self, rank)` |
| `_hessian_vector_product` | method | `src/training/convergence_theory.py:90` | `def _hessian_vector_product(self, x, y, v)` |
| `_rademacher_vector` | method | `src/training/convergence_theory.py:73` | `def _rademacher_vector(self)` |
| `compute_kappa_eff` | method | `src/training/convergence_theory.py:118` | `def compute_kappa_eff(self, data)` |
| `convergence_theorem` | method | `src/training/convergence_theory.py:193` | `def convergence_theorem()` |
| `estimate_noise` | method | `src/training/convergence_theory.py:148` | `def estimate_noise(self, data_loader, n_batches, n_threads)` |
| `estimate_trace` | method | `src/training/convergence_theory.py:47` | `def estimate_trace(self, data)` |
| `forward` | method | `src/training/convergence_theory.py:354` | `def forward(self, x)` |
| `verify_convergence_conditions` | method | `src/training/convergence_theory.py:265` | `def verify_convergence_conditions(model, loss_fn, train_data, noise_threshold)` |
| `detect_phase_transition` | function | `src/training/grokkit_physics.py:126` | `def detect_phase_transition(results)` |
| `main` | function | `src/training/grokkit_physics.py:148` | `def main()` |
| `measure_physics` | function | `src/training/grokkit_physics.py:64` | `def measure_physics(N, num_samples)` |
| `strassen_multiply` | function | `src/training/grokkit_physics.py:49` | `def strassen_multiply(A, B)` |
| `Config` | class | `src/training/main.py:44` | `class Config` |
| `Matrix4x4Dataset` | class | `src/training/main.py:199` | `class Matrix4x4Dataset(Dataset)` |
| `StrassenDiscovery` | class | `src/training/main.py:87` | `class StrassenDiscovery(Module)` |
| `Trainer` | class | `src/training/main.py:217` | `class Trainer` |
| `__getitem__` | method | `src/training/main.py:213` | `def __getitem__(self, idx)` |
| `__init__` | method | `src/training/main.py:96` | `def __init__(self, num_slots)` |
| `__init__` | method | `src/training/main.py:202` | `def __init__(self, num_samples, seed)` |
| `__init__` | method | `src/training/main.py:220` | `def __init__(self, config)` |
| `__len__` | method | `src/training/main.py:210` | `def __len__(self)` |
| `accuracy` | method | `src/training/main.py:242` | `def accuracy(self, pred, target)` |
| `evaluate` | method | `src/training/main.py:265` | `def evaluate(self)` |
| `forward` | method | `src/training/main.py:108` | `def forward(self, A, B)` |
| `get_active_slots` | method | `src/training/main.py:160` | `def get_active_slots(self)` |
| `get_slot_norms` | method | `src/training/main.py:155` | `def get_slot_norms(self)` |
| `get_weakest_slot` | method | `src/training/main.py:169` | `def get_weakest_slot(self)` |
| `main` | method | `src/training/main.py:345` | `def main()` |
| `mask_slot` | method | `src/training/main.py:164` | `def mask_slot(self, slot_idx)` |
| `print_coefficients` | method | `src/training/main.py:175` | `def print_coefficients(self)` |
| `set_seed` | method | `src/training/main.py:79` | `def set_seed(seed)` |
| `train` | method | `src/training/main.py:280` | `def train(self)` |
| `train_epoch` | method | `src/training/main.py:245` | `def train_epoch(self, optimizer)` |
| `StrassenModel` | class | `src/training/main_pure_math.py:22` | `class StrassenModel(Module)` |
| `__init__` | method | `src/training/main_pure_math.py:28` | `def __init__(self, rank)` |
| `active_count` | method | `src/training/main_pure_math.py:54` | `def active_count(self, thresh)` |
| `forward` | method | `src/training/main_pure_math.py:35` | `def forward(self, A, B)` |
| `gen_data` | method | `src/training/main_pure_math.py:58` | `def gen_data(n, scale)` |
| `hard_prune` | method | `src/training/main_pure_math.py:106` | `def hard_prune(model, keep)` |
| `main` | method | `src/training/main_pure_math.py:186` | `def main()` |
| `refine_pruned` | method | `src/training/main_pure_math.py:122` | `def refine_pruned(model, active, epochs, lr)` |
| `show_coeffs` | method | `src/training/main_pure_math.py:159` | `def show_coeffs(model, active)` |
| `slot_norms` | method | `src/training/main_pure_math.py:47` | `def slot_norms(self)` |
| `train` | method | `src/training/main_pure_math.py:64` | `def train(model, epochs, lr, l1, batch, verbose)` |
| `verify` | method | `src/training/main_pure_math.py:88` | `def verify(model, n)` |
| `_load_weights` | function | `src/training/strassen_core.py:14` | `def _load_weights()` |
| `get_coefficients` | function | `src/training/strassen_core.py:77` | `def get_coefficients()` |
| `multiplication_count` | function | `src/training/strassen_core.py:82` | `def multiplication_count(n)` |
| `strassen` | function | `src/training/strassen_core.py:44` | `def strassen(X, Y)` |
| `strassen_2x2` | function | `src/training/strassen_core.py:21` | `def strassen_2x2(A, B)` |
| `StrassenOperator` | class | `src/training/strassen_grokkit.py:27` | `class StrassenOperator(Module)` |
| `__init__` | method | `src/training/strassen_grokkit.py:40` | `def __init__(self, rank)` |
| `compute_LC` | method | `src/training/strassen_grokkit.py:67` | `def compute_LC(self)` |
| `compute_SP` | method | `src/training/strassen_grokkit.py:83` | `def compute_SP(self)` |
| `count_active` | method | `src/training/strassen_grokkit.py:108` | `def count_active(self, threshold)` |
| `forward` | method | `src/training/strassen_grokkit.py:49` | `def forward(self, A, B)` |
| `generate_batch` | method | `src/training/strassen_grokkit.py:113` | `def generate_batch(n, scale)` |
| `main` | method | `src/training/strassen_grokkit.py:351` | `def main()` |
| `progressive_sparsification` | method | `src/training/strassen_grokkit.py:254` | `def progressive_sparsification(model, target_slots)` |
| `slot_importance` | method | `src/training/strassen_grokkit.py:101` | `def slot_importance(self)` |
| `train_grokkit` | method | `src/training/strassen_grokkit.py:120` | `def train_grokkit(epochs, batch_size, lr, wd)` |
| `verify_grokking` | method | `src/training/strassen_grokkit.py:203` | `def verify_grokking(model, n_test)` |
| `StrassenOperator` | class | `src/training/train_strassen.py:26` | `class StrassenOperator(Module)` |
| `__init__` | method | `src/training/train_strassen.py:33` | `def __init__(self, rank)` |
| `count_active` | method | `src/training/train_strassen.py:56` | `def count_active(self, threshold)` |
| `discretize` | method | `src/training/train_strassen.py:183` | `def discretize(model, slots_to_prune)` |
| `forward` | method | `src/training/train_strassen.py:40` | `def forward(self, A, B)` |
| `generate_batch` | method | `src/training/train_strassen.py:60` | `def generate_batch(n, scale)` |
| `get_canonical_strassen` | method | `src/training/train_strassen.py:211` | `def get_canonical_strassen()` |
| `main` | method | `src/training/train_strassen.py:299` | `def main()` |
| `slot_importance` | method | `src/training/train_strassen.py:50` | `def slot_importance(self)` |
| `sparsify` | method | `src/training/train_strassen.py:104` | `def sparsify(model, target_slots)` |
| `train_phase1` | method | `src/training/train_strassen.py:66` | `def train_phase1(epochs, batch_size, lr, wd)` |
| `verify` | method | `src/training/train_strassen.py:260` | `def verify(U, V, W, n_test)` |
| `BilinearStrassenModel` | class | `superposition.py:258` | `class BilinearStrassenModel(Module)` |
| `CheckpointLoader` | class | `superposition.py:75` | `class CheckpointLoader` |
| `CheckpointLoadingError` | class | `superposition.py:71` | `class CheckpointLoadingError(Exception)` |
| `CheckpointMigrator` | class | `superposition.py:85` | `class CheckpointMigrator` |
| `Config` | class | `superposition.py:23` | `class Config` |
| `IAnalyzer` | class | `superposition.py:66` | `class IAnalyzer(ABC)` |
| `ICheckpointLoader` | class | `superposition.py:59` | `class ICheckpointLoader(Protocol)` |
| `IMetricsCalculator` | class | `superposition.py:62` | `class IMetricsCalculator(ABC)` |
| `SAETrainer` | class | `superposition.py:407` | `class SAETrainer` |
| `SparseAutoencoder` | class | `superposition.py:292` | `class SparseAutoencoder(Module)` |
| `StrassenCheckpointAnalyzer` | class | `superposition.py:475` | `class StrassenCheckpointAnalyzer(IAnalyzer)` |
| `StrassenDataGenerator` | class | `superposition.py:220` | `class StrassenDataGenerator` |
| `SuperpositionMetrics` | class | `superposition.py:334` | `class SuperpositionMetrics(IMetricsCalculator)` |
| `__init__` | method | `superposition.py:223` | `def __init__(self, config)` |
| `__init__` | method | `superposition.py:261` | `def __init__(self, config)` |
| `__init__` | method | `superposition.py:298` | `def __init__(self, config)` |
| `__init__` | method | `superposition.py:339` | `def __init__(self, config)` |
| `__init__` | method | `superposition.py:410` | `def __init__(self, sae, config)` |
| `__init__` | method | `superposition.py:480` | `def __init__(self, config)` |
| `__post_init__` | method | `superposition.py:54` | `def __post_init__(self)` |
| `_generate_comparison_plots` | method | `superposition.py:694` | `def _generate_comparison_plots(self, results)` |
| `_initialize_symmetric` | method | `superposition.py:271` | `def _initialize_symmetric(self)` |
| `_migrate_coefs_format` | method | `superposition.py:162` | `def _migrate_coefs_format(state_dict)` |
| `_migrate_custom_format` | method | `superposition.py:147` | `def _migrate_custom_format(state_dict)` |
| `_migrate_dict` | method | `superposition.py:135` | `def _migrate_dict(state_dict)` |
| `_migrate_encoder_format` | method | `superposition.py:170` | `def _migrate_encoder_format(state_dict)` |
| `_migrate_standard_format` | method | `superposition.py:215` | `def _migrate_standard_format(state_dict)` |
| `_save_final_results` | method | `superposition.py:687` | `def _save_final_results(self, results)` |
| `_save_intermediate_result` | method | `superposition.py:633` | `def _save_intermediate_result(self, result, name)` |
| `_save_progress_checkpoint` | method | `superposition.py:680` | `def _save_progress_checkpoint(self, results)` |
| `analyze_checkpoint` | method | `superposition.py:68` | `def analyze_checkpoint(self, checkpoint_path)` |
| `analyze_checkpoint` | method | `superposition.py:571` | `def analyze_checkpoint(self, checkpoint_path)` |
| `analyze_directory` | method | `superposition.py:641` | `def analyze_directory(self, checkpoint_dir)` |
| `compute` | method | `superposition.py:64` | `def compute(self)` |
| `compute` | method | `superposition.py:389` | `def compute(self, sae_activations, weight_matrix)` |
| `compute_entropy` | method | `superposition.py:352` | `def compute_entropy(self, probabilities)` |
| `compute_feature_probabilities` | method | `superposition.py:342` | `def compute_feature_probabilities(self, sae_activations)` |
| `compute_frobenius_metric` | method | `superposition.py:377` | `def compute_frobenius_metric(self, weight_matrix)` |
| `compute_interference_matrix` | method | `superposition.py:385` | `def compute_interference_matrix(self, weight_matrix)` |
| `compute_superposition` | method | `superposition.py:359` | `def compute_superposition(self, sae_activations)` |
| `decode` | method | `superposition.py:319` | `def decode(self, z)` |
| `detect_hidden_dim` | method | `superposition.py:89` | `def detect_hidden_dim(raw_data)` |
| `encode` | method | `superposition.py:310` | `def encode(self, x)` |
| `extract_bottleneck_activations` | method | `superposition.py:554` | `def extract_bottleneck_activations(self, model)` |
| `forward` | method | `superposition.py:276` | `def forward(self, a, b)` |
| `forward` | method | `superposition.py:328` | `def forward(self, x)` |
| `generate_batch` | method | `superposition.py:226` | `def generate_batch(self, batch_size)` |
| `generate_dataset` | method | `superposition.py:240` | `def generate_dataset(self, num_samples)` |
| `get_coefficients` | method | `superposition.py:284` | `def get_coefficients(self)` |
| `get_tensor` | method | `superposition.py:148` | `def get_tensor(key)` |
| `load_checkpoint` | method | `superposition.py:60` | `def load_checkpoint(self, path, device)` |
| `load_checkpoint` | method | `superposition.py:78` | `def load_checkpoint(self, path, device)` |
| `load_model` | method | `superposition.py:496` | `def load_model(self, checkpoint_path)` |
| `main` | method | `superposition.py:765` | `def main()` |
| `migrate_checkpoint` | method | `superposition.py:122` | `def migrate_checkpoint(raw_data)` |
| `train` | method | `superposition.py:421` | `def train(self, bottleneck_activations)` |
| `LocalConfig` | class | `train_batch_sweep.py:17` | `class LocalConfig` |
| `train_for_batch_size` | function | `train_batch_sweep.py:11` | `def train_for_batch_size(B, seed, output_dir)` |
| `AreaPruningStrategy` | class | `unified_hidden_connections_suite.py:988` | `class AreaPruningStrategy(IPruningStrategy)` |
| `CheckpointManager` | class | `unified_hidden_connections_suite.py:269` | `class CheckpointManager` |
| `Experiment1Config` | class | `unified_hidden_connections_suite.py:99` | `class Experiment1Config` |
| `Experiment1RicciMBLDuality` | class | `unified_hidden_connections_suite.py:657` | `class Experiment1RicciMBLDuality(IExperiment)` |
| `Experiment2AltlandZirnbauer` | class | `unified_hidden_connections_suite.py:735` | `class Experiment2AltlandZirnbauer(IExperiment)` |
| `Experiment2Config` | class | `unified_hidden_connections_suite.py:114` | `class Experiment2Config` |
| `Experiment3Config` | class | `unified_hidden_connections_suite.py:131` | `class Experiment3Config` |
| `Experiment3ConformalIsomorphism` | class | `unified_hidden_connections_suite.py:820` | `class Experiment3ConformalIsomorphism(IExperiment)` |
| `Experiment4CompressionFrontier` | class | `unified_hidden_connections_suite.py:886` | `class Experiment4CompressionFrontier(IExperiment)` |
| `Experiment4Config` | class | `unified_hidden_connections_suite.py:141` | `class Experiment4Config` |
| `Experiment5Config` | class | `unified_hidden_connections_suite.py:158` | `class Experiment5Config` |
| `Experiment5HolographicPruning` | class | `unified_hidden_connections_suite.py:1004` | `class Experiment5HolographicPruning(IExperiment)` |
| `ICheckpointManager` | class | `unified_hidden_connections_suite.py:262` | `class ICheckpointManager(Protocol)` |
| `IDataGenerator` | class | `unified_hidden_connections_suite.py:229` | `class IDataGenerator(Protocol)` |
| `IExperiment` | class | `unified_hidden_connections_suite.py:645` | `class IExperiment(ABC)` |
| `IMetricCalculator` | class | `unified_hidden_connections_suite.py:357` | `class IMetricCalculator(ABC)` |
| `IPruningStrategy` | class | `unified_hidden_connections_suite.py:965` | `class IPruningStrategy(ABC)` |
| `ITrainer` | class | `unified_hidden_connections_suite.py:293` | `class ITrainer(Protocol)` |
| `LevelSpacingRatioCalculator` | class | `unified_hidden_connections_suite.py:365` | `class LevelSpacingRatioCalculator` |
| `RicciScalarCalculator` | class | `unified_hidden_connections_suite.py:415` | `class RicciScalarCalculator` |
| `StrassStrassenConfig` | class | `unified_hidden_connections_suite.py:65` | `class StrassStrassenConfig` |
| `StrassStrassenModel` | class | `unified_hidden_connections_suite.py:183` | `class StrassStrassenModel(Module)` |
| `StrassenDataGenerator` | class | `unified_hidden_connections_suite.py:235` | `class StrassenDataGenerator` |
| `SuiteConfig` | class | `unified_hidden_connections_suite.py:169` | `class SuiteConfig` |
| `SuperpositionMetricCalculator` | class | `unified_hidden_connections_suite.py:552` | `class SuperpositionMetricCalculator` |
| `SyntheticPlanckCalculator` | class | `unified_hidden_connections_suite.py:492` | `class SyntheticPlanckCalculator` |
| `Trainer` | class | `unified_hidden_connections_suite.py:301` | `class Trainer` |
| `TrainingConfig` | class | `unified_hidden_connections_suite.py:84` | `class TrainingConfig` |
| `UnifiedSuite` | class | `unified_hidden_connections_suite.py:1078` | `class UnifiedSuite` |
| `VolumePruningStrategy` | class | `unified_hidden_connections_suite.py:973` | `class VolumePruningStrategy(IPruningStrategy)` |
| `__init__` | method | `unified_hidden_connections_suite.py:190` | `def __init__(self, config)` |
| `__init__` | method | `unified_hidden_connections_suite.py:238` | `def __init__(self, config)` |
| `__init__` | method | `unified_hidden_connections_suite.py:304` | `def __init__(self, model_config, training_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:368` | `def __init__(self, config, tolerance)` |
| `__init__` | method | `unified_hidden_connections_suite.py:418` | `def __init__(self, config, regularization)` |
| `__init__` | method | `unified_hidden_connections_suite.py:495` | `def __init__(self, config, noise_floor)` |
| `__init__` | method | `unified_hidden_connections_suite.py:555` | `def __init__(self, model_config, expansion_factor, l1_coefficient, sae_lr, sae_epochs, sae_batch_size, num_samples...` |
| `__init__` | method | `unified_hidden_connections_suite.py:663` | `def __init__(self, config, model_config, training_config, data_generator, checkpoint_manager)` |
| `__init__` | method | `unified_hidden_connections_suite.py:741` | `def __init__(self, config, model_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:826` | `def __init__(self, config, model_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:893` | `def __init__(self, config, model_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:1010` | `def __init__(self, config, model_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:1081` | `def __init__(self, config)` |
| `__post_init__` | method | `unified_hidden_connections_suite.py:76` | `def __post_init__(self)` |
| `_aggregate_verdicts` | method | `unified_hidden_connections_suite.py:1150` | `def _aggregate_verdicts(self, all_results)` |
| `_analyze_temporal_correlation` | method | `unified_hidden_connections_suite.py:717` | `def _analyze_temporal_correlation(self, results)` |
| `_apply_moebius` | method | `unified_hidden_connections_suite.py:869` | `def _apply_moebius(self, A, B)` |
| `_apply_moebius_to_output` | method | `unified_hidden_connections_suite.py:879` | `def _apply_moebius_to_output(self, C)` |
| `_build_experiments` | method | `unified_hidden_connections_suite.py:1088` | `def _build_experiments(self)` |
| `_build_hessian_approximation` | method | `unified_hidden_connections_suite.py:372` | `def _build_hessian_approximation(self, model)` |
| `_compute_hessian` | method | `unified_hidden_connections_suite.py:422` | `def _compute_hessian(self, model)` |
| `_create_gamma_model` | method | `unified_hidden_connections_suite.py:778` | `def _create_gamma_model(self, base_model, gamma)` |
| `_detect_critical_transition` | method | `unified_hidden_connections_suite.py:800` | `def _detect_critical_transition(self, results)` |
| `_diagonal_hessian_approximation` | method | `unified_hidden_connections_suite.py:456` | `def _diagonal_hessian_approximation(self, model, A, B, C_true)` |
| `_extract_activations` | method | `unified_hidden_connections_suite.py:576` | `def _extract_activations(self, model)` |
| `_generate_single_sample` | method | `unified_hidden_connections_suite.py:433` | `def _generate_single_sample(self)` |
| `_loss_from_flat` | method | `unified_hidden_connections_suite.py:437` | `def _loss_from_flat(self, flat_params, model, original_params, A, B, C_true)` |
| `_run_pruning_trials` | method | `unified_hidden_connections_suite.py:1046` | `def _run_pruning_trials(self, base_model, pruner, A, B, C_true)` |
| `_sae_forward` | method | `unified_hidden_connections_suite.py:616` | `def _sae_forward(self, x, W_enc, b_enc, b_dec)` |
| `_serialize_config` | method | `unified_hidden_connections_suite.py:1163` | `def _serialize_config(self)` |
| `_test_uncertainty_bound` | method | `unified_hidden_connections_suite.py:951` | `def _test_uncertainty_bound(self, results)` |
| `_train_gamma_model` | method | `unified_hidden_connections_suite.py:790` | `def _train_gamma_model(self, model)` |
| `_train_sae` | method | `unified_hidden_connections_suite.py:589` | `def _train_sae(self, activations)` |
| `calculate` | method | `unified_hidden_connections_suite.py:361` | `def calculate(self, model)` |
| `calculate` | method | `unified_hidden_connections_suite.py:384` | `def calculate(self, model)` |
| `calculate` | method | `unified_hidden_connections_suite.py:470` | `def calculate(self, model)` |
| `calculate` | method | `unified_hidden_connections_suite.py:499` | `def calculate(self, model)` |
| `calculate` | method | `unified_hidden_connections_suite.py:623` | `def calculate(self, model)` |
| `checkpoint_callback` | method | `unified_hidden_connections_suite.py:691` | `def checkpoint_callback(epoch, m, loss, acc)` |
| `count_active_slots` | method | `unified_hidden_connections_suite.py:225` | `def count_active_slots(self, threshold)` |
| `forward` | method | `unified_hidden_connections_suite.py:203` | `def forward(self, A, B)` |
| `generate_batch` | method | `unified_hidden_connections_suite.py:232` | `def generate_batch(self, batch_size)` |
| `generate_batch` | method | `unified_hidden_connections_suite.py:241` | `def generate_batch(self, batch_size)` |
| `get_coefficients` | method | `unified_hidden_connections_suite.py:213` | `def get_coefficients(self)` |
| `get_flat_parameters` | method | `unified_hidden_connections_suite.py:216` | `def get_flat_parameters(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:653` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:679` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:752` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:836` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:904` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:1022` | `def get_name(self)` |
| `load` | method | `unified_hidden_connections_suite.py:266` | `def load(self, path, model)` |
| `load` | method | `unified_hidden_connections_suite.py:284` | `def load(self, path, model)` |
| `main` | method | `unified_hidden_connections_suite.py:1182` | `def main()` |
| `prune` | method | `unified_hidden_connections_suite.py:969` | `def prune(self, model, fraction)` |
| `prune` | method | `unified_hidden_connections_suite.py:976` | `def prune(self, model, fraction)` |
| `prune` | method | `unified_hidden_connections_suite.py:991` | `def prune(self, model, fraction)` |
| `run` | method | `unified_hidden_connections_suite.py:649` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:682` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:755` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:839` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:907` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:1025` | `def run(self, model)` |
| `run_all` | method | `unified_hidden_connections_suite.py:1127` | `def run_all(self)` |
| `save` | method | `unified_hidden_connections_suite.py:265` | `def save(self, model, epoch, metrics, path)` |
| `save` | method | `unified_hidden_connections_suite.py:272` | `def save(self, model, epoch, metrics, path)` |
| `slot_importance` | method | `unified_hidden_connections_suite.py:219` | `def slot_importance(self)` |
| `train` | method | `unified_hidden_connections_suite.py:296` | `def train(self, model, epochs, callback)` |
| `train` | method | `unified_hidden_connections_suite.py:314` | `def train(self, model, epochs, callback)` |
| `BilinearStrassenModel` | class | `xray_tensor_diffractometer.py:187` | `class BilinearStrassenModel(Module)` |
| `BoltzmannAnalysisProgram` | class | `xray_tensor_diffractometer.py:1490` | `class BoltzmannAnalysisProgram` |
| `CheckpointLoader` | class | `xray_tensor_diffractometer.py:1371` | `class CheckpointLoader` |
| `CheckpointLoadingError` | class | `xray_tensor_diffractometer.py:155` | `class CheckpointLoadingError(Exception)` |
| `CheckpointMigrator` | class | `xray_tensor_diffractometer.py:1405` | `class CheckpointMigrator` |
| `Config` | class | `xray_tensor_diffractometer.py:27` | `class Config` |
| `CrystallographyMetrics` | class | `xray_tensor_diffractometer.py:1158` | `class CrystallographyMetrics` |
| `EpitaxialGrowthEngine` | class | `xray_tensor_diffractometer.py:216` | `class EpitaxialGrowthEngine` |
| `EpitaxyExperiment` | class | `xray_tensor_diffractometer.py:469` | `class EpitaxyExperiment` |
| `GreenCowExperiment` | class | `xray_tensor_diffractometer.py:1264` | `class GreenCowExperiment` |
| `ICheckpointLoader` | class | `xray_tensor_diffractometer.py:146` | `class ICheckpointLoader(Protocol)` |
| `IDataGenerator` | class | `xray_tensor_diffractometer.py:152` | `class IDataGenerator(Protocol)` |
| `IMetricsCalculator` | class | `xray_tensor_diffractometer.py:149` | `class IMetricsCalculator(Protocol)` |
| `MetricsComputationError` | class | `xray_tensor_diffractometer.py:158` | `class MetricsComputationError(Exception)` |
| `SpectroscopyMetrics` | class | `xray_tensor_diffractometer.py:693` | `class SpectroscopyMetrics` |
| `StrassenCrystallographer` | class | `xray_tensor_diffractometer.py:2610` | `class StrassenCrystallographer` |
| `StrassenDataGenerator` | class | `xray_tensor_diffractometer.py:166` | `class StrassenDataGenerator` |
| `ThermodynamicMetrics` | class | `xray_tensor_diffractometer.py:913` | `class ThermodynamicMetrics` |
| `ThermodynamicPotential` | class | `xray_tensor_diffractometer.py:671` | `class ThermodynamicPotential` |
| `TrainingError` | class | `xray_tensor_diffractometer.py:161` | `class TrainingError(Exception)` |
| `__init__` | method | `xray_tensor_diffractometer.py:188` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `xray_tensor_diffractometer.py:224` | `def __init__(self, seed_checkpoint_path, target_matrix_size, device)` |
| `__init__` | method | `xray_tensor_diffractometer.py:474` | `def __init__(self, results_dir)` |
| `__init__` | method | `xray_tensor_diffractometer.py:1270` | `def __init__(self, model, device)` |
| `__init__` | method | `xray_tensor_diffractometer.py:1491` | `def __init__(self, checkpoint_dir, results_dir)` |
| `__init__` | method | `xray_tensor_diffractometer.py:2611` | `def __init__(self, checkpoint_path, device)` |
| `_adjust_dimensions` | method | `xray_tensor_diffractometer.py:326` | `def _adjust_dimensions(self, tensor, target_shape)` |
| `_assign_grade` | method | `xray_tensor_diffractometer.py:2656` | `def _assign_grade(self, delta, alpha)` |
| `_check_strassen_equivalence` | method | `xray_tensor_diffractometer.py:868` | `def _check_strassen_equivalence(discretized_factors)` |
| `_classify_thermodynamic_phase` | method | `xray_tensor_diffractometer.py:2150` | `def _classify_thermodynamic_phase(self, t_eff, cv, alpha)` |
| `_compute_effective_volume` | method | `xray_tensor_diffractometer.py:2304` | `def _compute_effective_volume(self, params)` |
| `_compute_entropy` | method | `xray_tensor_diffractometer.py:2265` | `def _compute_entropy(self, params)` |
| `_compute_entropy_simple` | method | `xray_tensor_diffractometer.py:2239` | `def _compute_entropy_simple(self, params)` |

Next: [SYMBOLS_p5.md](SYMBOLS_p5.md)
