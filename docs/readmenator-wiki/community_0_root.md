# root

*Community 0 | 60 files | cohesion 1.00*

## Definition

This community groups 60 file(s) rooted at `root` with dominant language py (cohesion 1.00). Central symbols: `ALIGN`, `AccuracyCalculator`, `AdaptiveQuantizationLoss`, `AnalysisConfig`, `AnalysisPipeline`, `AreaPruningStrategy`, `ArithmeticDataset`, `AttractorLandscapeProbe`. Core file: `xray_tensor_diffractometer.py` (132 symbols). Documented purpose: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:.

## Files

### `.` (27 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `app.py` | py | utility | 3 | yes |
| `batch_size.py` | py | utility | 41 | yes |
| `boltzmann_experiments.py` | py | utility | 60 | no |

### `experiments/extended_experiments` (8 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `experiments/extended_experiments/all_test_extended.py` | py | testing | 114 | yes |
| `experiments/extended_experiments/exp1_covariance_spectrometry.py` | py | testing | 12 | yes |
| `experiments/extended_experiments/exp2_noise_ablation.py` | py | utility | 18 | yes |

### `src/training` (7 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `src/training/convergence_theory.py` | py | utility | 15 | yes |
| `src/training/grokkit_physics.py` | py | utility | 4 | yes |

### `experiments` (5 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `experiments/apendix_experiments.py` | py | utility | 17 | yes |
| `experiments/cache_analysis_v2.py` | py | infrastructure | 1 | yes |

### `src/benchmarks` (4 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `src/benchmarks/benchmark_final.py` | py | utility | 4 | yes |
| `src/benchmarks/benchmark_scientific.py` | py | utility | 5 | yes |

### `src/native` (3 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `src/native/strassen_c.c` | c | utility | 10 | no |
| `src/native/strassen_optimal.c` | c | utility | 3 | no |

### `experiments/ablation` (2 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `experiments/ablation/ablation_8192.py` | py | utility | 0 | no |
| `experiments/ablation/ablation_study.py` | py | utility | 13 | yes |

### `experiments/statistics` (2 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `experiments/statistics/coherence_analysis.py` | py | utility | 2 | yes |
| `experiments/statistics/rigorous_experiment.py` | py | utility | 19 | yes |

### `experiments/validation` (1 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `experiments/validation/benchmark.py` | py | utility | 3 | yes |

### `src/discovery` (1 files)

| File | Language | Layer | Symbols | Doc |
|------|----------|-------|---------|-----|
| `src/discovery/auto_T_discovery.py` | py | utility | 18 | yes |

*... and 40 more files in this community.*


## Key Symbols

- `StrassenNet` (class, `app.py:24`) `class StrassenNet(Module)`
- `__init__` (method, `app.py:25`) `def __init__(self, rank)`
- `forward` (method, `app.py:31`) `def forward(self, A, B)`
- `Configuration` (class, `batch_size.py:31`) `class Configuration`
- `set_random_seed` (method, `batch_size.py:71`) `def set_random_seed(seed)`
- `BilinearStrassenModel` (class, `batch_size.py:77`) `class BilinearStrassenModel(Module)`
- `__init__` (method, `batch_size.py:78`) `def __init__(self, config)`
- `forward` (method, `batch_size.py:88`) `def forward(self, a, b)`
- `get_coefficients` (method, `batch_size.py:91`) `def get_coefficients(self)`
- `compute_lambda_effective` (method, `batch_size.py:94`) `def compute_lambda_effective(self)`
- `CheckpointMigrator` (class, `batch_size.py:100`) `class CheckpointMigrator(ABC)`
- `can_migrate` (method, `batch_size.py:102`) `def can_migrate(self, state_dict)`
- `migrate` (method, `batch_size.py:106`) `def migrate(self, state_dict)`
- `CustomFormatMigrator` (class, `batch_size.py:110`) `class CustomFormatMigrator(CheckpointMigrator)`
- `can_migrate` (method, `batch_size.py:111`) `def can_migrate(self, state_dict)`
- `migrate` (method, `batch_size.py:114`) `def migrate(self, state_dict)`
- `StandardFormatMigrator` (class, `batch_size.py:122`) `class StandardFormatMigrator(CheckpointMigrator)`
- `can_migrate` (method, `batch_size.py:123`) `def can_migrate(self, state_dict)`
- `migrate` (method, `batch_size.py:126`) `def migrate(self, state_dict)`
- `CheckpointMigrationManager` (class, `batch_size.py:131`) `class CheckpointMigrationManager`
- `__init__` (method, `batch_size.py:132`) `def __init__(self)`
- `migrate_checkpoint` (method, `batch_size.py:135`) `def migrate_checkpoint(self, path, device)`
- `StrassenDataGenerator` (class, `batch_size.py:147`) `class StrassenDataGenerator`
- `generate_batch` (method, `batch_size.py:149`) `def generate_batch(batch_size, config)`
- `CrystallographyMetrics` (class, `batch_size.py:156`) `class CrystallographyMetrics`
- `compute_kappa` (method, `batch_size.py:158`) `def compute_kappa(model, num_batches, config)`
- `compute_discretization_margin` (method, `batch_size.py:174`) `def compute_discretization_margin(coeffs)`
- `compute_local_complexity` (method, `batch_size.py:178`) `def compute_local_complexity(model, config)`
- `PlanckConstantCalculator` (class, `batch_size.py:186`) `class PlanckConstantCalculator`
- `__init__` (method, `batch_size.py:187`) `def __init__(self, metrics, training_metrics, config)`

## Internal vs External Edges

- Internal resolved imports (EXTRACTED): 0
- Cross-boundary resolved imports (EXTRACTED): 0

## Connections

- No cross-community bridges recorded. This community is self-contained.

## Risks

- [taint high] `menu.py` -> `menu.py` via `subprocess` (0 hops)
- [dataflow DEAD_STORE] `src/native/strassen_c.c:72` `strassen_recursive` `hh`: `hh` assigned at line 72 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:32` `strassen_level` `M`: `M` assigned at line 32 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:36` `strassen_level` `A11`: `A11` assigned at line 36 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:37` `strassen_level` `A12`: `A12` assigned at line 37 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:38` `strassen_level` `A21`: `A21` assigned at line 38 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:39` `strassen_level` `A22`: `A22` assigned at line 39 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:41` `strassen_level` `B11`: `B11` assigned at line 41 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:42` `strassen_level` `B12`: `B12` assigned at line 42 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:43` `strassen_level` `B21`: `B21` assigned at line 43 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:44` `strassen_level` `B22`: `B22` assigned at line 44 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:46` `strassen_level` `C11`: `C11` assigned at line 46 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:47` `strassen_level` `C12`: `C12` assigned at line 47 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:48` `strassen_level` `C21`: `C21` assigned at line 48 but never read afterwards.
- [dataflow DEAD_STORE] `src/native/strassen_optimal.c:49` `strassen_level` `C22`: `C22` assigned at line 49 but never read afterwards.

## Open Questions

- Why do 16 file(s) lack file-level docs (e.g. `boltzmann_experiments.py`)? What purpose do they serve?
- Is the dangerous import `subprocess` in `menu.py` still required, or can it be isolated?
- What would break if the most connected file in root changed?
- Should root be split, given cohesion 1.00?

## Sources

- `app.py`
- `batch_size.py`
- `boltzmann_experiments.py`
- `compute_gns_checkpoints.py`
- `crystallography.py`
- `dirac_polos_zeros.py`
- `experiments/ablation/ablation_8192.py`
- `experiments/ablation/ablation_study.py`
- `experiments/apendix_experiments.py`
- `experiments/cache_analysis_v2.py`
- `experiments/extended_experiments/all_test_extended.py`
- `experiments/extended_experiments/exp1_covariance_spectrometry.py`
- `experiments/extended_experiments/exp2_noise_ablation.py`
- `experiments/extended_experiments/exp3_prospective_prediction.py`
- `experiments/extended_experiments/exp4_trajectory_perturbation.py`
- `experiments/extended_experiments/exp5_discreteness_attractors.py`
- `experiments/extended_experiments/run_all_experiments.py`
- `experiments/extended_experiments/validate2.py`
- `experiments/generate_figures.py`
- `experiments/statistics/coherence_analysis.py`
- *... and 40 more*
