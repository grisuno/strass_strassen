# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM, Ruby, Swift, Kotlin, Scala, Lua, Elixir.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 60 | **Total Symbols Extracted:** 2071 | **Total Imports:** 638

<!-- ranking_model: v1.0 | weights: {ppr:0.45,auth:0.2,test:0.15,doc:0.1,fresh:0.1} | alpha:0.85 | commit:4c8e0d2 | date:2026-07-18 -->


## Table of Contents

1. [Statistics Dashboard](#statistics-dashboard)
2. [Architectural Layers](#architectural-layers)
3. [Ranked Context](#ranked-context)
4. [God Nodes](#god-nodes)
5. [Suggested Questions](#suggested-questions)
6. [Taint Propagation Map](#taint-propagation-map)
7. [Hotspot Analysis](#hotspot-analysis)
8. [Change Impact Analysis](#change-impact-analysis)
9. [Suggested Linting Rules](#suggested-linting-rules)
10. [Orphans](#orphans)
11. [Query Recipes](#query-recipes)
12. [Structural Knowledge Map](#structural-knowledge-map)
13. [UML Class Diagram](#uml-class-diagram)
14. [Code Property Graph](#code-property-graph)
15. [Architecture Reference](#architecture-reference)
    - [C (3 files)](#c-3-files)
    - [PY (56 files)](#py-56-files)
    - [SH (1 files)](#sh-1-files)

---

## Statistics Dashboard

| Metric | Value |
|--------|-------|
| Total Files | 60 |
| Total Symbols | 2071 |
| Total Imports | 638 |
| Call Edges | 16016 |
| Inheritance Edges | 178 |
| Languages | 3 |
| Avg Symbols/File | 34.5 |
| Avg Imports/File | 10.6 |

### Top Files by Import Count (Fan-Out)

| File | Imports | Symbols | Language |
|------|---------|---------|----------|
| `xray_tensor_diffractometer.py` | 27 | 132 | py |
| `percolation_analysis.py` | 25 | 91 | py |
| `all_test_extended.py` | 22 | 114 | py |
| `validate2.py` | 22 | 71 | py |
| `repor_experiments.py` | 21 | 41 | py |
| `full_seed_prospector.py` | 20 | 116 | py |
| `grain.py` | 20 | 77 | py |
| `gravity.py` | 20 | 88 | py |
| `dirac_polos_zeros.py` | 19 | 123 | py |
| `experimetn2.py` | 19 | 58 | py |

---

## Architectural Layers

Auto-detected from path patterns, naming conventions, and imported frameworks.

| Layer | Files |
|-------|-------|
| utility | 47 |
| infrastructure | 9 |
| testing | 2 |
| data_access | 1 |
| presentation | 1 |

### utility

- `app.py` (py, 3 symbols)
- `batch_size.py` (py, 41 symbols)
- `boltzmann_experiments.py` (py, 60 symbols)
- `compute_gns_checkpoints.py` (py, 2 symbols)
- `crystallography.py` (py, 45 symbols)
- `ablation_8192.py` (py, 0 symbols)
- `ablation_study.py` (py, 13 symbols)
- `exp2_noise_ablation.py` (py, 18 symbols)
- `exp4_trajectory_perturbation.py` (py, 16 symbols)
- `run_all_experiments.py` (py, 14 symbols)
- `validate2.py` (py, 71 symbols)
- `generate_figures.py` (py, 9 symbols)
- `coherence_analysis.py` (py, 2 symbols)
- `rigorous_experiment.py` (py, 19 symbols)
- `benchmark.py` (py, 3 symbols)
- *... and 32 more*

### infrastructure

- `dirac_polos_zeros.py` (py, 123 symbols)
- `apendix_experiments.py` (py, 17 symbols)
- `cache_analysis_v2.py` (py, 1 symbols)
- `exp3_prospective_prediction.py` (py, 17 symbols)
- `exp5_discreteness_attractors.py` (py, 0 symbols)
- `hawking_radiation.py` (py, 71 symbols)
- `scrodingger.py` (py, 70 symbols)
- `auto_T_discovery.py` (py, 18 symbols)
- `xray_tensor_diffractometer.py` (py, 132 symbols)

### testing

- `all_test_extended.py` (py, 114 symbols)
- `exp1_covariance_spectrometry.py` (py, 12 symbols)

### data_access

- `full_seed_prospector.py` (py, 116 symbols)

### presentation

- `unified_hidden_connections_suite.py` (py, 99 symbols)

---

## Ranked Context

Files ranked by composite score for the current query context. The ranking combines Personalized PageRank (query relevance), global authority, test coverage, documentation coverage, and code freshness. Model: v1.0.

| Rank | File | Composite | PPR | Authority | Test | Doc |
|------|------|-----------|-----|-----------|------|-----|
| 1 | `cache_analysis_v2.py` | 0.1000 | 0.0000 | 0.0000 | 0.00 | 1.00 |
| 2 | `benchmark.py` | 0.1000 | 0.0000 | 0.0000 | 0.00 | 1.00 |
| 3 | `auto_T_discovery.py` | 0.0944 | 0.0000 | 0.0000 | 0.00 | 0.94 |
| 4 | `strassen_grokkit.py` | 0.0917 | 0.0000 | 0.0000 | 0.00 | 0.92 |
| 5 | `strassen_c.c` | 0.0900 | 0.0000 | 0.0000 | 0.00 | 0.90 |
| 6 | `generate_figures.py` | 0.0889 | 0.0000 | 0.0000 | 0.00 | 0.89 |
| 7 | `validation_experiments.py` | 0.0889 | 0.0000 | 0.0000 | 0.00 | 0.89 |
| 8 | `validate2.py` | 0.0817 | 0.0000 | 0.0000 | 0.00 | 0.82 |
| 9 | `strassen_numpy.py` | 0.0800 | 0.0000 | 0.0000 | 0.00 | 0.80 |
| 10 | `plank.py` | 0.0792 | 0.0000 | 0.0000 | 0.00 | 0.79 |

---

## God Nodes

Most architecturally central files ranked by combined import/export degree and symbol richness.

| File | Score | Connections | PageRank |
|------|-------|-------------|----------|
| `xray_tensor_diffractometer.py` | 13.2 | | 0.0000 |
| `dirac_polos_zeros.py` | 12.3 | | 0.0000 |
| `full_seed_prospector.py` | 11.6 | | 0.0000 |
| `all_test_extended.py` | 11.4 | | 0.0000 |
| `unified_hidden_connections_suite.py` | 9.9 | | 0.0000 |
| `percolation_analysis.py` | 9.1 | | 0.0000 |
| `mbl_analyzer.py` | 8.9 | | 0.0000 |
| `gravity.py` | 8.8 | | 0.0000 |
| `grain.py` | 7.7 | | 0.0000 |
| `plank.py` | 7.2 | | 0.0000 |

---

## Suggested Questions

Auto-generated exploration prompts based on graph structure:

- What does xray_tensor_diffractometer.py depend on, and what depends on it? (0 connections)
- What does dirac_polos_zeros.py depend on, and what depends on it? (0 connections)
- What does full_seed_prospector.py depend on, and what depends on it? (0 connections)
- What is StrassenNet in app.py and how is it used?
- What is Configuration in batch_size.py and how is it used?

---

## Taint Propagation Map

Taint analysis traces how dangerous imports propagate through the codebase via transitive dependencies. Source files import dangerous modules directly; sink files receive the danger indirectly.

**Taint Sources:** 1 | **Taint Sinks:** 1 | **Propagation Paths:** 1

- `menu.py` imports `subprocess` (0 hop to `menu.py`) [high]
  Path: menu.py

---

## Hotspot Analysis

Files ranked by combined complexity (symbol count) and centrality (connection count). High-scoring files are architecturally critical and may need refactoring attention.

| File | Complexity | Centrality | Combined | Symbols | Connections |
|------|-----------|------------|----------|---------|-------------|
| `cache_analysis_v2.py` | 0.008 | 0.037 | 0.025 | 1 | 1 |
| `benchmark.py` | 0.023 | 0.148 | 0.098 | 3 | 4 |
| `auto_T_discovery.py` | 0.136 | 0.222 | 0.188 | 18 | 6 |
| `strassen_grokkit.py` | 0.091 | 0.148 | 0.125 | 12 | 4 |
| `strassen_c.c` | 0.076 | 0.111 | 0.097 | 10 | 3 |
| `generate_figures.py` | 0.068 | 0.556 | 0.361 | 9 | 15 |
| `validation_experiments.py` | 0.068 | 0.222 | 0.161 | 9 | 6 |
| `validate2.py` | 0.538 | 0.815 | 0.704 | 71 | 22 |
| `strassen_numpy.py` | 0.038 | 0.148 | 0.104 | 5 | 4 |
| `plank.py` | 0.545 | 0.518 | 0.529 | 72 | 14 |
| `xray_tensor_diffractometer.py` | 1.000 | 1.000 | 1.000 | 132 | 27 |
| `all_test_extended.py` | 0.864 | 0.815 | 0.834 | 114 | 22 |
| `percolation_analysis.py` | 0.689 | 0.926 | 0.831 | 91 | 25 |
| `full_seed_prospector.py` | 0.879 | 0.741 | 0.796 | 116 | 20 |
| `dirac_polos_zeros.py` | 0.932 | 0.704 | 0.795 | 123 | 19 |

---

## Change Impact Analysis

Files sorted by how many other files would be affected if they changed. High-impact files should be changed with caution.

| File | Direct Dependents | Transitive Dependents | Total Impact |
|------|------------------|----------------------|--------------|
| `app.py` | 0 | 0 | 0 |
| `batch_size.py` | 0 | 0 | 0 |
| `boltzmann_experiments.py` | 0 | 0 | 0 |
| `compute_gns_checkpoints.py` | 0 | 0 | 0 |
| `crystallography.py` | 0 | 0 | 0 |
| `dirac_polos_zeros.py` | 0 | 0 | 0 |
| `ablation_8192.py` | 0 | 0 | 0 |
| `ablation_study.py` | 0 | 0 | 0 |
| `apendix_experiments.py` | 0 | 0 | 0 |
| `cache_analysis_v2.py` | 0 | 0 | 0 |
| `all_test_extended.py` | 0 | 0 | 0 |
| `exp1_covariance_spectrometry.py` | 0 | 0 | 0 |
| `exp2_noise_ablation.py` | 0 | 0 | 0 |
| `exp3_prospective_prediction.py` | 0 | 0 | 0 |
| `exp4_trajectory_perturbation.py` | 0 | 0 | 0 |

---

## Suggested Linting Rules

Automatically suggested linting and security rules based on patterns detected in the codebase. These can be exported as Semgrep rules using the `--export-rules` flag.

| Rule ID | Severity | Description | Language | Matches |
|---------|----------|-------------|----------|---------|
| `RM003` | warning | Bare except clause catches all exceptions including SystemExit | python | 33 |
| `RM001` | info | Large number of functions in py: 1598 total | py | 1598 |
| `RM002` | info | Large number of functions in c: 20 total | c | 20 |
| `RM004` | info | Print statement found (consider logging instead) | python | 1978 |

---

## Orphans

Files with no documentation or low connectivity. These are candidates for documentation investment or cleanup.

- `batch_size.py` (41 symbols, no doc)
- `dirac_polos_zeros.py` (123 symbols, no doc)
- `ablation_8192.py` (0 symbols, no doc)
- `apendix_experiments.py` (17 symbols, no doc)
- `exp5_discreteness_attractors.py` (0 symbols, no doc)
- `coherence_analysis.py` (2 symbols, no doc)
- `verify_checkpoints.py` (13 symbols, no doc)
- `grain.py` (77 symbols, no doc)
- `gravity.py` (88 symbols, no doc)
- `install.sh` (0 symbols, no doc)
- `measure_strassen.py` (0 symbols, no doc)
- `menu.py` (9 symbols, no doc)
- `purity_index.py` (56 symbols, no doc)
- `scrodingger.py` (70 symbols, no doc)
- `strassen_core.py` (5 symbols, no doc)

---

## Query Recipes

Example queries you can run against this knowledge base using the ranking engine:

```
# Find files most relevant to a concept
readmenator query "Where is the import resolver implemented?"

# Rank files by relevance to a topic
readmenator query "How does documentation generation work?"

# Explain why a file ranks highly
readmenator query "explain readmenator/_documentation.py"

# Trace dependency paths with ranked context
readmenator query "path from CLI to exporter"
```

The ranking model uses the following signals:

- **Personalized PageRank** (45% weight): query-specific relevance via seed propagation
- **Global Authority** (20% weight): structural importance via standard PageRank
- **Test Coverage** (15% weight): fraction of symbols referenced in test files
- **Doc Coverage** (10% weight): presence of docstrings and file-level docs
- **Freshness** (10% weight): recent modification activity

Results include score decomposition and justification paths for each ranked item.

---

## Structural Knowledge Map

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
    xray_tensor_diffractometer_py["xray_tensor_diffractometer.py (py)"]
    class xray_tensor_diffractometer_py mod;
    xray_tensor_diffractometer_py_Config["Config"]
    class xray_tensor_diffractometer_py_Config cls;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_Config
    xray_tensor_diffractometer_py_set_seed["set_seed"]
    class xray_tensor_diffractometer_py_set_seed fn;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_set_seed
    xray_tensor_diffractometer_py_setup_logger["setup_logger"]
    class xray_tensor_diffractometer_py_setup_logger fn;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_setup_logger
    xray_tensor_diffractometer_py_run_epitaxy_from_best_crystal["run_epitaxy_from_best_crystal"]
    class xray_tensor_diffractometer_py_run_epitaxy_from_best_crystal fn;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_run_epitaxy_from_best_crystal
    xray_tensor_diffractometer_py_ICheckpointLoader["ICheckpointLoader"]
    class xray_tensor_diffractometer_py_ICheckpointLoader cls;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_ICheckpointLoader
    percolation_analysis_py["percolation_analysis.py (py)"]
    class percolation_analysis_py mod;
    experiments_extended_experiments_all_test_extended_py["all_test_extended.py (py)"]
    class experiments_extended_experiments_all_test_extended_py mod;
    experiments_extended_experiments_validate2_py["validate2.py (py)"]
    class experiments_extended_experiments_validate2_py mod;
    repor_experiments_py["repor_experiments.py (py)"]
    class repor_experiments_py mod;
    full_seed_prospector_py["full_seed_prospector.py (py)"]
    class full_seed_prospector_py mod;
    gravity_py["gravity.py (py)"]
    class gravity_py mod;
    grain_py["grain.py (py)"]
    class grain_py mod;
    dirac_polos_zeros_py["dirac_polos_zeros.py (py)"]
    class dirac_polos_zeros_py mod;
    unified_hidden_connections_suite_py["unified_hidden_connections_suite.py (py)"]
    class unified_hidden_connections_suite_py mod;
    experimetn2_py["experimetn2.py (py)"]
    class experimetn2_py mod;
    superposition_py["superposition.py (py)"]
    class superposition_py mod;
    mbl_analyzer_py["mbl_analyzer.py (py)"]
    class mbl_analyzer_py mod;
    maxwell_strassen_analysis_py["maxwell_strassen_analysis.py (py)"]
    class maxwell_strassen_analysis_py mod;
    boltzmann_experiments_py["boltzmann_experiments.py (py)"]
    class boltzmann_experiments_py mod;
    grigori_perelmans_ricci_flow_py["grigori_perelmans_ricci_flow.py (py)"]
    class grigori_perelmans_ricci_flow_py mod;
    hawking_radiation_py["hawking_radiation.py (py)"]
    class hawking_radiation_py mod;
    experiments_generate_figures_py["generate_figures.py (py)"]
    class experiments_generate_figures_py mod;
    plank_py["plank.py (py)"]
    class plank_py mod;
    scrodingger_py["scrodingger.py (py)"]
    class scrodingger_py mod;
    fermi_py["fermi.py (py)"]
    class fermi_py mod;
    purity_index_py["purity_index.py (py)"]
    class purity_index_py mod;
    batch_size_py["batch_size.py (py)"]
    class batch_size_py mod;
    src_training_main_py["main.py (py)"]
    class src_training_main_py mod;
    experiments_extended_experiments_run_all_experiments_py["run_all_experiments.py (py)"]
    class experiments_extended_experiments_run_all_experiments_py mod;
    experiments_apendix_experiments_py["apendix_experiments.py (py)"]
    class experiments_apendix_experiments_py mod;
    experiments_extended_experiments_exp3_prospective_prediction_py["exp3_prospective_prediction.py (py)"]
    class experiments_extended_experiments_exp3_prospective_prediction_py mod;
    experiments_extended_experiments_exp1_covariance_spectrometry_py["exp1_covariance_spectrometry.py (py)"]
    class experiments_extended_experiments_exp1_covariance_spectrometry_py mod;
    src_benchmarks_benchmark_strassen_py["benchmark_strassen.py (py)"]
    class src_benchmarks_benchmark_strassen_py mod;
    crystallography_py["crystallography.py (py)"]
    class crystallography_py mod;
    experiments_statistics_rigorous_experiment_py["rigorous_experiment.py (py)"]
    class experiments_statistics_rigorous_experiment_py mod;
    experiments_extended_experiments_exp2_noise_ablation_py["exp2_noise_ablation.py (py)"]
    class experiments_extended_experiments_exp2_noise_ablation_py mod;
    experiments_extended_experiments_exp4_trajectory_perturbation_py["exp4_trajectory_perturbation.py (py)"]
    class experiments_extended_experiments_exp4_trajectory_perturbation_py mod;
    experiments_ablation_ablation_study_py["ablation_study.py (py)"]
    class experiments_ablation_ablation_study_py mod;
    src_benchmarks_benchmark_scientific_py["benchmark_scientific.py (py)"]
    class src_benchmarks_benchmark_scientific_py mod;
    app_py["app.py (py)"]
    class app_py mod;
    src_discovery_auto_T_discovery_py["auto_T_discovery.py (py)"]
    class src_discovery_auto_T_discovery_py mod;
    src_training_convergence_theory_py["convergence_theory.py (py)"]
    class src_training_convergence_theory_py mod;
    experiments_validation_experiments_py["validation_experiments.py (py)"]
    class experiments_validation_experiments_py mod;
    train_batch_sweep_py["train_batch_sweep.py (py)"]
    class train_batch_sweep_py mod;
    experiments_ablation_ablation_8192_py["ablation_8192.py (py)"]
    class experiments_ablation_ablation_8192_py mod;
    experiments_verify_checkpoints_py["verify_checkpoints.py (py)"]
    class experiments_verify_checkpoints_py mod;
    src_native_strassen_turbo_c["strassen_turbo.c (c)"]
    class src_native_strassen_turbo_c mod;
    src_training_main_pure_math_py["main_pure_math.py (py)"]
    class src_training_main_pure_math_py mod;
    src_training_grokkit_physics_py["grokkit_physics.py (py)"]
    class src_training_grokkit_physics_py mod;
    compute_gns_checkpoints_py["compute_gns_checkpoints.py (py)"]
    class compute_gns_checkpoints_py mod;
    src_training_strassen_grokkit_py["strassen_grokkit.py (py)"]
    class src_training_strassen_grokkit_py mod;
    src_training_train_strassen_py["train_strassen.py (py)"]
    class src_training_train_strassen_py mod;
    menu_py["menu.py (py)"]
    class menu_py mod;
    src_benchmarks_strassen_numpy_py["strassen_numpy.py (py)"]
    class src_benchmarks_strassen_numpy_py mod;
    src_benchmarks_benchmark_final_py["benchmark_final.py (py)"]
    class src_benchmarks_benchmark_final_py mod;
    experiments_validation_benchmark_py["benchmark.py (py)"]
    class experiments_validation_benchmark_py mod;
    src_native_strassen_optimal_c["strassen_optimal.c (c)"]
    class src_native_strassen_optimal_c mod;
    experiments_statistics_coherence_analysis_py["coherence_analysis.py (py)"]
    class experiments_statistics_coherence_analysis_py mod;
    src_native_strassen_c_c["strassen_c.c (c)"]
    class src_native_strassen_c_c mod;
    src_training_strassen_core_py["strassen_core.py (py)"]
    class src_training_strassen_core_py mod;
    experiments_cache_analysis_v2_py["cache_analysis_v2.py (py)"]
    class experiments_cache_analysis_v2_py mod;
    experiments_extended_experiments_exp5_discreteness_attractors_py["exp5_discreteness_attractors.py (py)"]
    class experiments_extended_experiments_exp5_discreteness_attractors_py mod;
    install_sh["install.sh (sh)"]
    class install_sh mod;
    measure_strassen_py["measure_strassen.py (py)"]
    class measure_strassen_py mod;
    ext_torch["torch"]
    class ext_torch ext;
    app_py -.->|imports| ext_torch
    ext_torch_nn["torch.nn"]
    class ext_torch_nn ext;
    app_py -.->|imports| ext_torch_nn
    ext_os["os"]
    class ext_os ext;
    app_py -.->|imports| ext_os
    ext_onnx["onnx"]
    class ext_onnx ext;
    app_py -.->|imports| ext_onnx
    ext_onnxruntime["onnxruntime"]
    class ext_onnxruntime ext;
    app_py -.->|imports| ext_onnxruntime
    app_py -.->|imports| ext_onnxruntime
    ext_numpy["numpy"]
    class ext_numpy ext;
    app_py -.->|imports| ext_numpy
    batch_size_py -.->|imports| ext_torch
    batch_size_py -.->|imports| ext_torch_nn
    batch_size_py -.->|imports| ext_numpy
    ext_random["random"]
    class ext_random ext;
    batch_size_py -.->|imports| ext_random
    ext_json["json"]
    class ext_json ext;
    batch_size_py -.->|imports| ext_json
    batch_size_py -.->|imports| ext_os
    ext_argparse["argparse"]
    class ext_argparse ext;
    batch_size_py -.->|imports| ext_argparse
    ext_typing["typing"]
    class ext_typing ext;
    batch_size_py -.->|imports| ext_typing
    ext_dataclasses["dataclasses"]
    class ext_dataclasses ext;
    batch_size_py -.->|imports| ext_dataclasses
    ext_abc["abc"]
    class ext_abc ext;
    batch_size_py -.->|imports| ext_abc
    ext_datetime["datetime"]
    class ext_datetime ext;
    batch_size_py -.->|imports| ext_datetime
    ext_matplotlib_pyplot["matplotlib.pyplot"]
    class ext_matplotlib_pyplot ext;
    batch_size_py -.->|imports| ext_matplotlib_pyplot
    ext_pathlib["pathlib"]
    class ext_pathlib ext;
    batch_size_py -.->|imports| ext_pathlib
    boltzmann_experiments_py -.->|imports| ext_argparse
    boltzmann_experiments_py -.->|imports| ext_torch
    boltzmann_experiments_py -.->|imports| ext_torch_nn
    boltzmann_experiments_py -.->|imports| ext_numpy
    boltzmann_experiments_py -.->|imports| ext_random
    boltzmann_experiments_py -.->|imports| ext_json
    boltzmann_experiments_py -.->|imports| ext_os
    boltzmann_experiments_py -.->|imports| ext_matplotlib_pyplot
    boltzmann_experiments_py -.->|imports| ext_datetime
    boltzmann_experiments_py -.->|imports| ext_typing
    boltzmann_experiments_py -.->|imports| ext_abc
    ext_seaborn["seaborn"]
    class ext_seaborn ext;
    boltzmann_experiments_py -.->|imports| ext_seaborn
    ext_scipy_stats["scipy.stats"]
    class ext_scipy_stats ext;
    boltzmann_experiments_py -.->|imports| ext_scipy_stats
    ext_scipy_linalg["scipy.linalg"]
    class ext_scipy_linalg ext;
    boltzmann_experiments_py -.->|imports| ext_scipy_linalg
    ext_scipy_optimize["scipy.optimize"]
    class ext_scipy_optimize ext;
    boltzmann_experiments_py -.->|imports| ext_scipy_optimize
    ext_sklearn_decomposition["sklearn.decomposition"]
    class ext_sklearn_decomposition ext;
    boltzmann_experiments_py -.->|imports| ext_sklearn_decomposition
    ext_warnings["warnings"]
    class ext_warnings ext;
    boltzmann_experiments_py -.->|imports| ext_warnings
    compute_gns_checkpoints_py -.->|imports| ext_torch
    compute_gns_checkpoints_py -.->|imports| ext_torch_nn
    compute_gns_checkpoints_py -.->|imports| ext_pathlib
    ext_collections["collections"]
    class ext_collections ext;
    compute_gns_checkpoints_py -.->|imports| ext_collections
    ext_vector8["vector8"]
    class ext_vector8 ext;
    compute_gns_checkpoints_py -.->|imports| ext_vector8
    crystallography_py -.->|imports| ext_torch
    crystallography_py -.->|imports| ext_torch_nn
    crystallography_py -.->|imports| ext_numpy
    crystallography_py -.->|imports| ext_random
    crystallography_py -.->|imports| ext_json
    crystallography_py -.->|imports| ext_os
    crystallography_py -.->|imports| ext_typing
    crystallography_py -.->|imports| ext_matplotlib_pyplot
    crystallography_py -.->|imports| ext_collections
    crystallography_py -.->|imports| ext_datetime
    dirac_polos_zeros_py -.->|imports| ext_torch
    dirac_polos_zeros_py -.->|imports| ext_torch_nn
    dirac_polos_zeros_py -.->|imports| ext_numpy
    dirac_polos_zeros_py -.->|imports| ext_json
    dirac_polos_zeros_py -.->|imports| ext_os
    dirac_polos_zeros_py -.->|imports| ext_argparse
    dirac_polos_zeros_py -.->|imports| ext_datetime
    dirac_polos_zeros_py -.->|imports| ext_typing
    dirac_polos_zeros_py -.->|imports| ext_pathlib
    ext_glob["glob"]
    class ext_glob ext;
    dirac_polos_zeros_py -.->|imports| ext_glob
    dirac_polos_zeros_py -.->|imports| ext_dataclasses
    dirac_polos_zeros_py -.->|imports| ext_abc
    ext_scipy["scipy"]
    class ext_scipy ext;
    dirac_polos_zeros_py -.->|imports| ext_scipy
    dirac_polos_zeros_py -.->|imports| ext_scipy_linalg
    dirac_polos_zeros_py -.->|imports| ext_matplotlib_pyplot
    ext_matplotlib_colors["matplotlib.colors"]
    class ext_matplotlib_colors ext;
    dirac_polos_zeros_py -.->|imports| ext_matplotlib_colors
    ext_matplotlib_cm["matplotlib.cm"]
    class ext_matplotlib_cm ext;
    dirac_polos_zeros_py -.->|imports| ext_matplotlib_cm
    ext_matplotlib_patches["matplotlib.patches"]
    class ext_matplotlib_patches ext;
    dirac_polos_zeros_py -.->|imports| ext_matplotlib_patches
    ext_traceback["traceback"]
    class ext_traceback ext;
    dirac_polos_zeros_py -.->|imports| ext_traceback
    ext_ctypes["ctypes"]
    class ext_ctypes ext;
    experiments_ablation_ablation_8192_py -.->|imports| ext_ctypes
    experiments_ablation_ablation_8192_py -.->|imports| ext_numpy
    ext_time["time"]
    class ext_time ext;
    experiments_ablation_ablation_8192_py -.->|imports| ext_time
    experiments_ablation_ablation_8192_py -.->|imports| ext_json
    ext_gc["gc"]
    class ext_gc ext;
    experiments_ablation_ablation_8192_py -.->|imports| ext_gc
    ext_sys["sys"]
    class ext_sys ext;
    experiments_ablation_ablation_8192_py -.->|imports| ext_sys
    experiments_ablation_ablation_study_py -.->|imports| ext_ctypes
    experiments_ablation_ablation_study_py -.->|imports| ext_numpy
    experiments_ablation_ablation_study_py -.->|imports| ext_time
    experiments_ablation_ablation_study_py -.->|imports| ext_json
    experiments_ablation_ablation_study_py -.->|imports| ext_gc
    experiments_ablation_ablation_study_py -.->|imports| ext_dataclasses
    experiments_ablation_ablation_study_py -.->|imports| ext_typing
    experiments_apendix_experiments_py -.->|imports| ext_torch
    experiments_apendix_experiments_py -.->|imports| ext_torch_nn
    ext_torch_optim["torch.optim"]
    class ext_torch_optim ext;
    experiments_apendix_experiments_py -.->|imports| ext_torch_optim
    experiments_apendix_experiments_py -.->|imports| ext_numpy
    experiments_apendix_experiments_py -.->|imports| ext_json
    experiments_apendix_experiments_py -.->|imports| ext_warnings
    experiments_apendix_experiments_py -.->|imports| ext_pathlib
    experiments_apendix_experiments_py -.->|imports| ext_datetime
    experiments_apendix_experiments_py -.->|imports| ext_time
    experiments_apendix_experiments_py -.->|imports| ext_matplotlib_pyplot
    experiments_apendix_experiments_py -.->|imports| ext_seaborn
    experiments_cache_analysis_v2_py -.->|imports| ext_json
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_torch
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_torch_nn
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_torch_optim
    ext_torch_utils_data["torch.utils.data"]
    class ext_torch_utils_data ext;
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_torch_utils_data
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_numpy
    ext_pandas["pandas"]
    class ext_pandas ext;
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_pandas
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_matplotlib_pyplot
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_scipy
    ext_scipy_spatial_distance["scipy.spatial.distance"]
    class ext_scipy_spatial_distance ext;
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_scipy_spatial_distance
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_datetime
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_pathlib
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_json
    ext_hashlib["hashlib"]
    class ext_hashlib ext;
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_hashlib
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_time
    ext_signal["signal"]
    class ext_signal ext;
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_signal
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_sys
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_typing
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_abc
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_dataclasses
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_collections
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_warnings
    experiments_extended_experiments_all_test_extended_py -.->|imports| ext_traceback
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_torch
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_torch_nn
    ext_torch_nn_functional["torch.nn.functional"]
    class ext_torch_nn_functional ext;
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_torch_nn_functional
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_numpy
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_os
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_json
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_pathlib
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_scipy
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_datetime
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_matplotlib_pyplot
    experiments_extended_experiments_exp1_covariance_spectrometry_py -.->|imports| ext_seaborn
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_torch
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_torch_nn
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_torch_nn_functional
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_numpy
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_json
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_pathlib
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_datetime
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_matplotlib_pyplot
    experiments_extended_experiments_exp2_noise_ablation_py -.->|imports| ext_seaborn
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_torch
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_torch_nn
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_torch_nn_functional
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_numpy
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_json
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_pathlib
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_scipy
    ext_sklearn_metrics["sklearn.metrics"]
    class ext_sklearn_metrics ext;
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_sklearn_metrics
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_datetime
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_matplotlib_pyplot
    experiments_extended_experiments_exp3_prospective_prediction_py -.->|imports| ext_seaborn
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_torch
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_torch_nn
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_torch_nn_functional
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_numpy
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_json
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_pathlib
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_datetime
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_matplotlib_pyplot
    experiments_extended_experiments_exp4_trajectory_perturbation_py -.->|imports| ext_seaborn
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_sys
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_os
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_torch
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_torch_nn
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_torch_nn_functional
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_numpy
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_json
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_pathlib
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_scipy
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_datetime
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_matplotlib_pyplot
    experiments_extended_experiments_run_all_experiments_py -.->|imports| ext_seaborn
    experiments_extended_experiments_validate2_py -.->|imports| ext_sys
    experiments_extended_experiments_validate2_py -.->|imports| ext_os
    experiments_extended_experiments_validate2_py -.->|imports| ext_json
    ext_copy["copy"]
    class ext_copy ext;
    experiments_extended_experiments_validate2_py -.->|imports| ext_copy
    experiments_extended_experiments_validate2_py -.->|imports| ext_argparse
    experiments_extended_experiments_validate2_py -.->|imports| ext_time
    experiments_extended_experiments_validate2_py -.->|imports| ext_pathlib
    experiments_extended_experiments_validate2_py -.->|imports| ext_datetime
    experiments_extended_experiments_validate2_py -.->|imports| ext_typing
    experiments_extended_experiments_validate2_py -.->|imports| ext_dataclasses
    experiments_extended_experiments_validate2_py -.->|imports| ext_collections
    experiments_extended_experiments_validate2_py -.->|imports| ext_warnings
    experiments_extended_experiments_validate2_py -.->|imports| ext_numpy
    experiments_extended_experiments_validate2_py -.->|imports| ext_matplotlib_pyplot
    experiments_extended_experiments_validate2_py -.->|imports| ext_sklearn_metrics
    experiments_extended_experiments_validate2_py -.->|imports| ext_scipy_stats
    experiments_extended_experiments_validate2_py -.->|imports| ext_torch
    experiments_extended_experiments_validate2_py -.->|imports| ext_torch_nn
    experiments_extended_experiments_validate2_py -.->|imports| ext_torch_optim
    experiments_extended_experiments_validate2_py -.->|imports| ext_torch_utils_data
    ext_validate_all_revisor_experiments["validate_all_revisor_experiments"]
    class ext_validate_all_revisor_experiments ext;
    experiments_extended_experiments_validate2_py -.->|imports| ext_validate_all_revisor_experiments
    experiments_extended_experiments_validate2_py -.->|imports| ext_sklearn_metrics
    experiments_generate_figures_py -.->|imports| ext_warnings
    experiments_generate_figures_py -.->|imports| ext_json
    experiments_generate_figures_py -.->|imports| ext_os
    experiments_generate_figures_py -.->|imports| ext_sys
    experiments_generate_figures_py -.->|imports| ext_numpy
    experiments_generate_figures_py -.->|imports| ext_matplotlib_pyplot
    experiments_generate_figures_py -.->|imports| ext_matplotlib_patches
    experiments_generate_figures_py -.->|imports| ext_matplotlib_patches
    ext_mpl_toolkits_mplot3d["mpl_toolkits.mplot3d"]
    class ext_mpl_toolkits_mplot3d ext;
    experiments_generate_figures_py -.->|imports| ext_mpl_toolkits_mplot3d
    experiments_generate_figures_py -.->|imports| ext_scipy
    experiments_generate_figures_py -.->|imports| ext_sklearn_decomposition
    ext_sklearn_cluster["sklearn.cluster"]
    class ext_sklearn_cluster ext;
    experiments_generate_figures_py -.->|imports| ext_sklearn_cluster
    experiments_generate_figures_py -.->|imports| ext_matplotlib_pyplot
    experiments_generate_figures_py -.->|imports| ext_seaborn
    experiments_generate_figures_py -.->|imports| ext_torch
    experiments_statistics_coherence_analysis_py -.->|imports| ext_numpy
    experiments_statistics_coherence_analysis_py -.->|imports| ext_time
    experiments_statistics_coherence_analysis_py -.->|imports| ext_json
    ext_threadpoolctl["threadpoolctl"]
    class ext_threadpoolctl ext;
    experiments_statistics_coherence_analysis_py -.->|imports| ext_threadpoolctl
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_torch
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_torch_nn
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_numpy
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_scipy
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_scipy_optimize
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_time
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_json
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_dataclasses
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_typing
    experiments_statistics_rigorous_experiment_py -.->|imports| ext_warnings
    experiments_validation_benchmark_py -.->|imports| ext_numpy
    experiments_validation_benchmark_py -.->|imports| ext_time
    experiments_validation_benchmark_py -.->|imports| ext_json
    experiments_validation_benchmark_py -.->|imports| ext_threadpoolctl
    experiments_validation_experiments_py -.->|imports| ext_torch
    experiments_validation_experiments_py -.->|imports| ext_numpy
    experiments_validation_experiments_py -.->|imports| ext_matplotlib_pyplot
    experiments_validation_experiments_py -.->|imports| ext_json
    experiments_validation_experiments_py -.->|imports| ext_pathlib
    ext_itertools["itertools"]
    class ext_itertools ext;
    experiments_validation_experiments_py -.->|imports| ext_itertools
    experiments_verify_checkpoints_py -.->|imports| ext_torch
    experiments_verify_checkpoints_py -.->|imports| ext_torch_nn
    experiments_verify_checkpoints_py -.->|imports| ext_numpy
    experiments_verify_checkpoints_py -.->|imports| ext_pathlib
    experiments_verify_checkpoints_py -.->|imports| ext_sys
    ext___future__["__future__"]
    class ext___future__ ext;
    experimetn2_py -.->|imports| ext___future__
    experimetn2_py -.->|imports| ext_argparse
    experimetn2_py -.->|imports| ext_copy
    experimetn2_py -.->|imports| ext_json
    ext_math["math"]
    class ext_math ext;
    experimetn2_py -.->|imports| ext_math
    experimetn2_py -.->|imports| ext_os
    experimetn2_py -.->|imports| ext_random
    experimetn2_py -.->|imports| ext_sys
    experimetn2_py -.->|imports| ext_warnings
    experimetn2_py -.->|imports| ext_abc
    experimetn2_py -.->|imports| ext_dataclasses
    experimetn2_py -.->|imports| ext_datetime
    experimetn2_py -.->|imports| ext_pathlib
    experimetn2_py -.->|imports| ext_typing
    experimetn2_py -.->|imports| ext_numpy
    experimetn2_py -.->|imports| ext_torch
    experimetn2_py -.->|imports| ext_torch_nn
    experimetn2_py -.->|imports| ext_torch_nn_functional
    experimetn2_py -.->|imports| ext_torch_utils_data
    fermi_py -.->|imports| ext_torch
    fermi_py -.->|imports| ext_torch_nn
    fermi_py -.->|imports| ext_numpy
    fermi_py -.->|imports| ext_json
    fermi_py -.->|imports| ext_os
    fermi_py -.->|imports| ext_argparse
    fermi_py -.->|imports| ext_datetime
    fermi_py -.->|imports| ext_typing
    fermi_py -.->|imports| ext_pathlib
    fermi_py -.->|imports| ext_glob
    fermi_py -.->|imports| ext_dataclasses
    fermi_py -.->|imports| ext_scipy_linalg
    fermi_py -.->|imports| ext_matplotlib_pyplot
    fermi_py -.->|imports| ext_traceback
    full_seed_prospector_py -.->|imports| ext_argparse
    full_seed_prospector_py -.->|imports| ext_torch
    full_seed_prospector_py -.->|imports| ext_torch_nn
    full_seed_prospector_py -.->|imports| ext_torch_optim
    full_seed_prospector_py -.->|imports| ext_numpy
    full_seed_prospector_py -.->|imports| ext_json
    full_seed_prospector_py -.->|imports| ext_os
    full_seed_prospector_py -.->|imports| ext_time
    full_seed_prospector_py -.->|imports| ext_signal
    full_seed_prospector_py -.->|imports| ext_sys
    full_seed_prospector_py -.->|imports| ext_math
    full_seed_prospector_py -.->|imports| ext_pathlib
    full_seed_prospector_py -.->|imports| ext_dataclasses
    full_seed_prospector_py -.->|imports| ext_typing
    full_seed_prospector_py -.->|imports| ext_abc
    full_seed_prospector_py -.->|imports| ext_collections
    full_seed_prospector_py -.->|imports| ext_datetime
    ext_enum["enum"]
    class ext_enum ext;
    full_seed_prospector_py -.->|imports| ext_enum
    full_seed_prospector_py -.->|imports| ext_warnings
    full_seed_prospector_py -.->|imports| ext_numpy
    grain_py -.->|imports| ext_torch
    grain_py -.->|imports| ext_torch_nn
    grain_py -.->|imports| ext_torch_optim
    grain_py -.->|imports| ext_numpy
    grain_py -.->|imports| ext_json
    grain_py -.->|imports| ext_os
    grain_py -.->|imports| ext_argparse
    grain_py -.->|imports| ext_time
    grain_py -.->|imports| ext_signal
    grain_py -.->|imports| ext_sys
    grain_py -.->|imports| ext_datetime
    grain_py -.->|imports| ext_typing
    grain_py -.->|imports| ext_pathlib
    grain_py -.->|imports| ext_glob
    grain_py -.->|imports| ext_dataclasses
    grain_py -.->|imports| ext_collections
    grain_py -.->|imports| ext_matplotlib_pyplot
    grain_py -.->|imports| ext_matplotlib_patches
    grain_py -.->|imports| ext_matplotlib_patches
    grain_py -.->|imports| ext_traceback
    gravity_py -.->|imports| ext_torch
    gravity_py -.->|imports| ext_torch_nn
    gravity_py -.->|imports| ext_numpy
    gravity_py -.->|imports| ext_json
    gravity_py -.->|imports| ext_os
    gravity_py -.->|imports| ext_argparse
    gravity_py -.->|imports| ext_datetime
    gravity_py -.->|imports| ext_typing
    gravity_py -.->|imports| ext_pathlib
    gravity_py -.->|imports| ext_glob
    gravity_py -.->|imports| ext_dataclasses
    gravity_py -.->|imports| ext_scipy
    gravity_py -.->|imports| ext_scipy_linalg
    gravity_py -.->|imports| ext_scipy_stats
    gravity_py -.->|imports| ext_matplotlib_pyplot
    gravity_py -.->|imports| ext_matplotlib_colors
    gravity_py -.->|imports| ext_matplotlib_cm
    gravity_py -.->|imports| ext_matplotlib_patches
    gravity_py -.->|imports| ext_sklearn_metrics
    gravity_py -.->|imports| ext_traceback
    grigori_perelmans_ricci_flow_py -.->|imports| ext_torch
    grigori_perelmans_ricci_flow_py -.->|imports| ext_torch_nn
    grigori_perelmans_ricci_flow_py -.->|imports| ext_torch_nn_functional
    grigori_perelmans_ricci_flow_py -.->|imports| ext_numpy
    grigori_perelmans_ricci_flow_py -.->|imports| ext_random
    grigori_perelmans_ricci_flow_py -.->|imports| ext_json
    grigori_perelmans_ricci_flow_py -.->|imports| ext_os
    grigori_perelmans_ricci_flow_py -.->|imports| ext_argparse
    grigori_perelmans_ricci_flow_py -.->|imports| ext_typing
    grigori_perelmans_ricci_flow_py -.->|imports| ext_dataclasses
    grigori_perelmans_ricci_flow_py -.->|imports| ext_abc
    grigori_perelmans_ricci_flow_py -.->|imports| ext_collections
    grigori_perelmans_ricci_flow_py -.->|imports| ext_datetime
    grigori_perelmans_ricci_flow_py -.->|imports| ext_matplotlib_pyplot
    grigori_perelmans_ricci_flow_py -.->|imports| ext_pathlib
    grigori_perelmans_ricci_flow_py -.->|imports| ext_traceback
    hawking_radiation_py -.->|imports| ext_torch
    hawking_radiation_py -.->|imports| ext_torch_nn
    hawking_radiation_py -.->|imports| ext_numpy
    hawking_radiation_py -.->|imports| ext_json
    hawking_radiation_py -.->|imports| ext_os
    hawking_radiation_py -.->|imports| ext_argparse
    ext_pickle["pickle"]
    class ext_pickle ext;
    hawking_radiation_py -.->|imports| ext_pickle
    ext_io["io"]
    class ext_io ext;
    hawking_radiation_py -.->|imports| ext_io
    hawking_radiation_py -.->|imports| ext_datetime
    hawking_radiation_py -.->|imports| ext_typing
    hawking_radiation_py -.->|imports| ext_pathlib
    hawking_radiation_py -.->|imports| ext_dataclasses
    hawking_radiation_py -.->|imports| ext_scipy_stats
    hawking_radiation_py -.->|imports| ext_warnings
    hawking_radiation_py -.->|imports| ext_traceback
    maxwell_strassen_analysis_py -.->|imports| ext_torch
    maxwell_strassen_analysis_py -.->|imports| ext_torch_nn
    maxwell_strassen_analysis_py -.->|imports| ext_numpy
    maxwell_strassen_analysis_py -.->|imports| ext_json
    maxwell_strassen_analysis_py -.->|imports| ext_os
    maxwell_strassen_analysis_py -.->|imports| ext_argparse
    maxwell_strassen_analysis_py -.->|imports| ext_time
    maxwell_strassen_analysis_py -.->|imports| ext_datetime
    maxwell_strassen_analysis_py -.->|imports| ext_typing
    maxwell_strassen_analysis_py -.->|imports| ext_pathlib
    maxwell_strassen_analysis_py -.->|imports| ext_dataclasses
    ext_scipy_fft["scipy.fft"]
    class ext_scipy_fft ext;
    maxwell_strassen_analysis_py -.->|imports| ext_scipy_fft
    maxwell_strassen_analysis_py -.->|imports| ext_scipy_stats
    ext_matplotlib["matplotlib"]
    class ext_matplotlib ext;
    maxwell_strassen_analysis_py -.->|imports| ext_matplotlib
    maxwell_strassen_analysis_py -.->|imports| ext_matplotlib_pyplot
    maxwell_strassen_analysis_py -.->|imports| ext_mpl_toolkits_mplot3d
    maxwell_strassen_analysis_py -.->|imports| ext_warnings
    mbl_analyzer_py -.->|imports| ext_torch
    mbl_analyzer_py -.->|imports| ext_torch_nn
    mbl_analyzer_py -.->|imports| ext_numpy
    mbl_analyzer_py -.->|imports| ext_json
    mbl_analyzer_py -.->|imports| ext_os
    mbl_analyzer_py -.->|imports| ext_argparse
    mbl_analyzer_py -.->|imports| ext_time
    mbl_analyzer_py -.->|imports| ext_datetime
    mbl_analyzer_py -.->|imports| ext_typing
    mbl_analyzer_py -.->|imports| ext_pathlib
    mbl_analyzer_py -.->|imports| ext_glob
    mbl_analyzer_py -.->|imports| ext_dataclasses
    mbl_analyzer_py -.->|imports| ext_scipy_stats
    mbl_analyzer_py -.->|imports| ext_scipy_linalg
    mbl_analyzer_py -.->|imports| ext_matplotlib_pyplot
    mbl_analyzer_py -.->|imports| ext_warnings
    mbl_analyzer_py -.->|imports| ext_traceback
    menu_py -.->|imports| ext_os
    menu_py -.->|imports| ext_sys
    ext_subprocess["subprocess"]
    class ext_subprocess ext;
    menu_py -.->|imports| ext_subprocess
    ext_textwrap["textwrap"]
    class ext_textwrap ext;
    menu_py -.->|imports| ext_textwrap
    percolation_analysis_py -.->|imports| ext_numpy
    percolation_analysis_py -.->|imports| ext_json
    percolation_analysis_py -.->|imports| ext_os
    percolation_analysis_py -.->|imports| ext_argparse
    percolation_analysis_py -.->|imports| ext_time
    percolation_analysis_py -.->|imports| ext_glob
    percolation_analysis_py -.->|imports| ext_warnings
    percolation_analysis_py -.->|imports| ext_datetime
    percolation_analysis_py -.->|imports| ext_typing
    percolation_analysis_py -.->|imports| ext_pathlib
    percolation_analysis_py -.->|imports| ext_dataclasses
    percolation_analysis_py -.->|imports| ext_scipy_stats
    percolation_analysis_py -.->|imports| ext_scipy_linalg
    ext_scipy_sparse["scipy.sparse"]
    class ext_scipy_sparse ext;
    percolation_analysis_py -.->|imports| ext_scipy_sparse
    ext_scipy_sparse_csgraph["scipy.sparse.csgraph"]
    class ext_scipy_sparse_csgraph ext;
    percolation_analysis_py -.->|imports| ext_scipy_sparse_csgraph
    percolation_analysis_py -.->|imports| ext_matplotlib
    percolation_analysis_py -.->|imports| ext_matplotlib_pyplot
    percolation_analysis_py -.->|imports| ext_torch
    percolation_analysis_py -.->|imports| ext_torch_nn
    percolation_analysis_py -.->|imports| ext_torch
    percolation_analysis_py -.->|imports| ext_sys
    ext_re["re"]
    class ext_re ext;
    percolation_analysis_py -.->|imports| ext_re
    ext_types["types"]
    class ext_types ext;
    percolation_analysis_py -.->|imports| ext_types
    percolation_analysis_py -.->|imports| ext_types
    percolation_analysis_py -.->|imports| ext_traceback
    plank_py -.->|imports| ext_torch
    plank_py -.->|imports| ext_torch_nn
    plank_py -.->|imports| ext_numpy
    plank_py -.->|imports| ext_random
    plank_py -.->|imports| ext_json
    plank_py -.->|imports| ext_os
    plank_py -.->|imports| ext_argparse
    plank_py -.->|imports| ext_typing
    plank_py -.->|imports| ext_dataclasses
    plank_py -.->|imports| ext_abc
    plank_py -.->|imports| ext_collections
    plank_py -.->|imports| ext_datetime
    plank_py -.->|imports| ext_matplotlib_pyplot
    plank_py -.->|imports| ext_pathlib
    purity_index_py -.->|imports| ext_torch
    purity_index_py -.->|imports| ext_torch_nn
    purity_index_py -.->|imports| ext_numpy
    purity_index_py -.->|imports| ext_json
    purity_index_py -.->|imports| ext_os
    purity_index_py -.->|imports| ext_argparse
    purity_index_py -.->|imports| ext_datetime
    purity_index_py -.->|imports| ext_typing
    purity_index_py -.->|imports| ext_pathlib
    purity_index_py -.->|imports| ext_glob
    purity_index_py -.->|imports| ext_dataclasses
    purity_index_py -.->|imports| ext_scipy_stats
    purity_index_py -.->|imports| ext_matplotlib_pyplot
    purity_index_py -.->|imports| ext_traceback
    repor_experiments_py -.->|imports| ext___future__
    repor_experiments_py -.->|imports| ext_argparse
    repor_experiments_py -.->|imports| ext_copy
    ext_csv["csv"]
    class ext_csv ext;
    repor_experiments_py -.->|imports| ext_csv
    repor_experiments_py -.->|imports| ext_json
    repor_experiments_py -.->|imports| ext_math
    repor_experiments_py -.->|imports| ext_os
    repor_experiments_py -.->|imports| ext_random
    repor_experiments_py -.->|imports| ext_sys
    repor_experiments_py -.->|imports| ext_warnings
    repor_experiments_py -.->|imports| ext_abc
    repor_experiments_py -.->|imports| ext_dataclasses
    repor_experiments_py -.->|imports| ext_datetime
    repor_experiments_py -.->|imports| ext_pathlib
    repor_experiments_py -.->|imports| ext_typing
    repor_experiments_py -.->|imports| ext_numpy
    repor_experiments_py -.->|imports| ext_torch
    repor_experiments_py -.->|imports| ext_torch_nn
    repor_experiments_py -.->|imports| ext_torch_nn_functional
    repor_experiments_py -.->|imports| ext_scipy_stats
    repor_experiments_py -.->|imports| ext_traceback
    scrodingger_py -.->|imports| ext_torch
    scrodingger_py -.->|imports| ext_torch_nn
    scrodingger_py -.->|imports| ext_numpy
    scrodingger_py -.->|imports| ext_json
    scrodingger_py -.->|imports| ext_os
    scrodingger_py -.->|imports| ext_argparse
    scrodingger_py -.->|imports| ext_datetime
    scrodingger_py -.->|imports| ext_typing
    scrodingger_py -.->|imports| ext_pathlib
    scrodingger_py -.->|imports| ext_glob
    scrodingger_py -.->|imports| ext_dataclasses
    scrodingger_py -.->|imports| ext_scipy_linalg
    scrodingger_py -.->|imports| ext_matplotlib_pyplot
    scrodingger_py -.->|imports| ext_traceback
    src_benchmarks_benchmark_final_py -.->|imports| ext_numpy
    src_benchmarks_benchmark_final_py -.->|imports| ext_ctypes
    src_benchmarks_benchmark_final_py -.->|imports| ext_time
    src_benchmarks_benchmark_final_py -.->|imports| ext_os
    src_benchmarks_benchmark_scientific_py -.->|imports| ext_numpy
    src_benchmarks_benchmark_scientific_py -.->|imports| ext_ctypes
    src_benchmarks_benchmark_scientific_py -.->|imports| ext_time
    src_benchmarks_benchmark_scientific_py -.->|imports| ext_json
    src_benchmarks_benchmark_scientific_py -.->|imports| ext_os
    src_benchmarks_benchmark_scientific_py -.->|imports| ext_datetime
    src_benchmarks_benchmark_scientific_py -.->|imports| ext_traceback
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_torch
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_time
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_json
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_gc
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_sys
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_pathlib
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_dataclasses
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_typing
    ext_strassen["strassen"]
    class ext_strassen ext;
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_strassen
    ext_tomllib["tomllib"]
    class ext_tomllib ext;
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_tomllib
    ext_tomli["tomli"]
    class ext_tomli ext;
    src_benchmarks_benchmark_strassen_py -.->|imports| ext_tomli
    src_benchmarks_strassen_numpy_py -.->|imports| ext_numpy
    src_benchmarks_strassen_numpy_py -.->|imports| ext_torch
    src_benchmarks_strassen_numpy_py -.->|imports| ext_pathlib
    ext_functools["functools"]
    class ext_functools ext;
    src_benchmarks_strassen_numpy_py -.->|imports| ext_functools
    src_discovery_auto_T_discovery_py -.->|imports| ext_torch
    src_discovery_auto_T_discovery_py -.->|imports| ext_numpy
    src_discovery_auto_T_discovery_py -.->|imports| ext_typing
    src_discovery_auto_T_discovery_py -.->|imports| ext_dataclasses
    src_discovery_auto_T_discovery_py -.->|imports| ext_torch_nn_functional
    src_discovery_auto_T_discovery_py -.->|imports| ext_sys
    ext_stdlib_h["stdlib.h"]
    class ext_stdlib_h ext;
    src_native_strassen_c_c -.->|imports| ext_stdlib_h
    ext_string_h["string.h"]
    class ext_string_h ext;
    src_native_strassen_c_c -.->|imports| ext_string_h
    ext_stdio_h["stdio.h"]
    class ext_stdio_h ext;
    src_native_strassen_c_c -.->|imports| ext_stdio_h
    src_native_strassen_optimal_c -.->|imports| ext_stdlib_h
    src_native_strassen_optimal_c -.->|imports| ext_string_h
    src_native_strassen_optimal_c -.->|imports| ext_stdio_h
    ext_cblas_h["cblas.h"]
    class ext_cblas_h ext;
    src_native_strassen_optimal_c -.->|imports| ext_cblas_h
    src_native_strassen_turbo_c -.->|imports| ext_stdlib_h
    src_native_strassen_turbo_c -.->|imports| ext_string_h
    src_native_strassen_turbo_c -.->|imports| ext_stdio_h
    ext_omp_h["omp.h"]
    class ext_omp_h ext;
    src_native_strassen_turbo_c -.->|imports| ext_omp_h
    ext_immintrin_h["immintrin.h"]
    class ext_immintrin_h ext;
    src_native_strassen_turbo_c -.->|imports| ext_immintrin_h
    src_training_convergence_theory_py -.->|imports| ext_torch
    src_training_convergence_theory_py -.->|imports| ext_torch_nn
    src_training_convergence_theory_py -.->|imports| ext_numpy
    src_training_convergence_theory_py -.->|imports| ext_typing
    src_training_convergence_theory_py -.->|imports| ext_dataclasses
    src_training_convergence_theory_py -.->|imports| ext_time
    src_training_grokkit_physics_py -.->|imports| ext_numpy
    src_training_grokkit_physics_py -.->|imports| ext_ctypes
    src_training_grokkit_physics_py -.->|imports| ext_time
    src_training_grokkit_physics_py -.->|imports| ext_sys
    src_training_grokkit_physics_py -.->|imports| ext_pathlib
    src_training_main_py -.->|imports| ext_os
    src_training_main_py -.->|imports| ext_sys
    src_training_main_py -.->|imports| ext_time
    ext_logging["logging"]
    class ext_logging ext;
    src_training_main_py -.->|imports| ext_logging
    src_training_main_py -.->|imports| ext_pathlib
    src_training_main_py -.->|imports| ext_typing
    src_training_main_py -.->|imports| ext_dataclasses
    src_training_main_py -.->|imports| ext_torch
    src_training_main_py -.->|imports| ext_torch_nn
    src_training_main_py -.->|imports| ext_torch_optim
    src_training_main_py -.->|imports| ext_torch_utils_data
    src_training_main_py -.->|imports| ext_numpy
    src_training_main_pure_math_py -.->|imports| ext_torch
    src_training_main_pure_math_py -.->|imports| ext_torch_nn
    src_training_main_pure_math_py -.->|imports| ext_torch_optim
    src_training_main_pure_math_py -.->|imports| ext_numpy
    src_training_main_pure_math_py -.->|imports| ext_pathlib
    src_training_strassen_core_py -.->|imports| ext_torch
    src_training_strassen_core_py -.->|imports| ext_pathlib
    src_training_strassen_grokkit_py -.->|imports| ext_torch
    src_training_strassen_grokkit_py -.->|imports| ext_torch_nn
    src_training_strassen_grokkit_py -.->|imports| ext_torch_optim
    src_training_strassen_grokkit_py -.->|imports| ext_math
    src_training_train_strassen_py -.->|imports| ext_torch
    src_training_train_strassen_py -.->|imports| ext_torch_nn
    src_training_train_strassen_py -.->|imports| ext_torch_optim
    src_training_train_strassen_py -.->|imports| ext_pathlib
    superposition_py -.->|imports| ext_argparse
    superposition_py -.->|imports| ext_json
    superposition_py -.->|imports| ext_logging
    superposition_py -.->|imports| ext_time
    superposition_py -.->|imports| ext_warnings
    superposition_py -.->|imports| ext_abc
    superposition_py -.->|imports| ext_dataclasses
    superposition_py -.->|imports| ext_datetime
    superposition_py -.->|imports| ext_pathlib
    superposition_py -.->|imports| ext_typing
    superposition_py -.->|imports| ext_numpy
    superposition_py -.->|imports| ext_torch
    superposition_py -.->|imports| ext_torch_nn
    superposition_py -.->|imports| ext_torch_nn_functional
    superposition_py -.->|imports| ext_torch_utils_data
    ext_tqdm["tqdm"]
    class ext_tqdm ext;
    superposition_py -.->|imports| ext_tqdm
    superposition_py -.->|imports| ext_matplotlib_pyplot
    superposition_py -.->|imports| ext_scipy_stats
    train_batch_sweep_py -.->|imports| ext_torch
    train_batch_sweep_py -.->|imports| ext_torch_nn
    train_batch_sweep_py -.->|imports| ext_os
    train_batch_sweep_py -.->|imports| ext_pathlib
    train_batch_sweep_py -.->|imports| ext_vector8
    train_batch_sweep_py -.->|imports| ext_argparse
    unified_hidden_connections_suite_py -.->|imports| ext___future__
    unified_hidden_connections_suite_py -.->|imports| ext_argparse
    unified_hidden_connections_suite_py -.->|imports| ext_copy
    unified_hidden_connections_suite_py -.->|imports| ext_json
    unified_hidden_connections_suite_py -.->|imports| ext_math
    unified_hidden_connections_suite_py -.->|imports| ext_os
    unified_hidden_connections_suite_py -.->|imports| ext_random
    unified_hidden_connections_suite_py -.->|imports| ext_sys
    unified_hidden_connections_suite_py -.->|imports| ext_warnings
    unified_hidden_connections_suite_py -.->|imports| ext_abc
    unified_hidden_connections_suite_py -.->|imports| ext_dataclasses
    unified_hidden_connections_suite_py -.->|imports| ext_datetime
    unified_hidden_connections_suite_py -.->|imports| ext_pathlib
    unified_hidden_connections_suite_py -.->|imports| ext_typing
    unified_hidden_connections_suite_py -.->|imports| ext_numpy
    unified_hidden_connections_suite_py -.->|imports| ext_torch
    unified_hidden_connections_suite_py -.->|imports| ext_torch_nn
    unified_hidden_connections_suite_py -.->|imports| ext_torch_nn_functional
    unified_hidden_connections_suite_py -.->|imports| ext_torch_utils_data
    xray_tensor_diffractometer_py -.->|imports| ext_argparse
    xray_tensor_diffractometer_py -.->|imports| ext_torch
    xray_tensor_diffractometer_py -.->|imports| ext_torch_nn
    xray_tensor_diffractometer_py -.->|imports| ext_numpy
    xray_tensor_diffractometer_py -.->|imports| ext_random
    xray_tensor_diffractometer_py -.->|imports| ext_json
    xray_tensor_diffractometer_py -.->|imports| ext_os
    xray_tensor_diffractometer_py -.->|imports| ext_matplotlib_pyplot
    xray_tensor_diffractometer_py -.->|imports| ext_datetime
    xray_tensor_diffractometer_py -.->|imports| ext_typing
    xray_tensor_diffractometer_py -.->|imports| ext_abc
    xray_tensor_diffractometer_py -.->|imports| ext_seaborn
    xray_tensor_diffractometer_py -.->|imports| ext_scipy_stats
    xray_tensor_diffractometer_py -.->|imports| ext_scipy_linalg
    xray_tensor_diffractometer_py -.->|imports| ext_scipy_optimize
    xray_tensor_diffractometer_py -.->|imports| ext_sklearn_decomposition
    xray_tensor_diffractometer_py -.->|imports| ext_warnings
    xray_tensor_diffractometer_py -.->|imports| ext_logging
    xray_tensor_diffractometer_py -.->|imports| ext_dataclasses
    xray_tensor_diffractometer_py -.->|imports| ext_pathlib
    ext_threading["threading"]
    class ext_threading ext;
    xray_tensor_diffractometer_py -.->|imports| ext_threading
    xray_tensor_diffractometer_py -.->|imports| ext_time
    xray_tensor_diffractometer_py -.->|imports| ext_collections
    xray_tensor_diffractometer_py -.->|imports| ext_sys
    xray_tensor_diffractometer_py -.->|imports| ext_sys
    xray_tensor_diffractometer_py -.->|imports| ext_json
    xray_tensor_diffractometer_py -.->|imports| ext_traceback
```

---

## UML Class Diagram

Auto-generated Mermaid class diagram from parsed class-level symbols. Shows classes, structs, interfaces, traits, and their methods with inheritance and dependency relationships.

```mermaid
classDiagram
  class app_py_StrassenNet {
    <<class>>
    +__init__(self, rank)
    +forward(self, A, B)
  }
  class batch_size_py_Configuration {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_BilinearStrassenModel {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_CheckpointMigrator {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_CustomFormatMigrator {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_StandardFormatMigrator {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_CheckpointMigrationManager {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_StrassenDataGenerator {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_CrystallographyMetrics {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_PlanckConstantCalculator {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_BatchSizeThermodynamics {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_StrassenCheckpointLoader {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class batch_size_py_StrassenPlanckAnalyzer {
    <<class>>
    +set_random_seed(seed)
    +main()
    +__init__(self, config)
    +forward(self, a, b)
    +get_coefficients(self)
    +compute_lambda_effective(self)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
    +can_migrate(self, state_dict)
    +migrate(self, state_dict)
  }
  class boltzmann_experiments_py_Config {
    <<class>>
    +set_seed(seed)
    +main()
    +_simulate_training_trajectory(self, final_params, final_delta)
    +_compute_generalization_entropy(self, params, successful_ckpts)
    +_fit_timescale(self, entropy_values)
    +_plot_entropy_production(self, t, S, dS_dt, ckpt_name)
    +phase3_extensivity_law(self)
    +load_checkpoint(self, path, device)
    +load_checkpoint(self, path, device)
    +migrate_checkpoint(raw_data)
  }
  class boltzmann_experiments_py_CheckpointLoadingError {
    <<class>>
    +set_seed(seed)
    +main()
    +_simulate_training_trajectory(self, final_params, final_delta)
    +_compute_generalization_entropy(self, params, successful_ckpts)
    +_fit_timescale(self, entropy_values)
    +_plot_entropy_production(self, t, S, dS_dt, ckpt_name)
    +phase3_extensivity_law(self)
    +load_checkpoint(self, path, device)
    +load_checkpoint(self, path, device)
    +migrate_checkpoint(raw_data)
  }
  class boltzmann_experiments_py_ICheckpointLoader {
    <<class>>
    +set_seed(seed)
    +main()
    +_simulate_training_trajectory(self, final_params, final_delta)
    +_compute_generalization_entropy(self, params, successful_ckpts)
    +_fit_timescale(self, entropy_values)
    +_plot_entropy_production(self, t, S, dS_dt, ckpt_name)
    +phase3_extensivity_law(self)
    +load_checkpoint(self, path, device)
    +load_checkpoint(self, path, device)
    +migrate_checkpoint(raw_data)
  }
  class boltzmann_experiments_py_CheckpointLoader {
    <<class>>
    +set_seed(seed)
    +main()
    +_simulate_training_trajectory(self, final_params, final_delta)
    +_compute_generalization_entropy(self, params, successful_ckpts)
    +_fit_timescale(self, entropy_values)
    +_plot_entropy_production(self, t, S, dS_dt, ckpt_name)
    +phase3_extensivity_law(self)
    +load_checkpoint(self, path, device)
    +load_checkpoint(self, path, device)
    +migrate_checkpoint(raw_data)
  }
  class boltzmann_experiments_py_CheckpointMigrator {
    <<class>>
    +set_seed(seed)
    +main()
    +_simulate_training_trajectory(self, final_params, final_delta)
    +_compute_generalization_entropy(self, params, successful_ckpts)
    +_fit_timescale(self, entropy_values)
    +_plot_entropy_production(self, t, S, dS_dt, ckpt_name)
    +phase3_extensivity_law(self)
    +load_checkpoint(self, path, device)
    +load_checkpoint(self, path, device)
    +migrate_checkpoint(raw_data)
  }
  class boltzmann_experiments_py_BilinearStrassenModel {
    <<class>>
    +set_seed(seed)
    +main()
    +_simulate_training_trajectory(self, final_params, final_delta)
    +_compute_generalization_entropy(self, params, successful_ckpts)
    +_fit_timescale(self, entropy_values)
    +_plot_entropy_production(self, t, S, dS_dt, ckpt_name)
    +phase3_extensivity_law(self)
    +load_checkpoint(self, path, device)
    +load_checkpoint(self, path, device)
    +migrate_checkpoint(raw_data)
  }
  class boltzmann_experiments_py_CrystallographyMetrics {
    <<class>>
    +set_seed(seed)
    +main()
    +_simulate_training_trajectory(self, final_params, final_delta)
    +_compute_generalization_entropy(self, params, successful_ckpts)
    +_fit_timescale(self, entropy_values)
    +_plot_entropy_production(self, t, S, dS_dt, ckpt_name)
    +phase3_extensivity_law(self)
    +load_checkpoint(self, path, device)
    +load_checkpoint(self, path, device)
    +migrate_checkpoint(raw_data)
  }
  class boltzmann_experiments_py_DLProgram {
    <<class>>
    +set_seed(seed)
    +main()
    +_simulate_training_trajectory(self, final_params, final_delta)
    +_compute_generalization_entropy(self, params, successful_ckpts)
    +_fit_timescale(self, entropy_values)
    +_plot_entropy_production(self, t, S, dS_dt, ckpt_name)
    +phase3_extensivity_law(self)
    +load_checkpoint(self, path, device)
    +load_checkpoint(self, path, device)
    +migrate_checkpoint(raw_data)
  }
  class crystallography_py_Config {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_BilinearStrassenModel {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_CheckpointMigrator {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_StrassenDataGenerator {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_SparsificationProtocol {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_CrystallographyMetrics {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_StrassenDiffractionTest {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_BasinResilienceSpectrometer {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_CrystalPurityIndex {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_StrassenCrystallographer {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class crystallography_py_LocalComplexity {
    <<class>>
    +set_seed(seed)
    +main()
    +__init__(self, n_slots)
    +_initialize_symmetric(self)
    +forward(self, a, b)
    +get_coefficients(self)
    +migrate_checkpoint(path, device)
    +_migrate_custom(state_dict)
    +_migrate_encoder(state_dict)
    +_migrate_standard(state_dict)
  }
  class dirac_polos_zeros_py_AnalysisConfig {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IModel {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IChargeDistributionExtractor {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IDiracAnalyzer {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IFieldCalculator {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IFluxCalculator {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IStateSpaceExtractor {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_ITransferFunctionComputer {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IPoleZeroAnalyzer {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IFrequencyAnalyzer {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_ITimeResponseAnalyzer {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_ICheckpointLoader {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_ICheckpointMigrator {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_IVisualizer {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_BilinearModel {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_ChargeDistributionExtractor {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_DiracDeltaAnalyzer {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
  class dirac_polos_zeros_py_ElectricFieldCalculator {
    <<class>>
    +main()
    +forward(self, a, b)
    +get_coefficients(self)
    +extract(self, model)
    +analyze(self, charge_density)
    +calculate(self, dirac_data, eval_points)
    +calculate(self, electric_field, surface_points)
    +extract(self, model)
    +compute(self, A, B, C, D)
    +analyze_stability(self)
  }
```

---

## Code Property Graph

Machine-readable Code Property Graph (CPG) in JSON-LD format. This block allows AI agents to parse the full structural graph without additional file reads. Compatible with GraphRAG pipelines.

```json
{"@context": "https://schema.org", "analysis": {"communities": [], "god_nodes": [{"node_id": "xray_tensor_diffractometer.py", "score": 13.2}, {"node_id": "dirac_polos_zeros.py", "score": 12.3}, {"node_id": "full_seed_prospector.py", "score": 11.6}, {"node_id": "experiments/extended_experiments/all_test_extended.py", "score": 11.4}, {"node_id": "unified_hidden_connections_suite.py", "score": 9.9}, {"node_id": "percolation_analysis.py", "score": 9.1}, {"node_id": "mbl_analyzer.py", "score": 8.9}, {"node_id": "gravity.py", "score": 8.8}, {"node_id": "grain.py", "score": 7.7}, {"node_id": "plank.py", "score": 7.2}], "surprising_connections": []}, "edges": [{"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "onnx"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "onnxruntime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "onnxruntime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "app.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "batch_size.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "sklearn.decomposition"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "boltzmann_experiments.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "compute_gns_checkpoints.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "compute_gns_checkpoints.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "compute_gns_checkpoints.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "compute_gns_checkpoints.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "compute_gns_checkpoints.py", "target": "vector8"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "crystallography.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "matplotlib.colors"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "matplotlib.cm"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "dirac_polos_zeros.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_8192.py", "target": "ctypes"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_8192.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_8192.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_8192.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_8192.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_8192.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_study.py", "target": "ctypes"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_study.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_study.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_study.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_study.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_study.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/ablation/ablation_study.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/apendix_experiments.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/cache_analysis_v2.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "pandas"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "scipy.spatial.distance"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "hashlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "signal"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/all_test_extended.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp2_noise_ablation.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "sklearn.metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp3_prospective_prediction.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/run_all_experiments.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "copy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "sklearn.metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "validate_all_revisor_experiments"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/extended_experiments/validate2.py", "target": "sklearn.metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "sklearn.decomposition"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "sklearn.cluster"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/generate_figures.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/coherence_analysis.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/coherence_analysis.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/coherence_analysis.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/coherence_analysis.py", "target": "threadpoolctl"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/statistics/rigorous_experiment.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation/benchmark.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation/benchmark.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation/benchmark.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation/benchmark.py", "target": "threadpoolctl"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation_experiments.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation_experiments.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation_experiments.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation_experiments.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation_experiments.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/validation_experiments.py", "target": "itertools"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/verify_checkpoints.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/verify_checkpoints.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/verify_checkpoints.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/verify_checkpoints.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experiments/verify_checkpoints.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "copy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "experimetn2.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "fermi.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "signal"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "enum"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "full_seed_prospector.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "signal"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grain.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "scipy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "matplotlib.colors"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "matplotlib.cm"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "matplotlib.patches"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "sklearn.metrics"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "gravity.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "grigori_perelmans_ricci_flow.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "pickle"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "io"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "hawking_radiation.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "scipy.fft"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "mpl_toolkits.mplot3d"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "maxwell_strassen_analysis.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "mbl_analyzer.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "menu.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "menu.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "menu.py", "target": "subprocess"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "menu.py", "target": "textwrap"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "scipy.sparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "scipy.sparse.csgraph"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "matplotlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "re"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "types"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "types"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "percolation_analysis.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "plank.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "purity_index.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "copy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "csv"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "repor_experiments.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "glob"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "scrodingger.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_final.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_final.py", "target": "ctypes"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_final.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_final.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_scientific.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_scientific.py", "target": "ctypes"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_scientific.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_scientific.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_scientific.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_scientific.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_scientific.py", "target": "traceback"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "gc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "strassen"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "tomllib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/benchmark_strassen.py", "target": "tomli"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/strassen_numpy.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/strassen_numpy.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/strassen_numpy.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/benchmarks/strassen_numpy.py", "target": "functools"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/discovery/auto_T_discovery.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/discovery/auto_T_discovery.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/discovery/auto_T_discovery.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/discovery/auto_T_discovery.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/discovery/auto_T_discovery.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/discovery/auto_T_discovery.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_c.c", "target": "stdlib.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_c.c", "target": "string.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_c.c", "target": "stdio.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_optimal.c", "target": "stdlib.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_optimal.c", "target": "string.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_optimal.c", "target": "stdio.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_optimal.c", "target": "cblas.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_turbo.c", "target": "stdlib.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_turbo.c", "target": "string.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_turbo.c", "target": "stdio.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_turbo.c", "target": "omp.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/native/strassen_turbo.c", "target": "immintrin.h"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/convergence_theory.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/convergence_theory.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/convergence_theory.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/convergence_theory.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/convergence_theory.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/convergence_theory.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/grokkit_physics.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/grokkit_physics.py", "target": "ctypes"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/grokkit_physics.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/grokkit_physics.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/grokkit_physics.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main_pure_math.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main_pure_math.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main_pure_math.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main_pure_math.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/main_pure_math.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/strassen_core.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/strassen_core.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/strassen_grokkit.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/strassen_grokkit.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/strassen_grokkit.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/strassen_grokkit.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/train_strassen.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/train_strassen.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/train_strassen.py", "target": "torch.optim"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "src/training/train_strassen.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "tqdm"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "superposition.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_batch_sweep.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_batch_sweep.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_batch_sweep.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_batch_sweep.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_batch_sweep.py", "target": "vector8"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "train_batch_sweep.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "__future__"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "copy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "math"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "torch.nn.functional"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "unified_hidden_connections_suite.py", "target": "torch.utils.data"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "argparse"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "torch"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "torch.nn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "numpy"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "random"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "os"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "matplotlib.pyplot"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "datetime"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "typing"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "abc"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "seaborn"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "scipy.stats"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "scipy.linalg"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "scipy.optimize"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "sklearn.decomposition"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "warnings"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "logging"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "dataclasses"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "pathlib"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "threading"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "time"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "collections"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "sys"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "json"}, {"confidence": "EXTRACTED", "relation": "imports", "source": "xray_tensor_diffractometer.py", "target": "traceback"}], "generator": "readmenator", "metadata": {"edge_count": 16832, "file_count": 60, "language_count": 3, "symbol_count": 2071}, "nodes": [{"doc": "_*_ coding: utf8 _*_", "id": "app.py", "kind": "module", "label": "app.py", "language": "py", "sha256": "77cb34a89f3c4483", "symbol_count": 3, "symbols": [{"kind": "class", "line": 24, "name": "StrassenNet", "signature": "class StrassenNet(Module)"}, {"kind": "method", "line": 25, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 31, "name": "forward", "signature": "def forward(self, A, B)"}]}, {"id": "batch_size.py", "kind": "module", "label": "batch_size.py", "language": "py", "sha256": "d4d0858b8b04aa47", "symbol_count": 41, "symbols": [{"kind": "class", "line": 31, "name": "Configuration", "signature": "class Configuration"}, {"kind": "method", "line": 71, "name": "set_random_seed", "signature": "def set_random_seed(seed)"}, {"kind": "class", "line": 77, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"kind": "class", "line": 100, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator(ABC)"}, {"kind": "class", "line": 110, "name": "CustomFormatMigrator", "signature": "class CustomFormatMigrator(CheckpointMigrator)"}, {"kind": "class", "line": 122, "name": "StandardFormatMigrator", "signature": "class StandardFormatMigrator(CheckpointMigrator)"}, {"kind": "class", "line": 131, "name": "CheckpointMigrationManager", "signature": "class CheckpointMigrationManager"}, {"kind": "class", "line": 147, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"kind": "class", "line": 156, "name": "CrystallographyMetrics", "signature": "class CrystallographyMetrics"}, {"kind": "class", "line": 186, "name": "PlanckConstantCalculator", "signature": "class PlanckConstantCalculator"}, {"kind": "class", "line": 217, "name": "BatchSizeThermodynamics", "signature": "class BatchSizeThermodynamics"}, {"kind": "class", "line": 262, "name": "StrassenCheckpointLoader", "signature": "class StrassenCheckpointLoader"}, {"kind": "class", "line": 292, "name": "StrassenPlanckAnalyzer", "signature": "class StrassenPlanckAnalyzer"}, {"kind": "method", "line": 341, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 78, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 88, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 91, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 94, "name": "compute_lambda_effective", "signature": "def compute_lambda_effective(self)"}, {"kind": "method", "line": 102, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 106, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 111, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 114, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 123, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 126, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 132, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 135, "name": "migrate_checkpoint", "signature": "def migrate_checkpoint(self, path, device)"}, {"kind": "method", "line": 149, "name": "generate_batch", "signature": "def generate_batch(batch_size, config)"}, {"kind": "method", "line": 158, "name": "compute_kappa", "signature": "def compute_kappa(model, num_batches, config)"}, {"kind": "method", "line": 174, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(coeffs)"}, {"kind": "method", "line": 178, "name": "compute_local_complexity", "signature": "def compute_local_complexity(model, config)"}, {"kind": "method", "line": 187, "name": "__init__", "signature": "def __init__(self, metrics, training_metrics, config)"}, {"kind": "method", "line": 196, "name": "calculate_all", "signature": "def calculate_all(self)"}, {"kind": "method", "line": 218, "name": "__init__", "signature": "def __init__(self, model, h_bar, delta_struct, config)"}, {"kind": "method", "line": 224, "name": "analyze_batch_size_spectrum", "signature": "def analyze_batch_size_spectrum(self)"}, {"kind": "method", "line": 246, "name": "_measure_gradients", "signature": "def _measure_gradients(self, batch_size)"}, {"kind": "method", "line": 263, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 267, "name": "load", "signature": "def load(self, path, device)"}, {"kind": "method", "line": 281, "name": "extract_training_metrics", "signature": "def extract_training_metrics(self, path)"}, {"kind": "method", "line": 293, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 297, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(self, path, device)"}, {"kind": "method", "line": 323, "name": "analyze_directory", "signature": "def analyze_directory(self, directory, device, pattern)"}]}, {"id": "boltzmann_experiments.py", "kind": "module", "label": "boltzmann_experiments.py", "language": "py", "sha256": "09848c341836aca9", "symbol_count": 60, "symbols": [{"kind": "class", "line": 19, "name": "Config", "signature": "class Config"}, {"kind": "method", "line": 30, "name": "set_seed", "signature": "def set_seed(seed)"}, {"kind": "class", "line": 37, "name": "CheckpointLoadingError", "signature": "class CheckpointLoadingError(Exception)"}, {"kind": "class", "line": 40, "name": "ICheckpointLoader", "signature": "class ICheckpointLoader(ABC)"}, {"kind": "class", "line": 45, "name": "CheckpointLoader", "signature": "class CheckpointLoader(ICheckpointLoader)"}, {"kind": "class", "line": 52, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"kind": "class", "line": 118, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"kind": "class", "line": 137, "name": "CrystallographyMetrics", "signature": "class CrystallographyMetrics"}, {"kind": "class", "line": 199, "name": "DLProgram", "signature": "class DLProgram"}, {"kind": "method", "line": 935, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 951, "name": "_simulate_training_trajectory", "signature": "def _simulate_training_trajectory(self, final_params, final_delta)"}, {"doc": "Entropía de generalización con manejo robusto de dimensionalidad", "kind": "method", "line": 962, "name": "_compute_generalization_entropy", "signature": "def _compute_generalization_entropy(self, params, successful_ckpts)"}, {"kind": "method", "line": 1016, "name": "_fit_timescale", "signature": "def _fit_timescale(self, entropy_values)"}, {"kind": "method", "line": 1026, "name": "_plot_entropy_production", "signature": "def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)"}, {"kind": "method", "line": 1044, "name": "phase3_extensivity_law", "signature": "def phase3_extensivity_law(self)"}, {"kind": "method", "line": 42, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path, device)"}, {"kind": "method", "line": 46, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path, device)"}, {"kind": "method", "line": 54, "name": "migrate_checkpoint", "signature": "def migrate_checkpoint(raw_data)"}, {"kind": "method", "line": 70, "name": "_format_direct_tensors", "signature": "def _format_direct_tensors(tensor_dict)"}, {"kind": "method", "line": 92, "name": "_migrate_dict", "signature": "def _migrate_dict(state_dict)"}, {"kind": "method", "line": 105, "name": "_migrate_encoder_format", "signature": "def _migrate_encoder_format(state_dict)"}, {"kind": "method", "line": 115, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(state_dict)"}, {"kind": "method", "line": 119, "name": "__init__", "signature": "def __init__(self, n_slots)"}, {"kind": "method", "line": 126, "name": "_initialize_symmetric", "signature": "def _initialize_symmetric(self)"}, {"kind": "method", "line": 131, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 134, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"doc": "Classical kappa - will be inf for discrete states", "kind": "method", "line": 139, "name": "compute_kappa", "signature": "def compute_kappa(coeffs)"}, {"doc": "Discretization error δ", "kind": "method", "line": 156, "name": "compute_delta", "signature": "def compute_delta(coeffs)"}, {"kind": "method", "line": 161, "name": "compute_local_complexity", "signature": "def compute_local_complexity(coeffs)"}, {"doc": "Alpha purity: α = -log(δ), inverse temperature metric for discrete states", "kind": "method", "line": 169, "name": "compute_alpha_purity", "signature": "def compute_alpha_purity(coeffs)"}, {"doc": "Quantum-regularized kappa for singular covariance states", "kind": "method", "line": 178, "name": "compute_kappa_quantum", "signature": "def compute_kappa_quantum(coeffs, hbar)"}, {"kind": "method", "line": 200, "name": "__init__", "signature": "def __init__(self, checkpoint_dir, results_dir)"}, {"kind": "method", "line": 207, "name": "_load_all_checkpoints", "signature": "def _load_all_checkpoints(self)"}, {"kind": "method", "line": 247, "name": "run_full_boltzmann_program", "signature": "def run_full_boltzmann_program(self)"}, {"kind": "method", "line": 267, "name": "_print_executive_summary", "signature": "def _print_executive_summary(self, results)"}, {"kind": "method", "line": 307, "name": "_save_results", "signature": "def _save_results(self, results, filename)"}, {"kind": "method", "line": 328, "name": "phase1_molecular_hypothesis", "signature": "def phase1_molecular_hypothesis(self)"}, {"doc": "Entropía simple sin KDE para datos de baja varianza", "kind": "method", "line": 435, "name": "_compute_entropy_simple", "signature": "def _compute_entropy_simple(self, params)"}, {"doc": "Entropía con manejo robusto de covarianza", "kind": "method", "line": 446, "name": "_compute_entropy", "signature": "def _compute_entropy(self, params)"}, {"kind": "method", "line": 466, "name": "_compute_effective_volume", "signature": "def _compute_effective_volume(self, kde)"}, {"kind": "method", "line": 475, "name": "_plot_parameter_distribution", "signature": "def _plot_parameter_distribution(self, params, group_name, kde)"}, {"kind": "method", "line": 506, "name": "phase2_entropy_production", "signature": "def phase2_entropy_production(self)"}, {"kind": "method", "line": 592, "name": "_simulate_training_trajectory", "signature": "def _simulate_training_trajectory(self, final_params, final_delta)"}, {"doc": "Entropía de generalización con manejo robusto de datos idénticos", "kind": "method", "line": 604, "name": "_compute_generalization_entropy", "signature": "def _compute_generalization_entropy(self, params, successful_ckpts)"}, {"kind": "method", "line": 691, "name": "_fit_timescale", "signature": "def _fit_timescale(self, entropy_values)"}, {"kind": "method", "line": 701, "name": "_plot_entropy_production", "signature": "def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)"}, {"kind": "method", "line": 719, "name": "phase3_extensivity_law", "signature": "def phase3_extensivity_law(self)"}, {"kind": "method", "line": 773, "name": "_verify_scaling", "signature": "def _verify_scaling(self, coeffs, N)"}, {"kind": "method", "line": 783, "name": "_recursive_strassen", "signature": "def _recursive_strassen(self, A, B, coeffs, N)"}, {"kind": "method", "line": 812, "name": "_fit_extensivity", "signature": "def _fit_extensivity(self, errors, sizes, purity)"}, {"kind": "method", "line": 824, "name": "_verify_extensivity_universality", "signature": "def _verify_extensivity_universality(self, results)"}, {"kind": "method", "line": 828, "name": "_plot_extensivity", "signature": "def _plot_extensivity(self, sizes, errors, purity, ckpt_name)"}, {"kind": "method", "line": 841, "name": "phase4_quantum_basis_transform", "signature": "def phase4_quantum_basis_transform(self)"}, {"kind": "method", "line": 895, "name": "_find_broken_symmetries", "signature": "def _find_broken_symmetries(self, coeffs)"}, {"kind": "method", "line": 903, "name": "_measure_uncertainty", "signature": "def _measure_uncertainty(self, coeffs, basis)"}, {"kind": "method", "line": 914, "name": "_plot_uncertainty_distribution", "signature": "def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)"}, {"kind": "method", "line": 1017, "name": "model", "signature": "def model(t, A, tau, C)"}, {"kind": "method", "line": 310, "name": "convert_to_serializable", "signature": "def convert_to_serializable(obj)"}, {"kind": "method", "line": 692, "name": "model", "signature": "def model(t, A, tau, C)"}, {"kind": "method", "line": 813, "name": "model", "signature": "def model(N, alpha, beta)"}]}, {"doc": "compute_gns_by_batch.py", "id": "compute_gns_checkpoints.py", "kind": "module", "label": "compute_gns_checkpoints.py", "language": "py", "sha256": "cd39cf88cafb8ab3", "symbol_count": 2, "symbols": [{"kind": "function", "line": 11, "name": "estimate_gns", "signature": "def estimate_gns(model, batch_size, num_batches)"}, {"kind": "function", "line": 37, "name": "main", "signature": "def main()"}]}, {"id": "crystallography.py", "kind": "module", "label": "crystallography.py", "language": "py", "sha256": "952a7bbd201f3cea", "symbol_count": 45, "symbols": [{"kind": "class", "line": 25, "name": "Config", "signature": "class Config"}, {"kind": "method", "line": 35, "name": "set_seed", "signature": "def set_seed(seed)"}, {"kind": "class", "line": 44, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"kind": "class", "line": 71, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"kind": "class", "line": 155, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"kind": "class", "line": 172, "name": "SparsificationProtocol", "signature": "class SparsificationProtocol"}, {"kind": "class", "line": 206, "name": "CrystallographyMetrics", "signature": "class CrystallographyMetrics"}, {"kind": "class", "line": 232, "name": "StrassenDiffractionTest", "signature": "class StrassenDiffractionTest"}, {"kind": "class", "line": 277, "name": "BasinResilienceSpectrometer", "signature": "class BasinResilienceSpectrometer"}, {"kind": "class", "line": 345, "name": "CrystalPurityIndex", "signature": "class CrystalPurityIndex"}, {"kind": "class", "line": 416, "name": "StrassenCrystallographer", "signature": "class StrassenCrystallographer"}, {"kind": "class", "line": 523, "name": "LocalComplexity", "signature": "class LocalComplexity"}, {"kind": "method", "line": 542, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 45, "name": "__init__", "signature": "def __init__(self, n_slots)"}, {"kind": "method", "line": 52, "name": "_initialize_symmetric", "signature": "def _initialize_symmetric(self)"}, {"kind": "method", "line": 57, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 60, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 73, "name": "migrate_checkpoint", "signature": "def migrate_checkpoint(path, device)"}, {"doc": "Maneja formatos custom U,V,W directos", "kind": "method", "line": 106, "name": "_migrate_custom", "signature": "def _migrate_custom(state_dict)"}, {"doc": "Extracción de encoder.layers", "kind": "method", "line": 123, "name": "_migrate_encoder", "signature": "def _migrate_encoder(state_dict)"}, {"doc": "Formato estándar U.weight, V.weight, W.weight", "kind": "method", "line": 147, "name": "_migrate_standard", "signature": "def _migrate_standard(state_dict)"}, {"kind": "method", "line": 157, "name": "generate_batch", "signature": "def generate_batch(batch_size)"}, {"kind": "method", "line": 164, "name": "verify_structure", "signature": "def verify_structure(coeffs)"}, {"kind": "method", "line": 173, "name": "__init__", "signature": "def __init__(self, model)"}, {"kind": "method", "line": 176, "name": "prune_to_target", "signature": "def prune_to_target(self, target)"}, {"kind": "method", "line": 192, "name": "discretize_weights", "signature": "def discretize_weights(self, margin)"}, {"kind": "method", "line": 208, "name": "compute_kappa", "signature": "def compute_kappa(model, dataloader, num_batches)"}, {"kind": "method", "line": 225, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(coeffs)"}, {"kind": "method", "line": 233, "name": "__init__", "signature": "def __init__(self, model)"}, {"kind": "method", "line": 236, "name": "test_gauge_invariance", "signature": "def test_gauge_invariance(self, n_samples)"}, {"kind": "method", "line": 262, "name": "_functional_error", "signature": "def _functional_error(self, test_coeffs)"}, {"kind": "method", "line": 278, "name": "__init__", "signature": "def __init__(self, model)"}, {"kind": "method", "line": 282, "name": "measure_resilience_spectrum", "signature": "def measure_resilience_spectrum(self, noise_levels)"}, {"kind": "method", "line": 293, "name": "_test_noise_recovery", "signature": "def _test_noise_recovery(self, sigma, n_trials)"}, {"kind": "method", "line": 312, "name": "_apply_noise", "signature": "def _apply_noise(self, sigma)"}, {"kind": "method", "line": 317, "name": "_anneal_to_attractor", "signature": "def _anneal_to_attractor(self, max_epochs)"}, {"kind": "method", "line": 329, "name": "_estimate_critical_noise", "signature": "def _estimate_critical_noise(self, results)"}, {"kind": "method", "line": 346, "name": "__init__", "signature": "def __init__(self, model, diffraction_results, resilience_results, metrics_results)"}, {"kind": "method", "line": 359, "name": "compute", "signature": "def compute(self)"}, {"kind": "method", "line": 399, "name": "_assign_grade", "signature": "def _assign_grade(self, index, delta)"}, {"kind": "method", "line": 417, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"kind": "method", "line": 445, "name": "run_full_analysis", "signature": "def run_full_analysis(self)"}, {"kind": "method", "line": 506, "name": "_save_report", "signature": "def _save_report(self, report)"}, {"doc": "Computa LC basado en Can't Stop Won't Stop paper", "kind": "method", "line": 525, "name": "compute", "signature": "def compute(model)"}, {"kind": "method", "line": 465, "name": "dataloader_gen", "signature": "def dataloader_gen()"}]}, {"id": "dirac_polos_zeros.py", "kind": "module", "label": "dirac_polos_zeros.py", "language": "py", "sha256": "26e01677af9e4f55", "symbol_count": 123, "symbols": [{"kind": "class", "line": 24, "name": "AnalysisConfig", "signature": "class AnalysisConfig"}, {"kind": "class", "line": 54, "name": "IModel", "signature": "class IModel(Protocol)"}, {"kind": "class", "line": 60, "name": "IChargeDistributionExtractor", "signature": "class IChargeDistributionExtractor(Protocol)"}, {"kind": "class", "line": 65, "name": "IDiracAnalyzer", "signature": "class IDiracAnalyzer(Protocol)"}, {"kind": "class", "line": 70, "name": "IFieldCalculator", "signature": "class IFieldCalculator(Protocol)"}, {"kind": "class", "line": 75, "name": "IFluxCalculator", "signature": "class IFluxCalculator(Protocol)"}, {"kind": "class", "line": 80, "name": "IStateSpaceExtractor", "signature": "class IStateSpaceExtractor(Protocol)"}, {"kind": "class", "line": 85, "name": "ITransferFunctionComputer", "signature": "class ITransferFunctionComputer(Protocol)"}, {"kind": "class", "line": 90, "name": "IPoleZeroAnalyzer", "signature": "class IPoleZeroAnalyzer(Protocol)"}, {"kind": "class", "line": 97, "name": "IFrequencyAnalyzer", "signature": "class IFrequencyAnalyzer(Protocol)"}, {"kind": "class", "line": 104, "name": "ITimeResponseAnalyzer", "signature": "class ITimeResponseAnalyzer(Protocol)"}, {"kind": "class", "line": 110, "name": "ICheckpointLoader", "signature": "class ICheckpointLoader(Protocol)"}, {"kind": "class", "line": 115, "name": "ICheckpointMigrator", "signature": "class ICheckpointMigrator(Protocol)"}, {"kind": "class", "line": 120, "name": "IVisualizer", "signature": "class IVisualizer(Protocol)"}, {"kind": "class", "line": 124, "name": "BilinearModel", "signature": "class BilinearModel(Module)"}, {"kind": "class", "line": 152, "name": "ChargeDistributionExtractor", "signature": "class ChargeDistributionExtractor"}, {"kind": "class", "line": 160, "name": "DiracDeltaAnalyzer", "signature": "class DiracDeltaAnalyzer"}, {"kind": "class", "line": 192, "name": "ElectricFieldCalculator", "signature": "class ElectricFieldCalculator"}, {"kind": "class", "line": 225, "name": "ElectricFluxCalculator", "signature": "class ElectricFluxCalculator"}, {"kind": "class", "line": 248, "name": "DivergenceCalculator", "signature": "class DivergenceCalculator"}, {"kind": "class", "line": 253, "name": "GaussLawVerifier", "signature": "class GaussLawVerifier"}, {"kind": "class", "line": 271, "name": "StateSpaceExtractor", "signature": "class StateSpaceExtractor"}, {"kind": "class", "line": 298, "name": "TransferFunctionComputer", "signature": "class TransferFunctionComputer"}, {"kind": "class", "line": 313, "name": "PoleZeroAnalyzer", "signature": "class PoleZeroAnalyzer"}, {"kind": "class", "line": 463, "name": "FrequencyResponseAnalyzer", "signature": "class FrequencyResponseAnalyzer"}, {"kind": "class", "line": 566, "name": "TimeResponseAnalyzer", "signature": "class TimeResponseAnalyzer"}, {"kind": "class", "line": 653, "name": "CheckpointLoader", "signature": "class CheckpointLoader"}, {"kind": "class", "line": 661, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"kind": "class", "line": 710, "name": "ChargeDistributionVisualizer", "signature": "class ChargeDistributionVisualizer"}, {"kind": "class", "line": 738, "name": "ElectricFieldVisualizer", "signature": "class ElectricFieldVisualizer"}, {"kind": "class", "line": 780, "name": "DivergenceVisualizer", "signature": "class DivergenceVisualizer"}, {"kind": "class", "line": 809, "name": "PoleZeroVisualizer", "signature": "class PoleZeroVisualizer"}, {"kind": "class", "line": 848, "name": "BodeVisualizer", "signature": "class BodeVisualizer"}, {"kind": "class", "line": 899, "name": "NyquistVisualizer", "signature": "class NyquistVisualizer"}, {"kind": "class", "line": 936, "name": "TimeResponseVisualizer", "signature": "class TimeResponseVisualizer"}, {"kind": "class", "line": 966, "name": "CombinedVisualizer", "signature": "class CombinedVisualizer"}, {"kind": "class", "line": 1070, "name": "SystemAnalyzer", "signature": "class SystemAnalyzer"}, {"kind": "class", "line": 1288, "name": "AnalysisPipeline", "signature": "class AnalysisPipeline"}, {"kind": "method", "line": 1545, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 55, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 56, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 61, "name": "extract", "signature": "def extract(self, model)"}, {"kind": "method", "line": 66, "name": "analyze", "signature": "def analyze(self, charge_density)"}, {"kind": "method", "line": 71, "name": "calculate", "signature": "def calculate(self, dirac_data, eval_points)"}, {"kind": "method", "line": 76, "name": "calculate", "signature": "def calculate(self, electric_field, surface_points)"}, {"kind": "method", "line": 81, "name": "extract", "signature": "def extract(self, model)"}, {"kind": "method", "line": 86, "name": "compute", "signature": "def compute(self, A, B, C, D)"}, {"kind": "method", "line": 91, "name": "analyze_stability", "signature": "def analyze_stability(self)"}, {"kind": "method", "line": 92, "name": "get_poles", "signature": "def get_poles(self)"}, {"kind": "method", "line": 93, "name": "get_zeros", "signature": "def get_zeros(self)"}, {"kind": "method", "line": 98, "name": "compute_bode", "signature": "def compute_bode(self)"}, {"kind": "method", "line": 99, "name": "compute_margins", "signature": "def compute_margins(self)"}, {"kind": "method", "line": 100, "name": "compute_nyquist", "signature": "def compute_nyquist(self)"}, {"kind": "method", "line": 105, "name": "compute_step", "signature": "def compute_step(self)"}, {"kind": "method", "line": 106, "name": "compute_impulse", "signature": "def compute_impulse(self)"}, {"kind": "method", "line": 111, "name": "load", "signature": "def load(self, path, device)"}, {"kind": "method", "line": 116, "name": "migrate", "signature": "def migrate(self, raw_data)"}, {"kind": "method", "line": 121, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 125, "name": "__init__", "signature": "def __init__(self, hidden_dim, matrix_size)"}, {"kind": "method", "line": 136, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 141, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 144, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 153, "name": "extract", "signature": "def extract(self, model)"}, {"kind": "method", "line": 161, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 164, "name": "analyze", "signature": "def analyze(self, charge_density)"}, {"kind": "method", "line": 193, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 196, "name": "calculate", "signature": "def calculate(self, dirac_data, eval_points)"}, {"kind": "method", "line": 226, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 229, "name": "calculate", "signature": "def calculate(self, electric_field, surface_points)"}, {"kind": "method", "line": 249, "name": "calculate", "signature": "def calculate(self, electric_field)"}, {"kind": "method", "line": 254, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 257, "name": "verify", "signature": "def verify(self, dirac_data, flux_data)"}, {"kind": "method", "line": 272, "name": "extract", "signature": "def extract(self, model)"}, {"kind": "method", "line": 299, "name": "compute", "signature": "def compute(self, A, B, C, D)"}, {"kind": "method", "line": 314, "name": "__init__", "signature": "def __init__(self, numerator, denominator, config)"}, {"kind": "method", "line": 323, "name": "_compute", "signature": "def _compute(self)"}, {"kind": "method", "line": 334, "name": "get_poles", "signature": "def get_poles(self)"}, {"kind": "method", "line": 337, "name": "get_zeros", "signature": "def get_zeros(self)"}, {"kind": "method", "line": 340, "name": "analyze_stability", "signature": "def analyze_stability(self)"}, {"kind": "method", "line": 378, "name": "classify_poles", "signature": "def classify_poles(self)"}, {"kind": "method", "line": 406, "name": "compute_damping", "signature": "def compute_damping(self)"}, {"kind": "method", "line": 445, "name": "compute_time_constants", "signature": "def compute_time_constants(self)"}, {"kind": "method", "line": 464, "name": "__init__", "signature": "def __init__(self, numerator, denominator, config)"}, {"kind": "method", "line": 474, "name": "compute_bode", "signature": "def compute_bode(self)"}, {"kind": "method", "line": 490, "name": "compute_margins", "signature": "def compute_margins(self)"}, {"kind": "method", "line": 516, "name": "compute_nyquist", "signature": "def compute_nyquist(self)"}, {"kind": "method", "line": 535, "name": "evaluate_nyquist_stability", "signature": "def evaluate_nyquist_stability(self, nyquist_data)"}, {"kind": "method", "line": 567, "name": "__init__", "signature": "def __init__(self, numerator, denominator, config)"}, {"kind": "method", "line": 577, "name": "compute_step", "signature": "def compute_step(self)"}, {"kind": "method", "line": 589, "name": "compute_impulse", "signature": "def compute_impulse(self)"}, {"kind": "method", "line": 601, "name": "analyze_step_characteristics", "signature": "def analyze_step_characteristics(self, step_data)"}, {"kind": "method", "line": 654, "name": "load", "signature": "def load(self, path, device)"}, {"kind": "method", "line": 662, "name": "migrate", "signature": "def migrate(self, raw_data)"}, {"kind": "method", "line": 674, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict)"}, {"kind": "method", "line": 683, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(self, state_dict)"}, {"kind": "method", "line": 699, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(self, state_dict)"}, {"kind": "method", "line": 706, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(self, state_dict)"}, {"kind": "method", "line": 711, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 714, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 739, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 742, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 781, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 784, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 810, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 813, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 849, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 852, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 900, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 903, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 937, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 940, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 967, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 970, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 1071, "name": "__init__", "signature": "def __init__(self, checkpoint_path, config)"}, {"kind": "method", "line": 1088, "name": "_load_model", "signature": "def _load_model(self)"}, {"kind": "method", "line": 1104, "name": "analyze", "signature": "def analyze(self)"}, {"kind": "method", "line": 1204, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 1289, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1300, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"kind": "method", "line": 1357, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 1385, "name": "generate_summary", "signature": "def generate_summary(self, all_results, output_dir)"}, {"kind": "method", "line": 1406, "name": "_compute_aggregate_statistics", "signature": "def _compute_aggregate_statistics(self, results)"}, {"kind": "method", "line": 1473, "name": "_generate_text_report", "signature": "def _generate_text_report(self, summary, output_dir)"}]}, {"id": "experiments/ablation/ablation_8192.py", "kind": "module", "label": "ablation_8192.py", "language": "py", "sha256": "8d9358c2cfa7dcc4", "symbol_count": 0, "symbols": []}, {"id": "experiments/ablation/ablation_study.py", "kind": "module", "label": "ablation_study.py", "language": "py", "sha256": "4da9d6afe81b539f", "symbol_count": 13, "symbols": [{"kind": "class", "line": 27, "name": "BenchmarkResult", "signature": "class BenchmarkResult"}, {"doc": "Cargar bibliotecas con manejo de errores", "kind": "method", "line": 54, "name": "load_libraries", "signature": "def load_libraries()"}, {"doc": "Ejecutar multiplicación con OpenBLAS", "kind": "method", "line": 111, "name": "run_openblas", "signature": "def run_openblas(libs, A, B, C, n)"}, {"doc": "Ejecutar multiplicación con Strassen", "kind": "method", "line": 122, "name": "run_strassen", "signature": "def run_strassen(libs, name, func_name, A, B, C, n)"}, {"doc": "Benchmark una implementación", "kind": "method", "line": 132, "name": "benchmark_single", "signature": "def benchmark_single(libs, algo_name, func_name, A, B, C, C_ref, n, n_runs, warmup)"}, {"doc": "Ejecutar ablación completa", "kind": "method", "line": 180, "name": "run_ablation", "signature": "def run_ablation(libs, sizes, n_runs, warmup)"}, {"doc": "Analizar y presentar resultados", "kind": "method", "line": 233, "name": "analyze_results", "signature": "def analyze_results(results)"}, {"kind": "method", "line": 273, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 35, "name": "mean_time", "signature": "def mean_time(self)"}, {"kind": "method", "line": 39, "name": "std_time", "signature": "def std_time(self)"}, {"kind": "method", "line": 43, "name": "min_time", "signature": "def min_time(self)"}, {"kind": "method", "line": 47, "name": "max_time", "signature": "def max_time(self)"}, {"kind": "method", "line": 51, "name": "mean_gflops", "signature": "def mean_gflops(self)"}]}, {"id": "experiments/apendix_experiments.py", "kind": "module", "label": "apendix_experiments.py", "language": "py", "sha256": "67b67bd5d4e2d284", "symbol_count": 17, "symbols": [{"kind": "function", "line": 34, "name": "setup_matplotlib", "signature": "def setup_matplotlib()"}, {"kind": "class", "line": 51, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"kind": "method", "line": 87, "name": "generate_batch", "signature": "def generate_batch(n, device)"}, {"kind": "method", "line": 93, "name": "generate_test_set", "signature": "def generate_test_set(n, device)"}, {"kind": "method", "line": 100, "name": "compute_delta", "signature": "def compute_delta(model)"}, {"kind": "method", "line": 117, "name": "verify_strassen_structure", "signature": "def verify_strassen_structure(U_disc, V_disc, W_disc, tolerance)"}, {"kind": "method", "line": 138, "name": "compute_S_theta", "signature": "def compute_S_theta(model)"}, {"kind": "method", "line": 154, "name": "compute_gradient_covariance", "signature": "def compute_gradient_covariance(model, batch_size, n_samples)"}, {"kind": "method", "line": 187, "name": "train_with_logging", "signature": "def train_with_logging(batch_size, total_epochs, lr, wd, symmetric_init, seed, log_interval)"}, {"kind": "method", "line": 274, "name": "sparsify_and_discretize", "signature": "def sparsify_and_discretize(model, batch_size)"}, {"kind": "method", "line": 327, "name": "run_phase_diagram", "signature": "def run_phase_diagram()"}, {"kind": "method", "line": 429, "name": "run_batch_size_effect", "signature": "def run_batch_size_effect()"}, {"kind": "method", "line": 526, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 53, "name": "__init__", "signature": "def __init__(self, rank, symmetric_init)"}, {"kind": "method", "line": 67, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 77, "name": "slot_importance", "signature": "def slot_importance(self)"}, {"kind": "method", "line": 83, "name": "count_active", "signature": "def count_active(self, threshold)"}]}, {"id": "experiments/cache_analysis_v2.py", "kind": "module", "label": "cache_analysis_v2.py", "language": "py", "sha256": "1278d5cd1beb4a59", "symbol_count": 1, "symbols": [{"doc": "Full memory analysis for training.\n\nModel: Bilinear with rank-8 (target 7 active)\n- U: 8 x 4 = 32 floats\n- V: 8 x 4 = 32 floats  \n- W: 4 x 8 = 32 floats\nTotal params: 96 floats = 384 bytes\n\nAdamW optimizer state (per param):\n- m (momentum): 96 floats\n- v (variance): 96 floats\nTotal optimizer: 192 floats = 768 bytes\n\nPer-batch memory:\n- Input batch: B x 8 floats (A, B flattened)\n- Hidden: B x 8 floats (intermediate M)\n- Output: B x 4 floats (C)\n- Gradients: same as forward\nTotal per-sample: ~40 floats x 2 (forward+backward) = 80 floats = 320 bytes\n\nTotal training memory = model + optimizer + batch", "kind": "function", "line": 7, "name": "cache_analysis", "signature": "def cache_analysis()"}]}, {"id": "experiments/extended_experiments/all_test_extended.py", "kind": "module", "label": "all_test_extended.py", "language": "py", "sha256": "144450a9ba44025b", "symbol_count": 114, "symbols": [{"kind": "class", "line": 41, "name": "Configuration", "signature": "class Configuration"}, {"kind": "class", "line": 95, "name": "Narrator", "signature": "class Narrator"}, {"kind": "class", "line": 152, "name": "SystemFingerprint", "signature": "class SystemFingerprint"}, {"doc": "Dataset for arithmetic operations based on original implementation.\n\nGenerates (a, b) pairs and their product c = (a * b) mod MODULUS.\nEach a, b, c is a one-hot vector of size MODULUS.\n\nThe BilinearModel learns: c = W @ ((U @ a) * (V @ b))\nThis is a lookup table for modular multiplication.", "kind": "class", "line": 190, "name": "ArithmeticDataset", "signature": "class ArithmeticDataset(Dataset)"}, {"doc": "Bilinear model from original implementation.\n\nArchitecture:\n- U: Linear(d_vocab, rank) -> [67, 8] weights\n- V: Linear(d_vocab, rank) -> [67, 8] weights  \n- W: Linear(rank, d_vocab) -> [8, 67] weights\n\nForward:\n- a = x[:, 0] -> (batch, d_vocab)\n- b = x[:, 1] -> (batch, d_vocab)\n- m = U(a) * V(b) -> (batch, rank)\n- logits = W(m) -> (batch, d_vocab)", "kind": "class", "line": 238, "name": "BilinearModel", "signature": "class BilinearModel(Module)"}, {"kind": "class", "line": 293, "name": "Task", "signature": "class Task(ABC)"}, {"kind": "class", "line": 311, "name": "MatrixMultiplicationTask", "signature": "class MatrixMultiplicationTask(Task)"}, {"doc": "Dataset for parity task.", "kind": "class", "line": 354, "name": "ParityDataset", "signature": "class ParityDataset(Dataset)"}, {"kind": "class", "line": 375, "name": "ParityTask", "signature": "class ParityTask(Task)"}, {"doc": "Analyzes gradient covariance to understand batch size effects.", "kind": "class", "line": 402, "name": "GradientCovarianceProbe", "signature": "class GradientCovarianceProbe"}, {"doc": "Actively intervenes on condition number during training.", "kind": "class", "line": 465, "name": "SpectralInterventionProbe", "signature": "class SpectralInterventionProbe"}, {"doc": "Analyzes attractor landscapes to understand failure modes.", "kind": "class", "line": 488, "name": "AttractorLandscapeProbe", "signature": "class AttractorLandscapeProbe"}, {"doc": "Estimates volume of success basin in weight space.", "kind": "class", "line": 510, "name": "VolumeEstimator", "signature": "class VolumeEstimator"}, {"doc": "Tests robustness against various perturbations.", "kind": "class", "line": 524, "name": "RobustnessTest", "signature": "class RobustnessTest"}, {"doc": "Manages training checkpoints with configurable intervals.", "kind": "class", "line": 581, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"doc": "Comprehensive metrics collection for training progress.", "kind": "class", "line": 627, "name": "TrainingMetrics", "signature": "class TrainingMetrics"}, {"doc": "Main orchestration engine for training and evaluation.", "kind": "class", "line": 700, "name": "ExperimentRunner", "signature": "class ExperimentRunner"}, {"doc": "Analyzes discretization quality and success probability.", "kind": "class", "line": 809, "name": "DiscretizationAnalyzer", "signature": "class DiscretizationAnalyzer"}, {"doc": "Verifies zero-shot transfer to larger problem sizes.", "kind": "class", "line": 851, "name": "ExpansionVerifier", "signature": "class ExpansionVerifier"}, {"doc": "Executes the complete battery of open-question experiments.", "kind": "class", "line": 879, "name": "ExperimentPipeline", "signature": "class ExperimentPipeline"}, {"kind": "method", "line": 1403, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 89, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 96, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 101, "name": "begin", "signature": "def begin(self, experiment_name)"}, {"kind": "method", "line": 109, "name": "progress", "signature": "def progress(self, current, total, metrics)"}, {"kind": "method", "line": 116, "name": "checkpoint", "signature": "def checkpoint(self, epoch, loss, accuracy)"}, {"kind": "method", "line": 121, "name": "result", "signature": "def result(self, name, value, context)"}, {"kind": "method", "line": 126, "name": "verdict", "signature": "def verdict(self, hypothesis, evidence, conclusion)"}, {"kind": "method", "line": 131, "name": "failure", "signature": "def failure(self, reason, details)"}, {"kind": "method", "line": 136, "name": "complete", "signature": "def complete(self, summary)"}, {"kind": "method", "line": 144, "name": "claim", "signature": "def claim(self, statement, confidence)"}, {"kind": "method", "line": 148, "name": "note", "signature": "def note(self, observation)"}, {"kind": "method", "line": 153, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 156, "name": "capture", "signature": "def capture(self)"}, {"kind": "method", "line": 174, "name": "report", "signature": "def report(self)"}, {"kind": "method", "line": 202, "name": "__init__", "signature": "def __init__(self, size, modulus)"}, {"kind": "method", "line": 207, "name": "_generate_data", "signature": "def _generate_data(self)"}, {"kind": "method", "line": 231, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 234, "name": "__getitem__", "signature": "def __getitem__(self, idx)"}, {"kind": "method", "line": 253, "name": "__init__", "signature": "def __init__(self, d_vocab, rank, scale)"}, {"kind": "method", "line": 265, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 277, "name": "get_weights", "signature": "def get_weights(self)"}, {"kind": "method", "line": 283, "name": "get_U_weights", "signature": "def get_U_weights(self)"}, {"kind": "method", "line": 286, "name": "get_V_weights", "signature": "def get_V_weights(self)"}, {"kind": "method", "line": 289, "name": "get_W_weights", "signature": "def get_W_weights(self)"}, {"kind": "method", "line": 295, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 299, "name": "d_vocab", "signature": "def d_vocab(self)"}, {"kind": "method", "line": 303, "name": "generate_dataset", "signature": "def generate_dataset(self, size)"}, {"kind": "method", "line": 307, "name": "verify", "signature": "def verify(self, model, x, y)"}, {"kind": "method", "line": 312, "name": "__init__", "signature": "def __init__(self, modulus)"}, {"kind": "method", "line": 316, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 319, "name": "d_vocab", "signature": "def d_vocab(self)"}, {"kind": "method", "line": 322, "name": "generate_dataset", "signature": "def generate_dataset(self, size)"}, {"kind": "method", "line": 344, "name": "verify", "signature": "def verify(self, model, x, y)"}, {"kind": "method", "line": 357, "name": "__init__", "signature": "def __init__(self, size, bit_length)"}, {"kind": "method", "line": 362, "name": "_generate_data", "signature": "def _generate_data(self)"}, {"kind": "method", "line": 368, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 371, "name": "__getitem__", "signature": "def __getitem__(self, idx)"}, {"kind": "method", "line": 376, "name": "__init__", "signature": "def __init__(self, bit_length, modulus)"}, {"kind": "method", "line": 381, "name": "name", "signature": "def name(self)"}, {"kind": "method", "line": 384, "name": "d_vocab", "signature": "def d_vocab(self)"}, {"kind": "method", "line": 387, "name": "generate_dataset", "signature": "def generate_dataset(self, size)"}, {"kind": "method", "line": 391, "name": "verify", "signature": "def verify(self, model, x, y)"}, {"kind": "method", "line": 405, "name": "__init__", "signature": "def __init__(self, model)"}, {"kind": "method", "line": 409, "name": "capture_gradients", "signature": "def capture_gradients(self, dataloader, n_batches)"}, {"kind": "method", "line": 429, "name": "compute_covariance", "signature": "def compute_covariance(self)"}, {"kind": "method", "line": 436, "name": "compute_condition_number", "signature": "def compute_condition_number(self)"}, {"kind": "method", "line": 442, "name": "compute_gradient_noise_scale", "signature": "def compute_gradient_noise_scale(self, batch_size, learning_rate)"}, {"kind": "method", "line": 450, "name": "analyze", "signature": "def analyze(self, dataloader, batch_size, learning_rate)"}, {"kind": "method", "line": 468, "name": "__init__", "signature": "def __init__(self, model, target_kappa)"}, {"kind": "method", "line": 472, "name": "spectral_regularizer", "signature": "def spectral_regularizer(self)"}, {"kind": "method", "line": 491, "name": "__init__", "signature": "def __init__(self, model)"}, {"kind": "method", "line": 494, "name": "count_local_minima", "signature": "def count_local_minima(self, directions, losses)"}, {"kind": "method", "line": 501, "name": "measure_basin_width", "signature": "def measure_basin_width(self, weights, direction, n_points)"}, {"kind": "method", "line": 505, "name": "classify_failure_mode", "signature": "def classify_failure_mode(self, final_weights, initial_weights)"}, {"kind": "method", "line": 513, "name": "__init__", "signature": "def __init__(self, success_radius)"}, {"kind": "method", "line": 516, "name": "estimate_volume_monte_carlo", "signature": "def estimate_volume_monte_carlo(self, model_class, n_samples, success_checker)"}, {"kind": "method", "line": 520, "name": "compute_fractal_dimension", "signature": "def compute_fractal_dimension(self, trajectory)"}, {"kind": "method", "line": 527, "name": "__init__", "signature": "def __init__(self, model)"}, {"kind": "method", "line": 530, "name": "add_gaussian_noise", "signature": "def add_gaussian_noise(self, sigma)"}, {"kind": "method", "line": 536, "name": "fgsm_attack", "signature": "def fgsm_attack(self, x, y, epsilon)"}, {"kind": "method", "line": 544, "name": "quantize_weights", "signature": "def quantize_weights(self, bits)"}, {"kind": "method", "line": 550, "name": "test_discretization_with_noise", "signature": "def test_discretization_with_noise(self, sigma, checker)"}, {"kind": "method", "line": 560, "name": "run_fragility_analysis", "signature": "def run_fragility_analysis(self, sigma_values, checker)"}, {"kind": "method", "line": 584, "name": "__init__", "signature": "def __init__(self, config, experiment_name)"}, {"kind": "method", "line": 593, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, optimizer, epoch, metrics)"}, {"kind": "method", "line": 619, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path, model, optimizer)"}, {"kind": "method", "line": 630, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 642, "name": "update", "signature": "def update(self, train_loss, train_acc, test_loss, test_acc, kappa, grad_norm, weight_norm, disc_margin)"}, {"kind": "method", "line": 660, "name": "detect_grokking", "signature": "def detect_grokking(self, loss_threshold, test_loss_threshold, min_duration)"}, {"kind": "method", "line": 679, "name": "progress_bar_string", "signature": "def progress_bar_string(self, epoch, total_epochs)"}, {"kind": "method", "line": 703, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 708, "name": "train_epoch", "signature": "def train_epoch(self, model, dataloader, optimizer)"}, {"kind": "method", "line": 730, "name": "evaluate", "signature": "def evaluate(self, model, dataloader)"}, {"kind": "method", "line": 748, "name": "run_training", "signature": "def run_training(self, model, train_loader, test_loader, experiment_name, epochs, batch_size, lr, wd, verbose)"}, {"kind": "method", "line": 812, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 815, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(self, model)"}, {"kind": "method", "line": 827, "name": "discretize_weights", "signature": "def discretize_weights(self, model)"}, {"kind": "method", "line": 832, "name": "check_strassen_structure", "signature": "def check_strassen_structure(self, model, modulus)"}, {"kind": "method", "line": 839, "name": "count_discretized_parameters", "signature": "def count_discretized_parameters(self, model)"}, {"kind": "method", "line": 854, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 857, "name": "verify_expansion", "signature": "def verify_expansion(self, model, task, sizes)"}, {"kind": "method", "line": 882, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Experiment 1: Why batch size [24,128] works.", "kind": "method", "line": 891, "name": "experiment_batch_size_mechanism", "signature": "def experiment_batch_size_mechanism(self)"}, {"doc": "Experiment 2: Active intervention on κ.", "kind": "method", "line": 962, "name": "experiment_kappa_intervention", "signature": "def experiment_kappa_intervention(self)"}, {"doc": "Experiment 3: Why 32% of runs fail.", "kind": "method", "line": 1026, "name": "experiment_failure_analysis", "signature": "def experiment_failure_analysis(self)"}, {"doc": "Experiment 4: Generalization to other tasks.", "kind": "method", "line": 1088, "name": "experiment_generalization", "signature": "def experiment_generalization(self)"}, {"doc": "Experiment 5: Basin volume estimation.", "kind": "method", "line": 1145, "name": "experiment_basin_volume", "signature": "def experiment_basin_volume(self)"}, {"doc": "Experiment 6: Hardware reproducibility testing.", "kind": "method", "line": 1158, "name": "experiment_hardware_reproducibility", "signature": "def experiment_hardware_reproducibility(self)"}, {"doc": "Experiment 7: Discretization fragility testing.", "kind": "method", "line": 1217, "name": "experiment_fragility", "signature": "def experiment_fragility(self)"}, {"kind": "method", "line": 1332, "name": "run_all_experiments", "signature": "def run_all_experiments(self)"}, {"kind": "method", "line": 1371, "name": "_save_results", "signature": "def _save_results(self)"}, {"kind": "method", "line": 1379, "name": "_generate_summary", "signature": "def _generate_summary(self)"}, {"kind": "method", "line": 1253, "name": "checker", "signature": "def checker()"}]}, {"id": "experiments/extended_experiments/exp1_covariance_spectrometry.py", "kind": "module", "label": "exp1_covariance_spectrometry.py", "language": "py", "sha256": "5e85802db878e523", "symbol_count": 12, "symbols": [{"kind": "function", "line": 22, "name": "setup_matplotlib", "signature": "def setup_matplotlib()"}, {"doc": "Spectral operator for 2x2 matrix multiplication.\nTensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)", "kind": "class", "line": 48, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"doc": "Generate batch of matrices.", "kind": "method", "line": 112, "name": "generate_batch", "signature": "def generate_batch(n, scale)"}, {"doc": "Compute the gradient covariance matrix Σₜ and its eigenvalues.\n\nReturns:\n    kappa: condition number λ_max / λ_min\n    lambda_max: maximum eigenvalue\n    lambda_min: minimum eigenvalue  \n    trace: trace of covariance (total gradient energy)\n    frobenius_norm: Frobenius norm of covariance", "kind": "method", "line": 120, "name": "compute_gradient_covariance", "signature": "def compute_gradient_covariance(model, batch_size, n_samples)"}, {"doc": "Load model from checkpoint file.", "kind": "method", "line": 196, "name": "load_checkpoint", "signature": "def load_checkpoint(checkpoint_path)"}, {"doc": "Analyze a single checkpoint with multiple batch sizes.", "kind": "method", "line": 222, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(checkpoint_path, batch_sizes, n_samples, n_runs)"}, {"doc": "Main execution for Experiment 1.", "kind": "method", "line": 277, "name": "main", "signature": "def main()"}, {"doc": "Generate publication-quality figures.", "kind": "method", "line": 380, "name": "generate_visualization", "signature": "def generate_visualization(results, output_dir)"}, {"kind": "method", "line": 54, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 61, "name": "forward", "signature": "def forward(self, A, B)"}, {"doc": "Get all parameters as a single flattened vector.", "kind": "method", "line": 71, "name": "get_all_parameters", "signature": "def get_all_parameters(self)"}, {"doc": "Compute per-sample gradients for covariance estimation.\nReturns: gradients shape [batch_size, num_parameters]", "kind": "method", "line": 78, "name": "compute_per_sample_gradients", "signature": "def compute_per_sample_gradients(self, A, B, C_true)"}]}, {"id": "experiments/extended_experiments/exp2_noise_ablation.py", "kind": "module", "label": "exp2_noise_ablation.py", "language": "py", "sha256": "8c492e5ad7682df1", "symbol_count": 18, "symbols": [{"kind": "function", "line": 25, "name": "setup_matplotlib", "signature": "def setup_matplotlib()"}, {"doc": "Spectral operator for 2x2 matrix multiplication.", "kind": "class", "line": 49, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"doc": "Generate batch of matrices.", "kind": "method", "line": 99, "name": "generate_batch", "signature": "def generate_batch(n, scale)"}, {"doc": "Compute the gradient covariance matrix Σ.", "kind": "method", "line": 107, "name": "compute_gradient_covariance_matrix", "signature": "def compute_gradient_covariance_matrix(model, n_samples, batch_size)"}, {"doc": "Get eigenvectors and eigenvalues of covariance matrix.", "kind": "method", "line": 139, "name": "get_eigenbasis", "signature": "def get_eigenbasis(covariance)"}, {"doc": "Load model from checkpoint file.", "kind": "method", "line": 150, "name": "load_checkpoint", "signature": "def load_checkpoint(checkpoint_path)"}, {"doc": "Treatment A: Add noise to gradients DURING forward/backward pass.\n\nThis simulates the effect of gradient noise during training without actual retraining.", "kind": "method", "line": 172, "name": "experiment_treatment_a_gradient_noise", "signature": "def experiment_treatment_a_gradient_noise(model, noise_std, n_test)"}, {"doc": "Treatment B: Noise on weights BEFORE evaluation (already done in paper).\nThis is the fallback mechanism test.", "kind": "method", "line": 212, "name": "experiment_treatment_b_weight_noise", "signature": "def experiment_treatment_b_weight_noise(model, noise_std, n_test)"}, {"doc": "Treatment C: Structured noise by eigenvectors of Σ.\n\nTests whether damage is isotropic or aligned with gradient covariance directions.", "kind": "method", "line": 246, "name": "experiment_treatment_c_structured_noise", "signature": "def experiment_treatment_c_structured_noise(model, covariance, noise_std, n_test)"}, {"doc": "Run complete noise ablation experiment on a checkpoint.", "kind": "method", "line": 315, "name": "run_noise_ablation", "signature": "def run_noise_ablation(checkpoint_path, noise_levels)"}, {"doc": "Main execution for Experiment 2.", "kind": "method", "line": 348, "name": "main", "signature": "def main()"}, {"doc": "Generate publication-quality figures.", "kind": "method", "line": 433, "name": "generate_visualization", "signature": "def generate_visualization(results, output_dir)"}, {"kind": "method", "line": 54, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 61, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 71, "name": "get_all_parameters", "signature": "def get_all_parameters(self)"}, {"doc": "Set parameters from a flattened tensor.", "kind": "method", "line": 77, "name": "set_parameters", "signature": "def set_parameters(self, new_params)"}, {"doc": "Compute MSE loss.", "kind": "method", "line": 85, "name": "compute_loss", "signature": "def compute_loss(self, A, B)"}, {"doc": "Compute accuracy (proportion of predictions within threshold).", "kind": "method", "line": 91, "name": "compute_accuracy", "signature": "def compute_accuracy(self, A, B, threshold)"}]}, {"id": "experiments/extended_experiments/exp3_prospective_prediction.py", "kind": "module", "label": "exp3_prospective_prediction.py", "language": "py", "sha256": "cf621cb5e24ebd4c", "symbol_count": 17, "symbols": [{"kind": "function", "line": 32, "name": "setup_matplotlib", "signature": "def setup_matplotlib()"}, {"doc": "Spectral operator for 2x2 matrix multiplication.", "kind": "class", "line": 56, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"doc": "Generate batch of matrices.", "kind": "method", "line": 114, "name": "generate_batch", "signature": "def generate_batch(n, scale)"}, {"doc": "Compute condition number κ(Σ) of gradient covariance matrix.", "kind": "method", "line": 121, "name": "compute_kappa", "signature": "def compute_kappa(model, n_samples, batch_size)"}, {"doc": "Load model from checkpoint file.", "kind": "method", "line": 166, "name": "load_checkpoint", "signature": "def load_checkpoint(checkpoint_path)"}, {"doc": "Simulate the prospective prediction experiment.\n\nSince we can't actually retrain, we use available checkpoints to simulate:\n- \"Early\" checkpoint = first checkpoint in sequence\n- \"Final\" checkpoint = last checkpoint in sequence\n\nThis tests if early-stage κ predicts final-stage success.", "kind": "method", "line": 188, "name": "simulate_early_prediction", "signature": "def simulate_early_prediction(checkpoint_path, early_epoch_fraction)"}, {"doc": "Run the full prospective prediction experiment across all checkpoints.", "kind": "method", "line": 236, "name": "run_prospective_prediction_experiment", "signature": "def run_prospective_prediction_experiment(checkpoint_files)"}, {"doc": "Compute ROC curve and AUC for κ as predictor of success.", "kind": "method", "line": 252, "name": "compute_roc_analysis", "signature": "def compute_roc_analysis(predictions)"}, {"doc": "Main execution for Experiment 3.", "kind": "method", "line": 313, "name": "main", "signature": "def main()"}, {"doc": "Generate publication-quality figures.", "kind": "method", "line": 410, "name": "generate_visualization", "signature": "def generate_visualization(results, predictions, output_dir)"}, {"kind": "method", "line": 59, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 66, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 76, "name": "get_all_parameters", "signature": "def get_all_parameters(self)"}, {"kind": "method", "line": 82, "name": "set_parameters", "signature": "def set_parameters(self, new_params)"}, {"doc": "Count active slots based on weight norms.", "kind": "method", "line": 89, "name": "count_active_slots", "signature": "def count_active_slots(self, threshold)"}, {"doc": "Compute how close weights are to discrete values {-1, 0, 1}.\nδ(θ) = mean(|w - round(w)|)", "kind": "method", "line": 97, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(self)"}, {"doc": "Check if model has grokked (discretized with low error).", "kind": "method", "line": 107, "name": "is_grokked", "signature": "def is_grokked(self, margin_threshold, active_slots_target)"}]}, {"id": "experiments/extended_experiments/exp4_trajectory_perturbation.py", "kind": "module", "label": "exp4_trajectory_perturbation.py", "language": "py", "sha256": "c5af444ca350b87a", "symbol_count": 16, "symbols": [{"kind": "function", "line": 27, "name": "setup_matplotlib", "signature": "def setup_matplotlib()"}, {"doc": "Spectral operator for 2x2 matrix multiplication.", "kind": "class", "line": 51, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"doc": "Generate batch of matrices.", "kind": "method", "line": 117, "name": "generate_batch", "signature": "def generate_batch(n, scale)"}, {"doc": "Load model from checkpoint file.", "kind": "method", "line": 124, "name": "load_checkpoint", "signature": "def load_checkpoint(checkpoint_path)"}, {"doc": "Simulate trajectory perturbation effects using available checkpoints.\n\nSince we can't retrain, we simulate the effect of perturbations by:\n1. Taking a \"final\" checkpoint as the target\n2. Using earlier checkpoints to simulate \"early training state\"\n3. Applying perturbations and measuring their effect", "kind": "method", "line": 146, "name": "simulate_trajectory_perturbation", "signature": "def simulate_trajectory_perturbation(checkpoint_path, perturbations)"}, {"doc": "Main execution for Experiment 4.", "kind": "method", "line": 277, "name": "main", "signature": "def main()"}, {"doc": "Generate publication-quality figures.", "kind": "method", "line": 398, "name": "generate_visualization", "signature": "def generate_visualization(results, output_dir)"}, {"kind": "method", "line": 54, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 61, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 71, "name": "get_all_parameters", "signature": "def get_all_parameters(self)"}, {"kind": "method", "line": 77, "name": "set_parameters", "signature": "def set_parameters(self, new_params)"}, {"doc": "Get total L2 norm of all parameters.", "kind": "method", "line": 84, "name": "get_weight_norm", "signature": "def get_weight_norm(self)"}, {"doc": "Get normalized weight vector direction.", "kind": "method", "line": 91, "name": "get_weight_direction", "signature": "def get_weight_direction(self)"}, {"doc": "Compute norm of gradients.", "kind": "method", "line": 96, "name": "compute_gradient_norm", "signature": "def compute_gradient_norm(self, A, B)"}, {"doc": "Compute cosine similarity between current weights and target weights.", "kind": "method", "line": 110, "name": "cosine_similarity", "signature": "def cosine_similarity(self, other_params)"}, {"doc": "Compute evaluation metrics.", "kind": "method", "line": 194, "name": "compute_metrics", "signature": "def compute_metrics(model, name)"}]}, {"id": "experiments/extended_experiments/exp5_discreteness_attractors.py", "kind": "module", "label": "exp5_discreteness_attractors.py", "language": "py", "sha256": "bbb309ba64f50a60", "symbol_count": 0, "symbols": []}, {"id": "experiments/extended_experiments/run_all_experiments.py", "kind": "module", "label": "run_all_experiments.py", "language": "py", "sha256": "f505e54c45804374", "symbol_count": 14, "symbols": [{"kind": "function", "line": 24, "name": "setup_matplotlib", "signature": "def setup_matplotlib()"}, {"doc": "Spectral operator for 2x2 matrix multiplication.", "kind": "class", "line": 48, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"kind": "method", "line": 95, "name": "generate_batch", "signature": "def generate_batch(n, scale)"}, {"doc": "Load checkpoint with multiple format fallback strategies.", "kind": "method", "line": 101, "name": "load_checkpoint_robust", "signature": "def load_checkpoint_robust(checkpoint_path, model)"}, {"doc": "Compute κ(Σₜ) with numerical safety.", "kind": "method", "line": 135, "name": "compute_gradient_covariance_safe", "signature": "def compute_gradient_covariance_safe(model, batch_size, n_samples)"}, {"doc": "Run all experiments.", "kind": "method", "line": 205, "name": "run_all_experiments", "signature": "def run_all_experiments()"}, {"doc": "Generate summary visualization.", "kind": "method", "line": 496, "name": "generate_summary_visualization", "signature": "def generate_summary_visualization(results, output_dir)"}, {"kind": "method", "line": 51, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 58, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 68, "name": "get_all_parameters", "signature": "def get_all_parameters(self)"}, {"kind": "method", "line": 74, "name": "set_parameters", "signature": "def set_parameters(self, new_params)"}, {"kind": "method", "line": 81, "name": "count_active_slots", "signature": "def count_active_slots(self, threshold)"}, {"kind": "method", "line": 88, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(self)"}, {"kind": "method", "line": 324, "name": "compute_accuracy", "signature": "def compute_accuracy()"}]}, {"id": "experiments/extended_experiments/validate2.py", "kind": "module", "label": "validate2.py", "language": "py", "sha256": "e6258466b309d328", "symbol_count": 71, "symbols": [{"doc": "Configuración centralizada para todos los experimentos.\n\nSigue el principio de responsabilidad única - solo gestiona parámetros.\nNo usa magic numbers; todos los valores están definidos aquí.", "kind": "class", "line": 60, "name": "ExperimentConfig", "signature": "class ExperimentConfig"}, {"doc": "Operador Strassen para multiplicación de matrices 2x2 vía descomposición tensorial.\n\nEl modelo representa el tensor de rango R:\nC_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)\n\nDonde:\n- U, V: Coeficientes de combinación lineal (LC)\n- W: Coeficientes de reconstrucción\n- Sparsity (SP): Cuántos slots están activos", "kind": "class", "line": 126, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"doc": "Generador de datos para multiplicación de matrices 2x2.", "kind": "class", "line": 225, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"doc": "Calculadora de Complejidad Local basada en la varianza del gradiente.\n\nLC = ||grad||^2 / N (Noise Scale normalizada)\n\nEsta métrica captura la \"dificultad\" del batch actual y su relación\ncon el aprendizaje del modelo.", "kind": "class", "line": 277, "name": "LocalComplexityCalculator", "signature": "class LocalComplexityCalculator"}, {"doc": "Verifica que un modelo ha grokkeado correctamente.", "kind": "class", "line": 340, "name": "GrokkingVerifier", "signature": "class GrokkingVerifier"}, {"doc": "Motor para poda iterativa con fine-tuning completo.\n\nProtocolo:\n1. Calcular importancia de pesos (magnitud L1)\n2. Podar p% de pesos menos importantes\n3. Fine-tune por épocas especificadas\n4. Chequear degradación δ\n5. Si δ < threshold, continuar; si no, detener\n6. Verificar discretización con δ < 0.1 (PUNTO A DEL REVISOR)\n\nEste protocolo es CRUCIAL para verificar la hipótesis de cuenca discreta.", "kind": "class", "line": 405, "name": "IterativePruningEngine", "signature": "class IterativePruningEngine"}, {"doc": "Experimento de Local Complexity entrenando desde cero.\n\nEsto es CRUCIAL para responder al revisor (PUNTO B):\n- Se entrena un modelo desde cero hasta grokking\n- Se mide LC en cada época para capturar la transición de fase\n- Si LC muestra un cambio alrededor del grokking, la métrica es útil\n- Si LC permanece constante, la métrica NO captura la transición", "kind": "class", "line": 764, "name": "LocalComplexityExperiment", "signature": "class LocalComplexityExperiment"}, {"doc": "Generador de runs balanceados para calcular AUC válido.\n\nEsto es CRUCIAL para responder al revisor (PUNTO C):\n- Entrenar múltiples modelos con diferentes hiperparámetros\n- Algunos grokkean, otros no (condiciones variadas)\n- Generar dataset balanceado para ROC/AUC\n\nSi todos los samples son de una sola clase, AUC es indefinido.\nNecesitamos mix de grokked + no-grokked para calcularlo.", "kind": "class", "line": 890, "name": "BalancedRunsGenerator", "signature": "class BalancedRunsGenerator"}, {"doc": "Generador de estadísticas con intervalos de confianza bootstrap.\n\nCalcula:\n- Curvas ROC con IC del 95%\n- AUC con IC del 95%\n- Kappa de Cohen con IC del 95%", "kind": "class", "line": 1135, "name": "BootstrapStatistics", "signature": "class BootstrapStatistics"}, {"doc": "Generador de visualizaciones con estilo académico.", "kind": "class", "line": 1281, "name": "VisualizationGenerator", "signature": "class VisualizationGenerator"}, {"doc": "Orquestador principal para todos los experimentos.\n\nCoordina:\n1. Carga de checkpoint grokkeado\n2. Verificación de grokking\n3. Experimento de Local Complexity\n4. Protocolo de poda iterativa\n5. Análisis ROC/AUC\n6. Generación de visualizaciones", "kind": "class", "line": 1646, "name": "ExperimentOrchestrator", "signature": "class ExperimentOrchestrator"}, {"doc": "Buscar checkpoint grokkeado en múltiples ubicaciones.\n\nReturns:\n    Path al archivo de checkpoint grokkeado", "kind": "method", "line": 2313, "name": "find_grokked_checkpoint", "signature": "def find_grokked_checkpoint()"}, {"doc": "Analizar todos los checkpoints disponibles para encontrar el grokkeado.\n\nReturns:\n    Diccionario con métricas de cada checkpoint", "kind": "method", "line": 2353, "name": "analyze_checkpoints", "signature": "def analyze_checkpoints()"}, {"doc": "Punto de entrada principal.", "kind": "method", "line": 2423, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 115, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 139, "name": "__init__", "signature": "def __init__(self, rank)"}, {"doc": "Inicializar pesos con consideración para grokking.", "kind": "method", "line": 150, "name": "_initialize_weights", "signature": "def _initialize_weights(self)"}, {"doc": "Computar A @ B usando descomposición tensorial.\n\nArgs:\n    A: Tensor de entrada de forma (batch, 2, 2)\n    B: Tensor de entrada de forma (batch, 2, 2)\n    \nReturns:\n    Tensor de salida de forma (batch, 2, 2)", "kind": "method", "line": 156, "name": "forward", "signature": "def forward(self, A, B)"}, {"doc": "Importancia de cada slot basada en normas.", "kind": "method", "line": 183, "name": "slot_importance", "signature": "def slot_importance(self)"}, {"doc": "Contar slots activos.", "kind": "method", "line": 190, "name": "count_active", "signature": "def count_active(self, threshold)"}, {"doc": "Métrica de Sparsity. SP -> 0 significa máxima sparsity.", "kind": "method", "line": 194, "name": "compute_SP", "signature": "def compute_SP(self)"}, {"doc": "Obtener estado completo para checkpointing.", "kind": "method", "line": 202, "name": "get_state_dict", "signature": "def get_state_dict(self)"}, {"doc": "Cargar estado completo desde checkpoint.", "kind": "method", "line": 211, "name": "load_state_dict", "signature": "def load_state_dict(self, state_dict)"}, {"kind": "method", "line": 228, "name": "__init__", "signature": "def __init__(self, num_samples, matrix_size, seed)"}, {"doc": "Generar matriz aleatoria con valores enteros.", "kind": "method", "line": 238, "name": "generate_matrix", "signature": "def generate_matrix(self)"}, {"doc": "Generar pares de matrices y sus productos.", "kind": "method", "line": 242, "name": "generate_data", "signature": "def generate_data(self)"}, {"doc": "Dividir en conjuntos de entrenamiento y prueba.", "kind": "method", "line": 260, "name": "get_train_test", "signature": "def get_train_test(self, test_ratio)"}, {"kind": "method", "line": 287, "name": "__init__", "signature": "def __init__(self, model, config)"}, {"doc": "Calcular LC para un batch específico.\n\nLC = ||g||^2 / N_batch\n\nDonde g es el gradiente de la pérdida respecto a los pesos.", "kind": "method", "line": 292, "name": "compute_lc", "signature": "def compute_lc(self, batch_inputs, batch_targets)"}, {"doc": "Calcular diversidad del batch basada en varianza de activaciones.", "kind": "method", "line": 326, "name": "compute_batch_diversity", "signature": "def compute_batch_diversity(self, batch_inputs)"}, {"kind": "method", "line": 343, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Verificar que el operador ha grokkeado correctamente.\n\nReturns:\n    Tupla de (éxito, métricas)", "kind": "method", "line": 347, "name": "verify", "signature": "def verify(self, model, n_test)"}, {"doc": "Generar batch de matrices aleatorias.", "kind": "method", "line": 393, "name": "_generate_batch", "signature": "def _generate_batch(self, n, scale)"}, {"kind": "method", "line": 420, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Obtener magnitud absoluta de todos los pesos.", "kind": "method", "line": 426, "name": "get_weight_magnitudes", "signature": "def get_weight_magnitudes(self, model)"}, {"doc": "Calcular porcentaje de pesos en cero.", "kind": "method", "line": 431, "name": "compute_sparsity", "signature": "def compute_sparsity(self, model)"}, {"doc": "Podar el porcentaje especificado de pesos menos importantes.\n\nReturns: (num_pruned, current_sparsity)", "kind": "method", "line": 437, "name": "prune_percent", "signature": "def prune_percent(self, model, percent)"}, {"doc": "Fine-tune del modelo podado con métricas completas.", "kind": "method", "line": 457, "name": "fine_tune", "signature": "def fine_tune(self, model, train_data)"}, {"doc": "Generar batch de matrices aleatorias.", "kind": "method", "line": 510, "name": "_generate_batch", "signature": "def _generate_batch(self, n, scale)"}, {"doc": "Ejecutar protocolo completo de poda iterativa.\n\nReturns:\n    Diccionario con resultados completos", "kind": "method", "line": 624, "name": "run_protocol", "signature": "def run_protocol(self, model, train_data)"}, {"kind": "method", "line": 775, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Ejecutar experimento completo de LC entrenando desde cero.", "kind": "method", "line": 779, "name": "run_full_experiment", "signature": "def run_full_experiment(self, target_epochs)"}, {"kind": "method", "line": 882, "name": "_generate_batch", "signature": "def _generate_batch(self, n, scale)"}, {"kind": "method", "line": 903, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Ejecutar multiples runs con condiciones disenhadas para producir mix.\n\nReturns:\n    Diccionario con resultados de todos los runs", "kind": "method", "line": 907, "name": "run_balanced_experiments", "signature": "def run_balanced_experiments(self, n_runs)"}, {"doc": "Entrenar un solo modelo con configuración específica.", "kind": "method", "line": 1021, "name": "_train_single_run", "signature": "def _train_single_run(self, run_idx, config)"}, {"doc": "Calcular ROC/AUC básico.", "kind": "method", "line": 1099, "name": "_compute_roc", "signature": "def _compute_roc(self, y_true, y_scores)"}, {"kind": "method", "line": 1127, "name": "_generate_batch", "signature": "def _generate_batch(self, n, scale)"}, {"kind": "method", "line": 1145, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calcular curva ROC con intervalos de confianza bootstrap.\n\nReturns:\n    Diccionario con resultados ROC", "kind": "method", "line": 1149, "name": "compute_roc_with_ci", "signature": "def compute_roc_with_ci(self, y_true, y_scores)"}, {"doc": "Calcular Kappa de Cohen con IC bootstrap.", "kind": "method", "line": 1236, "name": "compute_kappa_with_ci", "signature": "def compute_kappa_with_ci(self, y_true, y_pred)"}, {"doc": "Calcular accuracy con IC binomial.", "kind": "method", "line": 1260, "name": "compute_accuracy_with_ci", "signature": "def compute_accuracy_with_ci(self, correct)"}, {"kind": "method", "line": 1284, "name": "__init__", "signature": "def __init__(self, style)"}, {"doc": "Graficar evolución de Local Complexity y Accuracy.", "kind": "method", "line": 1298, "name": "plot_local_complexity", "signature": "def plot_local_complexity(self, epochs, lc_values, accuracy, save_path)"}, {"doc": "Graficar resultados de poda iterativa.", "kind": "method", "line": 1342, "name": "plot_pruning_results", "signature": "def plot_pruning_results(self, pruning_data, save_path)"}, {"doc": "Graficar curva ROC con intervalos de confianza.", "kind": "method", "line": 1404, "name": "plot_roc_with_ci", "signature": "def plot_roc_with_ci(self, roc_data, save_path)"}, {"doc": "Graficar resultados del experimento de runs balanceados.", "kind": "method", "line": 1467, "name": "plot_balanced_runs_results", "signature": "def plot_balanced_runs_results(self, balanced_data, save_path)"}, {"doc": "Graficar resultados de discretizacion.", "kind": "method", "line": 1556, "name": "plot_discretization_results", "signature": "def plot_discretization_results(self, pruning_data, save_path)"}, {"kind": "method", "line": 1659, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Buscar checkpoint grokkeado en múltiples ubicaciones.\n\nReturns:\n    Path al archivo de checkpoint o None si no se encuentra", "kind": "method", "line": 1689, "name": "find_grokked_checkpoint", "signature": "def find_grokked_checkpoint(self)"}, {"doc": "Cargar checkpoint grokkeado y verificar que grokkeó.\n\nReturns:\n    Modelo cargado y verificado", "kind": "method", "line": 1733, "name": "load_grokked_checkpoint", "signature": "def load_grokked_checkpoint(self, checkpoint_path)"}, {"doc": "Verificar que el checkpoint cargado realmente grokkeó.\n\nReturns:\n    Tupla de (es_grokked, métricas)", "kind": "method", "line": 1768, "name": "verify_checkpoint_is_grokked", "signature": "def verify_checkpoint_is_grokked(self)"}, {"doc": "Ejecutar experimento de Local Complexity vs Época.\n\nNota: Como usamos un checkpoint ya grokked, esto mide la LC\ndurante el fine-tuning post-poda, no durante el grokking inicial.\n\nReturns:\n    Diccionario con historial de LC y accuracy", "kind": "method", "line": 1795, "name": "run_local_complexity_experiment", "signature": "def run_local_complexity_experiment(self, epochs)"}, {"doc": "Ejecutar experimento de Local Complexity .\n\n- Entrenar un modelo desde cero hasta grokking\n- Medir LC en cada época para capturar la transición de fase\n- Si LC muestra un cambio alrededor del grokking, la métrica es útil\n- Si LC permanece constante, la métrica NO captura la transición\n\nReturns:\n    Diccionario con historial completo del experimento", "kind": "method", "line": 1897, "name": "run_lc_training_experiment", "signature": "def run_lc_training_experiment(self, epochs)"}, {"doc": "Ejecutar protocolo de poda iterativa + fine-tuning.\n\nReturns:\n    Diccionario con resultados de poda", "kind": "method", "line": 1936, "name": "run_pruning_experiment", "signature": "def run_pruning_experiment(self)"}, {"doc": "Ejecutar experimento de runs balanceados (PUNTO C DEL REVISOR).\n\nEsto es CRUCIAL para obtener un AUC valido:\n- Entrenar multiples modelos con diferentes hiperparametros\n- Algunos grokkean, otros no\n- Generar dataset balanceado para ROC/AUC\n\nReturns:\n    Diccionario con resultados de todos los runs", "kind": "method", "line": 1980, "name": "run_balanced_runs_experiment", "signature": "def run_balanced_runs_experiment(self, n_runs)"}, {"doc": "Ejecutar análisis ROC/AUC con bootstrap.\n\nReturns:\n    Diccionario con resultados ROC", "kind": "method", "line": 2043, "name": "run_roc_analysis", "signature": "def run_roc_analysis(self)"}, {"doc": "Generar batch de matrices aleatorias.", "kind": "method", "line": 2117, "name": "_generate_batch", "signature": "def _generate_batch(self, n, scale)"}, {"doc": "Generar reporte de resumen en markdown.", "kind": "method", "line": 2124, "name": "generate_summary_report", "signature": "def generate_summary_report(self)"}, {"doc": "Guardar todos los resultados.", "kind": "method", "line": 2197, "name": "save_results", "signature": "def save_results(self)"}, {"doc": "Ejecutar suite completa de experimentos.", "kind": "method", "line": 2235, "name": "run_all_experiments", "signature": "def run_all_experiments(self, checkpoint_path)"}]}, {"id": "experiments/generate_figures.py", "kind": "module", "label": "generate_figures.py", "language": "py", "sha256": "e2416056a9625281", "symbol_count": 9, "symbols": [{"doc": "Configure matplotlib and seaborn for proper rendering.", "kind": "function", "line": 15, "name": "setup_matplotlib_for_plotting", "signature": "def setup_matplotlib_for_plotting()"}, {"doc": "Generate benchmark performance comparison plot.", "kind": "function", "line": 63, "name": "generate_benchmark_figure", "signature": "def generate_benchmark_figure()"}, {"doc": "Generate ablation study visualization.", "kind": "function", "line": 124, "name": "generate_ablation_figure", "signature": "def generate_ablation_figure()"}, {"doc": "Load all checkpoint files and extract weight tensors.", "kind": "function", "line": 219, "name": "load_checkpoint_weights", "signature": "def load_checkpoint_weights()"}, {"doc": "Generate weight space geometry visualization.", "kind": "function", "line": 258, "name": "generate_weight_geometry_figure", "signature": "def generate_weight_geometry_figure()"}, {"doc": "Generate phase transition analysis from checkpoint evolution.", "kind": "function", "line": 354, "name": "generate_phase_transition_figure", "signature": "def generate_phase_transition_figure()"}, {"doc": "Generate cache coherence analysis visualization.", "kind": "function", "line": 465, "name": "generate_coherence_figure", "signature": "def generate_coherence_figure()"}, {"doc": "Visualize the crystallization of Strassen coefficients.", "kind": "function", "line": 534, "name": "generate_crystallization_figure", "signature": "def generate_crystallization_figure()"}, {"kind": "function", "line": 623, "name": "main", "signature": "def main()"}]}, {"id": "experiments/statistics/coherence_analysis.py", "kind": "module", "label": "coherence_analysis.py", "language": "py", "sha256": "206e9e4e972e9d90", "symbol_count": 2, "symbols": [{"kind": "function", "line": 15, "name": "strassen_numpy", "signature": "def strassen_numpy(A, B, threshold)"}, {"kind": "function", "line": 42, "name": "run_coherence_analysis", "signature": "def run_coherence_analysis()"}]}, {"id": "experiments/statistics/rigorous_experiment.py", "kind": "module", "label": "rigorous_experiment.py", "language": "py", "sha256": "397957d422092b4d", "symbol_count": 19, "symbols": [{"doc": "Complete hyperparameter specification for reproducibility", "kind": "class", "line": 58, "name": "ExperimentConfig", "signature": "class ExperimentConfig"}, {"doc": "Single experiment result", "kind": "class", "line": 84, "name": "ExperimentResult", "signature": "class ExperimentResult"}, {"doc": "Strassen-like bilinear model", "kind": "class", "line": 106, "name": "StrassenModel", "signature": "class StrassenModel(Module)"}, {"doc": "Generate matrix multiplication dataset", "kind": "method", "line": 131, "name": "generate_data", "signature": "def generate_data(n_samples, seed)"}, {"doc": "Compute mean distance to nearest discrete value", "kind": "method", "line": 143, "name": "compute_discretization_error", "signature": "def compute_discretization_error(model, values)"}, {"doc": "Compute maximum spectral gap ratio", "kind": "method", "line": 157, "name": "compute_spectral_gap", "signature": "def compute_spectral_gap(model)"}, {"doc": "Run a single controlled experiment", "kind": "method", "line": 169, "name": "run_single_experiment", "signature": "def run_single_experiment(batch_size, seed, run_id, config)"}, {"doc": "Run complete factorial experiment", "kind": "method", "line": 269, "name": "run_full_experiment", "signature": "def run_full_experiment(batch_sizes, n_seeds, n_runs_per_seed)"}, {"doc": "Perform full factorial ANOVA\n\nReturns complete ANOVA table with SS, df, MS, F, p, η²", "kind": "method", "line": 306, "name": "perform_anova", "signature": "def perform_anova(results)"}, {"doc": "Print formatted ANOVA table", "kind": "method", "line": 401, "name": "print_anova_table", "signature": "def print_anova_table(anova)"}, {"doc": "Fit theoretical noise model:\nVar(loss) = α/B + β·cache_miss(B) + γ\n\nCompare to null model: Var(loss) = α/B + γ", "kind": "method", "line": 448, "name": "fit_noise_model", "signature": "def fit_noise_model(results)"}, {"doc": "Find optimal batch size with bootstrap confidence interval", "kind": "method", "line": 519, "name": "find_optimal_B", "signature": "def find_optimal_B(results, n_bootstrap)"}, {"doc": "Generate complete statistical report", "kind": "method", "line": 555, "name": "generate_report", "signature": "def generate_report(results, config)"}, {"kind": "method", "line": 108, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 124, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 467, "name": "cache_miss_proxy", "signature": "def cache_miss_proxy(B)"}, {"kind": "method", "line": 472, "name": "full_model", "signature": "def full_model(B, alpha, beta, gamma)"}, {"kind": "method", "line": 476, "name": "null_model", "signature": "def null_model(B, alpha, gamma)"}, {"kind": "method", "line": 526, "name": "get_mean_error", "signature": "def get_mean_error(data, B)"}]}, {"id": "experiments/validation/benchmark.py", "kind": "module", "label": "benchmark.py", "language": "py", "sha256": "285cbfdecf0cd0a3", "symbol_count": 3, "symbols": [{"doc": "Strassen recursivo con NumPy para productos base.", "kind": "function", "line": 15, "name": "strassen_numpy", "signature": "def strassen_numpy(A, B, threshold)"}, {"doc": "Mide tiempo de un solo sgemm de tamaño n.", "kind": "function", "line": 49, "name": "measure_single_sgemm", "signature": "def measure_single_sgemm(n, threads)"}, {"doc": "Ejecuta el análisis del Límite de Planck.", "kind": "function", "line": 59, "name": "run_planck_analysis", "signature": "def run_planck_analysis()"}]}, {"id": "experiments/validation_experiments.py", "kind": "module", "label": "validation_experiments.py", "language": "py", "sha256": "3c4a06a2e3f65378", "symbol_count": 9, "symbols": [{"doc": "Compute 2x2 matrix multiplication using Strassen coefficients.", "kind": "function", "line": 47, "name": "strassen_2x2", "signature": "def strassen_2x2(A, B, U, V, W)"}, {"doc": "Recursive Strassen for NxN matrices.", "kind": "function", "line": 56, "name": "strassen_recursive", "signature": "def strassen_recursive(A, B, U, V, W, threshold)"}, {"doc": "Test that permuting slots produces equivalent computation.", "kind": "function", "line": 85, "name": "test_uniqueness_via_permutation", "signature": "def test_uniqueness_via_permutation()"}, {"doc": "Test stability under Gaussian noise.", "kind": "function", "line": 125, "name": "test_noise_stability", "signature": "def test_noise_stability()"}, {"doc": "Test expansion to larger sizes.", "kind": "function", "line": 162, "name": "test_expansion_sizes", "signature": "def test_expansion_sizes()"}, {"doc": "Simulate grokking dynamics for visualization.", "kind": "function", "line": 184, "name": "simulate_grokking_dynamics", "signature": "def simulate_grokking_dynamics()"}, {"doc": "Compute L3 cache requirements for different batch sizes.", "kind": "function", "line": 250, "name": "compute_cache_math", "signature": "def compute_cache_math()"}, {"doc": "Run all validation experiments.", "kind": "function", "line": 294, "name": "main", "signature": "def main()"}, {"kind": "function", "line": 313, "name": "convert_types", "signature": "def convert_types(obj)"}]}, {"id": "experiments/verify_checkpoints.py", "kind": "module", "label": "verify_checkpoints.py", "language": "py", "sha256": "b190c34fb7c22fae", "symbol_count": 13, "symbols": [{"kind": "class", "line": 25, "name": "StrassenBilinear", "signature": "class StrassenBilinear(Module)"}, {"kind": "method", "line": 51, "name": "compute_delta", "signature": "def compute_delta(model)"}, {"kind": "method", "line": 68, "name": "verify_2x2", "signature": "def verify_2x2(U, V, W, n_test)"}, {"kind": "method", "line": 89, "name": "strassen_expand", "signature": "def strassen_expand(A, B, U, V, W)"}, {"kind": "method", "line": 126, "name": "verify_expansion", "signature": "def verify_expansion(U, V, W, sizes)"}, {"kind": "method", "line": 152, "name": "compute_S_theta", "signature": "def compute_S_theta(model)"}, {"kind": "method", "line": 166, "name": "load_checkpoint", "signature": "def load_checkpoint(path)"}, {"kind": "method", "line": 183, "name": "verify_checkpoint", "signature": "def verify_checkpoint(checkpoint_path)"}, {"kind": "method", "line": 226, "name": "run_noise_stability_test", "signature": "def run_noise_stability_test(checkpoint_path, noise_levels)"}, {"kind": "method", "line": 249, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 27, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 34, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 44, "name": "get_discrete_coefficients", "signature": "def get_discrete_coefficients(self)"}]}, {"id": "experimetn2.py", "kind": "module", "label": "experimetn2.py", "language": "py", "sha256": "2ca93c9aed4d430d", "symbol_count": 58, "symbols": [{"doc": "Immutable canonical configuration for the Strassen bilinear model.", "kind": "class", "line": 53, "name": "StrassStrassenConfig", "signature": "class StrassStrassenConfig"}, {"doc": "Immutable training hyperparameters for crystallization runs.", "kind": "class", "line": 71, "name": "TrainingConfig", "signature": "class TrainingConfig"}, {"doc": "Top-level suite configuration orchestrator.", "kind": "class", "line": 84, "name": "SuiteConfig", "signature": "class SuiteConfig"}, {"doc": "Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.\nImplements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.\nU, V in R^{rank x input_dim}, W in R^{output_dim x rank}.", "kind": "class", "line": 95, "name": "StrassStrassenModel", "signature": "class StrassStrassenModel(Module)"}, {"doc": "Genuinely complex-valued bilinear model for Altland-Zirnbauer spectral testing.\nUses complex parameters and arithmetic to break time-reversal symmetry.", "kind": "class", "line": 135, "name": "ComplexStrassStrassenModel", "signature": "class ComplexStrassStrassenModel(Module)"}, {"doc": "Generates random 2x2 matrix pairs and their exact products.", "kind": "class", "line": 178, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"doc": "Handles serialization and metadata tracking of model checkpoints.", "kind": "class", "line": 190, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"doc": "Computes adjacent gap ratio 'r' of eigenvalues to determine spectral class.", "kind": "class", "line": 207, "name": "LevelSpacingRatioCalculator", "signature": "class LevelSpacingRatioCalculator"}, {"doc": "Computes the mathematically exact full-rank Hessian of the model loss.", "kind": "class", "line": 244, "name": "ExactHessianCalculator", "signature": "class ExactHessianCalculator"}, {"doc": "Computes the synthetic Planck constant from parameter and loss statistics.", "kind": "class", "line": 278, "name": "SyntheticPlanckCalculator", "signature": "class SyntheticPlanckCalculator"}, {"doc": "Measures representation density via Sparse Autoencoder (SAE) bottleneck analysis.", "kind": "class", "line": 328, "name": "SuperpositionMetricCalculator", "signature": "class SuperpositionMetricCalculator"}, {"doc": "Abstract interface defining the execution protocol for experiments.", "kind": "class", "line": 385, "name": "IExperiment", "signature": "class IExperiment(ABC)"}, {"doc": "Experiment 1: Ricci-MBL Duality.\nTracks geometric curvature (Hessian Ricci scalar) against adjacent gap ratio\nduring a long-duration optimization trajectory to capture crystallization.", "kind": "class", "line": 393, "name": "Experiment1RicciMBLDuality", "signature": "class Experiment1RicciMBLDuality(IExperiment)"}, {"doc": "Experiment 2: Altland-Zirnbauer Symmetry Dial.\nUses complex parameters and arithmetic to trigger a true symmetry crossover\nfrom real symmetric ensembles (GOE) to complex Hermitian (GUE).", "kind": "class", "line": 457, "name": "Experiment2AltlandZirnbauer", "signature": "class Experiment2AltlandZirnbauer(IExperiment)"}, {"doc": "Experiment 3: Conformal Isomorphism.\nConducts mathematical stress-testing of fractional-linear Möbius\ntransformations vs scale-conformal transformations to map the physical\nlimits of learned tensor models.", "kind": "class", "line": 520, "name": "Experiment3ConformalIsomorphism", "signature": "class Experiment3ConformalIsomorphism(IExperiment)"}, {"doc": "Experiment 4: Compression Frontier.\nTests the thermodynamic bound between parameter uncertainty (hbar) and\nrepresentation superposition (psi) under varied weight decay levels.", "kind": "class", "line": 570, "name": "Experiment4CompressionFrontier", "signature": "class Experiment4CompressionFrontier(IExperiment)"}, {"doc": "Experiment 5: Holographic Pruning.\nDistinguishes volumetric representation structures from structured boundary-state\nmechanisms via element-wise versus slot-wise pruning.", "kind": "class", "line": 624, "name": "Experiment5HolographicPruning", "signature": "class Experiment5HolographicPruning(IExperiment)"}, {"doc": "Orchestrates and structures the execution of the five experiments.", "kind": "class", "line": 699, "name": "UnifiedSuite", "signature": "class UnifiedSuite"}, {"kind": "method", "line": 745, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 63, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 101, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 115, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 125, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 128, "name": "slot_importance", "signature": "def slot_importance(self)"}, {"kind": "method", "line": 140, "name": "__init__", "signature": "def __init__(self, config, gamma)"}, {"kind": "method", "line": 153, "name": "get_complex_tensors", "signature": "def get_complex_tensors(self)"}, {"kind": "method", "line": 160, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 180, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 183, "name": "generate_batch", "signature": "def generate_batch(self, batch_size)"}, {"kind": "method", "line": 192, "name": "save", "signature": "def save(self, model, epoch, metrics, path)"}, {"kind": "method", "line": 209, "name": "__init__", "signature": "def __init__(self, tolerance)"}, {"kind": "method", "line": 212, "name": "calculate_r_ratio", "signature": "def calculate_r_ratio(self, eigenvalues)"}, {"kind": "method", "line": 246, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 249, "name": "compute_hessian", "signature": "def compute_hessian(self, model, A, B, C_true)"}, {"kind": "method", "line": 280, "name": "__init__", "signature": "def __init__(self, noise_floor)"}, {"kind": "method", "line": 283, "name": "calculate", "signature": "def calculate(self, model, current_loss)"}, {"kind": "method", "line": 330, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 333, "name": "calculate", "signature": "def calculate(self, model, datagen)"}, {"kind": "method", "line": 388, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 390, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 399, "name": "__init__", "signature": "def __init__(self, suite_config, datagen)"}, {"kind": "method", "line": 405, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 408, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 463, "name": "__init__", "signature": "def __init__(self, suite_config, datagen)"}, {"kind": "method", "line": 468, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 471, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 527, "name": "__init__", "signature": "def __init__(self, suite_config, datagen)"}, {"kind": "method", "line": 531, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 534, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 576, "name": "__init__", "signature": "def __init__(self, suite_config, datagen)"}, {"kind": "method", "line": 582, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 585, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 630, "name": "__init__", "signature": "def __init__(self, suite_config, datagen)"}, {"kind": "method", "line": 634, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 637, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 701, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 712, "name": "execute_all", "signature": "def execute_all(self)"}, {"kind": "method", "line": 254, "name": "loss_fn", "signature": "def loss_fn(flat_param_tensor)"}]}, {"id": "fermi.py", "kind": "module", "label": "fermi.py", "language": "py", "sha256": "8303e569ef7c9fb3", "symbol_count": 68, "symbols": [{"kind": "class", "line": 19, "name": "FermiConfig", "signature": "class FermiConfig"}, {"kind": "class", "line": 49, "name": "IModel", "signature": "class IModel(Protocol)"}, {"kind": "class", "line": 54, "name": "IBlochWaveConstructor", "signature": "class IBlochWaveConstructor(Protocol)"}, {"kind": "class", "line": 59, "name": "IBandStructureCalculator", "signature": "class IBandStructureCalculator(Protocol)"}, {"kind": "class", "line": 64, "name": "IFermiLevelCalculator", "signature": "class IFermiLevelCalculator(Protocol)"}, {"kind": "class", "line": 69, "name": "IDensityOfStatesCalculator", "signature": "class IDensityOfStatesCalculator(Protocol)"}, {"kind": "class", "line": 74, "name": "IElectronicPropertiesCalculator", "signature": "class IElectronicPropertiesCalculator(Protocol)"}, {"kind": "class", "line": 79, "name": "IMetalInsulatorClassifier", "signature": "class IMetalInsulatorClassifier(Protocol)"}, {"kind": "class", "line": 83, "name": "BilinearModel", "signature": "class BilinearModel(Module)"}, {"kind": "class", "line": 111, "name": "BlochWaveConstructor", "signature": "class BlochWaveConstructor"}, {"kind": "class", "line": 135, "name": "BandStructureCalculator", "signature": "class BandStructureCalculator"}, {"kind": "class", "line": 215, "name": "FermiLevelCalculator", "signature": "class FermiLevelCalculator"}, {"kind": "class", "line": 285, "name": "DensityOfStatesCalculator", "signature": "class DensityOfStatesCalculator"}, {"kind": "class", "line": 306, "name": "ElectronicPropertiesCalculator", "signature": "class ElectronicPropertiesCalculator"}, {"kind": "class", "line": 368, "name": "MetalInsulatorClassifier", "signature": "class MetalInsulatorClassifier"}, {"kind": "class", "line": 408, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"kind": "class", "line": 458, "name": "FermiLevelAnalyzer", "signature": "class FermiLevelAnalyzer"}, {"kind": "class", "line": 608, "name": "FermiPipeline", "signature": "class FermiPipeline"}, {"kind": "method", "line": 766, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 50, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 55, "name": "construct", "signature": "def construct(self, weights, k)"}, {"kind": "method", "line": 60, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 65, "name": "calculate", "signature": "def calculate(self, eigenvalues, num_electrons)"}, {"kind": "method", "line": 70, "name": "calculate", "signature": "def calculate(self, eigenvalues, energies)"}, {"kind": "method", "line": 75, "name": "calculate", "signature": "def calculate(self, eigenvalues, eigenvectors, fermi_level)"}, {"kind": "method", "line": 80, "name": "classify", "signature": "def classify(self, band_gap, dos_at_fermi)"}, {"kind": "method", "line": 84, "name": "__init__", "signature": "def __init__(self, hidden_dim, matrix_size)"}, {"kind": "method", "line": 95, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 100, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 103, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 112, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 115, "name": "construct", "signature": "def construct(self, weights, k)"}, {"kind": "method", "line": 136, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 140, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 171, "name": "_calculate_band_gap", "signature": "def _calculate_band_gap(self, band_structure)"}, {"kind": "method", "line": 188, "name": "_calculate_effective_masses", "signature": "def _calculate_effective_masses(self, k_points, band_structure, valence_idx, conduction_idx)"}, {"kind": "method", "line": 208, "name": "_is_direct_gap", "signature": "def _is_direct_gap(self, band_structure, valence_idx, conduction_idx)"}, {"kind": "method", "line": 216, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 219, "name": "calculate", "signature": "def calculate(self, eigenvalues, num_electrons)"}, {"kind": "method", "line": 240, "name": "_calculate_chemical_potential", "signature": "def _calculate_chemical_potential(self, eigenvalues, num_electrons)"}, {"kind": "method", "line": 253, "name": "_find_chemical_potential_iterative", "signature": "def _find_chemical_potential_iterative(self, eigenvalues, num_electrons, temperature, max_iter)"}, {"kind": "method", "line": 273, "name": "_fermi_dirac", "signature": "def _fermi_dirac(self, energy, mu, temperature)"}, {"kind": "method", "line": 286, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 289, "name": "calculate", "signature": "def calculate(self, eigenvalues, energies)"}, {"kind": "method", "line": 302, "name": "_gaussian", "signature": "def _gaussian(self, x, mu, sigma)"}, {"kind": "method", "line": 307, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 310, "name": "calculate", "signature": "def calculate(self, eigenvalues, eigenvectors, fermi_level)"}, {"kind": "method", "line": 337, "name": "_calculate_kinetic_energy", "signature": "def _calculate_kinetic_energy(self, occupied_states)"}, {"kind": "method", "line": 347, "name": "_calculate_electronic_pressure", "signature": "def _calculate_electronic_pressure(self, eigenvalues, fermi_level)"}, {"kind": "method", "line": 357, "name": "_calculate_compressibility", "signature": "def _calculate_compressibility(self, eigenvalues, fermi_level)"}, {"kind": "method", "line": 369, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 372, "name": "classify", "signature": "def classify(self, band_gap, dos_at_fermi)"}, {"kind": "method", "line": 386, "name": "classify_transport", "signature": "def classify_transport(self, effective_masses, band_gap)"}, {"kind": "method", "line": 409, "name": "migrate", "signature": "def migrate(self, raw_data, device)"}, {"kind": "method", "line": 419, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict, device)"}, {"kind": "method", "line": 428, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(self, state_dict, device)"}, {"kind": "method", "line": 447, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(self, state_dict)"}, {"kind": "method", "line": 454, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(self, state_dict)"}, {"kind": "method", "line": 459, "name": "__init__", "signature": "def __init__(self, checkpoint_path, config)"}, {"kind": "method", "line": 472, "name": "_load_checkpoint", "signature": "def _load_checkpoint(self)"}, {"kind": "method", "line": 495, "name": "analyze", "signature": "def analyze(self)"}, {"kind": "method", "line": 550, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 609, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 612, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"kind": "method", "line": 626, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 653, "name": "generate_summary", "signature": "def generate_summary(self, all_results, output_dir)"}, {"kind": "method", "line": 686, "name": "_generate_text_report", "signature": "def _generate_text_report(self, summary, output_dir)"}, {"doc": "Generate comparison plots of band structures across checkpoints.", "kind": "method", "line": 724, "name": "plot_band_structures", "signature": "def plot_band_structures(self, all_results, output_dir)"}]}, {"id": "full_seed_prospector.py", "kind": "module", "label": "full_seed_prospector.py", "language": "py", "sha256": "35535dd80d90a50d", "symbol_count": 116, "symbols": [{"kind": "class", "line": 46, "name": "ExecutionMode", "signature": "class ExecutionMode(Enum)"}, {"doc": "Immutable unified configuration for all execution modes.", "kind": "class", "line": 52, "name": "UnifiedConfig", "signature": "class UnifiedConfig"}, {"doc": "Interface for metric calculation strategies.", "kind": "class", "line": 162, "name": "IMetricCalculator", "signature": "class IMetricCalculator(ABC)"}, {"doc": "Interface for loss function components.", "kind": "class", "line": 170, "name": "ILossComponent", "signature": "class ILossComponent(ABC)"}, {"doc": "Interface for checkpoint management.", "kind": "class", "line": 179, "name": "ICheckpointManager", "signature": "class ICheckpointManager(ABC)"}, {"doc": "Interface for training phase execution.", "kind": "class", "line": 195, "name": "ITrainingPhase", "signature": "class ITrainingPhase(ABC)"}, {"doc": "Spectral operator for 2x2 matrix multiplication with Strassen structure.", "kind": "class", "line": 203, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"doc": "Calculate delta (discretization margin) metric.", "kind": "class", "line": 313, "name": "DeltaCalculator", "signature": "class DeltaCalculator(IMetricCalculator)"}, {"doc": "Calculate matrix multiplication accuracy.", "kind": "class", "line": 333, "name": "AccuracyCalculator", "signature": "class AccuracyCalculator(IMetricCalculator)"}, {"doc": "Calculate kappa (gradient covariance condition number).", "kind": "class", "line": 371, "name": "KappaCalculator", "signature": "class KappaCalculator"}, {"doc": "Calculate Perelman W-entropy with adaptive tau.", "kind": "class", "line": 455, "name": "PerelmanEntropyCalculator", "signature": "class PerelmanEntropyCalculator(IMetricCalculator)"}, {"doc": "Calculate sparsity metrics.", "kind": "class", "line": 553, "name": "SparsityCalculator", "signature": "class SparsityCalculator(IMetricCalculator)"}, {"doc": "Calculate gradient statistics.", "kind": "class", "line": 573, "name": "GradientMetricsCalculator", "signature": "class GradientMetricsCalculator"}, {"doc": "Measure structural stability under progressive pruning.", "kind": "class", "line": 599, "name": "ResilienceSpectrometer", "signature": "class ResilienceSpectrometer"}, {"doc": "Aggregate all metrics including LC, SP, kappa, delta, h_bar_eff, T_eff.", "kind": "class", "line": 694, "name": "ComprehensiveMetricsAggregator", "signature": "class ComprehensiveMetricsAggregator"}, {"doc": "Adaptive quantization loss pushing towards {-1, 0, 1}.", "kind": "class", "line": 802, "name": "AdaptiveQuantizationLoss", "signature": "class AdaptiveQuantizationLoss(ILossComponent)"}, {"doc": "Ricci scalar curvature penalty.", "kind": "class", "line": 854, "name": "RicciCurvaturePenalty", "signature": "class RicciCurvaturePenalty(ILossComponent)"}, {"doc": "Aggregate geometric loss components.", "kind": "class", "line": 868, "name": "GeometricLossAggregator", "signature": "class GeometricLossAggregator(ILossComponent)"}, {"doc": "Manage model checkpointing.", "kind": "class", "line": 884, "name": "CheckpointManager", "signature": "class CheckpointManager(ICheckpointManager)"}, {"doc": "Generate random matrix data for training.", "kind": "class", "line": 930, "name": "MatrixDataGenerator", "signature": "class MatrixDataGenerator"}, {"doc": "Schedule batch size with cosine annealing.", "kind": "class", "line": 947, "name": "DynamicBatchSizeScheduler", "signature": "class DynamicBatchSizeScheduler"}, {"doc": "Detect glass state (non-crystallizing seeds).", "kind": "class", "line": 969, "name": "GlassDetector", "signature": "class GlassDetector"}, {"doc": "Fast prospecting phase to identify crystal seeds.", "kind": "class", "line": 1025, "name": "ProspectorPhase", "signature": "class ProspectorPhase(ITrainingPhase)"}, {"doc": "Long training phase with full thermodynamic metrics.", "kind": "class", "line": 1152, "name": "LongTrainingPhase", "signature": "class LongTrainingPhase(ITrainingPhase)"}, {"doc": "Progressive sparsification guided by slot importance.", "kind": "class", "line": 1335, "name": "ProgressiveSparsificationPhase", "signature": "class ProgressiveSparsificationPhase(ITrainingPhase)"}, {"doc": "Discretize coefficients to {-1, 0, 1}.", "kind": "class", "line": 1442, "name": "CoefficientDiscretizationPhase", "signature": "class CoefficientDiscretizationPhase(ITrainingPhase)"}, {"doc": "Verify Strassen algorithm correctness.", "kind": "class", "line": 1532, "name": "StrassenVerifier", "signature": "class StrassenVerifier"}, {"doc": "Provide canonical Strassen algorithm coefficients.", "kind": "class", "line": 1588, "name": "CanonicalStrassenProvider", "signature": "class CanonicalStrassenProvider"}, {"doc": "Prospect seeds for crystallization candidates.", "kind": "class", "line": 1623, "name": "SeedProspector", "signature": "class SeedProspector"}, {"doc": "Long training pipeline with all phases.", "kind": "class", "line": 1742, "name": "LongTrainingPipeline", "signature": "class LongTrainingPipeline"}, {"doc": "Calculate Local Complexity as defined in the paper.", "kind": "class", "line": 1849, "name": "LocalComplexityCalculator", "signature": "class LocalComplexityCalculator(IMetricCalculator)"}, {"doc": "Calculate Superposition metrics using sparse autoencoder.", "kind": "class", "line": 1915, "name": "SuperpositionCalculator", "signature": "class SuperpositionCalculator(IMetricCalculator)"}, {"doc": "Calculate thermodynamic metrics: h_bar_eff and T_eff.", "kind": "class", "line": 1964, "name": "ThermodynamicMetricsCalculator", "signature": "class ThermodynamicMetricsCalculator(IMetricCalculator)"}, {"doc": "Main entry point.", "kind": "method", "line": 2022, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 166, "name": "calculate", "signature": "def calculate(self)"}, {"kind": "method", "line": 174, "name": "compute", "signature": "def compute(self, model, loss_mse, epoch)"}, {"kind": "method", "line": 183, "name": "save", "signature": "def save(self, state, path)"}, {"kind": "method", "line": 187, "name": "load", "signature": "def load(self, path)"}, {"kind": "method", "line": 191, "name": "should_checkpoint", "signature": "def should_checkpoint(self)"}, {"kind": "method", "line": 199, "name": "execute", "signature": "def execute(self, model)"}, {"kind": "method", "line": 206, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Initialize with bias towards canonical Strassen structure.", "kind": "method", "line": 220, "name": "_initialize_strassen_structure", "signature": "def _initialize_strassen_structure(self)"}, {"doc": "Forward pass computing C = A @ B via low-rank factorization.", "kind": "method", "line": 276, "name": "forward", "signature": "def forward(self, A, B)"}, {"doc": "Calculate importance of each rank slot.", "kind": "method", "line": 291, "name": "slot_importance", "signature": "def slot_importance(self)"}, {"doc": "Count active slots above threshold.", "kind": "method", "line": 298, "name": "count_active", "signature": "def count_active(self, threshold)"}, {"doc": "Get flattened parameter vector.", "kind": "method", "line": 304, "name": "get_flat_parameters", "signature": "def get_flat_parameters(self)"}, {"doc": "Get total parameter count.", "kind": "method", "line": 308, "name": "get_parameter_count", "signature": "def get_parameter_count(self)"}, {"kind": "method", "line": 316, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate delta: mean squared distance to {-1, 0, 1}.", "kind": "method", "line": 319, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 336, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate accuracy percentage.", "kind": "method", "line": 339, "name": "calculate", "signature": "def calculate(self, model, C_pred, C_true, n_test)"}, {"kind": "method", "line": 374, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Accumulate current gradient vector.", "kind": "method", "line": 379, "name": "accumulate_gradient", "signature": "def accumulate_gradient(self, model)"}, {"doc": "Calculate condition number of gradient covariance matrix.", "kind": "method", "line": 393, "name": "calculate_kappa", "signature": "def calculate_kappa(self)"}, {"doc": "Determine kappa trend direction.", "kind": "method", "line": 425, "name": "get_kappa_trend", "signature": "def get_kappa_trend(self)"}, {"doc": "Detect if system is in crystallization phase.", "kind": "method", "line": 439, "name": "is_crystallizing", "signature": "def is_crystallizing(self)"}, {"doc": "Reset calculator state.", "kind": "method", "line": 449, "name": "reset", "signature": "def reset(self)"}, {"kind": "method", "line": 458, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate W-entropy with adaptive tau coupling to GNS.", "kind": "method", "line": 465, "name": "calculate", "signature": "def calculate(self, model, loss, epoch, gradient_norm)"}, {"doc": "Calculate log(W) with numerical stability.", "kind": "method", "line": 522, "name": "_calculate_log_W", "signature": "def _calculate_log_W(self, tau, R, grad_f_sq, f, n_params)"}, {"doc": "Reset calculator state.", "kind": "method", "line": 546, "name": "reset", "signature": "def reset(self)"}, {"kind": "method", "line": 556, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate active slots and sparsity.", "kind": "method", "line": 559, "name": "calculate", "signature": "def calculate(self, model)"}, {"doc": "Calculate gradient norm statistics.", "kind": "method", "line": 577, "name": "calculate", "signature": "def calculate(model)"}, {"kind": "method", "line": 602, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Measure resilience via progressive magnitude pruning.", "kind": "method", "line": 605, "name": "measure", "signature": "def measure(self, model)"}, {"doc": "Prune parameters by magnitude threshold.", "kind": "method", "line": 684, "name": "_prune_by_magnitude", "signature": "def _prune_by_magnitude(self, model, sparsity)"}, {"kind": "method", "line": 697, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Compute all available metrics.", "kind": "method", "line": 711, "name": "compute_all", "signature": "def compute_all(self, model, C_pred, C_true, loss, epoch, force_kappa, force_lc, force_sp)"}, {"doc": "Accumulate gradient for kappa calculation.", "kind": "method", "line": 787, "name": "accumulate_gradient", "signature": "def accumulate_gradient(self, model)"}, {"doc": "Update current learning rate.", "kind": "method", "line": 791, "name": "update_lr", "signature": "def update_lr(self, lr)"}, {"doc": "Reset all calculators.", "kind": "method", "line": 795, "name": "reset", "signature": "def reset(self)"}, {"kind": "method", "line": 805, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Compute quantization loss with adaptive weighting.", "kind": "method", "line": 808, "name": "compute", "signature": "def compute(self, model, loss_mse, epoch, kappa)"}, {"doc": "Calculate adaptive weight based on epoch and kappa.", "kind": "method", "line": 835, "name": "_get_adaptive_weight", "signature": "def _get_adaptive_weight(self, epoch, kappa)"}, {"kind": "method", "line": 857, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Compute Ricci curvature penalty.", "kind": "method", "line": 860, "name": "compute", "signature": "def compute(self, model, loss_mse, epoch)"}, {"kind": "method", "line": 871, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Compute total geometric loss.", "kind": "method", "line": 876, "name": "compute", "signature": "def compute(self, model, loss_mse, epoch, kappa)"}, {"kind": "method", "line": 887, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Save checkpoint to disk.", "kind": "method", "line": 893, "name": "save", "signature": "def save(self, state, path)"}, {"doc": "Load checkpoint from disk.", "kind": "method", "line": 911, "name": "load", "signature": "def load(self, path)"}, {"doc": "Check if checkpoint interval has elapsed.", "kind": "method", "line": 919, "name": "should_checkpoint", "signature": "def should_checkpoint(self)"}, {"doc": "Get path to latest checkpoint if exists.", "kind": "method", "line": 924, "name": "get_latest_checkpoint_path", "signature": "def get_latest_checkpoint_path(self)"}, {"kind": "method", "line": 933, "name": "__init__", "signature": "def __init__(self, config, scale)"}, {"doc": "Generate batch of random matrices.", "kind": "method", "line": 938, "name": "generate_batch", "signature": "def generate_batch(self, n)"}, {"kind": "method", "line": 950, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Get batch size for current epoch.", "kind": "method", "line": 953, "name": "get_batch_size", "signature": "def get_batch_size(self, epoch)"}, {"kind": "method", "line": 972, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Determine if training should stop due to glass state.", "kind": "method", "line": 977, "name": "should_stop", "signature": "def should_stop(self, epoch, metrics)"}, {"kind": "method", "line": 1028, "name": "__init__", "signature": "def __init__(self, config, seed)"}, {"doc": "Execute prospecting phase with all metrics visible.", "kind": "method", "line": 1037, "name": "execute", "signature": "def execute(self, model)"}, {"kind": "method", "line": 1155, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Execute long training phase.", "kind": "method", "line": 1164, "name": "execute", "signature": "def execute(self, model)"}, {"kind": "method", "line": 1338, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Execute sparsification phase.", "kind": "method", "line": 1342, "name": "execute", "signature": "def execute(self, model)"}, {"doc": "Final refinement after pruning.", "kind": "method", "line": 1407, "name": "_final_refinement", "signature": "def _final_refinement(self, model, optimizer, slots_to_prune)"}, {"kind": "method", "line": 1445, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Execute discretization phase.", "kind": "method", "line": 1448, "name": "execute", "signature": "def execute(self, model)"}, {"kind": "method", "line": 1535, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Verify algorithm on random test matrices.", "kind": "method", "line": 1538, "name": "verify", "signature": "def verify(self, U, V, W, n_test)"}, {"doc": "Return canonical Strassen algorithm matrices.", "kind": "method", "line": 1592, "name": "get_canonical", "signature": "def get_canonical()"}, {"kind": "method", "line": 1626, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Prospect multiple seeds for crystals.", "kind": "method", "line": 1631, "name": "prospect", "signature": "def prospect(self, total_attempts, start_seed)"}, {"doc": "Set random seed for reproducibility.", "kind": "method", "line": 1733, "name": "_set_seed", "signature": "def _set_seed(self, seed)"}, {"kind": "method", "line": 1745, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Handle interrupt signal.", "kind": "method", "line": 1758, "name": "_signal_handler", "signature": "def _signal_handler(self, signum, frame)"}, {"doc": "Set random seed for reproducibility.", "kind": "method", "line": 1763, "name": "_set_seed", "signature": "def _set_seed(self, seed)"}, {"doc": "Run complete training pipeline.", "kind": "method", "line": 1771, "name": "run", "signature": "def run(self, resume_from, seed)"}, {"kind": "method", "line": 1852, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate LC as effective local dimensionality (paper definition).\nCompatible fraction approach → log(volume) can be negative.", "kind": "method", "line": 1855, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 1918, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Initialize sparse autoencoder if not already done.", "kind": "method", "line": 1923, "name": "_initialize_sae", "signature": "def _initialize_sae(self, input_dim, device)"}, {"doc": "Calculate superposition coefficient psi and effective features F.", "kind": "method", "line": 1930, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 1967, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate effective Planck constant and temperature.", "kind": "method", "line": 1970, "name": "calculate", "signature": "def calculate(self, model, gradient_covariance)"}]}, {"id": "grain.py", "kind": "module", "label": "grain.py", "language": "py", "sha256": "a2a65bcd3923bbd7", "symbol_count": 77, "symbols": [{"kind": "class", "line": 25, "name": "StrassenConfig", "signature": "class StrassenConfig"}, {"kind": "class", "line": 65, "name": "IModel", "signature": "class IModel(Protocol)"}, {"kind": "class", "line": 71, "name": "IGrainBoundaryDetector", "signature": "class IGrainBoundaryDetector(Protocol)"}, {"kind": "class", "line": 76, "name": "ILayerAnalyzer", "signature": "class ILayerAnalyzer(Protocol)"}, {"kind": "class", "line": 81, "name": "IDislocationCalculator", "signature": "class IDislocationCalculator(Protocol)"}, {"kind": "class", "line": 86, "name": "IDomainFragmentationAnalyzer", "signature": "class IDomainFragmentationAnalyzer(Protocol)"}, {"kind": "class", "line": 91, "name": "ICheckpointManager", "signature": "class ICheckpointManager(Protocol)"}, {"kind": "class", "line": 97, "name": "ITrainingMonitor", "signature": "class ITrainingMonitor(Protocol)"}, {"kind": "class", "line": 102, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"kind": "class", "line": 130, "name": "LayerAnalyzer", "signature": "class LayerAnalyzer"}, {"kind": "class", "line": 153, "name": "GrainBoundaryDetector", "signature": "class GrainBoundaryDetector"}, {"kind": "class", "line": 243, "name": "DomainFragmentationAnalyzer", "signature": "class DomainFragmentationAnalyzer"}, {"kind": "class", "line": 309, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"kind": "class", "line": 340, "name": "TrainingMetricsTracker", "signature": "class TrainingMetricsTracker"}, {"kind": "class", "line": 407, "name": "StrassenTrainer", "signature": "class StrassenTrainer"}, {"kind": "class", "line": 547, "name": "GrainBoundaryAnalyzer", "signature": "class GrainBoundaryAnalyzer"}, {"kind": "class", "line": 726, "name": "GrainBoundaryPipeline", "signature": "class GrainBoundaryPipeline"}, {"kind": "method", "line": 833, "name": "run_training", "signature": "def run_training(seed, config)"}, {"kind": "method", "line": 838, "name": "run_analysis", "signature": "def run_analysis(checkpoint_dir, output_dir, n_latest, config)"}, {"kind": "method", "line": 845, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 57, "name": "to_dict", "signature": "def to_dict(self)"}, {"kind": "method", "line": 66, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 67, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 72, "name": "detect", "signature": "def detect(self, model, pruning_level)"}, {"kind": "method", "line": 77, "name": "analyze_layer", "signature": "def analyze_layer(self, weights, layer_name)"}, {"kind": "method", "line": 82, "name": "calculate", "signature": "def calculate(self, layer_deltas)"}, {"kind": "method", "line": 87, "name": "analyze", "signature": "def analyze(self, model, pruning_level)"}, {"kind": "method", "line": 92, "name": "save", "signature": "def save(self, model, epoch, metrics, path)"}, {"kind": "method", "line": 93, "name": "load", "signature": "def load(self, path)"}, {"kind": "method", "line": 98, "name": "update", "signature": "def update(self, epoch, metrics)"}, {"kind": "method", "line": 99, "name": "should_checkpoint", "signature": "def should_checkpoint(self)"}, {"kind": "method", "line": 103, "name": "__init__", "signature": "def __init__(self, hidden_dim, matrix_size)"}, {"kind": "method", "line": 114, "name": "_initialize_symmetric", "signature": "def _initialize_symmetric(self)"}, {"kind": "method", "line": 119, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 122, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 131, "name": "analyze_layer", "signature": "def analyze_layer(self, weights, layer_name)"}, {"kind": "method", "line": 154, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 158, "name": "detect", "signature": "def detect(self, model, pruning_level)"}, {"kind": "method", "line": 185, "name": "_prune_model", "signature": "def _prune_model(self, model, sparsity)"}, {"kind": "method", "line": 194, "name": "_calculate_dislocation", "signature": "def _calculate_dislocation(self, layer_deltas)"}, {"kind": "method", "line": 227, "name": "_analyze_fragmentation", "signature": "def _analyze_fragmentation(self, layer_analysis)"}, {"kind": "method", "line": 244, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 248, "name": "analyze", "signature": "def analyze(self, model, pruning_level)"}, {"kind": "method", "line": 269, "name": "_calculate_coordination_loss", "signature": "def _calculate_coordination_loss(self, layer_analysis)"}, {"kind": "method", "line": 282, "name": "_estimate_domain_count", "signature": "def _estimate_domain_count(self, layer_analysis)"}, {"kind": "method", "line": 297, "name": "_calculate_coherence_length", "signature": "def _calculate_coherence_length(self, layer_analysis)"}, {"kind": "method", "line": 310, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 314, "name": "save", "signature": "def save(self, model, epoch, metrics, path)"}, {"kind": "method", "line": 332, "name": "load", "signature": "def load(self, path)"}, {"kind": "method", "line": 335, "name": "should_save", "signature": "def should_save(self)"}, {"kind": "method", "line": 341, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 358, "name": "update", "signature": "def update(self, epoch, loss, accuracy, model, grain_result)"}, {"kind": "method", "line": 382, "name": "get_current_metrics", "signature": "def get_current_metrics(self)"}, {"kind": "method", "line": 389, "name": "get_training_bar_string", "signature": "def get_training_bar_string(self, epoch, total_epochs)"}, {"kind": "method", "line": 408, "name": "__init__", "signature": "def __init__(self, config, seed)"}, {"kind": "method", "line": 438, "name": "_setup_signal_handlers", "signature": "def _setup_signal_handlers(self)"}, {"kind": "method", "line": 442, "name": "_signal_handler", "signature": "def _signal_handler(self, signum, frame)"}, {"kind": "method", "line": 447, "name": "_generate_batch", "signature": "def _generate_batch(self)"}, {"kind": "method", "line": 468, "name": "_compute_accuracy", "signature": "def _compute_accuracy(self, pred, target)"}, {"kind": "method", "line": 473, "name": "_save_checkpoint", "signature": "def _save_checkpoint(self, interrupted)"}, {"kind": "method", "line": 486, "name": "train", "signature": "def train(self)"}, {"kind": "method", "line": 548, "name": "__init__", "signature": "def __init__(self, checkpoint_path, config)"}, {"kind": "method", "line": 557, "name": "_load_checkpoint", "signature": "def _load_checkpoint(self)"}, {"kind": "method", "line": 580, "name": "_migrate_checkpoint", "signature": "def _migrate_checkpoint(self, raw_data)"}, {"kind": "method", "line": 590, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict)"}, {"kind": "method", "line": 599, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(self, state_dict)"}, {"kind": "method", "line": 618, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(self, state_dict)"}, {"kind": "method", "line": 625, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(self, state_dict)"}, {"kind": "method", "line": 628, "name": "analyze", "signature": "def analyze(self)"}, {"kind": "method", "line": 658, "name": "_analyze_dislocation_evolution", "signature": "def _analyze_dislocation_evolution(self, grain_results)"}, {"kind": "method", "line": 681, "name": "_find_critical_pruning_level", "signature": "def _find_critical_pruning_level(self, grain_results)"}, {"kind": "method", "line": 687, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 727, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 730, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"kind": "method", "line": 744, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 771, "name": "generate_summary", "signature": "def generate_summary(self, all_results, output_dir)"}, {"kind": "method", "line": 797, "name": "_generate_text_report", "signature": "def _generate_text_report(self, summary, output_dir)"}]}, {"id": "gravity.py", "kind": "module", "label": "gravity.py", "language": "py", "sha256": "5c3cd78a11ac7288", "symbol_count": 88, "symbols": [{"kind": "class", "line": 24, "name": "ThermodynamicConfig", "signature": "class ThermodynamicConfig"}, {"kind": "class", "line": 59, "name": "IModel", "signature": "class IModel(Protocol)"}, {"kind": "class", "line": 65, "name": "IOrderParameterCalculator", "signature": "class IOrderParameterCalculator(Protocol)"}, {"kind": "class", "line": 70, "name": "IEntropyCalculator", "signature": "class IEntropyCalculator(Protocol)"}, {"kind": "class", "line": 75, "name": "ISpecificHeatCalculator", "signature": "class ISpecificHeatCalculator(Protocol)"}, {"kind": "class", "line": 80, "name": "IGravitationalConstantCalculator", "signature": "class IGravitationalConstantCalculator(Protocol)"}, {"kind": "class", "line": 85, "name": "ILandauerConstantCalculator", "signature": "class ILandauerConstantCalculator(Protocol)"}, {"kind": "class", "line": 90, "name": "IHeisenbergUncertaintyCalculator", "signature": "class IHeisenbergUncertaintyCalculator(Protocol)"}, {"kind": "class", "line": 95, "name": "ILocalComplexityCalculator", "signature": "class ILocalComplexityCalculator(Protocol)"}, {"kind": "class", "line": 100, "name": "IBasinStabilityCalculator", "signature": "class IBasinStabilityCalculator(Protocol)"}, {"kind": "class", "line": 105, "name": "IZeroShotTransferCalculator", "signature": "class IZeroShotTransferCalculator(Protocol)"}, {"kind": "class", "line": 110, "name": "IConditionNumberCalculator", "signature": "class IConditionNumberCalculator(Protocol)"}, {"kind": "class", "line": 114, "name": "BilinearModel", "signature": "class BilinearModel(Module)"}, {"kind": "class", "line": 142, "name": "OrderParameterCalculator", "signature": "class OrderParameterCalculator"}, {"kind": "class", "line": 154, "name": "ConfigurationEntropyCalculator", "signature": "class ConfigurationEntropyCalculator"}, {"kind": "class", "line": 171, "name": "SpecificHeatCalculator", "signature": "class SpecificHeatCalculator"}, {"kind": "class", "line": 183, "name": "GravitationalConstantCalculator", "signature": "class GravitationalConstantCalculator"}, {"kind": "class", "line": 239, "name": "LandauerConstantCalculator", "signature": "class LandauerConstantCalculator"}, {"kind": "class", "line": 271, "name": "HeisenbergUncertaintyCalculator", "signature": "class HeisenbergUncertaintyCalculator"}, {"kind": "class", "line": 310, "name": "LocalComplexityCalculator", "signature": "class LocalComplexityCalculator"}, {"kind": "class", "line": 327, "name": "BasinStabilityCalculator", "signature": "class BasinStabilityCalculator"}, {"kind": "class", "line": 365, "name": "ZeroShotTransferCalculator", "signature": "class ZeroShotTransferCalculator"}, {"kind": "class", "line": 398, "name": "ConditionNumberCalculator", "signature": "class ConditionNumberCalculator"}, {"kind": "class", "line": 454, "name": "PhaseTransitionDetector", "signature": "class PhaseTransitionDetector"}, {"kind": "class", "line": 505, "name": "ThermodynamicAnalyzer", "signature": "class ThermodynamicAnalyzer"}, {"kind": "class", "line": 837, "name": "ThermodynamicPipeline", "signature": "class ThermodynamicPipeline"}, {"kind": "method", "line": 1114, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 60, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 61, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 66, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 71, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 76, "name": "calculate", "signature": "def calculate(self, loss_history)"}, {"kind": "method", "line": 81, "name": "calculate", "signature": "def calculate(self, model, gradient_history)"}, {"kind": "method", "line": 86, "name": "calculate", "signature": "def calculate(self, entropy_change, energy_dissipated)"}, {"kind": "method", "line": 91, "name": "calculate", "signature": "def calculate(self, model, temperature)"}, {"kind": "method", "line": 96, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 101, "name": "calculate", "signature": "def calculate(self, model, test_data)"}, {"kind": "method", "line": 106, "name": "calculate", "signature": "def calculate(self, model, target_size)"}, {"kind": "method", "line": 111, "name": "calculate", "signature": "def calculate(self, gradient_covariance)"}, {"kind": "method", "line": 115, "name": "__init__", "signature": "def __init__(self, hidden_dim, matrix_size)"}, {"kind": "method", "line": 126, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 131, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 134, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 143, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 146, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 155, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 158, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 172, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 175, "name": "calculate", "signature": "def calculate(self, loss_history)"}, {"kind": "method", "line": 184, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 187, "name": "calculate", "signature": "def calculate(self, model, gradient_history, loss_history, static_gradient)"}, {"kind": "method", "line": 240, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 243, "name": "calculate", "signature": "def calculate(self, entropy_change, energy_dissipated, has_transition, transition_window)"}, {"kind": "method", "line": 272, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 275, "name": "calculate", "signature": "def calculate(self, model, temperature, static_gradient)"}, {"kind": "method", "line": 311, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 314, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 328, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 331, "name": "calculate", "signature": "def calculate(self, model, test_data)"}, {"kind": "method", "line": 355, "name": "_prune_model", "signature": "def _prune_model(self, model, sparsity)"}, {"kind": "method", "line": 366, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 369, "name": "calculate", "signature": "def calculate(self, model, target_size)"}, {"kind": "method", "line": 391, "name": "_kronecker_recursive", "signature": "def _kronecker_recursive(self, matrix, power)"}, {"kind": "method", "line": 399, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 403, "name": "calculate", "signature": "def calculate(self, gradient_history, static_gradient)"}, {"kind": "method", "line": 455, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 458, "name": "detect", "signature": "def detect(self, loss_history, entropy_history)"}, {"kind": "method", "line": 506, "name": "__init__", "signature": "def __init__(self, checkpoint_path, config)"}, {"kind": "method", "line": 525, "name": "_load_checkpoint", "signature": "def _load_checkpoint(self)"}, {"kind": "method", "line": 548, "name": "_migrate_checkpoint", "signature": "def _migrate_checkpoint(self, raw_data)"}, {"kind": "method", "line": 560, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict)"}, {"kind": "method", "line": 569, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(self, state_dict)"}, {"kind": "method", "line": 588, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(self, state_dict)"}, {"kind": "method", "line": 595, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(self, state_dict)"}, {"kind": "method", "line": 598, "name": "_compute_static_gradient", "signature": "def _compute_static_gradient(self)"}, {"kind": "method", "line": 619, "name": "_generate_test_data", "signature": "def _generate_test_data(self)"}, {"kind": "method", "line": 630, "name": "analyze", "signature": "def analyze(self)"}, {"kind": "method", "line": 733, "name": "_determine_failure_mode", "signature": "def _determine_failure_mode(self, delta, basin, kappa, transition)"}, {"kind": "method", "line": 744, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 838, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 841, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"kind": "method", "line": 855, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 882, "name": "generate_summary", "signature": "def generate_summary(self, all_results, output_dir)"}, {"kind": "method", "line": 904, "name": "_compute_emergent_constants", "signature": "def _compute_emergent_constants(self, results)"}, {"kind": "method", "line": 959, "name": "_verify_universal_laws", "signature": "def _verify_universal_laws(self, results)"}, {"kind": "method", "line": 993, "name": "_compute_kappa_correlation", "signature": "def _compute_kappa_correlation(self, results)"}, {"kind": "method", "line": 1023, "name": "_generate_text_report", "signature": "def _generate_text_report(self, summary, output_dir)"}, {"kind": "method", "line": 908, "name": "extract_values", "signature": "def extract_values(data_list, key_path)"}]}, {"id": "grigori_perelmans_ricci_flow.py", "kind": "module", "label": "grigori_perelmans_ricci_flow.py", "language": "py", "sha256": "50f9c450756c0bec", "symbol_count": 46, "symbols": [{"doc": "Immutable configuration for Ricci Flow analysis.", "kind": "class", "line": 34, "name": "RicciConfig", "signature": "class RicciConfig"}, {"kind": "method", "line": 70, "name": "set_random_seed", "signature": "def set_random_seed(seed)"}, {"doc": "Bilinear model f(A,B) = W((U*A) ⊙ (V*B)).", "kind": "class", "line": 82, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"kind": "class", "line": 128, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"kind": "class", "line": 145, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator(ABC)"}, {"kind": "class", "line": 151, "name": "CustomFormatMigrator", "signature": "class CustomFormatMigrator(CheckpointMigrator)"}, {"kind": "class", "line": 166, "name": "StandardFormatMigrator", "signature": "class StandardFormatMigrator(CheckpointMigrator)"}, {"kind": "class", "line": 172, "name": "CheckpointMigrationManager", "signature": "class CheckpointMigrationManager"}, {"doc": "Calculates Ricci curvature metrics using Hessian as Metric Tensor proxy.\nIn Perelman's flow, dg/dt = -2Ric. \nHere we analyze instantaneous state of Metric (Hessian).", "kind": "class", "line": 193, "name": "RicciFlowAnalyzer", "signature": "class RicciFlowAnalyzer"}, {"doc": "Identifies 'necks' (singularities) in the geometry and proposes 'surgery' (pruning).\nA 'neck' is a parameter direction with extreme curvature (Hessian eigenvalue).", "kind": "class", "line": 310, "name": "SingularityEngine", "signature": "class SingularityEngine"}, {"doc": "Estimates effective Planck constant from Spectral Geometry.\nUses the Hessian eigenvalues to define an energy spectrum.", "kind": "class", "line": 366, "name": "GeometricPlanckCalculator", "signature": "class GeometricPlanckCalculator"}, {"doc": "Complete analysis pipeline for Ricci Flow and Planck Estimation.\nOrchestrates Hessian computation, Curvature Analysis, and Physics metrics.", "kind": "class", "line": 430, "name": "RicciFlowAnalyzerPipeline", "signature": "class RicciFlowAnalyzerPipeline"}, {"kind": "method", "line": 593, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 86, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 94, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 99, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 102, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"doc": "Return all parameters as a single flattened vector.", "kind": "method", "line": 109, "name": "get_flat_params", "signature": "def get_flat_params(self)"}, {"doc": "Set model parameters from a flattened vector.", "kind": "method", "line": 114, "name": "set_flat_params", "signature": "def set_flat_params(self, flat_params)"}, {"kind": "method", "line": 130, "name": "generate_batch", "signature": "def generate_batch(batch_size, config)"}, {"kind": "method", "line": 147, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 149, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 152, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 154, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 167, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 169, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 173, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 176, "name": "migrate_checkpoint", "signature": "def migrate_checkpoint(self, path, device)"}, {"kind": "method", "line": 199, "name": "__init__", "signature": "def __init__(self, model, config)"}, {"doc": "Computes exact Hessian of loss w.r.t parameters.\nH = d^2L / dtheta^2", "kind": "method", "line": 203, "name": "compute_hessian", "signature": "def compute_hessian(self, input_a, input_b, target_c)"}, {"doc": "Wrapper to compute loss from flat param vector.", "kind": "method", "line": 223, "name": "_loss_wrapper", "signature": "def _loss_wrapper(self, flat_params, original_params, a, b, c)"}, {"doc": "Approximation: Diagonal of Hessian (Gauss-Newton).", "kind": "method", "line": 240, "name": "_compute_diagonal_hessian", "signature": "def _compute_diagonal_hessian(self, a, b, c)"}, {"doc": "Analyze Hessian spectrum to derive Ricci Scalar and Topological invariants.", "kind": "method", "line": 247, "name": "analyze_curvature", "signature": "def analyze_curvature(self, hessian)"}, {"doc": "Trace of Heat Kernel: Z(t) = Sum( exp(-lambda_i * t) ).\nRelates to Partition Function in Quantum Mechanics.", "kind": "method", "line": 285, "name": "compute_heat_kernel_trace", "signature": "def compute_heat_kernel_trace(self, eigenvalues, t)"}, {"doc": "von Neumann Entropy / Spectral Entropy.\nS = - Sum( p_i * log(p_i) ) where p_i are normalized eigenvalue weights.", "kind": "method", "line": 296, "name": "compute_topological_entropy", "signature": "def compute_topological_entropy(self, eigenvalues)"}, {"kind": "method", "line": 316, "name": "__init__", "signature": "def __init__(self, model, eigenvalues, config)"}, {"doc": "Identify if the system is in a 'bottleneck' state.", "kind": "method", "line": 325, "name": "detect_necks", "signature": "def detect_necks(self, curvature_analysis)"}, {"doc": "Propose parameters to 'cut' (prune) based on curvature heuristics.\nIn Strassen, the 'bias' slot (8th) often carries the 'noise' or singular connection.", "kind": "method", "line": 336, "name": "propose_surgery", "signature": "def propose_surgery(self)"}, {"kind": "method", "line": 371, "name": "__init__", "signature": "def __init__(self, eigenvalues, ricci_scalar, config)"}, {"kind": "method", "line": 376, "name": "calculate", "signature": "def calculate(self)"}, {"kind": "method", "line": 412, "name": "_get_spectral_gap", "signature": "def _get_spectral_gap(self)"}, {"kind": "method", "line": 420, "name": "_compute_spectral_entropy", "signature": "def _compute_spectral_entropy(self)"}, {"kind": "method", "line": 435, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Perform complete analysis of a single checkpoint.", "kind": "method", "line": 439, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(self, checkpoint_path, device)"}, {"kind": "method", "line": 527, "name": "analyze_directory", "signature": "def analyze_directory(self, directory, device, pattern)"}, {"kind": "method", "line": 560, "name": "_print_summary", "signature": "def _print_summary(self, report)"}]}, {"id": "hawking_radiation.py", "kind": "module", "label": "hawking_radiation.py", "language": "py", "sha256": "54430ac85ceb2c74", "symbol_count": 71, "symbols": [{"doc": "Custom unpickler that handles unknown classes by creating dummy objects.\n\nThis solves the \"Can't get attribute 'UnifiedConfig'\" error by providing\nfallback objects for any class that can't be found.", "kind": "class", "line": 36, "name": "CustomUnpickler", "signature": "class CustomUnpickler(Unpickler)"}, {"doc": "Load checkpoint with robust handling of custom classes.\n\nTries multiple loading strategies in order:\n1. Standard torch.load\n2. Custom unpickler\n3. Weights_only mode with manual extraction", "kind": "method", "line": 93, "name": "load_checkpoint_robust", "signature": "def load_checkpoint_robust(path, device)"}, {"doc": "Immutable configuration for Hawking radiation analysis.", "kind": "class", "line": 138, "name": "HawkingConfiguration", "signature": "class HawkingConfiguration"}, {"kind": "class", "line": 188, "name": "IModel", "signature": "class IModel(Protocol)"}, {"doc": "Bilinear model for Strassen matrix multiplication.", "kind": "class", "line": 197, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"doc": "Enhanced checkpoint migrator that handles:\n- Direct U, V, W tensors\n- U_coefs format\n- Standard .weight format\n- Nested structures with config\n- Encoder format\n- State dicts within state dicts", "kind": "class", "line": 234, "name": "RobustCheckpointMigrator", "signature": "class RobustCheckpointMigrator"}, {"doc": "Extracts metadata from various checkpoint formats.", "kind": "class", "line": 514, "name": "MetadataExtractor", "signature": "class MetadataExtractor"}, {"doc": "Calculates effective gravitational constant G_alg.", "kind": "class", "line": 609, "name": "GravitationalConstantCalculator", "signature": "class GravitationalConstantCalculator"}, {"doc": "Calculates effective Planck constant h_bar_eff.", "kind": "class", "line": 676, "name": "PlanckConstantCalculator", "signature": "class PlanckConstantCalculator"}, {"doc": "Calculates effective Boltzmann constant k_B_eff.", "kind": "class", "line": 768, "name": "BoltzmannConstantCalculator", "signature": "class BoltzmannConstantCalculator"}, {"doc": "Calculates effective speed of light c_eff.", "kind": "class", "line": 823, "name": "SpeedOfLightCalculator", "signature": "class SpeedOfLightCalculator"}, {"doc": "Calculates effective mass M_eff.", "kind": "class", "line": 883, "name": "InformationalMassCalculator", "signature": "class InformationalMassCalculator"}, {"doc": "Calculates effective area A_eff.", "kind": "class", "line": 939, "name": "HorizonAreaCalculator", "signature": "class HorizonAreaCalculator"}, {"doc": "Calculates Hawking radiation metrics.", "kind": "class", "line": 991, "name": "HawkingRadiationCalculator", "signature": "class HawkingRadiationCalculator"}, {"doc": "Robust analyzer that handles exotic checkpoint formats.", "kind": "class", "line": 1166, "name": "RobustHawkingAnalyzer", "signature": "class RobustHawkingAnalyzer"}, {"kind": "method", "line": 1406, "name": "main", "signature": "def main()"}, {"doc": "Override find_class to handle missing classes gracefully.", "kind": "method", "line": 44, "name": "find_class", "signature": "def find_class(self, module, name)"}, {"doc": "Create a dummy class that can hold attributes.", "kind": "method", "line": 62, "name": "_create_dummy_class", "signature": "def _create_dummy_class(self, name)"}, {"kind": "method", "line": 172, "name": "get_effective_input_dim", "signature": "def get_effective_input_dim(self)"}, {"kind": "method", "line": 175, "name": "get_total_parameters", "signature": "def get_total_parameters(self)"}, {"kind": "method", "line": 189, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 190, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 200, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 211, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 216, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 219, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 226, "name": "get_flat_parameters", "signature": "def get_flat_parameters(self)"}, {"doc": "Main migration entry point with multiple strategies.", "kind": "method", "line": 245, "name": "migrate", "signature": "def migrate(self, raw_data, device)"}, {"doc": "Try standard state dict extraction methods.", "kind": "method", "line": 269, "name": "_try_extract_state_dict", "signature": "def _try_extract_state_dict(self, data, device)"}, {"doc": "Try to extract from nested structures.", "kind": "method", "line": 288, "name": "_try_nested_extraction", "signature": "def _try_nested_extraction(self, data, device)"}, {"doc": "Try to extract tensors directly from any structure.", "kind": "method", "line": 313, "name": "_try_direct_tensor_extraction", "signature": "def _try_direct_tensor_extraction(self, data, device)"}, {"doc": "Reconstruct U, V, W from found tensors.", "kind": "method", "line": 338, "name": "_reconstruct_from_tensors", "signature": "def _reconstruct_from_tensors(self, tensors, device)"}, {"doc": "Check if dict looks like a state dict.", "kind": "method", "line": 386, "name": "_is_state_dict", "signature": "def _is_state_dict(self, data)"}, {"doc": "Migrate a state dict to the expected format.", "kind": "method", "line": 398, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict, device)"}, {"kind": "method", "line": 424, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(self, state_dict, device)"}, {"kind": "method", "line": 449, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(self, state_dict, device)"}, {"kind": "method", "line": 456, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(self, state_dict, device)"}, {"doc": "Handle encoder.layers style checkpoints.", "kind": "method", "line": 466, "name": "_migrate_encoder_format", "signature": "def _migrate_encoder_format(self, state_dict, device)"}, {"doc": "Handle prefixed state dict keys.", "kind": "method", "line": 496, "name": "_migrate_prefixed_format", "signature": "def _migrate_prefixed_format(self, state_dict, prefix, device)"}, {"doc": "Extract all relevant metadata from checkpoint.", "kind": "method", "line": 518, "name": "extract", "signature": "def extract(checkpoint)"}, {"doc": "Recursively search for delta in nested structures.", "kind": "method", "line": 571, "name": "_extract_delta", "signature": "def _extract_delta(data, depth)"}, {"kind": "method", "line": 612, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 615, "name": "calculate", "signature": "def calculate(self, model, gradient, precomputed_delta)"}, {"kind": "method", "line": 679, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 682, "name": "calculate", "signature": "def calculate(self, model, loss, precomputed_delta)"}, {"kind": "method", "line": 771, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 774, "name": "calculate", "signature": "def calculate(self, model, loss, loss_history)"}, {"kind": "method", "line": 826, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 829, "name": "calculate", "signature": "def calculate(self, model, h_bar, G_alg)"}, {"kind": "method", "line": 886, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 889, "name": "calculate", "signature": "def calculate(self, model, G_alg, c_eff, h_bar)"}, {"kind": "method", "line": 942, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 945, "name": "calculate", "signature": "def calculate(self, model, M_eff)"}, {"kind": "method", "line": 994, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate all Hawking radiation metrics.", "kind": "method", "line": 1003, "name": "calculate_all", "signature": "def calculate_all(self, model, loss, loss_history, gradient, precomputed_delta)"}, {"kind": "method", "line": 1151, "name": "_classify_state", "signature": "def _classify_state(self, delta, T_hawking)"}, {"kind": "method", "line": 1169, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Analyze a single checkpoint with robust error handling.", "kind": "method", "line": 1174, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(self, checkpoint_path)"}, {"doc": "Compute gradient on random batch.", "kind": "method", "line": 1227, "name": "_compute_gradient", "signature": "def _compute_gradient(self, model)"}, {"doc": "Print formatted report.", "kind": "method", "line": 1255, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"doc": "Analyze all checkpoints in directory.", "kind": "method", "line": 1295, "name": "analyze_directory", "signature": "def analyze_directory(self, checkpoint_dir, output_dir, pattern)"}, {"doc": "Generate aggregate summary.", "kind": "method", "line": 1342, "name": "_generate_summary", "signature": "def _generate_summary(self, results, errors)"}, {"kind": "class", "line": 64, "name": "DummyClass", "signature": "class DummyClass"}, {"kind": "method", "line": 317, "name": "extract_tensors", "signature": "def extract_tensors(obj, prefix)"}, {"kind": "method", "line": 65, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 72, "name": "__repr__", "signature": "def __repr__(self)"}, {"kind": "method", "line": 75, "name": "__getitem__", "signature": "def __getitem__(self, key)"}, {"kind": "method", "line": 78, "name": "keys", "signature": "def keys(self)"}, {"kind": "method", "line": 81, "name": "values", "signature": "def values(self)"}, {"kind": "method", "line": 84, "name": "items", "signature": "def items(self)"}, {"kind": "method", "line": 87, "name": "get", "signature": "def get(self, key, default)"}]}, {"id": "install.sh", "kind": "module", "label": "install.sh", "language": "sh", "sha256": "c907d80fd6734993", "symbol_count": 0, "symbols": []}, {"id": "maxwell_strassen_analysis.py", "kind": "module", "label": "maxwell_strassen_analysis.py", "language": "py", "sha256": "769e8472db8aa189", "symbol_count": 61, "symbols": [{"doc": "Configuration for Maxwellian analysis of Strassen crystals.\n\nPhysics Parameters:\n-------------------\nLATTICE_CONSTANT: Spacing between weight nodes (arbitrary units, e.g., nm scale).\nPERMITTIVITY_VACUUM: ε_0 (scaled for numerical stability).\nPERMITTIVITY_WEIGHT_SCALE: Factor to convert weight magnitude to dielectric contrast.\n\nSimulation Parameters:\n----------------------\nGRID_DIMENSION: Size of the cubic lattice for weight embedding (N x N x N).\nFREQUENCY_SAMPLES: Number of frequency points for spectral analysis.\nWAVEVECTOR_RANGE: Range of k-vectors for scattering simulation.\n\nAnalysis Thresholds:\n--------------------\nCRYSTALLINITY_THRESHOLD: Entropy threshold to classify as Crystal vs Glass.\nBRAGG_PEAK_SHARPNESS: Minimum prominence for a peak to be considered Bragg scattering.", "kind": "class", "line": 42, "name": "MaxwellConfiguration", "signature": "class MaxwellConfiguration"}, {"kind": "class", "line": 116, "name": "IModel", "signature": "class IModel(Protocol)"}, {"kind": "class", "line": 121, "name": "IGeometryMapper", "signature": "class IGeometryMapper(Protocol)"}, {"kind": "class", "line": 126, "name": "IMaxwellSolver", "signature": "class IMaxwellSolver(Protocol)"}, {"kind": "class", "line": 132, "name": "IDielectricAnalyzer", "signature": "class IDielectricAnalyzer(Protocol)"}, {"kind": "class", "line": 137, "name": "IPhaseClassifier", "signature": "class IPhaseClassifier(Protocol)"}, {"kind": "class", "line": 145, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"kind": "class", "line": 171, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"doc": "Maps abstract weight vectors to a 3D Dielectric Lattice.\n\nStrategy:\n1. Flatten U, V, W into a single vector.\n2. Populate a 3D grid (NxNxN) using a space-filling curve (Z-order) \n   or layer-wise assignment.\n3. Compute Effective Charge Density ρ and Permittivity ε.\n\nCrystal Hypothesis:\n- Crystallized weights (discrete values) form ordered arrays.\n- Glass weights (random) form noise.", "kind": "class", "line": 227, "name": "StrassenGeometryMapper", "signature": "class StrassenGeometryMapper"}, {"doc": "Analyzes the anisotropy of the dielectric medium.\n\nGlass: Isotropic (ε ~ scalar)\nCrystal: Anisotropic (ε ~ tensor with specific principal axes)", "kind": "class", "line": 288, "name": "DielectricTensorAnalyzer", "signature": "class DielectricTensorAnalyzer"}, {"doc": "Solves Maxwell's equations in the frequency domain via FFT.\n\nKey Analysis:\n1. Electrostatics: Solve ∇·(ε∇φ) = -ρ.\n2. Scattering: Calculate Fourier Transform of Dielectric contrast.\n   - Crystal: Sharp Bragg Peaks at k-vectors determined by lattice periodicity.\n   - Glass: Broad diffuse scattering (Rayleigh).", "kind": "class", "line": 350, "name": "MaxwellScatteringSolver", "signature": "class MaxwellScatteringSolver"}, {"doc": "Calculates the entropy of the electromagnetic field distribution.\nS = -∑ p(E) log p(E)\n\nGlass: High Entropy (Disordered field).\nCrystal: Low Entropy (Ordered, localized modes).", "kind": "class", "line": 466, "name": "PhotonicEntropyCalculator", "signature": "class PhotonicEntropyCalculator"}, {"doc": "Estimates if a photonic bandgap exists.\nA bandgap implies certain frequencies cannot propagate.\n\nWe use the Fourier coefficients to estimate the gap.", "kind": "class", "line": 498, "name": "BandgapAnalyzer", "signature": "class BandgapAnalyzer"}, {"doc": "Classifies the material phase (Crystal vs Glass) based on EM metrics.", "kind": "class", "line": 544, "name": "CrystalPhaseClassifier", "signature": "class CrystalPhaseClassifier"}, {"kind": "class", "line": 605, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"kind": "class", "line": 625, "name": "MaxwellVisualizer", "signature": "class MaxwellVisualizer"}, {"kind": "class", "line": 677, "name": "MaxwellAnalyzer", "signature": "class MaxwellAnalyzer"}, {"kind": "method", "line": 788, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 103, "name": "get_effective_input_dim", "signature": "def get_effective_input_dim(self)"}, {"kind": "method", "line": 106, "name": "get_total_parameters", "signature": "def get_total_parameters(self)"}, {"kind": "method", "line": 117, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 122, "name": "map_weights_to_lattice", "signature": "def map_weights_to_lattice(self, weights)"}, {"kind": "method", "line": 127, "name": "solve_poisson", "signature": "def solve_poisson(self, charge_density, permittivity)"}, {"kind": "method", "line": 128, "name": "compute_scattering", "signature": "def compute_scattering(self, permittivity)"}, {"kind": "method", "line": 133, "name": "analyze_permittivity_tensor", "signature": "def analyze_permittivity_tensor(self, permittivity)"}, {"kind": "method", "line": 138, "name": "classify", "signature": "def classify(self, metrics)"}, {"kind": "method", "line": 146, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 156, "name": "_initialize_weights", "signature": "def _initialize_weights(self)"}, {"kind": "method", "line": 161, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 164, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 172, "name": "migrate", "signature": "def migrate(self, raw_data, config)"}, {"kind": "method", "line": 185, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict, config)"}, {"kind": "method", "line": 194, "name": "_migrate_custom", "signature": "def _migrate_custom(self, sd, config)"}, {"kind": "method", "line": 203, "name": "_migrate_coefs", "signature": "def _migrate_coefs(self, sd)"}, {"kind": "method", "line": 208, "name": "_migrate_standard", "signature": "def _migrate_standard(self, sd)"}, {"kind": "method", "line": 217, "name": "_np", "signature": "def _np(tensor)"}, {"kind": "method", "line": 241, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Returns:\n    charge_density: 3D array of charge distribution.\n    permittivity: 3D array of dielectric constants.", "kind": "method", "line": 244, "name": "map_weights_to_lattice", "signature": "def map_weights_to_lattice(self, weights)"}, {"kind": "method", "line": 295, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Analyze the effective permittivity tensor of the medium.", "kind": "method", "line": 298, "name": "analyze_permittivity_tensor", "signature": "def analyze_permittivity_tensor(self, permittivity)"}, {"kind": "method", "line": 360, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Solves Poisson equation for Electric Potential φ.\nUsing Spectral Method (FFT):\n∇²φ = -ρ / ε_0\n\nFor variable ε (heterogeneous medium), this is approximate.\nWe use the convolution theorem for the Green's function.", "kind": "method", "line": 363, "name": "solve_poisson", "signature": "def solve_poisson(self, charge_density, permittivity)"}, {"doc": "Compute the Scattering Amplitude S(k).\nS(k) ∝ |FT(Δε)|²\n\nΔε = ε(r) - ε_avg (Dielectric contrast)\n\nReturns:\n    Dict containing scattering intensity map and peak analysis.", "kind": "method", "line": 401, "name": "compute_scattering", "signature": "def compute_scattering(self, permittivity)"}, {"doc": "Detect sharp peaks indicative of crystallinity.", "kind": "method", "line": 438, "name": "_find_peaks", "signature": "def _find_peaks(self, intensity)"}, {"kind": "method", "line": 474, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 477, "name": "calculate", "signature": "def calculate(self, potential, intensity)"}, {"kind": "method", "line": 505, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 508, "name": "analyze", "signature": "def analyze(self, fourier_coeffs)"}, {"kind": "method", "line": 548, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Decision Logic:\n1. If Purity Alpha > Threshold AND Discretization is high -> Pre-cursor.\n2. If Anisotropy is High -> Crystal Order.\n3. If Scattering shows Bragg Peaks -> Long Range Order.\n4. If Entropy is Low -> Ordered System.", "kind": "method", "line": 551, "name": "classify", "signature": "def classify(self, em_metrics, purity_metrics)"}, {"kind": "method", "line": 606, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 610, "name": "should_save", "signature": "def should_save(self)"}, {"kind": "method", "line": 613, "name": "save", "signature": "def save(self, data, output_dir)"}, {"kind": "method", "line": 626, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 629, "name": "visualize_lattice", "signature": "def visualize_lattice(self, permittivity, output_dir, name)"}, {"kind": "method", "line": 648, "name": "visualize_scattering", "signature": "def visualize_scattering(self, scattering_slice, output_dir, name)"}, {"kind": "method", "line": 659, "name": "visualize_potential", "signature": "def visualize_potential(self, potential, output_dir, name)"}, {"kind": "method", "line": 678, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 690, "name": "_calculate_purity_metrics", "signature": "def _calculate_purity_metrics(self, weights)"}, {"doc": "Executes the full Maxwellian analysis pipeline on a single checkpoint.", "kind": "method", "line": 701, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(self, checkpoint_path, output_dir)"}, {"kind": "method", "line": 765, "name": "generate_report", "signature": "def generate_report(self, results, output_dir)"}]}, {"doc": "Remove tqdm dependency and use standard library", "id": "mbl_analyzer.py", "kind": "module", "label": "mbl_analyzer.py", "language": "py", "sha256": "50d73c9b61655858", "symbol_count": 89, "symbols": [{"doc": "Comprehensive configuration for MBL analysis of Strassen algorithm crystallization.\nAll parameters are centralized here following SOLID principles.", "kind": "class", "line": 24, "name": "MBLConfiguration", "signature": "class MBLConfiguration"}, {"doc": "Protocol for models compatible with MBL analysis.", "kind": "class", "line": 93, "name": "IModel", "signature": "class IModel(Protocol)"}, {"doc": "Protocol for level spacing ratio calculation.", "kind": "class", "line": 100, "name": "ILevelSpacingCalculator", "signature": "class ILevelSpacingCalculator(Protocol)"}, {"doc": "Protocol for participation ratio calculation.", "kind": "class", "line": 106, "name": "IParticipationRatioCalculator", "signature": "class IParticipationRatioCalculator(Protocol)"}, {"doc": "Protocol for synthetic Planck's constant calculation.", "kind": "class", "line": 112, "name": "ISyntheticPlanckCalculator", "signature": "class ISyntheticPlanckCalculator(Protocol)"}, {"doc": "Protocol for discretization dial analysis.", "kind": "class", "line": 118, "name": "IDiscretizationDialAnalyzer", "signature": "class IDiscretizationDialAnalyzer(Protocol)"}, {"doc": "Protocol for checkpoint management.", "kind": "class", "line": 124, "name": "ICheckpointManager", "signature": "class ICheckpointManager(Protocol)"}, {"doc": "Protocol for collecting all training metrics.", "kind": "class", "line": 132, "name": "ITrainingMetricsCollector", "signature": "class ITrainingMetricsCollector(Protocol)"}, {"doc": "Bilinear model for Strassen algorithm implementation.\nRepresents the 2x2 matrix multiplication with hidden dimension expansion.", "kind": "class", "line": 138, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"doc": "Calculates the level spacing ratio r for MBL phase detection.\n\nThe ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})\nwhere delta_n = E_{n+1} - E_n (energy level spacing).\n\nReferences:\n- Oganesyan & Huse (2008): r_WD ≈ 0.53 (Wigner-Dyson, thermal)\n- Poisson statistics: r_P ≈ 0.386 (MBL/localized phase)", "kind": "class", "line": 205, "name": "LevelSpacingRatioCalculator", "signature": "class LevelSpacingRatioCalculator"}, {"doc": "Calculates Inverse Participation Ratio (IPR) for localization analysis.\n\nIPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.\nIPR = 1 for fully localized state, IPR = 1/N for fully delocalized state.\n\nUsed to quantify the 'crystallinity' of the weight distribution.", "kind": "class", "line": 321, "name": "ParticipationRatioCalculator", "signature": "class ParticipationRatioCalculator"}, {"doc": "Calculates effective synthetic Planck's constant (hbar_eff) from model properties.\n\nBased on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)\nwhere PR is the Participation Ratio and Energy_Gap is the spectral gap.\n\nThis represents the quantum of action in the synthetic quantum system.", "kind": "class", "line": 420, "name": "SyntheticPlanckConstantCalculator", "signature": "class SyntheticPlanckConstantCalculator"}, {"doc": "Analyzes the discretization parameter delta as a phase transition control.\n\nThe discretization delta measures how close weights are to discrete values.\nIt acts as a \"dial\" that controls the quantum-classical transition.\n\nThis implements the noise robustness test: applying Gaussian perturbations\nand measuring when the energy gap collapses (loss of quantum protection).", "kind": "class", "line": 486, "name": "DiscretizationDialAnalyzer", "signature": "class DiscretizationDialAnalyzer"}, {"doc": "Original purity calculation preserved exactly as in user's code.\nCalculates the 'crystallinity' of the weight distribution.", "kind": "class", "line": 614, "name": "PurityIndexCalculator", "signature": "class PurityIndexCalculator"}, {"doc": "Original temperature calculation preserved exactly.", "kind": "class", "line": 674, "name": "EffectiveTemperatureCalculator", "signature": "class EffectiveTemperatureCalculator"}, {"doc": "Original phase classification preserved exactly.", "kind": "class", "line": 721, "name": "PhaseClassifier", "signature": "class PhaseClassifier"}, {"doc": "Original checkpoint migration logic preserved exactly.", "kind": "class", "line": 746, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"doc": "Manages checkpoint saving with 5-minute intervals and latest file maintenance.", "kind": "class", "line": 800, "name": "MBLCheckpointManager", "signature": "class MBLCheckpointManager"}, {"doc": "Collects all MBL metrics for comprehensive training monitoring.", "kind": "class", "line": 855, "name": "MBLMetricsCollector", "signature": "class MBLMetricsCollector"}, {"doc": "Comprehensive analyzer for MBL metrics from checkpoints.", "kind": "class", "line": 962, "name": "MBLCheckpointAnalyzer", "signature": "class MBLCheckpointAnalyzer"}, {"doc": "Main pipeline for processing checkpoints and generating reports.", "kind": "class", "line": 1102, "name": "MBLAnalysisPipeline", "signature": "class MBLAnalysisPipeline"}, {"kind": "method", "line": 1227, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 84, "name": "get_effective_input_dim", "signature": "def get_effective_input_dim(self)"}, {"kind": "method", "line": 87, "name": "get_total_parameters", "signature": "def get_total_parameters(self)"}, {"kind": "method", "line": 95, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 96, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 102, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 108, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 114, "name": "calculate", "signature": "def calculate(self, participation_ratio, energy_gap)"}, {"kind": "method", "line": 120, "name": "analyze_robustness", "signature": "def analyze_robustness(self, model, noise_levels)"}, {"kind": "method", "line": 126, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, epoch, metrics, loss_history, path)"}, {"kind": "method", "line": 128, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path)"}, {"kind": "method", "line": 134, "name": "collect", "signature": "def collect(self, model, loss, epoch, loss_history)"}, {"kind": "method", "line": 143, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Xavier initialization with symmetry constraint for U and V.", "kind": "method", "line": 154, "name": "_initialize_weights", "signature": "def _initialize_weights(self)"}, {"doc": "Forward pass implementing bilinear multiplication.", "kind": "method", "line": 160, "name": "forward", "signature": "def forward(self, a, b)"}, {"doc": "Returns weight matrices for analysis.", "kind": "method", "line": 164, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"doc": "Returns all parameters flattened for Hamiltonian construction.", "kind": "method", "line": 172, "name": "get_flat_parameters", "signature": "def get_flat_parameters(self)"}, {"doc": "Constructs approximate Hessian matrix from weight correlations.\nThis serves as the 'Hamiltonian' for MBL analysis.", "kind": "method", "line": 179, "name": "construct_hessian_approximation", "signature": "def construct_hessian_approximation(self)"}, {"kind": "method", "line": 217, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate level spacing statistics from model weights.\n\nReturns:\n    Dictionary containing mean ratio, variance, and phase classification.", "kind": "method", "line": 220, "name": "calculate", "signature": "def calculate(self, model)"}, {"doc": "Alternative Hessian construction for generic models.", "kind": "method", "line": 267, "name": "_construct_hessian_from_weights", "signature": "def _construct_hessian_from_weights(self, model)"}, {"doc": "Compute sorted eigenvalues of the Hamiltonian.", "kind": "method", "line": 283, "name": "_compute_eigenvalues", "signature": "def _compute_eigenvalues(self, hessian)"}, {"doc": "Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).", "kind": "method", "line": 288, "name": "_calculate_spacing_ratios", "signature": "def _calculate_spacing_ratios(self, spacings)"}, {"doc": "Classify the quantum phase based on level spacing ratio.", "kind": "method", "line": 303, "name": "_classify_phase", "signature": "def _classify_phase(self, mean_ratio)"}, {"kind": "method", "line": 331, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate participation ratios for all weight layers.\n\nReturns:\n    Dictionary containing global and layer-wise IPR metrics.", "kind": "method", "line": 334, "name": "calculate", "signature": "def calculate(self, model)"}, {"doc": "Calculate standard Inverse Participation Ratio.\nIPR = sum_i |c_i|^4 / (sum_i |c_i|^2)^2", "kind": "method", "line": 380, "name": "_calculate_ipr", "signature": "def _calculate_ipr(self, coefficients)"}, {"doc": "Calculate q-th order Rényi IPR.\nI_q = sum_i |c_i|^{2q} / (sum_i |c_i|^2)^q", "kind": "method", "line": 395, "name": "_calculate_renyi_ipr", "signature": "def _calculate_renyi_ipr(self, coefficients, q)"}, {"doc": "Calculate fractal dimension D_q from IPR.\nIPR ~ N^{-D_q} => D_q = -log(IPR) / log(N)", "kind": "method", "line": 409, "name": "_calculate_fractal_dimension", "signature": "def _calculate_fractal_dimension(self, ipr, n)"}, {"kind": "method", "line": 430, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate synthetic Planck's constant.\n\nArgs:\n    participation_ratio: Inverse participation ratio (measure of localization)\n    energy_gap: Energy gap from spectrum (measure of quantum discreteness)\n\nReturns:\n    Synthetic hbar value representing the quantum scale of the system.", "kind": "method", "line": 433, "name": "calculate", "signature": "def calculate(self, participation_ratio, energy_gap)"}, {"doc": "Comprehensive calculation from model and previous analyses.", "kind": "method", "line": 456, "name": "calculate_from_model", "signature": "def calculate_from_model(self, model, level_spacing_results, pr_results)"}, {"kind": "method", "line": 497, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate the base discretization level from weight rounding error.", "kind": "method", "line": 501, "name": "calculate_base_discretization", "signature": "def calculate_base_discretization(self, model)"}, {"doc": "Test robustness by applying noise and measuring gap collapse.\n\nArgs:\n    model: The neural network model\n    noise_levels: Tuple of noise magnitudes to test\n\nReturns:\n    Dictionary containing robustness metrics and phase transition points.", "kind": "method", "line": 528, "name": "analyze_robustness", "signature": "def analyze_robustness(self, model, noise_levels)"}, {"doc": "Apply noise to model and measure resulting metrics.", "kind": "method", "line": 584, "name": "_perturb_and_measure", "signature": "def _perturb_and_measure(self, model, noise_level)"}, {"doc": "Convert discretization error to purity alpha.", "kind": "method", "line": 607, "name": "_delta_to_alpha", "signature": "def _delta_to_alpha(self, delta)"}, {"kind": "method", "line": 620, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 623, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 652, "name": "_compute_layer_purity", "signature": "def _compute_layer_purity(self, weights)"}, {"kind": "method", "line": 658, "name": "_delta_to_alpha", "signature": "def _delta_to_alpha(self, delta)"}, {"kind": "method", "line": 663, "name": "_assess_purity_quality", "signature": "def _assess_purity_quality(self, alpha, variance)"}, {"kind": "method", "line": 679, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 682, "name": "calculate", "signature": "def calculate(self, loss_history)"}, {"kind": "method", "line": 726, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 729, "name": "classify", "signature": "def classify(self, alpha, temperature)"}, {"kind": "method", "line": 751, "name": "migrate", "signature": "def migrate(self, raw_data, device)"}, {"kind": "method", "line": 761, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict, device)"}, {"kind": "method", "line": 770, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(self, state_dict, device)"}, {"kind": "method", "line": 789, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(self, state_dict)"}, {"kind": "method", "line": 796, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(self, state_dict)"}, {"kind": "method", "line": 805, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Check if 5 minutes have elapsed since last checkpoint.", "kind": "method", "line": 810, "name": "should_save_checkpoint", "signature": "def should_save_checkpoint(self)"}, {"doc": "Save checkpoint with all MBL metrics.", "kind": "method", "line": 816, "name": "save_checkpoint", "signature": "def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)"}, {"doc": "Load checkpoint with automatic device placement.", "kind": "method", "line": 850, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path)"}, {"kind": "method", "line": 860, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Collect all metrics for the current training state.", "kind": "method", "line": 870, "name": "collect", "signature": "def collect(self, model, loss, epoch, loss_history)"}, {"doc": "Classify the combined quantum phase.", "kind": "method", "line": 946, "name": "_classify_quantum_phase", "signature": "def _classify_quantum_phase(self, level_spacing, hbar_results)"}, {"kind": "method", "line": 967, "name": "__init__", "signature": "def __init__(self, checkpoint_path, config)"}, {"doc": "Load and migrate checkpoint to model.", "kind": "method", "line": 975, "name": "_load_checkpoint", "signature": "def _load_checkpoint(self)"}, {"doc": "Perform complete MBL analysis on checkpoint.", "kind": "method", "line": 998, "name": "analyze", "signature": "def analyze(self)"}, {"doc": "Generate executive summary of analysis.", "kind": "method", "line": 1026, "name": "_generate_summary", "signature": "def _generate_summary(self, metrics, robustness)"}, {"doc": "Print formatted analysis report.", "kind": "method", "line": 1043, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 1107, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Process single checkpoint and save results.", "kind": "method", "line": 1110, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"doc": "Process multiple checkpoints from directory.", "kind": "method", "line": 1125, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"doc": "Generate aggregate summary report.", "kind": "method", "line": 1155, "name": "generate_summary", "signature": "def generate_summary(self, all_results, output_dir)"}, {"doc": "Generate human-readable text report.", "kind": "method", "line": 1188, "name": "_generate_text_report", "signature": "def _generate_text_report(self, summary, output_dir)"}]}, {"id": "measure_strassen.py", "kind": "module", "label": "measure_strassen.py", "language": "py", "sha256": "67a9ba7e1518ed1e", "symbol_count": 0, "symbols": []}, {"id": "menu.py", "kind": "module", "label": "menu.py", "language": "py", "sha256": "09c4a59fc46f1674", "symbol_count": 9, "symbols": [{"kind": "function", "line": 304, "name": "clear_screen", "signature": "def clear_screen()"}, {"kind": "function", "line": 308, "name": "print_header", "signature": "def print_header(title, subtitle)"}, {"kind": "function", "line": 318, "name": "print_wrapped", "signature": "def print_wrapped(text, indent)"}, {"kind": "function", "line": 323, "name": "wait_for_enter", "signature": "def wait_for_enter()"}, {"kind": "function", "line": 332, "name": "run_script", "signature": "def run_script(entry)"}, {"kind": "function", "line": 362, "name": "show_checkpoints", "signature": "def show_checkpoints()"}, {"kind": "function", "line": 397, "name": "show_results", "signature": "def show_results()"}, {"kind": "function", "line": 433, "name": "show_category", "signature": "def show_category(cat)"}, {"kind": "function", "line": 473, "name": "main_menu", "signature": "def main_menu()"}]}, {"id": "percolation_analysis.py", "kind": "module", "label": "percolation_analysis.py", "language": "py", "sha256": "ec6178b1979e8a99", "symbol_count": 91, "symbols": [{"kind": "class", "line": 31, "name": "PercolationConfiguration", "signature": "class PercolationConfiguration"}, {"kind": "class", "line": 97, "name": "IModel", "signature": "class IModel(Protocol)"}, {"kind": "class", "line": 101, "name": "NumpyModelWrapper", "signature": "class NumpyModelWrapper"}, {"doc": "Transparent stand-in for any unpicklable class found inside a checkpoint.\nStores all keyword and positional constructor arguments as attributes so\nthat downstream dict-key lookups still work on objects that behave like\nnamespaces (e.g. UnifiedConfig, TrainingConfig, etc.).", "kind": "class", "line": 148, "name": "_DummyObject", "signature": "class _DummyObject"}, {"doc": "Load a torch checkpoint that may contain unknown serialized classes\n(UnifiedConfig, TrainingConfig, custom dataclasses, etc.).\n\nStrategy:\n  1. Try normal torch.load.  If it works, return immediately.\n  2. On AttributeError / ModuleNotFoundError, extract the missing\n     class names from the exception, inject _DummyObject into\n     sys.modules / __main__ under those names, and retry.\n  3. Repeat up to MAX_RETRIES times (each retry may reveal a\n     new missing class that was nested deeper in the pickle stream).\n  4. Clean up injected names after loading.\n\nThis keeps torch's own deserialization path (including correct\npersistent_load for tensor storage) fully intact.", "kind": "method", "line": 165, "name": "_safe_torch_load", "signature": "def _safe_torch_load(path)"}, {"kind": "class", "line": 264, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"kind": "class", "line": 350, "name": "WeightGraphConstructor", "signature": "class WeightGraphConstructor"}, {"kind": "class", "line": 402, "name": "BondPercolationAnalyzer", "signature": "class BondPercolationAnalyzer"}, {"kind": "class", "line": 488, "name": "SitePercolationAnalyzer", "signature": "class SitePercolationAnalyzer"}, {"kind": "class", "line": 542, "name": "PruningPercolationAnalyzer", "signature": "class PruningPercolationAnalyzer"}, {"kind": "class", "line": 725, "name": "ClusterSizeDistributionAnalyzer", "signature": "class ClusterSizeDistributionAnalyzer"}, {"kind": "class", "line": 775, "name": "PercolationUniversalityAnalyzer", "signature": "class PercolationUniversalityAnalyzer"}, {"kind": "class", "line": 808, "name": "PercolationCheckpointManager", "signature": "class PercolationCheckpointManager"}, {"kind": "class", "line": 837, "name": "PercolationVisualizationEngine", "signature": "class PercolationVisualizationEngine"}, {"kind": "class", "line": 1049, "name": "PercolationReportGenerator", "signature": "class PercolationReportGenerator"}, {"kind": "class", "line": 1139, "name": "PercolationAnalysisPipeline", "signature": "class PercolationAnalysisPipeline"}, {"kind": "method", "line": 1264, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 82, "name": "get_effective_input_dim", "signature": "def get_effective_input_dim(self)"}, {"kind": "method", "line": 85, "name": "get_total_parameters", "signature": "def get_total_parameters(self)"}, {"kind": "method", "line": 89, "name": "get_percolation_thresholds", "signature": "def get_percolation_thresholds(self)"}, {"kind": "method", "line": 98, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 102, "name": "__init__", "signature": "def __init__(self, weights)"}, {"kind": "method", "line": 105, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 108, "name": "get_flat_parameters", "signature": "def get_flat_parameters(self)"}, {"kind": "class", "line": 117, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"kind": "method", "line": 155, "name": "__init__", "signature": "def __init__(self)"}, {"kind": "method", "line": 161, "name": "__repr__", "signature": "def __repr__(self)"}, {"kind": "method", "line": 189, "name": "_patch_missing", "signature": "def _patch_missing(exc)"}, {"kind": "method", "line": 230, "name": "_cleanup", "signature": "def _cleanup()"}, {"kind": "method", "line": 265, "name": "migrate", "signature": "def migrate(self, raw_data, config)"}, {"kind": "method", "line": 285, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict, config)"}, {"kind": "method", "line": 300, "name": "_try_migrate_nested", "signature": "def _try_migrate_nested(self, candidate, config)"}, {"kind": "method", "line": 310, "name": "_migrate_custom", "signature": "def _migrate_custom(self, sd, config)"}, {"kind": "method", "line": 326, "name": "_migrate_coefs", "signature": "def _migrate_coefs(self, sd)"}, {"kind": "method", "line": 331, "name": "_migrate_standard", "signature": "def _migrate_standard(self, sd)"}, {"kind": "method", "line": 340, "name": "_np", "signature": "def _np(tensor)"}, {"kind": "method", "line": 351, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 354, "name": "construct_adjacency_from_weights", "signature": "def construct_adjacency_from_weights(self, weights)"}, {"kind": "method", "line": 375, "name": "construct_weight_correlation_graph", "signature": "def construct_weight_correlation_graph(self, weights)"}, {"kind": "method", "line": 386, "name": "construct_slot_interaction_graph", "signature": "def construct_slot_interaction_graph(self, weights)"}, {"kind": "method", "line": 403, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 406, "name": "analyze", "signature": "def analyze(self, adjacency, thresholds)"}, {"kind": "method", "line": 449, "name": "_susceptibility", "signature": "def _susceptibility(self, sizes, n)"}, {"kind": "method", "line": 456, "name": "_find_pc", "signature": "def _find_pc(self, thresholds, res)"}, {"kind": "method", "line": 462, "name": "_exponents", "signature": "def _exponents(self, thresholds, res, pc)"}, {"kind": "method", "line": 478, "name": "_fit_pl", "signature": "def _fit_pl(x, y)"}, {"kind": "method", "line": 489, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 492, "name": "analyze", "signature": "def analyze(self, weights, thresholds)"}, {"kind": "method", "line": 543, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 546, "name": "analyze", "signature": "def analyze(self, weights)"}, {"kind": "method", "line": 626, "name": "_d2a", "signature": "def _d2a(self, delta)"}, {"kind": "method", "line": 631, "name": "_kappa", "signature": "def _kappa(self, wv)"}, {"kind": "method", "line": 644, "name": "_hbar", "signature": "def _hbar(self, wv)"}, {"kind": "method", "line": 656, "name": "_teff", "signature": "def _teff(self, wv)"}, {"kind": "method", "line": 659, "name": "_lc", "signature": "def _lc(self, wv, delta)"}, {"kind": "method", "line": 669, "name": "_entropy", "signature": "def _entropy(self, wv)"}, {"kind": "method", "line": 680, "name": "_ipr", "signature": "def _ipr(self, wv)"}, {"kind": "method", "line": 687, "name": "_lsr", "signature": "def _lsr(self, wv)"}, {"kind": "method", "line": 699, "name": "_fractal", "signature": "def _fractal(self, ipr, n)"}, {"kind": "method", "line": 704, "name": "_phase", "signature": "def _phase(self, alpha, temp, delta)"}, {"kind": "method", "line": 726, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 729, "name": "analyze_at_threshold", "signature": "def analyze_at_threshold(self, adjacency, threshold)"}, {"kind": "method", "line": 753, "name": "_tau", "signature": "def _tau(self, sizes)"}, {"kind": "method", "line": 767, "name": "_critical", "signature": "def _critical(self, sizes, n)"}, {"kind": "method", "line": 776, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 779, "name": "classify_universality", "signature": "def classify_universality(self, measured)"}, {"kind": "method", "line": 809, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 814, "name": "should_save", "signature": "def should_save(self)"}, {"kind": "method", "line": 817, "name": "save", "signature": "def save(self, results, output_dir)"}, {"kind": "method", "line": 829, "name": "load", "signature": "def load(self, output_dir)"}, {"kind": "method", "line": 838, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 841, "name": "generate_all_figures", "signature": "def generate_all_figures(self, results, output_dir)"}, {"kind": "method", "line": 856, "name": "_plot_bond", "signature": "def _plot_bond(self, data, out)"}, {"kind": "method", "line": 889, "name": "_plot_pruning", "signature": "def _plot_pruning(self, data, out)"}, {"kind": "method", "line": 952, "name": "_plot_dashboard", "signature": "def _plot_dashboard(self, data, out)"}, {"kind": "method", "line": 993, "name": "_plot_site", "signature": "def _plot_site(self, data, out)"}, {"kind": "method", "line": 1019, "name": "_plot_cluster", "signature": "def _plot_cluster(self, data, out)"}, {"kind": "method", "line": 1050, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1053, "name": "generate_text_report", "signature": "def generate_text_report(self, results, output_dir)"}, {"kind": "method", "line": 1131, "name": "generate_json_report", "signature": "def generate_json_report(self, results, output_dir)"}, {"kind": "method", "line": 1140, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1153, "name": "_load_weights", "signature": "def _load_weights(self, checkpoint_path)"}, {"kind": "method", "line": 1164, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"kind": "method", "line": 1212, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 1234, "name": "_maybe_save", "signature": "def _maybe_save(self, results, output_dir)"}, {"kind": "method", "line": 1238, "name": "_comparative_summary", "signature": "def _comparative_summary(self, all_res, output_dir)"}, {"kind": "method", "line": 118, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 127, "name": "_initialize_weights", "signature": "def _initialize_weights(self)"}, {"kind": "method", "line": 132, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 135, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 141, "name": "get_flat_parameters", "signature": "def get_flat_parameters(self)"}]}, {"id": "plank.py", "kind": "module", "label": "plank.py", "language": "py", "sha256": "9ca42739b0c2e0da", "symbol_count": 72, "symbols": [{"doc": "Immutable configuration container following Single Responsibility Principle.", "kind": "class", "line": 32, "name": "Configuration", "signature": "class Configuration"}, {"doc": "Set random seeds for reproducibility.", "kind": "method", "line": 114, "name": "set_random_seed", "signature": "def set_random_seed(seed)"}, {"doc": "Bilinear model for Strassen matrix multiplication.\nImplements f(A,B) = W((U*A) ⊙ (V*B)) where ⊙ is element-wise product.", "kind": "class", "line": 127, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"doc": "Abstract base for checkpoint migration strategies.", "kind": "class", "line": 194, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator(ABC)"}, {"doc": "Handles custom U,V,W direct formats.", "kind": "class", "line": 208, "name": "CustomFormatMigrator", "signature": "class CustomFormatMigrator(CheckpointMigrator)"}, {"doc": "Handles encoder.layers format.", "kind": "class", "line": 237, "name": "EncoderFormatMigrator", "signature": "class EncoderFormatMigrator(CheckpointMigrator)"}, {"doc": "Handles standard U.weight, V.weight, W.weight format.", "kind": "class", "line": 270, "name": "StandardFormatMigrator", "signature": "class StandardFormatMigrator(CheckpointMigrator)"}, {"doc": "Manages multiple migration strategies.", "kind": "class", "line": 284, "name": "CheckpointMigrationManager", "signature": "class CheckpointMigrationManager"}, {"doc": "Generates training data for 2x2 matrix multiplication.", "kind": "class", "line": 341, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"doc": "Computes crystallographic quality metrics for Strassen models.", "kind": "class", "line": 384, "name": "CrystallographyMetrics", "signature": "class CrystallographyMetrics"}, {"doc": "Tests gauge invariance through permutation symmetry.", "kind": "class", "line": 483, "name": "StrassenDiffractionTest", "signature": "class StrassenDiffractionTest"}, {"doc": "Measures basin of attraction through noise injection and recovery.", "kind": "class", "line": 560, "name": "BasinResilienceSpectrometer", "signature": "class BasinResilienceSpectrometer"}, {"doc": "Computes normalized purity index from component metrics.", "kind": "class", "line": 677, "name": "CrystalPurityIndex", "signature": "class CrystalPurityIndex"}, {"doc": "Calculates effective Planck constant from Strassen model parameters.\n\nMaps crystallographic metrics to quantum thermodynamic quantities.", "kind": "class", "line": 763, "name": "PlanckConstantCalculator", "signature": "class PlanckConstantCalculator"}, {"doc": "Loads and migrates Strassen checkpoints with fallback strategies.", "kind": "class", "line": 936, "name": "StrassenCheckpointLoader", "signature": "class StrassenCheckpointLoader"}, {"doc": "Complete analysis pipeline for Strassen checkpoints.\n\nOrchestrates crystallographic analysis and Planck constant calculation.", "kind": "class", "line": 1014, "name": "StrassenPlanckAnalyzer", "signature": "class StrassenPlanckAnalyzer"}, {"doc": "Generates reports and visualizations from analysis results.", "kind": "class", "line": 1165, "name": "ReportGenerator", "signature": "class ReportGenerator"}, {"doc": "Parse command line arguments.", "kind": "method", "line": 1339, "name": "parse_arguments", "signature": "def parse_arguments()"}, {"doc": "Create configuration from command line arguments.", "kind": "method", "line": 1387, "name": "create_config_from_args", "signature": "def create_config_from_args(args)"}, {"doc": "Main execution entry point.", "kind": "method", "line": 1399, "name": "main", "signature": "def main()"}, {"doc": "Validate configuration parameters.", "kind": "method", "line": 100, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 133, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Initialize with Xavier uniform, symmetric U and V.", "kind": "method", "line": 143, "name": "_initialize_symmetric", "signature": "def _initialize_symmetric(self)"}, {"doc": "Forward pass computing approximate matrix product.\n\nArgs:\n    matrix_a: Flattened input matrix A [batch, INPUT_DIM]\n    matrix_b: Flattened input matrix B [batch, INPUT_DIM]\n\nReturns:\n    Approximate product C = A @ B [batch, OUTPUT_DIM]", "kind": "method", "line": 149, "name": "forward", "signature": "def forward(self, matrix_a, matrix_b)"}, {"doc": "Return current coefficient matrices.", "kind": "method", "line": 166, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"doc": "Compute effective lambda (confinement potential) from weight magnitudes.\nDerived from weight decay interpretation as harmonic confinement.", "kind": "method", "line": 174, "name": "compute_lambda_effective", "signature": "def compute_lambda_effective(self)"}, {"doc": "Check if this strategy can handle the given state dict.", "kind": "method", "line": 198, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"doc": "Migrate state dict to standard format.", "kind": "method", "line": 203, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 211, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 214, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 240, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 243, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 273, "name": "can_migrate", "signature": "def can_migrate(self, state_dict)"}, {"kind": "method", "line": 276, "name": "migrate", "signature": "def migrate(self, state_dict)"}, {"kind": "method", "line": 287, "name": "__init__", "signature": "def __init__(self)"}, {"doc": "Attempt to migrate checkpoint using available strategies.\n\nArgs:\n    path: Path to checkpoint file\n    device: Device to load tensors to\n\nReturns:\n    Migrated state dict or None if migration fails", "kind": "method", "line": 294, "name": "migrate_checkpoint", "signature": "def migrate_checkpoint(self, path, device)"}, {"doc": "Generate a batch of random matrix pairs and their products.\n\nReturns:\n    Tuple of (A_flat, B_flat, C_flat) where C = A @ B", "kind": "method", "line": 345, "name": "generate_batch", "signature": "def generate_batch(batch_size, config)"}, {"doc": "Verify if coefficients represent valid Strassen structure.", "kind": "method", "line": 366, "name": "verify_structure", "signature": "def verify_structure(coeffs, config)"}, {"doc": "Compute condition number of gradient covariance matrix.\n\nHigh kappa indicates ill-conditioned optimization landscape.\nLow kappa (approaching 1.0) indicates well-conditioned, crystalline structure.", "kind": "method", "line": 388, "name": "compute_kappa", "signature": "def compute_kappa(model, num_batches, config)"}, {"doc": "Compute maximum deviation from nearest integer values.\n\nDelta measures how close coefficients are to discrete (integer) values,\nindicating crystalline structure formation.", "kind": "method", "line": 429, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(coeffs)"}, {"doc": "Compute local complexity based on active parameters.\n\nFrom \"Can't Stop Won't Stop\" paper - measures effective parameter count.", "kind": "method", "line": 445, "name": "compute_local_complexity", "signature": "def compute_local_complexity(model, config)"}, {"doc": "Compute all crystallographic metrics at once.", "kind": "method", "line": 464, "name": "compute_all_metrics", "signature": "def compute_all_metrics(model, config)"}, {"kind": "method", "line": 486, "name": "__init__", "signature": "def __init__(self, model, config)"}, {"doc": "Test if model exhibits true Strassen structure through permutation invariance.\n\nGenuine Strassen algorithm should have exactly one valid permutation (identity).", "kind": "method", "line": 490, "name": "test_gauge_invariance", "signature": "def test_gauge_invariance(self)"}, {"doc": "Compute functional error between original and permuted coefficients.", "kind": "method", "line": 531, "name": "_compute_functional_error", "signature": "def _compute_functional_error(self, test_coeffs)"}, {"kind": "method", "line": 563, "name": "__init__", "signature": "def __init__(self, model, config)"}, {"doc": "Measure resilience across multiple noise levels.\n\nReturns spectrum showing critical noise level where recovery fails.", "kind": "method", "line": 570, "name": "measure_resilience_spectrum", "signature": "def measure_resilience_spectrum(self)"}, {"doc": "Test recovery from noise level sigma.", "kind": "method", "line": 585, "name": "_test_noise_recovery", "signature": "def _test_noise_recovery(self, sigma)"}, {"doc": "Apply Gaussian noise to model parameters.", "kind": "method", "line": 614, "name": "_apply_noise", "signature": "def _apply_noise(self, sigma)"}, {"doc": "Anneal model back to attractor using fine-tuning.\n\nReturns number of epochs needed for recovery.", "kind": "method", "line": 621, "name": "_anneal_to_attractor", "signature": "def _anneal_to_attractor(self)"}, {"doc": "Estimate critical noise level where success rate drops below 50%.", "kind": "method", "line": 654, "name": "_estimate_critical_noise", "signature": "def _estimate_critical_noise(self, results)"}, {"kind": "method", "line": 680, "name": "__init__", "signature": "def __init__(self, metrics, diffraction_results, resilience_results, config)"}, {"doc": "Compute normalized purity index and grade.", "kind": "method", "line": 692, "name": "compute", "signature": "def compute(self)"}, {"doc": "Assign crystallographic grade based on delta (primary indicator).", "kind": "method", "line": 745, "name": "_assign_grade", "signature": "def _assign_grade(self, index, delta)"}, {"kind": "method", "line": 770, "name": "__init__", "signature": "def __init__(self, metrics, training_metrics, config)"}, {"doc": "Execute all Planck constant calculation methods.", "kind": "method", "line": 789, "name": "calculate_all", "signature": "def calculate_all(self)"}, {"doc": "Determine confinement regime and corresponding weights.", "kind": "method", "line": 875, "name": "_determine_regime_and_weights", "signature": "def _determine_regime_and_weights(self)"}, {"doc": "Compute derived Planck-scale constants.", "kind": "method", "line": 886, "name": "_compute_derived_constants", "signature": "def _compute_derived_constants(self, h_bar)"}, {"doc": "Compare calculated constants with physical universe.", "kind": "method", "line": 917, "name": "_compute_universe_comparison", "signature": "def _compute_universe_comparison(self, h_bar)"}, {"kind": "method", "line": 939, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Load checkpoint into model instance.\n\nArgs:\n    checkpoint_path: Path to checkpoint file\n    device: Target device\n\nReturns:\n    Loaded model or None if loading fails", "kind": "method", "line": 943, "name": "load", "signature": "def load(self, checkpoint_path, device)"}, {"doc": "Extract training metrics from checkpoint if available.", "kind": "method", "line": 986, "name": "extract_training_metrics", "signature": "def extract_training_metrics(self, checkpoint_path)"}, {"kind": "method", "line": 1021, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Perform complete analysis of a single checkpoint.\n\nArgs:\n    checkpoint_path: Path to checkpoint file\n    device: Computation device\n\nReturns:\n    Complete analysis report", "kind": "method", "line": 1025, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(self, checkpoint_path, device)"}, {"doc": "Analyze all checkpoints in a directory.\n\nArgs:\n    directory: Directory containing checkpoints\n    device: Computation device\n    pattern: File pattern to match\n\nReturns:\n    List of analysis reports", "kind": "method", "line": 1104, "name": "analyze_directory", "signature": "def analyze_directory(self, directory, device, pattern)"}, {"doc": "Print formatted summary of analysis results.", "kind": "method", "line": 1146, "name": "_print_summary", "signature": "def _print_summary(self, report)"}, {"kind": "method", "line": 1168, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Save individual report as JSON.", "kind": "method", "line": 1173, "name": "save_json_report", "signature": "def save_json_report(self, report, suffix)"}, {"doc": "Save aggregate report from multiple analyses.", "kind": "method", "line": 1185, "name": "save_aggregate_report", "signature": "def save_aggregate_report(self, results)"}, {"doc": "Compute aggregate statistics from summaries.", "kind": "method", "line": 1216, "name": "_compute_statistics", "signature": "def _compute_statistics(self, summaries)"}, {"doc": "Count distribution of grades.", "kind": "method", "line": 1247, "name": "_count_grades", "signature": "def _count_grades(self, summaries)"}, {"doc": "Generate visualization plots.", "kind": "method", "line": 1255, "name": "generate_visualizations", "signature": "def generate_visualizations(self, results)"}]}, {"id": "purity_index.py", "kind": "module", "label": "purity_index.py", "language": "py", "sha256": "3181512c5a14c202", "symbol_count": 56, "symbols": [{"kind": "class", "line": 19, "name": "PurityConfig", "signature": "class PurityConfig"}, {"kind": "class", "line": 44, "name": "IModel", "signature": "class IModel(Protocol)"}, {"kind": "class", "line": 49, "name": "IPurityIndexCalculator", "signature": "class IPurityIndexCalculator(Protocol)"}, {"kind": "class", "line": 54, "name": "IEffectiveTemperatureCalculator", "signature": "class IEffectiveTemperatureCalculator(Protocol)"}, {"kind": "class", "line": 59, "name": "IPhaseClassifier", "signature": "class IPhaseClassifier(Protocol)"}, {"kind": "class", "line": 64, "name": "IPolycrystalAnalyzer", "signature": "class IPolycrystalAnalyzer(Protocol)"}, {"kind": "class", "line": 69, "name": "IPurityComparator", "signature": "class IPurityComparator(Protocol)"}, {"kind": "class", "line": 73, "name": "BilinearModel", "signature": "class BilinearModel(Module)"}, {"kind": "class", "line": 101, "name": "PurityIndexCalculator", "signature": "class PurityIndexCalculator"}, {"kind": "class", "line": 156, "name": "EffectiveTemperatureCalculator", "signature": "class EffectiveTemperatureCalculator"}, {"kind": "class", "line": 199, "name": "PhaseClassifier", "signature": "class PhaseClassifier"}, {"kind": "class", "line": 236, "name": "PolycrystalAnalyzer", "signature": "class PolycrystalAnalyzer"}, {"kind": "class", "line": 274, "name": "PurityComparator", "signature": "class PurityComparator"}, {"kind": "class", "line": 311, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"kind": "class", "line": 361, "name": "PurityAnalyzer", "signature": "class PurityAnalyzer"}, {"kind": "class", "line": 492, "name": "PurityPipeline", "signature": "class PurityPipeline"}, {"kind": "method", "line": 611, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 45, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 50, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 55, "name": "calculate", "signature": "def calculate(self, loss_history)"}, {"kind": "method", "line": 60, "name": "classify", "signature": "def classify(self, alpha, temperature)"}, {"kind": "method", "line": 65, "name": "analyze_polycrystal", "signature": "def analyze_polycrystal(self, model, pruning_level)"}, {"kind": "method", "line": 70, "name": "compare", "signature": "def compare(self, original, polycrystal)"}, {"kind": "method", "line": 74, "name": "__init__", "signature": "def __init__(self, hidden_dim, matrix_size)"}, {"kind": "method", "line": 85, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 90, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 93, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 102, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 105, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 134, "name": "_compute_layer_purity", "signature": "def _compute_layer_purity(self, weights)"}, {"kind": "method", "line": 140, "name": "_delta_to_alpha", "signature": "def _delta_to_alpha(self, delta)"}, {"kind": "method", "line": 145, "name": "_assess_purity_quality", "signature": "def _assess_purity_quality(self, alpha, variance)"}, {"kind": "method", "line": 157, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 160, "name": "calculate", "signature": "def calculate(self, loss_history)"}, {"kind": "method", "line": 200, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 203, "name": "classify", "signature": "def classify(self, alpha, temperature)"}, {"kind": "method", "line": 219, "name": "classify_polycrystal_state", "signature": "def classify_polycrystal_state(self, original_alpha, original_temp, poly_alpha, poly_temp)"}, {"kind": "method", "line": 237, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 243, "name": "analyze_polycrystal", "signature": "def analyze_polycrystal(self, model, pruning_level, loss_history)"}, {"kind": "method", "line": 264, "name": "_prune_model", "signature": "def _prune_model(self, model, sparsity)"}, {"kind": "method", "line": 275, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 279, "name": "compare", "signature": "def compare(self, original, polycrystal)"}, {"kind": "method", "line": 312, "name": "migrate", "signature": "def migrate(self, raw_data, device)"}, {"kind": "method", "line": 322, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict, device)"}, {"kind": "method", "line": 331, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(self, state_dict, device)"}, {"kind": "method", "line": 350, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(self, state_dict)"}, {"kind": "method", "line": 357, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(self, state_dict)"}, {"kind": "method", "line": 362, "name": "__init__", "signature": "def __init__(self, checkpoint_path, config)"}, {"kind": "method", "line": 375, "name": "_load_checkpoint", "signature": "def _load_checkpoint(self)"}, {"kind": "method", "line": 399, "name": "analyze", "signature": "def analyze(self)"}, {"kind": "method", "line": 445, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 493, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 496, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"kind": "method", "line": 510, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 537, "name": "generate_summary", "signature": "def generate_summary(self, all_results, output_dir)"}, {"kind": "method", "line": 574, "name": "_generate_text_report", "signature": "def _generate_text_report(self, summary, output_dir)"}]}, {"id": "repor_experiments.py", "kind": "module", "label": "repor_experiments.py", "language": "py", "sha256": "642c3818df6a311f", "symbol_count": 41, "symbols": [{"kind": "class", "line": 78, "name": "ModelConfig", "signature": "class ModelConfig"}, {"kind": "class", "line": 86, "name": "TrainConfig", "signature": "class TrainConfig"}, {"kind": "class", "line": 96, "name": "Exp1Config", "signature": "class Exp1Config"}, {"kind": "class", "line": 103, "name": "Exp2Config", "signature": "class Exp2Config"}, {"kind": "class", "line": 108, "name": "Exp3Config", "signature": "class Exp3Config"}, {"kind": "class", "line": 114, "name": "Exp4Config", "signature": "class Exp4Config"}, {"kind": "class", "line": 120, "name": "Exp5Config", "signature": "class Exp5Config"}, {"kind": "class", "line": 126, "name": "SuiteConfig", "signature": "class SuiteConfig"}, {"kind": "class", "line": 141, "name": "BilinearModel", "signature": "class BilinearModel(Module)"}, {"kind": "method", "line": 173, "name": "discretize_q", "signature": "def discretize_q(w)"}, {"kind": "method", "line": 176, "name": "compute_delta", "signature": "def compute_delta(U, V, W)"}, {"doc": "Poda a target_rank slots, discretiza a {-1,0,1}, verifica.", "kind": "method", "line": 181, "name": "phase2", "signature": "def phase2(model)"}, {"kind": "method", "line": 204, "name": "_verify_2x2", "signature": "def _verify_2x2(U, V, W, n)"}, {"kind": "method", "line": 214, "name": "zero_shot_verify", "signature": "def zero_shot_verify(U, V, W, sizes)"}, {"kind": "method", "line": 222, "name": "_recursive_strassen", "signature": "def _recursive_strassen(U, V, W, n, trials)"}, {"kind": "method", "line": 234, "name": "_strassen_rec", "signature": "def _strassen_rec(A, B, U, V, W, n)"}, {"kind": "method", "line": 262, "name": "compute_kappa", "signature": "def compute_kappa(model, num_batches, bs)"}, {"kind": "method", "line": 279, "name": "compute_alpha", "signature": "def compute_alpha(delta)"}, {"kind": "method", "line": 284, "name": "compute_teff", "signature": "def compute_teff(model, num_batches, bs)"}, {"kind": "method", "line": 298, "name": "classify_phase", "signature": "def classify_phase(delta)"}, {"kind": "method", "line": 309, "name": "load_checkpoint", "signature": "def load_checkpoint(path, device)"}, {"kind": "method", "line": 329, "name": "_extract_state", "signature": "def _extract_state(d)"}, {"kind": "method", "line": 351, "name": "train_model", "signature": "def train_model(cfg, model, epochs, bs, wd, lr, callback)"}, {"kind": "method", "line": 378, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(path, device)"}, {"kind": "method", "line": 428, "name": "experiment1", "signature": "def experiment1(cfg)"}, {"kind": "method", "line": 475, "name": "experiment2", "signature": "def experiment2(cfg)"}, {"kind": "method", "line": 507, "name": "experiment3", "signature": "def experiment3(cfg)"}, {"kind": "method", "line": 563, "name": "experiment4", "signature": "def experiment4(cfg)"}, {"kind": "method", "line": 607, "name": "experiment5", "signature": "def experiment5(cfg)"}, {"kind": "method", "line": 633, "name": "_test_accuracy", "signature": "def _test_accuracy(model, n, device)"}, {"kind": "method", "line": 644, "name": "_random_prune", "signature": "def _random_prune(model, fraction)"}, {"kind": "method", "line": 655, "name": "_boundary_prune", "signature": "def _boundary_prune(model, fraction)"}, {"kind": "method", "line": 665, "name": "analyze_checkpoints", "signature": "def analyze_checkpoints(ckpt_dir, device)"}, {"kind": "method", "line": 705, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 783, "name": "_save", "signature": "def _save(data, path)"}, {"kind": "method", "line": 142, "name": "__init__", "signature": "def __init__(self, cfg)"}, {"kind": "method", "line": 149, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 156, "name": "slot_importance", "signature": "def slot_importance(self)"}, {"kind": "method", "line": 163, "name": "get_weights", "signature": "def get_weights(self)"}, {"kind": "method", "line": 167, "name": "get_flat", "signature": "def get_flat(self)"}, {"kind": "method", "line": 434, "name": "cb", "signature": "def cb(ep, m, loss, acc)"}]}, {"id": "scrodingger.py", "kind": "module", "label": "scrodingger.py", "language": "py", "sha256": "2de8344e909f3bf3", "symbol_count": 70, "symbols": [{"kind": "class", "line": 19, "name": "SchrodingerConfig", "signature": "class SchrodingerConfig"}, {"kind": "class", "line": 44, "name": "IModel", "signature": "class IModel(Protocol)"}, {"kind": "class", "line": 49, "name": "IWaveFunctionExtractor", "signature": "class IWaveFunctionExtractor(Protocol)"}, {"kind": "class", "line": 54, "name": "IPotentialCalculator", "signature": "class IPotentialCalculator(Protocol)"}, {"kind": "class", "line": 59, "name": "IHamiltonianConstructor", "signature": "class IHamiltonianConstructor(Protocol)"}, {"kind": "class", "line": 64, "name": "IEigenvalueSolver", "signature": "class IEigenvalueSolver(Protocol)"}, {"kind": "class", "line": 69, "name": "ITimeEvolver", "signature": "class ITimeEvolver(Protocol)"}, {"kind": "class", "line": 74, "name": "IExpectationValueCalculator", "signature": "class IExpectationValueCalculator(Protocol)"}, {"kind": "class", "line": 79, "name": "IUncertaintyCalculator", "signature": "class IUncertaintyCalculator(Protocol)"}, {"kind": "class", "line": 84, "name": "ICheckpointLoader", "signature": "class ICheckpointLoader(Protocol)"}, {"kind": "class", "line": 89, "name": "ICheckpointMigrator", "signature": "class ICheckpointMigrator(Protocol)"}, {"kind": "class", "line": 93, "name": "BilinearModel", "signature": "class BilinearModel(Module)"}, {"kind": "class", "line": 121, "name": "WaveFunctionExtractor", "signature": "class WaveFunctionExtractor"}, {"kind": "class", "line": 128, "name": "PotentialCalculator", "signature": "class PotentialCalculator"}, {"kind": "class", "line": 150, "name": "HamiltonianConstructor", "signature": "class HamiltonianConstructor"}, {"kind": "class", "line": 170, "name": "EigenvalueSolver", "signature": "class EigenvalueSolver"}, {"kind": "class", "line": 194, "name": "TimeEvolver", "signature": "class TimeEvolver"}, {"kind": "class", "line": 216, "name": "ExpectationValueCalculator", "signature": "class ExpectationValueCalculator"}, {"kind": "class", "line": 226, "name": "UncertaintyCalculator", "signature": "class UncertaintyCalculator"}, {"kind": "class", "line": 263, "name": "CheckpointLoader", "signature": "class CheckpointLoader"}, {"kind": "class", "line": 271, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"kind": "class", "line": 320, "name": "SchrodingerAnalyzer", "signature": "class SchrodingerAnalyzer"}, {"kind": "class", "line": 542, "name": "WaveFunctionVisualizer", "signature": "class WaveFunctionVisualizer"}, {"kind": "class", "line": 609, "name": "SchrodingerPipeline", "signature": "class SchrodingerPipeline"}, {"kind": "method", "line": 779, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 45, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 50, "name": "extract", "signature": "def extract(self, model)"}, {"kind": "method", "line": 55, "name": "calculate", "signature": "def calculate(self, weights)"}, {"kind": "method", "line": 60, "name": "construct", "signature": "def construct(self, potential, mass)"}, {"kind": "method", "line": 65, "name": "solve", "signature": "def solve(self, hamiltonian, count)"}, {"kind": "method", "line": 70, "name": "evolve", "signature": "def evolve(self, initial_state, hamiltonian, time_steps, dt)"}, {"kind": "method", "line": 75, "name": "calculate", "signature": "def calculate(self, wave_function, operator)"}, {"kind": "method", "line": 80, "name": "calculate", "signature": "def calculate(self, wave_function, position_grid)"}, {"kind": "method", "line": 85, "name": "load", "signature": "def load(self, path, device)"}, {"kind": "method", "line": 90, "name": "migrate", "signature": "def migrate(self, raw_data)"}, {"kind": "method", "line": 94, "name": "__init__", "signature": "def __init__(self, hidden_dim, matrix_size)"}, {"kind": "method", "line": 105, "name": "_initialize", "signature": "def _initialize(self)"}, {"kind": "method", "line": 110, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 113, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 122, "name": "extract", "signature": "def extract(self, model)"}, {"kind": "method", "line": 129, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 132, "name": "calculate", "signature": "def calculate(self, weights)"}, {"kind": "method", "line": 151, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 154, "name": "construct", "signature": "def construct(self, potential, mass)"}, {"kind": "method", "line": 171, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 174, "name": "solve", "signature": "def solve(self, hamiltonian, count)"}, {"kind": "method", "line": 195, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 198, "name": "evolve", "signature": "def evolve(self, initial_state, hamiltonian, time_steps, dt)"}, {"kind": "method", "line": 217, "name": "calculate", "signature": "def calculate(self, wave_function, operator)"}, {"kind": "method", "line": 227, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 230, "name": "calculate", "signature": "def calculate(self, wave_function, position_grid)"}, {"kind": "method", "line": 264, "name": "load", "signature": "def load(self, path, device)"}, {"kind": "method", "line": 272, "name": "migrate", "signature": "def migrate(self, raw_data)"}, {"kind": "method", "line": 284, "name": "_migrate_dict", "signature": "def _migrate_dict(self, state_dict)"}, {"kind": "method", "line": 293, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(self, state_dict)"}, {"kind": "method", "line": 309, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(self, state_dict)"}, {"kind": "method", "line": 316, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(self, state_dict)"}, {"kind": "method", "line": 321, "name": "__init__", "signature": "def __init__(self, checkpoint_path, config)"}, {"kind": "method", "line": 335, "name": "_load_checkpoint", "signature": "def _load_checkpoint(self)"}, {"kind": "method", "line": 357, "name": "analyze", "signature": "def analyze(self)"}, {"kind": "method", "line": 451, "name": "_calculate_tunneling_probability", "signature": "def _calculate_tunneling_probability(self, potential, wave_function)"}, {"kind": "method", "line": 465, "name": "_count_degeneracy", "signature": "def _count_degeneracy(self, eigenvalues)"}, {"kind": "method", "line": 478, "name": "_print_report", "signature": "def _print_report(self, results)"}, {"kind": "method", "line": 543, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 546, "name": "visualize", "signature": "def visualize(self, data, output_path)"}, {"kind": "method", "line": 610, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 614, "name": "process_checkpoint", "signature": "def process_checkpoint(self, checkpoint_path, output_dir)"}, {"kind": "method", "line": 628, "name": "process_directory", "signature": "def process_directory(self, checkpoint_dir, n_latest, output_dir)"}, {"kind": "method", "line": 655, "name": "generate_summary", "signature": "def generate_summary(self, all_results, output_dir)"}, {"kind": "method", "line": 718, "name": "_generate_text_report", "signature": "def _generate_text_report(self, summary, output_dir)"}]}, {"id": "src/benchmarks/benchmark_final.py", "kind": "module", "label": "benchmark_final.py", "language": "py", "sha256": "478e76d6328bcc22", "symbol_count": 4, "symbols": [{"doc": "Multiply using our Strassen Hybrid implementation", "kind": "function", "line": 45, "name": "strassen_hybrid_multiply", "signature": "def strassen_hybrid_multiply(A, B)"}, {"doc": "Standard NumPy BLAS multiplication", "kind": "function", "line": 60, "name": "numpy_multiply", "signature": "def numpy_multiply(A, B)"}, {"doc": "Run benchmark with warmup", "kind": "function", "line": 64, "name": "benchmark", "signature": "def benchmark(func, A, B, warmup, runs)"}, {"kind": "function", "line": 80, "name": "main", "signature": "def main()"}]}, {"id": "src/benchmarks/benchmark_scientific.py", "kind": "module", "label": "benchmark_scientific.py", "language": "py", "sha256": "a2ae739a2896c85a", "symbol_count": 5, "symbols": [{"kind": "function", "line": 38, "name": "strassen_multiply", "signature": "def strassen_multiply(A, B)"}, {"kind": "function", "line": 51, "name": "standard_avx512_multiply", "signature": "def standard_avx512_multiply(A, B)"}, {"kind": "function", "line": 64, "name": "numpy_multiply", "signature": "def numpy_multiply(A, B)"}, {"doc": "Benchmark with statistical analysis", "kind": "function", "line": 67, "name": "benchmark_function", "signature": "def benchmark_function(func, A, B, runs, warmup)"}, {"kind": "function", "line": 91, "name": "main", "signature": "def main()"}]}, {"id": "src/benchmarks/benchmark_strassen.py", "kind": "module", "label": "benchmark_strassen.py", "language": "py", "sha256": "a1d9c60bc8b048b1", "symbol_count": 9, "symbols": [{"kind": "class", "line": 29, "name": "BenchmarkResult", "signature": "class BenchmarkResult"}, {"kind": "class", "line": 45, "name": "BenchmarkConfig", "signature": "class BenchmarkConfig"}, {"doc": "Load configuration from TOML file.", "kind": "method", "line": 61, "name": "load_config", "signature": "def load_config(config_path)"}, {"doc": "Convert dtype string to torch dtype.", "kind": "method", "line": 93, "name": "get_dtype", "signature": "def get_dtype(dtype_str)"}, {"doc": "Estimate memory usage for matrix multiplication.", "kind": "method", "line": 104, "name": "estimate_memory_mb", "signature": "def estimate_memory_mb(n, dtype, batch_size)"}, {"doc": "Benchmark Strassen vs standard matmul for given resolution.", "kind": "method", "line": 114, "name": "benchmark_resolution", "signature": "def benchmark_resolution(n, cfg, dtype)"}, {"doc": "Run full benchmark suite.", "kind": "method", "line": 218, "name": "run_benchmark", "signature": "def run_benchmark(cfg)"}, {"doc": "Save benchmark results to JSON.", "kind": "method", "line": 310, "name": "save_results", "signature": "def save_results(results, filepath)"}, {"kind": "method", "line": 322, "name": "main", "signature": "def main()"}]}, {"id": "src/benchmarks/strassen_numpy.py", "kind": "module", "label": "strassen_numpy.py", "language": "py", "sha256": "8a81b149fdf42ee6", "symbol_count": 5, "symbols": [{"kind": "function", "line": 19, "name": "_load_weights", "signature": "def _load_weights()"}, {"doc": "Strassen 2x2 using grokked coefficients.", "kind": "function", "line": 29, "name": "strassen_2x2_numpy", "signature": "def strassen_2x2_numpy(A, B)"}, {"doc": "Recursive Strassen using NumPy.", "kind": "function", "line": 44, "name": "strassen_numpy", "signature": "def strassen_numpy(A, B)"}, {"doc": "Hybrid Strassen: use Strassen for large matrices, NumPy for small.\nThis is faster because NumPy matmul is highly optimized for small matrices.", "kind": "function", "line": 78, "name": "strassen_hybrid", "signature": "def strassen_hybrid(A, B, threshold)"}, {"doc": "Count multiplications used by Strassen.", "kind": "function", "line": 112, "name": "multiplication_count", "signature": "def multiplication_count(n)"}]}, {"id": "src/discovery/auto_T_discovery.py", "kind": "module", "label": "auto_T_discovery.py", "language": "py", "sha256": "8498357ca9147a0f", "symbol_count": 18, "symbols": [{"doc": "Discovered symmetries in weight matrix", "kind": "class", "line": 22, "name": "SymmetryStructure", "signature": "class SymmetryStructure"}, {"doc": "Automatic discovery of expansion operator T from converged weights.\n\nThe algorithm works in three phases:\n1. Spectral Analysis: Extract dominant singular subspace\n2. Symmetry Detection: Find block/permutation structure\n3. T Construction: Build expansion operator preserving invariants", "kind": "class", "line": 32, "name": "AutoTDiscovery", "signature": "class AutoTDiscovery"}, {"doc": "Verify T discovery on Strassen model", "kind": "method", "line": 306, "name": "verify_strassen_T", "signature": "def verify_strassen_T(model_path, target_sizes)"}, {"doc": "Verify that expanded operator correctly computes matrix multiplication", "kind": "method", "line": 353, "name": "verify_expanded_correctness", "signature": "def verify_expanded_correctness(U, V, W, target_size, expanded)"}, {"doc": "Recursively apply learned Strassen decomposition\n\nThis is the IMPLEMENTATION of T: it shows how the base 2x2 decomposition\nextends to arbitrary sizes via recursive block application.", "kind": "method", "line": 389, "name": "recursive_strassen_multiply", "signature": "def recursive_strassen_multiply(A, B, U, V, W, base_size)"}, {"kind": "method", "line": 42, "name": "__init__", "signature": "def __init__(self, tolerance, verbose)"}, {"doc": "Phase 1 & 2: Analyze weight matrix structure", "kind": "method", "line": 46, "name": "analyze_structure", "signature": "def analyze_structure(self, W)"}, {"doc": "Detect if weights cluster around discrete values", "kind": "method", "line": 89, "name": "_detect_discrete_values", "signature": "def _detect_discrete_values(self, W_flat)"}, {"doc": "Detect repeating block patterns", "kind": "method", "line": 105, "name": "_detect_block_structure", "signature": "def _detect_block_structure(self, W)"}, {"doc": "Score how well blocks repeat (lower = more repetitive)", "kind": "method", "line": 127, "name": "_block_repetition_score", "signature": "def _block_repetition_score(self, W, bm, bn)"}, {"doc": "Detect type of symmetry in weight matrix", "kind": "method", "line": 144, "name": "_detect_symmetry_type", "signature": "def _detect_symmetry_type(self, W)"}, {"doc": "Check if rows are permutations of a base pattern", "kind": "method", "line": 165, "name": "_is_permutation_symmetric", "signature": "def _is_permutation_symmetric(self, W)"}, {"doc": "Check for cyclic/Toeplitz structure", "kind": "method", "line": 171, "name": "_is_cyclic", "signature": "def _is_cyclic(self, W)"}, {"doc": "Estimate dimension of truly invariant subspace", "kind": "method", "line": 184, "name": "_invariant_subspace_dim", "signature": "def _invariant_subspace_dim(self, U, S, rank)"}, {"doc": "Compute error when discretizing to given values", "kind": "method", "line": 198, "name": "_discretization_error", "signature": "def _discretization_error(self, W, values)"}, {"doc": "Print analysis results", "kind": "method", "line": 213, "name": "_print_analysis", "signature": "def _print_analysis(self, W, S, structure)"}, {"doc": "Phase 3: Construct expansion operator T\n\nFor matrix multiplication (U, V, W tensors), T expands via\nrecursive block structure discovered from the base case.\n\nArgs:\n    W_dict: Dictionary with 'U', 'V', 'W' tensors\n    target_size: Target matrix dimension (n' in the paper)\n\nReturns:\n    Expanded weight dictionary", "kind": "method", "line": 229, "name": "construct_T", "signature": "def construct_T(self, W_dict, target_size)"}, {"doc": "Validate that expansion preserves key invariants", "kind": "method", "line": 290, "name": "_validate_expansion", "signature": "def _validate_expansion(self, expanded, structure)"}]}, {"id": "src/native/strassen_c.c", "kind": "module", "label": "strassen_c.c", "language": "c", "sha256": "81b11c24e9e7fc51", "symbol_count": 10, "symbols": [{"doc": "Strassen Matrix Multiplication - C Implementation Author: grisun0  Compila: gcc -O3 -ffast-math -march=native -shared -fPIC -o libstrassen.so strassen_c.c  #include <stdlib.h> #include <string.h> #include <stdio.h> #define THRESHOLD 64 /* Allocate matrix", "kind": "function", "line": 15, "name": "alloc_matrix", "signature": "static float* alloc_matrix(int n)"}, {"doc": "#include <stdlib.h> #include <string.h> #include <stdio.h> #define THRESHOLD 64 /* Allocate matrix static float* alloc_matrix(int n) { return (float*)aligned_alloc(32, n * n * sizeof(float)); } /* Standard matrix multiplication for small matrices", "kind": "function", "line": 20, "name": "matmul_standard", "signature": "static void matmul_standard(float* C, float* A, float* B, int n)"}, {"doc": "/* Standard matrix multiplication for small matrices static void matmul_standard(float* C, float* A, float* B, int n) { memset(C, 0, n * n * sizeof(float)); for (int i = 0; i < n; i++) { for (int k = 0; k < n; k++) { float a_ik = A[i * n + k]; for (int j = 0; j < n; j++) { C[i * n + j] += a_ik * B[k * n + j]; } } } } /* Add matrices: C = A + B", "kind": "function", "line": 33, "name": "mat_add", "signature": "static void mat_add(float* C, float* A, float* B, int n)"}, {"doc": "} } } } /* Add matrices: C = A + B static void mat_add(float* C, float* A, float* B, int n) { int nn = n * n; for (int i = 0; i < nn; i++) { C[i] = A[i] + B[i]; } } /* Subtract matrices: C = A - B", "kind": "function", "line": 41, "name": "mat_sub", "signature": "static void mat_sub(float* C, float* A, float* B, int n)"}, {"doc": "for (int i = 0; i < nn; i++) { C[i] = A[i] + B[i]; } } /* Subtract matrices: C = A - B static void mat_sub(float* C, float* A, float* B, int n) { int nn = n * n; for (int i = 0; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Extract quadrant from matrix", "kind": "function", "line": 49, "name": "extract_quadrant", "signature": "static void extract_quadrant(float* Q, float* M, int n, int row, int col)"}, {"doc": "for (int i = 0; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Extract quadrant from matrix static void extract_quadrant(float* Q, float* M, int n, int row, int col) { int h = n / 2; for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant into matrix", "kind": "function", "line": 57, "name": "insert_quadrant", "signature": "static void insert_quadrant(float* M, float* Q, int n, int row, int col)"}, {"doc": "for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant into matrix static void insert_quadrant(float* M, float* Q, int n, int row, int col) { int h = n / 2; for (int i = 0; i < h; i++) { memcpy(&M[(row + i) * n + col], &Q[i * h], h * sizeof(float)); } } /* Strassen recursive", "kind": "function", "line": 65, "name": "strassen_recursive", "signature": "void strassen_recursive(float* C, float* A, float* B, int n)"}, {"doc": "insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C, C22, n, h, h); /* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7); free(T1); free(T2); free(C11); free(C12); free(C21); free(C22); } /* Public API", "kind": "function", "line": 164, "name": "strassen_multiply", "signature": "void strassen_multiply(float* C, float* A, float* B, int n)"}, {"doc": "/* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7); free(T1); free(T2); free(C11); free(C12); free(C21); free(C22); } /* Public API void strassen_multiply(float* C, float* A, float* B, int n) { strassen_recursive(C, A, B, n); } /* Standard multiply for comparison", "kind": "function", "line": 169, "name": "standard_multiply", "signature": "void standard_multiply(float* C, float* A, float* B, int n)"}, {"kind": "macro", "line": 11, "name": "THRESHOLD"}]}, {"id": "src/native/strassen_optimal.c", "kind": "module", "label": "strassen_optimal.c", "language": "c", "sha256": "9138be30ed50280c", "symbol_count": 3, "symbols": [{"doc": "Uses in-place operations where possible and only applies Strassen for very large matrices where the asymptotic advantage overcomes overhead.  #include <stdlib.h> #include <string.h> #include <stdio.h> #include <cblas.h> /* Only use Strassen for huge matrices where O(n^2.807) wins #define STRASSEN_THRESHOLD 4096 /* Strassen for matrices >= threshold", "kind": "function", "line": 18, "name": "strassen_level", "signature": "static void strassen_level(float* C, float* A, float* B, int n, \n                           float..."}, {"kind": "function", "line": 130, "name": "strassen_optimal", "signature": "void strassen_optimal(float* C, float* A, float* B, int n)"}, {"kind": "macro", "line": 15, "name": "STRASSEN_THRESHOLD"}]}, {"id": "src/native/strassen_turbo.c", "kind": "module", "label": "strassen_turbo.c", "language": "c", "sha256": "c92fc5185093f887", "symbol_count": 12, "symbols": [{"doc": "Compile: gcc -O3 -ffast-math -march=native -fopenmp -mavx2 -shared -fPIC -o libstrassen_turbo.so strassen_turbo.c  #include <stdlib.h> #include <string.h> #include <stdio.h> #include <omp.h> #include <immintrin.h> #define THRESHOLD 128 #define BLOCK_SIZE 32 #define ALIGN 32 /* Aligned allocation", "kind": "function", "line": 25, "name": "alloc_matrix", "signature": "static inline float* alloc_matrix(int n)"}, {"doc": "#include <stdio.h> #include <omp.h> #include <immintrin.h> #define THRESHOLD 128 #define BLOCK_SIZE 32 #define ALIGN 32 /* Aligned allocation static inline float* alloc_matrix(int n) { return (float*)aligned_alloc(ALIGN, n * n * sizeof(float)); } /* AVX2 vectorized matrix addition: C = A + B", "kind": "function", "line": 30, "name": "mat_add_avx", "signature": "static void mat_add_avx(float* __restrict C, const float* __restrict A, \n                        ..."}, {"doc": "for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_add_ps(va, vb); _mm256_store_ps(&C[i], vc); } /* Handle remainder for (; i < nn; i++) { C[i] = A[i] + B[i]; } } /* AVX2 vectorized matrix subtraction: C = A - B", "kind": "function", "line": 50, "name": "mat_sub_avx", "signature": "static void mat_sub_avx(float* __restrict C, const float* __restrict A, \n                        ..."}, {"doc": "for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_sub_ps(va, vb); _mm256_store_ps(&C[i], vc); } for (; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Cache-blocked matrix multiplication with AVX2", "kind": "function", "line": 68, "name": "matmul_blocked_avx", "signature": "static void matmul_blocked_avx(float* __restrict C, const float* __restrict A, \n                 ..."}, {"doc": "_mm256_storeu_ps(&C[i * n + j], vc); } for (; j < j_end; j++) { C[i * n + j] += a_ik * B[k * n + j]; } } } } } } } /* Extract quadrant", "kind": "function", "line": 104, "name": "extract_quadrant", "signature": "static void extract_quadrant(float* __restrict Q, const float* __restrict M, \n                   ..."}, {"doc": "} } /* Extract quadrant static void extract_quadrant(float* __restrict Q, const float* __restrict M, int n, int row, int col) { int h = n / 2; #pragma omp parallel for if(h > 64) for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant", "kind": "function", "line": 114, "name": "insert_quadrant", "signature": "static void insert_quadrant(float* __restrict M, const float* __restrict Q, \n                    ..."}, {"doc": "} } /* Insert quadrant static void insert_quadrant(float* __restrict M, const float* __restrict Q, int n, int row, int col) { int h = n / 2; #pragma omp parallel for if(h > 64) for (int i = 0; i < h; i++) { memcpy(&M[(row + i) * n + col], &Q[i * h], h * sizeof(float)); } } /* Strassen recursive with parallelism", "kind": "function", "line": 124, "name": "strassen_turbo_recursive", "signature": "void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)"}, {"doc": "insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C, C22, n, h, h); /* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7); free(T1); free(T2); free(C11); free(C12); free(C21); free(C22); } /* Public API", "kind": "function", "line": 261, "name": "strassen_turbo", "signature": "void strassen_turbo(float* C, float* A, float* B, int n)"}, {"doc": "free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7); free(T1); free(T2); free(C11); free(C12); free(C21); free(C22); } /* Public API void strassen_turbo(float* C, float* A, float* B, int n) { omp_set_num_threads(omp_get_max_threads()); strassen_turbo_recursive(C, A, B, n, 0); } /* Get number of threads", "kind": "function", "line": 267, "name": "get_num_threads", "signature": "int get_num_threads(void)"}, {"kind": "macro", "line": 19, "name": "THRESHOLD"}, {"kind": "macro", "line": 21, "name": "BLOCK_SIZE"}, {"kind": "macro", "line": 22, "name": "ALIGN"}]}, {"id": "src/training/convergence_theory.py", "kind": "module", "label": "convergence_theory.py", "language": "py", "sha256": "8e8d1fc11a07ee4f", "symbol_count": 15, "symbols": [{"doc": "Metrics for tracking convergence to algorithmic invariance", "kind": "class", "line": 21, "name": "ConvergenceMetrics", "signature": "class ConvergenceMetrics"}, {"doc": "Efficient Hessian trace estimation using Hutchinson's method.\n\ntr(H) ≈ E[v^T H v] where v ~ Rademacher(±1)\n\nComplexity: O(n_samples * forward_backward_pass) instead of O(n²)", "kind": "class", "line": 31, "name": "HutchinsonTraceEstimator", "signature": "class HutchinsonTraceEstimator"}, {"doc": "Estimate ε_hw(B, T) - hardware-induced variance\n\nExtended model addressing reviewer's criticism:\nε_hw(B, T) = α/B + β*cache_miss_rate(B) + γ*thread_contention(T)", "kind": "class", "line": 136, "name": "HardwareNoiseEstimator", "signature": "class HardwareNoiseEstimator"}, {"doc": "THEOREM (Convergence to Algorithmic Invariance)\n\nLet W_t denote the weights at step t under SGD with:\n    W_{t+1} = W_t - η∇L(W_t) + ξ_t\n\nwhere ξ_t is stochastic gradient noise with Var(ξ_t) = σ²/B + ε_hw(B,T).\n\nASSUMPTIONS:\nA1. The target function f admits a rank-r tensor decomposition\nA2. The loss L is twice differentiable with Lipschitz Hessian\nA3. The effective curvature satisfies κ_eff < 0 after step t₀\n\nTHEOREM: Under A1-A3, if\n    \n    Var(ξ_t) < σ_min(W*)² / (η · condition(H))\n\nwhere σ_min(W*) is the smallest non-zero singular value of the optimal\ndecomposition, then with probability 1-δ:\n\n    lim_{t->∞} d(W_t, W*) = 0\n\nwhere d is the subspace distance and W* is the algorithmically invariant solution.\n\nPROOF SKETCH:\n\n1. By A3 (κ_eff < 0), the loss landscape is locally convex near convergence\n\n2. The noise condition ensures gradient updates don't overshoot the\n   invariant subspace defined by σ_min(W*)\n\n3. By spectral gap analysis, W_t projects onto the dominant singular\n   subspace with increasing precision as t -> ∞\n\n4. The discretization {-1, 0, 1} emerges because integer solutions\n   are fixed points of the projection when noise is controlled\n\nIMPLICATIONS FOR T:\n\nThe expansion operator T is constructible because:\n- T preserves the dominant singular subspace (by definition)\n- The rank-r structure is independent of problem scale (by A1)\n- Therefore T = block_embed(W_r) where W_r is the converged rank-r solution", "kind": "method", "line": 193, "name": "convergence_theorem", "signature": "def convergence_theorem()"}, {"doc": "Verify that convergence conditions are satisfied for a trained model.", "kind": "method", "line": 265, "name": "verify_convergence_conditions", "signature": "def verify_convergence_conditions(model, loss_fn, train_data, noise_threshold)"}, {"doc": "Simple model for testing convergence verification", "kind": "class", "line": 346, "name": "SimpleStrassenModel", "signature": "class SimpleStrassenModel(Module)"}, {"kind": "method", "line": 40, "name": "__init__", "signature": "def __init__(self, model, loss_fn, n_samples, device)"}, {"doc": "Estimate tr(H) using Hutchinson's stochastic trace estimator.\n\nReturns:\n    (mean_trace, std_trace)", "kind": "method", "line": 47, "name": "estimate_trace", "signature": "def estimate_trace(self, data)"}, {"doc": "Generate Rademacher random vector (±1 with equal probability)", "kind": "method", "line": 73, "name": "_rademacher_vector", "signature": "def _rademacher_vector(self)"}, {"doc": "Compute H @ v using the \"double backward\" trick.\n\nH @ v = ∂/∂θ (∇L · v)", "kind": "method", "line": 90, "name": "_hessian_vector_product", "signature": "def _hessian_vector_product(self, x, y, v)"}, {"doc": "Compute κ_eff = -tr(H) / N\n\nInterpretation:\n- κ_eff < 0 and stable -> grokking likely\n- κ_eff > 0 or oscillating -> grokking unlikely", "kind": "method", "line": 118, "name": "compute_kappa_eff", "signature": "def compute_kappa_eff(self, data)"}, {"kind": "method", "line": 144, "name": "__init__", "signature": "def __init__(self, model, loss_fn)"}, {"doc": "Estimate hardware noise by measuring gradient variance across batches", "kind": "method", "line": 148, "name": "estimate_noise", "signature": "def estimate_noise(self, data_loader, n_batches, n_threads)"}, {"kind": "method", "line": 348, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 354, "name": "forward", "signature": "def forward(self, x)"}]}, {"id": "src/training/grokkit_physics.py", "kind": "module", "label": "grokkit_physics.py", "language": "py", "sha256": "b47f5cf4b017d494", "symbol_count": 4, "symbols": [{"doc": "Wrapper for Strassen multiplication (uses float32).", "kind": "function", "line": 49, "name": "strassen_multiply", "signature": "def strassen_multiply(A, B)"}, {"doc": "Measure the 'physical quantities' for a given matrix size.\n\nReturns:\n    dict with: speedup, hbar_strassen, hbar_numpy, coherence, error", "kind": "function", "line": 64, "name": "measure_physics", "signature": "def measure_physics(N, num_samples)"}, {"doc": "Find the critical size N_c where the phase transition occurs.\nUses the maximum derivative of speedup curve.", "kind": "function", "line": 126, "name": "detect_phase_transition", "signature": "def detect_phase_transition(results)"}, {"kind": "function", "line": 148, "name": "main", "signature": "def main()"}]}, {"id": "src/training/main.py", "kind": "module", "label": "main.py", "language": "py", "sha256": "8f8ee26aa34943d6", "symbol_count": 21, "symbols": [{"doc": "Configuración del experimento.", "kind": "class", "line": 44, "name": "Config", "signature": "class Config"}, {"doc": "Fijar semilla para reproducibilidad.", "kind": "method", "line": 79, "name": "set_seed", "signature": "def set_seed(seed)"}, {"doc": "Modelo para descubrir Strassen mediante coeficientes aprendibles.\n\nArquitectura:\n- U_coefs[i]: Combinación de bloques de A para producto M_i\n- V_coefs[i]: Combinación de bloques de B para producto M_i  \n- W_coefs[j,i]: Contribución de M_i al bloque C_j del resultado", "kind": "class", "line": 87, "name": "StrassenDiscovery", "signature": "class StrassenDiscovery(Module)"}, {"doc": "Dataset de multiplicación de matrices 4x4.", "kind": "class", "line": 199, "name": "Matrix4x4Dataset", "signature": "class Matrix4x4Dataset(Dataset)"}, {"doc": "Entrenador con enmascaramiento progresivo.", "kind": "class", "line": 217, "name": "Trainer", "signature": "class Trainer"}, {"kind": "method", "line": 345, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 96, "name": "__init__", "signature": "def __init__(self, num_slots)"}, {"doc": "Forward pass con multiplicación matemática pura.\nA, B: (batch, 4, 4) -> C: (batch, 4, 4)", "kind": "method", "line": 108, "name": "forward", "signature": "def forward(self, A, B)"}, {"doc": "Norma promedio de cada slot.", "kind": "method", "line": 155, "name": "get_slot_norms", "signature": "def get_slot_norms(self)"}, {"doc": "Número de slots activos.", "kind": "method", "line": 160, "name": "get_active_slots", "signature": "def get_active_slots(self)"}, {"doc": "Desactiva un slot.", "kind": "method", "line": 164, "name": "mask_slot", "signature": "def mask_slot(self, slot_idx)"}, {"doc": "Slot con menor norma entre los activos.", "kind": "method", "line": 169, "name": "get_weakest_slot", "signature": "def get_weakest_slot(self)"}, {"doc": "Muestra coeficientes descubiertos.", "kind": "method", "line": 175, "name": "print_coefficients", "signature": "def print_coefficients(self)"}, {"kind": "method", "line": 202, "name": "__init__", "signature": "def __init__(self, num_samples, seed)"}, {"kind": "method", "line": 210, "name": "__len__", "signature": "def __len__(self)"}, {"kind": "method", "line": 213, "name": "__getitem__", "signature": "def __getitem__(self, idx)"}, {"kind": "method", "line": 220, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 242, "name": "accuracy", "signature": "def accuracy(self, pred, target)"}, {"kind": "method", "line": 245, "name": "train_epoch", "signature": "def train_epoch(self, optimizer)"}, {"kind": "method", "line": 265, "name": "evaluate", "signature": "def evaluate(self)"}, {"kind": "method", "line": 280, "name": "train", "signature": "def train(self)"}]}, {"id": "src/training/main_pure_math.py", "kind": "module", "label": "main_pure_math.py", "language": "py", "sha256": "cfcc12b7921579ce", "symbol_count": 12, "symbols": [{"doc": "Descomposición de tensor para multiplicación de matrices 2x2.\nC_ij = sum_r W[ij,r] * (U[r,:] @ a) * (V[r,:] @ b)", "kind": "class", "line": 22, "name": "StrassenModel", "signature": "class StrassenModel(Module)"}, {"kind": "method", "line": 58, "name": "gen_data", "signature": "def gen_data(n, scale)"}, {"kind": "method", "line": 64, "name": "train", "signature": "def train(model, epochs, lr, l1, batch, verbose)"}, {"kind": "method", "line": 88, "name": "verify", "signature": "def verify(model, n)"}, {"doc": "Poda los slots más débiles, mantiene top-k.", "kind": "method", "line": 106, "name": "hard_prune", "signature": "def hard_prune(model, keep)"}, {"doc": "Refina manteniendo slots podados en cero.", "kind": "method", "line": 122, "name": "refine_pruned", "signature": "def refine_pruned(model, active, epochs, lr)"}, {"kind": "method", "line": 159, "name": "show_coeffs", "signature": "def show_coeffs(model, active)"}, {"kind": "method", "line": 186, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 28, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 35, "name": "forward", "signature": "def forward(self, A, B)"}, {"doc": "Norma combinada de cada slot.", "kind": "method", "line": 47, "name": "slot_norms", "signature": "def slot_norms(self)"}, {"kind": "method", "line": 54, "name": "active_count", "signature": "def active_count(self, thresh)"}]}, {"id": "src/training/strassen_core.py", "kind": "module", "label": "strassen_core.py", "language": "py", "sha256": "be6d231397927a1a", "symbol_count": 5, "symbols": [{"kind": "function", "line": 14, "name": "_load_weights", "signature": "def _load_weights()"}, {"kind": "function", "line": 21, "name": "strassen_2x2", "signature": "def strassen_2x2(A, B)"}, {"kind": "function", "line": 44, "name": "strassen", "signature": "def strassen(X, Y)"}, {"kind": "function", "line": 77, "name": "get_coefficients", "signature": "def get_coefficients()"}, {"kind": "function", "line": 82, "name": "multiplication_count", "signature": "def multiplication_count(n)"}]}, {"id": "src/training/strassen_grokkit.py", "kind": "module", "label": "strassen_grokkit.py", "language": "py", "sha256": "da1f0444a8b1ccfe", "symbol_count": 12, "symbols": [{"doc": "Operador espectral para multiplicación de matrices 2x2.\n\nRepresenta el tensor de rango R:\nC_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)\n\nDonde:\n- U, V: Coeficientes de combinación lineal (LC)\n- W: Coeficientes de reconstrucción\n- Esparsidad (SP): Cuántos slots están activos", "kind": "class", "line": 27, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"doc": "Genera batch de matrices aleatorias.", "kind": "method", "line": 113, "name": "generate_batch", "signature": "def generate_batch(n, scale)"}, {"doc": "Entrena usando el framework Grokkit.\n\nWD (Weight Decay) actúa como presión termodinámica que:\n1. Empuja hacia soluciones de menor norma\n2. Promueve esparsidad natural (slots débiles -> 0)\n3. Cristaliza el operador en el mínimo de energía (rango 7)", "kind": "method", "line": 120, "name": "train_grokkit", "signature": "def train_grokkit(epochs, batch_size, lr, wd)"}, {"doc": "Verifica que el operador ha grokkeado correctamente.", "kind": "method", "line": 203, "name": "verify_grokking", "signature": "def verify_grokking(model, n_test)"}, {"doc": "Fase 2: Esparsificación progresiva.\nReduce gradualmente a 7 slots manteniendo accuracy.", "kind": "method", "line": 254, "name": "progressive_sparsification", "signature": "def progressive_sparsification(model, target_slots)"}, {"doc": "Pipeline principal Grokkit para Strassen.", "kind": "method", "line": 351, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 40, "name": "__init__", "signature": "def __init__(self, rank)"}, {"doc": "Computa A @ B usando la descomposición tensorial.", "kind": "method", "line": 49, "name": "forward", "signature": "def forward(self, A, B)"}, {"doc": "Linear Combination metric.\nMide qué tan bien los coeficientes forman combinaciones válidas.\nLC -> 1 significa combinaciones perfectas.", "kind": "method", "line": 67, "name": "compute_LC", "signature": "def compute_LC(self)"}, {"doc": "Sparsity metric.\nSP -> 0 significa máxima esparsidad (menos slots activos).\nSP = (slots_activos - 7) / rank para normalizar", "kind": "method", "line": 83, "name": "compute_SP", "signature": "def compute_SP(self)"}, {"doc": "Importancia de cada slot basada en normas.", "kind": "method", "line": 101, "name": "slot_importance", "signature": "def slot_importance(self)"}, {"doc": "Cuenta slots activos.", "kind": "method", "line": 108, "name": "count_active", "signature": "def count_active(self, threshold)"}]}, {"id": "src/training/train_strassen.py", "kind": "module", "label": "train_strassen.py", "language": "py", "sha256": "22b4731c2dca15d5", "symbol_count": 12, "symbols": [{"doc": "Spectral operator for 2x2 matrix multiplication.\n\nTensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)", "kind": "class", "line": 26, "name": "StrassenOperator", "signature": "class StrassenOperator(Module)"}, {"kind": "method", "line": 60, "name": "generate_batch", "signature": "def generate_batch(n, scale)"}, {"doc": "Phase 1: Grokking with Weight Decay as thermodynamic pressure.", "kind": "method", "line": 66, "name": "train_phase1", "signature": "def train_phase1(epochs, batch_size, lr, wd)"}, {"doc": "Phase 2: Progressive sparsification to target rank.", "kind": "method", "line": 104, "name": "sparsify", "signature": "def sparsify(model, target_slots)"}, {"doc": "Phase 3: Discretize coefficients to {-1, 0, 1}.", "kind": "method", "line": 183, "name": "discretize", "signature": "def discretize(model, slots_to_prune)"}, {"doc": "Returns the canonical Strassen coefficients.\nExact discrete coefficients for rank-7 tensor decomposition.\n\nStrassen's 7 products:\nM1 = (a11 + a22)(b11 + b22)\nM2 = (a21 + a22) * b11\nM3 = a11 * (b12 - b22)\nM4 = a22 * (b21 - b11)\nM5 = (a11 + a12) * b22\nM6 = (a21 - a11)(b11 + b12)\nM7 = (a12 - a22)(b21 + b22)\n\nResult reconstruction:\nc11 = M1 + M4 - M5 + M7\nc12 = M3 + M5\nc21 = M2 + M4\nc22 = M1 - M2 + M3 + M6", "kind": "method", "line": 211, "name": "get_canonical_strassen", "signature": "def get_canonical_strassen()"}, {"doc": "Verify the discretized operator.", "kind": "method", "line": 260, "name": "verify", "signature": "def verify(U, V, W, n_test)"}, {"doc": "Main training pipeline.", "kind": "method", "line": 299, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 33, "name": "__init__", "signature": "def __init__(self, rank)"}, {"kind": "method", "line": 40, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 50, "name": "slot_importance", "signature": "def slot_importance(self)"}, {"kind": "method", "line": 56, "name": "count_active", "signature": "def count_active(self, threshold)"}]}, {"id": "superposition.py", "kind": "module", "label": "superposition.py", "language": "py", "sha256": "8bff078b4e30d410", "symbol_count": 56, "symbols": [{"doc": "Configuration for Superposition Analysis on Strassen Checkpoints.", "kind": "class", "line": 23, "name": "Config", "signature": "class Config"}, {"kind": "class", "line": 59, "name": "ICheckpointLoader", "signature": "class ICheckpointLoader(Protocol)"}, {"kind": "class", "line": 62, "name": "IMetricsCalculator", "signature": "class IMetricsCalculator(ABC)"}, {"kind": "class", "line": 66, "name": "IAnalyzer", "signature": "class IAnalyzer(ABC)"}, {"kind": "class", "line": 71, "name": "CheckpointLoadingError", "signature": "class CheckpointLoadingError(Exception)"}, {"doc": "Loads raw checkpoint files.", "kind": "class", "line": 75, "name": "CheckpointLoader", "signature": "class CheckpointLoader"}, {"doc": "Migrates various checkpoint formats to standard state_dict.", "kind": "class", "line": 85, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"doc": "Generates matrix multiplication data for activation extraction.", "kind": "class", "line": 220, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"doc": "Your existing Strassen model architecture.", "kind": "class", "line": 258, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"doc": "SAE with tied weights (W_dec = W_enc^T).\nCorrected dimensions: W_enc: [D, N], encode uses W_enc^T, decode uses W_enc.", "kind": "class", "line": 292, "name": "SparseAutoencoder", "signature": "class SparseAutoencoder(Module)"}, {"doc": "Calculates superposition metrics from Section 4 of the paper.", "kind": "class", "line": 334, "name": "SuperpositionMetrics", "signature": "class SuperpositionMetrics(IMetricsCalculator)"}, {"doc": "Trains SAE on bottleneck activations extracted from Strassen model.", "kind": "class", "line": 407, "name": "SAETrainer", "signature": "class SAETrainer"}, {"doc": "Analyzes existing Strassen checkpoints for superposition metrics.", "kind": "class", "line": 475, "name": "StrassenCheckpointAnalyzer", "signature": "class StrassenCheckpointAnalyzer(IAnalyzer)"}, {"kind": "method", "line": 765, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 54, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 60, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path, device)"}, {"kind": "method", "line": 64, "name": "compute", "signature": "def compute(self)"}, {"kind": "method", "line": 68, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(self, checkpoint_path)"}, {"kind": "method", "line": 78, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path, device)"}, {"doc": "Detect hidden dimension from checkpoint data structure by inspecting\ntensor shapes in various known formats.\nReturns None if cannot be determined unambiguously.", "kind": "method", "line": 89, "name": "detect_hidden_dim", "signature": "def detect_hidden_dim(raw_data)"}, {"kind": "method", "line": 122, "name": "migrate_checkpoint", "signature": "def migrate_checkpoint(raw_data)"}, {"kind": "method", "line": 135, "name": "_migrate_dict", "signature": "def _migrate_dict(state_dict)"}, {"kind": "method", "line": 147, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(state_dict)"}, {"kind": "method", "line": 162, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(state_dict)"}, {"doc": "Handle encoder-based format from specific experimental architectures.\nExtracts U, V, W from sequential encoder layers assuming specific indexing.", "kind": "method", "line": 170, "name": "_migrate_encoder_format", "signature": "def _migrate_encoder_format(state_dict)"}, {"kind": "method", "line": 215, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(state_dict)"}, {"kind": "method", "line": 223, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Generate batch of matrix pairs and their product.", "kind": "method", "line": 226, "name": "generate_batch", "signature": "def generate_batch(self, batch_size)"}, {"doc": "Generate full dataset.", "kind": "method", "line": 240, "name": "generate_dataset", "signature": "def generate_dataset(self, num_samples)"}, {"kind": "method", "line": 261, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 271, "name": "_initialize_symmetric", "signature": "def _initialize_symmetric(self)"}, {"doc": "Forward pass returning output and bottleneck activations.", "kind": "method", "line": 276, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 284, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 298, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Encode bottleneck activations to sparse features.\nx: [batch, N]\nW_enc: [D, N]\nReturns: [batch, D]", "kind": "method", "line": 310, "name": "encode", "signature": "def encode(self, x)"}, {"doc": "Decode sparse features back to bottleneck.\nz: [batch, D]\nW_enc: [D, N]\nReturns: [batch, N]", "kind": "method", "line": 319, "name": "decode", "signature": "def decode(self, z)"}, {"kind": "method", "line": 328, "name": "forward", "signature": "def forward(self, x)"}, {"kind": "method", "line": 339, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Calculate feature probabilities from SAE activations.\np_i = Σ_s |z_i,s| / Σ_j Σ_s |z_j,s|", "kind": "method", "line": 342, "name": "compute_feature_probabilities", "signature": "def compute_feature_probabilities(self, sae_activations)"}, {"doc": "Shannon entropy H(p) = -Σ p_i log p_i.", "kind": "method", "line": 352, "name": "compute_entropy", "signature": "def compute_entropy(self, probabilities)"}, {"doc": "Main metric: ψ = F/N where F = e^{H(p)}.", "kind": "method", "line": 359, "name": "compute_superposition", "signature": "def compute_superposition(self, sae_activations)"}, {"doc": "Baseline from Eq 2: ψ_Frob = ||W||_F^2 / N.\nApplied to the bottleneck transformation (W matrix of Strassen).", "kind": "method", "line": 377, "name": "compute_frobenius_metric", "signature": "def compute_frobenius_metric(self, weight_matrix)"}, {"doc": "Compute W^T @ W to analyze interference patterns.", "kind": "method", "line": 385, "name": "compute_interference_matrix", "signature": "def compute_interference_matrix(self, weight_matrix)"}, {"doc": "Unified interface.", "kind": "method", "line": 389, "name": "compute", "signature": "def compute(self, sae_activations, weight_matrix)"}, {"kind": "method", "line": 410, "name": "__init__", "signature": "def __init__(self, sae, config)"}, {"doc": "Train SAE on extracted activations.", "kind": "method", "line": 421, "name": "train", "signature": "def train(self, bottleneck_activations)"}, {"kind": "method", "line": 480, "name": "__init__", "signature": "def __init__(self, config)"}, {"doc": "Load and migrate checkpoint to model.\nReturns tuple of (model, effective_config) where effective_config has \nthe correct HIDDEN_DIM for this specific checkpoint to avoid dimension mismatches.", "kind": "method", "line": 496, "name": "load_model", "signature": "def load_model(self, checkpoint_path)"}, {"doc": "Extract bottleneck activations (U(a) * V(b)) from model.", "kind": "method", "line": 554, "name": "extract_bottleneck_activations", "signature": "def extract_bottleneck_activations(self, model)"}, {"doc": "Full analysis pipeline for a single checkpoint:\n1. Load model with correct dimensions (detecting hidden_dim from checkpoint)\n2. Extract bottleneck activations\n3. Train SAE with matching dimensions (using effective_config)\n4. Calculate superposition metrics with correct normalization (N=hidden_dim)\n5. Calculate baseline Frobenius metric on W weights", "kind": "method", "line": 571, "name": "analyze_checkpoint", "signature": "def analyze_checkpoint(self, checkpoint_path)"}, {"doc": "Save result for individual checkpoint.", "kind": "method", "line": 633, "name": "_save_intermediate_result", "signature": "def _save_intermediate_result(self, result, name)"}, {"doc": "Analyze all checkpoints in directory.", "kind": "method", "line": 641, "name": "analyze_directory", "signature": "def analyze_directory(self, checkpoint_dir)"}, {"doc": "Save intermediate progress.", "kind": "method", "line": 680, "name": "_save_progress_checkpoint", "signature": "def _save_progress_checkpoint(self, results)"}, {"doc": "Save complete results.", "kind": "method", "line": 687, "name": "_save_final_results", "signature": "def _save_final_results(self, results)"}, {"doc": "Generate comparison plots across checkpoints.", "kind": "method", "line": 694, "name": "_generate_comparison_plots", "signature": "def _generate_comparison_plots(self, results)"}, {"kind": "method", "line": 148, "name": "get_tensor", "signature": "def get_tensor(key)"}]}, {"doc": "train_batch_sweep.py", "id": "train_batch_sweep.py", "kind": "module", "label": "train_batch_sweep.py", "language": "py", "sha256": "5010d7de798d96d3", "symbol_count": 2, "symbols": [{"kind": "function", "line": 11, "name": "train_for_batch_size", "signature": "def train_for_batch_size(B, seed, output_dir)"}, {"kind": "class", "line": 17, "name": "LocalConfig", "signature": "class LocalConfig"}]}, {"id": "unified_hidden_connections_suite.py", "kind": "module", "label": "unified_hidden_connections_suite.py", "language": "py", "sha256": "678a19fb361fbe3a", "symbol_count": 99, "symbols": [{"doc": "Immutable canonical configuration for the Strassen bilinear model.", "kind": "class", "line": 65, "name": "StrassStrassenConfig", "signature": "class StrassStrassenConfig"}, {"doc": "Immutable training hyperparameters.", "kind": "class", "line": 84, "name": "TrainingConfig", "signature": "class TrainingConfig"}, {"doc": "Configuration for Experiment 1: Ricci-MBL Duality.", "kind": "class", "line": 99, "name": "Experiment1Config", "signature": "class Experiment1Config"}, {"doc": "Configuration for Experiment 2: Altland-Zirnbauer Symmetry Dial.", "kind": "class", "line": 114, "name": "Experiment2Config", "signature": "class Experiment2Config"}, {"doc": "Configuration for Experiment 3: Conformal Isomorphism.", "kind": "class", "line": 131, "name": "Experiment3Config", "signature": "class Experiment3Config"}, {"doc": "Configuration for Experiment 4: Compression Frontier.", "kind": "class", "line": 141, "name": "Experiment4Config", "signature": "class Experiment4Config"}, {"doc": "Configuration for Experiment 5: Holographic Pruning.", "kind": "class", "line": 158, "name": "Experiment5Config", "signature": "class Experiment5Config"}, {"doc": "Top-level suite orchestration configuration.", "kind": "class", "line": 169, "name": "SuiteConfig", "signature": "class SuiteConfig"}, {"doc": "Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.\nImplements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.\nU, V ∈ R^{rank x input_dim}, W ∈ R^{output_dim x rank}.", "kind": "class", "line": 183, "name": "StrassStrassenModel", "signature": "class StrassStrassenModel(Module)"}, {"doc": "Protocol for deterministic data generation.", "kind": "class", "line": 229, "name": "IDataGenerator", "signature": "class IDataGenerator(Protocol)"}, {"doc": "Generates random 2x2 matrix pairs and their exact products.", "kind": "class", "line": 235, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"doc": "Protocol for checkpoint persistence.", "kind": "class", "line": 262, "name": "ICheckpointManager", "signature": "class ICheckpointManager(Protocol)"}, {"doc": "Handles safe serialization and deserialization of model checkpoints.", "kind": "class", "line": 269, "name": "CheckpointManager", "signature": "class CheckpointManager"}, {"doc": "Protocol for training routines.", "kind": "class", "line": 293, "name": "ITrainer", "signature": "class ITrainer(Protocol)"}, {"doc": "Standard trainer with AdamW, cosine annealing, and gradient clipping.", "kind": "class", "line": 301, "name": "Trainer", "signature": "class Trainer"}, {"doc": "Abstract base for all metric calculators.", "kind": "class", "line": 357, "name": "IMetricCalculator", "signature": "class IMetricCalculator(ABC)"}, {"doc": "Computes the adjacent gap ratio r for Hessian eigenvalue spectra.", "kind": "class", "line": 365, "name": "LevelSpacingRatioCalculator", "signature": "class LevelSpacingRatioCalculator"}, {"doc": "Computes Ricci scalar and geometric curvature metrics from Hessian.", "kind": "class", "line": 415, "name": "RicciScalarCalculator", "signature": "class RicciScalarCalculator"}, {"doc": "Estimates the synthetic Planck constant from model weight statistics.", "kind": "class", "line": 492, "name": "SyntheticPlanckCalculator", "signature": "class SyntheticPlanckCalculator"}, {"doc": "Measures superposition via sparse autoencoder bottleneck analysis.", "kind": "class", "line": 552, "name": "SuperpositionMetricCalculator", "signature": "class SuperpositionMetricCalculator"}, {"doc": "Abstract base for all experiments in the suite.", "kind": "class", "line": 645, "name": "IExperiment", "signature": "class IExperiment(ABC)"}, {"doc": "Tests the claim that Ricci curvature smoothing is the geometric mechanism\ndriving the Wigner-Dyson to Poisson/MBL spectral transition.", "kind": "class", "line": 657, "name": "Experiment1RicciMBLDuality", "signature": "class Experiment1RicciMBLDuality(IExperiment)"}, {"doc": "Tests the claim that the imaginary-weight control parameter gamma drives\nthe system between GOE (orthogonal) and GUE (unitary) random matrix classes.", "kind": "class", "line": 735, "name": "Experiment2AltlandZirnbauer", "signature": "class Experiment2AltlandZirnbauer(IExperiment)"}, {"doc": "Tests the claim that the network learns the underlying conformal operator\nby applying Moebius transformations to inputs and measuring equivariance.", "kind": "class", "line": 820, "name": "Experiment3ConformalIsomorphism", "signature": "class Experiment3ConformalIsomorphism(IExperiment)"}, {"doc": "Tests the thermodynamic uncertainty relation between synthetic Planck\nconstant and superposition metric across a sweep of batch sizes and\nweight decay values.", "kind": "class", "line": 886, "name": "Experiment4CompressionFrontier", "signature": "class Experiment4CompressionFrontier(IExperiment)"}, {"doc": "Abstract base for structured pruning strategies.", "kind": "class", "line": 965, "name": "IPruningStrategy", "signature": "class IPruningStrategy(ABC)"}, {"doc": "Prunes weights uniformly at random across all layers.", "kind": "class", "line": 973, "name": "VolumePruningStrategy", "signature": "class VolumePruningStrategy(IPruningStrategy)"}, {"doc": "Prunes weights only from tensor boundaries (slot edges).", "kind": "class", "line": 988, "name": "AreaPruningStrategy", "signature": "class AreaPruningStrategy(IPruningStrategy)"}, {"doc": "Tests whether the crystal phase encodes information on boundaries\n(area law) versus the glass phase encoding it volumetrically.", "kind": "class", "line": 1004, "name": "Experiment5HolographicPruning", "signature": "class Experiment5HolographicPruning(IExperiment)"}, {"doc": "Orchestrates the execution of all five hidden-connection experiments.", "kind": "class", "line": 1078, "name": "UnifiedSuite", "signature": "class UnifiedSuite"}, {"kind": "method", "line": 1182, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 76, "name": "__post_init__", "signature": "def __post_init__(self)"}, {"kind": "method", "line": 190, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 203, "name": "forward", "signature": "def forward(self, A, B)"}, {"kind": "method", "line": 213, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 216, "name": "get_flat_parameters", "signature": "def get_flat_parameters(self)"}, {"kind": "method", "line": 219, "name": "slot_importance", "signature": "def slot_importance(self)"}, {"kind": "method", "line": 225, "name": "count_active_slots", "signature": "def count_active_slots(self, threshold)"}, {"kind": "method", "line": 232, "name": "generate_batch", "signature": "def generate_batch(self, batch_size)"}, {"kind": "method", "line": 238, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 241, "name": "generate_batch", "signature": "def generate_batch(self, batch_size)"}, {"kind": "method", "line": 265, "name": "save", "signature": "def save(self, model, epoch, metrics, path)"}, {"kind": "method", "line": 266, "name": "load", "signature": "def load(self, path, model)"}, {"kind": "method", "line": 272, "name": "save", "signature": "def save(self, model, epoch, metrics, path)"}, {"kind": "method", "line": 284, "name": "load", "signature": "def load(self, path, model)"}, {"kind": "method", "line": 296, "name": "train", "signature": "def train(self, model, epochs, callback)"}, {"kind": "method", "line": 304, "name": "__init__", "signature": "def __init__(self, model_config, training_config, data_generator)"}, {"kind": "method", "line": 314, "name": "train", "signature": "def train(self, model, epochs, callback)"}, {"kind": "method", "line": 361, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 368, "name": "__init__", "signature": "def __init__(self, config, tolerance)"}, {"kind": "method", "line": 372, "name": "_build_hessian_approximation", "signature": "def _build_hessian_approximation(self, model)"}, {"kind": "method", "line": 384, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 418, "name": "__init__", "signature": "def __init__(self, config, regularization)"}, {"kind": "method", "line": 422, "name": "_compute_hessian", "signature": "def _compute_hessian(self, model)"}, {"kind": "method", "line": 433, "name": "_generate_single_sample", "signature": "def _generate_single_sample(self)"}, {"kind": "method", "line": 437, "name": "_loss_from_flat", "signature": "def _loss_from_flat(self, flat_params, model, original_params, A, B, C_true)"}, {"kind": "method", "line": 456, "name": "_diagonal_hessian_approximation", "signature": "def _diagonal_hessian_approximation(self, model, A, B, C_true)"}, {"kind": "method", "line": 470, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 495, "name": "__init__", "signature": "def __init__(self, config, noise_floor)"}, {"kind": "method", "line": 499, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 555, "name": "__init__", "signature": "def __init__(self, model_config, expansion_factor, l1_coefficient, sae_lr, sae_epochs, sae_batch_size, num_samples, epsilon)"}, {"kind": "method", "line": 576, "name": "_extract_activations", "signature": "def _extract_activations(self, model)"}, {"kind": "method", "line": 589, "name": "_train_sae", "signature": "def _train_sae(self, activations)"}, {"kind": "method", "line": 616, "name": "_sae_forward", "signature": "def _sae_forward(self, x, W_enc, b_enc, b_dec)"}, {"kind": "method", "line": 623, "name": "calculate", "signature": "def calculate(self, model)"}, {"kind": "method", "line": 649, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 653, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 663, "name": "__init__", "signature": "def __init__(self, config, model_config, training_config, data_generator, checkpoint_manager)"}, {"kind": "method", "line": 679, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 682, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 717, "name": "_analyze_temporal_correlation", "signature": "def _analyze_temporal_correlation(self, results)"}, {"kind": "method", "line": 741, "name": "__init__", "signature": "def __init__(self, config, model_config, data_generator)"}, {"kind": "method", "line": 752, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 755, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 778, "name": "_create_gamma_model", "signature": "def _create_gamma_model(self, base_model, gamma)"}, {"kind": "method", "line": 790, "name": "_train_gamma_model", "signature": "def _train_gamma_model(self, model)"}, {"kind": "method", "line": 800, "name": "_detect_critical_transition", "signature": "def _detect_critical_transition(self, results)"}, {"kind": "method", "line": 826, "name": "__init__", "signature": "def __init__(self, config, model_config, data_generator)"}, {"kind": "method", "line": 836, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 839, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 869, "name": "_apply_moebius", "signature": "def _apply_moebius(self, A, B)"}, {"kind": "method", "line": 879, "name": "_apply_moebius_to_output", "signature": "def _apply_moebius_to_output(self, C)"}, {"kind": "method", "line": 893, "name": "__init__", "signature": "def __init__(self, config, model_config, data_generator)"}, {"kind": "method", "line": 904, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 907, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 951, "name": "_test_uncertainty_bound", "signature": "def _test_uncertainty_bound(self, results)"}, {"kind": "method", "line": 969, "name": "prune", "signature": "def prune(self, model, fraction)"}, {"kind": "method", "line": 976, "name": "prune", "signature": "def prune(self, model, fraction)"}, {"kind": "method", "line": 991, "name": "prune", "signature": "def prune(self, model, fraction)"}, {"kind": "method", "line": 1010, "name": "__init__", "signature": "def __init__(self, config, model_config, data_generator)"}, {"kind": "method", "line": 1022, "name": "get_name", "signature": "def get_name(self)"}, {"kind": "method", "line": 1025, "name": "run", "signature": "def run(self, model)"}, {"kind": "method", "line": 1046, "name": "_run_pruning_trials", "signature": "def _run_pruning_trials(self, base_model, pruner, A, B, C_true)"}, {"kind": "method", "line": 1081, "name": "__init__", "signature": "def __init__(self, config)"}, {"kind": "method", "line": 1088, "name": "_build_experiments", "signature": "def _build_experiments(self)"}, {"kind": "method", "line": 1127, "name": "run_all", "signature": "def run_all(self)"}, {"kind": "method", "line": 1150, "name": "_aggregate_verdicts", "signature": "def _aggregate_verdicts(self, all_results)"}, {"kind": "method", "line": 1163, "name": "_serialize_config", "signature": "def _serialize_config(self)"}, {"kind": "method", "line": 691, "name": "checkpoint_callback", "signature": "def checkpoint_callback(epoch, m, loss, acc)"}]}, {"id": "xray_tensor_diffractometer.py", "kind": "module", "label": "xray_tensor_diffractometer.py", "language": "py", "sha256": "348313ca85c3ac55", "symbol_count": 132, "symbols": [{"kind": "class", "line": 27, "name": "Config", "signature": "class Config"}, {"kind": "method", "line": 64, "name": "set_seed", "signature": "def set_seed(seed)"}, {"kind": "method", "line": 72, "name": "setup_logger", "signature": "def setup_logger(name, level)"}, {"doc": "Pipeline automático: encuentra el mejor cristal y lo usa como semilla.", "kind": "method", "line": 86, "name": "run_epitaxy_from_best_crystal", "signature": "def run_epitaxy_from_best_crystal(checkpoint_dir, target_sizes)"}, {"kind": "class", "line": 146, "name": "ICheckpointLoader", "signature": "class ICheckpointLoader(Protocol)"}, {"kind": "class", "line": 149, "name": "IMetricsCalculator", "signature": "class IMetricsCalculator(Protocol)"}, {"kind": "class", "line": 152, "name": "IDataGenerator", "signature": "class IDataGenerator(Protocol)"}, {"kind": "class", "line": 155, "name": "CheckpointLoadingError", "signature": "class CheckpointLoadingError(Exception)"}, {"kind": "class", "line": 158, "name": "MetricsComputationError", "signature": "class MetricsComputationError(Exception)"}, {"kind": "class", "line": 161, "name": "TrainingError", "signature": "class TrainingError(Exception)"}, {"kind": "class", "line": 166, "name": "StrassenDataGenerator", "signature": "class StrassenDataGenerator"}, {"kind": "class", "line": 187, "name": "BilinearStrassenModel", "signature": "class BilinearStrassenModel(Module)"}, {"doc": "Motor de crecimiento epitaxial para cristales algorítmicos.\n\nFÍSICA: Imita el crecimiento de cristales en sustratos donde la estructura\natómica del sustrato guía la formación del nuevo cristal.", "kind": "class", "line": 216, "name": "EpitaxialGrowthEngine", "signature": "class EpitaxialGrowthEngine"}, {"doc": "Experimento completo de epitaxia: sembrar, crecer, analizar.", "kind": "class", "line": 469, "name": "EpitaxyExperiment", "signature": "class EpitaxyExperiment"}, {"doc": "Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C", "kind": "class", "line": 671, "name": "ThermodynamicPotential", "signature": "class ThermodynamicPotential"}, {"kind": "class", "line": 693, "name": "SpectroscopyMetrics", "signature": "class SpectroscopyMetrics"}, {"kind": "class", "line": 913, "name": "ThermodynamicMetrics", "signature": "class ThermodynamicMetrics"}, {"kind": "class", "line": 1158, "name": "CrystallographyMetrics", "signature": "class CrystallographyMetrics"}, {"doc": "🐄 Green's Cow: Uses integration-by-parts analogy to split gradient into bulk and boundary terms.\nInspired by Green's identities: ∫_Ω ∇u·v = ∫_∂Ω u(v·n) - ∫_Ω u∇·v\nApplied to weight tensors as discrete manifolds.", "kind": "class", "line": 1264, "name": "GreenCowExperiment", "signature": "class GreenCowExperiment"}, {"kind": "class", "line": 1371, "name": "CheckpointLoader", "signature": "class CheckpointLoader"}, {"kind": "class", "line": 1405, "name": "CheckpointMigrator", "signature": "class CheckpointMigrator"}, {"kind": "class", "line": 1490, "name": "BoltzmannAnalysisProgram", "signature": "class BoltzmannAnalysisProgram"}, {"kind": "class", "line": 2610, "name": "StrassenCrystallographer", "signature": "class StrassenCrystallographer"}, {"kind": "method", "line": 2683, "name": "main", "signature": "def main()"}, {"kind": "method", "line": 147, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path, device)"}, {"kind": "method", "line": 150, "name": "compute", "signature": "def compute(self, model)"}, {"kind": "method", "line": 153, "name": "generate_batch", "signature": "def generate_batch(self, batch_size)"}, {"kind": "method", "line": 168, "name": "generate_batch", "signature": "def generate_batch(batch_size)"}, {"kind": "method", "line": 179, "name": "verify_structure", "signature": "def verify_structure(coeffs)"}, {"kind": "method", "line": 188, "name": "__init__", "signature": "def __init__(self, hidden_dim, matrix_size)"}, {"kind": "method", "line": 199, "name": "_initialize_symmetric", "signature": "def _initialize_symmetric(self)"}, {"kind": "method", "line": 204, "name": "forward", "signature": "def forward(self, a, b)"}, {"kind": "method", "line": 207, "name": "get_coefficients", "signature": "def get_coefficients(self)"}, {"kind": "method", "line": 224, "name": "__init__", "signature": "def __init__(self, seed_checkpoint_path, target_matrix_size, device)"}, {"doc": "Carga el cristal semilla verificando su pureza", "kind": "method", "line": 242, "name": "_load_seed_crystal", "signature": "def _load_seed_crystal(self)"}, {"doc": "Crece un cristal epitaxial desde la semilla.\n\nMÉTODO: Kronecker product preserva la estructura periódica:\nSi A es cristal Strassen de 2x2, entonces A ⊗ I_n es cristal de (2n)x(2n)", "kind": "method", "line": 265, "name": "grow_epitaxial_crystal", "signature": "def grow_epitaxial_crystal(self)"}, {"doc": "Ajusta dimensiones del tensor epitaxial para coincidir con el modelo objetivo.\nRellena con ruido térmico pequeño o trunca según sea necesario.", "kind": "method", "line": 326, "name": "_adjust_dimensions", "signature": "def _adjust_dimensions(self, tensor, target_shape)"}, {"doc": "Recocido térmico del cristal epitaxial.\n\nFÍSICA: En lugar de \"entrenar desde cero\", aplicamos temperatura decreciente\npara que el cristal se auto-organice alrededor de la semilla.", "kind": "method", "line": 360, "name": "anneal_crystal", "signature": "def anneal_crystal(self, model, max_epochs, early_stop_threshold)"}, {"kind": "method", "line": 474, "name": "__init__", "signature": "def __init__(self, results_dir)"}, {"doc": "Experimento completo: cultiva cristales de múltiples tamaños desde una semilla.", "kind": "method", "line": 479, "name": "run_epitaxial_growth_experiment", "signature": "def run_epitaxial_growth_experiment(self, seed_checkpoint, target_sizes)"}, {"doc": "Visualiza la evolución del cristal durante el recocido", "kind": "method", "line": 556, "name": "_plot_epitaxial_evolution", "signature": "def _plot_epitaxial_evolution(self, annealing_results, target_size, seed_name)"}, {"doc": "Genera reporte comparativo de todos los experimentos epitaxiales", "kind": "method", "line": 604, "name": "_generate_comparative_report", "signature": "def _generate_comparative_report(self, results)"}, {"doc": "F = U - T*S (a μ y N constantes)", "kind": "method", "line": 680, "name": "helmholtz_free_energy", "signature": "def helmholtz_free_energy(self)"}, {"doc": "G = F + μ*N + P*V (presión algorítmica)", "kind": "method", "line": 684, "name": "gibbs_free_energy", "signature": "def gibbs_free_energy(self)"}, {"doc": "Criterio de estabilidad: dG < 0", "kind": "method", "line": 689, "name": "is_stable", "signature": "def is_stable(self)"}, {"kind": "method", "line": 696, "name": "compute_weight_diffraction", "signature": "def compute_weight_diffraction(coeffs)"}, {"kind": "method", "line": 719, "name": "_compute_spectral_entropy", "signature": "def _compute_spectral_entropy(power_spectrum)"}, {"doc": "Extrae parámetros de red preservando la geometría física del tensor.\n\nFIX: En lugar de reshape arbitrario, aplicamos SVD sobre la matriz \nde covarianza que preserva la estructura de correlaciones.", "kind": "method", "line": 726, "name": "extract_lattice_parameters", "signature": "def extract_lattice_parameters(weight_tensor, rank)"}, {"kind": "method", "line": 785, "name": "compute_gibbs_free_energy", "signature": "def compute_gibbs_free_energy(loss, temp, entropy)"}, {"doc": "Descomposición Canónica del tensor tripartito (U, V, W).\n\nFIX: Preserva la estructura bilineal en lugar de tratar como matriz plana.\nAplicamos HOSVD (Higher-Order SVD) para tensores de orden 3.", "kind": "method", "line": 791, "name": "extract_canonical_decomposition", "signature": "def extract_canonical_decomposition(coeffs, rank)"}, {"doc": "Proyecta factores continuos a la red cristalina discreta {-1, 0, 1}.", "kind": "method", "line": 847, "name": "_discretize_to_integers", "signature": "def _discretize_to_integers(factors)"}, {"doc": "Verifica si los factores discretizados corresponden a la estructura de Strassen.", "kind": "method", "line": 868, "name": "_check_strassen_equivalence", "signature": "def _check_strassen_equivalence(discretized_factors)"}, {"kind": "method", "line": 892, "name": "create_superlattice_seed", "signature": "def create_superlattice_seed(base_tensor, scale_factor)"}, {"kind": "method", "line": 916, "name": "compute_effective_temperature", "signature": "def compute_effective_temperature(gradient_buffer, learning_rate)"}, {"doc": "Calcula exponentes críticos cerca de transiciones de fase.\n\nLeyes de escala:\n- C_v ~ |T - T_c|^{-α_exp}  (calor específico)\n- ξ ~ |T - T_c|^{-ν}        (longitud de correlación)\n- τ ~ |T - T_c|^{-z}        (tiempo de grokking)", "kind": "method", "line": 930, "name": "compute_critical_exponents", "signature": "def compute_critical_exponents(temp_history, cv_history, alpha_history)"}, {"doc": "Ecuación de estado: T_c(α) = T_0 * exp(-c*α)\n\nFIX: Relación constitutiva que describe la curva de coexistencia cristal-vidrio.", "kind": "method", "line": 1009, "name": "compute_equation_of_state", "signature": "def compute_equation_of_state(temp_eff, alpha, kappa)"}, {"kind": "method", "line": 1048, "name": "compute_specific_heat", "signature": "def compute_specific_heat(loss_history, temp_history, cv_threshold)"}, {"kind": "method", "line": 1061, "name": "estimate_hbar_algorithmic", "signature": "def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)"}, {"kind": "method", "line": 1069, "name": "compute_mutual_information", "signature": "def compute_mutual_information(weights, gradients)"}, {"kind": "method", "line": 1083, "name": "check_extensivity", "signature": "def check_extensivity(entropy_list, scale_factors)"}, {"kind": "method", "line": 1107, "name": "compute_fisher_information_matrix", "signature": "def compute_fisher_information_matrix(model, samples)"}, {"kind": "method", "line": 1126, "name": "compute_ricci_curvature", "signature": "def compute_ricci_curvature(fisher_matrix)"}, {"kind": "method", "line": 1137, "name": "calculate_carnot_efficiency", "signature": "def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)"}, {"kind": "method", "line": 1161, "name": "compute_kappa", "signature": "def compute_kappa(model, dataloader, num_batches)"}, {"kind": "method", "line": 1187, "name": "compute_discretization_margin", "signature": "def compute_discretization_margin(coeffs)"}, {"kind": "method", "line": 1191, "name": "compute_local_complexity", "signature": "def compute_local_complexity(model)"}, {"kind": "method", "line": 1200, "name": "compute_alpha_purity", "signature": "def compute_alpha_purity(coeffs)"}, {"kind": "method", "line": 1207, "name": "compute_kappa_quantum", "signature": "def compute_kappa_quantum(coeffs, hbar)"}, {"kind": "method", "line": 1224, "name": "compute_poynting_vector", "signature": "def compute_poynting_vector(coeffs)"}, {"kind": "method", "line": 1246, "name": "compute_all_metrics", "signature": "def compute_all_metrics(model, dataloader)"}, {"kind": "method", "line": 1270, "name": "__init__", "signature": "def __init__(self, model, device)"}, {"doc": "Approximate surface term: gradient concentrated on tensor boundaries.\nFor a matrix W ∈ ℝ^{m×n}, boundary = first/last row + first/last column.", "kind": "method", "line": 1275, "name": "compute_boundary_gradient", "signature": "def compute_boundary_gradient(self, weight)"}, {"doc": "Interior (volume) term: everything except boundary.", "kind": "method", "line": 1290, "name": "compute_bulk_gradient", "signature": "def compute_bulk_gradient(self, weight)"}, {"doc": "Custom backward pass using Green-inspired decomposition.\nLoss = MSE + λ_boundary * ||boundary_grad||²", "kind": "method", "line": 1296, "name": "run_green_backprop_step", "signature": "def run_green_backprop_step(self, A, B, C_true, lambda_boundary)"}, {"doc": "Returns a binary mask marking boundary elements of a tensor.", "kind": "method", "line": 1329, "name": "_get_boundary_mask", "signature": "def _get_boundary_mask(self, weight)"}, {"kind": "method", "line": 1341, "name": "train_with_green_cow", "signature": "def train_with_green_cow(self, epochs, lr, lambda_boundary)"}, {"doc": "Load checkpoint with robust deserialization handling.\nInjects Config as UnifiedConfig alias to handle cross-script compatibility.", "kind": "method", "line": 1372, "name": "load_checkpoint", "signature": "def load_checkpoint(self, path, device)"}, {"doc": "Migrate checkpoint to standard format, extracting config if present.\nReturns migrated state_dict compatible with BilinearStrassenModel.", "kind": "method", "line": 1407, "name": "migrate_checkpoint", "signature": "def migrate_checkpoint(raw_data)"}, {"kind": "method", "line": 1431, "name": "_migrate_dict", "signature": "def _migrate_dict(state_dict)"}, {"kind": "method", "line": 1443, "name": "_migrate_custom_format", "signature": "def _migrate_custom_format(state_dict)"}, {"kind": "method", "line": 1467, "name": "_migrate_coefs_format", "signature": "def _migrate_coefs_format(state_dict)"}, {"kind": "method", "line": 1475, "name": "_migrate_encoder_format", "signature": "def _migrate_encoder_format(state_dict)"}, {"kind": "method", "line": 1487, "name": "_migrate_standard_format", "signature": "def _migrate_standard_format(state_dict)"}, {"kind": "method", "line": 1491, "name": "__init__", "signature": "def __init__(self, checkpoint_dir, results_dir)"}, {"kind": "method", "line": 1503, "name": "_load_all_checkpoints", "signature": "def _load_all_checkpoints(self)"}, {"kind": "method", "line": 1548, "name": "run_full_boltzmann_program", "signature": "def run_full_boltzmann_program(self)"}, {"kind": "method", "line": 1575, "name": "phase1_molecular_hypothesis", "signature": "def phase1_molecular_hypothesis(self)"}, {"kind": "method", "line": 1660, "name": "phase2_entropy_production", "signature": "def phase2_entropy_production(self)"}, {"kind": "method", "line": 1742, "name": "phase3_extensivity_law", "signature": "def phase3_extensivity_law(self)"}, {"kind": "method", "line": 1796, "name": "phase4_quantum_basis_transform", "signature": "def phase4_quantum_basis_transform(self)"}, {"kind": "method", "line": 1849, "name": "analyze_poynting_flow", "signature": "def analyze_poynting_flow(self)"}, {"doc": "PHASE 5: THERMODYNAMIC ANALYSIS con exponentes críticos y ecuación de estado.\n\nFIX: Ahora calcula exponentes críticos y ecuación de estado para cada checkpoint.", "kind": "method", "line": 1882, "name": "phase5_thermodynamic_analysis", "signature": "def phase5_thermodynamic_analysis(self)"}, {"kind": "method", "line": 2011, "name": "phase6_spectroscopic_analysis", "signature": "def phase6_spectroscopic_analysis(self)"}, {"kind": "method", "line": 2097, "name": "_plot_diffraction_pattern", "signature": "def _plot_diffraction_pattern(self, diffraction_data, ckpt_name)"}, {"kind": "method", "line": 2131, "name": "_save_superlattice_seed", "signature": "def _save_superlattice_seed(self, superlattice, ckpt_name)"}, {"kind": "method", "line": 2150, "name": "_classify_thermodynamic_phase", "signature": "def _classify_thermodynamic_phase(self, t_eff, cv, alpha)"}, {"kind": "method", "line": 2162, "name": "_estimate_critical_temperature", "signature": "def _estimate_critical_temperature(self, results)"}, {"kind": "method", "line": 2174, "name": "_verify_entropy_extensivity", "signature": "def _verify_entropy_extensivity(self, results)"}, {"kind": "method", "line": 2188, "name": "_plot_phase_diagram", "signature": "def _plot_phase_diagram(self, results)"}, {"kind": "method", "line": 2213, "name": "_plot_temperature_vs_purity", "signature": "def _plot_temperature_vs_purity(self, results)"}, {"kind": "method", "line": 2239, "name": "_compute_entropy_simple", "signature": "def _compute_entropy_simple(self, params)"}, {"kind": "method", "line": 2265, "name": "_compute_entropy", "signature": "def _compute_entropy(self, params)"}, {"kind": "method", "line": 2304, "name": "_compute_effective_volume", "signature": "def _compute_effective_volume(self, params)"}, {"kind": "method", "line": 2322, "name": "_plot_parameter_distribution", "signature": "def _plot_parameter_distribution(self, params, group_name, kde)"}, {"kind": "method", "line": 2350, "name": "_simulate_training_trajectory", "signature": "def _simulate_training_trajectory(self, final_params, final_delta)"}, {"kind": "method", "line": 2361, "name": "_compute_generalization_entropy", "signature": "def _compute_generalization_entropy(self, params, successful_ckpts)"}, {"kind": "method", "line": 2414, "name": "_fit_timescale", "signature": "def _fit_timescale(self, entropy_values)"}, {"kind": "method", "line": 2424, "name": "_plot_entropy_production", "signature": "def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)"}, {"kind": "method", "line": 2442, "name": "_verify_scaling", "signature": "def _verify_scaling(self, coeffs, N)"}, {"kind": "method", "line": 2453, "name": "_recursive_strassen", "signature": "def _recursive_strassen(self, A, B, coeffs, N)"}, {"kind": "method", "line": 2487, "name": "_fit_extensivity", "signature": "def _fit_extensivity(self, errors, sizes, purity)"}, {"kind": "method", "line": 2501, "name": "_verify_extensivity_universality", "signature": "def _verify_extensivity_universality(self, results)"}, {"kind": "method", "line": 2505, "name": "_plot_extensivity", "signature": "def _plot_extensivity(self, sizes, errors, purity, ckpt_name)"}, {"kind": "method", "line": 2518, "name": "_find_broken_symmetries", "signature": "def _find_broken_symmetries(self, coeffs)"}, {"kind": "method", "line": 2526, "name": "_measure_uncertainty", "signature": "def _measure_uncertainty(self, coeffs, basis)"}, {"kind": "method", "line": 2537, "name": "_plot_uncertainty_distribution", "signature": "def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)"}, {"kind": "method", "line": 2558, "name": "_print_executive_summary", "signature": "def _print_executive_summary(self, results)"}, {"kind": "method", "line": 2587, "name": "_save_results", "signature": "def _save_results(self, results, filename)"}, {"kind": "method", "line": 2611, "name": "__init__", "signature": "def __init__(self, checkpoint_path, device)"}, {"kind": "method", "line": 2617, "name": "_load_model", "signature": "def _load_model(self, path, device)"}, {"kind": "method", "line": 2634, "name": "run_full_analysis", "signature": "def run_full_analysis(self)"}, {"kind": "method", "line": 2656, "name": "_assign_grade", "signature": "def _assign_grade(self, delta, alpha)"}, {"kind": "method", "line": 2668, "name": "_save_report", "signature": "def _save_report(self, report)"}, {"kind": "method", "line": 373, "name": "generate_batch", "signature": "def generate_batch(batch_size)"}, {"kind": "method", "line": 1444, "name": "get_tensor", "signature": "def get_tensor(key)"}, {"kind": "method", "line": 2415, "name": "model", "signature": "def model(t, A, tau, C)"}, {"kind": "method", "line": 2488, "name": "model", "signature": "def model(N, alpha, beta)"}, {"kind": "method", "line": 2590, "name": "convert_to_serializable", "signature": "def convert_to_serializable(obj)"}, {"kind": "method", "line": 2637, "name": "dataloader", "signature": "def dataloader()"}, {"kind": "method", "line": 1906, "name": "sample_dataloader", "signature": "def sample_dataloader()"}, {"kind": "method", "line": 2039, "name": "sample_dataloader", "signature": "def sample_dataloader()"}, {"kind": "method", "line": 1522, "name": "dataloader", "signature": "def dataloader()"}]}], "type": "CodePropertyGraph", "version": "1.0"}
```

---

## Architecture Reference

### C (3 files)

#### `strassen_c.c`
**Path:** `src/native/strassen_c.c`

**Functions:**
- `alloc_matrix` (line 15) `static float* alloc_matrix(int n)` - *Strassen Matrix Multiplication - C Implementation Author: grisun0  Compila: gcc -O3 -ffast-math -march=native -shared -fPIC -o libstrassen.so strassen_c.c  #include <stdlib.h> #include <string.h> #include <stdio.h> #define THRESHOLD 64 /* Allocate matrix*
- `matmul_standard` (line 20) `static void matmul_standard(float* C, float* A, float* B, int n)` - *#include <stdlib.h> #include <string.h> #include <stdio.h> #define THRESHOLD 64 /* Allocate matrix static float* alloc_matrix(int n) { return (float*)aligned_alloc(32, n * n * sizeof(float)); } /* Standard matrix multiplication for small matrices*
- `mat_add` (line 33) `static void mat_add(float* C, float* A, float* B, int n)` - */* Standard matrix multiplication for small matrices static void matmul_standard(float* C, float* A, float* B, int n) { memset(C, 0, n * n * sizeof(float)); for (int i = 0; i < n; i++) { for (int k = 0; k < n; k++) { float a_ik = A[i * n + k]; for (int j = 0; j < n; j++) { C[i * n + j] += a_ik * B[k * n + j]; } } } } /* Add matrices: C = A + B*
- `mat_sub` (line 41) `static void mat_sub(float* C, float* A, float* B, int n)` - *} } } } /* Add matrices: C = A + B static void mat_add(float* C, float* A, float* B, int n) { int nn = n * n; for (int i = 0; i < nn; i++) { C[i] = A[i] + B[i]; } } /* Subtract matrices: C = A - B*
- `extract_quadrant` (line 49) `static void extract_quadrant(float* Q, float* M, int n, int row, int col)` - *for (int i = 0; i < nn; i++) { C[i] = A[i] + B[i]; } } /* Subtract matrices: C = A - B static void mat_sub(float* C, float* A, float* B, int n) { int nn = n * n; for (int i = 0; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Extract quadrant from matrix*
- `insert_quadrant` (line 57) `static void insert_quadrant(float* M, float* Q, int n, int row, int col)` - *for (int i = 0; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Extract quadrant from matrix static void extract_quadrant(float* Q, float* M, int n, int row, int col) { int h = n / 2; for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant into matrix*
- `strassen_recursive` (line 65) `void strassen_recursive(float* C, float* A, float* B, int n)` - *for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant into matrix static void insert_quadrant(float* M, float* Q, int n, int row, int col) { int h = n / 2; for (int i = 0; i < h; i++) { memcpy(&M[(row + i) * n + col], &Q[i * h], h * sizeof(float)); } } /* Strassen recursive*
- `strassen_multiply` (line 164) `void strassen_multiply(float* C, float* A, float* B, int n)` - *insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C, C22, n, h, h); /* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7); free(T1); free(T2); free(C11); free(C12); free(C21); free(C22); } /* Public API*
- `standard_multiply` (line 169) `void standard_multiply(float* C, float* A, float* B, int n)` - */* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7); free(T1); free(T2); free(C11); free(C12); free(C21); free(C22); } /* Public API void strassen_multiply(float* C, float* A, float* B, int n) { strassen_recursive(C, A, B, n); } /* Standard multiply for comparison*

**Macros:**
- `THRESHOLD` (line 11)

#### `strassen_optimal.c`
**Path:** `src/native/strassen_optimal.c`

**Functions:**
- `strassen_level` (line 18) `static void strassen_level(float* C, float* A, float* B, int n, 
                           float...` - *Uses in-place operations where possible and only applies Strassen for very large matrices where the asymptotic advantage overcomes overhead.  #include <stdlib.h> #include <string.h> #include <stdio.h> #include <cblas.h> /* Only use Strassen for huge matrices where O(n^2.807) wins #define STRASSEN_THRESHOLD 4096 /* Strassen for matrices >= threshold*
- `strassen_optimal` (line 130) `void strassen_optimal(float* C, float* A, float* B, int n)`

**Macros:**
- `STRASSEN_THRESHOLD` (line 15)

#### `strassen_turbo.c`
**Path:** `src/native/strassen_turbo.c`

**Functions:**
- `alloc_matrix` (line 25) `static inline float* alloc_matrix(int n)` - *Compile: gcc -O3 -ffast-math -march=native -fopenmp -mavx2 -shared -fPIC -o libstrassen_turbo.so strassen_turbo.c  #include <stdlib.h> #include <string.h> #include <stdio.h> #include <omp.h> #include <immintrin.h> #define THRESHOLD 128 #define BLOCK_SIZE 32 #define ALIGN 32 /* Aligned allocation*
- `mat_add_avx` (line 30) `static void mat_add_avx(float* __restrict C, const float* __restrict A, 
                        ...` - *#include <stdio.h> #include <omp.h> #include <immintrin.h> #define THRESHOLD 128 #define BLOCK_SIZE 32 #define ALIGN 32 /* Aligned allocation static inline float* alloc_matrix(int n) { return (float*)aligned_alloc(ALIGN, n * n * sizeof(float)); } /* AVX2 vectorized matrix addition: C = A + B*
- `mat_sub_avx` (line 50) `static void mat_sub_avx(float* __restrict C, const float* __restrict A, 
                        ...` - *for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_add_ps(va, vb); _mm256_store_ps(&C[i], vc); } /* Handle remainder for (; i < nn; i++) { C[i] = A[i] + B[i]; } } /* AVX2 vectorized matrix subtraction: C = A - B*
- `matmul_blocked_avx` (line 68) `static void matmul_blocked_avx(float* __restrict C, const float* __restrict A, 
                 ...` - *for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_sub_ps(va, vb); _mm256_store_ps(&C[i], vc); } for (; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Cache-blocked matrix multiplication with AVX2*
- `extract_quadrant` (line 104) `static void extract_quadrant(float* __restrict Q, const float* __restrict M, 
                   ...` - *_mm256_storeu_ps(&C[i * n + j], vc); } for (; j < j_end; j++) { C[i * n + j] += a_ik * B[k * n + j]; } } } } } } } /* Extract quadrant*
- `insert_quadrant` (line 114) `static void insert_quadrant(float* __restrict M, const float* __restrict Q, 
                    ...` - *} } /* Extract quadrant static void extract_quadrant(float* __restrict Q, const float* __restrict M, int n, int row, int col) { int h = n / 2; #pragma omp parallel for if(h > 64) for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant*
- `strassen_turbo_recursive` (line 124) `void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)` - *} } /* Insert quadrant static void insert_quadrant(float* __restrict M, const float* __restrict Q, int n, int row, int col) { int h = n / 2; #pragma omp parallel for if(h > 64) for (int i = 0; i < h; i++) { memcpy(&M[(row + i) * n + col], &Q[i * h], h * sizeof(float)); } } /* Strassen recursive with parallelism*
- `strassen_turbo` (line 261) `void strassen_turbo(float* C, float* A, float* B, int n)` - *insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C, C22, n, h, h); /* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7); free(T1); free(T2); free(C11); free(C12); free(C21); free(C22); } /* Public API*
- `get_num_threads` (line 267) `int get_num_threads(void)` - *free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6); free(M7); free(T1); free(T2); free(C11); free(C12); free(C21); free(C22); } /* Public API void strassen_turbo(float* C, float* A, float* B, int n) { omp_set_num_threads(omp_get_max_threads()); strassen_turbo_recursive(C, A, B, n, 0); } /* Get number of threads*

**Macros:**
- `THRESHOLD` (line 19)
- `BLOCK_SIZE` (line 21)
- `ALIGN` (line 22)

### PY (56 files)

#### `app.py`
**Path:** `app.py`
**File Doc:** *_*_ coding: utf8 _*_*

**Classes:**
- `StrassenNet` (line 24) `class StrassenNet(Module)`

**Methods:**
- `__init__` (line 25) `def __init__(self, rank)`
- `forward` (line 31) `def forward(self, A, B)`

#### `batch_size.py`
**Path:** `batch_size.py`

**Classes:**
- `Configuration` (line 31) `class Configuration`
- `BilinearStrassenModel` (line 77) `class BilinearStrassenModel(Module)`
- `CheckpointMigrator` (line 100) `class CheckpointMigrator(ABC)`
- `CustomFormatMigrator` (line 110) `class CustomFormatMigrator(CheckpointMigrator)`
- `StandardFormatMigrator` (line 122) `class StandardFormatMigrator(CheckpointMigrator)`
- `CheckpointMigrationManager` (line 131) `class CheckpointMigrationManager`
- `StrassenDataGenerator` (line 147) `class StrassenDataGenerator`
- `CrystallographyMetrics` (line 156) `class CrystallographyMetrics`
- `PlanckConstantCalculator` (line 186) `class PlanckConstantCalculator`
- `BatchSizeThermodynamics` (line 217) `class BatchSizeThermodynamics`
- `StrassenCheckpointLoader` (line 262) `class StrassenCheckpointLoader`
- `StrassenPlanckAnalyzer` (line 292) `class StrassenPlanckAnalyzer`

**Methods:**
- `set_random_seed` (line 71) `def set_random_seed(seed)`
- `main` (line 341) `def main()`
- `__init__` (line 78) `def __init__(self, config)`
- `forward` (line 88) `def forward(self, a, b)`
- `get_coefficients` (line 91) `def get_coefficients(self)`
- `compute_lambda_effective` (line 94) `def compute_lambda_effective(self)`
- `can_migrate` (line 102) `def can_migrate(self, state_dict)`
- `migrate` (line 106) `def migrate(self, state_dict)`
- `can_migrate` (line 111) `def can_migrate(self, state_dict)`
- `migrate` (line 114) `def migrate(self, state_dict)`
- `can_migrate` (line 123) `def can_migrate(self, state_dict)`
- `migrate` (line 126) `def migrate(self, state_dict)`
- `__init__` (line 132) `def __init__(self)`
- `migrate_checkpoint` (line 135) `def migrate_checkpoint(self, path, device)`
- `generate_batch` (line 149) `def generate_batch(batch_size, config)`
- `compute_kappa` (line 158) `def compute_kappa(model, num_batches, config)`
- `compute_discretization_margin` (line 174) `def compute_discretization_margin(coeffs)`
- `compute_local_complexity` (line 178) `def compute_local_complexity(model, config)`
- `__init__` (line 187) `def __init__(self, metrics, training_metrics, config)`
- `calculate_all` (line 196) `def calculate_all(self)`
- `__init__` (line 218) `def __init__(self, model, h_bar, delta_struct, config)`
- `analyze_batch_size_spectrum` (line 224) `def analyze_batch_size_spectrum(self)`
- `_measure_gradients` (line 246) `def _measure_gradients(self, batch_size)`
- `__init__` (line 263) `def __init__(self, config)`
- `load` (line 267) `def load(self, path, device)`
- `extract_training_metrics` (line 281) `def extract_training_metrics(self, path)`
- `__init__` (line 293) `def __init__(self, config)`
- `analyze_checkpoint` (line 297) `def analyze_checkpoint(self, path, device)`
- `analyze_directory` (line 323) `def analyze_directory(self, directory, device, pattern)`

#### `boltzmann_experiments.py`
**Path:** `boltzmann_experiments.py`

**Classes:**
- `Config` (line 19) `class Config`
- `CheckpointLoadingError` (line 37) `class CheckpointLoadingError(Exception)`
- `ICheckpointLoader` (line 40) `class ICheckpointLoader(ABC)`
- `CheckpointLoader` (line 45) `class CheckpointLoader(ICheckpointLoader)`
- `CheckpointMigrator` (line 52) `class CheckpointMigrator`
- `BilinearStrassenModel` (line 118) `class BilinearStrassenModel(Module)`
- `CrystallographyMetrics` (line 137) `class CrystallographyMetrics`
- `DLProgram` (line 199) `class DLProgram`

**Methods:**
- `set_seed` (line 30) `def set_seed(seed)`
- `main` (line 935) `def main()`
- `_simulate_training_trajectory` (line 951) `def _simulate_training_trajectory(self, final_params, final_delta)`
- `_compute_generalization_entropy` (line 962) `def _compute_generalization_entropy(self, params, successful_ckpts)` - *Entropía de generalización con manejo robusto de dimensionalidad*
- `_fit_timescale` (line 1016) `def _fit_timescale(self, entropy_values)`
- `_plot_entropy_production` (line 1026) `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- `phase3_extensivity_law` (line 1044) `def phase3_extensivity_law(self)`
- `load_checkpoint` (line 42) `def load_checkpoint(self, path, device)`
- `load_checkpoint` (line 46) `def load_checkpoint(self, path, device)`
- `migrate_checkpoint` (line 54) `def migrate_checkpoint(raw_data)`
- `_format_direct_tensors` (line 70) `def _format_direct_tensors(tensor_dict)`
- `_migrate_dict` (line 92) `def _migrate_dict(state_dict)`
- `_migrate_encoder_format` (line 105) `def _migrate_encoder_format(state_dict)`
- `_migrate_coefs_format` (line 115) `def _migrate_coefs_format(state_dict)`
- `__init__` (line 119) `def __init__(self, n_slots)`
- `_initialize_symmetric` (line 126) `def _initialize_symmetric(self)`
- `forward` (line 131) `def forward(self, a, b)`
- `get_coefficients` (line 134) `def get_coefficients(self)`
- `compute_kappa` (line 139) `def compute_kappa(coeffs)` - *Classical kappa - will be inf for discrete states*
- `compute_delta` (line 156) `def compute_delta(coeffs)` - *Discretization error δ*
- `compute_local_complexity` (line 161) `def compute_local_complexity(coeffs)`
- `compute_alpha_purity` (line 169) `def compute_alpha_purity(coeffs)` - *Alpha purity: α = -log(δ), inverse temperature metric for discrete states*
- `compute_kappa_quantum` (line 178) `def compute_kappa_quantum(coeffs, hbar)` - *Quantum-regularized kappa for singular covariance states*
- `__init__` (line 200) `def __init__(self, checkpoint_dir, results_dir)`
- `_load_all_checkpoints` (line 207) `def _load_all_checkpoints(self)`
- `run_full_boltzmann_program` (line 247) `def run_full_boltzmann_program(self)`
- `_print_executive_summary` (line 267) `def _print_executive_summary(self, results)`
- `_save_results` (line 307) `def _save_results(self, results, filename)`
- `phase1_molecular_hypothesis` (line 328) `def phase1_molecular_hypothesis(self)`
- `_compute_entropy_simple` (line 435) `def _compute_entropy_simple(self, params)` - *Entropía simple sin KDE para datos de baja varianza*
- `_compute_entropy` (line 446) `def _compute_entropy(self, params)` - *Entropía con manejo robusto de covarianza*
- `_compute_effective_volume` (line 466) `def _compute_effective_volume(self, kde)`
- `_plot_parameter_distribution` (line 475) `def _plot_parameter_distribution(self, params, group_name, kde)`
- `phase2_entropy_production` (line 506) `def phase2_entropy_production(self)`
- `_simulate_training_trajectory` (line 592) `def _simulate_training_trajectory(self, final_params, final_delta)`
- `_compute_generalization_entropy` (line 604) `def _compute_generalization_entropy(self, params, successful_ckpts)` - *Entropía de generalización con manejo robusto de datos idénticos*
- `_fit_timescale` (line 691) `def _fit_timescale(self, entropy_values)`
- `_plot_entropy_production` (line 701) `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- `phase3_extensivity_law` (line 719) `def phase3_extensivity_law(self)`
- `_verify_scaling` (line 773) `def _verify_scaling(self, coeffs, N)`
- `_recursive_strassen` (line 783) `def _recursive_strassen(self, A, B, coeffs, N)`
- `_fit_extensivity` (line 812) `def _fit_extensivity(self, errors, sizes, purity)`
- `_verify_extensivity_universality` (line 824) `def _verify_extensivity_universality(self, results)`
- `_plot_extensivity` (line 828) `def _plot_extensivity(self, sizes, errors, purity, ckpt_name)`
- `phase4_quantum_basis_transform` (line 841) `def phase4_quantum_basis_transform(self)`
- `_find_broken_symmetries` (line 895) `def _find_broken_symmetries(self, coeffs)`
- `_measure_uncertainty` (line 903) `def _measure_uncertainty(self, coeffs, basis)`
- `_plot_uncertainty_distribution` (line 914) `def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)`
- `model` (line 1017) `def model(t, A, tau, C)`
- `convert_to_serializable` (line 310) `def convert_to_serializable(obj)`
- `model` (line 692) `def model(t, A, tau, C)`
- `model` (line 813) `def model(N, alpha, beta)`

#### `compute_gns_checkpoints.py`
**Path:** `compute_gns_checkpoints.py`
**File Doc:** *compute_gns_by_batch.py*

**Functions:**
- `estimate_gns` (line 11) `def estimate_gns(model, batch_size, num_batches)`
- `main` (line 37) `def main()`

#### `crystallography.py`
**Path:** `crystallography.py`

**Classes:**
- `Config` (line 25) `class Config`
- `BilinearStrassenModel` (line 44) `class BilinearStrassenModel(Module)`
- `CheckpointMigrator` (line 71) `class CheckpointMigrator`
- `StrassenDataGenerator` (line 155) `class StrassenDataGenerator`
- `SparsificationProtocol` (line 172) `class SparsificationProtocol`
- `CrystallographyMetrics` (line 206) `class CrystallographyMetrics`
- `StrassenDiffractionTest` (line 232) `class StrassenDiffractionTest`
- `BasinResilienceSpectrometer` (line 277) `class BasinResilienceSpectrometer`
- `CrystalPurityIndex` (line 345) `class CrystalPurityIndex`
- `StrassenCrystallographer` (line 416) `class StrassenCrystallographer`
- `LocalComplexity` (line 523) `class LocalComplexity`

**Methods:**
- `set_seed` (line 35) `def set_seed(seed)`
- `main` (line 542) `def main()`
- `__init__` (line 45) `def __init__(self, n_slots)`
- `_initialize_symmetric` (line 52) `def _initialize_symmetric(self)`
- `forward` (line 57) `def forward(self, a, b)`
- `get_coefficients` (line 60) `def get_coefficients(self)`
- `migrate_checkpoint` (line 73) `def migrate_checkpoint(path, device)`
- `_migrate_custom` (line 106) `def _migrate_custom(state_dict)` - *Maneja formatos custom U,V,W directos*
- `_migrate_encoder` (line 123) `def _migrate_encoder(state_dict)` - *Extracción de encoder.layers*
- `_migrate_standard` (line 147) `def _migrate_standard(state_dict)` - *Formato estándar U.weight, V.weight, W.weight*
- `generate_batch` (line 157) `def generate_batch(batch_size)`
- `verify_structure` (line 164) `def verify_structure(coeffs)`
- `__init__` (line 173) `def __init__(self, model)`
- `prune_to_target` (line 176) `def prune_to_target(self, target)`
- `discretize_weights` (line 192) `def discretize_weights(self, margin)`
- `compute_kappa` (line 208) `def compute_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin` (line 225) `def compute_discretization_margin(coeffs)`
- `__init__` (line 233) `def __init__(self, model)`
- `test_gauge_invariance` (line 236) `def test_gauge_invariance(self, n_samples)`
- `_functional_error` (line 262) `def _functional_error(self, test_coeffs)`
- `__init__` (line 278) `def __init__(self, model)`
- `measure_resilience_spectrum` (line 282) `def measure_resilience_spectrum(self, noise_levels)`
- `_test_noise_recovery` (line 293) `def _test_noise_recovery(self, sigma, n_trials)`
- `_apply_noise` (line 312) `def _apply_noise(self, sigma)`
- `_anneal_to_attractor` (line 317) `def _anneal_to_attractor(self, max_epochs)`
- `_estimate_critical_noise` (line 329) `def _estimate_critical_noise(self, results)`
- `__init__` (line 346) `def __init__(self, model, diffraction_results, resilience_results, metrics_results)`
- `compute` (line 359) `def compute(self)`
- `_assign_grade` (line 399) `def _assign_grade(self, index, delta)`
- `__init__` (line 417) `def __init__(self, checkpoint_path, device)`
- `run_full_analysis` (line 445) `def run_full_analysis(self)`
- `_save_report` (line 506) `def _save_report(self, report)`
- `compute` (line 525) `def compute(model)` - *Computa LC basado en Can't Stop Won't Stop paper*
- `dataloader_gen` (line 465) `def dataloader_gen()`

#### `dirac_polos_zeros.py`
**Path:** `dirac_polos_zeros.py`

**Classes:**
- `AnalysisConfig` (line 24) `class AnalysisConfig`
- `IModel` (line 54) `class IModel(Protocol)`
- `IChargeDistributionExtractor` (line 60) `class IChargeDistributionExtractor(Protocol)`
- `IDiracAnalyzer` (line 65) `class IDiracAnalyzer(Protocol)`
- `IFieldCalculator` (line 70) `class IFieldCalculator(Protocol)`
- `IFluxCalculator` (line 75) `class IFluxCalculator(Protocol)`
- `IStateSpaceExtractor` (line 80) `class IStateSpaceExtractor(Protocol)`
- `ITransferFunctionComputer` (line 85) `class ITransferFunctionComputer(Protocol)`
- `IPoleZeroAnalyzer` (line 90) `class IPoleZeroAnalyzer(Protocol)`
- `IFrequencyAnalyzer` (line 97) `class IFrequencyAnalyzer(Protocol)`
- `ITimeResponseAnalyzer` (line 104) `class ITimeResponseAnalyzer(Protocol)`
- `ICheckpointLoader` (line 110) `class ICheckpointLoader(Protocol)`
- `ICheckpointMigrator` (line 115) `class ICheckpointMigrator(Protocol)`
- `IVisualizer` (line 120) `class IVisualizer(Protocol)`
- `BilinearModel` (line 124) `class BilinearModel(Module)`
- `ChargeDistributionExtractor` (line 152) `class ChargeDistributionExtractor`
- `DiracDeltaAnalyzer` (line 160) `class DiracDeltaAnalyzer`
- `ElectricFieldCalculator` (line 192) `class ElectricFieldCalculator`
- `ElectricFluxCalculator` (line 225) `class ElectricFluxCalculator`
- `DivergenceCalculator` (line 248) `class DivergenceCalculator`
- `GaussLawVerifier` (line 253) `class GaussLawVerifier`
- `StateSpaceExtractor` (line 271) `class StateSpaceExtractor`
- `TransferFunctionComputer` (line 298) `class TransferFunctionComputer`
- `PoleZeroAnalyzer` (line 313) `class PoleZeroAnalyzer`
- `FrequencyResponseAnalyzer` (line 463) `class FrequencyResponseAnalyzer`
- `TimeResponseAnalyzer` (line 566) `class TimeResponseAnalyzer`
- `CheckpointLoader` (line 653) `class CheckpointLoader`
- `CheckpointMigrator` (line 661) `class CheckpointMigrator`
- `ChargeDistributionVisualizer` (line 710) `class ChargeDistributionVisualizer`
- `ElectricFieldVisualizer` (line 738) `class ElectricFieldVisualizer`
- `DivergenceVisualizer` (line 780) `class DivergenceVisualizer`
- `PoleZeroVisualizer` (line 809) `class PoleZeroVisualizer`
- `BodeVisualizer` (line 848) `class BodeVisualizer`
- `NyquistVisualizer` (line 899) `class NyquistVisualizer`
- `TimeResponseVisualizer` (line 936) `class TimeResponseVisualizer`
- `CombinedVisualizer` (line 966) `class CombinedVisualizer`
- `SystemAnalyzer` (line 1070) `class SystemAnalyzer`
- `AnalysisPipeline` (line 1288) `class AnalysisPipeline`

**Methods:**
- `main` (line 1545) `def main()`
- `forward` (line 55) `def forward(self, a, b)`
- `get_coefficients` (line 56) `def get_coefficients(self)`
- `extract` (line 61) `def extract(self, model)`
- `analyze` (line 66) `def analyze(self, charge_density)`
- `calculate` (line 71) `def calculate(self, dirac_data, eval_points)`
- `calculate` (line 76) `def calculate(self, electric_field, surface_points)`
- `extract` (line 81) `def extract(self, model)`
- `compute` (line 86) `def compute(self, A, B, C, D)`
- `analyze_stability` (line 91) `def analyze_stability(self)`
- `get_poles` (line 92) `def get_poles(self)`
- `get_zeros` (line 93) `def get_zeros(self)`
- `compute_bode` (line 98) `def compute_bode(self)`
- `compute_margins` (line 99) `def compute_margins(self)`
- `compute_nyquist` (line 100) `def compute_nyquist(self)`
- `compute_step` (line 105) `def compute_step(self)`
- `compute_impulse` (line 106) `def compute_impulse(self)`
- `load` (line 111) `def load(self, path, device)`
- `migrate` (line 116) `def migrate(self, raw_data)`
- `visualize` (line 121) `def visualize(self, data, output_path)`
- `__init__` (line 125) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 136) `def _initialize(self)`
- `forward` (line 141) `def forward(self, a, b)`
- `get_coefficients` (line 144) `def get_coefficients(self)`
- `extract` (line 153) `def extract(self, model)`
- `__init__` (line 161) `def __init__(self, config)`
- `analyze` (line 164) `def analyze(self, charge_density)`
- `__init__` (line 193) `def __init__(self, config)`
- `calculate` (line 196) `def calculate(self, dirac_data, eval_points)`
- `__init__` (line 226) `def __init__(self, config)`
- `calculate` (line 229) `def calculate(self, electric_field, surface_points)`
- `calculate` (line 249) `def calculate(self, electric_field)`
- `__init__` (line 254) `def __init__(self, config)`
- `verify` (line 257) `def verify(self, dirac_data, flux_data)`
- `extract` (line 272) `def extract(self, model)`
- `compute` (line 299) `def compute(self, A, B, C, D)`
- `__init__` (line 314) `def __init__(self, numerator, denominator, config)`
- `_compute` (line 323) `def _compute(self)`
- `get_poles` (line 334) `def get_poles(self)`
- `get_zeros` (line 337) `def get_zeros(self)`
- `analyze_stability` (line 340) `def analyze_stability(self)`
- `classify_poles` (line 378) `def classify_poles(self)`
- `compute_damping` (line 406) `def compute_damping(self)`
- `compute_time_constants` (line 445) `def compute_time_constants(self)`
- `__init__` (line 464) `def __init__(self, numerator, denominator, config)`
- `compute_bode` (line 474) `def compute_bode(self)`
- `compute_margins` (line 490) `def compute_margins(self)`
- `compute_nyquist` (line 516) `def compute_nyquist(self)`
- `evaluate_nyquist_stability` (line 535) `def evaluate_nyquist_stability(self, nyquist_data)`
- `__init__` (line 567) `def __init__(self, numerator, denominator, config)`
- `compute_step` (line 577) `def compute_step(self)`
- `compute_impulse` (line 589) `def compute_impulse(self)`
- `analyze_step_characteristics` (line 601) `def analyze_step_characteristics(self, step_data)`
- `load` (line 654) `def load(self, path, device)`
- `migrate` (line 662) `def migrate(self, raw_data)`
- `_migrate_dict` (line 674) `def _migrate_dict(self, state_dict)`
- `_migrate_custom_format` (line 683) `def _migrate_custom_format(self, state_dict)`
- `_migrate_coefs_format` (line 699) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 706) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 711) `def __init__(self, config)`
- `visualize` (line 714) `def visualize(self, data, output_path)`
- `__init__` (line 739) `def __init__(self, config)`
- `visualize` (line 742) `def visualize(self, data, output_path)`
- `__init__` (line 781) `def __init__(self, config)`
- `visualize` (line 784) `def visualize(self, data, output_path)`
- `__init__` (line 810) `def __init__(self, config)`
- `visualize` (line 813) `def visualize(self, data, output_path)`
- `__init__` (line 849) `def __init__(self, config)`
- `visualize` (line 852) `def visualize(self, data, output_path)`
- `__init__` (line 900) `def __init__(self, config)`
- `visualize` (line 903) `def visualize(self, data, output_path)`
- `__init__` (line 937) `def __init__(self, config)`
- `visualize` (line 940) `def visualize(self, data, output_path)`
- `__init__` (line 967) `def __init__(self, config)`
- `visualize` (line 970) `def visualize(self, data, output_path)`
- `__init__` (line 1071) `def __init__(self, checkpoint_path, config)`
- `_load_model` (line 1088) `def _load_model(self)`
- `analyze` (line 1104) `def analyze(self)`
- `_print_report` (line 1204) `def _print_report(self, results)`
- `__init__` (line 1289) `def __init__(self, config)`
- `process_checkpoint` (line 1300) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 1357) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 1385) `def generate_summary(self, all_results, output_dir)`
- `_compute_aggregate_statistics` (line 1406) `def _compute_aggregate_statistics(self, results)`
- `_generate_text_report` (line 1473) `def _generate_text_report(self, summary, output_dir)`

#### `ablation_8192.py`
**Path:** `experiments/ablation/ablation_8192.py`

*No symbols extracted*

#### `ablation_study.py`
**Path:** `experiments/ablation/ablation_study.py`

**Classes:**
- `BenchmarkResult` (line 27) `class BenchmarkResult`

**Methods:**
- `load_libraries` (line 54) `def load_libraries()` - *Cargar bibliotecas con manejo de errores*
- `run_openblas` (line 111) `def run_openblas(libs, A, B, C, n)` - *Ejecutar multiplicación con OpenBLAS*
- `run_strassen` (line 122) `def run_strassen(libs, name, func_name, A, B, C, n)` - *Ejecutar multiplicación con Strassen*
- `benchmark_single` (line 132) `def benchmark_single(libs, algo_name, func_name, A, B, C, C_ref, n, n_runs, warmup)` - *Benchmark una implementación*
- `run_ablation` (line 180) `def run_ablation(libs, sizes, n_runs, warmup)` - *Ejecutar ablación completa*
- `analyze_results` (line 233) `def analyze_results(results)` - *Analizar y presentar resultados*
- `main` (line 273) `def main()`
- `mean_time` (line 35) `def mean_time(self)`
- `std_time` (line 39) `def std_time(self)`
- `min_time` (line 43) `def min_time(self)`
- `max_time` (line 47) `def max_time(self)`
- `mean_gflops` (line 51) `def mean_gflops(self)`

#### `apendix_experiments.py`
**Path:** `experiments/apendix_experiments.py`

**Classes:**
- `StrassenOperator` (line 51) `class StrassenOperator(Module)`

**Functions:**
- `setup_matplotlib` (line 34) `def setup_matplotlib()`

**Methods:**
- `generate_batch` (line 87) `def generate_batch(n, device)`
- `generate_test_set` (line 93) `def generate_test_set(n, device)`
- `compute_delta` (line 100) `def compute_delta(model)`
- `verify_strassen_structure` (line 117) `def verify_strassen_structure(U_disc, V_disc, W_disc, tolerance)`
- `compute_S_theta` (line 138) `def compute_S_theta(model)`
- `compute_gradient_covariance` (line 154) `def compute_gradient_covariance(model, batch_size, n_samples)`
- `train_with_logging` (line 187) `def train_with_logging(batch_size, total_epochs, lr, wd, symmetric_init, seed, log_interval)`
- `sparsify_and_discretize` (line 274) `def sparsify_and_discretize(model, batch_size)`
- `run_phase_diagram` (line 327) `def run_phase_diagram()`
- `run_batch_size_effect` (line 429) `def run_batch_size_effect()`
- `main` (line 526) `def main()`
- `__init__` (line 53) `def __init__(self, rank, symmetric_init)`
- `forward` (line 67) `def forward(self, A, B)`
- `slot_importance` (line 77) `def slot_importance(self)`
- `count_active` (line 83) `def count_active(self, threshold)`

#### `cache_analysis_v2.py`
**Path:** `experiments/cache_analysis_v2.py`

**Functions:**
- `cache_analysis` (line 7) `def cache_analysis()` - *Full memory analysis for training.

Model: Bilinear with rank-8 (target 7 active)
- U: 8 x 4 = 32 floats
- V: 8 x 4 = 32 floats  
- W: 4 x 8 = 32 floats
Total params: 96 floats = 384 bytes

AdamW optimizer state (per param):
- m (momentum): 96 floats
- v (variance): 96 floats
Total optimizer: 192 floats = 768 bytes

Per-batch memory:
- Input batch: B x 8 floats (A, B flattened)
- Hidden: B x 8 floats (intermediate M)
- Output: B x 4 floats (C)
- Gradients: same as forward
Total per-sample: ~40 floats x 2 (forward+backward) = 80 floats = 320 bytes

Total training memory = model + optimizer + batch*

#### `all_test_extended.py`
**Path:** `experiments/extended_experiments/all_test_extended.py`

**Classes:**
- `Configuration` (line 41) `class Configuration`
- `Narrator` (line 95) `class Narrator`
- `SystemFingerprint` (line 152) `class SystemFingerprint`
- `ArithmeticDataset` (line 190) `class ArithmeticDataset(Dataset)` - *Dataset for arithmetic operations based on original implementation.

Generates (a, b) pairs and their product c = (a * b) mod MODULUS.
Each a, b, c is a one-hot vector of size MODULUS.

The BilinearModel learns: c = W @ ((U @ a) * (V @ b))
This is a lookup table for modular multiplication.*
- `BilinearModel` (line 238) `class BilinearModel(Module)` - *Bilinear model from original implementation.

Architecture:
- U: Linear(d_vocab, rank) -> [67, 8] weights
- V: Linear(d_vocab, rank) -> [67, 8] weights  
- W: Linear(rank, d_vocab) -> [8, 67] weights

Forward:
- a = x[:, 0] -> (batch, d_vocab)
- b = x[:, 1] -> (batch, d_vocab)
- m = U(a) * V(b) -> (batch, rank)
- logits = W(m) -> (batch, d_vocab)*
- `Task` (line 293) `class Task(ABC)`
- `MatrixMultiplicationTask` (line 311) `class MatrixMultiplicationTask(Task)`
- `ParityDataset` (line 354) `class ParityDataset(Dataset)` - *Dataset for parity task.*
- `ParityTask` (line 375) `class ParityTask(Task)`
- `GradientCovarianceProbe` (line 402) `class GradientCovarianceProbe` - *Analyzes gradient covariance to understand batch size effects.*
- `SpectralInterventionProbe` (line 465) `class SpectralInterventionProbe` - *Actively intervenes on condition number during training.*
- `AttractorLandscapeProbe` (line 488) `class AttractorLandscapeProbe` - *Analyzes attractor landscapes to understand failure modes.*
- `VolumeEstimator` (line 510) `class VolumeEstimator` - *Estimates volume of success basin in weight space.*
- `RobustnessTest` (line 524) `class RobustnessTest` - *Tests robustness against various perturbations.*
- `CheckpointManager` (line 581) `class CheckpointManager` - *Manages training checkpoints with configurable intervals.*
- `TrainingMetrics` (line 627) `class TrainingMetrics` - *Comprehensive metrics collection for training progress.*
- `ExperimentRunner` (line 700) `class ExperimentRunner` - *Main orchestration engine for training and evaluation.*
- `DiscretizationAnalyzer` (line 809) `class DiscretizationAnalyzer` - *Analyzes discretization quality and success probability.*
- `ExpansionVerifier` (line 851) `class ExpansionVerifier` - *Verifies zero-shot transfer to larger problem sizes.*
- `ExperimentPipeline` (line 879) `class ExperimentPipeline` - *Executes the complete battery of open-question experiments.*

**Methods:**
- `main` (line 1403) `def main()`
- `__init__` (line 89) `def __init__(self)`
- `__init__` (line 96) `def __init__(self, config)`
- `begin` (line 101) `def begin(self, experiment_name)`
- `progress` (line 109) `def progress(self, current, total, metrics)`
- `checkpoint` (line 116) `def checkpoint(self, epoch, loss, accuracy)`
- `result` (line 121) `def result(self, name, value, context)`
- `verdict` (line 126) `def verdict(self, hypothesis, evidence, conclusion)`
- `failure` (line 131) `def failure(self, reason, details)`
- `complete` (line 136) `def complete(self, summary)`
- `claim` (line 144) `def claim(self, statement, confidence)`
- `note` (line 148) `def note(self, observation)`
- `__init__` (line 153) `def __init__(self, config)`
- `capture` (line 156) `def capture(self)`
- `report` (line 174) `def report(self)`
- `__init__` (line 202) `def __init__(self, size, modulus)`
- `_generate_data` (line 207) `def _generate_data(self)`
- `__len__` (line 231) `def __len__(self)`
- `__getitem__` (line 234) `def __getitem__(self, idx)`
- `__init__` (line 253) `def __init__(self, d_vocab, rank, scale)`
- `forward` (line 265) `def forward(self, x)`
- `get_weights` (line 277) `def get_weights(self)`
- `get_U_weights` (line 283) `def get_U_weights(self)`
- `get_V_weights` (line 286) `def get_V_weights(self)`
- `get_W_weights` (line 289) `def get_W_weights(self)`
- `name` (line 295) `def name(self)`
- `d_vocab` (line 299) `def d_vocab(self)`
- `generate_dataset` (line 303) `def generate_dataset(self, size)`
- `verify` (line 307) `def verify(self, model, x, y)`
- `__init__` (line 312) `def __init__(self, modulus)`
- `name` (line 316) `def name(self)`
- `d_vocab` (line 319) `def d_vocab(self)`
- `generate_dataset` (line 322) `def generate_dataset(self, size)`
- `verify` (line 344) `def verify(self, model, x, y)`
- `__init__` (line 357) `def __init__(self, size, bit_length)`
- `_generate_data` (line 362) `def _generate_data(self)`
- `__len__` (line 368) `def __len__(self)`
- `__getitem__` (line 371) `def __getitem__(self, idx)`
- `__init__` (line 376) `def __init__(self, bit_length, modulus)`
- `name` (line 381) `def name(self)`
- `d_vocab` (line 384) `def d_vocab(self)`
- `generate_dataset` (line 387) `def generate_dataset(self, size)`
- `verify` (line 391) `def verify(self, model, x, y)`
- `__init__` (line 405) `def __init__(self, model)`
- `capture_gradients` (line 409) `def capture_gradients(self, dataloader, n_batches)`
- `compute_covariance` (line 429) `def compute_covariance(self)`
- `compute_condition_number` (line 436) `def compute_condition_number(self)`
- `compute_gradient_noise_scale` (line 442) `def compute_gradient_noise_scale(self, batch_size, learning_rate)`
- `analyze` (line 450) `def analyze(self, dataloader, batch_size, learning_rate)`
- `__init__` (line 468) `def __init__(self, model, target_kappa)`
- `spectral_regularizer` (line 472) `def spectral_regularizer(self)`
- `__init__` (line 491) `def __init__(self, model)`
- `count_local_minima` (line 494) `def count_local_minima(self, directions, losses)`
- `measure_basin_width` (line 501) `def measure_basin_width(self, weights, direction, n_points)`
- `classify_failure_mode` (line 505) `def classify_failure_mode(self, final_weights, initial_weights)`
- `__init__` (line 513) `def __init__(self, success_radius)`
- `estimate_volume_monte_carlo` (line 516) `def estimate_volume_monte_carlo(self, model_class, n_samples, success_checker)`
- `compute_fractal_dimension` (line 520) `def compute_fractal_dimension(self, trajectory)`
- `__init__` (line 527) `def __init__(self, model)`
- `add_gaussian_noise` (line 530) `def add_gaussian_noise(self, sigma)`
- `fgsm_attack` (line 536) `def fgsm_attack(self, x, y, epsilon)`
- `quantize_weights` (line 544) `def quantize_weights(self, bits)`
- `test_discretization_with_noise` (line 550) `def test_discretization_with_noise(self, sigma, checker)`
- `run_fragility_analysis` (line 560) `def run_fragility_analysis(self, sigma_values, checker)`
- `__init__` (line 584) `def __init__(self, config, experiment_name)`
- `save_checkpoint` (line 593) `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `load_checkpoint` (line 619) `def load_checkpoint(self, path, model, optimizer)`
- `__init__` (line 630) `def __init__(self)`
- `update` (line 642) `def update(self, train_loss, train_acc, test_loss, test_acc, kappa, grad_norm, weight_norm, disc_margin)`
- `detect_grokking` (line 660) `def detect_grokking(self, loss_threshold, test_loss_threshold, min_duration)`
- `progress_bar_string` (line 679) `def progress_bar_string(self, epoch, total_epochs)`
- `__init__` (line 703) `def __init__(self, config)`
- `train_epoch` (line 708) `def train_epoch(self, model, dataloader, optimizer)`
- `evaluate` (line 730) `def evaluate(self, model, dataloader)`
- `run_training` (line 748) `def run_training(self, model, train_loader, test_loader, experiment_name, epochs, batch_size, lr, wd, verbose)`
- `__init__` (line 812) `def __init__(self, config)`
- `compute_discretization_margin` (line 815) `def compute_discretization_margin(self, model)`
- `discretize_weights` (line 827) `def discretize_weights(self, model)`
- `check_strassen_structure` (line 832) `def check_strassen_structure(self, model, modulus)`
- `count_discretized_parameters` (line 839) `def count_discretized_parameters(self, model)`
- `__init__` (line 854) `def __init__(self, config)`
- `verify_expansion` (line 857) `def verify_expansion(self, model, task, sizes)`
- `__init__` (line 882) `def __init__(self, config)`
- `experiment_batch_size_mechanism` (line 891) `def experiment_batch_size_mechanism(self)` - *Experiment 1: Why batch size [24,128] works.*
- `experiment_kappa_intervention` (line 962) `def experiment_kappa_intervention(self)` - *Experiment 2: Active intervention on κ.*
- `experiment_failure_analysis` (line 1026) `def experiment_failure_analysis(self)` - *Experiment 3: Why 32% of runs fail.*
- `experiment_generalization` (line 1088) `def experiment_generalization(self)` - *Experiment 4: Generalization to other tasks.*
- `experiment_basin_volume` (line 1145) `def experiment_basin_volume(self)` - *Experiment 5: Basin volume estimation.*
- `experiment_hardware_reproducibility` (line 1158) `def experiment_hardware_reproducibility(self)` - *Experiment 6: Hardware reproducibility testing.*
- `experiment_fragility` (line 1217) `def experiment_fragility(self)` - *Experiment 7: Discretization fragility testing.*
- `run_all_experiments` (line 1332) `def run_all_experiments(self)`
- `_save_results` (line 1371) `def _save_results(self)`
- `_generate_summary` (line 1379) `def _generate_summary(self)`
- `checker` (line 1253) `def checker()`

#### `exp1_covariance_spectrometry.py`
**Path:** `experiments/extended_experiments/exp1_covariance_spectrometry.py`

**Classes:**
- `StrassenOperator` (line 48) `class StrassenOperator(Module)` - *Spectral operator for 2x2 matrix multiplication.
Tensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)*

**Functions:**
- `setup_matplotlib` (line 22) `def setup_matplotlib()`

**Methods:**
- `generate_batch` (line 112) `def generate_batch(n, scale)` - *Generate batch of matrices.*
- `compute_gradient_covariance` (line 120) `def compute_gradient_covariance(model, batch_size, n_samples)` - *Compute the gradient covariance matrix Σₜ and its eigenvalues.

Returns:
    kappa: condition number λ_max / λ_min
    lambda_max: maximum eigenvalue
    lambda_min: minimum eigenvalue  
    trace: trace of covariance (total gradient energy)
    frobenius_norm: Frobenius norm of covariance*
- `load_checkpoint` (line 196) `def load_checkpoint(checkpoint_path)` - *Load model from checkpoint file.*
- `analyze_checkpoint` (line 222) `def analyze_checkpoint(checkpoint_path, batch_sizes, n_samples, n_runs)` - *Analyze a single checkpoint with multiple batch sizes.*
- `main` (line 277) `def main()` - *Main execution for Experiment 1.*
- `generate_visualization` (line 380) `def generate_visualization(results, output_dir)` - *Generate publication-quality figures.*
- `__init__` (line 54) `def __init__(self, rank)`
- `forward` (line 61) `def forward(self, A, B)`
- `get_all_parameters` (line 71) `def get_all_parameters(self)` - *Get all parameters as a single flattened vector.*
- `compute_per_sample_gradients` (line 78) `def compute_per_sample_gradients(self, A, B, C_true)` - *Compute per-sample gradients for covariance estimation.
Returns: gradients shape [batch_size, num_parameters]*

#### `exp2_noise_ablation.py`
**Path:** `experiments/extended_experiments/exp2_noise_ablation.py`

**Classes:**
- `StrassenOperator` (line 49) `class StrassenOperator(Module)` - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 25) `def setup_matplotlib()`

**Methods:**
- `generate_batch` (line 99) `def generate_batch(n, scale)` - *Generate batch of matrices.*
- `compute_gradient_covariance_matrix` (line 107) `def compute_gradient_covariance_matrix(model, n_samples, batch_size)` - *Compute the gradient covariance matrix Σ.*
- `get_eigenbasis` (line 139) `def get_eigenbasis(covariance)` - *Get eigenvectors and eigenvalues of covariance matrix.*
- `load_checkpoint` (line 150) `def load_checkpoint(checkpoint_path)` - *Load model from checkpoint file.*
- `experiment_treatment_a_gradient_noise` (line 172) `def experiment_treatment_a_gradient_noise(model, noise_std, n_test)` - *Treatment A: Add noise to gradients DURING forward/backward pass.

This simulates the effect of gradient noise during training without actual retraining.*
- `experiment_treatment_b_weight_noise` (line 212) `def experiment_treatment_b_weight_noise(model, noise_std, n_test)` - *Treatment B: Noise on weights BEFORE evaluation (already done in paper).
This is the fallback mechanism test.*
- `experiment_treatment_c_structured_noise` (line 246) `def experiment_treatment_c_structured_noise(model, covariance, noise_std, n_test)` - *Treatment C: Structured noise by eigenvectors of Σ.

Tests whether damage is isotropic or aligned with gradient covariance directions.*
- `run_noise_ablation` (line 315) `def run_noise_ablation(checkpoint_path, noise_levels)` - *Run complete noise ablation experiment on a checkpoint.*
- `main` (line 348) `def main()` - *Main execution for Experiment 2.*
- `generate_visualization` (line 433) `def generate_visualization(results, output_dir)` - *Generate publication-quality figures.*
- `__init__` (line 54) `def __init__(self, rank)`
- `forward` (line 61) `def forward(self, A, B)`
- `get_all_parameters` (line 71) `def get_all_parameters(self)`
- `set_parameters` (line 77) `def set_parameters(self, new_params)` - *Set parameters from a flattened tensor.*
- `compute_loss` (line 85) `def compute_loss(self, A, B)` - *Compute MSE loss.*
- `compute_accuracy` (line 91) `def compute_accuracy(self, A, B, threshold)` - *Compute accuracy (proportion of predictions within threshold).*

#### `exp3_prospective_prediction.py`
**Path:** `experiments/extended_experiments/exp3_prospective_prediction.py`

**Classes:**
- `StrassenOperator` (line 56) `class StrassenOperator(Module)` - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 32) `def setup_matplotlib()`

**Methods:**
- `generate_batch` (line 114) `def generate_batch(n, scale)` - *Generate batch of matrices.*
- `compute_kappa` (line 121) `def compute_kappa(model, n_samples, batch_size)` - *Compute condition number κ(Σ) of gradient covariance matrix.*
- `load_checkpoint` (line 166) `def load_checkpoint(checkpoint_path)` - *Load model from checkpoint file.*
- `simulate_early_prediction` (line 188) `def simulate_early_prediction(checkpoint_path, early_epoch_fraction)` - *Simulate the prospective prediction experiment.

Since we can't actually retrain, we use available checkpoints to simulate:
- "Early" checkpoint = first checkpoint in sequence
- "Final" checkpoint = last checkpoint in sequence

This tests if early-stage κ predicts final-stage success.*
- `run_prospective_prediction_experiment` (line 236) `def run_prospective_prediction_experiment(checkpoint_files)` - *Run the full prospective prediction experiment across all checkpoints.*
- `compute_roc_analysis` (line 252) `def compute_roc_analysis(predictions)` - *Compute ROC curve and AUC for κ as predictor of success.*
- `main` (line 313) `def main()` - *Main execution for Experiment 3.*
- `generate_visualization` (line 410) `def generate_visualization(results, predictions, output_dir)` - *Generate publication-quality figures.*
- `__init__` (line 59) `def __init__(self, rank)`
- `forward` (line 66) `def forward(self, A, B)`
- `get_all_parameters` (line 76) `def get_all_parameters(self)`
- `set_parameters` (line 82) `def set_parameters(self, new_params)`
- `count_active_slots` (line 89) `def count_active_slots(self, threshold)` - *Count active slots based on weight norms.*
- `compute_discretization_margin` (line 97) `def compute_discretization_margin(self)` - *Compute how close weights are to discrete values {-1, 0, 1}.
δ(θ) = mean(|w - round(w)|)*
- `is_grokked` (line 107) `def is_grokked(self, margin_threshold, active_slots_target)` - *Check if model has grokked (discretized with low error).*

#### `exp4_trajectory_perturbation.py`
**Path:** `experiments/extended_experiments/exp4_trajectory_perturbation.py`

**Classes:**
- `StrassenOperator` (line 51) `class StrassenOperator(Module)` - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 27) `def setup_matplotlib()`

**Methods:**
- `generate_batch` (line 117) `def generate_batch(n, scale)` - *Generate batch of matrices.*
- `load_checkpoint` (line 124) `def load_checkpoint(checkpoint_path)` - *Load model from checkpoint file.*
- `simulate_trajectory_perturbation` (line 146) `def simulate_trajectory_perturbation(checkpoint_path, perturbations)` - *Simulate trajectory perturbation effects using available checkpoints.

Since we can't retrain, we simulate the effect of perturbations by:
1. Taking a "final" checkpoint as the target
2. Using earlier checkpoints to simulate "early training state"
3. Applying perturbations and measuring their effect*
- `main` (line 277) `def main()` - *Main execution for Experiment 4.*
- `generate_visualization` (line 398) `def generate_visualization(results, output_dir)` - *Generate publication-quality figures.*
- `__init__` (line 54) `def __init__(self, rank)`
- `forward` (line 61) `def forward(self, A, B)`
- `get_all_parameters` (line 71) `def get_all_parameters(self)`
- `set_parameters` (line 77) `def set_parameters(self, new_params)`
- `get_weight_norm` (line 84) `def get_weight_norm(self)` - *Get total L2 norm of all parameters.*
- `get_weight_direction` (line 91) `def get_weight_direction(self)` - *Get normalized weight vector direction.*
- `compute_gradient_norm` (line 96) `def compute_gradient_norm(self, A, B)` - *Compute norm of gradients.*
- `cosine_similarity` (line 110) `def cosine_similarity(self, other_params)` - *Compute cosine similarity between current weights and target weights.*
- `compute_metrics` (line 194) `def compute_metrics(model, name)` - *Compute evaluation metrics.*

#### `exp5_discreteness_attractors.py`
**Path:** `experiments/extended_experiments/exp5_discreteness_attractors.py`

*No symbols extracted*

#### `run_all_experiments.py`
**Path:** `experiments/extended_experiments/run_all_experiments.py`

**Classes:**
- `StrassenOperator` (line 48) `class StrassenOperator(Module)` - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 24) `def setup_matplotlib()`

**Methods:**
- `generate_batch` (line 95) `def generate_batch(n, scale)`
- `load_checkpoint_robust` (line 101) `def load_checkpoint_robust(checkpoint_path, model)` - *Load checkpoint with multiple format fallback strategies.*
- `compute_gradient_covariance_safe` (line 135) `def compute_gradient_covariance_safe(model, batch_size, n_samples)` - *Compute κ(Σₜ) with numerical safety.*
- `run_all_experiments` (line 205) `def run_all_experiments()` - *Run all experiments.*
- `generate_summary_visualization` (line 496) `def generate_summary_visualization(results, output_dir)` - *Generate summary visualization.*
- `__init__` (line 51) `def __init__(self, rank)`
- `forward` (line 58) `def forward(self, A, B)`
- `get_all_parameters` (line 68) `def get_all_parameters(self)`
- `set_parameters` (line 74) `def set_parameters(self, new_params)`
- `count_active_slots` (line 81) `def count_active_slots(self, threshold)`
- `compute_discretization_margin` (line 88) `def compute_discretization_margin(self)`
- `compute_accuracy` (line 324) `def compute_accuracy()`

#### `validate2.py`
**Path:** `experiments/extended_experiments/validate2.py`

**Classes:**
- `ExperimentConfig` (line 60) `class ExperimentConfig` - *Configuración centralizada para todos los experimentos.

Sigue el principio de responsabilidad única - solo gestiona parámetros.
No usa magic numbers; todos los valores están definidos aquí.*
- `StrassenOperator` (line 126) `class StrassenOperator(Module)` - *Operador Strassen para multiplicación de matrices 2x2 vía descomposición tensorial.

El modelo representa el tensor de rango R:
C_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)

Donde:
- U, V: Coeficientes de combinación lineal (LC)
- W: Coeficientes de reconstrucción
- Sparsity (SP): Cuántos slots están activos*
- `StrassenDataGenerator` (line 225) `class StrassenDataGenerator` - *Generador de datos para multiplicación de matrices 2x2.*
- `LocalComplexityCalculator` (line 277) `class LocalComplexityCalculator` - *Calculadora de Complejidad Local basada en la varianza del gradiente.

LC = ||grad||^2 / N (Noise Scale normalizada)

Esta métrica captura la "dificultad" del batch actual y su relación
con el aprendizaje del modelo.*
- `GrokkingVerifier` (line 340) `class GrokkingVerifier` - *Verifica que un modelo ha grokkeado correctamente.*
- `IterativePruningEngine` (line 405) `class IterativePruningEngine` - *Motor para poda iterativa con fine-tuning completo.

Protocolo:
1. Calcular importancia de pesos (magnitud L1)
2. Podar p% de pesos menos importantes
3. Fine-tune por épocas especificadas
4. Chequear degradación δ
5. Si δ < threshold, continuar; si no, detener
6. Verificar discretización con δ < 0.1 (PUNTO A DEL REVISOR)

Este protocolo es CRUCIAL para verificar la hipótesis de cuenca discreta.*
- `LocalComplexityExperiment` (line 764) `class LocalComplexityExperiment` - *Experimento de Local Complexity entrenando desde cero.

Esto es CRUCIAL para responder al revisor (PUNTO B):
- Se entrena un modelo desde cero hasta grokking
- Se mide LC en cada época para capturar la transición de fase
- Si LC muestra un cambio alrededor del grokking, la métrica es útil
- Si LC permanece constante, la métrica NO captura la transición*
- `BalancedRunsGenerator` (line 890) `class BalancedRunsGenerator` - *Generador de runs balanceados para calcular AUC válido.

Esto es CRUCIAL para responder al revisor (PUNTO C):
- Entrenar múltiples modelos con diferentes hiperparámetros
- Algunos grokkean, otros no (condiciones variadas)
- Generar dataset balanceado para ROC/AUC

Si todos los samples son de una sola clase, AUC es indefinido.
Necesitamos mix de grokked + no-grokked para calcularlo.*
- `BootstrapStatistics` (line 1135) `class BootstrapStatistics` - *Generador de estadísticas con intervalos de confianza bootstrap.

Calcula:
- Curvas ROC con IC del 95%
- AUC con IC del 95%
- Kappa de Cohen con IC del 95%*
- `VisualizationGenerator` (line 1281) `class VisualizationGenerator` - *Generador de visualizaciones con estilo académico.*
- `ExperimentOrchestrator` (line 1646) `class ExperimentOrchestrator` - *Orquestador principal para todos los experimentos.

Coordina:
1. Carga de checkpoint grokkeado
2. Verificación de grokking
3. Experimento de Local Complexity
4. Protocolo de poda iterativa
5. Análisis ROC/AUC
6. Generación de visualizaciones*

**Methods:**
- `find_grokked_checkpoint` (line 2313) `def find_grokked_checkpoint()` - *Buscar checkpoint grokkeado en múltiples ubicaciones.

Returns:
    Path al archivo de checkpoint grokkeado*
- `analyze_checkpoints` (line 2353) `def analyze_checkpoints()` - *Analizar todos los checkpoints disponibles para encontrar el grokkeado.

Returns:
    Diccionario con métricas de cada checkpoint*
- `main` (line 2423) `def main()` - *Punto de entrada principal.*
- `__post_init__` (line 115) `def __post_init__(self)`
- `__init__` (line 139) `def __init__(self, rank)`
- `_initialize_weights` (line 150) `def _initialize_weights(self)` - *Inicializar pesos con consideración para grokking.*
- `forward` (line 156) `def forward(self, A, B)` - *Computar A @ B usando descomposición tensorial.

Args:
    A: Tensor de entrada de forma (batch, 2, 2)
    B: Tensor de entrada de forma (batch, 2, 2)
    
Returns:
    Tensor de salida de forma (batch, 2, 2)*
- `slot_importance` (line 183) `def slot_importance(self)` - *Importancia de cada slot basada en normas.*
- `count_active` (line 190) `def count_active(self, threshold)` - *Contar slots activos.*
- `compute_SP` (line 194) `def compute_SP(self)` - *Métrica de Sparsity. SP -> 0 significa máxima sparsity.*
- `get_state_dict` (line 202) `def get_state_dict(self)` - *Obtener estado completo para checkpointing.*
- `load_state_dict` (line 211) `def load_state_dict(self, state_dict)` - *Cargar estado completo desde checkpoint.*
- `__init__` (line 228) `def __init__(self, num_samples, matrix_size, seed)`
- `generate_matrix` (line 238) `def generate_matrix(self)` - *Generar matriz aleatoria con valores enteros.*
- `generate_data` (line 242) `def generate_data(self)` - *Generar pares de matrices y sus productos.*
- `get_train_test` (line 260) `def get_train_test(self, test_ratio)` - *Dividir en conjuntos de entrenamiento y prueba.*
- `__init__` (line 287) `def __init__(self, model, config)`
- `compute_lc` (line 292) `def compute_lc(self, batch_inputs, batch_targets)` - *Calcular LC para un batch específico.

LC = ||g||^2 / N_batch

Donde g es el gradiente de la pérdida respecto a los pesos.*
- `compute_batch_diversity` (line 326) `def compute_batch_diversity(self, batch_inputs)` - *Calcular diversidad del batch basada en varianza de activaciones.*
- `__init__` (line 343) `def __init__(self, config)`
- `verify` (line 347) `def verify(self, model, n_test)` - *Verificar que el operador ha grokkeado correctamente.

Returns:
    Tupla de (éxito, métricas)*
- `_generate_batch` (line 393) `def _generate_batch(self, n, scale)` - *Generar batch de matrices aleatorias.*
- `__init__` (line 420) `def __init__(self, config)`
- `get_weight_magnitudes` (line 426) `def get_weight_magnitudes(self, model)` - *Obtener magnitud absoluta de todos los pesos.*
- `compute_sparsity` (line 431) `def compute_sparsity(self, model)` - *Calcular porcentaje de pesos en cero.*
- `prune_percent` (line 437) `def prune_percent(self, model, percent)` - *Podar el porcentaje especificado de pesos menos importantes.

Returns: (num_pruned, current_sparsity)*
- `fine_tune` (line 457) `def fine_tune(self, model, train_data)` - *Fine-tune del modelo podado con métricas completas.*
- `_generate_batch` (line 510) `def _generate_batch(self, n, scale)` - *Generar batch de matrices aleatorias.*
- `run_protocol` (line 624) `def run_protocol(self, model, train_data)` - *Ejecutar protocolo completo de poda iterativa.

Returns:
    Diccionario con resultados completos*
- `__init__` (line 775) `def __init__(self, config)`
- `run_full_experiment` (line 779) `def run_full_experiment(self, target_epochs)` - *Ejecutar experimento completo de LC entrenando desde cero.*
- `_generate_batch` (line 882) `def _generate_batch(self, n, scale)`
- `__init__` (line 903) `def __init__(self, config)`
- `run_balanced_experiments` (line 907) `def run_balanced_experiments(self, n_runs)` - *Ejecutar multiples runs con condiciones disenhadas para producir mix.

Returns:
    Diccionario con resultados de todos los runs*
- `_train_single_run` (line 1021) `def _train_single_run(self, run_idx, config)` - *Entrenar un solo modelo con configuración específica.*
- `_compute_roc` (line 1099) `def _compute_roc(self, y_true, y_scores)` - *Calcular ROC/AUC básico.*
- `_generate_batch` (line 1127) `def _generate_batch(self, n, scale)`
- `__init__` (line 1145) `def __init__(self, config)`
- `compute_roc_with_ci` (line 1149) `def compute_roc_with_ci(self, y_true, y_scores)` - *Calcular curva ROC con intervalos de confianza bootstrap.

Returns:
    Diccionario con resultados ROC*
- `compute_kappa_with_ci` (line 1236) `def compute_kappa_with_ci(self, y_true, y_pred)` - *Calcular Kappa de Cohen con IC bootstrap.*
- `compute_accuracy_with_ci` (line 1260) `def compute_accuracy_with_ci(self, correct)` - *Calcular accuracy con IC binomial.*
- `__init__` (line 1284) `def __init__(self, style)`
- `plot_local_complexity` (line 1298) `def plot_local_complexity(self, epochs, lc_values, accuracy, save_path)` - *Graficar evolución de Local Complexity y Accuracy.*
- `plot_pruning_results` (line 1342) `def plot_pruning_results(self, pruning_data, save_path)` - *Graficar resultados de poda iterativa.*
- `plot_roc_with_ci` (line 1404) `def plot_roc_with_ci(self, roc_data, save_path)` - *Graficar curva ROC con intervalos de confianza.*
- `plot_balanced_runs_results` (line 1467) `def plot_balanced_runs_results(self, balanced_data, save_path)` - *Graficar resultados del experimento de runs balanceados.*
- `plot_discretization_results` (line 1556) `def plot_discretization_results(self, pruning_data, save_path)` - *Graficar resultados de discretizacion.*
- `__init__` (line 1659) `def __init__(self, config)`
- `find_grokked_checkpoint` (line 1689) `def find_grokked_checkpoint(self)` - *Buscar checkpoint grokkeado en múltiples ubicaciones.

Returns:
    Path al archivo de checkpoint o None si no se encuentra*
- `load_grokked_checkpoint` (line 1733) `def load_grokked_checkpoint(self, checkpoint_path)` - *Cargar checkpoint grokkeado y verificar que grokkeó.

Returns:
    Modelo cargado y verificado*
- `verify_checkpoint_is_grokked` (line 1768) `def verify_checkpoint_is_grokked(self)` - *Verificar que el checkpoint cargado realmente grokkeó.

Returns:
    Tupla de (es_grokked, métricas)*
- `run_local_complexity_experiment` (line 1795) `def run_local_complexity_experiment(self, epochs)` - *Ejecutar experimento de Local Complexity vs Época.

Nota: Como usamos un checkpoint ya grokked, esto mide la LC
durante el fine-tuning post-poda, no durante el grokking inicial.

Returns:
    Diccionario con historial de LC y accuracy*
- `run_lc_training_experiment` (line 1897) `def run_lc_training_experiment(self, epochs)` - *Ejecutar experimento de Local Complexity .

- Entrenar un modelo desde cero hasta grokking
- Medir LC en cada época para capturar la transición de fase
- Si LC muestra un cambio alrededor del grokking, la métrica es útil
- Si LC permanece constante, la métrica NO captura la transición

Returns:
    Diccionario con historial completo del experimento*
- `run_pruning_experiment` (line 1936) `def run_pruning_experiment(self)` - *Ejecutar protocolo de poda iterativa + fine-tuning.

Returns:
    Diccionario con resultados de poda*
- `run_balanced_runs_experiment` (line 1980) `def run_balanced_runs_experiment(self, n_runs)` - *Ejecutar experimento de runs balanceados (PUNTO C DEL REVISOR).

Esto es CRUCIAL para obtener un AUC valido:
- Entrenar multiples modelos con diferentes hiperparametros
- Algunos grokkean, otros no
- Generar dataset balanceado para ROC/AUC

Returns:
    Diccionario con resultados de todos los runs*
- `run_roc_analysis` (line 2043) `def run_roc_analysis(self)` - *Ejecutar análisis ROC/AUC con bootstrap.

Returns:
    Diccionario con resultados ROC*
- `_generate_batch` (line 2117) `def _generate_batch(self, n, scale)` - *Generar batch de matrices aleatorias.*
- `generate_summary_report` (line 2124) `def generate_summary_report(self)` - *Generar reporte de resumen en markdown.*
- `save_results` (line 2197) `def save_results(self)` - *Guardar todos los resultados.*
- `run_all_experiments` (line 2235) `def run_all_experiments(self, checkpoint_path)` - *Ejecutar suite completa de experimentos.*

#### `generate_figures.py`
**Path:** `experiments/generate_figures.py`

**Functions:**
- `setup_matplotlib_for_plotting` (line 15) `def setup_matplotlib_for_plotting()` - *Configure matplotlib and seaborn for proper rendering.*
- `generate_benchmark_figure` (line 63) `def generate_benchmark_figure()` - *Generate benchmark performance comparison plot.*
- `generate_ablation_figure` (line 124) `def generate_ablation_figure()` - *Generate ablation study visualization.*
- `load_checkpoint_weights` (line 219) `def load_checkpoint_weights()` - *Load all checkpoint files and extract weight tensors.*
- `generate_weight_geometry_figure` (line 258) `def generate_weight_geometry_figure()` - *Generate weight space geometry visualization.*
- `generate_phase_transition_figure` (line 354) `def generate_phase_transition_figure()` - *Generate phase transition analysis from checkpoint evolution.*
- `generate_coherence_figure` (line 465) `def generate_coherence_figure()` - *Generate cache coherence analysis visualization.*
- `generate_crystallization_figure` (line 534) `def generate_crystallization_figure()` - *Visualize the crystallization of Strassen coefficients.*
- `main` (line 623) `def main()`

#### `coherence_analysis.py`
**Path:** `experiments/statistics/coherence_analysis.py`

**Functions:**
- `strassen_numpy` (line 15) `def strassen_numpy(A, B, threshold)`
- `run_coherence_analysis` (line 42) `def run_coherence_analysis()`

#### `rigorous_experiment.py`
**Path:** `experiments/statistics/rigorous_experiment.py`

**Classes:**
- `ExperimentConfig` (line 58) `class ExperimentConfig` - *Complete hyperparameter specification for reproducibility*
- `ExperimentResult` (line 84) `class ExperimentResult` - *Single experiment result*
- `StrassenModel` (line 106) `class StrassenModel(Module)` - *Strassen-like bilinear model*

**Methods:**
- `generate_data` (line 131) `def generate_data(n_samples, seed)` - *Generate matrix multiplication dataset*
- `compute_discretization_error` (line 143) `def compute_discretization_error(model, values)` - *Compute mean distance to nearest discrete value*
- `compute_spectral_gap` (line 157) `def compute_spectral_gap(model)` - *Compute maximum spectral gap ratio*
- `run_single_experiment` (line 169) `def run_single_experiment(batch_size, seed, run_id, config)` - *Run a single controlled experiment*
- `run_full_experiment` (line 269) `def run_full_experiment(batch_sizes, n_seeds, n_runs_per_seed)` - *Run complete factorial experiment*
- `perform_anova` (line 306) `def perform_anova(results)` - *Perform full factorial ANOVA

Returns complete ANOVA table with SS, df, MS, F, p, η²*
- `print_anova_table` (line 401) `def print_anova_table(anova)` - *Print formatted ANOVA table*
- `fit_noise_model` (line 448) `def fit_noise_model(results)` - *Fit theoretical noise model:
Var(loss) = α/B + β·cache_miss(B) + γ

Compare to null model: Var(loss) = α/B + γ*
- `find_optimal_B` (line 519) `def find_optimal_B(results, n_bootstrap)` - *Find optimal batch size with bootstrap confidence interval*
- `generate_report` (line 555) `def generate_report(results, config)` - *Generate complete statistical report*
- `__init__` (line 108) `def __init__(self, config)`
- `forward` (line 124) `def forward(self, x)`
- `cache_miss_proxy` (line 467) `def cache_miss_proxy(B)`
- `full_model` (line 472) `def full_model(B, alpha, beta, gamma)`
- `null_model` (line 476) `def null_model(B, alpha, gamma)`
- `get_mean_error` (line 526) `def get_mean_error(data, B)`

#### `benchmark.py`
**Path:** `experiments/validation/benchmark.py`

**Functions:**
- `strassen_numpy` (line 15) `def strassen_numpy(A, B, threshold)` - *Strassen recursivo con NumPy para productos base.*
- `measure_single_sgemm` (line 49) `def measure_single_sgemm(n, threads)` - *Mide tiempo de un solo sgemm de tamaño n.*
- `run_planck_analysis` (line 59) `def run_planck_analysis()` - *Ejecuta el análisis del Límite de Planck.*

#### `validation_experiments.py`
**Path:** `experiments/validation_experiments.py`

**Functions:**
- `strassen_2x2` (line 47) `def strassen_2x2(A, B, U, V, W)` - *Compute 2x2 matrix multiplication using Strassen coefficients.*
- `strassen_recursive` (line 56) `def strassen_recursive(A, B, U, V, W, threshold)` - *Recursive Strassen for NxN matrices.*
- `test_uniqueness_via_permutation` (line 85) `def test_uniqueness_via_permutation()` - *Test that permuting slots produces equivalent computation.*
- `test_noise_stability` (line 125) `def test_noise_stability()` - *Test stability under Gaussian noise.*
- `test_expansion_sizes` (line 162) `def test_expansion_sizes()` - *Test expansion to larger sizes.*
- `simulate_grokking_dynamics` (line 184) `def simulate_grokking_dynamics()` - *Simulate grokking dynamics for visualization.*
- `compute_cache_math` (line 250) `def compute_cache_math()` - *Compute L3 cache requirements for different batch sizes.*
- `main` (line 294) `def main()` - *Run all validation experiments.*
- `convert_types` (line 313) `def convert_types(obj)`

#### `verify_checkpoints.py`
**Path:** `experiments/verify_checkpoints.py`

**Classes:**
- `StrassenBilinear` (line 25) `class StrassenBilinear(Module)`

**Methods:**
- `compute_delta` (line 51) `def compute_delta(model)`
- `verify_2x2` (line 68) `def verify_2x2(U, V, W, n_test)`
- `strassen_expand` (line 89) `def strassen_expand(A, B, U, V, W)`
- `verify_expansion` (line 126) `def verify_expansion(U, V, W, sizes)`
- `compute_S_theta` (line 152) `def compute_S_theta(model)`
- `load_checkpoint` (line 166) `def load_checkpoint(path)`
- `verify_checkpoint` (line 183) `def verify_checkpoint(checkpoint_path)`
- `run_noise_stability_test` (line 226) `def run_noise_stability_test(checkpoint_path, noise_levels)`
- `main` (line 249) `def main()`
- `__init__` (line 27) `def __init__(self, rank)`
- `forward` (line 34) `def forward(self, A, B)`
- `get_discrete_coefficients` (line 44) `def get_discrete_coefficients(self)`

#### `experimetn2.py`
**Path:** `experimetn2.py`

**Classes:**
- `StrassStrassenConfig` (line 53) `class StrassStrassenConfig` - *Immutable canonical configuration for the Strassen bilinear model.*
- `TrainingConfig` (line 71) `class TrainingConfig` - *Immutable training hyperparameters for crystallization runs.*
- `SuiteConfig` (line 84) `class SuiteConfig` - *Top-level suite configuration orchestrator.*
- `StrassStrassenModel` (line 95) `class StrassStrassenModel(Module)` - *Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.
Implements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.
U, V in R^{rank x input_dim}, W in R^{output_dim x rank}.*
- `ComplexStrassStrassenModel` (line 135) `class ComplexStrassStrassenModel(Module)` - *Genuinely complex-valued bilinear model for Altland-Zirnbauer spectral testing.
Uses complex parameters and arithmetic to break time-reversal symmetry.*
- `StrassenDataGenerator` (line 178) `class StrassenDataGenerator` - *Generates random 2x2 matrix pairs and their exact products.*
- `CheckpointManager` (line 190) `class CheckpointManager` - *Handles serialization and metadata tracking of model checkpoints.*
- `LevelSpacingRatioCalculator` (line 207) `class LevelSpacingRatioCalculator` - *Computes adjacent gap ratio 'r' of eigenvalues to determine spectral class.*
- `ExactHessianCalculator` (line 244) `class ExactHessianCalculator` - *Computes the mathematically exact full-rank Hessian of the model loss.*
- `SyntheticPlanckCalculator` (line 278) `class SyntheticPlanckCalculator` - *Computes the synthetic Planck constant from parameter and loss statistics.*
- `SuperpositionMetricCalculator` (line 328) `class SuperpositionMetricCalculator` - *Measures representation density via Sparse Autoencoder (SAE) bottleneck analysis.*
- `IExperiment` (line 385) `class IExperiment(ABC)` - *Abstract interface defining the execution protocol for experiments.*
- `Experiment1RicciMBLDuality` (line 393) `class Experiment1RicciMBLDuality(IExperiment)` - *Experiment 1: Ricci-MBL Duality.
Tracks geometric curvature (Hessian Ricci scalar) against adjacent gap ratio
during a long-duration optimization trajectory to capture crystallization.*
- `Experiment2AltlandZirnbauer` (line 457) `class Experiment2AltlandZirnbauer(IExperiment)` - *Experiment 2: Altland-Zirnbauer Symmetry Dial.
Uses complex parameters and arithmetic to trigger a true symmetry crossover
from real symmetric ensembles (GOE) to complex Hermitian (GUE).*
- `Experiment3ConformalIsomorphism` (line 520) `class Experiment3ConformalIsomorphism(IExperiment)` - *Experiment 3: Conformal Isomorphism.
Conducts mathematical stress-testing of fractional-linear Möbius
transformations vs scale-conformal transformations to map the physical
limits of learned tensor models.*
- `Experiment4CompressionFrontier` (line 570) `class Experiment4CompressionFrontier(IExperiment)` - *Experiment 4: Compression Frontier.
Tests the thermodynamic bound between parameter uncertainty (hbar) and
representation superposition (psi) under varied weight decay levels.*
- `Experiment5HolographicPruning` (line 624) `class Experiment5HolographicPruning(IExperiment)` - *Experiment 5: Holographic Pruning.
Distinguishes volumetric representation structures from structured boundary-state
mechanisms via element-wise versus slot-wise pruning.*
- `UnifiedSuite` (line 699) `class UnifiedSuite` - *Orchestrates and structures the execution of the five experiments.*

**Methods:**
- `main` (line 745) `def main()`
- `__post_init__` (line 63) `def __post_init__(self)`
- `__init__` (line 101) `def __init__(self, config)`
- `forward` (line 115) `def forward(self, A, B)`
- `get_coefficients` (line 125) `def get_coefficients(self)`
- `slot_importance` (line 128) `def slot_importance(self)`
- `__init__` (line 140) `def __init__(self, config, gamma)`
- `get_complex_tensors` (line 153) `def get_complex_tensors(self)`
- `forward` (line 160) `def forward(self, A, B)`
- `__init__` (line 180) `def __init__(self, config)`
- `generate_batch` (line 183) `def generate_batch(self, batch_size)`
- `save` (line 192) `def save(self, model, epoch, metrics, path)`
- `__init__` (line 209) `def __init__(self, tolerance)`
- `calculate_r_ratio` (line 212) `def calculate_r_ratio(self, eigenvalues)`
- `__init__` (line 246) `def __init__(self, config)`
- `compute_hessian` (line 249) `def compute_hessian(self, model, A, B, C_true)`
- `__init__` (line 280) `def __init__(self, noise_floor)`
- `calculate` (line 283) `def calculate(self, model, current_loss)`
- `__init__` (line 330) `def __init__(self, config)`
- `calculate` (line 333) `def calculate(self, model, datagen)`
- `run` (line 388) `def run(self, model)`
- `get_name` (line 390) `def get_name(self)`
- `__init__` (line 399) `def __init__(self, suite_config, datagen)`
- `get_name` (line 405) `def get_name(self)`
- `run` (line 408) `def run(self, model)`
- `__init__` (line 463) `def __init__(self, suite_config, datagen)`
- `get_name` (line 468) `def get_name(self)`
- `run` (line 471) `def run(self, model)`
- `__init__` (line 527) `def __init__(self, suite_config, datagen)`
- `get_name` (line 531) `def get_name(self)`
- `run` (line 534) `def run(self, model)`
- `__init__` (line 576) `def __init__(self, suite_config, datagen)`
- `get_name` (line 582) `def get_name(self)`
- `run` (line 585) `def run(self, model)`
- `__init__` (line 630) `def __init__(self, suite_config, datagen)`
- `get_name` (line 634) `def get_name(self)`
- `run` (line 637) `def run(self, model)`
- `__init__` (line 701) `def __init__(self, config)`
- `execute_all` (line 712) `def execute_all(self)`
- `loss_fn` (line 254) `def loss_fn(flat_param_tensor)`

#### `fermi.py`
**Path:** `fermi.py`

**Classes:**
- `FermiConfig` (line 19) `class FermiConfig`
- `IModel` (line 49) `class IModel(Protocol)`
- `IBlochWaveConstructor` (line 54) `class IBlochWaveConstructor(Protocol)`
- `IBandStructureCalculator` (line 59) `class IBandStructureCalculator(Protocol)`
- `IFermiLevelCalculator` (line 64) `class IFermiLevelCalculator(Protocol)`
- `IDensityOfStatesCalculator` (line 69) `class IDensityOfStatesCalculator(Protocol)`
- `IElectronicPropertiesCalculator` (line 74) `class IElectronicPropertiesCalculator(Protocol)`
- `IMetalInsulatorClassifier` (line 79) `class IMetalInsulatorClassifier(Protocol)`
- `BilinearModel` (line 83) `class BilinearModel(Module)`
- `BlochWaveConstructor` (line 111) `class BlochWaveConstructor`
- `BandStructureCalculator` (line 135) `class BandStructureCalculator`
- `FermiLevelCalculator` (line 215) `class FermiLevelCalculator`
- `DensityOfStatesCalculator` (line 285) `class DensityOfStatesCalculator`
- `ElectronicPropertiesCalculator` (line 306) `class ElectronicPropertiesCalculator`
- `MetalInsulatorClassifier` (line 368) `class MetalInsulatorClassifier`
- `CheckpointMigrator` (line 408) `class CheckpointMigrator`
- `FermiLevelAnalyzer` (line 458) `class FermiLevelAnalyzer`
- `FermiPipeline` (line 608) `class FermiPipeline`

**Methods:**
- `main` (line 766) `def main()`
- `get_coefficients` (line 50) `def get_coefficients(self)`
- `construct` (line 55) `def construct(self, weights, k)`
- `calculate` (line 60) `def calculate(self, model)`
- `calculate` (line 65) `def calculate(self, eigenvalues, num_electrons)`
- `calculate` (line 70) `def calculate(self, eigenvalues, energies)`
- `calculate` (line 75) `def calculate(self, eigenvalues, eigenvectors, fermi_level)`
- `classify` (line 80) `def classify(self, band_gap, dos_at_fermi)`
- `__init__` (line 84) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 95) `def _initialize(self)`
- `forward` (line 100) `def forward(self, a, b)`
- `get_coefficients` (line 103) `def get_coefficients(self)`
- `__init__` (line 112) `def __init__(self, config)`
- `construct` (line 115) `def construct(self, weights, k)`
- `__init__` (line 136) `def __init__(self, config)`
- `calculate` (line 140) `def calculate(self, model)`
- `_calculate_band_gap` (line 171) `def _calculate_band_gap(self, band_structure)`
- `_calculate_effective_masses` (line 188) `def _calculate_effective_masses(self, k_points, band_structure, valence_idx, conduction_idx)`
- `_is_direct_gap` (line 208) `def _is_direct_gap(self, band_structure, valence_idx, conduction_idx)`
- `__init__` (line 216) `def __init__(self, config)`
- `calculate` (line 219) `def calculate(self, eigenvalues, num_electrons)`
- `_calculate_chemical_potential` (line 240) `def _calculate_chemical_potential(self, eigenvalues, num_electrons)`
- `_find_chemical_potential_iterative` (line 253) `def _find_chemical_potential_iterative(self, eigenvalues, num_electrons, temperature, max_iter)`
- `_fermi_dirac` (line 273) `def _fermi_dirac(self, energy, mu, temperature)`
- `__init__` (line 286) `def __init__(self, config)`
- `calculate` (line 289) `def calculate(self, eigenvalues, energies)`
- `_gaussian` (line 302) `def _gaussian(self, x, mu, sigma)`
- `__init__` (line 307) `def __init__(self, config)`
- `calculate` (line 310) `def calculate(self, eigenvalues, eigenvectors, fermi_level)`
- `_calculate_kinetic_energy` (line 337) `def _calculate_kinetic_energy(self, occupied_states)`
- `_calculate_electronic_pressure` (line 347) `def _calculate_electronic_pressure(self, eigenvalues, fermi_level)`
- `_calculate_compressibility` (line 357) `def _calculate_compressibility(self, eigenvalues, fermi_level)`
- `__init__` (line 369) `def __init__(self, config)`
- `classify` (line 372) `def classify(self, band_gap, dos_at_fermi)`
- `classify_transport` (line 386) `def classify_transport(self, effective_masses, band_gap)`
- `migrate` (line 409) `def migrate(self, raw_data, device)`
- `_migrate_dict` (line 419) `def _migrate_dict(self, state_dict, device)`
- `_migrate_custom_format` (line 428) `def _migrate_custom_format(self, state_dict, device)`
- `_migrate_coefs_format` (line 447) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 454) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 459) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 472) `def _load_checkpoint(self)`
- `analyze` (line 495) `def analyze(self)`
- `_print_report` (line 550) `def _print_report(self, results)`
- `__init__` (line 609) `def __init__(self, config)`
- `process_checkpoint` (line 612) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 626) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 653) `def generate_summary(self, all_results, output_dir)`
- `_generate_text_report` (line 686) `def _generate_text_report(self, summary, output_dir)`
- `plot_band_structures` (line 724) `def plot_band_structures(self, all_results, output_dir)` - *Generate comparison plots of band structures across checkpoints.*

#### `full_seed_prospector.py`
**Path:** `full_seed_prospector.py`

**Classes:**
- `ExecutionMode` (line 46) `class ExecutionMode(Enum)`
- `UnifiedConfig` (line 52) `class UnifiedConfig` - *Immutable unified configuration for all execution modes.*
- `IMetricCalculator` (line 162) `class IMetricCalculator(ABC)` - *Interface for metric calculation strategies.*
- `ILossComponent` (line 170) `class ILossComponent(ABC)` - *Interface for loss function components.*
- `ICheckpointManager` (line 179) `class ICheckpointManager(ABC)` - *Interface for checkpoint management.*
- `ITrainingPhase` (line 195) `class ITrainingPhase(ABC)` - *Interface for training phase execution.*
- `StrassenOperator` (line 203) `class StrassenOperator(Module)` - *Spectral operator for 2x2 matrix multiplication with Strassen structure.*
- `DeltaCalculator` (line 313) `class DeltaCalculator(IMetricCalculator)` - *Calculate delta (discretization margin) metric.*
- `AccuracyCalculator` (line 333) `class AccuracyCalculator(IMetricCalculator)` - *Calculate matrix multiplication accuracy.*
- `KappaCalculator` (line 371) `class KappaCalculator` - *Calculate kappa (gradient covariance condition number).*
- `PerelmanEntropyCalculator` (line 455) `class PerelmanEntropyCalculator(IMetricCalculator)` - *Calculate Perelman W-entropy with adaptive tau.*
- `SparsityCalculator` (line 553) `class SparsityCalculator(IMetricCalculator)` - *Calculate sparsity metrics.*
- `GradientMetricsCalculator` (line 573) `class GradientMetricsCalculator` - *Calculate gradient statistics.*
- `ResilienceSpectrometer` (line 599) `class ResilienceSpectrometer` - *Measure structural stability under progressive pruning.*
- `ComprehensiveMetricsAggregator` (line 694) `class ComprehensiveMetricsAggregator` - *Aggregate all metrics including LC, SP, kappa, delta, h_bar_eff, T_eff.*
- `AdaptiveQuantizationLoss` (line 802) `class AdaptiveQuantizationLoss(ILossComponent)` - *Adaptive quantization loss pushing towards {-1, 0, 1}.*
- `RicciCurvaturePenalty` (line 854) `class RicciCurvaturePenalty(ILossComponent)` - *Ricci scalar curvature penalty.*
- `GeometricLossAggregator` (line 868) `class GeometricLossAggregator(ILossComponent)` - *Aggregate geometric loss components.*
- `CheckpointManager` (line 884) `class CheckpointManager(ICheckpointManager)` - *Manage model checkpointing.*
- `MatrixDataGenerator` (line 930) `class MatrixDataGenerator` - *Generate random matrix data for training.*
- `DynamicBatchSizeScheduler` (line 947) `class DynamicBatchSizeScheduler` - *Schedule batch size with cosine annealing.*
- `GlassDetector` (line 969) `class GlassDetector` - *Detect glass state (non-crystallizing seeds).*
- `ProspectorPhase` (line 1025) `class ProspectorPhase(ITrainingPhase)` - *Fast prospecting phase to identify crystal seeds.*
- `LongTrainingPhase` (line 1152) `class LongTrainingPhase(ITrainingPhase)` - *Long training phase with full thermodynamic metrics.*
- `ProgressiveSparsificationPhase` (line 1335) `class ProgressiveSparsificationPhase(ITrainingPhase)` - *Progressive sparsification guided by slot importance.*
- `CoefficientDiscretizationPhase` (line 1442) `class CoefficientDiscretizationPhase(ITrainingPhase)` - *Discretize coefficients to {-1, 0, 1}.*
- `StrassenVerifier` (line 1532) `class StrassenVerifier` - *Verify Strassen algorithm correctness.*
- `CanonicalStrassenProvider` (line 1588) `class CanonicalStrassenProvider` - *Provide canonical Strassen algorithm coefficients.*
- `SeedProspector` (line 1623) `class SeedProspector` - *Prospect seeds for crystallization candidates.*
- `LongTrainingPipeline` (line 1742) `class LongTrainingPipeline` - *Long training pipeline with all phases.*
- `LocalComplexityCalculator` (line 1849) `class LocalComplexityCalculator(IMetricCalculator)` - *Calculate Local Complexity as defined in the paper.*
- `SuperpositionCalculator` (line 1915) `class SuperpositionCalculator(IMetricCalculator)` - *Calculate Superposition metrics using sparse autoencoder.*
- `ThermodynamicMetricsCalculator` (line 1964) `class ThermodynamicMetricsCalculator(IMetricCalculator)` - *Calculate thermodynamic metrics: h_bar_eff and T_eff.*

**Methods:**
- `main` (line 2022) `def main()` - *Main entry point.*
- `calculate` (line 166) `def calculate(self)`
- `compute` (line 174) `def compute(self, model, loss_mse, epoch)`
- `save` (line 183) `def save(self, state, path)`
- `load` (line 187) `def load(self, path)`
- `should_checkpoint` (line 191) `def should_checkpoint(self)`
- `execute` (line 199) `def execute(self, model)`
- `__init__` (line 206) `def __init__(self, config)`
- `_initialize_strassen_structure` (line 220) `def _initialize_strassen_structure(self)` - *Initialize with bias towards canonical Strassen structure.*
- `forward` (line 276) `def forward(self, A, B)` - *Forward pass computing C = A @ B via low-rank factorization.*
- `slot_importance` (line 291) `def slot_importance(self)` - *Calculate importance of each rank slot.*
- `count_active` (line 298) `def count_active(self, threshold)` - *Count active slots above threshold.*
- `get_flat_parameters` (line 304) `def get_flat_parameters(self)` - *Get flattened parameter vector.*
- `get_parameter_count` (line 308) `def get_parameter_count(self)` - *Get total parameter count.*
- `__init__` (line 316) `def __init__(self, config)`
- `calculate` (line 319) `def calculate(self, model)` - *Calculate delta: mean squared distance to {-1, 0, 1}.*
- `__init__` (line 336) `def __init__(self, config)`
- `calculate` (line 339) `def calculate(self, model, C_pred, C_true, n_test)` - *Calculate accuracy percentage.*
- `__init__` (line 374) `def __init__(self, config)`
- `accumulate_gradient` (line 379) `def accumulate_gradient(self, model)` - *Accumulate current gradient vector.*
- `calculate_kappa` (line 393) `def calculate_kappa(self)` - *Calculate condition number of gradient covariance matrix.*
- `get_kappa_trend` (line 425) `def get_kappa_trend(self)` - *Determine kappa trend direction.*
- `is_crystallizing` (line 439) `def is_crystallizing(self)` - *Detect if system is in crystallization phase.*
- `reset` (line 449) `def reset(self)` - *Reset calculator state.*
- `__init__` (line 458) `def __init__(self, config)`
- `calculate` (line 465) `def calculate(self, model, loss, epoch, gradient_norm)` - *Calculate W-entropy with adaptive tau coupling to GNS.*
- `_calculate_log_W` (line 522) `def _calculate_log_W(self, tau, R, grad_f_sq, f, n_params)` - *Calculate log(W) with numerical stability.*
- `reset` (line 546) `def reset(self)` - *Reset calculator state.*
- `__init__` (line 556) `def __init__(self, config)`
- `calculate` (line 559) `def calculate(self, model)` - *Calculate active slots and sparsity.*
- `calculate` (line 577) `def calculate(model)` - *Calculate gradient norm statistics.*
- `__init__` (line 602) `def __init__(self, config)`
- `measure` (line 605) `def measure(self, model)` - *Measure resilience via progressive magnitude pruning.*
- `_prune_by_magnitude` (line 684) `def _prune_by_magnitude(self, model, sparsity)` - *Prune parameters by magnitude threshold.*
- `__init__` (line 697) `def __init__(self, config)`
- `compute_all` (line 711) `def compute_all(self, model, C_pred, C_true, loss, epoch, force_kappa, force_lc, force_sp)` - *Compute all available metrics.*
- `accumulate_gradient` (line 787) `def accumulate_gradient(self, model)` - *Accumulate gradient for kappa calculation.*
- `update_lr` (line 791) `def update_lr(self, lr)` - *Update current learning rate.*
- `reset` (line 795) `def reset(self)` - *Reset all calculators.*
- `__init__` (line 805) `def __init__(self, config)`
- `compute` (line 808) `def compute(self, model, loss_mse, epoch, kappa)` - *Compute quantization loss with adaptive weighting.*
- `_get_adaptive_weight` (line 835) `def _get_adaptive_weight(self, epoch, kappa)` - *Calculate adaptive weight based on epoch and kappa.*
- `__init__` (line 857) `def __init__(self, config)`
- `compute` (line 860) `def compute(self, model, loss_mse, epoch)` - *Compute Ricci curvature penalty.*
- `__init__` (line 871) `def __init__(self, config)`
- `compute` (line 876) `def compute(self, model, loss_mse, epoch, kappa)` - *Compute total geometric loss.*
- `__init__` (line 887) `def __init__(self, config)`
- `save` (line 893) `def save(self, state, path)` - *Save checkpoint to disk.*
- `load` (line 911) `def load(self, path)` - *Load checkpoint from disk.*
- `should_checkpoint` (line 919) `def should_checkpoint(self)` - *Check if checkpoint interval has elapsed.*
- `get_latest_checkpoint_path` (line 924) `def get_latest_checkpoint_path(self)` - *Get path to latest checkpoint if exists.*
- `__init__` (line 933) `def __init__(self, config, scale)`
- `generate_batch` (line 938) `def generate_batch(self, n)` - *Generate batch of random matrices.*
- `__init__` (line 950) `def __init__(self, config)`
- `get_batch_size` (line 953) `def get_batch_size(self, epoch)` - *Get batch size for current epoch.*
- `__init__` (line 972) `def __init__(self, config)`
- `should_stop` (line 977) `def should_stop(self, epoch, metrics)` - *Determine if training should stop due to glass state.*
- `__init__` (line 1028) `def __init__(self, config, seed)`
- `execute` (line 1037) `def execute(self, model)` - *Execute prospecting phase with all metrics visible.*
- `__init__` (line 1155) `def __init__(self, config)`
- `execute` (line 1164) `def execute(self, model)` - *Execute long training phase.*
- `__init__` (line 1338) `def __init__(self, config)`
- `execute` (line 1342) `def execute(self, model)` - *Execute sparsification phase.*
- `_final_refinement` (line 1407) `def _final_refinement(self, model, optimizer, slots_to_prune)` - *Final refinement after pruning.*
- `__init__` (line 1445) `def __init__(self, config)`
- `execute` (line 1448) `def execute(self, model)` - *Execute discretization phase.*
- `__init__` (line 1535) `def __init__(self, config)`
- `verify` (line 1538) `def verify(self, U, V, W, n_test)` - *Verify algorithm on random test matrices.*
- `get_canonical` (line 1592) `def get_canonical()` - *Return canonical Strassen algorithm matrices.*
- `__init__` (line 1626) `def __init__(self, config)`
- `prospect` (line 1631) `def prospect(self, total_attempts, start_seed)` - *Prospect multiple seeds for crystals.*
- `_set_seed` (line 1733) `def _set_seed(self, seed)` - *Set random seed for reproducibility.*
- `__init__` (line 1745) `def __init__(self, config)`
- `_signal_handler` (line 1758) `def _signal_handler(self, signum, frame)` - *Handle interrupt signal.*
- `_set_seed` (line 1763) `def _set_seed(self, seed)` - *Set random seed for reproducibility.*
- `run` (line 1771) `def run(self, resume_from, seed)` - *Run complete training pipeline.*
- `__init__` (line 1852) `def __init__(self, config)`
- `calculate` (line 1855) `def calculate(self, model)` - *Calculate LC as effective local dimensionality (paper definition).
Compatible fraction approach → log(volume) can be negative.*
- `__init__` (line 1918) `def __init__(self, config)`
- `_initialize_sae` (line 1923) `def _initialize_sae(self, input_dim, device)` - *Initialize sparse autoencoder if not already done.*
- `calculate` (line 1930) `def calculate(self, model)` - *Calculate superposition coefficient psi and effective features F.*
- `__init__` (line 1967) `def __init__(self, config)`
- `calculate` (line 1970) `def calculate(self, model, gradient_covariance)` - *Calculate effective Planck constant and temperature.*

#### `grain.py`
**Path:** `grain.py`

**Classes:**
- `StrassenConfig` (line 25) `class StrassenConfig`
- `IModel` (line 65) `class IModel(Protocol)`
- `IGrainBoundaryDetector` (line 71) `class IGrainBoundaryDetector(Protocol)`
- `ILayerAnalyzer` (line 76) `class ILayerAnalyzer(Protocol)`
- `IDislocationCalculator` (line 81) `class IDislocationCalculator(Protocol)`
- `IDomainFragmentationAnalyzer` (line 86) `class IDomainFragmentationAnalyzer(Protocol)`
- `ICheckpointManager` (line 91) `class ICheckpointManager(Protocol)`
- `ITrainingMonitor` (line 97) `class ITrainingMonitor(Protocol)`
- `BilinearStrassenModel` (line 102) `class BilinearStrassenModel(Module)`
- `LayerAnalyzer` (line 130) `class LayerAnalyzer`
- `GrainBoundaryDetector` (line 153) `class GrainBoundaryDetector`
- `DomainFragmentationAnalyzer` (line 243) `class DomainFragmentationAnalyzer`
- `CheckpointManager` (line 309) `class CheckpointManager`
- `TrainingMetricsTracker` (line 340) `class TrainingMetricsTracker`
- `StrassenTrainer` (line 407) `class StrassenTrainer`
- `GrainBoundaryAnalyzer` (line 547) `class GrainBoundaryAnalyzer`
- `GrainBoundaryPipeline` (line 726) `class GrainBoundaryPipeline`

**Methods:**
- `run_training` (line 833) `def run_training(seed, config)`
- `run_analysis` (line 838) `def run_analysis(checkpoint_dir, output_dir, n_latest, config)`
- `main` (line 845) `def main()`
- `to_dict` (line 57) `def to_dict(self)`
- `forward` (line 66) `def forward(self, a, b)`
- `get_coefficients` (line 67) `def get_coefficients(self)`
- `detect` (line 72) `def detect(self, model, pruning_level)`
- `analyze_layer` (line 77) `def analyze_layer(self, weights, layer_name)`
- `calculate` (line 82) `def calculate(self, layer_deltas)`
- `analyze` (line 87) `def analyze(self, model, pruning_level)`
- `save` (line 92) `def save(self, model, epoch, metrics, path)`
- `load` (line 93) `def load(self, path)`
- `update` (line 98) `def update(self, epoch, metrics)`
- `should_checkpoint` (line 99) `def should_checkpoint(self)`
- `__init__` (line 103) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize_symmetric` (line 114) `def _initialize_symmetric(self)`
- `forward` (line 119) `def forward(self, a, b)`
- `get_coefficients` (line 122) `def get_coefficients(self)`
- `analyze_layer` (line 131) `def analyze_layer(self, weights, layer_name)`
- `__init__` (line 154) `def __init__(self, config)`
- `detect` (line 158) `def detect(self, model, pruning_level)`
- `_prune_model` (line 185) `def _prune_model(self, model, sparsity)`
- `_calculate_dislocation` (line 194) `def _calculate_dislocation(self, layer_deltas)`
- `_analyze_fragmentation` (line 227) `def _analyze_fragmentation(self, layer_analysis)`
- `__init__` (line 244) `def __init__(self, config)`
- `analyze` (line 248) `def analyze(self, model, pruning_level)`
- `_calculate_coordination_loss` (line 269) `def _calculate_coordination_loss(self, layer_analysis)`
- `_estimate_domain_count` (line 282) `def _estimate_domain_count(self, layer_analysis)`
- `_calculate_coherence_length` (line 297) `def _calculate_coherence_length(self, layer_analysis)`
- `__init__` (line 310) `def __init__(self, config)`
- `save` (line 314) `def save(self, model, epoch, metrics, path)`
- `load` (line 332) `def load(self, path)`
- `should_save` (line 335) `def should_save(self)`
- `__init__` (line 341) `def __init__(self, config)`
- `update` (line 358) `def update(self, epoch, loss, accuracy, model, grain_result)`
- `get_current_metrics` (line 382) `def get_current_metrics(self)`
- `get_training_bar_string` (line 389) `def get_training_bar_string(self, epoch, total_epochs)`
- `__init__` (line 408) `def __init__(self, config, seed)`
- `_setup_signal_handlers` (line 438) `def _setup_signal_handlers(self)`
- `_signal_handler` (line 442) `def _signal_handler(self, signum, frame)`
- `_generate_batch` (line 447) `def _generate_batch(self)`
- `_compute_accuracy` (line 468) `def _compute_accuracy(self, pred, target)`
- `_save_checkpoint` (line 473) `def _save_checkpoint(self, interrupted)`
- `train` (line 486) `def train(self)`
- `__init__` (line 548) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 557) `def _load_checkpoint(self)`
- `_migrate_checkpoint` (line 580) `def _migrate_checkpoint(self, raw_data)`
- `_migrate_dict` (line 590) `def _migrate_dict(self, state_dict)`
- `_migrate_custom_format` (line 599) `def _migrate_custom_format(self, state_dict)`
- `_migrate_coefs_format` (line 618) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 625) `def _migrate_standard_format(self, state_dict)`
- `analyze` (line 628) `def analyze(self)`
- `_analyze_dislocation_evolution` (line 658) `def _analyze_dislocation_evolution(self, grain_results)`
- `_find_critical_pruning_level` (line 681) `def _find_critical_pruning_level(self, grain_results)`
- `_print_report` (line 687) `def _print_report(self, results)`
- `__init__` (line 727) `def __init__(self, config)`
- `process_checkpoint` (line 730) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 744) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 771) `def generate_summary(self, all_results, output_dir)`
- `_generate_text_report` (line 797) `def _generate_text_report(self, summary, output_dir)`

#### `gravity.py`
**Path:** `gravity.py`

**Classes:**
- `ThermodynamicConfig` (line 24) `class ThermodynamicConfig`
- `IModel` (line 59) `class IModel(Protocol)`
- `IOrderParameterCalculator` (line 65) `class IOrderParameterCalculator(Protocol)`
- `IEntropyCalculator` (line 70) `class IEntropyCalculator(Protocol)`
- `ISpecificHeatCalculator` (line 75) `class ISpecificHeatCalculator(Protocol)`
- `IGravitationalConstantCalculator` (line 80) `class IGravitationalConstantCalculator(Protocol)`
- `ILandauerConstantCalculator` (line 85) `class ILandauerConstantCalculator(Protocol)`
- `IHeisenbergUncertaintyCalculator` (line 90) `class IHeisenbergUncertaintyCalculator(Protocol)`
- `ILocalComplexityCalculator` (line 95) `class ILocalComplexityCalculator(Protocol)`
- `IBasinStabilityCalculator` (line 100) `class IBasinStabilityCalculator(Protocol)`
- `IZeroShotTransferCalculator` (line 105) `class IZeroShotTransferCalculator(Protocol)`
- `IConditionNumberCalculator` (line 110) `class IConditionNumberCalculator(Protocol)`
- `BilinearModel` (line 114) `class BilinearModel(Module)`
- `OrderParameterCalculator` (line 142) `class OrderParameterCalculator`
- `ConfigurationEntropyCalculator` (line 154) `class ConfigurationEntropyCalculator`
- `SpecificHeatCalculator` (line 171) `class SpecificHeatCalculator`
- `GravitationalConstantCalculator` (line 183) `class GravitationalConstantCalculator`
- `LandauerConstantCalculator` (line 239) `class LandauerConstantCalculator`
- `HeisenbergUncertaintyCalculator` (line 271) `class HeisenbergUncertaintyCalculator`
- `LocalComplexityCalculator` (line 310) `class LocalComplexityCalculator`
- `BasinStabilityCalculator` (line 327) `class BasinStabilityCalculator`
- `ZeroShotTransferCalculator` (line 365) `class ZeroShotTransferCalculator`
- `ConditionNumberCalculator` (line 398) `class ConditionNumberCalculator`
- `PhaseTransitionDetector` (line 454) `class PhaseTransitionDetector`
- `ThermodynamicAnalyzer` (line 505) `class ThermodynamicAnalyzer`
- `ThermodynamicPipeline` (line 837) `class ThermodynamicPipeline`

**Methods:**
- `main` (line 1114) `def main()`
- `forward` (line 60) `def forward(self, a, b)`
- `get_coefficients` (line 61) `def get_coefficients(self)`
- `calculate` (line 66) `def calculate(self, model)`
- `calculate` (line 71) `def calculate(self, model)`
- `calculate` (line 76) `def calculate(self, loss_history)`
- `calculate` (line 81) `def calculate(self, model, gradient_history)`
- `calculate` (line 86) `def calculate(self, entropy_change, energy_dissipated)`
- `calculate` (line 91) `def calculate(self, model, temperature)`
- `calculate` (line 96) `def calculate(self, model)`
- `calculate` (line 101) `def calculate(self, model, test_data)`
- `calculate` (line 106) `def calculate(self, model, target_size)`
- `calculate` (line 111) `def calculate(self, gradient_covariance)`
- `__init__` (line 115) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 126) `def _initialize(self)`
- `forward` (line 131) `def forward(self, a, b)`
- `get_coefficients` (line 134) `def get_coefficients(self)`
- `__init__` (line 143) `def __init__(self, config)`
- `calculate` (line 146) `def calculate(self, model)`
- `__init__` (line 155) `def __init__(self, config)`
- `calculate` (line 158) `def calculate(self, model)`
- `__init__` (line 172) `def __init__(self, config)`
- `calculate` (line 175) `def calculate(self, loss_history)`
- `__init__` (line 184) `def __init__(self, config)`
- `calculate` (line 187) `def calculate(self, model, gradient_history, loss_history, static_gradient)`
- `__init__` (line 240) `def __init__(self, config)`
- `calculate` (line 243) `def calculate(self, entropy_change, energy_dissipated, has_transition, transition_window)`
- `__init__` (line 272) `def __init__(self, config)`
- `calculate` (line 275) `def calculate(self, model, temperature, static_gradient)`
- `__init__` (line 311) `def __init__(self, config)`
- `calculate` (line 314) `def calculate(self, model)`
- `__init__` (line 328) `def __init__(self, config)`
- `calculate` (line 331) `def calculate(self, model, test_data)`
- `_prune_model` (line 355) `def _prune_model(self, model, sparsity)`
- `__init__` (line 366) `def __init__(self, config)`
- `calculate` (line 369) `def calculate(self, model, target_size)`
- `_kronecker_recursive` (line 391) `def _kronecker_recursive(self, matrix, power)`
- `__init__` (line 399) `def __init__(self, config)`
- `calculate` (line 403) `def calculate(self, gradient_history, static_gradient)`
- `__init__` (line 455) `def __init__(self, config)`
- `detect` (line 458) `def detect(self, loss_history, entropy_history)`
- `__init__` (line 506) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 525) `def _load_checkpoint(self)`
- `_migrate_checkpoint` (line 548) `def _migrate_checkpoint(self, raw_data)`
- `_migrate_dict` (line 560) `def _migrate_dict(self, state_dict)`
- `_migrate_custom_format` (line 569) `def _migrate_custom_format(self, state_dict)`
- `_migrate_coefs_format` (line 588) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 595) `def _migrate_standard_format(self, state_dict)`
- `_compute_static_gradient` (line 598) `def _compute_static_gradient(self)`
- `_generate_test_data` (line 619) `def _generate_test_data(self)`
- `analyze` (line 630) `def analyze(self)`
- `_determine_failure_mode` (line 733) `def _determine_failure_mode(self, delta, basin, kappa, transition)`
- `_print_report` (line 744) `def _print_report(self, results)`
- `__init__` (line 838) `def __init__(self, config)`
- `process_checkpoint` (line 841) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 855) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 882) `def generate_summary(self, all_results, output_dir)`
- `_compute_emergent_constants` (line 904) `def _compute_emergent_constants(self, results)`
- `_verify_universal_laws` (line 959) `def _verify_universal_laws(self, results)`
- `_compute_kappa_correlation` (line 993) `def _compute_kappa_correlation(self, results)`
- `_generate_text_report` (line 1023) `def _generate_text_report(self, summary, output_dir)`
- `extract_values` (line 908) `def extract_values(data_list, key_path)`

#### `grigori_perelmans_ricci_flow.py`
**Path:** `grigori_perelmans_ricci_flow.py`

**Classes:**
- `RicciConfig` (line 34) `class RicciConfig` - *Immutable configuration for Ricci Flow analysis.*
- `BilinearStrassenModel` (line 82) `class BilinearStrassenModel(Module)` - *Bilinear model f(A,B) = W((U*A) ⊙ (V*B)).*
- `StrassenDataGenerator` (line 128) `class StrassenDataGenerator`
- `CheckpointMigrator` (line 145) `class CheckpointMigrator(ABC)`
- `CustomFormatMigrator` (line 151) `class CustomFormatMigrator(CheckpointMigrator)`
- `StandardFormatMigrator` (line 166) `class StandardFormatMigrator(CheckpointMigrator)`
- `CheckpointMigrationManager` (line 172) `class CheckpointMigrationManager`
- `RicciFlowAnalyzer` (line 193) `class RicciFlowAnalyzer` - *Calculates Ricci curvature metrics using Hessian as Metric Tensor proxy.
In Perelman's flow, dg/dt = -2Ric. 
Here we analyze instantaneous state of Metric (Hessian).*
- `SingularityEngine` (line 310) `class SingularityEngine` - *Identifies 'necks' (singularities) in the geometry and proposes 'surgery' (pruning).
A 'neck' is a parameter direction with extreme curvature (Hessian eigenvalue).*
- `GeometricPlanckCalculator` (line 366) `class GeometricPlanckCalculator` - *Estimates effective Planck constant from Spectral Geometry.
Uses the Hessian eigenvalues to define an energy spectrum.*
- `RicciFlowAnalyzerPipeline` (line 430) `class RicciFlowAnalyzerPipeline` - *Complete analysis pipeline for Ricci Flow and Planck Estimation.
Orchestrates Hessian computation, Curvature Analysis, and Physics metrics.*

**Methods:**
- `set_random_seed` (line 70) `def set_random_seed(seed)`
- `main` (line 593) `def main()`
- `__init__` (line 86) `def __init__(self, config)`
- `_initialize` (line 94) `def _initialize(self)`
- `forward` (line 99) `def forward(self, a, b)`
- `get_coefficients` (line 102) `def get_coefficients(self)`
- `get_flat_params` (line 109) `def get_flat_params(self)` - *Return all parameters as a single flattened vector.*
- `set_flat_params` (line 114) `def set_flat_params(self, flat_params)` - *Set model parameters from a flattened vector.*
- `generate_batch` (line 130) `def generate_batch(batch_size, config)`
- `can_migrate` (line 147) `def can_migrate(self, state_dict)`
- `migrate` (line 149) `def migrate(self, state_dict)`
- `can_migrate` (line 152) `def can_migrate(self, state_dict)`
- `migrate` (line 154) `def migrate(self, state_dict)`
- `can_migrate` (line 167) `def can_migrate(self, state_dict)`
- `migrate` (line 169) `def migrate(self, state_dict)`
- `__init__` (line 173) `def __init__(self)`
- `migrate_checkpoint` (line 176) `def migrate_checkpoint(self, path, device)`
- `__init__` (line 199) `def __init__(self, model, config)`
- `compute_hessian` (line 203) `def compute_hessian(self, input_a, input_b, target_c)` - *Computes exact Hessian of loss w.r.t parameters.
H = d^2L / dtheta^2*
- `_loss_wrapper` (line 223) `def _loss_wrapper(self, flat_params, original_params, a, b, c)` - *Wrapper to compute loss from flat param vector.*
- `_compute_diagonal_hessian` (line 240) `def _compute_diagonal_hessian(self, a, b, c)` - *Approximation: Diagonal of Hessian (Gauss-Newton).*
- `analyze_curvature` (line 247) `def analyze_curvature(self, hessian)` - *Analyze Hessian spectrum to derive Ricci Scalar and Topological invariants.*
- `compute_heat_kernel_trace` (line 285) `def compute_heat_kernel_trace(self, eigenvalues, t)` - *Trace of Heat Kernel: Z(t) = Sum( exp(-lambda_i * t) ).
Relates to Partition Function in Quantum Mechanics.*
- `compute_topological_entropy` (line 296) `def compute_topological_entropy(self, eigenvalues)` - *von Neumann Entropy / Spectral Entropy.
S = - Sum( p_i * log(p_i) ) where p_i are normalized eigenvalue weights.*
- `__init__` (line 316) `def __init__(self, model, eigenvalues, config)`
- `detect_necks` (line 325) `def detect_necks(self, curvature_analysis)` - *Identify if the system is in a 'bottleneck' state.*
- `propose_surgery` (line 336) `def propose_surgery(self)` - *Propose parameters to 'cut' (prune) based on curvature heuristics.
In Strassen, the 'bias' slot (8th) often carries the 'noise' or singular connection.*
- `__init__` (line 371) `def __init__(self, eigenvalues, ricci_scalar, config)`
- `calculate` (line 376) `def calculate(self)`
- `_get_spectral_gap` (line 412) `def _get_spectral_gap(self)`
- `_compute_spectral_entropy` (line 420) `def _compute_spectral_entropy(self)`
- `__init__` (line 435) `def __init__(self, config)`
- `analyze_checkpoint` (line 439) `def analyze_checkpoint(self, checkpoint_path, device)` - *Perform complete analysis of a single checkpoint.*
- `analyze_directory` (line 527) `def analyze_directory(self, directory, device, pattern)`
- `_print_summary` (line 560) `def _print_summary(self, report)`

#### `hawking_radiation.py`
**Path:** `hawking_radiation.py`

**Classes:**
- `CustomUnpickler` (line 36) `class CustomUnpickler(Unpickler)` - *Custom unpickler that handles unknown classes by creating dummy objects.

This solves the "Can't get attribute 'UnifiedConfig'" error by providing
fallback objects for any class that can't be found.*
- `HawkingConfiguration` (line 138) `class HawkingConfiguration` - *Immutable configuration for Hawking radiation analysis.*
- `IModel` (line 188) `class IModel(Protocol)`
- `BilinearStrassenModel` (line 197) `class BilinearStrassenModel(Module)` - *Bilinear model for Strassen matrix multiplication.*
- `RobustCheckpointMigrator` (line 234) `class RobustCheckpointMigrator` - *Enhanced checkpoint migrator that handles:
- Direct U, V, W tensors
- U_coefs format
- Standard .weight format
- Nested structures with config
- Encoder format
- State dicts within state dicts*
- `MetadataExtractor` (line 514) `class MetadataExtractor` - *Extracts metadata from various checkpoint formats.*
- `GravitationalConstantCalculator` (line 609) `class GravitationalConstantCalculator` - *Calculates effective gravitational constant G_alg.*
- `PlanckConstantCalculator` (line 676) `class PlanckConstantCalculator` - *Calculates effective Planck constant h_bar_eff.*
- `BoltzmannConstantCalculator` (line 768) `class BoltzmannConstantCalculator` - *Calculates effective Boltzmann constant k_B_eff.*
- `SpeedOfLightCalculator` (line 823) `class SpeedOfLightCalculator` - *Calculates effective speed of light c_eff.*
- `InformationalMassCalculator` (line 883) `class InformationalMassCalculator` - *Calculates effective mass M_eff.*
- `HorizonAreaCalculator` (line 939) `class HorizonAreaCalculator` - *Calculates effective area A_eff.*
- `HawkingRadiationCalculator` (line 991) `class HawkingRadiationCalculator` - *Calculates Hawking radiation metrics.*
- `RobustHawkingAnalyzer` (line 1166) `class RobustHawkingAnalyzer` - *Robust analyzer that handles exotic checkpoint formats.*
- `DummyClass` (line 64) `class DummyClass`

**Methods:**
- `load_checkpoint_robust` (line 93) `def load_checkpoint_robust(path, device)` - *Load checkpoint with robust handling of custom classes.

Tries multiple loading strategies in order:
1. Standard torch.load
2. Custom unpickler
3. Weights_only mode with manual extraction*
- `main` (line 1406) `def main()`
- `find_class` (line 44) `def find_class(self, module, name)` - *Override find_class to handle missing classes gracefully.*
- `_create_dummy_class` (line 62) `def _create_dummy_class(self, name)` - *Create a dummy class that can hold attributes.*
- `get_effective_input_dim` (line 172) `def get_effective_input_dim(self)`
- `get_total_parameters` (line 175) `def get_total_parameters(self)`
- `get_coefficients` (line 189) `def get_coefficients(self)`
- `forward` (line 190) `def forward(self, a, b)`
- `__init__` (line 200) `def __init__(self, config)`
- `_initialize` (line 211) `def _initialize(self)`
- `forward` (line 216) `def forward(self, a, b)`
- `get_coefficients` (line 219) `def get_coefficients(self)`
- `get_flat_parameters` (line 226) `def get_flat_parameters(self)`
- `migrate` (line 245) `def migrate(self, raw_data, device)` - *Main migration entry point with multiple strategies.*
- `_try_extract_state_dict` (line 269) `def _try_extract_state_dict(self, data, device)` - *Try standard state dict extraction methods.*
- `_try_nested_extraction` (line 288) `def _try_nested_extraction(self, data, device)` - *Try to extract from nested structures.*
- `_try_direct_tensor_extraction` (line 313) `def _try_direct_tensor_extraction(self, data, device)` - *Try to extract tensors directly from any structure.*
- `_reconstruct_from_tensors` (line 338) `def _reconstruct_from_tensors(self, tensors, device)` - *Reconstruct U, V, W from found tensors.*
- `_is_state_dict` (line 386) `def _is_state_dict(self, data)` - *Check if dict looks like a state dict.*
- `_migrate_dict` (line 398) `def _migrate_dict(self, state_dict, device)` - *Migrate a state dict to the expected format.*
- `_migrate_custom_format` (line 424) `def _migrate_custom_format(self, state_dict, device)`
- `_migrate_coefs_format` (line 449) `def _migrate_coefs_format(self, state_dict, device)`
- `_migrate_standard_format` (line 456) `def _migrate_standard_format(self, state_dict, device)`
- `_migrate_encoder_format` (line 466) `def _migrate_encoder_format(self, state_dict, device)` - *Handle encoder.layers style checkpoints.*
- `_migrate_prefixed_format` (line 496) `def _migrate_prefixed_format(self, state_dict, prefix, device)` - *Handle prefixed state dict keys.*
- `extract` (line 518) `def extract(checkpoint)` - *Extract all relevant metadata from checkpoint.*
- `_extract_delta` (line 571) `def _extract_delta(data, depth)` - *Recursively search for delta in nested structures.*
- `__init__` (line 612) `def __init__(self, config)`
- `calculate` (line 615) `def calculate(self, model, gradient, precomputed_delta)`
- `__init__` (line 679) `def __init__(self, config)`
- `calculate` (line 682) `def calculate(self, model, loss, precomputed_delta)`
- `__init__` (line 771) `def __init__(self, config)`
- `calculate` (line 774) `def calculate(self, model, loss, loss_history)`
- `__init__` (line 826) `def __init__(self, config)`
- `calculate` (line 829) `def calculate(self, model, h_bar, G_alg)`
- `__init__` (line 886) `def __init__(self, config)`
- `calculate` (line 889) `def calculate(self, model, G_alg, c_eff, h_bar)`
- `__init__` (line 942) `def __init__(self, config)`
- `calculate` (line 945) `def calculate(self, model, M_eff)`
- `__init__` (line 994) `def __init__(self, config)`
- `calculate_all` (line 1003) `def calculate_all(self, model, loss, loss_history, gradient, precomputed_delta)` - *Calculate all Hawking radiation metrics.*
- `_classify_state` (line 1151) `def _classify_state(self, delta, T_hawking)`
- `__init__` (line 1169) `def __init__(self, config)`
- `analyze_checkpoint` (line 1174) `def analyze_checkpoint(self, checkpoint_path)` - *Analyze a single checkpoint with robust error handling.*
- `_compute_gradient` (line 1227) `def _compute_gradient(self, model)` - *Compute gradient on random batch.*
- `_print_report` (line 1255) `def _print_report(self, results)` - *Print formatted report.*
- `analyze_directory` (line 1295) `def analyze_directory(self, checkpoint_dir, output_dir, pattern)` - *Analyze all checkpoints in directory.*
- `_generate_summary` (line 1342) `def _generate_summary(self, results, errors)` - *Generate aggregate summary.*
- `extract_tensors` (line 317) `def extract_tensors(obj, prefix)`
- `__init__` (line 65) `def __init__(self)`
- `__repr__` (line 72) `def __repr__(self)`
- `__getitem__` (line 75) `def __getitem__(self, key)`
- `keys` (line 78) `def keys(self)`
- `values` (line 81) `def values(self)`
- `items` (line 84) `def items(self)`
- `get` (line 87) `def get(self, key, default)`

#### `maxwell_strassen_analysis.py`
**Path:** `maxwell_strassen_analysis.py`

**Classes:**
- `MaxwellConfiguration` (line 42) `class MaxwellConfiguration` - *Configuration for Maxwellian analysis of Strassen crystals.

Physics Parameters:
-------------------
LATTICE_CONSTANT: Spacing between weight nodes (arbitrary units, e.g., nm scale).
PERMITTIVITY_VACUUM: ε_0 (scaled for numerical stability).
PERMITTIVITY_WEIGHT_SCALE: Factor to convert weight magnitude to dielectric contrast.

Simulation Parameters:
----------------------
GRID_DIMENSION: Size of the cubic lattice for weight embedding (N x N x N).
FREQUENCY_SAMPLES: Number of frequency points for spectral analysis.
WAVEVECTOR_RANGE: Range of k-vectors for scattering simulation.

Analysis Thresholds:
--------------------
CRYSTALLINITY_THRESHOLD: Entropy threshold to classify as Crystal vs Glass.
BRAGG_PEAK_SHARPNESS: Minimum prominence for a peak to be considered Bragg scattering.*
- `IModel` (line 116) `class IModel(Protocol)`
- `IGeometryMapper` (line 121) `class IGeometryMapper(Protocol)`
- `IMaxwellSolver` (line 126) `class IMaxwellSolver(Protocol)`
- `IDielectricAnalyzer` (line 132) `class IDielectricAnalyzer(Protocol)`
- `IPhaseClassifier` (line 137) `class IPhaseClassifier(Protocol)`
- `BilinearStrassenModel` (line 145) `class BilinearStrassenModel(Module)`
- `CheckpointMigrator` (line 171) `class CheckpointMigrator`
- `StrassenGeometryMapper` (line 227) `class StrassenGeometryMapper` - *Maps abstract weight vectors to a 3D Dielectric Lattice.

Strategy:
1. Flatten U, V, W into a single vector.
2. Populate a 3D grid (NxNxN) using a space-filling curve (Z-order) 
   or layer-wise assignment.
3. Compute Effective Charge Density ρ and Permittivity ε.

Crystal Hypothesis:
- Crystallized weights (discrete values) form ordered arrays.
- Glass weights (random) form noise.*
- `DielectricTensorAnalyzer` (line 288) `class DielectricTensorAnalyzer` - *Analyzes the anisotropy of the dielectric medium.

Glass: Isotropic (ε ~ scalar)
Crystal: Anisotropic (ε ~ tensor with specific principal axes)*
- `MaxwellScatteringSolver` (line 350) `class MaxwellScatteringSolver` - *Solves Maxwell's equations in the frequency domain via FFT.

Key Analysis:
1. Electrostatics: Solve ∇·(ε∇φ) = -ρ.
2. Scattering: Calculate Fourier Transform of Dielectric contrast.
   - Crystal: Sharp Bragg Peaks at k-vectors determined by lattice periodicity.
   - Glass: Broad diffuse scattering (Rayleigh).*
- `PhotonicEntropyCalculator` (line 466) `class PhotonicEntropyCalculator` - *Calculates the entropy of the electromagnetic field distribution.
S = -∑ p(E) log p(E)

Glass: High Entropy (Disordered field).
Crystal: Low Entropy (Ordered, localized modes).*
- `BandgapAnalyzer` (line 498) `class BandgapAnalyzer` - *Estimates if a photonic bandgap exists.
A bandgap implies certain frequencies cannot propagate.

We use the Fourier coefficients to estimate the gap.*
- `CrystalPhaseClassifier` (line 544) `class CrystalPhaseClassifier` - *Classifies the material phase (Crystal vs Glass) based on EM metrics.*
- `CheckpointManager` (line 605) `class CheckpointManager`
- `MaxwellVisualizer` (line 625) `class MaxwellVisualizer`
- `MaxwellAnalyzer` (line 677) `class MaxwellAnalyzer`

**Methods:**
- `main` (line 788) `def main()`
- `get_effective_input_dim` (line 103) `def get_effective_input_dim(self)`
- `get_total_parameters` (line 106) `def get_total_parameters(self)`
- `get_coefficients` (line 117) `def get_coefficients(self)`
- `map_weights_to_lattice` (line 122) `def map_weights_to_lattice(self, weights)`
- `solve_poisson` (line 127) `def solve_poisson(self, charge_density, permittivity)`
- `compute_scattering` (line 128) `def compute_scattering(self, permittivity)`
- `analyze_permittivity_tensor` (line 133) `def analyze_permittivity_tensor(self, permittivity)`
- `classify` (line 138) `def classify(self, metrics)`
- `__init__` (line 146) `def __init__(self, config)`
- `_initialize_weights` (line 156) `def _initialize_weights(self)`
- `forward` (line 161) `def forward(self, a, b)`
- `get_coefficients` (line 164) `def get_coefficients(self)`
- `migrate` (line 172) `def migrate(self, raw_data, config)`
- `_migrate_dict` (line 185) `def _migrate_dict(self, state_dict, config)`
- `_migrate_custom` (line 194) `def _migrate_custom(self, sd, config)`
- `_migrate_coefs` (line 203) `def _migrate_coefs(self, sd)`
- `_migrate_standard` (line 208) `def _migrate_standard(self, sd)`
- `_np` (line 217) `def _np(tensor)`
- `__init__` (line 241) `def __init__(self, config)`
- `map_weights_to_lattice` (line 244) `def map_weights_to_lattice(self, weights)` - *Returns:
    charge_density: 3D array of charge distribution.
    permittivity: 3D array of dielectric constants.*
- `__init__` (line 295) `def __init__(self, config)`
- `analyze_permittivity_tensor` (line 298) `def analyze_permittivity_tensor(self, permittivity)` - *Analyze the effective permittivity tensor of the medium.*
- `__init__` (line 360) `def __init__(self, config)`
- `solve_poisson` (line 363) `def solve_poisson(self, charge_density, permittivity)` - *Solves Poisson equation for Electric Potential φ.
Using Spectral Method (FFT):
∇²φ = -ρ / ε_0

For variable ε (heterogeneous medium), this is approximate.
We use the convolution theorem for the Green's function.*
- `compute_scattering` (line 401) `def compute_scattering(self, permittivity)` - *Compute the Scattering Amplitude S(k).
S(k) ∝ |FT(Δε)|²

Δε = ε(r) - ε_avg (Dielectric contrast)

Returns:
    Dict containing scattering intensity map and peak analysis.*
- `_find_peaks` (line 438) `def _find_peaks(self, intensity)` - *Detect sharp peaks indicative of crystallinity.*
- `__init__` (line 474) `def __init__(self, config)`
- `calculate` (line 477) `def calculate(self, potential, intensity)`
- `__init__` (line 505) `def __init__(self, config)`
- `analyze` (line 508) `def analyze(self, fourier_coeffs)`
- `__init__` (line 548) `def __init__(self, config)`
- `classify` (line 551) `def classify(self, em_metrics, purity_metrics)` - *Decision Logic:
1. If Purity Alpha > Threshold AND Discretization is high -> Pre-cursor.
2. If Anisotropy is High -> Crystal Order.
3. If Scattering shows Bragg Peaks -> Long Range Order.
4. If Entropy is Low -> Ordered System.*
- `__init__` (line 606) `def __init__(self, config)`
- `should_save` (line 610) `def should_save(self)`
- `save` (line 613) `def save(self, data, output_dir)`
- `__init__` (line 626) `def __init__(self, config)`
- `visualize_lattice` (line 629) `def visualize_lattice(self, permittivity, output_dir, name)`
- `visualize_scattering` (line 648) `def visualize_scattering(self, scattering_slice, output_dir, name)`
- `visualize_potential` (line 659) `def visualize_potential(self, potential, output_dir, name)`
- `__init__` (line 678) `def __init__(self, config)`
- `_calculate_purity_metrics` (line 690) `def _calculate_purity_metrics(self, weights)`
- `analyze_checkpoint` (line 701) `def analyze_checkpoint(self, checkpoint_path, output_dir)` - *Executes the full Maxwellian analysis pipeline on a single checkpoint.*
- `generate_report` (line 765) `def generate_report(self, results, output_dir)`

#### `mbl_analyzer.py`
**Path:** `mbl_analyzer.py`
**File Doc:** *Remove tqdm dependency and use standard library*

**Classes:**
- `MBLConfiguration` (line 24) `class MBLConfiguration` - *Comprehensive configuration for MBL analysis of Strassen algorithm crystallization.
All parameters are centralized here following SOLID principles.*
- `IModel` (line 93) `class IModel(Protocol)` - *Protocol for models compatible with MBL analysis.*
- `ILevelSpacingCalculator` (line 100) `class ILevelSpacingCalculator(Protocol)` - *Protocol for level spacing ratio calculation.*
- `IParticipationRatioCalculator` (line 106) `class IParticipationRatioCalculator(Protocol)` - *Protocol for participation ratio calculation.*
- `ISyntheticPlanckCalculator` (line 112) `class ISyntheticPlanckCalculator(Protocol)` - *Protocol for synthetic Planck's constant calculation.*
- `IDiscretizationDialAnalyzer` (line 118) `class IDiscretizationDialAnalyzer(Protocol)` - *Protocol for discretization dial analysis.*
- `ICheckpointManager` (line 124) `class ICheckpointManager(Protocol)` - *Protocol for checkpoint management.*
- `ITrainingMetricsCollector` (line 132) `class ITrainingMetricsCollector(Protocol)` - *Protocol for collecting all training metrics.*
- `BilinearStrassenModel` (line 138) `class BilinearStrassenModel(Module)` - *Bilinear model for Strassen algorithm implementation.
Represents the 2x2 matrix multiplication with hidden dimension expansion.*
- `LevelSpacingRatioCalculator` (line 205) `class LevelSpacingRatioCalculator` - *Calculates the level spacing ratio r for MBL phase detection.

The ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})
where delta_n = E_{n+1} - E_n (energy level spacing).

References:
- Oganesyan & Huse (2008): r_WD ≈ 0.53 (Wigner-Dyson, thermal)
- Poisson statistics: r_P ≈ 0.386 (MBL/localized phase)*
- `ParticipationRatioCalculator` (line 321) `class ParticipationRatioCalculator` - *Calculates Inverse Participation Ratio (IPR) for localization analysis.

IPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.
IPR = 1 for fully localized state, IPR = 1/N for fully delocalized state.

Used to quantify the 'crystallinity' of the weight distribution.*
- `SyntheticPlanckConstantCalculator` (line 420) `class SyntheticPlanckConstantCalculator` - *Calculates effective synthetic Planck's constant (hbar_eff) from model properties.

Based on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)
where PR is the Participation Ratio and Energy_Gap is the spectral gap.

This represents the quantum of action in the synthetic quantum system.*
- `DiscretizationDialAnalyzer` (line 486) `class DiscretizationDialAnalyzer` - *Analyzes the discretization parameter delta as a phase transition control.

The discretization delta measures how close weights are to discrete values.
It acts as a "dial" that controls the quantum-classical transition.

This implements the noise robustness test: applying Gaussian perturbations
and measuring when the energy gap collapses (loss of quantum protection).*
- `PurityIndexCalculator` (line 614) `class PurityIndexCalculator` - *Original purity calculation preserved exactly as in user's code.
Calculates the 'crystallinity' of the weight distribution.*
- `EffectiveTemperatureCalculator` (line 674) `class EffectiveTemperatureCalculator` - *Original temperature calculation preserved exactly.*
- `PhaseClassifier` (line 721) `class PhaseClassifier` - *Original phase classification preserved exactly.*
- `CheckpointMigrator` (line 746) `class CheckpointMigrator` - *Original checkpoint migration logic preserved exactly.*
- `MBLCheckpointManager` (line 800) `class MBLCheckpointManager` - *Manages checkpoint saving with 5-minute intervals and latest file maintenance.*
- `MBLMetricsCollector` (line 855) `class MBLMetricsCollector` - *Collects all MBL metrics for comprehensive training monitoring.*
- `MBLCheckpointAnalyzer` (line 962) `class MBLCheckpointAnalyzer` - *Comprehensive analyzer for MBL metrics from checkpoints.*
- `MBLAnalysisPipeline` (line 1102) `class MBLAnalysisPipeline` - *Main pipeline for processing checkpoints and generating reports.*

**Methods:**
- `main` (line 1227) `def main()`
- `get_effective_input_dim` (line 84) `def get_effective_input_dim(self)`
- `get_total_parameters` (line 87) `def get_total_parameters(self)`
- `get_coefficients` (line 95) `def get_coefficients(self)`
- `forward` (line 96) `def forward(self, a, b)`
- `calculate` (line 102) `def calculate(self, model)`
- `calculate` (line 108) `def calculate(self, model)`
- `calculate` (line 114) `def calculate(self, participation_ratio, energy_gap)`
- `analyze_robustness` (line 120) `def analyze_robustness(self, model, noise_levels)`
- `save_checkpoint` (line 126) `def save_checkpoint(self, model, epoch, metrics, loss_history, path)`
- `load_checkpoint` (line 128) `def load_checkpoint(self, path)`
- `collect` (line 134) `def collect(self, model, loss, epoch, loss_history)`
- `__init__` (line 143) `def __init__(self, config)`
- `_initialize_weights` (line 154) `def _initialize_weights(self)` - *Xavier initialization with symmetry constraint for U and V.*
- `forward` (line 160) `def forward(self, a, b)` - *Forward pass implementing bilinear multiplication.*
- `get_coefficients` (line 164) `def get_coefficients(self)` - *Returns weight matrices for analysis.*
- `get_flat_parameters` (line 172) `def get_flat_parameters(self)` - *Returns all parameters flattened for Hamiltonian construction.*
- `construct_hessian_approximation` (line 179) `def construct_hessian_approximation(self)` - *Constructs approximate Hessian matrix from weight correlations.
This serves as the 'Hamiltonian' for MBL analysis.*
- `__init__` (line 217) `def __init__(self, config)`
- `calculate` (line 220) `def calculate(self, model)` - *Calculate level spacing statistics from model weights.

Returns:
    Dictionary containing mean ratio, variance, and phase classification.*
- `_construct_hessian_from_weights` (line 267) `def _construct_hessian_from_weights(self, model)` - *Alternative Hessian construction for generic models.*
- `_compute_eigenvalues` (line 283) `def _compute_eigenvalues(self, hessian)` - *Compute sorted eigenvalues of the Hamiltonian.*
- `_calculate_spacing_ratios` (line 288) `def _calculate_spacing_ratios(self, spacings)` - *Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).*
- `_classify_phase` (line 303) `def _classify_phase(self, mean_ratio)` - *Classify the quantum phase based on level spacing ratio.*
- `__init__` (line 331) `def __init__(self, config)`
- `calculate` (line 334) `def calculate(self, model)` - *Calculate participation ratios for all weight layers.

Returns:
    Dictionary containing global and layer-wise IPR metrics.*
- `_calculate_ipr` (line 380) `def _calculate_ipr(self, coefficients)` - *Calculate standard Inverse Participation Ratio.
IPR = sum_i |c_i|^4 / (sum_i |c_i|^2)^2*
- `_calculate_renyi_ipr` (line 395) `def _calculate_renyi_ipr(self, coefficients, q)` - *Calculate q-th order Rényi IPR.
I_q = sum_i |c_i|^{2q} / (sum_i |c_i|^2)^q*
- `_calculate_fractal_dimension` (line 409) `def _calculate_fractal_dimension(self, ipr, n)` - *Calculate fractal dimension D_q from IPR.
IPR ~ N^{-D_q} => D_q = -log(IPR) / log(N)*
- `__init__` (line 430) `def __init__(self, config)`
- `calculate` (line 433) `def calculate(self, participation_ratio, energy_gap)` - *Calculate synthetic Planck's constant.

Args:
    participation_ratio: Inverse participation ratio (measure of localization)
    energy_gap: Energy gap from spectrum (measure of quantum discreteness)

Returns:
    Synthetic hbar value representing the quantum scale of the system.*
- `calculate_from_model` (line 456) `def calculate_from_model(self, model, level_spacing_results, pr_results)` - *Comprehensive calculation from model and previous analyses.*
- `__init__` (line 497) `def __init__(self, config)`
- `calculate_base_discretization` (line 501) `def calculate_base_discretization(self, model)` - *Calculate the base discretization level from weight rounding error.*
- `analyze_robustness` (line 528) `def analyze_robustness(self, model, noise_levels)` - *Test robustness by applying noise and measuring gap collapse.

Args:
    model: The neural network model
    noise_levels: Tuple of noise magnitudes to test

Returns:
    Dictionary containing robustness metrics and phase transition points.*
- `_perturb_and_measure` (line 584) `def _perturb_and_measure(self, model, noise_level)` - *Apply noise to model and measure resulting metrics.*
- `_delta_to_alpha` (line 607) `def _delta_to_alpha(self, delta)` - *Convert discretization error to purity alpha.*
- `__init__` (line 620) `def __init__(self, config)`
- `calculate` (line 623) `def calculate(self, model)`
- `_compute_layer_purity` (line 652) `def _compute_layer_purity(self, weights)`
- `_delta_to_alpha` (line 658) `def _delta_to_alpha(self, delta)`
- `_assess_purity_quality` (line 663) `def _assess_purity_quality(self, alpha, variance)`
- `__init__` (line 679) `def __init__(self, config)`
- `calculate` (line 682) `def calculate(self, loss_history)`
- `__init__` (line 726) `def __init__(self, config)`
- `classify` (line 729) `def classify(self, alpha, temperature)`
- `migrate` (line 751) `def migrate(self, raw_data, device)`
- `_migrate_dict` (line 761) `def _migrate_dict(self, state_dict, device)`
- `_migrate_custom_format` (line 770) `def _migrate_custom_format(self, state_dict, device)`
- `_migrate_coefs_format` (line 789) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 796) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 805) `def __init__(self, config)`
- `should_save_checkpoint` (line 810) `def should_save_checkpoint(self)` - *Check if 5 minutes have elapsed since last checkpoint.*
- `save_checkpoint` (line 816) `def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)` - *Save checkpoint with all MBL metrics.*
- `load_checkpoint` (line 850) `def load_checkpoint(self, path)` - *Load checkpoint with automatic device placement.*
- `__init__` (line 860) `def __init__(self, config)`
- `collect` (line 870) `def collect(self, model, loss, epoch, loss_history)` - *Collect all metrics for the current training state.*
- `_classify_quantum_phase` (line 946) `def _classify_quantum_phase(self, level_spacing, hbar_results)` - *Classify the combined quantum phase.*
- `__init__` (line 967) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 975) `def _load_checkpoint(self)` - *Load and migrate checkpoint to model.*
- `analyze` (line 998) `def analyze(self)` - *Perform complete MBL analysis on checkpoint.*
- `_generate_summary` (line 1026) `def _generate_summary(self, metrics, robustness)` - *Generate executive summary of analysis.*
- `_print_report` (line 1043) `def _print_report(self, results)` - *Print formatted analysis report.*
- `__init__` (line 1107) `def __init__(self, config)`
- `process_checkpoint` (line 1110) `def process_checkpoint(self, checkpoint_path, output_dir)` - *Process single checkpoint and save results.*
- `process_directory` (line 1125) `def process_directory(self, checkpoint_dir, n_latest, output_dir)` - *Process multiple checkpoints from directory.*
- `generate_summary` (line 1155) `def generate_summary(self, all_results, output_dir)` - *Generate aggregate summary report.*
- `_generate_text_report` (line 1188) `def _generate_text_report(self, summary, output_dir)` - *Generate human-readable text report.*

#### `measure_strassen.py`
**Path:** `measure_strassen.py`

*No symbols extracted*

#### `menu.py`
**Path:** `menu.py`

**Functions:**
- `clear_screen` (line 304) `def clear_screen()`
- `print_header` (line 308) `def print_header(title, subtitle)`
- `print_wrapped` (line 318) `def print_wrapped(text, indent)`
- `wait_for_enter` (line 323) `def wait_for_enter()`
- `run_script` (line 332) `def run_script(entry)`
- `show_checkpoints` (line 362) `def show_checkpoints()`
- `show_results` (line 397) `def show_results()`
- `show_category` (line 433) `def show_category(cat)`
- `main_menu` (line 473) `def main_menu()`

#### `percolation_analysis.py`
**Path:** `percolation_analysis.py`

**Classes:**
- `PercolationConfiguration` (line 31) `class PercolationConfiguration`
- `IModel` (line 97) `class IModel(Protocol)`
- `NumpyModelWrapper` (line 101) `class NumpyModelWrapper`
- `_DummyObject` (line 148) `class _DummyObject` - *Transparent stand-in for any unpicklable class found inside a checkpoint.
Stores all keyword and positional constructor arguments as attributes so
that downstream dict-key lookups still work on objects that behave like
namespaces (e.g. UnifiedConfig, TrainingConfig, etc.).*
- `CheckpointMigrator` (line 264) `class CheckpointMigrator`
- `WeightGraphConstructor` (line 350) `class WeightGraphConstructor`
- `BondPercolationAnalyzer` (line 402) `class BondPercolationAnalyzer`
- `SitePercolationAnalyzer` (line 488) `class SitePercolationAnalyzer`
- `PruningPercolationAnalyzer` (line 542) `class PruningPercolationAnalyzer`
- `ClusterSizeDistributionAnalyzer` (line 725) `class ClusterSizeDistributionAnalyzer`
- `PercolationUniversalityAnalyzer` (line 775) `class PercolationUniversalityAnalyzer`
- `PercolationCheckpointManager` (line 808) `class PercolationCheckpointManager`
- `PercolationVisualizationEngine` (line 837) `class PercolationVisualizationEngine`
- `PercolationReportGenerator` (line 1049) `class PercolationReportGenerator`
- `PercolationAnalysisPipeline` (line 1139) `class PercolationAnalysisPipeline`
- `BilinearStrassenModel` (line 117) `class BilinearStrassenModel(Module)`

**Methods:**
- `_safe_torch_load` (line 165) `def _safe_torch_load(path)` - *Load a torch checkpoint that may contain unknown serialized classes
(UnifiedConfig, TrainingConfig, custom dataclasses, etc.).

Strategy:
  1. Try normal torch.load.  If it works, return immediately.
  2. On AttributeError / ModuleNotFoundError, extract the missing
     class names from the exception, inject _DummyObject into
     sys.modules / __main__ under those names, and retry.
  3. Repeat up to MAX_RETRIES times (each retry may reveal a
     new missing class that was nested deeper in the pickle stream).
  4. Clean up injected names after loading.

This keeps torch's own deserialization path (including correct
persistent_load for tensor storage) fully intact.*
- `main` (line 1264) `def main()`
- `get_effective_input_dim` (line 82) `def get_effective_input_dim(self)`
- `get_total_parameters` (line 85) `def get_total_parameters(self)`
- `get_percolation_thresholds` (line 89) `def get_percolation_thresholds(self)`
- `get_coefficients` (line 98) `def get_coefficients(self)`
- `__init__` (line 102) `def __init__(self, weights)`
- `get_coefficients` (line 105) `def get_coefficients(self)`
- `get_flat_parameters` (line 108) `def get_flat_parameters(self)`
- `__init__` (line 155) `def __init__(self)`
- `__repr__` (line 161) `def __repr__(self)`
- `_patch_missing` (line 189) `def _patch_missing(exc)`
- `_cleanup` (line 230) `def _cleanup()`
- `migrate` (line 265) `def migrate(self, raw_data, config)`
- `_migrate_dict` (line 285) `def _migrate_dict(self, state_dict, config)`
- `_try_migrate_nested` (line 300) `def _try_migrate_nested(self, candidate, config)`
- `_migrate_custom` (line 310) `def _migrate_custom(self, sd, config)`
- `_migrate_coefs` (line 326) `def _migrate_coefs(self, sd)`
- `_migrate_standard` (line 331) `def _migrate_standard(self, sd)`
- `_np` (line 340) `def _np(tensor)`
- `__init__` (line 351) `def __init__(self, config)`
- `construct_adjacency_from_weights` (line 354) `def construct_adjacency_from_weights(self, weights)`
- `construct_weight_correlation_graph` (line 375) `def construct_weight_correlation_graph(self, weights)`
- `construct_slot_interaction_graph` (line 386) `def construct_slot_interaction_graph(self, weights)`
- `__init__` (line 403) `def __init__(self, config)`
- `analyze` (line 406) `def analyze(self, adjacency, thresholds)`
- `_susceptibility` (line 449) `def _susceptibility(self, sizes, n)`
- `_find_pc` (line 456) `def _find_pc(self, thresholds, res)`
- `_exponents` (line 462) `def _exponents(self, thresholds, res, pc)`
- `_fit_pl` (line 478) `def _fit_pl(x, y)`
- `__init__` (line 489) `def __init__(self, config)`
- `analyze` (line 492) `def analyze(self, weights, thresholds)`
- `__init__` (line 543) `def __init__(self, config)`
- `analyze` (line 546) `def analyze(self, weights)`
- `_d2a` (line 626) `def _d2a(self, delta)`
- `_kappa` (line 631) `def _kappa(self, wv)`
- `_hbar` (line 644) `def _hbar(self, wv)`
- `_teff` (line 656) `def _teff(self, wv)`
- `_lc` (line 659) `def _lc(self, wv, delta)`
- `_entropy` (line 669) `def _entropy(self, wv)`
- `_ipr` (line 680) `def _ipr(self, wv)`
- `_lsr` (line 687) `def _lsr(self, wv)`
- `_fractal` (line 699) `def _fractal(self, ipr, n)`
- `_phase` (line 704) `def _phase(self, alpha, temp, delta)`
- `__init__` (line 726) `def __init__(self, config)`
- `analyze_at_threshold` (line 729) `def analyze_at_threshold(self, adjacency, threshold)`
- `_tau` (line 753) `def _tau(self, sizes)`
- `_critical` (line 767) `def _critical(self, sizes, n)`
- `__init__` (line 776) `def __init__(self, config)`
- `classify_universality` (line 779) `def classify_universality(self, measured)`
- `__init__` (line 809) `def __init__(self, config)`
- `should_save` (line 814) `def should_save(self)`
- `save` (line 817) `def save(self, results, output_dir)`
- `load` (line 829) `def load(self, output_dir)`
- `__init__` (line 838) `def __init__(self, config)`
- `generate_all_figures` (line 841) `def generate_all_figures(self, results, output_dir)`
- `_plot_bond` (line 856) `def _plot_bond(self, data, out)`
- `_plot_pruning` (line 889) `def _plot_pruning(self, data, out)`
- `_plot_dashboard` (line 952) `def _plot_dashboard(self, data, out)`
- `_plot_site` (line 993) `def _plot_site(self, data, out)`
- `_plot_cluster` (line 1019) `def _plot_cluster(self, data, out)`
- `__init__` (line 1050) `def __init__(self, config)`
- `generate_text_report` (line 1053) `def generate_text_report(self, results, output_dir)`
- `generate_json_report` (line 1131) `def generate_json_report(self, results, output_dir)`
- `__init__` (line 1140) `def __init__(self, config)`
- `_load_weights` (line 1153) `def _load_weights(self, checkpoint_path)`
- `process_checkpoint` (line 1164) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 1212) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `_maybe_save` (line 1234) `def _maybe_save(self, results, output_dir)`
- `_comparative_summary` (line 1238) `def _comparative_summary(self, all_res, output_dir)`
- `__init__` (line 118) `def __init__(self, config)`
- `_initialize_weights` (line 127) `def _initialize_weights(self)`
- `forward` (line 132) `def forward(self, a, b)`
- `get_coefficients` (line 135) `def get_coefficients(self)`
- `get_flat_parameters` (line 141) `def get_flat_parameters(self)`

#### `plank.py`
**Path:** `plank.py`

**Classes:**
- `Configuration` (line 32) `class Configuration` - *Immutable configuration container following Single Responsibility Principle.*
- `BilinearStrassenModel` (line 127) `class BilinearStrassenModel(Module)` - *Bilinear model for Strassen matrix multiplication.
Implements f(A,B) = W((U*A) ⊙ (V*B)) where ⊙ is element-wise product.*
- `CheckpointMigrator` (line 194) `class CheckpointMigrator(ABC)` - *Abstract base for checkpoint migration strategies.*
- `CustomFormatMigrator` (line 208) `class CustomFormatMigrator(CheckpointMigrator)` - *Handles custom U,V,W direct formats.*
- `EncoderFormatMigrator` (line 237) `class EncoderFormatMigrator(CheckpointMigrator)` - *Handles encoder.layers format.*
- `StandardFormatMigrator` (line 270) `class StandardFormatMigrator(CheckpointMigrator)` - *Handles standard U.weight, V.weight, W.weight format.*
- `CheckpointMigrationManager` (line 284) `class CheckpointMigrationManager` - *Manages multiple migration strategies.*
- `StrassenDataGenerator` (line 341) `class StrassenDataGenerator` - *Generates training data for 2x2 matrix multiplication.*
- `CrystallographyMetrics` (line 384) `class CrystallographyMetrics` - *Computes crystallographic quality metrics for Strassen models.*
- `StrassenDiffractionTest` (line 483) `class StrassenDiffractionTest` - *Tests gauge invariance through permutation symmetry.*
- `BasinResilienceSpectrometer` (line 560) `class BasinResilienceSpectrometer` - *Measures basin of attraction through noise injection and recovery.*
- `CrystalPurityIndex` (line 677) `class CrystalPurityIndex` - *Computes normalized purity index from component metrics.*
- `PlanckConstantCalculator` (line 763) `class PlanckConstantCalculator` - *Calculates effective Planck constant from Strassen model parameters.

Maps crystallographic metrics to quantum thermodynamic quantities.*
- `StrassenCheckpointLoader` (line 936) `class StrassenCheckpointLoader` - *Loads and migrates Strassen checkpoints with fallback strategies.*
- `StrassenPlanckAnalyzer` (line 1014) `class StrassenPlanckAnalyzer` - *Complete analysis pipeline for Strassen checkpoints.

Orchestrates crystallographic analysis and Planck constant calculation.*
- `ReportGenerator` (line 1165) `class ReportGenerator` - *Generates reports and visualizations from analysis results.*

**Methods:**
- `set_random_seed` (line 114) `def set_random_seed(seed)` - *Set random seeds for reproducibility.*
- `parse_arguments` (line 1339) `def parse_arguments()` - *Parse command line arguments.*
- `create_config_from_args` (line 1387) `def create_config_from_args(args)` - *Create configuration from command line arguments.*
- `main` (line 1399) `def main()` - *Main execution entry point.*
- `__post_init__` (line 100) `def __post_init__(self)` - *Validate configuration parameters.*
- `__init__` (line 133) `def __init__(self, config)`
- `_initialize_symmetric` (line 143) `def _initialize_symmetric(self)` - *Initialize with Xavier uniform, symmetric U and V.*
- `forward` (line 149) `def forward(self, matrix_a, matrix_b)` - *Forward pass computing approximate matrix product.

Args:
    matrix_a: Flattened input matrix A [batch, INPUT_DIM]
    matrix_b: Flattened input matrix B [batch, INPUT_DIM]

Returns:
    Approximate product C = A @ B [batch, OUTPUT_DIM]*
- `get_coefficients` (line 166) `def get_coefficients(self)` - *Return current coefficient matrices.*
- `compute_lambda_effective` (line 174) `def compute_lambda_effective(self)` - *Compute effective lambda (confinement potential) from weight magnitudes.
Derived from weight decay interpretation as harmonic confinement.*
- `can_migrate` (line 198) `def can_migrate(self, state_dict)` - *Check if this strategy can handle the given state dict.*
- `migrate` (line 203) `def migrate(self, state_dict)` - *Migrate state dict to standard format.*
- `can_migrate` (line 211) `def can_migrate(self, state_dict)`
- `migrate` (line 214) `def migrate(self, state_dict)`
- `can_migrate` (line 240) `def can_migrate(self, state_dict)`
- `migrate` (line 243) `def migrate(self, state_dict)`
- `can_migrate` (line 273) `def can_migrate(self, state_dict)`
- `migrate` (line 276) `def migrate(self, state_dict)`
- `__init__` (line 287) `def __init__(self)`
- `migrate_checkpoint` (line 294) `def migrate_checkpoint(self, path, device)` - *Attempt to migrate checkpoint using available strategies.

Args:
    path: Path to checkpoint file
    device: Device to load tensors to

Returns:
    Migrated state dict or None if migration fails*
- `generate_batch` (line 345) `def generate_batch(batch_size, config)` - *Generate a batch of random matrix pairs and their products.

Returns:
    Tuple of (A_flat, B_flat, C_flat) where C = A @ B*
- `verify_structure` (line 366) `def verify_structure(coeffs, config)` - *Verify if coefficients represent valid Strassen structure.*
- `compute_kappa` (line 388) `def compute_kappa(model, num_batches, config)` - *Compute condition number of gradient covariance matrix.

High kappa indicates ill-conditioned optimization landscape.
Low kappa (approaching 1.0) indicates well-conditioned, crystalline structure.*
- `compute_discretization_margin` (line 429) `def compute_discretization_margin(coeffs)` - *Compute maximum deviation from nearest integer values.

Delta measures how close coefficients are to discrete (integer) values,
indicating crystalline structure formation.*
- `compute_local_complexity` (line 445) `def compute_local_complexity(model, config)` - *Compute local complexity based on active parameters.

From "Can't Stop Won't Stop" paper - measures effective parameter count.*
- `compute_all_metrics` (line 464) `def compute_all_metrics(model, config)` - *Compute all crystallographic metrics at once.*
- `__init__` (line 486) `def __init__(self, model, config)`
- `test_gauge_invariance` (line 490) `def test_gauge_invariance(self)` - *Test if model exhibits true Strassen structure through permutation invariance.

Genuine Strassen algorithm should have exactly one valid permutation (identity).*
- `_compute_functional_error` (line 531) `def _compute_functional_error(self, test_coeffs)` - *Compute functional error between original and permuted coefficients.*
- `__init__` (line 563) `def __init__(self, model, config)`
- `measure_resilience_spectrum` (line 570) `def measure_resilience_spectrum(self)` - *Measure resilience across multiple noise levels.

Returns spectrum showing critical noise level where recovery fails.*
- `_test_noise_recovery` (line 585) `def _test_noise_recovery(self, sigma)` - *Test recovery from noise level sigma.*
- `_apply_noise` (line 614) `def _apply_noise(self, sigma)` - *Apply Gaussian noise to model parameters.*
- `_anneal_to_attractor` (line 621) `def _anneal_to_attractor(self)` - *Anneal model back to attractor using fine-tuning.

Returns number of epochs needed for recovery.*
- `_estimate_critical_noise` (line 654) `def _estimate_critical_noise(self, results)` - *Estimate critical noise level where success rate drops below 50%.*
- `__init__` (line 680) `def __init__(self, metrics, diffraction_results, resilience_results, config)`
- `compute` (line 692) `def compute(self)` - *Compute normalized purity index and grade.*
- `_assign_grade` (line 745) `def _assign_grade(self, index, delta)` - *Assign crystallographic grade based on delta (primary indicator).*
- `__init__` (line 770) `def __init__(self, metrics, training_metrics, config)`
- `calculate_all` (line 789) `def calculate_all(self)` - *Execute all Planck constant calculation methods.*
- `_determine_regime_and_weights` (line 875) `def _determine_regime_and_weights(self)` - *Determine confinement regime and corresponding weights.*
- `_compute_derived_constants` (line 886) `def _compute_derived_constants(self, h_bar)` - *Compute derived Planck-scale constants.*
- `_compute_universe_comparison` (line 917) `def _compute_universe_comparison(self, h_bar)` - *Compare calculated constants with physical universe.*
- `__init__` (line 939) `def __init__(self, config)`
- `load` (line 943) `def load(self, checkpoint_path, device)` - *Load checkpoint into model instance.

Args:
    checkpoint_path: Path to checkpoint file
    device: Target device

Returns:
    Loaded model or None if loading fails*
- `extract_training_metrics` (line 986) `def extract_training_metrics(self, checkpoint_path)` - *Extract training metrics from checkpoint if available.*
- `__init__` (line 1021) `def __init__(self, config)`
- `analyze_checkpoint` (line 1025) `def analyze_checkpoint(self, checkpoint_path, device)` - *Perform complete analysis of a single checkpoint.

Args:
    checkpoint_path: Path to checkpoint file
    device: Computation device

Returns:
    Complete analysis report*
- `analyze_directory` (line 1104) `def analyze_directory(self, directory, device, pattern)` - *Analyze all checkpoints in a directory.

Args:
    directory: Directory containing checkpoints
    device: Computation device
    pattern: File pattern to match

Returns:
    List of analysis reports*
- `_print_summary` (line 1146) `def _print_summary(self, report)` - *Print formatted summary of analysis results.*
- `__init__` (line 1168) `def __init__(self, config)`
- `save_json_report` (line 1173) `def save_json_report(self, report, suffix)` - *Save individual report as JSON.*
- `save_aggregate_report` (line 1185) `def save_aggregate_report(self, results)` - *Save aggregate report from multiple analyses.*
- `_compute_statistics` (line 1216) `def _compute_statistics(self, summaries)` - *Compute aggregate statistics from summaries.*
- `_count_grades` (line 1247) `def _count_grades(self, summaries)` - *Count distribution of grades.*
- `generate_visualizations` (line 1255) `def generate_visualizations(self, results)` - *Generate visualization plots.*

#### `purity_index.py`
**Path:** `purity_index.py`

**Classes:**
- `PurityConfig` (line 19) `class PurityConfig`
- `IModel` (line 44) `class IModel(Protocol)`
- `IPurityIndexCalculator` (line 49) `class IPurityIndexCalculator(Protocol)`
- `IEffectiveTemperatureCalculator` (line 54) `class IEffectiveTemperatureCalculator(Protocol)`
- `IPhaseClassifier` (line 59) `class IPhaseClassifier(Protocol)`
- `IPolycrystalAnalyzer` (line 64) `class IPolycrystalAnalyzer(Protocol)`
- `IPurityComparator` (line 69) `class IPurityComparator(Protocol)`
- `BilinearModel` (line 73) `class BilinearModel(Module)`
- `PurityIndexCalculator` (line 101) `class PurityIndexCalculator`
- `EffectiveTemperatureCalculator` (line 156) `class EffectiveTemperatureCalculator`
- `PhaseClassifier` (line 199) `class PhaseClassifier`
- `PolycrystalAnalyzer` (line 236) `class PolycrystalAnalyzer`
- `PurityComparator` (line 274) `class PurityComparator`
- `CheckpointMigrator` (line 311) `class CheckpointMigrator`
- `PurityAnalyzer` (line 361) `class PurityAnalyzer`
- `PurityPipeline` (line 492) `class PurityPipeline`

**Methods:**
- `main` (line 611) `def main()`
- `get_coefficients` (line 45) `def get_coefficients(self)`
- `calculate` (line 50) `def calculate(self, model)`
- `calculate` (line 55) `def calculate(self, loss_history)`
- `classify` (line 60) `def classify(self, alpha, temperature)`
- `analyze_polycrystal` (line 65) `def analyze_polycrystal(self, model, pruning_level)`
- `compare` (line 70) `def compare(self, original, polycrystal)`
- `__init__` (line 74) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 85) `def _initialize(self)`
- `forward` (line 90) `def forward(self, a, b)`
- `get_coefficients` (line 93) `def get_coefficients(self)`
- `__init__` (line 102) `def __init__(self, config)`
- `calculate` (line 105) `def calculate(self, model)`
- `_compute_layer_purity` (line 134) `def _compute_layer_purity(self, weights)`
- `_delta_to_alpha` (line 140) `def _delta_to_alpha(self, delta)`
- `_assess_purity_quality` (line 145) `def _assess_purity_quality(self, alpha, variance)`
- `__init__` (line 157) `def __init__(self, config)`
- `calculate` (line 160) `def calculate(self, loss_history)`
- `__init__` (line 200) `def __init__(self, config)`
- `classify` (line 203) `def classify(self, alpha, temperature)`
- `classify_polycrystal_state` (line 219) `def classify_polycrystal_state(self, original_alpha, original_temp, poly_alpha, poly_temp)`
- `__init__` (line 237) `def __init__(self, config)`
- `analyze_polycrystal` (line 243) `def analyze_polycrystal(self, model, pruning_level, loss_history)`
- `_prune_model` (line 264) `def _prune_model(self, model, sparsity)`
- `__init__` (line 275) `def __init__(self, config)`
- `compare` (line 279) `def compare(self, original, polycrystal)`
- `migrate` (line 312) `def migrate(self, raw_data, device)`
- `_migrate_dict` (line 322) `def _migrate_dict(self, state_dict, device)`
- `_migrate_custom_format` (line 331) `def _migrate_custom_format(self, state_dict, device)`
- `_migrate_coefs_format` (line 350) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 357) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 362) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 375) `def _load_checkpoint(self)`
- `analyze` (line 399) `def analyze(self)`
- `_print_report` (line 445) `def _print_report(self, results)`
- `__init__` (line 493) `def __init__(self, config)`
- `process_checkpoint` (line 496) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 510) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 537) `def generate_summary(self, all_results, output_dir)`
- `_generate_text_report` (line 574) `def _generate_text_report(self, summary, output_dir)`

#### `repor_experiments.py`
**Path:** `repor_experiments.py`

**Classes:**
- `ModelConfig` (line 78) `class ModelConfig`
- `TrainConfig` (line 86) `class TrainConfig`
- `Exp1Config` (line 96) `class Exp1Config`
- `Exp2Config` (line 103) `class Exp2Config`
- `Exp3Config` (line 108) `class Exp3Config`
- `Exp4Config` (line 114) `class Exp4Config`
- `Exp5Config` (line 120) `class Exp5Config`
- `SuiteConfig` (line 126) `class SuiteConfig`
- `BilinearModel` (line 141) `class BilinearModel(Module)`

**Methods:**
- `discretize_q` (line 173) `def discretize_q(w)`
- `compute_delta` (line 176) `def compute_delta(U, V, W)`
- `phase2` (line 181) `def phase2(model)` - *Poda a target_rank slots, discretiza a {-1,0,1}, verifica.*
- `_verify_2x2` (line 204) `def _verify_2x2(U, V, W, n)`
- `zero_shot_verify` (line 214) `def zero_shot_verify(U, V, W, sizes)`
- `_recursive_strassen` (line 222) `def _recursive_strassen(U, V, W, n, trials)`
- `_strassen_rec` (line 234) `def _strassen_rec(A, B, U, V, W, n)`
- `compute_kappa` (line 262) `def compute_kappa(model, num_batches, bs)`
- `compute_alpha` (line 279) `def compute_alpha(delta)`
- `compute_teff` (line 284) `def compute_teff(model, num_batches, bs)`
- `classify_phase` (line 298) `def classify_phase(delta)`
- `load_checkpoint` (line 309) `def load_checkpoint(path, device)`
- `_extract_state` (line 329) `def _extract_state(d)`
- `train_model` (line 351) `def train_model(cfg, model, epochs, bs, wd, lr, callback)`
- `analyze_checkpoint` (line 378) `def analyze_checkpoint(path, device)`
- `experiment1` (line 428) `def experiment1(cfg)`
- `experiment2` (line 475) `def experiment2(cfg)`
- `experiment3` (line 507) `def experiment3(cfg)`
- `experiment4` (line 563) `def experiment4(cfg)`
- `experiment5` (line 607) `def experiment5(cfg)`
- `_test_accuracy` (line 633) `def _test_accuracy(model, n, device)`
- `_random_prune` (line 644) `def _random_prune(model, fraction)`
- `_boundary_prune` (line 655) `def _boundary_prune(model, fraction)`
- `analyze_checkpoints` (line 665) `def analyze_checkpoints(ckpt_dir, device)`
- `main` (line 705) `def main()`
- `_save` (line 783) `def _save(data, path)`
- `__init__` (line 142) `def __init__(self, cfg)`
- `forward` (line 149) `def forward(self, A, B)`
- `slot_importance` (line 156) `def slot_importance(self)`
- `get_weights` (line 163) `def get_weights(self)`
- `get_flat` (line 167) `def get_flat(self)`
- `cb` (line 434) `def cb(ep, m, loss, acc)`

#### `scrodingger.py`
**Path:** `scrodingger.py`

**Classes:**
- `SchrodingerConfig` (line 19) `class SchrodingerConfig`
- `IModel` (line 44) `class IModel(Protocol)`
- `IWaveFunctionExtractor` (line 49) `class IWaveFunctionExtractor(Protocol)`
- `IPotentialCalculator` (line 54) `class IPotentialCalculator(Protocol)`
- `IHamiltonianConstructor` (line 59) `class IHamiltonianConstructor(Protocol)`
- `IEigenvalueSolver` (line 64) `class IEigenvalueSolver(Protocol)`
- `ITimeEvolver` (line 69) `class ITimeEvolver(Protocol)`
- `IExpectationValueCalculator` (line 74) `class IExpectationValueCalculator(Protocol)`
- `IUncertaintyCalculator` (line 79) `class IUncertaintyCalculator(Protocol)`
- `ICheckpointLoader` (line 84) `class ICheckpointLoader(Protocol)`
- `ICheckpointMigrator` (line 89) `class ICheckpointMigrator(Protocol)`
- `BilinearModel` (line 93) `class BilinearModel(Module)`
- `WaveFunctionExtractor` (line 121) `class WaveFunctionExtractor`
- `PotentialCalculator` (line 128) `class PotentialCalculator`
- `HamiltonianConstructor` (line 150) `class HamiltonianConstructor`
- `EigenvalueSolver` (line 170) `class EigenvalueSolver`
- `TimeEvolver` (line 194) `class TimeEvolver`
- `ExpectationValueCalculator` (line 216) `class ExpectationValueCalculator`
- `UncertaintyCalculator` (line 226) `class UncertaintyCalculator`
- `CheckpointLoader` (line 263) `class CheckpointLoader`
- `CheckpointMigrator` (line 271) `class CheckpointMigrator`
- `SchrodingerAnalyzer` (line 320) `class SchrodingerAnalyzer`
- `WaveFunctionVisualizer` (line 542) `class WaveFunctionVisualizer`
- `SchrodingerPipeline` (line 609) `class SchrodingerPipeline`

**Methods:**
- `main` (line 779) `def main()`
- `get_coefficients` (line 45) `def get_coefficients(self)`
- `extract` (line 50) `def extract(self, model)`
- `calculate` (line 55) `def calculate(self, weights)`
- `construct` (line 60) `def construct(self, potential, mass)`
- `solve` (line 65) `def solve(self, hamiltonian, count)`
- `evolve` (line 70) `def evolve(self, initial_state, hamiltonian, time_steps, dt)`
- `calculate` (line 75) `def calculate(self, wave_function, operator)`
- `calculate` (line 80) `def calculate(self, wave_function, position_grid)`
- `load` (line 85) `def load(self, path, device)`
- `migrate` (line 90) `def migrate(self, raw_data)`
- `__init__` (line 94) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 105) `def _initialize(self)`
- `forward` (line 110) `def forward(self, a, b)`
- `get_coefficients` (line 113) `def get_coefficients(self)`
- `extract` (line 122) `def extract(self, model)`
- `__init__` (line 129) `def __init__(self, config)`
- `calculate` (line 132) `def calculate(self, weights)`
- `__init__` (line 151) `def __init__(self, config)`
- `construct` (line 154) `def construct(self, potential, mass)`
- `__init__` (line 171) `def __init__(self, config)`
- `solve` (line 174) `def solve(self, hamiltonian, count)`
- `__init__` (line 195) `def __init__(self, config)`
- `evolve` (line 198) `def evolve(self, initial_state, hamiltonian, time_steps, dt)`
- `calculate` (line 217) `def calculate(self, wave_function, operator)`
- `__init__` (line 227) `def __init__(self, config)`
- `calculate` (line 230) `def calculate(self, wave_function, position_grid)`
- `load` (line 264) `def load(self, path, device)`
- `migrate` (line 272) `def migrate(self, raw_data)`
- `_migrate_dict` (line 284) `def _migrate_dict(self, state_dict)`
- `_migrate_custom_format` (line 293) `def _migrate_custom_format(self, state_dict)`
- `_migrate_coefs_format` (line 309) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 316) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 321) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 335) `def _load_checkpoint(self)`
- `analyze` (line 357) `def analyze(self)`
- `_calculate_tunneling_probability` (line 451) `def _calculate_tunneling_probability(self, potential, wave_function)`
- `_count_degeneracy` (line 465) `def _count_degeneracy(self, eigenvalues)`
- `_print_report` (line 478) `def _print_report(self, results)`
- `__init__` (line 543) `def __init__(self, config)`
- `visualize` (line 546) `def visualize(self, data, output_path)`
- `__init__` (line 610) `def __init__(self, config)`
- `process_checkpoint` (line 614) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 628) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 655) `def generate_summary(self, all_results, output_dir)`
- `_generate_text_report` (line 718) `def _generate_text_report(self, summary, output_dir)`

#### `benchmark_final.py`
**Path:** `src/benchmarks/benchmark_final.py`

**Functions:**
- `strassen_hybrid_multiply` (line 45) `def strassen_hybrid_multiply(A, B)` - *Multiply using our Strassen Hybrid implementation*
- `numpy_multiply` (line 60) `def numpy_multiply(A, B)` - *Standard NumPy BLAS multiplication*
- `benchmark` (line 64) `def benchmark(func, A, B, warmup, runs)` - *Run benchmark with warmup*
- `main` (line 80) `def main()`

#### `benchmark_scientific.py`
**Path:** `src/benchmarks/benchmark_scientific.py`

**Functions:**
- `strassen_multiply` (line 38) `def strassen_multiply(A, B)`
- `standard_avx512_multiply` (line 51) `def standard_avx512_multiply(A, B)`
- `numpy_multiply` (line 64) `def numpy_multiply(A, B)`
- `benchmark_function` (line 67) `def benchmark_function(func, A, B, runs, warmup)` - *Benchmark with statistical analysis*
- `main` (line 91) `def main()`

#### `benchmark_strassen.py`
**Path:** `src/benchmarks/benchmark_strassen.py`

**Classes:**
- `BenchmarkResult` (line 29) `class BenchmarkResult`
- `BenchmarkConfig` (line 45) `class BenchmarkConfig`

**Methods:**
- `load_config` (line 61) `def load_config(config_path)` - *Load configuration from TOML file.*
- `get_dtype` (line 93) `def get_dtype(dtype_str)` - *Convert dtype string to torch dtype.*
- `estimate_memory_mb` (line 104) `def estimate_memory_mb(n, dtype, batch_size)` - *Estimate memory usage for matrix multiplication.*
- `benchmark_resolution` (line 114) `def benchmark_resolution(n, cfg, dtype)` - *Benchmark Strassen vs standard matmul for given resolution.*
- `run_benchmark` (line 218) `def run_benchmark(cfg)` - *Run full benchmark suite.*
- `save_results` (line 310) `def save_results(results, filepath)` - *Save benchmark results to JSON.*
- `main` (line 322) `def main()`

#### `strassen_numpy.py`
**Path:** `src/benchmarks/strassen_numpy.py`

**Functions:**
- `_load_weights` (line 19) `def _load_weights()`
- `strassen_2x2_numpy` (line 29) `def strassen_2x2_numpy(A, B)` - *Strassen 2x2 using grokked coefficients.*
- `strassen_numpy` (line 44) `def strassen_numpy(A, B)` - *Recursive Strassen using NumPy.*
- `strassen_hybrid` (line 78) `def strassen_hybrid(A, B, threshold)` - *Hybrid Strassen: use Strassen for large matrices, NumPy for small.
This is faster because NumPy matmul is highly optimized for small matrices.*
- `multiplication_count` (line 112) `def multiplication_count(n)` - *Count multiplications used by Strassen.*

#### `auto_T_discovery.py`
**Path:** `src/discovery/auto_T_discovery.py`

**Classes:**
- `SymmetryStructure` (line 22) `class SymmetryStructure` - *Discovered symmetries in weight matrix*
- `AutoTDiscovery` (line 32) `class AutoTDiscovery` - *Automatic discovery of expansion operator T from converged weights.

The algorithm works in three phases:
1. Spectral Analysis: Extract dominant singular subspace
2. Symmetry Detection: Find block/permutation structure
3. T Construction: Build expansion operator preserving invariants*

**Methods:**
- `verify_strassen_T` (line 306) `def verify_strassen_T(model_path, target_sizes)` - *Verify T discovery on Strassen model*
- `verify_expanded_correctness` (line 353) `def verify_expanded_correctness(U, V, W, target_size, expanded)` - *Verify that expanded operator correctly computes matrix multiplication*
- `recursive_strassen_multiply` (line 389) `def recursive_strassen_multiply(A, B, U, V, W, base_size)` - *Recursively apply learned Strassen decomposition

This is the IMPLEMENTATION of T: it shows how the base 2x2 decomposition
extends to arbitrary sizes via recursive block application.*
- `__init__` (line 42) `def __init__(self, tolerance, verbose)`
- `analyze_structure` (line 46) `def analyze_structure(self, W)` - *Phase 1 & 2: Analyze weight matrix structure*
- `_detect_discrete_values` (line 89) `def _detect_discrete_values(self, W_flat)` - *Detect if weights cluster around discrete values*
- `_detect_block_structure` (line 105) `def _detect_block_structure(self, W)` - *Detect repeating block patterns*
- `_block_repetition_score` (line 127) `def _block_repetition_score(self, W, bm, bn)` - *Score how well blocks repeat (lower = more repetitive)*
- `_detect_symmetry_type` (line 144) `def _detect_symmetry_type(self, W)` - *Detect type of symmetry in weight matrix*
- `_is_permutation_symmetric` (line 165) `def _is_permutation_symmetric(self, W)` - *Check if rows are permutations of a base pattern*
- `_is_cyclic` (line 171) `def _is_cyclic(self, W)` - *Check for cyclic/Toeplitz structure*
- `_invariant_subspace_dim` (line 184) `def _invariant_subspace_dim(self, U, S, rank)` - *Estimate dimension of truly invariant subspace*
- `_discretization_error` (line 198) `def _discretization_error(self, W, values)` - *Compute error when discretizing to given values*
- `_print_analysis` (line 213) `def _print_analysis(self, W, S, structure)` - *Print analysis results*
- `construct_T` (line 229) `def construct_T(self, W_dict, target_size)` - *Phase 3: Construct expansion operator T

For matrix multiplication (U, V, W tensors), T expands via
recursive block structure discovered from the base case.

Args:
    W_dict: Dictionary with 'U', 'V', 'W' tensors
    target_size: Target matrix dimension (n' in the paper)

Returns:
    Expanded weight dictionary*
- `_validate_expansion` (line 290) `def _validate_expansion(self, expanded, structure)` - *Validate that expansion preserves key invariants*

#### `convergence_theory.py`
**Path:** `src/training/convergence_theory.py`

**Classes:**
- `ConvergenceMetrics` (line 21) `class ConvergenceMetrics` - *Metrics for tracking convergence to algorithmic invariance*
- `HutchinsonTraceEstimator` (line 31) `class HutchinsonTraceEstimator` - *Efficient Hessian trace estimation using Hutchinson's method.

tr(H) ≈ E[v^T H v] where v ~ Rademacher(±1)

Complexity: O(n_samples * forward_backward_pass) instead of O(n²)*
- `HardwareNoiseEstimator` (line 136) `class HardwareNoiseEstimator` - *Estimate ε_hw(B, T) - hardware-induced variance

Extended model addressing reviewer's criticism:
ε_hw(B, T) = α/B + β*cache_miss_rate(B) + γ*thread_contention(T)*
- `SimpleStrassenModel` (line 346) `class SimpleStrassenModel(Module)` - *Simple model for testing convergence verification*

**Methods:**
- `convergence_theorem` (line 193) `def convergence_theorem()` - *THEOREM (Convergence to Algorithmic Invariance)

Let W_t denote the weights at step t under SGD with:
    W_{t+1} = W_t - η∇L(W_t) + ξ_t

where ξ_t is stochastic gradient noise with Var(ξ_t) = σ²/B + ε_hw(B,T).

ASSUMPTIONS:
A1. The target function f admits a rank-r tensor decomposition
A2. The loss L is twice differentiable with Lipschitz Hessian
A3. The effective curvature satisfies κ_eff < 0 after step t₀

THEOREM: Under A1-A3, if
    
    Var(ξ_t) < σ_min(W*)² / (η · condition(H))

where σ_min(W*) is the smallest non-zero singular value of the optimal
decomposition, then with probability 1-δ:

    lim_{t->∞} d(W_t, W*) = 0

where d is the subspace distance and W* is the algorithmically invariant solution.

PROOF SKETCH:

1. By A3 (κ_eff < 0), the loss landscape is locally convex near convergence

2. The noise condition ensures gradient updates don't overshoot the
   invariant subspace defined by σ_min(W*)

3. By spectral gap analysis, W_t projects onto the dominant singular
   subspace with increasing precision as t -> ∞

4. The discretization {-1, 0, 1} emerges because integer solutions
   are fixed points of the projection when noise is controlled

IMPLICATIONS FOR T:

The expansion operator T is constructible because:
- T preserves the dominant singular subspace (by definition)
- The rank-r structure is independent of problem scale (by A1)
- Therefore T = block_embed(W_r) where W_r is the converged rank-r solution*
- `verify_convergence_conditions` (line 265) `def verify_convergence_conditions(model, loss_fn, train_data, noise_threshold)` - *Verify that convergence conditions are satisfied for a trained model.*
- `__init__` (line 40) `def __init__(self, model, loss_fn, n_samples, device)`
- `estimate_trace` (line 47) `def estimate_trace(self, data)` - *Estimate tr(H) using Hutchinson's stochastic trace estimator.

Returns:
    (mean_trace, std_trace)*
- `_rademacher_vector` (line 73) `def _rademacher_vector(self)` - *Generate Rademacher random vector (±1 with equal probability)*
- `_hessian_vector_product` (line 90) `def _hessian_vector_product(self, x, y, v)` - *Compute H @ v using the "double backward" trick.

H @ v = ∂/∂θ (∇L · v)*
- `compute_kappa_eff` (line 118) `def compute_kappa_eff(self, data)` - *Compute κ_eff = -tr(H) / N

Interpretation:
- κ_eff < 0 and stable -> grokking likely
- κ_eff > 0 or oscillating -> grokking unlikely*
- `__init__` (line 144) `def __init__(self, model, loss_fn)`
- `estimate_noise` (line 148) `def estimate_noise(self, data_loader, n_batches, n_threads)` - *Estimate hardware noise by measuring gradient variance across batches*
- `__init__` (line 348) `def __init__(self, rank)`
- `forward` (line 354) `def forward(self, x)`

#### `grokkit_physics.py`
**Path:** `src/training/grokkit_physics.py`

**Functions:**
- `strassen_multiply` (line 49) `def strassen_multiply(A, B)` - *Wrapper for Strassen multiplication (uses float32).*
- `measure_physics` (line 64) `def measure_physics(N, num_samples)` - *Measure the 'physical quantities' for a given matrix size.

Returns:
    dict with: speedup, hbar_strassen, hbar_numpy, coherence, error*
- `detect_phase_transition` (line 126) `def detect_phase_transition(results)` - *Find the critical size N_c where the phase transition occurs.
Uses the maximum derivative of speedup curve.*
- `main` (line 148) `def main()`

#### `main.py`
**Path:** `src/training/main.py`

**Classes:**
- `Config` (line 44) `class Config` - *Configuración del experimento.*
- `StrassenDiscovery` (line 87) `class StrassenDiscovery(Module)` - *Modelo para descubrir Strassen mediante coeficientes aprendibles.

Arquitectura:
- U_coefs[i]: Combinación de bloques de A para producto M_i
- V_coefs[i]: Combinación de bloques de B para producto M_i  
- W_coefs[j,i]: Contribución de M_i al bloque C_j del resultado*
- `Matrix4x4Dataset` (line 199) `class Matrix4x4Dataset(Dataset)` - *Dataset de multiplicación de matrices 4x4.*
- `Trainer` (line 217) `class Trainer` - *Entrenador con enmascaramiento progresivo.*

**Methods:**
- `set_seed` (line 79) `def set_seed(seed)` - *Fijar semilla para reproducibilidad.*
- `main` (line 345) `def main()`
- `__init__` (line 96) `def __init__(self, num_slots)`
- `forward` (line 108) `def forward(self, A, B)` - *Forward pass con multiplicación matemática pura.
A, B: (batch, 4, 4) -> C: (batch, 4, 4)*
- `get_slot_norms` (line 155) `def get_slot_norms(self)` - *Norma promedio de cada slot.*
- `get_active_slots` (line 160) `def get_active_slots(self)` - *Número de slots activos.*
- `mask_slot` (line 164) `def mask_slot(self, slot_idx)` - *Desactiva un slot.*
- `get_weakest_slot` (line 169) `def get_weakest_slot(self)` - *Slot con menor norma entre los activos.*
- `print_coefficients` (line 175) `def print_coefficients(self)` - *Muestra coeficientes descubiertos.*
- `__init__` (line 202) `def __init__(self, num_samples, seed)`
- `__len__` (line 210) `def __len__(self)`
- `__getitem__` (line 213) `def __getitem__(self, idx)`
- `__init__` (line 220) `def __init__(self, config)`
- `accuracy` (line 242) `def accuracy(self, pred, target)`
- `train_epoch` (line 245) `def train_epoch(self, optimizer)`
- `evaluate` (line 265) `def evaluate(self)`
- `train` (line 280) `def train(self)`

#### `main_pure_math.py`
**Path:** `src/training/main_pure_math.py`

**Classes:**
- `StrassenModel` (line 22) `class StrassenModel(Module)` - *Descomposición de tensor para multiplicación de matrices 2x2.
C_ij = sum_r W[ij,r] * (U[r,:] @ a) * (V[r,:] @ b)*

**Methods:**
- `gen_data` (line 58) `def gen_data(n, scale)`
- `train` (line 64) `def train(model, epochs, lr, l1, batch, verbose)`
- `verify` (line 88) `def verify(model, n)`
- `hard_prune` (line 106) `def hard_prune(model, keep)` - *Poda los slots más débiles, mantiene top-k.*
- `refine_pruned` (line 122) `def refine_pruned(model, active, epochs, lr)` - *Refina manteniendo slots podados en cero.*
- `show_coeffs` (line 159) `def show_coeffs(model, active)`
- `main` (line 186) `def main()`
- `__init__` (line 28) `def __init__(self, rank)`
- `forward` (line 35) `def forward(self, A, B)`
- `slot_norms` (line 47) `def slot_norms(self)` - *Norma combinada de cada slot.*
- `active_count` (line 54) `def active_count(self, thresh)`

#### `strassen_core.py`
**Path:** `src/training/strassen_core.py`

**Functions:**
- `_load_weights` (line 14) `def _load_weights()`
- `strassen_2x2` (line 21) `def strassen_2x2(A, B)`
- `strassen` (line 44) `def strassen(X, Y)`
- `get_coefficients` (line 77) `def get_coefficients()`
- `multiplication_count` (line 82) `def multiplication_count(n)`

#### `strassen_grokkit.py`
**Path:** `src/training/strassen_grokkit.py`

**Classes:**
- `StrassenOperator` (line 27) `class StrassenOperator(Module)` - *Operador espectral para multiplicación de matrices 2x2.

Representa el tensor de rango R:
C_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)

Donde:
- U, V: Coeficientes de combinación lineal (LC)
- W: Coeficientes de reconstrucción
- Esparsidad (SP): Cuántos slots están activos*

**Methods:**
- `generate_batch` (line 113) `def generate_batch(n, scale)` - *Genera batch de matrices aleatorias.*
- `train_grokkit` (line 120) `def train_grokkit(epochs, batch_size, lr, wd)` - *Entrena usando el framework Grokkit.

WD (Weight Decay) actúa como presión termodinámica que:
1. Empuja hacia soluciones de menor norma
2. Promueve esparsidad natural (slots débiles -> 0)
3. Cristaliza el operador en el mínimo de energía (rango 7)*
- `verify_grokking` (line 203) `def verify_grokking(model, n_test)` - *Verifica que el operador ha grokkeado correctamente.*
- `progressive_sparsification` (line 254) `def progressive_sparsification(model, target_slots)` - *Fase 2: Esparsificación progresiva.
Reduce gradualmente a 7 slots manteniendo accuracy.*
- `main` (line 351) `def main()` - *Pipeline principal Grokkit para Strassen.*
- `__init__` (line 40) `def __init__(self, rank)`
- `forward` (line 49) `def forward(self, A, B)` - *Computa A @ B usando la descomposición tensorial.*
- `compute_LC` (line 67) `def compute_LC(self)` - *Linear Combination metric.
Mide qué tan bien los coeficientes forman combinaciones válidas.
LC -> 1 significa combinaciones perfectas.*
- `compute_SP` (line 83) `def compute_SP(self)` - *Sparsity metric.
SP -> 0 significa máxima esparsidad (menos slots activos).
SP = (slots_activos - 7) / rank para normalizar*
- `slot_importance` (line 101) `def slot_importance(self)` - *Importancia de cada slot basada en normas.*
- `count_active` (line 108) `def count_active(self, threshold)` - *Cuenta slots activos.*

#### `train_strassen.py`
**Path:** `src/training/train_strassen.py`

**Classes:**
- `StrassenOperator` (line 26) `class StrassenOperator(Module)` - *Spectral operator for 2x2 matrix multiplication.

Tensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)*

**Methods:**
- `generate_batch` (line 60) `def generate_batch(n, scale)`
- `train_phase1` (line 66) `def train_phase1(epochs, batch_size, lr, wd)` - *Phase 1: Grokking with Weight Decay as thermodynamic pressure.*
- `sparsify` (line 104) `def sparsify(model, target_slots)` - *Phase 2: Progressive sparsification to target rank.*
- `discretize` (line 183) `def discretize(model, slots_to_prune)` - *Phase 3: Discretize coefficients to {-1, 0, 1}.*
- `get_canonical_strassen` (line 211) `def get_canonical_strassen()` - *Returns the canonical Strassen coefficients.
Exact discrete coefficients for rank-7 tensor decomposition.

Strassen's 7 products:
M1 = (a11 + a22)(b11 + b22)
M2 = (a21 + a22) * b11
M3 = a11 * (b12 - b22)
M4 = a22 * (b21 - b11)
M5 = (a11 + a12) * b22
M6 = (a21 - a11)(b11 + b12)
M7 = (a12 - a22)(b21 + b22)

Result reconstruction:
c11 = M1 + M4 - M5 + M7
c12 = M3 + M5
c21 = M2 + M4
c22 = M1 - M2 + M3 + M6*
- `verify` (line 260) `def verify(U, V, W, n_test)` - *Verify the discretized operator.*
- `main` (line 299) `def main()` - *Main training pipeline.*
- `__init__` (line 33) `def __init__(self, rank)`
- `forward` (line 40) `def forward(self, A, B)`
- `slot_importance` (line 50) `def slot_importance(self)`
- `count_active` (line 56) `def count_active(self, threshold)`

#### `superposition.py`
**Path:** `superposition.py`

**Classes:**
- `Config` (line 23) `class Config` - *Configuration for Superposition Analysis on Strassen Checkpoints.*
- `ICheckpointLoader` (line 59) `class ICheckpointLoader(Protocol)`
- `IMetricsCalculator` (line 62) `class IMetricsCalculator(ABC)`
- `IAnalyzer` (line 66) `class IAnalyzer(ABC)`
- `CheckpointLoadingError` (line 71) `class CheckpointLoadingError(Exception)`
- `CheckpointLoader` (line 75) `class CheckpointLoader` - *Loads raw checkpoint files.*
- `CheckpointMigrator` (line 85) `class CheckpointMigrator` - *Migrates various checkpoint formats to standard state_dict.*
- `StrassenDataGenerator` (line 220) `class StrassenDataGenerator` - *Generates matrix multiplication data for activation extraction.*
- `BilinearStrassenModel` (line 258) `class BilinearStrassenModel(Module)` - *Your existing Strassen model architecture.*
- `SparseAutoencoder` (line 292) `class SparseAutoencoder(Module)` - *SAE with tied weights (W_dec = W_enc^T).
Corrected dimensions: W_enc: [D, N], encode uses W_enc^T, decode uses W_enc.*
- `SuperpositionMetrics` (line 334) `class SuperpositionMetrics(IMetricsCalculator)` - *Calculates superposition metrics from Section 4 of the paper.*
- `SAETrainer` (line 407) `class SAETrainer` - *Trains SAE on bottleneck activations extracted from Strassen model.*
- `StrassenCheckpointAnalyzer` (line 475) `class StrassenCheckpointAnalyzer(IAnalyzer)` - *Analyzes existing Strassen checkpoints for superposition metrics.*

**Methods:**
- `main` (line 765) `def main()`
- `__post_init__` (line 54) `def __post_init__(self)`
- `load_checkpoint` (line 60) `def load_checkpoint(self, path, device)`
- `compute` (line 64) `def compute(self)`
- `analyze_checkpoint` (line 68) `def analyze_checkpoint(self, checkpoint_path)`
- `load_checkpoint` (line 78) `def load_checkpoint(self, path, device)`
- `detect_hidden_dim` (line 89) `def detect_hidden_dim(raw_data)` - *Detect hidden dimension from checkpoint data structure by inspecting
tensor shapes in various known formats.
Returns None if cannot be determined unambiguously.*
- `migrate_checkpoint` (line 122) `def migrate_checkpoint(raw_data)`
- `_migrate_dict` (line 135) `def _migrate_dict(state_dict)`
- `_migrate_custom_format` (line 147) `def _migrate_custom_format(state_dict)`
- `_migrate_coefs_format` (line 162) `def _migrate_coefs_format(state_dict)`
- `_migrate_encoder_format` (line 170) `def _migrate_encoder_format(state_dict)` - *Handle encoder-based format from specific experimental architectures.
Extracts U, V, W from sequential encoder layers assuming specific indexing.*
- `_migrate_standard_format` (line 215) `def _migrate_standard_format(state_dict)`
- `__init__` (line 223) `def __init__(self, config)`
- `generate_batch` (line 226) `def generate_batch(self, batch_size)` - *Generate batch of matrix pairs and their product.*
- `generate_dataset` (line 240) `def generate_dataset(self, num_samples)` - *Generate full dataset.*
- `__init__` (line 261) `def __init__(self, config)`
- `_initialize_symmetric` (line 271) `def _initialize_symmetric(self)`
- `forward` (line 276) `def forward(self, a, b)` - *Forward pass returning output and bottleneck activations.*
- `get_coefficients` (line 284) `def get_coefficients(self)`
- `__init__` (line 298) `def __init__(self, config)`
- `encode` (line 310) `def encode(self, x)` - *Encode bottleneck activations to sparse features.
x: [batch, N]
W_enc: [D, N]
Returns: [batch, D]*
- `decode` (line 319) `def decode(self, z)` - *Decode sparse features back to bottleneck.
z: [batch, D]
W_enc: [D, N]
Returns: [batch, N]*
- `forward` (line 328) `def forward(self, x)`
- `__init__` (line 339) `def __init__(self, config)`
- `compute_feature_probabilities` (line 342) `def compute_feature_probabilities(self, sae_activations)` - *Calculate feature probabilities from SAE activations.
p_i = Σ_s |z_i,s| / Σ_j Σ_s |z_j,s|*
- `compute_entropy` (line 352) `def compute_entropy(self, probabilities)` - *Shannon entropy H(p) = -Σ p_i log p_i.*
- `compute_superposition` (line 359) `def compute_superposition(self, sae_activations)` - *Main metric: ψ = F/N where F = e^{H(p)}.*
- `compute_frobenius_metric` (line 377) `def compute_frobenius_metric(self, weight_matrix)` - *Baseline from Eq 2: ψ_Frob = ||W||_F^2 / N.
Applied to the bottleneck transformation (W matrix of Strassen).*
- `compute_interference_matrix` (line 385) `def compute_interference_matrix(self, weight_matrix)` - *Compute W^T @ W to analyze interference patterns.*
- `compute` (line 389) `def compute(self, sae_activations, weight_matrix)` - *Unified interface.*
- `__init__` (line 410) `def __init__(self, sae, config)`
- `train` (line 421) `def train(self, bottleneck_activations)` - *Train SAE on extracted activations.*
- `__init__` (line 480) `def __init__(self, config)`
- `load_model` (line 496) `def load_model(self, checkpoint_path)` - *Load and migrate checkpoint to model.
Returns tuple of (model, effective_config) where effective_config has 
the correct HIDDEN_DIM for this specific checkpoint to avoid dimension mismatches.*
- `extract_bottleneck_activations` (line 554) `def extract_bottleneck_activations(self, model)` - *Extract bottleneck activations (U(a) * V(b)) from model.*
- `analyze_checkpoint` (line 571) `def analyze_checkpoint(self, checkpoint_path)` - *Full analysis pipeline for a single checkpoint:
1. Load model with correct dimensions (detecting hidden_dim from checkpoint)
2. Extract bottleneck activations
3. Train SAE with matching dimensions (using effective_config)
4. Calculate superposition metrics with correct normalization (N=hidden_dim)
5. Calculate baseline Frobenius metric on W weights*
- `_save_intermediate_result` (line 633) `def _save_intermediate_result(self, result, name)` - *Save result for individual checkpoint.*
- `analyze_directory` (line 641) `def analyze_directory(self, checkpoint_dir)` - *Analyze all checkpoints in directory.*
- `_save_progress_checkpoint` (line 680) `def _save_progress_checkpoint(self, results)` - *Save intermediate progress.*
- `_save_final_results` (line 687) `def _save_final_results(self, results)` - *Save complete results.*
- `_generate_comparison_plots` (line 694) `def _generate_comparison_plots(self, results)` - *Generate comparison plots across checkpoints.*
- `get_tensor` (line 148) `def get_tensor(key)`

#### `train_batch_sweep.py`
**Path:** `train_batch_sweep.py`
**File Doc:** *train_batch_sweep.py*

**Classes:**
- `LocalConfig` (line 17) `class LocalConfig`

**Functions:**
- `train_for_batch_size` (line 11) `def train_for_batch_size(B, seed, output_dir)`

#### `unified_hidden_connections_suite.py`
**Path:** `unified_hidden_connections_suite.py`

**Classes:**
- `StrassStrassenConfig` (line 65) `class StrassStrassenConfig` - *Immutable canonical configuration for the Strassen bilinear model.*
- `TrainingConfig` (line 84) `class TrainingConfig` - *Immutable training hyperparameters.*
- `Experiment1Config` (line 99) `class Experiment1Config` - *Configuration for Experiment 1: Ricci-MBL Duality.*
- `Experiment2Config` (line 114) `class Experiment2Config` - *Configuration for Experiment 2: Altland-Zirnbauer Symmetry Dial.*
- `Experiment3Config` (line 131) `class Experiment3Config` - *Configuration for Experiment 3: Conformal Isomorphism.*
- `Experiment4Config` (line 141) `class Experiment4Config` - *Configuration for Experiment 4: Compression Frontier.*
- `Experiment5Config` (line 158) `class Experiment5Config` - *Configuration for Experiment 5: Holographic Pruning.*
- `SuiteConfig` (line 169) `class SuiteConfig` - *Top-level suite orchestration configuration.*
- `StrassStrassenModel` (line 183) `class StrassStrassenModel(Module)` - *Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.
Implements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.
U, V ∈ R^{rank x input_dim}, W ∈ R^{output_dim x rank}.*
- `IDataGenerator` (line 229) `class IDataGenerator(Protocol)` - *Protocol for deterministic data generation.*
- `StrassenDataGenerator` (line 235) `class StrassenDataGenerator` - *Generates random 2x2 matrix pairs and their exact products.*
- `ICheckpointManager` (line 262) `class ICheckpointManager(Protocol)` - *Protocol for checkpoint persistence.*
- `CheckpointManager` (line 269) `class CheckpointManager` - *Handles safe serialization and deserialization of model checkpoints.*
- `ITrainer` (line 293) `class ITrainer(Protocol)` - *Protocol for training routines.*
- `Trainer` (line 301) `class Trainer` - *Standard trainer with AdamW, cosine annealing, and gradient clipping.*
- `IMetricCalculator` (line 357) `class IMetricCalculator(ABC)` - *Abstract base for all metric calculators.*
- `LevelSpacingRatioCalculator` (line 365) `class LevelSpacingRatioCalculator` - *Computes the adjacent gap ratio r for Hessian eigenvalue spectra.*
- `RicciScalarCalculator` (line 415) `class RicciScalarCalculator` - *Computes Ricci scalar and geometric curvature metrics from Hessian.*
- `SyntheticPlanckCalculator` (line 492) `class SyntheticPlanckCalculator` - *Estimates the synthetic Planck constant from model weight statistics.*
- `SuperpositionMetricCalculator` (line 552) `class SuperpositionMetricCalculator` - *Measures superposition via sparse autoencoder bottleneck analysis.*
- `IExperiment` (line 645) `class IExperiment(ABC)` - *Abstract base for all experiments in the suite.*
- `Experiment1RicciMBLDuality` (line 657) `class Experiment1RicciMBLDuality(IExperiment)` - *Tests the claim that Ricci curvature smoothing is the geometric mechanism
driving the Wigner-Dyson to Poisson/MBL spectral transition.*
- `Experiment2AltlandZirnbauer` (line 735) `class Experiment2AltlandZirnbauer(IExperiment)` - *Tests the claim that the imaginary-weight control parameter gamma drives
the system between GOE (orthogonal) and GUE (unitary) random matrix classes.*
- `Experiment3ConformalIsomorphism` (line 820) `class Experiment3ConformalIsomorphism(IExperiment)` - *Tests the claim that the network learns the underlying conformal operator
by applying Moebius transformations to inputs and measuring equivariance.*
- `Experiment4CompressionFrontier` (line 886) `class Experiment4CompressionFrontier(IExperiment)` - *Tests the thermodynamic uncertainty relation between synthetic Planck
constant and superposition metric across a sweep of batch sizes and
weight decay values.*
- `IPruningStrategy` (line 965) `class IPruningStrategy(ABC)` - *Abstract base for structured pruning strategies.*
- `VolumePruningStrategy` (line 973) `class VolumePruningStrategy(IPruningStrategy)` - *Prunes weights uniformly at random across all layers.*
- `AreaPruningStrategy` (line 988) `class AreaPruningStrategy(IPruningStrategy)` - *Prunes weights only from tensor boundaries (slot edges).*
- `Experiment5HolographicPruning` (line 1004) `class Experiment5HolographicPruning(IExperiment)` - *Tests whether the crystal phase encodes information on boundaries
(area law) versus the glass phase encoding it volumetrically.*
- `UnifiedSuite` (line 1078) `class UnifiedSuite` - *Orchestrates the execution of all five hidden-connection experiments.*

**Methods:**
- `main` (line 1182) `def main()`
- `__post_init__` (line 76) `def __post_init__(self)`
- `__init__` (line 190) `def __init__(self, config)`
- `forward` (line 203) `def forward(self, A, B)`
- `get_coefficients` (line 213) `def get_coefficients(self)`
- `get_flat_parameters` (line 216) `def get_flat_parameters(self)`
- `slot_importance` (line 219) `def slot_importance(self)`
- `count_active_slots` (line 225) `def count_active_slots(self, threshold)`
- `generate_batch` (line 232) `def generate_batch(self, batch_size)`
- `__init__` (line 238) `def __init__(self, config)`
- `generate_batch` (line 241) `def generate_batch(self, batch_size)`
- `save` (line 265) `def save(self, model, epoch, metrics, path)`
- `load` (line 266) `def load(self, path, model)`
- `save` (line 272) `def save(self, model, epoch, metrics, path)`
- `load` (line 284) `def load(self, path, model)`
- `train` (line 296) `def train(self, model, epochs, callback)`
- `__init__` (line 304) `def __init__(self, model_config, training_config, data_generator)`
- `train` (line 314) `def train(self, model, epochs, callback)`
- `calculate` (line 361) `def calculate(self, model)`
- `__init__` (line 368) `def __init__(self, config, tolerance)`
- `_build_hessian_approximation` (line 372) `def _build_hessian_approximation(self, model)`
- `calculate` (line 384) `def calculate(self, model)`
- `__init__` (line 418) `def __init__(self, config, regularization)`
- `_compute_hessian` (line 422) `def _compute_hessian(self, model)`
- `_generate_single_sample` (line 433) `def _generate_single_sample(self)`
- `_loss_from_flat` (line 437) `def _loss_from_flat(self, flat_params, model, original_params, A, B, C_true)`
- `_diagonal_hessian_approximation` (line 456) `def _diagonal_hessian_approximation(self, model, A, B, C_true)`
- `calculate` (line 470) `def calculate(self, model)`
- `__init__` (line 495) `def __init__(self, config, noise_floor)`
- `calculate` (line 499) `def calculate(self, model)`
- `__init__` (line 555) `def __init__(self, model_config, expansion_factor, l1_coefficient, sae_lr, sae_epochs, sae_batch_size, num_samples, epsilon)`
- `_extract_activations` (line 576) `def _extract_activations(self, model)`
- `_train_sae` (line 589) `def _train_sae(self, activations)`
- `_sae_forward` (line 616) `def _sae_forward(self, x, W_enc, b_enc, b_dec)`
- `calculate` (line 623) `def calculate(self, model)`
- `run` (line 649) `def run(self, model)`
- `get_name` (line 653) `def get_name(self)`
- `__init__` (line 663) `def __init__(self, config, model_config, training_config, data_generator, checkpoint_manager)`
- `get_name` (line 679) `def get_name(self)`
- `run` (line 682) `def run(self, model)`
- `_analyze_temporal_correlation` (line 717) `def _analyze_temporal_correlation(self, results)`
- `__init__` (line 741) `def __init__(self, config, model_config, data_generator)`
- `get_name` (line 752) `def get_name(self)`
- `run` (line 755) `def run(self, model)`
- `_create_gamma_model` (line 778) `def _create_gamma_model(self, base_model, gamma)`
- `_train_gamma_model` (line 790) `def _train_gamma_model(self, model)`
- `_detect_critical_transition` (line 800) `def _detect_critical_transition(self, results)`
- `__init__` (line 826) `def __init__(self, config, model_config, data_generator)`
- `get_name` (line 836) `def get_name(self)`
- `run` (line 839) `def run(self, model)`
- `_apply_moebius` (line 869) `def _apply_moebius(self, A, B)`
- `_apply_moebius_to_output` (line 879) `def _apply_moebius_to_output(self, C)`
- `__init__` (line 893) `def __init__(self, config, model_config, data_generator)`
- `get_name` (line 904) `def get_name(self)`
- `run` (line 907) `def run(self, model)`
- `_test_uncertainty_bound` (line 951) `def _test_uncertainty_bound(self, results)`
- `prune` (line 969) `def prune(self, model, fraction)`
- `prune` (line 976) `def prune(self, model, fraction)`
- `prune` (line 991) `def prune(self, model, fraction)`
- `__init__` (line 1010) `def __init__(self, config, model_config, data_generator)`
- `get_name` (line 1022) `def get_name(self)`
- `run` (line 1025) `def run(self, model)`
- `_run_pruning_trials` (line 1046) `def _run_pruning_trials(self, base_model, pruner, A, B, C_true)`
- `__init__` (line 1081) `def __init__(self, config)`
- `_build_experiments` (line 1088) `def _build_experiments(self)`
- `run_all` (line 1127) `def run_all(self)`
- `_aggregate_verdicts` (line 1150) `def _aggregate_verdicts(self, all_results)`
- `_serialize_config` (line 1163) `def _serialize_config(self)`
- `checkpoint_callback` (line 691) `def checkpoint_callback(epoch, m, loss, acc)`

#### `xray_tensor_diffractometer.py`
**Path:** `xray_tensor_diffractometer.py`

**Classes:**
- `Config` (line 27) `class Config`
- `ICheckpointLoader` (line 146) `class ICheckpointLoader(Protocol)`
- `IMetricsCalculator` (line 149) `class IMetricsCalculator(Protocol)`
- `IDataGenerator` (line 152) `class IDataGenerator(Protocol)`
- `CheckpointLoadingError` (line 155) `class CheckpointLoadingError(Exception)`
- `MetricsComputationError` (line 158) `class MetricsComputationError(Exception)`
- `TrainingError` (line 161) `class TrainingError(Exception)`
- `StrassenDataGenerator` (line 166) `class StrassenDataGenerator`
- `BilinearStrassenModel` (line 187) `class BilinearStrassenModel(Module)`
- `EpitaxialGrowthEngine` (line 216) `class EpitaxialGrowthEngine` - *Motor de crecimiento epitaxial para cristales algorítmicos.

FÍSICA: Imita el crecimiento de cristales en sustratos donde la estructura
atómica del sustrato guía la formación del nuevo cristal.*
- `EpitaxyExperiment` (line 469) `class EpitaxyExperiment` - *Experimento completo de epitaxia: sembrar, crecer, analizar.*
- `ThermodynamicPotential` (line 671) `class ThermodynamicPotential` - *Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C*
- `SpectroscopyMetrics` (line 693) `class SpectroscopyMetrics`
- `ThermodynamicMetrics` (line 913) `class ThermodynamicMetrics`
- `CrystallographyMetrics` (line 1158) `class CrystallographyMetrics`
- `GreenCowExperiment` (line 1264) `class GreenCowExperiment` - *🐄 Green's Cow: Uses integration-by-parts analogy to split gradient into bulk and boundary terms.
Inspired by Green's identities: ∫_Ω ∇u·v = ∫_∂Ω u(v·n) - ∫_Ω u∇·v
Applied to weight tensors as discrete manifolds.*
- `CheckpointLoader` (line 1371) `class CheckpointLoader`
- `CheckpointMigrator` (line 1405) `class CheckpointMigrator`
- `BoltzmannAnalysisProgram` (line 1490) `class BoltzmannAnalysisProgram`
- `StrassenCrystallographer` (line 2610) `class StrassenCrystallographer`

**Methods:**
- `set_seed` (line 64) `def set_seed(seed)`
- `setup_logger` (line 72) `def setup_logger(name, level)`
- `run_epitaxy_from_best_crystal` (line 86) `def run_epitaxy_from_best_crystal(checkpoint_dir, target_sizes)` - *Pipeline automático: encuentra el mejor cristal y lo usa como semilla.*
- `main` (line 2683) `def main()`
- `load_checkpoint` (line 147) `def load_checkpoint(self, path, device)`
- `compute` (line 150) `def compute(self, model)`
- `generate_batch` (line 153) `def generate_batch(self, batch_size)`
- `generate_batch` (line 168) `def generate_batch(batch_size)`
- `verify_structure` (line 179) `def verify_structure(coeffs)`
- `__init__` (line 188) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize_symmetric` (line 199) `def _initialize_symmetric(self)`
- `forward` (line 204) `def forward(self, a, b)`
- `get_coefficients` (line 207) `def get_coefficients(self)`
- `__init__` (line 224) `def __init__(self, seed_checkpoint_path, target_matrix_size, device)`
- `_load_seed_crystal` (line 242) `def _load_seed_crystal(self)` - *Carga el cristal semilla verificando su pureza*
- `grow_epitaxial_crystal` (line 265) `def grow_epitaxial_crystal(self)` - *Crece un cristal epitaxial desde la semilla.

MÉTODO: Kronecker product preserva la estructura periódica:
Si A es cristal Strassen de 2x2, entonces A ⊗ I_n es cristal de (2n)x(2n)*
- `_adjust_dimensions` (line 326) `def _adjust_dimensions(self, tensor, target_shape)` - *Ajusta dimensiones del tensor epitaxial para coincidir con el modelo objetivo.
Rellena con ruido térmico pequeño o trunca según sea necesario.*
- `anneal_crystal` (line 360) `def anneal_crystal(self, model, max_epochs, early_stop_threshold)` - *Recocido térmico del cristal epitaxial.

FÍSICA: En lugar de "entrenar desde cero", aplicamos temperatura decreciente
para que el cristal se auto-organice alrededor de la semilla.*
- `__init__` (line 474) `def __init__(self, results_dir)`
- `run_epitaxial_growth_experiment` (line 479) `def run_epitaxial_growth_experiment(self, seed_checkpoint, target_sizes)` - *Experimento completo: cultiva cristales de múltiples tamaños desde una semilla.*
- `_plot_epitaxial_evolution` (line 556) `def _plot_epitaxial_evolution(self, annealing_results, target_size, seed_name)` - *Visualiza la evolución del cristal durante el recocido*
- `_generate_comparative_report` (line 604) `def _generate_comparative_report(self, results)` - *Genera reporte comparativo de todos los experimentos epitaxiales*
- `helmholtz_free_energy` (line 680) `def helmholtz_free_energy(self)` - *F = U - T*S (a μ y N constantes)*
- `gibbs_free_energy` (line 684) `def gibbs_free_energy(self)` - *G = F + μ*N + P*V (presión algorítmica)*
- `is_stable` (line 689) `def is_stable(self)` - *Criterio de estabilidad: dG < 0*
- `compute_weight_diffraction` (line 696) `def compute_weight_diffraction(coeffs)`
- `_compute_spectral_entropy` (line 719) `def _compute_spectral_entropy(power_spectrum)`
- `extract_lattice_parameters` (line 726) `def extract_lattice_parameters(weight_tensor, rank)` - *Extrae parámetros de red preservando la geometría física del tensor.

FIX: En lugar de reshape arbitrario, aplicamos SVD sobre la matriz 
de covarianza que preserva la estructura de correlaciones.*
- `compute_gibbs_free_energy` (line 785) `def compute_gibbs_free_energy(loss, temp, entropy)`
- `extract_canonical_decomposition` (line 791) `def extract_canonical_decomposition(coeffs, rank)` - *Descomposición Canónica del tensor tripartito (U, V, W).

FIX: Preserva la estructura bilineal en lugar de tratar como matriz plana.
Aplicamos HOSVD (Higher-Order SVD) para tensores de orden 3.*
- `_discretize_to_integers` (line 847) `def _discretize_to_integers(factors)` - *Proyecta factores continuos a la red cristalina discreta {-1, 0, 1}.*
- `_check_strassen_equivalence` (line 868) `def _check_strassen_equivalence(discretized_factors)` - *Verifica si los factores discretizados corresponden a la estructura de Strassen.*
- `create_superlattice_seed` (line 892) `def create_superlattice_seed(base_tensor, scale_factor)`
- `compute_effective_temperature` (line 916) `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `compute_critical_exponents` (line 930) `def compute_critical_exponents(temp_history, cv_history, alpha_history)` - *Calcula exponentes críticos cerca de transiciones de fase.

Leyes de escala:
- C_v ~ |T - T_c|^{-α_exp}  (calor específico)
- ξ ~ |T - T_c|^{-ν}        (longitud de correlación)
- τ ~ |T - T_c|^{-z}        (tiempo de grokking)*
- `compute_equation_of_state` (line 1009) `def compute_equation_of_state(temp_eff, alpha, kappa)` - *Ecuación de estado: T_c(α) = T_0 * exp(-c*α)

FIX: Relación constitutiva que describe la curva de coexistencia cristal-vidrio.*
- `compute_specific_heat` (line 1048) `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `estimate_hbar_algorithmic` (line 1061) `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)`
- `compute_mutual_information` (line 1069) `def compute_mutual_information(weights, gradients)`
- `check_extensivity` (line 1083) `def check_extensivity(entropy_list, scale_factors)`
- `compute_fisher_information_matrix` (line 1107) `def compute_fisher_information_matrix(model, samples)`
- `compute_ricci_curvature` (line 1126) `def compute_ricci_curvature(fisher_matrix)`
- `calculate_carnot_efficiency` (line 1137) `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)`
- `compute_kappa` (line 1161) `def compute_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin` (line 1187) `def compute_discretization_margin(coeffs)`
- `compute_local_complexity` (line 1191) `def compute_local_complexity(model)`
- `compute_alpha_purity` (line 1200) `def compute_alpha_purity(coeffs)`
- `compute_kappa_quantum` (line 1207) `def compute_kappa_quantum(coeffs, hbar)`
- `compute_poynting_vector` (line 1224) `def compute_poynting_vector(coeffs)`
- `compute_all_metrics` (line 1246) `def compute_all_metrics(model, dataloader)`
- `__init__` (line 1270) `def __init__(self, model, device)`
- `compute_boundary_gradient` (line 1275) `def compute_boundary_gradient(self, weight)` - *Approximate surface term: gradient concentrated on tensor boundaries.
For a matrix W ∈ ℝ^{m×n}, boundary = first/last row + first/last column.*
- `compute_bulk_gradient` (line 1290) `def compute_bulk_gradient(self, weight)` - *Interior (volume) term: everything except boundary.*
- `run_green_backprop_step` (line 1296) `def run_green_backprop_step(self, A, B, C_true, lambda_boundary)` - *Custom backward pass using Green-inspired decomposition.
Loss = MSE + λ_boundary * ||boundary_grad||²*
- `_get_boundary_mask` (line 1329) `def _get_boundary_mask(self, weight)` - *Returns a binary mask marking boundary elements of a tensor.*
- `train_with_green_cow` (line 1341) `def train_with_green_cow(self, epochs, lr, lambda_boundary)`
- `load_checkpoint` (line 1372) `def load_checkpoint(self, path, device)` - *Load checkpoint with robust deserialization handling.
Injects Config as UnifiedConfig alias to handle cross-script compatibility.*
- `migrate_checkpoint` (line 1407) `def migrate_checkpoint(raw_data)` - *Migrate checkpoint to standard format, extracting config if present.
Returns migrated state_dict compatible with BilinearStrassenModel.*
- `_migrate_dict` (line 1431) `def _migrate_dict(state_dict)`
- `_migrate_custom_format` (line 1443) `def _migrate_custom_format(state_dict)`
- `_migrate_coefs_format` (line 1467) `def _migrate_coefs_format(state_dict)`
- `_migrate_encoder_format` (line 1475) `def _migrate_encoder_format(state_dict)`
- `_migrate_standard_format` (line 1487) `def _migrate_standard_format(state_dict)`
- `__init__` (line 1491) `def __init__(self, checkpoint_dir, results_dir)`
- `_load_all_checkpoints` (line 1503) `def _load_all_checkpoints(self)`
- `run_full_boltzmann_program` (line 1548) `def run_full_boltzmann_program(self)`
- `phase1_molecular_hypothesis` (line 1575) `def phase1_molecular_hypothesis(self)`
- `phase2_entropy_production` (line 1660) `def phase2_entropy_production(self)`
- `phase3_extensivity_law` (line 1742) `def phase3_extensivity_law(self)`
- `phase4_quantum_basis_transform` (line 1796) `def phase4_quantum_basis_transform(self)`
- `analyze_poynting_flow` (line 1849) `def analyze_poynting_flow(self)`
- `phase5_thermodynamic_analysis` (line 1882) `def phase5_thermodynamic_analysis(self)` - *PHASE 5: THERMODYNAMIC ANALYSIS con exponentes críticos y ecuación de estado.

FIX: Ahora calcula exponentes críticos y ecuación de estado para cada checkpoint.*
- `phase6_spectroscopic_analysis` (line 2011) `def phase6_spectroscopic_analysis(self)`
- `_plot_diffraction_pattern` (line 2097) `def _plot_diffraction_pattern(self, diffraction_data, ckpt_name)`
- `_save_superlattice_seed` (line 2131) `def _save_superlattice_seed(self, superlattice, ckpt_name)`
- `_classify_thermodynamic_phase` (line 2150) `def _classify_thermodynamic_phase(self, t_eff, cv, alpha)`
- `_estimate_critical_temperature` (line 2162) `def _estimate_critical_temperature(self, results)`
- `_verify_entropy_extensivity` (line 2174) `def _verify_entropy_extensivity(self, results)`
- `_plot_phase_diagram` (line 2188) `def _plot_phase_diagram(self, results)`
- `_plot_temperature_vs_purity` (line 2213) `def _plot_temperature_vs_purity(self, results)`
- `_compute_entropy_simple` (line 2239) `def _compute_entropy_simple(self, params)`
- `_compute_entropy` (line 2265) `def _compute_entropy(self, params)`
- `_compute_effective_volume` (line 2304) `def _compute_effective_volume(self, params)`
- `_plot_parameter_distribution` (line 2322) `def _plot_parameter_distribution(self, params, group_name, kde)`
- `_simulate_training_trajectory` (line 2350) `def _simulate_training_trajectory(self, final_params, final_delta)`
- `_compute_generalization_entropy` (line 2361) `def _compute_generalization_entropy(self, params, successful_ckpts)`
- `_fit_timescale` (line 2414) `def _fit_timescale(self, entropy_values)`
- `_plot_entropy_production` (line 2424) `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- `_verify_scaling` (line 2442) `def _verify_scaling(self, coeffs, N)`
- `_recursive_strassen` (line 2453) `def _recursive_strassen(self, A, B, coeffs, N)`
- `_fit_extensivity` (line 2487) `def _fit_extensivity(self, errors, sizes, purity)`
- `_verify_extensivity_universality` (line 2501) `def _verify_extensivity_universality(self, results)`
- `_plot_extensivity` (line 2505) `def _plot_extensivity(self, sizes, errors, purity, ckpt_name)`
- `_find_broken_symmetries` (line 2518) `def _find_broken_symmetries(self, coeffs)`
- `_measure_uncertainty` (line 2526) `def _measure_uncertainty(self, coeffs, basis)`
- `_plot_uncertainty_distribution` (line 2537) `def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)`
- `_print_executive_summary` (line 2558) `def _print_executive_summary(self, results)`
- `_save_results` (line 2587) `def _save_results(self, results, filename)`
- `__init__` (line 2611) `def __init__(self, checkpoint_path, device)`
- `_load_model` (line 2617) `def _load_model(self, path, device)`
- `run_full_analysis` (line 2634) `def run_full_analysis(self)`
- `_assign_grade` (line 2656) `def _assign_grade(self, delta, alpha)`
- `_save_report` (line 2668) `def _save_report(self, report)`
- `generate_batch` (line 373) `def generate_batch(batch_size)`
- `get_tensor` (line 1444) `def get_tensor(key)`
- `model` (line 2415) `def model(t, A, tau, C)`
- `model` (line 2488) `def model(N, alpha, beta)`
- `convert_to_serializable` (line 2590) `def convert_to_serializable(obj)`
- `dataloader` (line 2637) `def dataloader()`
- `sample_dataloader` (line 1906) `def sample_dataloader()`
- `sample_dataloader` (line 2039) `def sample_dataloader()`
- `dataloader` (line 1522) `def dataloader()`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
