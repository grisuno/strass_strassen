# Gotchas

## God Nodes (high connectivity)

These files have the most connections. Changes here have high blast radius.

- `xray_tensor_diffractometer.py` (score: 13.20)
- `dirac_polos_zeros.py` (score: 12.30)
- `full_seed_prospector.py` (score: 11.60)
- `experiments/extended_experiments/all_test_extended.py` (score: 11.40)
- `unified_hidden_connections_suite.py` (score: 9.90)
- `percolation_analysis.py` (score: 9.10)
- `mbl_analyzer.py` (score: 8.90)
- `gravity.py` (score: 8.80)
- `grain.py` (score: 7.70)
- `plank.py` (score: 7.20)

## Hotspots (complexity + centrality)

- `xray_tensor_diffractometer.py` -- complexity: 1.0, centrality: 1.0, combined: 1.0
- `experiments/extended_experiments/all_test_extended.py` -- complexity: 0.9, centrality: 0.8, combined: 0.8
- `percolation_analysis.py` -- complexity: 0.7, centrality: 0.9, combined: 0.8
- `full_seed_prospector.py` -- complexity: 0.9, centrality: 0.7, combined: 0.8
- `dirac_polos_zeros.py` -- complexity: 0.9, centrality: 0.7, combined: 0.8
- `unified_hidden_connections_suite.py` -- complexity: 0.8, centrality: 0.7, combined: 0.7
- `gravity.py` -- complexity: 0.7, centrality: 0.7, combined: 0.7
- `experiments/extended_experiments/validate2.py` -- complexity: 0.5, centrality: 0.8, combined: 0.7
- `grain.py` -- complexity: 0.6, centrality: 0.7, combined: 0.7
- `mbl_analyzer.py` -- complexity: 0.7, centrality: 0.6, combined: 0.6

## Dataflow Issues (INFERRED, review each lead)

- `src/native/strassen_c.c:72` `strassen_recursive` [DEAD_STORE] `hh`: `hh` assigned at line 72 but never read afterwards.
- `src/native/strassen_optimal.c:32` `strassen_level` [DEAD_STORE] `M`: `M` assigned at line 32 but never read afterwards.
- `src/native/strassen_optimal.c:36` `strassen_level` [DEAD_STORE] `A11`: `A11` assigned at line 36 but never read afterwards.
- `src/native/strassen_optimal.c:37` `strassen_level` [DEAD_STORE] `A12`: `A12` assigned at line 37 but never read afterwards.
- `src/native/strassen_optimal.c:38` `strassen_level` [DEAD_STORE] `A21`: `A21` assigned at line 38 but never read afterwards.
- `src/native/strassen_optimal.c:39` `strassen_level` [DEAD_STORE] `A22`: `A22` assigned at line 39 but never read afterwards.
- `src/native/strassen_optimal.c:41` `strassen_level` [DEAD_STORE] `B11`: `B11` assigned at line 41 but never read afterwards.
- `src/native/strassen_optimal.c:42` `strassen_level` [DEAD_STORE] `B12`: `B12` assigned at line 42 but never read afterwards.
- `src/native/strassen_optimal.c:43` `strassen_level` [DEAD_STORE] `B21`: `B21` assigned at line 43 but never read afterwards.
- `src/native/strassen_optimal.c:44` `strassen_level` [DEAD_STORE] `B22`: `B22` assigned at line 44 but never read afterwards.
