# GraphRAG Community Reports

Entities: 2215 | Relationships: 5593 | Communities: 1 | Themes: 0 | Text units: 2076

Query with `readmenator . ask "<question>"` (local: BM25 + Personalized PageRank; global: map-reduce over these reports) or the MCP tool `readmenator.graphrag`.

## Project overview (`root`, root, rating 7.0)

60 files in 1 communities. Highest-impact communities: unassigned files (7.0). God nodes: xray_tensor_diffractometer.py, dirac_polos_zeros.py, full_seed_prospector.py, experiments/extended_experiments/all_test_extended.py, unified_hidden_connections_suite.py.

- [unassigned files] rating 7.0: `app.py` ranks 1 by PageRank, 0 importers, 3 symbols: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:
- Key entities: file:dirac_polos_zeros.py, file:full_seed_prospector.py
- Children: c0
- Root rating = highest community rating.

## unassigned files (`c0`, community, rating 7.0)

60 files under root (py 56, c 3, sh 1), mostly utility. Core file app.py (PageRank 0.0167, imported by 0 files): Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción: Key abstractions: StrassenNet, forward, Configuration, BilinearStrassenModel, CheckpointMigrator, CustomFormatMigrator.

- `app.py` ranks 1 by PageRank, 0 importers, 3 symbols: Autor: Gris Iscomeback Correo electrónico: grisiscomeback[at]gmail[dot]com Fecha de creación: xx/xx/xxxx Licencia: GPL v3  Descripción:
- `batch_size.py` ranks 2 by PageRank, 0 importers, 41 symbols: StrassenPlanckCalculator - Extended with Batch Size Thermodynamics  Calculates effective Planck constant (h_bar) from Strassen model checkpoints and derives optimal batch size from quantum-thermodynamic principles.
- `boltzmann_experiments.py` ranks 3 by PageRank, 0 importers, 60 symbols: compute_kappa: Classical kappa - will be inf for discrete states
- Hotspot `xray_tensor_diffractometer.py`: 132 symbols, 27 connections (score 1.00).
- Hotspot `experiments/extended_experiments/all_test_extended.py`: 114 symbols, 22 connections (score 0.83).
- Taint: 1 paths reach this group via subprocess.
- Dataflow: 14 INFERRED issues (first: DEAD_STORE in strassen_recursive).
- Key entities: file:dirac_polos_zeros.py, file:full_seed_prospector.py, file:experiments/extended_experiments/all_test_extended.py, file:xray_tensor_diffractometer.py, file:unified_hidden_connections_suite.py, file:gravity.py, file:mbl_analyzer.py, file:percolation_analysis.py
- Rating 7.0/10 = 7 x PageRank share 1.00 + 3 x risk 0.00. Internal imports: 0.
