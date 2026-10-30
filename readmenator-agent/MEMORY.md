# Project Memory

> Cross-session context for agents. Sections 1-6 are regenerated from the source tree with zero LLM tokens: declared rules are quoted verbatim with `file:line`, measured baselines come from the scan. Section 7 is written by agents and humans and is preserved across rebuilds.

Generated from 60 files at commit `f08f59266567`. Read this first, then `readmenator-wiki/index.md`, then `readmenator . ask "<question>"` for anything specific.

## 1. Purpose and domain

- What it is: This article presents what I learned by attempting to induce the Strassen matrix multiplication structure in neural networks, and why I now view this work as materials engineering rather than theory, and how it may serve as a bridge between statistical thermodynamics and deep learning. (`README.md:12`)
- Domain vocabulary (term, files): `strassen` (48), `forward` (37), `get` (34), `compute` (33), `generate` (32), `model` (30), `load` (30), `checkpoint` (29), `batch` (28), `matrix` (28), `analysis` (27), `coefficients` (27), `training` (26), `multiplication` (24), `all` (23)
- Business rules that the code cannot show live in section 7: record them there.

## 2. Workflow

Detected commands:
- `python -m pytest -q` (tests/ layout)
- `CI workflow github-actions-demo.yml` (workflows/github-actions-demo.yml)

Session protocol:
1. Start: read this file, then `readmenator . fresh` (exit 1 means run `readmenator . --rebuild`).
2. Orient: `readmenator-wiki/index.md`; for a question use `readmenator . ask "<question>"` (local = entities + sources, `--global` = community reports).
3. Before editing a file: `grep -n '<file>' readmenator-agent/GOTCHAS.md readmenator-agent/SECURITY.md`.
4. After the change: run the tests above, then `readmenator . --rebuild` so the maps, wiki, and this file stay true.
5. End: record decisions, business rules, and gotchas with `readmenator . remember "<note>" --kind decision`.

## 3. Rules and constraints

Declared:
- none declared in instruction files (add them to AGENTS.md or record them in section 7)

Measured baseline:
- Security findings at medium or above: 0 (see `readmenator-agent/SECURITY.md`); do not add new ones.
- Dependency cycles: 0; layer violations: 0 (see `readmenator-agent/GOTCHAS.md`).

## 4. Style norms

Declared:
- none declared in instruction files (add them to AGENTS.md or record them in section 7)

Measured baseline:
- py: 56 files, 2007 symbols; docstrings on 31% of symbols; functions snake_case (99%); types PascalCase (100%); median file 553 lines, max 2801.
- c: 3 files, 25 symbols; docstrings on 76% of symbols; functions snake_case (100%); median file 171 lines, max 269.
- sh: 1 files, 0 symbols; median file 4 lines, max 4.
- Tests: 1 files under tests; follow the existing naming (e.g. `test_main.py`).

## 5. Minimum deliverables

Declared:
- none declared in instruction files (add them to AGENTS.md or record them in section 7)

Measured baseline:
- Tests pass: `python -m pytest -q`.
- Docstring coverage stays at or above 32%.
- No new security findings at medium or above (current: 0).
- No new dependency cycles (current: 0).
- Files stay under 300 lines where possible (`readmenator . lint`).
- Docs refreshed: `readmenator . --rebuild`, and decisions recorded in section 7.

## 6. Risks to respect

- God nodes (changes ripple widely): `xray_tensor_diffractometer.py`, `dirac_polos_zeros.py`, `full_seed_prospector.py`, `experiments/extended_experiments/all_test_extended.py`, `unified_hidden_connections_suite.py`
- Hotspots (complex and central): `xray_tensor_diffractometer.py`, `experiments/extended_experiments/all_test_extended.py`, `percolation_analysis.py`, `full_seed_prospector.py`, `dirac_polos_zeros.py`
- Full blast radius: `readmenator-agent/GOTCHAS.md`; findings: `readmenator-agent/SECURITY.md`.

## 7. Session log (preserved across rebuilds)

Append with `readmenator . remember "<note>" --kind <kind>` (kinds: business, decision, rule, workflow, style, deliverable, gotcha, todo, note) or the MCP tool `readmenator.remember`. Record business rules, decisions and their reasons, workflow changes, and anything the next session must not rediscover.

<!-- readmenator:memory:notes:begin -->
<!-- readmenator:memory:notes:end -->
