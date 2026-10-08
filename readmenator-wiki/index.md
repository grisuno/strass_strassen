# Second Brain

*Last synthesized: 2026-10-07 | 60 files | 1 concept pages | offline, zero tokens*

> Raw sources -> readmenator wiki -> links (Karpathy LLM Wiki Pattern, deterministic).
> Start here, then open one community page. Prefer grep over full reads.

## Vault Overview

The codebase centres on `xray_tensor_diffractometer.py`, `dirac_polos_zeros.py`, `full_seed_prospector.py`. Architecturally it is 4 layers, dominant utility (56 files) across 1 import-based communities. Recorded risk surface: 0 security findings and 0 dependency cycles.

Communities are self-contained in the resolved import graph; no cross-boundary bridges were recorded.

Open work clusters around documentation (73% file coverage), 0 security findings, 1 taint paths, and 5 suggested exploration questions in `queries.md`.

## Stats

| Metric | Value |
|--------|-------|
| Files | 60 |
| Symbols | 2071 |
| Resolved imports | 0 |
| Languages | c, py, sh |
| Communities | 1 |
| Doc coverage | 73% (44/60 files) |
| Security findings | 0 |
| Estimated read cost | ~26648 tokens (chars/4, offline so $0) |

## Reading Order

1. Skim Stats and God Nodes below for blast radius.
2. Open the largest community page first, then follow Connections.
3. Use `queries.md` for the next question; log the answer there.

```
grep -rn '<keyword>' index.md community_*.md
readmenator query "<question>" --target readmenator_strass_strassen_lj84bygq
```

## Concept Wiki

- [root (60 files, cohesion 1.00)](./community_0_root.md)

## God Nodes

| File | Score |
|------|-------|
| `xray_tensor_diffractometer.py` | 13.2 |
| `dirac_polos_zeros.py` | 12.3 |
| `full_seed_prospector.py` | 11.6 |
| `experiments/extended_experiments/all_test_extended.py` | 11.4 |
| `unified_hidden_connections_suite.py` | 9.9 |

## Strongest Connections

- No cross-community connections recorded.

## Navigation Tips

- Obsidian Graph View works: every community page links back here.
- `connections.json` is machine-readable for GraphRAG pipelines.
- `REPORT.md` states what was extracted vs inferred and current limits.
- Regenerate offline: `readmenator . --rebuild` (no network, no tokens).
