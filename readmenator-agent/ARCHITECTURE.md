# Architecture

## Internal Dependencies

- (no internal resolved imports)

## External Imports

- `app.py` -> numpy, onnx, onnxruntime, os, torch, torch.nn
- `batch_size.py` -> abc, argparse, dataclasses, datetime, json, matplotlib.pyplot, numpy, os, pathlib, random, torch, torch.nn, typing
- `boltzmann_experiments.py` -> abc, argparse, datetime, json, matplotlib.pyplot, numpy, os, random, scipy.linalg, scipy.optimize, scipy.stats, seaborn, sklearn.decomposition, torch, torch.nn, typing, warnings
- `compute_gns_checkpoints.py` -> collections, pathlib, torch, torch.nn, vector8
- `crystallography.py` -> collections, datetime, json, matplotlib.pyplot, numpy, os, random, torch, torch.nn, typing
- `dirac_polos_zeros.py` -> abc, argparse, dataclasses, datetime, glob, json, matplotlib.cm, matplotlib.colors, matplotlib.patches, matplotlib.pyplot, numpy, os, pathlib, scipy, scipy.linalg, torch, torch.nn, traceback, typing
- `experiments/ablation/ablation_8192.py` -> ctypes, gc, json, numpy, sys, time
- `experiments/ablation/ablation_study.py` -> ctypes, dataclasses, gc, json, numpy, time, typing
- `experiments/apendix_experiments.py` -> datetime, json, matplotlib.pyplot, numpy, pathlib, seaborn, time, torch, torch.nn, torch.optim, warnings
- `experiments/cache_analysis_v2.py` -> json
- `experiments/extended_experiments/all_test_extended.py` -> abc, collections, dataclasses, datetime, hashlib, json, matplotlib.pyplot, numpy, pandas, pathlib, scipy, scipy.spatial.distance, signal, sys, time, torch, torch.nn, torch.optim, torch.utils.data, traceback, typing, warnings
- `experiments/extended_experiments/exp1_covariance_spectrometry.py` -> datetime, json, matplotlib.pyplot, numpy, os, pathlib, scipy, seaborn, torch, torch.nn, torch.nn.functional
- `experiments/extended_experiments/exp2_noise_ablation.py` -> datetime, json, matplotlib.pyplot, numpy, pathlib, seaborn, torch, torch.nn, torch.nn.functional
- `experiments/extended_experiments/exp3_prospective_prediction.py` -> datetime, json, matplotlib.pyplot, numpy, pathlib, scipy, seaborn, sklearn.metrics, torch, torch.nn, torch.nn.functional
- `experiments/extended_experiments/exp4_trajectory_perturbation.py` -> datetime, json, matplotlib.pyplot, numpy, pathlib, seaborn, torch, torch.nn, torch.nn.functional
- `experiments/extended_experiments/run_all_experiments.py` -> datetime, json, matplotlib.pyplot, numpy, os, pathlib, scipy, seaborn, sys, torch, torch.nn, torch.nn.functional
- `experiments/extended_experiments/validate2.py` -> argparse, collections, copy, dataclasses, datetime, json, matplotlib.pyplot, numpy, os, pathlib, scipy.stats, sklearn.metrics, sys, time, torch, torch.nn, torch.optim, torch.utils.data, typing, validate_all_revisor_experiments, warnings
- `experiments/generate_figures.py` -> json, matplotlib.patches, matplotlib.pyplot, mpl_toolkits.mplot3d, numpy, os, scipy, seaborn, sklearn.cluster, sklearn.decomposition, sys, torch, warnings
- `experiments/statistics/coherence_analysis.py` -> json, numpy, threadpoolctl, time
- `experiments/statistics/rigorous_experiment.py` -> dataclasses, json, numpy, scipy, scipy.optimize, time, torch, torch.nn, typing, warnings
- `experiments/validation/benchmark.py` -> json, numpy, threadpoolctl, time
- `experiments/validation_experiments.py` -> itertools, json, matplotlib.pyplot, numpy, pathlib, torch
- `experiments/verify_checkpoints.py` -> numpy, pathlib, sys, torch, torch.nn
- `experimetn2.py` -> __future__, abc, argparse, copy, dataclasses, datetime, json, math, numpy, os, pathlib, random, sys, torch, torch.nn, torch.nn.functional, torch.utils.data, typing, warnings
- `fermi.py` -> argparse, dataclasses, datetime, glob, json, matplotlib.pyplot, numpy, os, pathlib, scipy.linalg, torch, torch.nn, traceback, typing
- `full_seed_prospector.py` -> abc, argparse, collections, dataclasses, datetime, enum, json, math, numpy, os, pathlib, signal, sys, time, torch, torch.nn, torch.optim, typing, warnings
- `grain.py` -> argparse, collections, dataclasses, datetime, glob, json, matplotlib.patches, matplotlib.pyplot, numpy, os, pathlib, signal, sys, time, torch, torch.nn, torch.optim, traceback, typing
- `gravity.py` -> argparse, dataclasses, datetime, glob, json, matplotlib.cm, matplotlib.colors, matplotlib.patches, matplotlib.pyplot, numpy, os, pathlib, scipy, scipy.linalg, scipy.stats, sklearn.metrics, torch, torch.nn, traceback, typing
- `grigori_perelmans_ricci_flow.py` -> abc, argparse, collections, dataclasses, datetime, json, matplotlib.pyplot, numpy, os, pathlib, random, torch, torch.nn, torch.nn.functional, traceback, typing
- `hawking_radiation.py` -> argparse, dataclasses, datetime, io, json, numpy, os, pathlib, pickle, scipy.stats, torch, torch.nn, traceback, typing, warnings
- `maxwell_strassen_analysis.py` -> argparse, dataclasses, datetime, json, matplotlib, matplotlib.pyplot, mpl_toolkits.mplot3d, numpy, os, pathlib, scipy.fft, scipy.stats, time, torch, torch.nn, typing, warnings
- `mbl_analyzer.py` -> argparse, dataclasses, datetime, glob, json, matplotlib.pyplot, numpy, os, pathlib, scipy.linalg, scipy.stats, time, torch, torch.nn, traceback, typing, warnings
- `menu.py` -> os, subprocess, sys, textwrap
- `percolation_analysis.py` -> argparse, dataclasses, datetime, glob, json, matplotlib, matplotlib.pyplot, numpy, os, pathlib, re, scipy.linalg, scipy.sparse, scipy.sparse.csgraph, scipy.stats, sys, time, torch, torch.nn, traceback, types, typing, warnings
- `plank.py` -> abc, argparse, collections, dataclasses, datetime, json, matplotlib.pyplot, numpy, os, pathlib, random, torch, torch.nn, typing
- `purity_index.py` -> argparse, dataclasses, datetime, glob, json, matplotlib.pyplot, numpy, os, pathlib, scipy.stats, torch, torch.nn, traceback, typing
- `repor_experiments.py` -> __future__, abc, argparse, copy, csv, dataclasses, datetime, json, math, numpy, os, pathlib, random, scipy.stats, sys, torch, torch.nn, torch.nn.functional, traceback, typing, warnings
- `scrodingger.py` -> argparse, dataclasses, datetime, glob, json, matplotlib.pyplot, numpy, os, pathlib, scipy.linalg, torch, torch.nn, traceback, typing
- `src/benchmarks/benchmark_final.py` -> ctypes, numpy, os, time
- `src/benchmarks/benchmark_scientific.py` -> ctypes, datetime, json, numpy, os, time, traceback
- `src/benchmarks/benchmark_strassen.py` -> dataclasses, gc, json, pathlib, strassen, sys, time, tomli, tomllib, torch, typing
- `src/benchmarks/strassen_numpy.py` -> functools, numpy, pathlib, torch
- `src/discovery/auto_T_discovery.py` -> dataclasses, numpy, sys, torch, torch.nn.functional, typing
- `src/native/strassen_c.c` -> stdio.h, stdlib.h, string.h
- `src/native/strassen_optimal.c` -> cblas.h, stdio.h, stdlib.h, string.h
- `src/native/strassen_turbo.c` -> immintrin.h, omp.h, stdio.h, stdlib.h, string.h
- `src/training/convergence_theory.py` -> dataclasses, numpy, time, torch, torch.nn, typing
- `src/training/grokkit_physics.py` -> ctypes, numpy, pathlib, sys, time
- `src/training/main.py` -> dataclasses, logging, numpy, os, pathlib, sys, time, torch, torch.nn, torch.optim, torch.utils.data, typing
- `src/training/main_pure_math.py` -> numpy, pathlib, torch, torch.nn, torch.optim
- `src/training/strassen_core.py` -> pathlib, torch
- `src/training/strassen_grokkit.py` -> math, torch, torch.nn, torch.optim
- `src/training/train_strassen.py` -> pathlib, torch, torch.nn, torch.optim
- `superposition.py` -> abc, argparse, dataclasses, datetime, json, logging, matplotlib.pyplot, numpy, pathlib, scipy.stats, time, torch, torch.nn, torch.nn.functional, torch.utils.data, tqdm, typing, warnings
- `train_batch_sweep.py` -> argparse, os, pathlib, torch, torch.nn, vector8
- `unified_hidden_connections_suite.py` -> __future__, abc, argparse, copy, dataclasses, datetime, json, math, numpy, os, pathlib, random, sys, torch, torch.nn, torch.nn.functional, torch.utils.data, typing, warnings
- `xray_tensor_diffractometer.py` -> abc, argparse, collections, dataclasses, datetime, json, logging, matplotlib.pyplot, numpy, os, pathlib, random, scipy.linalg, scipy.optimize, scipy.stats, seaborn, sklearn.decomposition, sys, threading, time, torch, torch.nn, traceback, typing, warnings
