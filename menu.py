#!/usr/bin/env python3
"""
Unified Menu for Strassen Algorithmic Crystallization Experiments
================================================================

Interactive navigator for all training, experiments, analysis,
benchmarks, and physics-inspired scripts in this repository.

Usage:
    python menu.py
"""

import os
import sys
import subprocess
import textwrap

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------------------
# Registry: every runnable script grouped by category
# ---------------------------------------------------------------------------

CATEGORIES = [
    {
        "name": "Training",
        "description": "Train neural networks to discover Strassen's algorithm",
        "items": [
            {
                "label": "Progressive Masking Discovery",
                "script": "src/training/main.py",
                "info": "Two-phase protocol: train 8 slots then mask weakest to 7.",
            },
            {
                "label": "Thermodynamic Grokking",
                "script": "src/training/train_strassen.py",
                "info": "Weight decay as pressure -> grokking -> sparsification -> discretization.",
            },
            {
                "label": "Grokkit Framework",
                "script": "src/training/strassen_grokkit.py",
                "info": "Spectral tensor decomposition with LC/SP metrics.",
            },
            {
                "label": "Physics-Inspired Grokking",
                "script": "src/training/grokkit_physics.py",
                "info": "Physics constraints applied to the grokking process.",
            },
            {
                "label": "Pure Mathematical Approach",
                "script": "src/training/main_pure_math.py",
                "info": "Algebraic formulation without physics heuristics.",
            },
            {
                "label": "Convergence Theory",
                "script": "src/training/convergence_theory.py",
                "info": "Theoretical convergence analysis of the training process.",
            },
            {
                "label": "Seed Prospector (Unified Framework)",
                "script": "full_seed_prospector.py",
                "info": "Seed mining (prospect) or full training (train) with Riemann flow.",
                "args_prompt": "Enter mode and arguments (e.g. 'prospect' or 'train --seed 42'): ",
            },
            {
                "label": "Batch Size Sweep Trainer",
                "script": "train_batch_sweep.py",
                "info": "Hyperparameter sweep across batch sizes.",
            },
        ],
    },
    {
        "name": "Extended Experiments",
        "description": "Core experiments validating kappa as order parameter",
        "items": [
            {
                "label": "Exp 1 - Gradient Covariance Spectrometry",
                "script": "experiments/extended_experiments/exp1_covariance_spectrometry.py",
                "info": "Measures kappa = lambda_max/lambda_min of gradient covariance.",
            },
            {
                "label": "Exp 2 - Noise Ablation",
                "script": "experiments/extended_experiments/exp2_noise_ablation.py",
                "info": "Robustness to gradient, weight, and structured noise.",
            },
            {
                "label": "Exp 3 - Prospective Prediction",
                "script": "experiments/extended_experiments/exp3_prospective_prediction.py",
                "info": "Can kappa predict grokking before training completes? (AUC=1.000)",
            },
            {
                "label": "Exp 4 - Trajectory Perturbation",
                "script": "experiments/extended_experiments/exp4_trajectory_perturbation.py",
                "info": "Phase-space perturbation analysis: basin width and stability.",
            },
            {
                "label": "Exp 5 - Discreteness Attractors",
                "script": "experiments/extended_experiments/exp5_discreteness_attractors.py",
                "info": "Basin stability under iterative pruning (stable to 50%, collapses at 55%).",
            },
            {
                "label": "Run All Extended Experiments",
                "script": "experiments/extended_experiments/run_all_experiments.py",
                "info": "Master orchestrator: runs exp1-exp5 and generates summary.",
            },
            {
                "label": "Comprehensive Validation (validate2)",
                "script": "experiments/extended_experiments/validate2.py",
                "info": "Full validation suite for checkpoints, migration, and metrics.",
            },
        ],
    },
    {
        "name": "Ablation and Validation",
        "description": "Ablation studies and checkpoint verification",
        "items": [
            {
                "label": "Ablation Study",
                "script": "experiments/ablation/ablation_study.py",
                "info": "Architecture ablation experiments.",
            },
            {
                "label": "Ablation 8192",
                "script": "experiments/ablation/ablation_8192.py",
                "info": "Large-scale ablation at 8192 samples.",
            },
            {
                "label": "Validation Experiments",
                "script": "experiments/validation_experiments.py",
                "info": "Cross-validation experiments.",
            },
            {
                "label": "Verify Checkpoints",
                "script": "experiments/verify_checkpoints.py",
                "info": "Checkpoint integrity and correctness verification.",
            },
            {
                "label": "Appendix Experiments",
                "script": "experiments/apendix_experiments.py",
                "info": "Supplementary experiments referenced in the appendix.",
            },
            {
                "label": "Rigorous Statistical Experiment",
                "script": "experiments/statistics/rigorous_experiment.py",
                "info": "Statistically rigorous experimental protocol.",
            },
            {
                "label": "Coherence Analysis",
                "script": "experiments/statistics/coherence_analysis.py",
                "info": "Statistical coherence analysis across runs.",
            },
            {
                "label": "Validation Benchmark",
                "script": "experiments/validation/benchmark.py",
                "info": "Validation benchmark suite.",
            },
        ],
    },
    {
        "name": "Analysis",
        "description": "Metrics, crystallography, and materials-science analysis",
        "items": [
            {
                "label": "Measure Strassen (delta, kappa, LC, transfer)",
                "script": "measure_strassen.py",
                "info": "Comprehensive metrics: delta, kappa, LC, pruning robustness, zero-shot.",
                "args_prompt": "Enter arguments (e.g. '--checkpoint checkpoints/strassen_exact.pt --output out.json'): ",
            },
            {
                "label": "Crystallography Phase Analysis",
                "script": "crystallography.py",
                "info": "Crystal vs glass phase classification via materials science.",
            },
            {
                "label": "Purity Index",
                "script": "purity_index.py",
                "info": "Purity index: phase classification, entropy, effective temperature.",
            },
            {
                "label": "Superposition Analysis (SAE)",
                "script": "superposition.py",
                "info": "Sparse autoencoder feature discovery and superposition measurement.",
            },
            {
                "label": "Boltzmann Statistics",
                "script": "boltzmann_experiments.py",
                "info": "Gradient noise resilience, perturbation effects, gauge invariance.",
            },
            {
                "label": "Percolation Analysis",
                "script": "percolation_analysis.py",
                "info": "Percolation thresholds, level spacing, participation ratio.",
            },
            {
                "label": "Grain Boundary Detection",
                "script": "grain.py",
                "info": "Microstructural grain boundary analysis in weight space.",
            },
            {
                "label": "Cache Analysis v2",
                "script": "experiments/cache_analysis_v2.py",
                "info": "Memory cache effects on training dynamics.",
            },
            {
                "label": "Compute GNS Checkpoints",
                "script": "compute_gns_checkpoints.py",
                "info": "Generalized Noise Scale metrics for all checkpoints.",
            },
            {
                "label": "Generate Figures",
                "script": "experiments/generate_figures.py",
                "info": "Generate publication-ready figures from results.",
            },
        ],
    },
    {
        "name": "Benchmarks",
        "description": "Performance and accuracy benchmarks",
        "items": [
            {
                "label": "Zero-Shot Expansion Benchmark",
                "script": "src/benchmarks/benchmark_strassen.py",
                "info": "Benchmarks from 2x2 to 2048x2048: speedup, error, memory.",
            },
            {
                "label": "Scientific Accuracy Benchmark",
                "script": "src/benchmarks/benchmark_scientific.py",
                "info": "Numerical precision across float32, float64, int32.",
            },
            {
                "label": "Final Performance Benchmark",
                "script": "src/benchmarks/benchmark_final.py",
                "info": "Consolidated final benchmark results.",
            },
            {
                "label": "NumPy Baseline",
                "script": "src/benchmarks/strassen_numpy.py",
                "info": "Reference NumPy implementation for comparison.",
            },
        ],
    },
    {
        "name": "Physics-Inspired Analysis",
        "description": "Quantum, thermodynamic, and field-theoretic perspectives",
        "items": [
            {
                "label": "Fermi Surface Analysis",
                "script": "fermi.py",
                "info": "Band structure, DOS, conductivity, metal-insulator classification.",
            },
            {
                "label": "Gravitational / EM Field Analysis",
                "script": "gravity.py",
                "info": "Poisson equation, Gauss law, Landauer and Heisenberg inequalities.",
            },
            {
                "label": "Planck Constant Extraction",
                "script": "plank.py",
                "info": "Planck constant from checkpoints, resilience and gauge tests.",
            },
            {
                "label": "Dirac Poles and Zeros",
                "script": "dirac_polos_zeros.py",
                "info": "Control-theory pole/zero analysis in Dirac notation.",
            },
            {
                "label": "Schrodinger-Inspired Analysis",
                "script": "scrodingger.py",
                "info": "Schrodinger equation perspective on weight dynamics.",
            },
            {
                "label": "Ricci Flow (Perelman)",
                "script": "grigori_perelmans_ricci_flow.py",
                "info": "Geometric curvature flow through weight space.",
            },
            {
                "label": "X-Ray Tensor Diffractometer",
                "script": "xray_tensor_diffractometer.py",
                "info": "Reciprocal space, Miller indices, Bragg conditions, structure factors.",
            },
            {
                "label": "Batch Size Thermodynamics",
                "script": "batch_size.py",
                "info": "Batch size as inverse temperature: Var(grad L) ~ 1/B.",
            },
            {
                "label": "Auto Temperature Discovery",
                "script": "src/discovery/auto_T_discovery.py",
                "info": "Automatic temperature parameter discovery.",
            },
        ],
    },
]


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

TERM_WIDTH = min(os.get_terminal_size().columns, 90) if sys.stdout.isatty() else 80
SEP = "-" * TERM_WIDTH


def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def print_header(title, subtitle=""):
    clear_screen()
    print(SEP)
    print(title.center(TERM_WIDTH))
    if subtitle:
        print(subtitle.center(TERM_WIDTH))
    print(SEP)
    print()


def print_wrapped(text, indent=4):
    for line in textwrap.wrap(text, width=TERM_WIDTH - indent):
        print(" " * indent + line)


def wait_for_enter():
    print()
    input("Press ENTER to return to the menu...")


# ---------------------------------------------------------------------------
# Run a script
# ---------------------------------------------------------------------------

def run_script(entry):
    script_path = os.path.join(BASE_DIR, entry["script"])

    if not os.path.isfile(script_path):
        print(f"\n[ERROR] Script not found: {entry['script']}")
        wait_for_enter()
        return

    extra_args = []
    if "args_prompt" in entry:
        raw = input(f"\n{entry['args_prompt']}").strip()
        if raw:
            extra_args = raw.split()

    cmd = [sys.executable, script_path] + extra_args
    print(f"\nRunning: {' '.join(cmd)}\n")
    print(SEP)

    try:
        subprocess.run(cmd, cwd=BASE_DIR)
    except KeyboardInterrupt:
        print("\n\n[Interrupted by user]")

    wait_for_enter()


# ---------------------------------------------------------------------------
# List checkpoints
# ---------------------------------------------------------------------------

def show_checkpoints():
    print_header("Available Checkpoints")
    ckpt_dir = os.path.join(BASE_DIR, "checkpoints")
    if not os.path.isdir(ckpt_dir):
        print("  No checkpoints/ directory found.")
        wait_for_enter()
        return

    files = sorted(f for f in os.listdir(ckpt_dir) if f.endswith(".pt"))
    if not files:
        print("  No .pt files found in checkpoints/.")
        wait_for_enter()
        return

    for i, f in enumerate(files, 1):
        size_kb = os.path.getsize(os.path.join(ckpt_dir, f)) / 1024
        print(f"  {i:2d}. {f:<45s} ({size_kb:.1f} KB)")

    extra_pts = []
    training_dir = os.path.join(BASE_DIR, "src", "training")
    if os.path.isdir(training_dir):
        extra_pts = sorted(f for f in os.listdir(training_dir) if f.endswith(".pt"))
    if extra_pts:
        print(f"\n  Also in src/training/:")
        for f in extra_pts:
            size_kb = os.path.getsize(os.path.join(training_dir, f)) / 1024
            print(f"      {f:<45s} ({size_kb:.1f} KB)")

    wait_for_enter()


# ---------------------------------------------------------------------------
# Show results summary
# ---------------------------------------------------------------------------

def show_results():
    print_header("Results and Outputs")

    result_files = [
        ("experiments/extended_experiments/all_experiments_results.json",
         "Extended experiments compiled results"),
        ("analysis_output/summary.json",
         "Analysis output summary"),
        ("analysis_output/summary_report.txt",
         "Analysis report (text)"),
        ("batch_size_experiment.csv",
         "Batch size sweep results"),
        ("RESULTS.md",
         "Main results document"),
    ]

    found_any = False
    for rel_path, desc in result_files:
        full = os.path.join(BASE_DIR, rel_path)
        if os.path.isfile(full):
            size_kb = os.path.getsize(full) / 1024
            print(f"  [{size_kb:7.1f} KB]  {rel_path}")
            print(f"              {desc}")
            print()
            found_any = True

    if not found_any:
        print("  No result files found yet. Run some experiments first.")

    wait_for_enter()


# ---------------------------------------------------------------------------
# Category menu
# ---------------------------------------------------------------------------

def show_category(cat):
    while True:
        print_header(cat["name"], cat["description"])
        items = cat["items"]
        for i, item in enumerate(items, 1):
            exists = os.path.isfile(os.path.join(BASE_DIR, item["script"]))
            marker = " " if exists else "?"
            print(f"  {marker} {i:2d}. {item['label']}")
        print()
        print(f"   0. Back")
        print()

        choice = input("Select option: ").strip()
        if choice == "0" or choice == "":
            return

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(items):
                item = items[idx]
                print_header(item["label"])
                print_wrapped(item["info"])
                print(f"\n    Script: {item['script']}")
                exists = os.path.isfile(os.path.join(BASE_DIR, item["script"]))
                if not exists:
                    print("    [WARNING] File not found")
                print()
                action = input("Run this script? [y/N]: ").strip().lower()
                if action == "y":
                    run_script(item)
            else:
                print("Invalid option.")
        except ValueError:
            print("Invalid input.")


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def main_menu():
    while True:
        print_header(
            "Strassen Algorithmic Crystallization",
            "Engineering Algorithmic Structure in Neural Networks"
        )

        for i, cat in enumerate(CATEGORIES, 1):
            count = len(cat["items"])
            print(f"  {i}. {cat['name']:<35s} ({count} scripts)")

        print()
        print(f"  C. Checkpoints")
        print(f"  R. Results and outputs")
        print(f"  Q. Quit")
        print()

        choice = input("Select category: ").strip().upper()

        if choice == "Q":
            print("\nGoodbye.\n")
            sys.exit(0)
        elif choice == "C":
            show_checkpoints()
        elif choice == "R":
            show_results()
        else:
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(CATEGORIES):
                    show_category(CATEGORIES[idx])
                else:
                    print("Invalid option.")
            except ValueError:
                print("Invalid input.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    try:
        main_menu()
    except KeyboardInterrupt:
        print("\n\nGoodbye.\n")
        sys.exit(0)
