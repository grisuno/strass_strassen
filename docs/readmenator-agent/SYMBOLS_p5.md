# Symbols (page 5 of 5)
Previous: [SYMBOLS_p4.md](SYMBOLS_p4.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `_compute_generalization_entropy` | method | `xray_tensor_diffractometer.py:2361` | `def _compute_generalization_entropy(self, params, successful_ckpts)` |
| `_compute_spectral_entropy` | method | `xray_tensor_diffractometer.py:719` | `def _compute_spectral_entropy(power_spectrum)` |
| `_discretize_to_integers` | method | `xray_tensor_diffractometer.py:847` | `def _discretize_to_integers(factors)` |
| `_estimate_critical_temperature` | method | `xray_tensor_diffractometer.py:2162` | `def _estimate_critical_temperature(self, results)` |
| `_find_broken_symmetries` | method | `xray_tensor_diffractometer.py:2518` | `def _find_broken_symmetries(self, coeffs)` |
| `_fit_extensivity` | method | `xray_tensor_diffractometer.py:2487` | `def _fit_extensivity(self, errors, sizes, purity)` |
| `_fit_timescale` | method | `xray_tensor_diffractometer.py:2414` | `def _fit_timescale(self, entropy_values)` |
| `_generate_comparative_report` | method | `xray_tensor_diffractometer.py:604` | `def _generate_comparative_report(self, results)` |
| `_get_boundary_mask` | method | `xray_tensor_diffractometer.py:1329` | `def _get_boundary_mask(self, weight)` |
| `_initialize_symmetric` | method | `xray_tensor_diffractometer.py:199` | `def _initialize_symmetric(self)` |
| `_load_all_checkpoints` | method | `xray_tensor_diffractometer.py:1503` | `def _load_all_checkpoints(self)` |
| `_load_model` | method | `xray_tensor_diffractometer.py:2617` | `def _load_model(self, path, device)` |
| `_load_seed_crystal` | method | `xray_tensor_diffractometer.py:242` | `def _load_seed_crystal(self)` |
| `_measure_uncertainty` | method | `xray_tensor_diffractometer.py:2526` | `def _measure_uncertainty(self, coeffs, basis)` |
| `_migrate_coefs_format` | method | `xray_tensor_diffractometer.py:1467` | `def _migrate_coefs_format(state_dict)` |
| `_migrate_custom_format` | method | `xray_tensor_diffractometer.py:1443` | `def _migrate_custom_format(state_dict)` |
| `_migrate_dict` | method | `xray_tensor_diffractometer.py:1431` | `def _migrate_dict(state_dict)` |
| `_migrate_encoder_format` | method | `xray_tensor_diffractometer.py:1475` | `def _migrate_encoder_format(state_dict)` |
| `_migrate_standard_format` | method | `xray_tensor_diffractometer.py:1487` | `def _migrate_standard_format(state_dict)` |
| `_plot_diffraction_pattern` | method | `xray_tensor_diffractometer.py:2097` | `def _plot_diffraction_pattern(self, diffraction_data, ckpt_name)` |
| `_plot_entropy_production` | method | `xray_tensor_diffractometer.py:2424` | `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)` |
| `_plot_epitaxial_evolution` | method | `xray_tensor_diffractometer.py:556` | `def _plot_epitaxial_evolution(self, annealing_results, target_size, seed_name)` |
| `_plot_extensivity` | method | `xray_tensor_diffractometer.py:2505` | `def _plot_extensivity(self, sizes, errors, purity, ckpt_name)` |
| `_plot_parameter_distribution` | method | `xray_tensor_diffractometer.py:2322` | `def _plot_parameter_distribution(self, params, group_name, kde)` |
| `_plot_phase_diagram` | method | `xray_tensor_diffractometer.py:2188` | `def _plot_phase_diagram(self, results)` |
| `_plot_temperature_vs_purity` | method | `xray_tensor_diffractometer.py:2213` | `def _plot_temperature_vs_purity(self, results)` |
| `_plot_uncertainty_distribution` | method | `xray_tensor_diffractometer.py:2537` | `def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)` |
| `_print_executive_summary` | method | `xray_tensor_diffractometer.py:2558` | `def _print_executive_summary(self, results)` |
| `_recursive_strassen` | method | `xray_tensor_diffractometer.py:2453` | `def _recursive_strassen(self, A, B, coeffs, N)` |
| `_save_report` | method | `xray_tensor_diffractometer.py:2668` | `def _save_report(self, report)` |
| `_save_results` | method | `xray_tensor_diffractometer.py:2587` | `def _save_results(self, results, filename)` |
| `_save_superlattice_seed` | method | `xray_tensor_diffractometer.py:2131` | `def _save_superlattice_seed(self, superlattice, ckpt_name)` |
| `_simulate_training_trajectory` | method | `xray_tensor_diffractometer.py:2350` | `def _simulate_training_trajectory(self, final_params, final_delta)` |
| `_verify_entropy_extensivity` | method | `xray_tensor_diffractometer.py:2174` | `def _verify_entropy_extensivity(self, results)` |
| `_verify_extensivity_universality` | method | `xray_tensor_diffractometer.py:2501` | `def _verify_extensivity_universality(self, results)` |
| `_verify_scaling` | method | `xray_tensor_diffractometer.py:2442` | `def _verify_scaling(self, coeffs, N)` |
| `analyze_poynting_flow` | method | `xray_tensor_diffractometer.py:1849` | `def analyze_poynting_flow(self)` |
| `anneal_crystal` | method | `xray_tensor_diffractometer.py:360` | `def anneal_crystal(self, model, max_epochs, early_stop_threshold)` |
| `calculate_carnot_efficiency` | method | `xray_tensor_diffractometer.py:1137` | `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)` |
| `check_extensivity` | method | `xray_tensor_diffractometer.py:1083` | `def check_extensivity(entropy_list, scale_factors)` |
| `compute` | method | `xray_tensor_diffractometer.py:150` | `def compute(self, model)` |
| `compute_all_metrics` | method | `xray_tensor_diffractometer.py:1246` | `def compute_all_metrics(model, dataloader)` |
| `compute_alpha_purity` | method | `xray_tensor_diffractometer.py:1200` | `def compute_alpha_purity(coeffs)` |
| `compute_boundary_gradient` | method | `xray_tensor_diffractometer.py:1275` | `def compute_boundary_gradient(self, weight)` |
| `compute_bulk_gradient` | method | `xray_tensor_diffractometer.py:1290` | `def compute_bulk_gradient(self, weight)` |
| `compute_critical_exponents` | method | `xray_tensor_diffractometer.py:930` | `def compute_critical_exponents(temp_history, cv_history, alpha_history)` |
| `compute_discretization_margin` | method | `xray_tensor_diffractometer.py:1187` | `def compute_discretization_margin(coeffs)` |
| `compute_effective_temperature` | method | `xray_tensor_diffractometer.py:916` | `def compute_effective_temperature(gradient_buffer, learning_rate)` |
| `compute_equation_of_state` | method | `xray_tensor_diffractometer.py:1009` | `def compute_equation_of_state(temp_eff, alpha, kappa)` |
| `compute_fisher_information_matrix` | method | `xray_tensor_diffractometer.py:1107` | `def compute_fisher_information_matrix(model, samples)` |
| `compute_gibbs_free_energy` | method | `xray_tensor_diffractometer.py:785` | `def compute_gibbs_free_energy(loss, temp, entropy)` |
| `compute_kappa` | method | `xray_tensor_diffractometer.py:1161` | `def compute_kappa(model, dataloader, num_batches)` |
| `compute_kappa_quantum` | method | `xray_tensor_diffractometer.py:1207` | `def compute_kappa_quantum(coeffs, hbar)` |
| `compute_local_complexity` | method | `xray_tensor_diffractometer.py:1191` | `def compute_local_complexity(model)` |
| `compute_mutual_information` | method | `xray_tensor_diffractometer.py:1069` | `def compute_mutual_information(weights, gradients)` |
| `compute_poynting_vector` | method | `xray_tensor_diffractometer.py:1224` | `def compute_poynting_vector(coeffs)` |
| `compute_ricci_curvature` | method | `xray_tensor_diffractometer.py:1126` | `def compute_ricci_curvature(fisher_matrix)` |
| `compute_specific_heat` | method | `xray_tensor_diffractometer.py:1048` | `def compute_specific_heat(loss_history, temp_history, cv_threshold)` |
| `compute_weight_diffraction` | method | `xray_tensor_diffractometer.py:696` | `def compute_weight_diffraction(coeffs)` |
| `convert_to_serializable` | method | `xray_tensor_diffractometer.py:2590` | `def convert_to_serializable(obj)` |
| `create_superlattice_seed` | method | `xray_tensor_diffractometer.py:892` | `def create_superlattice_seed(base_tensor, scale_factor)` |
| `dataloader` | method | `xray_tensor_diffractometer.py:1522` | `def dataloader()` |
| `dataloader` | method | `xray_tensor_diffractometer.py:2637` | `def dataloader()` |
| `estimate_hbar_algorithmic` | method | `xray_tensor_diffractometer.py:1061` | `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)` |
| `extract_canonical_decomposition` | method | `xray_tensor_diffractometer.py:791` | `def extract_canonical_decomposition(coeffs, rank)` |
| `extract_lattice_parameters` | method | `xray_tensor_diffractometer.py:726` | `def extract_lattice_parameters(weight_tensor, rank)` |
| `forward` | method | `xray_tensor_diffractometer.py:204` | `def forward(self, a, b)` |
| `generate_batch` | method | `xray_tensor_diffractometer.py:153` | `def generate_batch(self, batch_size)` |
| `generate_batch` | method | `xray_tensor_diffractometer.py:168` | `def generate_batch(batch_size)` |
| `generate_batch` | method | `xray_tensor_diffractometer.py:373` | `def generate_batch(batch_size)` |
| `get_coefficients` | method | `xray_tensor_diffractometer.py:207` | `def get_coefficients(self)` |
| `get_tensor` | method | `xray_tensor_diffractometer.py:1444` | `def get_tensor(key)` |
| `gibbs_free_energy` | method | `xray_tensor_diffractometer.py:684` | `def gibbs_free_energy(self)` |
| `grow_epitaxial_crystal` | method | `xray_tensor_diffractometer.py:265` | `def grow_epitaxial_crystal(self)` |
| `helmholtz_free_energy` | method | `xray_tensor_diffractometer.py:680` | `def helmholtz_free_energy(self)` |
| `is_stable` | method | `xray_tensor_diffractometer.py:689` | `def is_stable(self)` |
| `load_checkpoint` | method | `xray_tensor_diffractometer.py:147` | `def load_checkpoint(self, path, device)` |
| `load_checkpoint` | method | `xray_tensor_diffractometer.py:1372` | `def load_checkpoint(self, path, device)` |
| `main` | method | `xray_tensor_diffractometer.py:2683` | `def main()` |
| `migrate_checkpoint` | method | `xray_tensor_diffractometer.py:1407` | `def migrate_checkpoint(raw_data)` |
| `model` | method | `xray_tensor_diffractometer.py:2415` | `def model(t, A, tau, C)` |
| `model` | method | `xray_tensor_diffractometer.py:2488` | `def model(N, alpha, beta)` |
| `phase1_molecular_hypothesis` | method | `xray_tensor_diffractometer.py:1575` | `def phase1_molecular_hypothesis(self)` |
| `phase2_entropy_production` | method | `xray_tensor_diffractometer.py:1660` | `def phase2_entropy_production(self)` |
| `phase3_extensivity_law` | method | `xray_tensor_diffractometer.py:1742` | `def phase3_extensivity_law(self)` |
| `phase4_quantum_basis_transform` | method | `xray_tensor_diffractometer.py:1796` | `def phase4_quantum_basis_transform(self)` |
| `phase5_thermodynamic_analysis` | method | `xray_tensor_diffractometer.py:1882` | `def phase5_thermodynamic_analysis(self)` |
| `phase6_spectroscopic_analysis` | method | `xray_tensor_diffractometer.py:2011` | `def phase6_spectroscopic_analysis(self)` |
| `run_epitaxial_growth_experiment` | method | `xray_tensor_diffractometer.py:479` | `def run_epitaxial_growth_experiment(self, seed_checkpoint, target_sizes)` |
| `run_epitaxy_from_best_crystal` | method | `xray_tensor_diffractometer.py:86` | `def run_epitaxy_from_best_crystal(checkpoint_dir, target_sizes)` |
| `run_full_analysis` | method | `xray_tensor_diffractometer.py:2634` | `def run_full_analysis(self)` |
| `run_full_boltzmann_program` | method | `xray_tensor_diffractometer.py:1548` | `def run_full_boltzmann_program(self)` |
| `run_green_backprop_step` | method | `xray_tensor_diffractometer.py:1296` | `def run_green_backprop_step(self, A, B, C_true, lambda_boundary)` |
| `sample_dataloader` | method | `xray_tensor_diffractometer.py:1906` | `def sample_dataloader()` |
| `sample_dataloader` | method | `xray_tensor_diffractometer.py:2039` | `def sample_dataloader()` |
| `set_seed` | method | `xray_tensor_diffractometer.py:64` | `def set_seed(seed)` |
| `setup_logger` | method | `xray_tensor_diffractometer.py:72` | `def setup_logger(name, level)` |
| `train_with_green_cow` | method | `xray_tensor_diffractometer.py:1341` | `def train_with_green_cow(self, epochs, lr, lambda_boundary)` |
| `verify_structure` | method | `xray_tensor_diffractometer.py:179` | `def verify_structure(coeffs)` |

