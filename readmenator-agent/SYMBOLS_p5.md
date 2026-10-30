# Symbols (page 5 of 5)
Previous: [SYMBOLS_p4.md](SYMBOLS_p4.md)

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
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
| `convert_to_serializable` | function | `xray_tensor_diffractometer.py:2590` | `def convert_to_serializable(obj)` |
| `create_superlattice_seed` | method | `xray_tensor_diffractometer.py:892` | `def create_superlattice_seed(base_tensor, scale_factor)` |
| `dataloader` | function | `xray_tensor_diffractometer.py:1522` | `def dataloader()` |
| `dataloader` | function | `xray_tensor_diffractometer.py:2637` | `def dataloader()` |
| `estimate_hbar_algorithmic` | method | `xray_tensor_diffractometer.py:1061` | `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)` |
| `extract_canonical_decomposition` | method | `xray_tensor_diffractometer.py:791` | `def extract_canonical_decomposition(coeffs, rank)` |
| `extract_lattice_parameters` | method | `xray_tensor_diffractometer.py:726` | `def extract_lattice_parameters(weight_tensor, rank)` |
| `forward` | method | `xray_tensor_diffractometer.py:204` | `def forward(self, a, b)` |
| `generate_batch` | method | `xray_tensor_diffractometer.py:153` | `def generate_batch(self, batch_size)` |
| `generate_batch` | method | `xray_tensor_diffractometer.py:168` | `def generate_batch(batch_size)` |
| `generate_batch` | function | `xray_tensor_diffractometer.py:373` | `def generate_batch(batch_size)` |
| `get_coefficients` | method | `xray_tensor_diffractometer.py:207` | `def get_coefficients(self)` |
| `get_tensor` | function | `xray_tensor_diffractometer.py:1444` | `def get_tensor(key)` |
| `gibbs_free_energy` | method | `xray_tensor_diffractometer.py:684` | `def gibbs_free_energy(self)` |
| `grow_epitaxial_crystal` | method | `xray_tensor_diffractometer.py:265` | `def grow_epitaxial_crystal(self)` |
| `helmholtz_free_energy` | method | `xray_tensor_diffractometer.py:680` | `def helmholtz_free_energy(self)` |
| `is_stable` | method | `xray_tensor_diffractometer.py:689` | `def is_stable(self)` |
| `load_checkpoint` | method | `xray_tensor_diffractometer.py:147` | `def load_checkpoint(self, path, device)` |
| `load_checkpoint` | method | `xray_tensor_diffractometer.py:1372` | `def load_checkpoint(self, path, device)` |
| `main` | function | `xray_tensor_diffractometer.py:2683` | `def main()` |
| `migrate_checkpoint` | method | `xray_tensor_diffractometer.py:1407` | `def migrate_checkpoint(raw_data)` |
| `model` | function | `xray_tensor_diffractometer.py:2415` | `def model(t, A, tau, C)` |
| `model` | function | `xray_tensor_diffractometer.py:2488` | `def model(N, alpha, beta)` |
| `phase1_molecular_hypothesis` | method | `xray_tensor_diffractometer.py:1575` | `def phase1_molecular_hypothesis(self)` |
| `phase2_entropy_production` | method | `xray_tensor_diffractometer.py:1660` | `def phase2_entropy_production(self)` |
| `phase3_extensivity_law` | method | `xray_tensor_diffractometer.py:1742` | `def phase3_extensivity_law(self)` |
| `phase4_quantum_basis_transform` | method | `xray_tensor_diffractometer.py:1796` | `def phase4_quantum_basis_transform(self)` |
| `phase5_thermodynamic_analysis` | method | `xray_tensor_diffractometer.py:1882` | `def phase5_thermodynamic_analysis(self)` |
| `phase6_spectroscopic_analysis` | method | `xray_tensor_diffractometer.py:2011` | `def phase6_spectroscopic_analysis(self)` |
| `run_epitaxial_growth_experiment` | method | `xray_tensor_diffractometer.py:479` | `def run_epitaxial_growth_experiment(self, seed_checkpoint, target_sizes)` |
| `run_epitaxy_from_best_crystal` | function | `xray_tensor_diffractometer.py:86` | `def run_epitaxy_from_best_crystal(checkpoint_dir, target_sizes)` |
| `run_full_analysis` | method | `xray_tensor_diffractometer.py:2634` | `def run_full_analysis(self)` |
| `run_full_boltzmann_program` | method | `xray_tensor_diffractometer.py:1548` | `def run_full_boltzmann_program(self)` |
| `run_green_backprop_step` | method | `xray_tensor_diffractometer.py:1296` | `def run_green_backprop_step(self, A, B, C_true, lambda_boundary)` |
| `sample_dataloader` | function | `xray_tensor_diffractometer.py:1906` | `def sample_dataloader()` |
| `sample_dataloader` | function | `xray_tensor_diffractometer.py:2039` | `def sample_dataloader()` |
| `set_seed` | function | `xray_tensor_diffractometer.py:64` | `def set_seed(seed)` |
| `setup_logger` | function | `xray_tensor_diffractometer.py:72` | `def setup_logger(name, level)` |
| `train_with_green_cow` | method | `xray_tensor_diffractometer.py:1341` | `def train_with_green_cow(self, epochs, lr, lambda_boundary)` |
| `verify_structure` | method | `xray_tensor_diffractometer.py:179` | `def verify_structure(coeffs)` |

