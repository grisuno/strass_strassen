# API

## app.py

### __init__ `def __init__(self, rank)`
- Defined: `app.py:25`

### forward `def forward(self, A, B)`
- Defined: `app.py:31`

## batch_size.py

### set_random_seed `def set_random_seed(seed)`
- Defined: `batch_size.py:71`

### main `def main()`
- Defined: `batch_size.py:341`

### __init__ `def __init__(self, config)`
- Defined: `batch_size.py:78`

### forward `def forward(self, a, b)`
- Defined: `batch_size.py:88`

### get_coefficients `def get_coefficients(self)`
- Defined: `batch_size.py:91`

### compute_lambda_effective `def compute_lambda_effective(self)`
- Defined: `batch_size.py:94`

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `batch_size.py:102`

### migrate `def migrate(self, state_dict)`
- Defined: `batch_size.py:106`

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `batch_size.py:111`

### migrate `def migrate(self, state_dict)`
- Defined: `batch_size.py:114`

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `batch_size.py:123`

### migrate `def migrate(self, state_dict)`
- Defined: `batch_size.py:126`

### __init__ `def __init__(self)`
- Defined: `batch_size.py:132`

### migrate_checkpoint `def migrate_checkpoint(self, path, device)`
- Defined: `batch_size.py:135`

### generate_batch `def generate_batch(batch_size, config)`
- Defined: `batch_size.py:149`

### compute_kappa `def compute_kappa(model, num_batches, config)`
- Defined: `batch_size.py:158`

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `batch_size.py:174`

### compute_local_complexity `def compute_local_complexity(model, config)`
- Defined: `batch_size.py:178`

### __init__ `def __init__(self, metrics, training_metrics, config)`
- Defined: `batch_size.py:187`

### calculate_all `def calculate_all(self)`
- Defined: `batch_size.py:196`

### __init__ `def __init__(self, model, h_bar, delta_struct, config)`
- Defined: `batch_size.py:218`

### analyze_batch_size_spectrum `def analyze_batch_size_spectrum(self)`
- Defined: `batch_size.py:224`

### _measure_gradients `def _measure_gradients(self, batch_size)`
- Defined: `batch_size.py:246`

### __init__ `def __init__(self, config)`
- Defined: `batch_size.py:263`

### load `def load(self, path, device)`
- Defined: `batch_size.py:267`

### extract_training_metrics `def extract_training_metrics(self, path)`
- Defined: `batch_size.py:281`

### __init__ `def __init__(self, config)`
- Defined: `batch_size.py:293`

### analyze_checkpoint `def analyze_checkpoint(self, path, device)`
- Defined: `batch_size.py:297`

### analyze_directory `def analyze_directory(self, directory, device, pattern)`
- Defined: `batch_size.py:323`

## boltzmann_experiments.py

### set_seed `def set_seed(seed)`
- Defined: `boltzmann_experiments.py:30`

### main `def main()`
- Defined: `boltzmann_experiments.py:935`

### _simulate_training_trajectory `def _simulate_training_trajectory(self, final_params, final_delta)`
- Defined: `boltzmann_experiments.py:951`

### _compute_generalization_entropy `def _compute_generalization_entropy(self, params, successful_ckpts)`
- Defined: `boltzmann_experiments.py:962`
- Doc: Entropía de generalización con manejo robusto de dimensionalidad

### _fit_timescale `def _fit_timescale(self, entropy_values)`
- Defined: `boltzmann_experiments.py:1016`

### _plot_entropy_production `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- Defined: `boltzmann_experiments.py:1026`

### phase3_extensivity_law `def phase3_extensivity_law(self)`
- Defined: `boltzmann_experiments.py:1044`

### load_checkpoint `def load_checkpoint(self, path, device)`
- Defined: `boltzmann_experiments.py:42`

### load_checkpoint `def load_checkpoint(self, path, device)`
- Defined: `boltzmann_experiments.py:46`

### migrate_checkpoint `def migrate_checkpoint(raw_data)`
- Defined: `boltzmann_experiments.py:54`

### _format_direct_tensors `def _format_direct_tensors(tensor_dict)`
- Defined: `boltzmann_experiments.py:70`

### _migrate_dict `def _migrate_dict(state_dict)`
- Defined: `boltzmann_experiments.py:92`

### _migrate_encoder_format `def _migrate_encoder_format(state_dict)`
- Defined: `boltzmann_experiments.py:105`

### _migrate_coefs_format `def _migrate_coefs_format(state_dict)`
- Defined: `boltzmann_experiments.py:115`

### __init__ `def __init__(self, n_slots)`
- Defined: `boltzmann_experiments.py:119`

### _initialize_symmetric `def _initialize_symmetric(self)`
- Defined: `boltzmann_experiments.py:126`

### forward `def forward(self, a, b)`
- Defined: `boltzmann_experiments.py:131`

### get_coefficients `def get_coefficients(self)`
- Defined: `boltzmann_experiments.py:134`

### compute_kappa `def compute_kappa(coeffs)`
- Defined: `boltzmann_experiments.py:139`
- Doc: Classical kappa - will be inf for discrete states

### compute_delta `def compute_delta(coeffs)`
- Defined: `boltzmann_experiments.py:156`
- Doc: Discretization error δ

### compute_local_complexity `def compute_local_complexity(coeffs)`
- Defined: `boltzmann_experiments.py:161`

### compute_alpha_purity `def compute_alpha_purity(coeffs)`
- Defined: `boltzmann_experiments.py:169`
- Doc: Alpha purity: α = -log(δ), inverse temperature metric for discrete states

### compute_kappa_quantum `def compute_kappa_quantum(coeffs, hbar)`
- Defined: `boltzmann_experiments.py:178`
- Doc: Quantum-regularized kappa for singular covariance states

### __init__ `def __init__(self, checkpoint_dir, results_dir)`
- Defined: `boltzmann_experiments.py:200`

### _load_all_checkpoints `def _load_all_checkpoints(self)`
- Defined: `boltzmann_experiments.py:207`

### run_full_boltzmann_program `def run_full_boltzmann_program(self)`
- Defined: `boltzmann_experiments.py:247`

### _print_executive_summary `def _print_executive_summary(self, results)`
- Defined: `boltzmann_experiments.py:267`

### _save_results `def _save_results(self, results, filename)`
- Defined: `boltzmann_experiments.py:307`

### phase1_molecular_hypothesis `def phase1_molecular_hypothesis(self)`
- Defined: `boltzmann_experiments.py:328`

### _compute_entropy_simple `def _compute_entropy_simple(self, params)`
- Defined: `boltzmann_experiments.py:435`
- Doc: Entropía simple sin KDE para datos de baja varianza

### _compute_entropy `def _compute_entropy(self, params)`
- Defined: `boltzmann_experiments.py:446`
- Doc: Entropía con manejo robusto de covarianza

### _compute_effective_volume `def _compute_effective_volume(self, kde)`
- Defined: `boltzmann_experiments.py:466`

### _plot_parameter_distribution `def _plot_parameter_distribution(self, params, group_name, kde)`
- Defined: `boltzmann_experiments.py:475`

### phase2_entropy_production `def phase2_entropy_production(self)`
- Defined: `boltzmann_experiments.py:506`

### _simulate_training_trajectory `def _simulate_training_trajectory(self, final_params, final_delta)`
- Defined: `boltzmann_experiments.py:592`

### _compute_generalization_entropy `def _compute_generalization_entropy(self, params, successful_ckpts)`
- Defined: `boltzmann_experiments.py:604`
- Doc: Entropía de generalización con manejo robusto de datos idénticos

### _fit_timescale `def _fit_timescale(self, entropy_values)`
- Defined: `boltzmann_experiments.py:691`

### _plot_entropy_production `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- Defined: `boltzmann_experiments.py:701`

### phase3_extensivity_law `def phase3_extensivity_law(self)`
- Defined: `boltzmann_experiments.py:719`

### _verify_scaling `def _verify_scaling(self, coeffs, N)`
- Defined: `boltzmann_experiments.py:773`

### _recursive_strassen `def _recursive_strassen(self, A, B, coeffs, N)`
- Defined: `boltzmann_experiments.py:783`

### _fit_extensivity `def _fit_extensivity(self, errors, sizes, purity)`
- Defined: `boltzmann_experiments.py:812`

### _verify_extensivity_universality `def _verify_extensivity_universality(self, results)`
- Defined: `boltzmann_experiments.py:824`

### _plot_extensivity `def _plot_extensivity(self, sizes, errors, purity, ckpt_name)`
- Defined: `boltzmann_experiments.py:828`

### phase4_quantum_basis_transform `def phase4_quantum_basis_transform(self)`
- Defined: `boltzmann_experiments.py:841`

### _find_broken_symmetries `def _find_broken_symmetries(self, coeffs)`
- Defined: `boltzmann_experiments.py:895`

### _measure_uncertainty `def _measure_uncertainty(self, coeffs, basis)`
- Defined: `boltzmann_experiments.py:903`

### _plot_uncertainty_distribution `def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)`
- Defined: `boltzmann_experiments.py:914`

### model `def model(t, A, tau, C)`
- Defined: `boltzmann_experiments.py:1017`

### convert_to_serializable `def convert_to_serializable(obj)`
- Defined: `boltzmann_experiments.py:310`

### model `def model(t, A, tau, C)`
- Defined: `boltzmann_experiments.py:692`

### model `def model(N, alpha, beta)`
- Defined: `boltzmann_experiments.py:813`

## compute_gns_checkpoints.py

### estimate_gns `def estimate_gns(model, batch_size, num_batches)`
- Defined: `compute_gns_checkpoints.py:11`

### main `def main()`
- Defined: `compute_gns_checkpoints.py:37`

## crystallography.py

### set_seed `def set_seed(seed)`
- Defined: `crystallography.py:35`

### main `def main()`
- Defined: `crystallography.py:542`

### __init__ `def __init__(self, n_slots)`
- Defined: `crystallography.py:45`

### _initialize_symmetric `def _initialize_symmetric(self)`
- Defined: `crystallography.py:52`

### forward `def forward(self, a, b)`
- Defined: `crystallography.py:57`

### get_coefficients `def get_coefficients(self)`
- Defined: `crystallography.py:60`

### migrate_checkpoint `def migrate_checkpoint(path, device)`
- Defined: `crystallography.py:73`

### _migrate_custom `def _migrate_custom(state_dict)`
- Defined: `crystallography.py:106`
- Doc: Maneja formatos custom U,V,W directos

### _migrate_encoder `def _migrate_encoder(state_dict)`
- Defined: `crystallography.py:123`
- Doc: Extracción de encoder.layers

### _migrate_standard `def _migrate_standard(state_dict)`
- Defined: `crystallography.py:147`
- Doc: Formato estándar U.weight, V.weight, W.weight

### generate_batch `def generate_batch(batch_size)`
- Defined: `crystallography.py:157`

### verify_structure `def verify_structure(coeffs)`
- Defined: `crystallography.py:164`

### __init__ `def __init__(self, model)`
- Defined: `crystallography.py:173`

### prune_to_target `def prune_to_target(self, target)`
- Defined: `crystallography.py:176`

### discretize_weights `def discretize_weights(self, margin)`
- Defined: `crystallography.py:192`

### compute_kappa `def compute_kappa(model, dataloader, num_batches)`
- Defined: `crystallography.py:208`

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `crystallography.py:225`

### __init__ `def __init__(self, model)`
- Defined: `crystallography.py:233`

### test_gauge_invariance `def test_gauge_invariance(self, n_samples)`
- Defined: `crystallography.py:236`

### _functional_error `def _functional_error(self, test_coeffs)`
- Defined: `crystallography.py:262`

### __init__ `def __init__(self, model)`
- Defined: `crystallography.py:278`

### measure_resilience_spectrum `def measure_resilience_spectrum(self, noise_levels)`
- Defined: `crystallography.py:282`

### _test_noise_recovery `def _test_noise_recovery(self, sigma, n_trials)`
- Defined: `crystallography.py:293`

### _apply_noise `def _apply_noise(self, sigma)`
- Defined: `crystallography.py:312`

### _anneal_to_attractor `def _anneal_to_attractor(self, max_epochs)`
- Defined: `crystallography.py:317`

### _estimate_critical_noise `def _estimate_critical_noise(self, results)`
- Defined: `crystallography.py:329`

### __init__ `def __init__(self, model, diffraction_results, resilience_results, metrics_results)`
- Defined: `crystallography.py:346`

### compute `def compute(self)`
- Defined: `crystallography.py:359`

### _assign_grade `def _assign_grade(self, index, delta)`
- Defined: `crystallography.py:399`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `crystallography.py:417`

### run_full_analysis `def run_full_analysis(self)`
- Defined: `crystallography.py:445`

### _save_report `def _save_report(self, report)`
- Defined: `crystallography.py:506`

### compute `def compute(model)`
- Defined: `crystallography.py:525`
- Doc: Computa LC basado en Can't Stop Won't Stop paper

### dataloader_gen `def dataloader_gen()`
- Defined: `crystallography.py:465`

## dirac_polos_zeros.py

### main `def main()`
- Defined: `dirac_polos_zeros.py:1545`

### forward `def forward(self, a, b)`
- Defined: `dirac_polos_zeros.py:55`

### get_coefficients `def get_coefficients(self)`
- Defined: `dirac_polos_zeros.py:56`

### extract `def extract(self, model)`
- Defined: `dirac_polos_zeros.py:61`

### analyze `def analyze(self, charge_density)`
- Defined: `dirac_polos_zeros.py:66`

### calculate `def calculate(self, dirac_data, eval_points)`
- Defined: `dirac_polos_zeros.py:71`

### calculate `def calculate(self, electric_field, surface_points)`
- Defined: `dirac_polos_zeros.py:76`

### extract `def extract(self, model)`
- Defined: `dirac_polos_zeros.py:81`

### compute `def compute(self, A, B, C, D)`
- Defined: `dirac_polos_zeros.py:86`

### analyze_stability `def analyze_stability(self)`
- Defined: `dirac_polos_zeros.py:91`

### get_poles `def get_poles(self)`
- Defined: `dirac_polos_zeros.py:92`

### get_zeros `def get_zeros(self)`
- Defined: `dirac_polos_zeros.py:93`

### compute_bode `def compute_bode(self)`
- Defined: `dirac_polos_zeros.py:98`

### compute_margins `def compute_margins(self)`
- Defined: `dirac_polos_zeros.py:99`

### compute_nyquist `def compute_nyquist(self)`
- Defined: `dirac_polos_zeros.py:100`

### compute_step `def compute_step(self)`
- Defined: `dirac_polos_zeros.py:105`

### compute_impulse `def compute_impulse(self)`
- Defined: `dirac_polos_zeros.py:106`

### load `def load(self, path, device)`
- Defined: `dirac_polos_zeros.py:111`

### migrate `def migrate(self, raw_data)`
- Defined: `dirac_polos_zeros.py:116`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:121`

### __init__ `def __init__(self, hidden_dim, matrix_size)`
- Defined: `dirac_polos_zeros.py:125`

### _initialize `def _initialize(self)`
- Defined: `dirac_polos_zeros.py:136`

### forward `def forward(self, a, b)`
- Defined: `dirac_polos_zeros.py:141`

### get_coefficients `def get_coefficients(self)`
- Defined: `dirac_polos_zeros.py:144`

### extract `def extract(self, model)`
- Defined: `dirac_polos_zeros.py:153`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:161`

### analyze `def analyze(self, charge_density)`
- Defined: `dirac_polos_zeros.py:164`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:193`

### calculate `def calculate(self, dirac_data, eval_points)`
- Defined: `dirac_polos_zeros.py:196`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:226`

### calculate `def calculate(self, electric_field, surface_points)`
- Defined: `dirac_polos_zeros.py:229`

### calculate `def calculate(self, electric_field)`
- Defined: `dirac_polos_zeros.py:249`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:254`

### verify `def verify(self, dirac_data, flux_data)`
- Defined: `dirac_polos_zeros.py:257`

### extract `def extract(self, model)`
- Defined: `dirac_polos_zeros.py:272`

### compute `def compute(self, A, B, C, D)`
- Defined: `dirac_polos_zeros.py:299`

### __init__ `def __init__(self, numerator, denominator, config)`
- Defined: `dirac_polos_zeros.py:314`

### _compute `def _compute(self)`
- Defined: `dirac_polos_zeros.py:323`

### get_poles `def get_poles(self)`
- Defined: `dirac_polos_zeros.py:334`

### get_zeros `def get_zeros(self)`
- Defined: `dirac_polos_zeros.py:337`

### analyze_stability `def analyze_stability(self)`
- Defined: `dirac_polos_zeros.py:340`

### classify_poles `def classify_poles(self)`
- Defined: `dirac_polos_zeros.py:378`

### compute_damping `def compute_damping(self)`
- Defined: `dirac_polos_zeros.py:406`

### compute_time_constants `def compute_time_constants(self)`
- Defined: `dirac_polos_zeros.py:445`

### __init__ `def __init__(self, numerator, denominator, config)`
- Defined: `dirac_polos_zeros.py:464`

### compute_bode `def compute_bode(self)`
- Defined: `dirac_polos_zeros.py:474`

### compute_margins `def compute_margins(self)`
- Defined: `dirac_polos_zeros.py:490`

### compute_nyquist `def compute_nyquist(self)`
- Defined: `dirac_polos_zeros.py:516`

### evaluate_nyquist_stability `def evaluate_nyquist_stability(self, nyquist_data)`
- Defined: `dirac_polos_zeros.py:535`

### __init__ `def __init__(self, numerator, denominator, config)`
- Defined: `dirac_polos_zeros.py:567`

### compute_step `def compute_step(self)`
- Defined: `dirac_polos_zeros.py:577`

### compute_impulse `def compute_impulse(self)`
- Defined: `dirac_polos_zeros.py:589`

### analyze_step_characteristics `def analyze_step_characteristics(self, step_data)`
- Defined: `dirac_polos_zeros.py:601`

### load `def load(self, path, device)`
- Defined: `dirac_polos_zeros.py:654`

### migrate `def migrate(self, raw_data)`
- Defined: `dirac_polos_zeros.py:662`

### _migrate_dict `def _migrate_dict(self, state_dict)`
- Defined: `dirac_polos_zeros.py:674`

### _migrate_custom_format `def _migrate_custom_format(self, state_dict)`
- Defined: `dirac_polos_zeros.py:683`

### _migrate_coefs_format `def _migrate_coefs_format(self, state_dict)`
- Defined: `dirac_polos_zeros.py:699`

### _migrate_standard_format `def _migrate_standard_format(self, state_dict)`
- Defined: `dirac_polos_zeros.py:706`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:711`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:714`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:739`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:742`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:781`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:784`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:810`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:813`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:849`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:852`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:900`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:903`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:937`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:940`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:967`

### visualize `def visualize(self, data, output_path)`
- Defined: `dirac_polos_zeros.py:970`

### __init__ `def __init__(self, checkpoint_path, config)`
- Defined: `dirac_polos_zeros.py:1071`

### _load_model `def _load_model(self)`
- Defined: `dirac_polos_zeros.py:1088`

### analyze `def analyze(self)`
- Defined: `dirac_polos_zeros.py:1104`

### _print_report `def _print_report(self, results)`
- Defined: `dirac_polos_zeros.py:1204`

### __init__ `def __init__(self, config)`
- Defined: `dirac_polos_zeros.py:1289`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `dirac_polos_zeros.py:1300`

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `dirac_polos_zeros.py:1357`

### generate_summary `def generate_summary(self, all_results, output_dir)`
- Defined: `dirac_polos_zeros.py:1385`

### _compute_aggregate_statistics `def _compute_aggregate_statistics(self, results)`
- Defined: `dirac_polos_zeros.py:1406`

### _generate_text_report `def _generate_text_report(self, summary, output_dir)`
- Defined: `dirac_polos_zeros.py:1473`

## experiments/ablation/ablation_study.py

### load_libraries `def load_libraries()`
- Defined: `experiments/ablation/ablation_study.py:54`
- Doc: Cargar bibliotecas con manejo de errores

### run_openblas `def run_openblas(libs, A, B, C, n)`
- Defined: `experiments/ablation/ablation_study.py:111`
- Doc: Ejecutar multiplicación con OpenBLAS

### run_strassen `def run_strassen(libs, name, func_name, A, B, C, n)`
- Defined: `experiments/ablation/ablation_study.py:122`
- Doc: Ejecutar multiplicación con Strassen

### benchmark_single `def benchmark_single(libs, algo_name, func_name, A, B, C, C_ref, n, n_runs, warmup)`
- Defined: `experiments/ablation/ablation_study.py:132`
- Doc: Benchmark una implementación

### run_ablation `def run_ablation(libs, sizes, n_runs, warmup)`
- Defined: `experiments/ablation/ablation_study.py:180`
- Doc: Ejecutar ablación completa

### analyze_results `def analyze_results(results)`
- Defined: `experiments/ablation/ablation_study.py:233`
- Doc: Analizar y presentar resultados

### main `def main()`
- Defined: `experiments/ablation/ablation_study.py:273`

### mean_time `def mean_time(self)`
- Defined: `experiments/ablation/ablation_study.py:35`

### std_time `def std_time(self)`
- Defined: `experiments/ablation/ablation_study.py:39`

### min_time `def min_time(self)`
- Defined: `experiments/ablation/ablation_study.py:43`

### max_time `def max_time(self)`
- Defined: `experiments/ablation/ablation_study.py:47`

### mean_gflops `def mean_gflops(self)`
- Defined: `experiments/ablation/ablation_study.py:51`

## experiments/apendix_experiments.py

### setup_matplotlib `def setup_matplotlib()`
- Defined: `experiments/apendix_experiments.py:34`

### generate_batch `def generate_batch(n, device)`
- Defined: `experiments/apendix_experiments.py:87`

### generate_test_set `def generate_test_set(n, device)`
- Defined: `experiments/apendix_experiments.py:93`

### compute_delta `def compute_delta(model)`
- Defined: `experiments/apendix_experiments.py:100`

### verify_strassen_structure `def verify_strassen_structure(U_disc, V_disc, W_disc, tolerance)`
- Defined: `experiments/apendix_experiments.py:117`

### compute_S_theta `def compute_S_theta(model)`
- Defined: `experiments/apendix_experiments.py:138`

### compute_gradient_covariance `def compute_gradient_covariance(model, batch_size, n_samples)`
- Defined: `experiments/apendix_experiments.py:154`

### train_with_logging `def train_with_logging(batch_size, total_epochs, lr, wd, symmetric_init, seed, log_interval)`
- Defined: `experiments/apendix_experiments.py:187`

### sparsify_and_discretize `def sparsify_and_discretize(model, batch_size)`
- Defined: `experiments/apendix_experiments.py:274`

### run_phase_diagram `def run_phase_diagram()`
- Defined: `experiments/apendix_experiments.py:327`

### run_batch_size_effect `def run_batch_size_effect()`
- Defined: `experiments/apendix_experiments.py:429`

### main `def main()`
- Defined: `experiments/apendix_experiments.py:526`

### __init__ `def __init__(self, rank, symmetric_init)`
- Defined: `experiments/apendix_experiments.py:53`

### forward `def forward(self, A, B)`
- Defined: `experiments/apendix_experiments.py:67`

### slot_importance `def slot_importance(self)`
- Defined: `experiments/apendix_experiments.py:77`

### count_active `def count_active(self, threshold)`
- Defined: `experiments/apendix_experiments.py:83`

## experiments/cache_analysis_v2.py

### cache_analysis `def cache_analysis()`
- Defined: `experiments/cache_analysis_v2.py:7`
- Doc: Full memory analysis for training.

## experiments/extended_experiments/all_test_extended.py

### main `def main()`
- Defined: `experiments/extended_experiments/all_test_extended.py:1403`

### __init__ `def __init__(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:89`

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/all_test_extended.py:96`

### begin `def begin(self, experiment_name)`
- Defined: `experiments/extended_experiments/all_test_extended.py:101`

### progress `def progress(self, current, total, metrics)`
- Defined: `experiments/extended_experiments/all_test_extended.py:109`

### checkpoint `def checkpoint(self, epoch, loss, accuracy)`
- Defined: `experiments/extended_experiments/all_test_extended.py:116`

### result `def result(self, name, value, context)`
- Defined: `experiments/extended_experiments/all_test_extended.py:121`

### verdict `def verdict(self, hypothesis, evidence, conclusion)`
- Defined: `experiments/extended_experiments/all_test_extended.py:126`

### failure `def failure(self, reason, details)`
- Defined: `experiments/extended_experiments/all_test_extended.py:131`

### complete `def complete(self, summary)`
- Defined: `experiments/extended_experiments/all_test_extended.py:136`

### claim `def claim(self, statement, confidence)`
- Defined: `experiments/extended_experiments/all_test_extended.py:144`

### note `def note(self, observation)`
- Defined: `experiments/extended_experiments/all_test_extended.py:148`

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/all_test_extended.py:153`

### capture `def capture(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:156`

### report `def report(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:174`

### __init__ `def __init__(self, size, modulus)`
- Defined: `experiments/extended_experiments/all_test_extended.py:202`

### _generate_data `def _generate_data(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:207`

### __len__ `def __len__(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:231`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `experiments/extended_experiments/all_test_extended.py:234`

### __init__ `def __init__(self, d_vocab, rank, scale)`
- Defined: `experiments/extended_experiments/all_test_extended.py:253`

### forward `def forward(self, x)`
- Defined: `experiments/extended_experiments/all_test_extended.py:265`

### get_weights `def get_weights(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:277`

### get_U_weights `def get_U_weights(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:283`

### get_V_weights `def get_V_weights(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:286`

### get_W_weights `def get_W_weights(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:289`

### name `def name(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:295`

### d_vocab `def d_vocab(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:299`

### generate_dataset `def generate_dataset(self, size)`
- Defined: `experiments/extended_experiments/all_test_extended.py:303`

### verify `def verify(self, model, x, y)`
- Defined: `experiments/extended_experiments/all_test_extended.py:307`

### __init__ `def __init__(self, modulus)`
- Defined: `experiments/extended_experiments/all_test_extended.py:312`

### name `def name(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:316`

### d_vocab `def d_vocab(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:319`

### generate_dataset `def generate_dataset(self, size)`
- Defined: `experiments/extended_experiments/all_test_extended.py:322`

### verify `def verify(self, model, x, y)`
- Defined: `experiments/extended_experiments/all_test_extended.py:344`

### __init__ `def __init__(self, size, bit_length)`
- Defined: `experiments/extended_experiments/all_test_extended.py:357`

### _generate_data `def _generate_data(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:362`

### __len__ `def __len__(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:368`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `experiments/extended_experiments/all_test_extended.py:371`

### __init__ `def __init__(self, bit_length, modulus)`
- Defined: `experiments/extended_experiments/all_test_extended.py:376`

### name `def name(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:381`

### d_vocab `def d_vocab(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:384`

### generate_dataset `def generate_dataset(self, size)`
- Defined: `experiments/extended_experiments/all_test_extended.py:387`

### verify `def verify(self, model, x, y)`
- Defined: `experiments/extended_experiments/all_test_extended.py:391`

### __init__ `def __init__(self, model)`
- Defined: `experiments/extended_experiments/all_test_extended.py:405`

### capture_gradients `def capture_gradients(self, dataloader, n_batches)`
- Defined: `experiments/extended_experiments/all_test_extended.py:409`

### compute_covariance `def compute_covariance(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:429`

### compute_condition_number `def compute_condition_number(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:436`

### compute_gradient_noise_scale `def compute_gradient_noise_scale(self, batch_size, learning_rate)`
- Defined: `experiments/extended_experiments/all_test_extended.py:442`

### analyze `def analyze(self, dataloader, batch_size, learning_rate)`
- Defined: `experiments/extended_experiments/all_test_extended.py:450`

### __init__ `def __init__(self, model, target_kappa)`
- Defined: `experiments/extended_experiments/all_test_extended.py:468`

### spectral_regularizer `def spectral_regularizer(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:472`

### __init__ `def __init__(self, model)`
- Defined: `experiments/extended_experiments/all_test_extended.py:491`

### count_local_minima `def count_local_minima(self, directions, losses)`
- Defined: `experiments/extended_experiments/all_test_extended.py:494`

### measure_basin_width `def measure_basin_width(self, weights, direction, n_points)`
- Defined: `experiments/extended_experiments/all_test_extended.py:501`

### classify_failure_mode `def classify_failure_mode(self, final_weights, initial_weights)`
- Defined: `experiments/extended_experiments/all_test_extended.py:505`

### __init__ `def __init__(self, success_radius)`
- Defined: `experiments/extended_experiments/all_test_extended.py:513`

### estimate_volume_monte_carlo `def estimate_volume_monte_carlo(self, model_class, n_samples, success_checker)`
- Defined: `experiments/extended_experiments/all_test_extended.py:516`

### compute_fractal_dimension `def compute_fractal_dimension(self, trajectory)`
- Defined: `experiments/extended_experiments/all_test_extended.py:520`

### __init__ `def __init__(self, model)`
- Defined: `experiments/extended_experiments/all_test_extended.py:527`

### add_gaussian_noise `def add_gaussian_noise(self, sigma)`
- Defined: `experiments/extended_experiments/all_test_extended.py:530`

### fgsm_attack `def fgsm_attack(self, x, y, epsilon)`
- Defined: `experiments/extended_experiments/all_test_extended.py:536`

### quantize_weights `def quantize_weights(self, bits)`
- Defined: `experiments/extended_experiments/all_test_extended.py:544`

### test_discretization_with_noise `def test_discretization_with_noise(self, sigma, checker)`
- Defined: `experiments/extended_experiments/all_test_extended.py:550`

### run_fragility_analysis `def run_fragility_analysis(self, sigma_values, checker)`
- Defined: `experiments/extended_experiments/all_test_extended.py:560`

### __init__ `def __init__(self, config, experiment_name)`
- Defined: `experiments/extended_experiments/all_test_extended.py:584`

### save_checkpoint `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- Defined: `experiments/extended_experiments/all_test_extended.py:593`

### load_checkpoint `def load_checkpoint(self, path, model, optimizer)`
- Defined: `experiments/extended_experiments/all_test_extended.py:619`

### __init__ `def __init__(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:630`

### update `def update(self, train_loss, train_acc, test_loss, test_acc, kappa, grad_norm, weight_norm, disc_margin)`
- Defined: `experiments/extended_experiments/all_test_extended.py:642`

### detect_grokking `def detect_grokking(self, loss_threshold, test_loss_threshold, min_duration)`
- Defined: `experiments/extended_experiments/all_test_extended.py:660`

### progress_bar_string `def progress_bar_string(self, epoch, total_epochs)`
- Defined: `experiments/extended_experiments/all_test_extended.py:679`

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/all_test_extended.py:703`

### train_epoch `def train_epoch(self, model, dataloader, optimizer)`
- Defined: `experiments/extended_experiments/all_test_extended.py:708`

### evaluate `def evaluate(self, model, dataloader)`
- Defined: `experiments/extended_experiments/all_test_extended.py:730`

### run_training `def run_training(self, model, train_loader, test_loader, experiment_name, epochs, batch_size, lr, wd, verbose)`
- Defined: `experiments/extended_experiments/all_test_extended.py:748`

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/all_test_extended.py:812`

### compute_discretization_margin `def compute_discretization_margin(self, model)`
- Defined: `experiments/extended_experiments/all_test_extended.py:815`

### discretize_weights `def discretize_weights(self, model)`
- Defined: `experiments/extended_experiments/all_test_extended.py:827`

### check_strassen_structure `def check_strassen_structure(self, model, modulus)`
- Defined: `experiments/extended_experiments/all_test_extended.py:832`

### count_discretized_parameters `def count_discretized_parameters(self, model)`
- Defined: `experiments/extended_experiments/all_test_extended.py:839`

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/all_test_extended.py:854`

### verify_expansion `def verify_expansion(self, model, task, sizes)`
- Defined: `experiments/extended_experiments/all_test_extended.py:857`

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/all_test_extended.py:882`

### experiment_batch_size_mechanism `def experiment_batch_size_mechanism(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:891`
- Doc: Experiment 1: Why batch size [24,128] works.

### experiment_kappa_intervention `def experiment_kappa_intervention(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:962`
- Doc: Experiment 2: Active intervention on κ.

### experiment_failure_analysis `def experiment_failure_analysis(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:1026`
- Doc: Experiment 3: Why 32% of runs fail.

### experiment_generalization `def experiment_generalization(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:1088`
- Doc: Experiment 4: Generalization to other tasks.

### experiment_basin_volume `def experiment_basin_volume(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:1145`
- Doc: Experiment 5: Basin volume estimation.

### experiment_hardware_reproducibility `def experiment_hardware_reproducibility(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:1158`
- Doc: Experiment 6: Hardware reproducibility testing.

### experiment_fragility `def experiment_fragility(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:1217`
- Doc: Experiment 7: Discretization fragility testing.

### run_all_experiments `def run_all_experiments(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:1332`

### _save_results `def _save_results(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:1371`

### _generate_summary `def _generate_summary(self)`
- Defined: `experiments/extended_experiments/all_test_extended.py:1379`

### checker `def checker()`
- Defined: `experiments/extended_experiments/all_test_extended.py:1253`

## experiments/extended_experiments/exp1_covariance_spectrometry.py

### setup_matplotlib `def setup_matplotlib()`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:22`

### generate_batch `def generate_batch(n, scale)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:112`
- Doc: Generate batch of matrices.

### compute_gradient_covariance `def compute_gradient_covariance(model, batch_size, n_samples)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:120`
- Doc: Compute the gradient covariance matrix Σₜ and its eigenvalues.

### load_checkpoint `def load_checkpoint(checkpoint_path)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:196`
- Doc: Load model from checkpoint file.

### analyze_checkpoint `def analyze_checkpoint(checkpoint_path, batch_sizes, n_samples, n_runs)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:222`
- Doc: Analyze a single checkpoint with multiple batch sizes.

### main `def main()`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:277`
- Doc: Main execution for Experiment 1.

### generate_visualization `def generate_visualization(results, output_dir)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:380`
- Doc: Generate publication-quality figures.

### __init__ `def __init__(self, rank)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:54`

### forward `def forward(self, A, B)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:61`

### get_all_parameters `def get_all_parameters(self)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:71`
- Doc: Get all parameters as a single flattened vector.

### compute_per_sample_gradients `def compute_per_sample_gradients(self, A, B, C_true)`
- Defined: `experiments/extended_experiments/exp1_covariance_spectrometry.py:78`
- Doc: Compute per-sample gradients for covariance estimation.

## experiments/extended_experiments/exp2_noise_ablation.py

### setup_matplotlib `def setup_matplotlib()`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:25`

### generate_batch `def generate_batch(n, scale)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:99`
- Doc: Generate batch of matrices.

### compute_gradient_covariance_matrix `def compute_gradient_covariance_matrix(model, n_samples, batch_size)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:107`
- Doc: Compute the gradient covariance matrix Σ.

### get_eigenbasis `def get_eigenbasis(covariance)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:139`
- Doc: Get eigenvectors and eigenvalues of covariance matrix.

### load_checkpoint `def load_checkpoint(checkpoint_path)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:150`
- Doc: Load model from checkpoint file.

### experiment_treatment_a_gradient_noise `def experiment_treatment_a_gradient_noise(model, noise_std, n_test)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:172`
- Doc: Treatment A: Add noise to gradients DURING forward/backward pass.

### experiment_treatment_b_weight_noise `def experiment_treatment_b_weight_noise(model, noise_std, n_test)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:212`
- Doc: Treatment B: Noise on weights BEFORE evaluation (already done in paper).

### experiment_treatment_c_structured_noise `def experiment_treatment_c_structured_noise(model, covariance, noise_std, n_test)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:246`
- Doc: Treatment C: Structured noise by eigenvectors of Σ.

### run_noise_ablation `def run_noise_ablation(checkpoint_path, noise_levels)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:315`
- Doc: Run complete noise ablation experiment on a checkpoint.

### main `def main()`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:348`
- Doc: Main execution for Experiment 2.

### generate_visualization `def generate_visualization(results, output_dir)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:433`
- Doc: Generate publication-quality figures.

### __init__ `def __init__(self, rank)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:54`

### forward `def forward(self, A, B)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:61`

### get_all_parameters `def get_all_parameters(self)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:71`

### set_parameters `def set_parameters(self, new_params)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:77`
- Doc: Set parameters from a flattened tensor.

### compute_loss `def compute_loss(self, A, B)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:85`
- Doc: Compute MSE loss.

### compute_accuracy `def compute_accuracy(self, A, B, threshold)`
- Defined: `experiments/extended_experiments/exp2_noise_ablation.py:91`
- Doc: Compute accuracy (proportion of predictions within threshold).

## experiments/extended_experiments/exp3_prospective_prediction.py

### setup_matplotlib `def setup_matplotlib()`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:32`

### generate_batch `def generate_batch(n, scale)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:114`
- Doc: Generate batch of matrices.

### compute_kappa `def compute_kappa(model, n_samples, batch_size)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:121`
- Doc: Compute condition number κ(Σ) of gradient covariance matrix.

### load_checkpoint `def load_checkpoint(checkpoint_path)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:166`
- Doc: Load model from checkpoint file.

### simulate_early_prediction `def simulate_early_prediction(checkpoint_path, early_epoch_fraction)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:188`
- Doc: Simulate the prospective prediction experiment.

### run_prospective_prediction_experiment `def run_prospective_prediction_experiment(checkpoint_files)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:236`
- Doc: Run the full prospective prediction experiment across all checkpoints.

### compute_roc_analysis `def compute_roc_analysis(predictions)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:252`
- Doc: Compute ROC curve and AUC for κ as predictor of success.

### main `def main()`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:313`
- Doc: Main execution for Experiment 3.

### generate_visualization `def generate_visualization(results, predictions, output_dir)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:410`
- Doc: Generate publication-quality figures.

### __init__ `def __init__(self, rank)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:59`

### forward `def forward(self, A, B)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:66`

### get_all_parameters `def get_all_parameters(self)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:76`

### set_parameters `def set_parameters(self, new_params)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:82`

### count_active_slots `def count_active_slots(self, threshold)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:89`
- Doc: Count active slots based on weight norms.

### compute_discretization_margin `def compute_discretization_margin(self)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:97`
- Doc: Compute how close weights are to discrete values {-1, 0, 1}.

### is_grokked `def is_grokked(self, margin_threshold, active_slots_target)`
- Defined: `experiments/extended_experiments/exp3_prospective_prediction.py:107`
- Doc: Check if model has grokked (discretized with low error).

## experiments/extended_experiments/exp4_trajectory_perturbation.py

### setup_matplotlib `def setup_matplotlib()`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:27`

### generate_batch `def generate_batch(n, scale)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:117`
- Doc: Generate batch of matrices.

### load_checkpoint `def load_checkpoint(checkpoint_path)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:124`
- Doc: Load model from checkpoint file.

### simulate_trajectory_perturbation `def simulate_trajectory_perturbation(checkpoint_path, perturbations)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:146`
- Doc: Simulate trajectory perturbation effects using available checkpoints.

### main `def main()`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:277`
- Doc: Main execution for Experiment 4.

### generate_visualization `def generate_visualization(results, output_dir)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:398`
- Doc: Generate publication-quality figures.

### __init__ `def __init__(self, rank)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:54`

### forward `def forward(self, A, B)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:61`

### get_all_parameters `def get_all_parameters(self)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:71`

### set_parameters `def set_parameters(self, new_params)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:77`

### get_weight_norm `def get_weight_norm(self)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:84`
- Doc: Get total L2 norm of all parameters.

### get_weight_direction `def get_weight_direction(self)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:91`
- Doc: Get normalized weight vector direction.

### compute_gradient_norm `def compute_gradient_norm(self, A, B)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:96`
- Doc: Compute norm of gradients.

### cosine_similarity `def cosine_similarity(self, other_params)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:110`
- Doc: Compute cosine similarity between current weights and target weights.

### compute_metrics `def compute_metrics(model, name)`
- Defined: `experiments/extended_experiments/exp4_trajectory_perturbation.py:194`
- Doc: Compute evaluation metrics.

## experiments/extended_experiments/run_all_experiments.py

### setup_matplotlib `def setup_matplotlib()`
- Defined: `experiments/extended_experiments/run_all_experiments.py:24`

### generate_batch `def generate_batch(n, scale)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:95`

### load_checkpoint_robust `def load_checkpoint_robust(checkpoint_path, model)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:101`
- Doc: Load checkpoint with multiple format fallback strategies.

### compute_gradient_covariance_safe `def compute_gradient_covariance_safe(model, batch_size, n_samples)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:135`
- Doc: Compute κ(Σₜ) with numerical safety.

### run_all_experiments `def run_all_experiments()`
- Defined: `experiments/extended_experiments/run_all_experiments.py:205`
- Doc: Run all experiments.

### generate_summary_visualization `def generate_summary_visualization(results, output_dir)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:496`
- Doc: Generate summary visualization.

### __init__ `def __init__(self, rank)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:51`

### forward `def forward(self, A, B)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:58`

### get_all_parameters `def get_all_parameters(self)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:68`

### set_parameters `def set_parameters(self, new_params)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:74`

### count_active_slots `def count_active_slots(self, threshold)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:81`

### compute_discretization_margin `def compute_discretization_margin(self)`
- Defined: `experiments/extended_experiments/run_all_experiments.py:88`

### compute_accuracy `def compute_accuracy()`
- Defined: `experiments/extended_experiments/run_all_experiments.py:324`

## experiments/extended_experiments/validate2.py

### find_grokked_checkpoint `def find_grokked_checkpoint()`
- Defined: `experiments/extended_experiments/validate2.py:2313`
- Doc: Buscar checkpoint grokkeado en múltiples ubicaciones.

### analyze_checkpoints `def analyze_checkpoints()`
- Defined: `experiments/extended_experiments/validate2.py:2353`
- Doc: Analizar todos los checkpoints disponibles para encontrar el grokkeado.

### main `def main()`
- Defined: `experiments/extended_experiments/validate2.py:2423`
- Doc: Punto de entrada principal.

### __post_init__ `def __post_init__(self)`
- Defined: `experiments/extended_experiments/validate2.py:115`

### __init__ `def __init__(self, rank)`
- Defined: `experiments/extended_experiments/validate2.py:139`

### _initialize_weights `def _initialize_weights(self)`
- Defined: `experiments/extended_experiments/validate2.py:150`
- Doc: Inicializar pesos con consideración para grokking.

### forward `def forward(self, A, B)`
- Defined: `experiments/extended_experiments/validate2.py:156`
- Doc: Computar A @ B usando descomposición tensorial.

### slot_importance `def slot_importance(self)`
- Defined: `experiments/extended_experiments/validate2.py:183`
- Doc: Importancia de cada slot basada en normas.

### count_active `def count_active(self, threshold)`
- Defined: `experiments/extended_experiments/validate2.py:190`
- Doc: Contar slots activos.

### compute_SP `def compute_SP(self)`
- Defined: `experiments/extended_experiments/validate2.py:194`
- Doc: Métrica de Sparsity. SP -> 0 significa máxima sparsity.

### get_state_dict `def get_state_dict(self)`
- Defined: `experiments/extended_experiments/validate2.py:202`
- Doc: Obtener estado completo para checkpointing.

### load_state_dict `def load_state_dict(self, state_dict)`
- Defined: `experiments/extended_experiments/validate2.py:211`
- Doc: Cargar estado completo desde checkpoint.

### __init__ `def __init__(self, num_samples, matrix_size, seed)`
- Defined: `experiments/extended_experiments/validate2.py:228`

### generate_matrix `def generate_matrix(self)`
- Defined: `experiments/extended_experiments/validate2.py:238`
- Doc: Generar matriz aleatoria con valores enteros.

### generate_data `def generate_data(self)`
- Defined: `experiments/extended_experiments/validate2.py:242`
- Doc: Generar pares de matrices y sus productos.

### get_train_test `def get_train_test(self, test_ratio)`
- Defined: `experiments/extended_experiments/validate2.py:260`
- Doc: Dividir en conjuntos de entrenamiento y prueba.

### __init__ `def __init__(self, model, config)`
- Defined: `experiments/extended_experiments/validate2.py:287`

### compute_lc `def compute_lc(self, batch_inputs, batch_targets)`
- Defined: `experiments/extended_experiments/validate2.py:292`
- Doc: Calcular LC para un batch específico.

### compute_batch_diversity `def compute_batch_diversity(self, batch_inputs)`
- Defined: `experiments/extended_experiments/validate2.py:326`
- Doc: Calcular diversidad del batch basada en varianza de activaciones.

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/validate2.py:343`

### verify `def verify(self, model, n_test)`
- Defined: `experiments/extended_experiments/validate2.py:347`
- Doc: Verificar que el operador ha grokkeado correctamente.

### _generate_batch `def _generate_batch(self, n, scale)`
- Defined: `experiments/extended_experiments/validate2.py:393`
- Doc: Generar batch de matrices aleatorias.

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/validate2.py:420`

### get_weight_magnitudes `def get_weight_magnitudes(self, model)`
- Defined: `experiments/extended_experiments/validate2.py:426`
- Doc: Obtener magnitud absoluta de todos los pesos.

### compute_sparsity `def compute_sparsity(self, model)`
- Defined: `experiments/extended_experiments/validate2.py:431`
- Doc: Calcular porcentaje de pesos en cero.

### prune_percent `def prune_percent(self, model, percent)`
- Defined: `experiments/extended_experiments/validate2.py:437`
- Doc: Podar el porcentaje especificado de pesos menos importantes.

### fine_tune `def fine_tune(self, model, train_data)`
- Defined: `experiments/extended_experiments/validate2.py:457`
- Doc: Fine-tune del modelo podado con métricas completas.

### _generate_batch `def _generate_batch(self, n, scale)`
- Defined: `experiments/extended_experiments/validate2.py:510`
- Doc: Generar batch de matrices aleatorias.

### run_protocol `def run_protocol(self, model, train_data)`
- Defined: `experiments/extended_experiments/validate2.py:624`
- Doc: Ejecutar protocolo completo de poda iterativa.

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/validate2.py:775`

### run_full_experiment `def run_full_experiment(self, target_epochs)`
- Defined: `experiments/extended_experiments/validate2.py:779`
- Doc: Ejecutar experimento completo de LC entrenando desde cero.

### _generate_batch `def _generate_batch(self, n, scale)`
- Defined: `experiments/extended_experiments/validate2.py:882`

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/validate2.py:903`

### run_balanced_experiments `def run_balanced_experiments(self, n_runs)`
- Defined: `experiments/extended_experiments/validate2.py:907`
- Doc: Ejecutar multiples runs con condiciones disenhadas para producir mix.

### _train_single_run `def _train_single_run(self, run_idx, config)`
- Defined: `experiments/extended_experiments/validate2.py:1021`
- Doc: Entrenar un solo modelo con configuración específica.

### _compute_roc `def _compute_roc(self, y_true, y_scores)`
- Defined: `experiments/extended_experiments/validate2.py:1099`
- Doc: Calcular ROC/AUC básico.

### _generate_batch `def _generate_batch(self, n, scale)`
- Defined: `experiments/extended_experiments/validate2.py:1127`

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/validate2.py:1145`

### compute_roc_with_ci `def compute_roc_with_ci(self, y_true, y_scores)`
- Defined: `experiments/extended_experiments/validate2.py:1149`
- Doc: Calcular curva ROC con intervalos de confianza bootstrap.

### compute_kappa_with_ci `def compute_kappa_with_ci(self, y_true, y_pred)`
- Defined: `experiments/extended_experiments/validate2.py:1236`
- Doc: Calcular Kappa de Cohen con IC bootstrap.

### compute_accuracy_with_ci `def compute_accuracy_with_ci(self, correct)`
- Defined: `experiments/extended_experiments/validate2.py:1260`
- Doc: Calcular accuracy con IC binomial.

### __init__ `def __init__(self, style)`
- Defined: `experiments/extended_experiments/validate2.py:1284`

### plot_local_complexity `def plot_local_complexity(self, epochs, lc_values, accuracy, save_path)`
- Defined: `experiments/extended_experiments/validate2.py:1298`
- Doc: Graficar evolución de Local Complexity y Accuracy.

### plot_pruning_results `def plot_pruning_results(self, pruning_data, save_path)`
- Defined: `experiments/extended_experiments/validate2.py:1342`
- Doc: Graficar resultados de poda iterativa.

### plot_roc_with_ci `def plot_roc_with_ci(self, roc_data, save_path)`
- Defined: `experiments/extended_experiments/validate2.py:1404`
- Doc: Graficar curva ROC con intervalos de confianza.

### plot_balanced_runs_results `def plot_balanced_runs_results(self, balanced_data, save_path)`
- Defined: `experiments/extended_experiments/validate2.py:1467`
- Doc: Graficar resultados del experimento de runs balanceados.

### plot_discretization_results `def plot_discretization_results(self, pruning_data, save_path)`
- Defined: `experiments/extended_experiments/validate2.py:1556`
- Doc: Graficar resultados de discretizacion.

### __init__ `def __init__(self, config)`
- Defined: `experiments/extended_experiments/validate2.py:1659`

### find_grokked_checkpoint `def find_grokked_checkpoint(self)`
- Defined: `experiments/extended_experiments/validate2.py:1689`
- Doc: Buscar checkpoint grokkeado en múltiples ubicaciones.

### load_grokked_checkpoint `def load_grokked_checkpoint(self, checkpoint_path)`
- Defined: `experiments/extended_experiments/validate2.py:1733`
- Doc: Cargar checkpoint grokkeado y verificar que grokkeó.

### verify_checkpoint_is_grokked `def verify_checkpoint_is_grokked(self)`
- Defined: `experiments/extended_experiments/validate2.py:1768`
- Doc: Verificar que el checkpoint cargado realmente grokkeó.

### run_local_complexity_experiment `def run_local_complexity_experiment(self, epochs)`
- Defined: `experiments/extended_experiments/validate2.py:1795`
- Doc: Ejecutar experimento de Local Complexity vs Época.

### run_lc_training_experiment `def run_lc_training_experiment(self, epochs)`
- Defined: `experiments/extended_experiments/validate2.py:1897`
- Doc: Ejecutar experimento de Local Complexity .

### run_pruning_experiment `def run_pruning_experiment(self)`
- Defined: `experiments/extended_experiments/validate2.py:1936`
- Doc: Ejecutar protocolo de poda iterativa + fine-tuning.

### run_balanced_runs_experiment `def run_balanced_runs_experiment(self, n_runs)`
- Defined: `experiments/extended_experiments/validate2.py:1980`
- Doc: Ejecutar experimento de runs balanceados (PUNTO C DEL REVISOR).

### run_roc_analysis `def run_roc_analysis(self)`
- Defined: `experiments/extended_experiments/validate2.py:2043`
- Doc: Ejecutar análisis ROC/AUC con bootstrap.

### _generate_batch `def _generate_batch(self, n, scale)`
- Defined: `experiments/extended_experiments/validate2.py:2117`
- Doc: Generar batch de matrices aleatorias.

### generate_summary_report `def generate_summary_report(self)`
- Defined: `experiments/extended_experiments/validate2.py:2124`
- Doc: Generar reporte de resumen en markdown.

### save_results `def save_results(self)`
- Defined: `experiments/extended_experiments/validate2.py:2197`
- Doc: Guardar todos los resultados.

### run_all_experiments `def run_all_experiments(self, checkpoint_path)`
- Defined: `experiments/extended_experiments/validate2.py:2235`
- Doc: Ejecutar suite completa de experimentos.

## experiments/generate_figures.py

### setup_matplotlib_for_plotting `def setup_matplotlib_for_plotting()`
- Defined: `experiments/generate_figures.py:15`
- Doc: Configure matplotlib and seaborn for proper rendering.

### generate_benchmark_figure `def generate_benchmark_figure()`
- Defined: `experiments/generate_figures.py:63`
- Doc: Generate benchmark performance comparison plot.

### generate_ablation_figure `def generate_ablation_figure()`
- Defined: `experiments/generate_figures.py:124`
- Doc: Generate ablation study visualization.

### load_checkpoint_weights `def load_checkpoint_weights()`
- Defined: `experiments/generate_figures.py:219`
- Doc: Load all checkpoint files and extract weight tensors.

### generate_weight_geometry_figure `def generate_weight_geometry_figure()`
- Defined: `experiments/generate_figures.py:258`
- Doc: Generate weight space geometry visualization.

### generate_phase_transition_figure `def generate_phase_transition_figure()`
- Defined: `experiments/generate_figures.py:354`
- Doc: Generate phase transition analysis from checkpoint evolution.

### generate_coherence_figure `def generate_coherence_figure()`
- Defined: `experiments/generate_figures.py:465`
- Doc: Generate cache coherence analysis visualization.

### generate_crystallization_figure `def generate_crystallization_figure()`
- Defined: `experiments/generate_figures.py:534`
- Doc: Visualize the crystallization of Strassen coefficients.

### main `def main()`
- Defined: `experiments/generate_figures.py:623`

## experiments/statistics/coherence_analysis.py

### strassen_numpy `def strassen_numpy(A, B, threshold)`
- Defined: `experiments/statistics/coherence_analysis.py:15`

### run_coherence_analysis `def run_coherence_analysis()`
- Defined: `experiments/statistics/coherence_analysis.py:42`

## experiments/statistics/rigorous_experiment.py

### generate_data `def generate_data(n_samples, seed)`
- Defined: `experiments/statistics/rigorous_experiment.py:131`
- Doc: Generate matrix multiplication dataset

### compute_discretization_error `def compute_discretization_error(model, values)`
- Defined: `experiments/statistics/rigorous_experiment.py:143`
- Doc: Compute mean distance to nearest discrete value

### compute_spectral_gap `def compute_spectral_gap(model)`
- Defined: `experiments/statistics/rigorous_experiment.py:157`
- Doc: Compute maximum spectral gap ratio

### run_single_experiment `def run_single_experiment(batch_size, seed, run_id, config)`
- Defined: `experiments/statistics/rigorous_experiment.py:169`
- Doc: Run a single controlled experiment

### run_full_experiment `def run_full_experiment(batch_sizes, n_seeds, n_runs_per_seed)`
- Defined: `experiments/statistics/rigorous_experiment.py:269`
- Doc: Run complete factorial experiment

### perform_anova `def perform_anova(results)`
- Defined: `experiments/statistics/rigorous_experiment.py:306`
- Doc: Perform full factorial ANOVA

### print_anova_table `def print_anova_table(anova)`
- Defined: `experiments/statistics/rigorous_experiment.py:401`
- Doc: Print formatted ANOVA table

### fit_noise_model `def fit_noise_model(results)`
- Defined: `experiments/statistics/rigorous_experiment.py:448`
- Doc: Fit theoretical noise model:

### find_optimal_B `def find_optimal_B(results, n_bootstrap)`
- Defined: `experiments/statistics/rigorous_experiment.py:519`
- Doc: Find optimal batch size with bootstrap confidence interval

### generate_report `def generate_report(results, config)`
- Defined: `experiments/statistics/rigorous_experiment.py:555`
- Doc: Generate complete statistical report

### __init__ `def __init__(self, config)`
- Defined: `experiments/statistics/rigorous_experiment.py:108`

### forward `def forward(self, x)`
- Defined: `experiments/statistics/rigorous_experiment.py:124`

### cache_miss_proxy `def cache_miss_proxy(B)`
- Defined: `experiments/statistics/rigorous_experiment.py:467`

### full_model `def full_model(B, alpha, beta, gamma)`
- Defined: `experiments/statistics/rigorous_experiment.py:472`

### null_model `def null_model(B, alpha, gamma)`
- Defined: `experiments/statistics/rigorous_experiment.py:476`

### get_mean_error `def get_mean_error(data, B)`
- Defined: `experiments/statistics/rigorous_experiment.py:526`

## experiments/validation/benchmark.py

### strassen_numpy `def strassen_numpy(A, B, threshold)`
- Defined: `experiments/validation/benchmark.py:15`
- Doc: Strassen recursivo con NumPy para productos base.

### measure_single_sgemm `def measure_single_sgemm(n, threads)`
- Defined: `experiments/validation/benchmark.py:49`
- Doc: Mide tiempo de un solo sgemm de tamaño n.

### run_planck_analysis `def run_planck_analysis()`
- Defined: `experiments/validation/benchmark.py:59`
- Doc: Ejecuta el análisis del Límite de Planck.

## experiments/validation_experiments.py

### strassen_2x2 `def strassen_2x2(A, B, U, V, W)`
- Defined: `experiments/validation_experiments.py:47`
- Doc: Compute 2x2 matrix multiplication using Strassen coefficients.

### strassen_recursive `def strassen_recursive(A, B, U, V, W, threshold)`
- Defined: `experiments/validation_experiments.py:56`
- Doc: Recursive Strassen for NxN matrices.

### test_uniqueness_via_permutation `def test_uniqueness_via_permutation()`
- Defined: `experiments/validation_experiments.py:85`
- Doc: Test that permuting slots produces equivalent computation.

### test_noise_stability `def test_noise_stability()`
- Defined: `experiments/validation_experiments.py:125`
- Doc: Test stability under Gaussian noise.

### test_expansion_sizes `def test_expansion_sizes()`
- Defined: `experiments/validation_experiments.py:162`
- Doc: Test expansion to larger sizes.

### simulate_grokking_dynamics `def simulate_grokking_dynamics()`
- Defined: `experiments/validation_experiments.py:184`
- Doc: Simulate grokking dynamics for visualization.

### compute_cache_math `def compute_cache_math()`
- Defined: `experiments/validation_experiments.py:250`
- Doc: Compute L3 cache requirements for different batch sizes.

### main `def main()`
- Defined: `experiments/validation_experiments.py:294`
- Doc: Run all validation experiments.

### convert_types `def convert_types(obj)`
- Defined: `experiments/validation_experiments.py:313`

## experiments/verify_checkpoints.py

### compute_delta `def compute_delta(model)`
- Defined: `experiments/verify_checkpoints.py:51`

### verify_2x2 `def verify_2x2(U, V, W, n_test)`
- Defined: `experiments/verify_checkpoints.py:68`

### strassen_expand `def strassen_expand(A, B, U, V, W)`
- Defined: `experiments/verify_checkpoints.py:89`

### verify_expansion `def verify_expansion(U, V, W, sizes)`
- Defined: `experiments/verify_checkpoints.py:126`

### compute_S_theta `def compute_S_theta(model)`
- Defined: `experiments/verify_checkpoints.py:152`

### load_checkpoint `def load_checkpoint(path)`
- Defined: `experiments/verify_checkpoints.py:166`

### verify_checkpoint `def verify_checkpoint(checkpoint_path)`
- Defined: `experiments/verify_checkpoints.py:183`

### run_noise_stability_test `def run_noise_stability_test(checkpoint_path, noise_levels)`
- Defined: `experiments/verify_checkpoints.py:226`

### main `def main()`
- Defined: `experiments/verify_checkpoints.py:249`

### __init__ `def __init__(self, rank)`
- Defined: `experiments/verify_checkpoints.py:27`

### forward `def forward(self, A, B)`
- Defined: `experiments/verify_checkpoints.py:34`

### get_discrete_coefficients `def get_discrete_coefficients(self)`
- Defined: `experiments/verify_checkpoints.py:44`

## experimetn2.py

### main `def main()`
- Defined: `experimetn2.py:745`

### __post_init__ `def __post_init__(self)`
- Defined: `experimetn2.py:63`

### __init__ `def __init__(self, config)`
- Defined: `experimetn2.py:101`

### forward `def forward(self, A, B)`
- Defined: `experimetn2.py:115`

### get_coefficients `def get_coefficients(self)`
- Defined: `experimetn2.py:125`

### slot_importance `def slot_importance(self)`
- Defined: `experimetn2.py:128`

### __init__ `def __init__(self, config, gamma)`
- Defined: `experimetn2.py:140`

### get_complex_tensors `def get_complex_tensors(self)`
- Defined: `experimetn2.py:153`

### forward `def forward(self, A, B)`
- Defined: `experimetn2.py:160`

### __init__ `def __init__(self, config)`
- Defined: `experimetn2.py:180`

### generate_batch `def generate_batch(self, batch_size)`
- Defined: `experimetn2.py:183`

### save `def save(self, model, epoch, metrics, path)`
- Defined: `experimetn2.py:192`

### __init__ `def __init__(self, tolerance)`
- Defined: `experimetn2.py:209`

### calculate_r_ratio `def calculate_r_ratio(self, eigenvalues)`
- Defined: `experimetn2.py:212`

### __init__ `def __init__(self, config)`
- Defined: `experimetn2.py:246`

### compute_hessian `def compute_hessian(self, model, A, B, C_true)`
- Defined: `experimetn2.py:249`

### __init__ `def __init__(self, noise_floor)`
- Defined: `experimetn2.py:280`

### calculate `def calculate(self, model, current_loss)`
- Defined: `experimetn2.py:283`

### __init__ `def __init__(self, config)`
- Defined: `experimetn2.py:330`

### calculate `def calculate(self, model, datagen)`
- Defined: `experimetn2.py:333`

### run `def run(self, model)`
- Defined: `experimetn2.py:388`

### get_name `def get_name(self)`
- Defined: `experimetn2.py:390`

### __init__ `def __init__(self, suite_config, datagen)`
- Defined: `experimetn2.py:399`

### get_name `def get_name(self)`
- Defined: `experimetn2.py:405`

### run `def run(self, model)`
- Defined: `experimetn2.py:408`

### __init__ `def __init__(self, suite_config, datagen)`
- Defined: `experimetn2.py:463`

### get_name `def get_name(self)`
- Defined: `experimetn2.py:468`

### run `def run(self, model)`
- Defined: `experimetn2.py:471`

### __init__ `def __init__(self, suite_config, datagen)`
- Defined: `experimetn2.py:527`

### get_name `def get_name(self)`
- Defined: `experimetn2.py:531`

### run `def run(self, model)`
- Defined: `experimetn2.py:534`

### __init__ `def __init__(self, suite_config, datagen)`
- Defined: `experimetn2.py:576`

### get_name `def get_name(self)`
- Defined: `experimetn2.py:582`

### run `def run(self, model)`
- Defined: `experimetn2.py:585`

### __init__ `def __init__(self, suite_config, datagen)`
- Defined: `experimetn2.py:630`

### get_name `def get_name(self)`
- Defined: `experimetn2.py:634`

### run `def run(self, model)`
- Defined: `experimetn2.py:637`

### __init__ `def __init__(self, config)`
- Defined: `experimetn2.py:701`

### execute_all `def execute_all(self)`
- Defined: `experimetn2.py:712`

### loss_fn `def loss_fn(flat_param_tensor)`
- Defined: `experimetn2.py:254`

## fermi.py

### main `def main()`
- Defined: `fermi.py:766`

### get_coefficients `def get_coefficients(self)`
- Defined: `fermi.py:50`

### construct `def construct(self, weights, k)`
- Defined: `fermi.py:55`

### calculate `def calculate(self, model)`
- Defined: `fermi.py:60`

### calculate `def calculate(self, eigenvalues, num_electrons)`
- Defined: `fermi.py:65`

### calculate `def calculate(self, eigenvalues, energies)`
- Defined: `fermi.py:70`

### calculate `def calculate(self, eigenvalues, eigenvectors, fermi_level)`
- Defined: `fermi.py:75`

### classify `def classify(self, band_gap, dos_at_fermi)`
- Defined: `fermi.py:80`

### __init__ `def __init__(self, hidden_dim, matrix_size)`
- Defined: `fermi.py:84`

### _initialize `def _initialize(self)`
- Defined: `fermi.py:95`

### forward `def forward(self, a, b)`
- Defined: `fermi.py:100`

### get_coefficients `def get_coefficients(self)`
- Defined: `fermi.py:103`

### __init__ `def __init__(self, config)`
- Defined: `fermi.py:112`

### construct `def construct(self, weights, k)`
- Defined: `fermi.py:115`

### __init__ `def __init__(self, config)`
- Defined: `fermi.py:136`

### calculate `def calculate(self, model)`
- Defined: `fermi.py:140`

### _calculate_band_gap `def _calculate_band_gap(self, band_structure)`
- Defined: `fermi.py:171`

### _calculate_effective_masses `def _calculate_effective_masses(self, k_points, band_structure, valence_idx, conduction_idx)`
- Defined: `fermi.py:188`

### _is_direct_gap `def _is_direct_gap(self, band_structure, valence_idx, conduction_idx)`
- Defined: `fermi.py:208`

### __init__ `def __init__(self, config)`
- Defined: `fermi.py:216`

### calculate `def calculate(self, eigenvalues, num_electrons)`
- Defined: `fermi.py:219`

### _calculate_chemical_potential `def _calculate_chemical_potential(self, eigenvalues, num_electrons)`
- Defined: `fermi.py:240`

### _find_chemical_potential_iterative `def _find_chemical_potential_iterative(self, eigenvalues, num_electrons, temperature, max_iter)`
- Defined: `fermi.py:253`

### _fermi_dirac `def _fermi_dirac(self, energy, mu, temperature)`
- Defined: `fermi.py:273`

### __init__ `def __init__(self, config)`
- Defined: `fermi.py:286`

### calculate `def calculate(self, eigenvalues, energies)`
- Defined: `fermi.py:289`

### _gaussian `def _gaussian(self, x, mu, sigma)`
- Defined: `fermi.py:302`

### __init__ `def __init__(self, config)`
- Defined: `fermi.py:307`

### calculate `def calculate(self, eigenvalues, eigenvectors, fermi_level)`
- Defined: `fermi.py:310`

### _calculate_kinetic_energy `def _calculate_kinetic_energy(self, occupied_states)`
- Defined: `fermi.py:337`

### _calculate_electronic_pressure `def _calculate_electronic_pressure(self, eigenvalues, fermi_level)`
- Defined: `fermi.py:347`

### _calculate_compressibility `def _calculate_compressibility(self, eigenvalues, fermi_level)`
- Defined: `fermi.py:357`

### __init__ `def __init__(self, config)`
- Defined: `fermi.py:369`

### classify `def classify(self, band_gap, dos_at_fermi)`
- Defined: `fermi.py:372`

### classify_transport `def classify_transport(self, effective_masses, band_gap)`
- Defined: `fermi.py:386`

### migrate `def migrate(self, raw_data, device)`
- Defined: `fermi.py:409`

### _migrate_dict `def _migrate_dict(self, state_dict, device)`
- Defined: `fermi.py:419`

### _migrate_custom_format `def _migrate_custom_format(self, state_dict, device)`
- Defined: `fermi.py:428`

### _migrate_coefs_format `def _migrate_coefs_format(self, state_dict)`
- Defined: `fermi.py:447`

### _migrate_standard_format `def _migrate_standard_format(self, state_dict)`
- Defined: `fermi.py:454`

### __init__ `def __init__(self, checkpoint_path, config)`
- Defined: `fermi.py:459`

### _load_checkpoint `def _load_checkpoint(self)`
- Defined: `fermi.py:472`

### analyze `def analyze(self)`
- Defined: `fermi.py:495`

### _print_report `def _print_report(self, results)`
- Defined: `fermi.py:550`

### __init__ `def __init__(self, config)`
- Defined: `fermi.py:609`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `fermi.py:612`

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `fermi.py:626`

### generate_summary `def generate_summary(self, all_results, output_dir)`
- Defined: `fermi.py:653`

### _generate_text_report `def _generate_text_report(self, summary, output_dir)`
- Defined: `fermi.py:686`

### plot_band_structures `def plot_band_structures(self, all_results, output_dir)`
- Defined: `fermi.py:724`
- Doc: Generate comparison plots of band structures across checkpoints.

## full_seed_prospector.py

### main `def main()`
- Defined: `full_seed_prospector.py:2022`
- Doc: Main entry point.

### calculate `def calculate(self)`
- Defined: `full_seed_prospector.py:166`

### compute `def compute(self, model, loss_mse, epoch)`
- Defined: `full_seed_prospector.py:174`

### save `def save(self, state, path)`
- Defined: `full_seed_prospector.py:183`

### load `def load(self, path)`
- Defined: `full_seed_prospector.py:187`

### should_checkpoint `def should_checkpoint(self)`
- Defined: `full_seed_prospector.py:191`

### execute `def execute(self, model)`
- Defined: `full_seed_prospector.py:199`

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:206`

### _initialize_strassen_structure `def _initialize_strassen_structure(self)`
- Defined: `full_seed_prospector.py:220`
- Doc: Initialize with bias towards canonical Strassen structure.

### forward `def forward(self, A, B)`
- Defined: `full_seed_prospector.py:276`
- Doc: Forward pass computing C = A @ B via low-rank factorization.

### slot_importance `def slot_importance(self)`
- Defined: `full_seed_prospector.py:291`
- Doc: Calculate importance of each rank slot.

### count_active `def count_active(self, threshold)`
- Defined: `full_seed_prospector.py:298`
- Doc: Count active slots above threshold.

### get_flat_parameters `def get_flat_parameters(self)`
- Defined: `full_seed_prospector.py:304`
- Doc: Get flattened parameter vector.

### get_parameter_count `def get_parameter_count(self)`
- Defined: `full_seed_prospector.py:308`
- Doc: Get total parameter count.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:316`

### calculate `def calculate(self, model)`
- Defined: `full_seed_prospector.py:319`
- Doc: Calculate delta: mean squared distance to {-1, 0, 1}.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:336`

### calculate `def calculate(self, model, C_pred, C_true, n_test)`
- Defined: `full_seed_prospector.py:339`
- Doc: Calculate accuracy percentage.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:374`

### accumulate_gradient `def accumulate_gradient(self, model)`
- Defined: `full_seed_prospector.py:379`
- Doc: Accumulate current gradient vector.

### calculate_kappa `def calculate_kappa(self)`
- Defined: `full_seed_prospector.py:393`
- Doc: Calculate condition number of gradient covariance matrix.

### get_kappa_trend `def get_kappa_trend(self)`
- Defined: `full_seed_prospector.py:425`
- Doc: Determine kappa trend direction.

### is_crystallizing `def is_crystallizing(self)`
- Defined: `full_seed_prospector.py:439`
- Doc: Detect if system is in crystallization phase.

### reset `def reset(self)`
- Defined: `full_seed_prospector.py:449`
- Doc: Reset calculator state.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:458`

### calculate `def calculate(self, model, loss, epoch, gradient_norm)`
- Defined: `full_seed_prospector.py:465`
- Doc: Calculate W-entropy with adaptive tau coupling to GNS.

### _calculate_log_W `def _calculate_log_W(self, tau, R, grad_f_sq, f, n_params)`
- Defined: `full_seed_prospector.py:522`
- Doc: Calculate log(W) with numerical stability.

### reset `def reset(self)`
- Defined: `full_seed_prospector.py:546`
- Doc: Reset calculator state.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:556`

### calculate `def calculate(self, model)`
- Defined: `full_seed_prospector.py:559`
- Doc: Calculate active slots and sparsity.

### calculate `def calculate(model)`
- Defined: `full_seed_prospector.py:577`
- Doc: Calculate gradient norm statistics.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:602`

### measure `def measure(self, model)`
- Defined: `full_seed_prospector.py:605`
- Doc: Measure resilience via progressive magnitude pruning.

### _prune_by_magnitude `def _prune_by_magnitude(self, model, sparsity)`
- Defined: `full_seed_prospector.py:684`
- Doc: Prune parameters by magnitude threshold.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:697`

### compute_all `def compute_all(self, model, C_pred, C_true, loss, epoch, force_kappa, force_lc, force_sp)`
- Defined: `full_seed_prospector.py:711`
- Doc: Compute all available metrics.

### accumulate_gradient `def accumulate_gradient(self, model)`
- Defined: `full_seed_prospector.py:787`
- Doc: Accumulate gradient for kappa calculation.

### update_lr `def update_lr(self, lr)`
- Defined: `full_seed_prospector.py:791`
- Doc: Update current learning rate.

### reset `def reset(self)`
- Defined: `full_seed_prospector.py:795`
- Doc: Reset all calculators.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:805`

### compute `def compute(self, model, loss_mse, epoch, kappa)`
- Defined: `full_seed_prospector.py:808`
- Doc: Compute quantization loss with adaptive weighting.

### _get_adaptive_weight `def _get_adaptive_weight(self, epoch, kappa)`
- Defined: `full_seed_prospector.py:835`
- Doc: Calculate adaptive weight based on epoch and kappa.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:857`

### compute `def compute(self, model, loss_mse, epoch)`
- Defined: `full_seed_prospector.py:860`
- Doc: Compute Ricci curvature penalty.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:871`

### compute `def compute(self, model, loss_mse, epoch, kappa)`
- Defined: `full_seed_prospector.py:876`
- Doc: Compute total geometric loss.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:887`

### save `def save(self, state, path)`
- Defined: `full_seed_prospector.py:893`
- Doc: Save checkpoint to disk.

### load `def load(self, path)`
- Defined: `full_seed_prospector.py:911`
- Doc: Load checkpoint from disk.

### should_checkpoint `def should_checkpoint(self)`
- Defined: `full_seed_prospector.py:919`
- Doc: Check if checkpoint interval has elapsed.

### get_latest_checkpoint_path `def get_latest_checkpoint_path(self)`
- Defined: `full_seed_prospector.py:924`
- Doc: Get path to latest checkpoint if exists.

### __init__ `def __init__(self, config, scale)`
- Defined: `full_seed_prospector.py:933`

### generate_batch `def generate_batch(self, n)`
- Defined: `full_seed_prospector.py:938`
- Doc: Generate batch of random matrices.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:950`

### get_batch_size `def get_batch_size(self, epoch)`
- Defined: `full_seed_prospector.py:953`
- Doc: Get batch size for current epoch.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:972`

### should_stop `def should_stop(self, epoch, metrics)`
- Defined: `full_seed_prospector.py:977`
- Doc: Determine if training should stop due to glass state.

### __init__ `def __init__(self, config, seed)`
- Defined: `full_seed_prospector.py:1028`

### execute `def execute(self, model)`
- Defined: `full_seed_prospector.py:1037`
- Doc: Execute prospecting phase with all metrics visible.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1155`

### execute `def execute(self, model)`
- Defined: `full_seed_prospector.py:1164`
- Doc: Execute long training phase.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1338`

### execute `def execute(self, model)`
- Defined: `full_seed_prospector.py:1342`
- Doc: Execute sparsification phase.

### _final_refinement `def _final_refinement(self, model, optimizer, slots_to_prune)`
- Defined: `full_seed_prospector.py:1407`
- Doc: Final refinement after pruning.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1445`

### execute `def execute(self, model)`
- Defined: `full_seed_prospector.py:1448`
- Doc: Execute discretization phase.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1535`

### verify `def verify(self, U, V, W, n_test)`
- Defined: `full_seed_prospector.py:1538`
- Doc: Verify algorithm on random test matrices.

### get_canonical `def get_canonical()`
- Defined: `full_seed_prospector.py:1592`
- Doc: Return canonical Strassen algorithm matrices.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1626`

### prospect `def prospect(self, total_attempts, start_seed)`
- Defined: `full_seed_prospector.py:1631`
- Doc: Prospect multiple seeds for crystals.

### _set_seed `def _set_seed(self, seed)`
- Defined: `full_seed_prospector.py:1733`
- Doc: Set random seed for reproducibility.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1745`

### _signal_handler `def _signal_handler(self, signum, frame)`
- Defined: `full_seed_prospector.py:1758`
- Doc: Handle interrupt signal.

### _set_seed `def _set_seed(self, seed)`
- Defined: `full_seed_prospector.py:1763`
- Doc: Set random seed for reproducibility.

### run `def run(self, resume_from, seed)`
- Defined: `full_seed_prospector.py:1771`
- Doc: Run complete training pipeline.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1852`

### calculate `def calculate(self, model)`
- Defined: `full_seed_prospector.py:1855`
- Doc: Calculate LC as effective local dimensionality (paper definition).

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1918`

### _initialize_sae `def _initialize_sae(self, input_dim, device)`
- Defined: `full_seed_prospector.py:1923`
- Doc: Initialize sparse autoencoder if not already done.

### calculate `def calculate(self, model)`
- Defined: `full_seed_prospector.py:1930`
- Doc: Calculate superposition coefficient psi and effective features F.

### __init__ `def __init__(self, config)`
- Defined: `full_seed_prospector.py:1967`

### calculate `def calculate(self, model, gradient_covariance)`
- Defined: `full_seed_prospector.py:1970`
- Doc: Calculate effective Planck constant and temperature.

## grain.py

### run_training `def run_training(seed, config)`
- Defined: `grain.py:833`

### run_analysis `def run_analysis(checkpoint_dir, output_dir, n_latest, config)`
- Defined: `grain.py:838`

### main `def main()`
- Defined: `grain.py:845`

### to_dict `def to_dict(self)`
- Defined: `grain.py:57`

### forward `def forward(self, a, b)`
- Defined: `grain.py:66`

### get_coefficients `def get_coefficients(self)`
- Defined: `grain.py:67`

### detect `def detect(self, model, pruning_level)`
- Defined: `grain.py:72`

### analyze_layer `def analyze_layer(self, weights, layer_name)`
- Defined: `grain.py:77`

### calculate `def calculate(self, layer_deltas)`
- Defined: `grain.py:82`

### analyze `def analyze(self, model, pruning_level)`
- Defined: `grain.py:87`

### save `def save(self, model, epoch, metrics, path)`
- Defined: `grain.py:92`

### load `def load(self, path)`
- Defined: `grain.py:93`

### update `def update(self, epoch, metrics)`
- Defined: `grain.py:98`

### should_checkpoint `def should_checkpoint(self)`
- Defined: `grain.py:99`

### __init__ `def __init__(self, hidden_dim, matrix_size)`
- Defined: `grain.py:103`

### _initialize_symmetric `def _initialize_symmetric(self)`
- Defined: `grain.py:114`

### forward `def forward(self, a, b)`
- Defined: `grain.py:119`

### get_coefficients `def get_coefficients(self)`
- Defined: `grain.py:122`

### analyze_layer `def analyze_layer(self, weights, layer_name)`
- Defined: `grain.py:131`

### __init__ `def __init__(self, config)`
- Defined: `grain.py:154`

### detect `def detect(self, model, pruning_level)`
- Defined: `grain.py:158`

### _prune_model `def _prune_model(self, model, sparsity)`
- Defined: `grain.py:185`

### _calculate_dislocation `def _calculate_dislocation(self, layer_deltas)`
- Defined: `grain.py:194`

### _analyze_fragmentation `def _analyze_fragmentation(self, layer_analysis)`
- Defined: `grain.py:227`

### __init__ `def __init__(self, config)`
- Defined: `grain.py:244`

### analyze `def analyze(self, model, pruning_level)`
- Defined: `grain.py:248`

### _calculate_coordination_loss `def _calculate_coordination_loss(self, layer_analysis)`
- Defined: `grain.py:269`

### _estimate_domain_count `def _estimate_domain_count(self, layer_analysis)`
- Defined: `grain.py:282`

### _calculate_coherence_length `def _calculate_coherence_length(self, layer_analysis)`
- Defined: `grain.py:297`

### __init__ `def __init__(self, config)`
- Defined: `grain.py:310`

### save `def save(self, model, epoch, metrics, path)`
- Defined: `grain.py:314`

### load `def load(self, path)`
- Defined: `grain.py:332`

### should_save `def should_save(self)`
- Defined: `grain.py:335`

### __init__ `def __init__(self, config)`
- Defined: `grain.py:341`

### update `def update(self, epoch, loss, accuracy, model, grain_result)`
- Defined: `grain.py:358`

### get_current_metrics `def get_current_metrics(self)`
- Defined: `grain.py:382`

### get_training_bar_string `def get_training_bar_string(self, epoch, total_epochs)`
- Defined: `grain.py:389`

### __init__ `def __init__(self, config, seed)`
- Defined: `grain.py:408`

### _setup_signal_handlers `def _setup_signal_handlers(self)`
- Defined: `grain.py:438`

### _signal_handler `def _signal_handler(self, signum, frame)`
- Defined: `grain.py:442`

### _generate_batch `def _generate_batch(self)`
- Defined: `grain.py:447`

### _compute_accuracy `def _compute_accuracy(self, pred, target)`
- Defined: `grain.py:468`

### _save_checkpoint `def _save_checkpoint(self, interrupted)`
- Defined: `grain.py:473`

### train `def train(self)`
- Defined: `grain.py:486`

### __init__ `def __init__(self, checkpoint_path, config)`
- Defined: `grain.py:548`

### _load_checkpoint `def _load_checkpoint(self)`
- Defined: `grain.py:557`

### _migrate_checkpoint `def _migrate_checkpoint(self, raw_data)`
- Defined: `grain.py:580`

### _migrate_dict `def _migrate_dict(self, state_dict)`
- Defined: `grain.py:590`

### _migrate_custom_format `def _migrate_custom_format(self, state_dict)`
- Defined: `grain.py:599`

### _migrate_coefs_format `def _migrate_coefs_format(self, state_dict)`
- Defined: `grain.py:618`

### _migrate_standard_format `def _migrate_standard_format(self, state_dict)`
- Defined: `grain.py:625`

### analyze `def analyze(self)`
- Defined: `grain.py:628`

### _analyze_dislocation_evolution `def _analyze_dislocation_evolution(self, grain_results)`
- Defined: `grain.py:658`

### _find_critical_pruning_level `def _find_critical_pruning_level(self, grain_results)`
- Defined: `grain.py:681`

### _print_report `def _print_report(self, results)`
- Defined: `grain.py:687`

### __init__ `def __init__(self, config)`
- Defined: `grain.py:727`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `grain.py:730`

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `grain.py:744`

### generate_summary `def generate_summary(self, all_results, output_dir)`
- Defined: `grain.py:771`

### _generate_text_report `def _generate_text_report(self, summary, output_dir)`
- Defined: `grain.py:797`

## gravity.py

### main `def main()`
- Defined: `gravity.py:1114`

### forward `def forward(self, a, b)`
- Defined: `gravity.py:60`

### get_coefficients `def get_coefficients(self)`
- Defined: `gravity.py:61`

### calculate `def calculate(self, model)`
- Defined: `gravity.py:66`

### calculate `def calculate(self, model)`
- Defined: `gravity.py:71`

### calculate `def calculate(self, loss_history)`
- Defined: `gravity.py:76`

### calculate `def calculate(self, model, gradient_history)`
- Defined: `gravity.py:81`

### calculate `def calculate(self, entropy_change, energy_dissipated)`
- Defined: `gravity.py:86`

### calculate `def calculate(self, model, temperature)`
- Defined: `gravity.py:91`

### calculate `def calculate(self, model)`
- Defined: `gravity.py:96`

### calculate `def calculate(self, model, test_data)`
- Defined: `gravity.py:101`

### calculate `def calculate(self, model, target_size)`
- Defined: `gravity.py:106`

### calculate `def calculate(self, gradient_covariance)`
- Defined: `gravity.py:111`

### __init__ `def __init__(self, hidden_dim, matrix_size)`
- Defined: `gravity.py:115`

### _initialize `def _initialize(self)`
- Defined: `gravity.py:126`

### forward `def forward(self, a, b)`
- Defined: `gravity.py:131`

### get_coefficients `def get_coefficients(self)`
- Defined: `gravity.py:134`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:143`

### calculate `def calculate(self, model)`
- Defined: `gravity.py:146`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:155`

### calculate `def calculate(self, model)`
- Defined: `gravity.py:158`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:172`

### calculate `def calculate(self, loss_history)`
- Defined: `gravity.py:175`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:184`

### calculate `def calculate(self, model, gradient_history, loss_history, static_gradient)`
- Defined: `gravity.py:187`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:240`

### calculate `def calculate(self, entropy_change, energy_dissipated, has_transition, transition_window)`
- Defined: `gravity.py:243`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:272`

### calculate `def calculate(self, model, temperature, static_gradient)`
- Defined: `gravity.py:275`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:311`

### calculate `def calculate(self, model)`
- Defined: `gravity.py:314`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:328`

### calculate `def calculate(self, model, test_data)`
- Defined: `gravity.py:331`

### _prune_model `def _prune_model(self, model, sparsity)`
- Defined: `gravity.py:355`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:366`

### calculate `def calculate(self, model, target_size)`
- Defined: `gravity.py:369`

### _kronecker_recursive `def _kronecker_recursive(self, matrix, power)`
- Defined: `gravity.py:391`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:399`

### calculate `def calculate(self, gradient_history, static_gradient)`
- Defined: `gravity.py:403`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:455`

### detect `def detect(self, loss_history, entropy_history)`
- Defined: `gravity.py:458`

### __init__ `def __init__(self, checkpoint_path, config)`
- Defined: `gravity.py:506`

### _load_checkpoint `def _load_checkpoint(self)`
- Defined: `gravity.py:525`

### _migrate_checkpoint `def _migrate_checkpoint(self, raw_data)`
- Defined: `gravity.py:548`

### _migrate_dict `def _migrate_dict(self, state_dict)`
- Defined: `gravity.py:560`

### _migrate_custom_format `def _migrate_custom_format(self, state_dict)`
- Defined: `gravity.py:569`

### _migrate_coefs_format `def _migrate_coefs_format(self, state_dict)`
- Defined: `gravity.py:588`

### _migrate_standard_format `def _migrate_standard_format(self, state_dict)`
- Defined: `gravity.py:595`

### _compute_static_gradient `def _compute_static_gradient(self)`
- Defined: `gravity.py:598`

### _generate_test_data `def _generate_test_data(self)`
- Defined: `gravity.py:619`

### analyze `def analyze(self)`
- Defined: `gravity.py:630`

### _determine_failure_mode `def _determine_failure_mode(self, delta, basin, kappa, transition)`
- Defined: `gravity.py:733`

### _print_report `def _print_report(self, results)`
- Defined: `gravity.py:744`

### __init__ `def __init__(self, config)`
- Defined: `gravity.py:838`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `gravity.py:841`

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `gravity.py:855`

### generate_summary `def generate_summary(self, all_results, output_dir)`
- Defined: `gravity.py:882`

### _compute_emergent_constants `def _compute_emergent_constants(self, results)`
- Defined: `gravity.py:904`

### _verify_universal_laws `def _verify_universal_laws(self, results)`
- Defined: `gravity.py:959`

### _compute_kappa_correlation `def _compute_kappa_correlation(self, results)`
- Defined: `gravity.py:993`

### _generate_text_report `def _generate_text_report(self, summary, output_dir)`
- Defined: `gravity.py:1023`

### extract_values `def extract_values(data_list, key_path)`
- Defined: `gravity.py:908`

## grigori_perelmans_ricci_flow.py

### set_random_seed `def set_random_seed(seed)`
- Defined: `grigori_perelmans_ricci_flow.py:70`

### main `def main()`
- Defined: `grigori_perelmans_ricci_flow.py:593`

### __init__ `def __init__(self, config)`
- Defined: `grigori_perelmans_ricci_flow.py:86`

### _initialize `def _initialize(self)`
- Defined: `grigori_perelmans_ricci_flow.py:94`

### forward `def forward(self, a, b)`
- Defined: `grigori_perelmans_ricci_flow.py:99`

### get_coefficients `def get_coefficients(self)`
- Defined: `grigori_perelmans_ricci_flow.py:102`

### get_flat_params `def get_flat_params(self)`
- Defined: `grigori_perelmans_ricci_flow.py:109`
- Doc: Return all parameters as a single flattened vector.

### set_flat_params `def set_flat_params(self, flat_params)`
- Defined: `grigori_perelmans_ricci_flow.py:114`
- Doc: Set model parameters from a flattened vector.

### generate_batch `def generate_batch(batch_size, config)`
- Defined: `grigori_perelmans_ricci_flow.py:130`

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `grigori_perelmans_ricci_flow.py:147`

### migrate `def migrate(self, state_dict)`
- Defined: `grigori_perelmans_ricci_flow.py:149`

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `grigori_perelmans_ricci_flow.py:152`

### migrate `def migrate(self, state_dict)`
- Defined: `grigori_perelmans_ricci_flow.py:154`

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `grigori_perelmans_ricci_flow.py:167`

### migrate `def migrate(self, state_dict)`
- Defined: `grigori_perelmans_ricci_flow.py:169`

### __init__ `def __init__(self)`
- Defined: `grigori_perelmans_ricci_flow.py:173`

### migrate_checkpoint `def migrate_checkpoint(self, path, device)`
- Defined: `grigori_perelmans_ricci_flow.py:176`

### __init__ `def __init__(self, model, config)`
- Defined: `grigori_perelmans_ricci_flow.py:199`

### compute_hessian `def compute_hessian(self, input_a, input_b, target_c)`
- Defined: `grigori_perelmans_ricci_flow.py:203`
- Doc: Computes exact Hessian of loss w.r.t parameters.

### _loss_wrapper `def _loss_wrapper(self, flat_params, original_params, a, b, c)`
- Defined: `grigori_perelmans_ricci_flow.py:223`
- Doc: Wrapper to compute loss from flat param vector.

### _compute_diagonal_hessian `def _compute_diagonal_hessian(self, a, b, c)`
- Defined: `grigori_perelmans_ricci_flow.py:240`
- Doc: Approximation: Diagonal of Hessian (Gauss-Newton).

### analyze_curvature `def analyze_curvature(self, hessian)`
- Defined: `grigori_perelmans_ricci_flow.py:247`
- Doc: Analyze Hessian spectrum to derive Ricci Scalar and Topological invariants.

### compute_heat_kernel_trace `def compute_heat_kernel_trace(self, eigenvalues, t)`
- Defined: `grigori_perelmans_ricci_flow.py:285`
- Doc: Trace of Heat Kernel: Z(t) = Sum( exp(-lambda_i * t) ).

### compute_topological_entropy `def compute_topological_entropy(self, eigenvalues)`
- Defined: `grigori_perelmans_ricci_flow.py:296`
- Doc: von Neumann Entropy / Spectral Entropy.

### __init__ `def __init__(self, model, eigenvalues, config)`
- Defined: `grigori_perelmans_ricci_flow.py:316`

### detect_necks `def detect_necks(self, curvature_analysis)`
- Defined: `grigori_perelmans_ricci_flow.py:325`
- Doc: Identify if the system is in a 'bottleneck' state.

### propose_surgery `def propose_surgery(self)`
- Defined: `grigori_perelmans_ricci_flow.py:336`
- Doc: Propose parameters to 'cut' (prune) based on curvature heuristics.

### __init__ `def __init__(self, eigenvalues, ricci_scalar, config)`
- Defined: `grigori_perelmans_ricci_flow.py:371`

### calculate `def calculate(self)`
- Defined: `grigori_perelmans_ricci_flow.py:376`

### _get_spectral_gap `def _get_spectral_gap(self)`
- Defined: `grigori_perelmans_ricci_flow.py:412`

### _compute_spectral_entropy `def _compute_spectral_entropy(self)`
- Defined: `grigori_perelmans_ricci_flow.py:420`

### __init__ `def __init__(self, config)`
- Defined: `grigori_perelmans_ricci_flow.py:435`

### analyze_checkpoint `def analyze_checkpoint(self, checkpoint_path, device)`
- Defined: `grigori_perelmans_ricci_flow.py:439`
- Doc: Perform complete analysis of a single checkpoint.

### analyze_directory `def analyze_directory(self, directory, device, pattern)`
- Defined: `grigori_perelmans_ricci_flow.py:527`

### _print_summary `def _print_summary(self, report)`
- Defined: `grigori_perelmans_ricci_flow.py:560`

## hawking_radiation.py

### load_checkpoint_robust `def load_checkpoint_robust(path, device)`
- Defined: `hawking_radiation.py:93`
- Doc: Load checkpoint with robust handling of custom classes.

### main `def main()`
- Defined: `hawking_radiation.py:1406`

### find_class `def find_class(self, module, name)`
- Defined: `hawking_radiation.py:44`
- Doc: Override find_class to handle missing classes gracefully.

### _create_dummy_class `def _create_dummy_class(self, name)`
- Defined: `hawking_radiation.py:62`
- Doc: Create a dummy class that can hold attributes.

### get_effective_input_dim `def get_effective_input_dim(self)`
- Defined: `hawking_radiation.py:172`

### get_total_parameters `def get_total_parameters(self)`
- Defined: `hawking_radiation.py:175`

### get_coefficients `def get_coefficients(self)`
- Defined: `hawking_radiation.py:189`

### forward `def forward(self, a, b)`
- Defined: `hawking_radiation.py:190`

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:200`

### _initialize `def _initialize(self)`
- Defined: `hawking_radiation.py:211`

### forward `def forward(self, a, b)`
- Defined: `hawking_radiation.py:216`

### get_coefficients `def get_coefficients(self)`
- Defined: `hawking_radiation.py:219`

### get_flat_parameters `def get_flat_parameters(self)`
- Defined: `hawking_radiation.py:226`

### migrate `def migrate(self, raw_data, device)`
- Defined: `hawking_radiation.py:245`
- Doc: Main migration entry point with multiple strategies.

### _try_extract_state_dict `def _try_extract_state_dict(self, data, device)`
- Defined: `hawking_radiation.py:269`
- Doc: Try standard state dict extraction methods.

### _try_nested_extraction `def _try_nested_extraction(self, data, device)`
- Defined: `hawking_radiation.py:288`
- Doc: Try to extract from nested structures.

### _try_direct_tensor_extraction `def _try_direct_tensor_extraction(self, data, device)`
- Defined: `hawking_radiation.py:313`
- Doc: Try to extract tensors directly from any structure.

### _reconstruct_from_tensors `def _reconstruct_from_tensors(self, tensors, device)`
- Defined: `hawking_radiation.py:338`
- Doc: Reconstruct U, V, W from found tensors.

### _is_state_dict `def _is_state_dict(self, data)`
- Defined: `hawking_radiation.py:386`
- Doc: Check if dict looks like a state dict.

### _migrate_dict `def _migrate_dict(self, state_dict, device)`
- Defined: `hawking_radiation.py:398`
- Doc: Migrate a state dict to the expected format.

### _migrate_custom_format `def _migrate_custom_format(self, state_dict, device)`
- Defined: `hawking_radiation.py:424`

### _migrate_coefs_format `def _migrate_coefs_format(self, state_dict, device)`
- Defined: `hawking_radiation.py:449`

### _migrate_standard_format `def _migrate_standard_format(self, state_dict, device)`
- Defined: `hawking_radiation.py:456`

### _migrate_encoder_format `def _migrate_encoder_format(self, state_dict, device)`
- Defined: `hawking_radiation.py:466`
- Doc: Handle encoder.layers style checkpoints.

### _migrate_prefixed_format `def _migrate_prefixed_format(self, state_dict, prefix, device)`
- Defined: `hawking_radiation.py:496`
- Doc: Handle prefixed state dict keys.

### extract `def extract(checkpoint)`
- Defined: `hawking_radiation.py:518`
- Doc: Extract all relevant metadata from checkpoint.

### _extract_delta `def _extract_delta(data, depth)`
- Defined: `hawking_radiation.py:571`
- Doc: Recursively search for delta in nested structures.

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:612`

### calculate `def calculate(self, model, gradient, precomputed_delta)`
- Defined: `hawking_radiation.py:615`

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:679`

### calculate `def calculate(self, model, loss, precomputed_delta)`
- Defined: `hawking_radiation.py:682`

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:771`

### calculate `def calculate(self, model, loss, loss_history)`
- Defined: `hawking_radiation.py:774`

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:826`

### calculate `def calculate(self, model, h_bar, G_alg)`
- Defined: `hawking_radiation.py:829`

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:886`

### calculate `def calculate(self, model, G_alg, c_eff, h_bar)`
- Defined: `hawking_radiation.py:889`

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:942`

### calculate `def calculate(self, model, M_eff)`
- Defined: `hawking_radiation.py:945`

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:994`

### calculate_all `def calculate_all(self, model, loss, loss_history, gradient, precomputed_delta)`
- Defined: `hawking_radiation.py:1003`
- Doc: Calculate all Hawking radiation metrics.

### _classify_state `def _classify_state(self, delta, T_hawking)`
- Defined: `hawking_radiation.py:1151`

### __init__ `def __init__(self, config)`
- Defined: `hawking_radiation.py:1169`

### analyze_checkpoint `def analyze_checkpoint(self, checkpoint_path)`
- Defined: `hawking_radiation.py:1174`
- Doc: Analyze a single checkpoint with robust error handling.

### _compute_gradient `def _compute_gradient(self, model)`
- Defined: `hawking_radiation.py:1227`
- Doc: Compute gradient on random batch.

### _print_report `def _print_report(self, results)`
- Defined: `hawking_radiation.py:1255`
- Doc: Print formatted report.

### analyze_directory `def analyze_directory(self, checkpoint_dir, output_dir, pattern)`
- Defined: `hawking_radiation.py:1295`
- Doc: Analyze all checkpoints in directory.

### _generate_summary `def _generate_summary(self, results, errors)`
- Defined: `hawking_radiation.py:1342`
- Doc: Generate aggregate summary.

### extract_tensors `def extract_tensors(obj, prefix)`
- Defined: `hawking_radiation.py:317`

### __init__ `def __init__(self)`
- Defined: `hawking_radiation.py:65`

### __repr__ `def __repr__(self)`
- Defined: `hawking_radiation.py:72`

### __getitem__ `def __getitem__(self, key)`
- Defined: `hawking_radiation.py:75`

### keys `def keys(self)`
- Defined: `hawking_radiation.py:78`

### values `def values(self)`
- Defined: `hawking_radiation.py:81`

### items `def items(self)`
- Defined: `hawking_radiation.py:84`

### get `def get(self, key, default)`
- Defined: `hawking_radiation.py:87`

## maxwell_strassen_analysis.py

### main `def main()`
- Defined: `maxwell_strassen_analysis.py:788`

### get_effective_input_dim `def get_effective_input_dim(self)`
- Defined: `maxwell_strassen_analysis.py:103`

### get_total_parameters `def get_total_parameters(self)`
- Defined: `maxwell_strassen_analysis.py:106`

### get_coefficients `def get_coefficients(self)`
- Defined: `maxwell_strassen_analysis.py:117`

### map_weights_to_lattice `def map_weights_to_lattice(self, weights)`
- Defined: `maxwell_strassen_analysis.py:122`

### solve_poisson `def solve_poisson(self, charge_density, permittivity)`
- Defined: `maxwell_strassen_analysis.py:127`

### compute_scattering `def compute_scattering(self, permittivity)`
- Defined: `maxwell_strassen_analysis.py:128`

### analyze_permittivity_tensor `def analyze_permittivity_tensor(self, permittivity)`
- Defined: `maxwell_strassen_analysis.py:133`

### classify `def classify(self, metrics)`
- Defined: `maxwell_strassen_analysis.py:138`

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:146`

### _initialize_weights `def _initialize_weights(self)`
- Defined: `maxwell_strassen_analysis.py:156`

### forward `def forward(self, a, b)`
- Defined: `maxwell_strassen_analysis.py:161`

### get_coefficients `def get_coefficients(self)`
- Defined: `maxwell_strassen_analysis.py:164`

### migrate `def migrate(self, raw_data, config)`
- Defined: `maxwell_strassen_analysis.py:172`

### _migrate_dict `def _migrate_dict(self, state_dict, config)`
- Defined: `maxwell_strassen_analysis.py:185`

### _migrate_custom `def _migrate_custom(self, sd, config)`
- Defined: `maxwell_strassen_analysis.py:194`

### _migrate_coefs `def _migrate_coefs(self, sd)`
- Defined: `maxwell_strassen_analysis.py:203`

### _migrate_standard `def _migrate_standard(self, sd)`
- Defined: `maxwell_strassen_analysis.py:208`

### _np `def _np(tensor)`
- Defined: `maxwell_strassen_analysis.py:217`

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:241`

### map_weights_to_lattice `def map_weights_to_lattice(self, weights)`
- Defined: `maxwell_strassen_analysis.py:244`
- Doc: Returns:

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:295`

### analyze_permittivity_tensor `def analyze_permittivity_tensor(self, permittivity)`
- Defined: `maxwell_strassen_analysis.py:298`
- Doc: Analyze the effective permittivity tensor of the medium.

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:360`

### solve_poisson `def solve_poisson(self, charge_density, permittivity)`
- Defined: `maxwell_strassen_analysis.py:363`
- Doc: Solves Poisson equation for Electric Potential φ.

### compute_scattering `def compute_scattering(self, permittivity)`
- Defined: `maxwell_strassen_analysis.py:401`
- Doc: Compute the Scattering Amplitude S(k).

### _find_peaks `def _find_peaks(self, intensity)`
- Defined: `maxwell_strassen_analysis.py:438`
- Doc: Detect sharp peaks indicative of crystallinity.

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:474`

### calculate `def calculate(self, potential, intensity)`
- Defined: `maxwell_strassen_analysis.py:477`

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:505`

### analyze `def analyze(self, fourier_coeffs)`
- Defined: `maxwell_strassen_analysis.py:508`

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:548`

### classify `def classify(self, em_metrics, purity_metrics)`
- Defined: `maxwell_strassen_analysis.py:551`
- Doc: Decision Logic:

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:606`

### should_save `def should_save(self)`
- Defined: `maxwell_strassen_analysis.py:610`

### save `def save(self, data, output_dir)`
- Defined: `maxwell_strassen_analysis.py:613`

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:626`

### visualize_lattice `def visualize_lattice(self, permittivity, output_dir, name)`
- Defined: `maxwell_strassen_analysis.py:629`

### visualize_scattering `def visualize_scattering(self, scattering_slice, output_dir, name)`
- Defined: `maxwell_strassen_analysis.py:648`

### visualize_potential `def visualize_potential(self, potential, output_dir, name)`
- Defined: `maxwell_strassen_analysis.py:659`

### __init__ `def __init__(self, config)`
- Defined: `maxwell_strassen_analysis.py:678`

### _calculate_purity_metrics `def _calculate_purity_metrics(self, weights)`
- Defined: `maxwell_strassen_analysis.py:690`

### analyze_checkpoint `def analyze_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `maxwell_strassen_analysis.py:701`
- Doc: Executes the full Maxwellian analysis pipeline on a single checkpoint.

### generate_report `def generate_report(self, results, output_dir)`
- Defined: `maxwell_strassen_analysis.py:765`

## mbl_analyzer.py

### main `def main()`
- Defined: `mbl_analyzer.py:1227`

### get_effective_input_dim `def get_effective_input_dim(self)`
- Defined: `mbl_analyzer.py:84`

### get_total_parameters `def get_total_parameters(self)`
- Defined: `mbl_analyzer.py:87`

### get_coefficients `def get_coefficients(self)`
- Defined: `mbl_analyzer.py:95`

### forward `def forward(self, a, b)`
- Defined: `mbl_analyzer.py:96`

### calculate `def calculate(self, model)`
- Defined: `mbl_analyzer.py:102`

### calculate `def calculate(self, model)`
- Defined: `mbl_analyzer.py:108`

### calculate `def calculate(self, participation_ratio, energy_gap)`
- Defined: `mbl_analyzer.py:114`

### analyze_robustness `def analyze_robustness(self, model, noise_levels)`
- Defined: `mbl_analyzer.py:120`

### save_checkpoint `def save_checkpoint(self, model, epoch, metrics, loss_history, path)`
- Defined: `mbl_analyzer.py:126`

### load_checkpoint `def load_checkpoint(self, path)`
- Defined: `mbl_analyzer.py:128`

### collect `def collect(self, model, loss, epoch, loss_history)`
- Defined: `mbl_analyzer.py:134`

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:143`

### _initialize_weights `def _initialize_weights(self)`
- Defined: `mbl_analyzer.py:154`
- Doc: Xavier initialization with symmetry constraint for U and V.

### forward `def forward(self, a, b)`
- Defined: `mbl_analyzer.py:160`
- Doc: Forward pass implementing bilinear multiplication.

### get_coefficients `def get_coefficients(self)`
- Defined: `mbl_analyzer.py:164`
- Doc: Returns weight matrices for analysis.

### get_flat_parameters `def get_flat_parameters(self)`
- Defined: `mbl_analyzer.py:172`
- Doc: Returns all parameters flattened for Hamiltonian construction.

### construct_hessian_approximation `def construct_hessian_approximation(self)`
- Defined: `mbl_analyzer.py:179`
- Doc: Constructs approximate Hessian matrix from weight correlations.

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:217`

### calculate `def calculate(self, model)`
- Defined: `mbl_analyzer.py:220`
- Doc: Calculate level spacing statistics from model weights.

### _construct_hessian_from_weights `def _construct_hessian_from_weights(self, model)`
- Defined: `mbl_analyzer.py:267`
- Doc: Alternative Hessian construction for generic models.

### _compute_eigenvalues `def _compute_eigenvalues(self, hessian)`
- Defined: `mbl_analyzer.py:283`
- Doc: Compute sorted eigenvalues of the Hamiltonian.

### _calculate_spacing_ratios `def _calculate_spacing_ratios(self, spacings)`
- Defined: `mbl_analyzer.py:288`
- Doc: Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).

### _classify_phase `def _classify_phase(self, mean_ratio)`
- Defined: `mbl_analyzer.py:303`
- Doc: Classify the quantum phase based on level spacing ratio.

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:331`

### calculate `def calculate(self, model)`
- Defined: `mbl_analyzer.py:334`
- Doc: Calculate participation ratios for all weight layers.

### _calculate_ipr `def _calculate_ipr(self, coefficients)`
- Defined: `mbl_analyzer.py:380`
- Doc: Calculate standard Inverse Participation Ratio.

### _calculate_renyi_ipr `def _calculate_renyi_ipr(self, coefficients, q)`
- Defined: `mbl_analyzer.py:395`
- Doc: Calculate q-th order Rényi IPR.

### _calculate_fractal_dimension `def _calculate_fractal_dimension(self, ipr, n)`
- Defined: `mbl_analyzer.py:409`
- Doc: Calculate fractal dimension D_q from IPR.

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:430`

### calculate `def calculate(self, participation_ratio, energy_gap)`
- Defined: `mbl_analyzer.py:433`
- Doc: Calculate synthetic Planck's constant.

### calculate_from_model `def calculate_from_model(self, model, level_spacing_results, pr_results)`
- Defined: `mbl_analyzer.py:456`
- Doc: Comprehensive calculation from model and previous analyses.

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:497`

### calculate_base_discretization `def calculate_base_discretization(self, model)`
- Defined: `mbl_analyzer.py:501`
- Doc: Calculate the base discretization level from weight rounding error.

### analyze_robustness `def analyze_robustness(self, model, noise_levels)`
- Defined: `mbl_analyzer.py:528`
- Doc: Test robustness by applying noise and measuring gap collapse.

### _perturb_and_measure `def _perturb_and_measure(self, model, noise_level)`
- Defined: `mbl_analyzer.py:584`
- Doc: Apply noise to model and measure resulting metrics.

### _delta_to_alpha `def _delta_to_alpha(self, delta)`
- Defined: `mbl_analyzer.py:607`
- Doc: Convert discretization error to purity alpha.

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:620`

### calculate `def calculate(self, model)`
- Defined: `mbl_analyzer.py:623`

### _compute_layer_purity `def _compute_layer_purity(self, weights)`
- Defined: `mbl_analyzer.py:652`

### _delta_to_alpha `def _delta_to_alpha(self, delta)`
- Defined: `mbl_analyzer.py:658`

### _assess_purity_quality `def _assess_purity_quality(self, alpha, variance)`
- Defined: `mbl_analyzer.py:663`

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:679`

### calculate `def calculate(self, loss_history)`
- Defined: `mbl_analyzer.py:682`

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:726`

### classify `def classify(self, alpha, temperature)`
- Defined: `mbl_analyzer.py:729`

### migrate `def migrate(self, raw_data, device)`
- Defined: `mbl_analyzer.py:751`

### _migrate_dict `def _migrate_dict(self, state_dict, device)`
- Defined: `mbl_analyzer.py:761`

### _migrate_custom_format `def _migrate_custom_format(self, state_dict, device)`
- Defined: `mbl_analyzer.py:770`

### _migrate_coefs_format `def _migrate_coefs_format(self, state_dict)`
- Defined: `mbl_analyzer.py:789`

### _migrate_standard_format `def _migrate_standard_format(self, state_dict)`
- Defined: `mbl_analyzer.py:796`

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:805`

### should_save_checkpoint `def should_save_checkpoint(self)`
- Defined: `mbl_analyzer.py:810`
- Doc: Check if 5 minutes have elapsed since last checkpoint.

### save_checkpoint `def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)`
- Defined: `mbl_analyzer.py:816`
- Doc: Save checkpoint with all MBL metrics.

### load_checkpoint `def load_checkpoint(self, path)`
- Defined: `mbl_analyzer.py:850`
- Doc: Load checkpoint with automatic device placement.

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:860`

### collect `def collect(self, model, loss, epoch, loss_history)`
- Defined: `mbl_analyzer.py:870`
- Doc: Collect all metrics for the current training state.

### _classify_quantum_phase `def _classify_quantum_phase(self, level_spacing, hbar_results)`
- Defined: `mbl_analyzer.py:946`
- Doc: Classify the combined quantum phase.

### __init__ `def __init__(self, checkpoint_path, config)`
- Defined: `mbl_analyzer.py:967`

### _load_checkpoint `def _load_checkpoint(self)`
- Defined: `mbl_analyzer.py:975`
- Doc: Load and migrate checkpoint to model.

### analyze `def analyze(self)`
- Defined: `mbl_analyzer.py:998`
- Doc: Perform complete MBL analysis on checkpoint.

### _generate_summary `def _generate_summary(self, metrics, robustness)`
- Defined: `mbl_analyzer.py:1026`
- Doc: Generate executive summary of analysis.

### _print_report `def _print_report(self, results)`
- Defined: `mbl_analyzer.py:1043`
- Doc: Print formatted analysis report.

### __init__ `def __init__(self, config)`
- Defined: `mbl_analyzer.py:1107`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `mbl_analyzer.py:1110`
- Doc: Process single checkpoint and save results.

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `mbl_analyzer.py:1125`
- Doc: Process multiple checkpoints from directory.

### generate_summary `def generate_summary(self, all_results, output_dir)`
- Defined: `mbl_analyzer.py:1155`
- Doc: Generate aggregate summary report.

### _generate_text_report `def _generate_text_report(self, summary, output_dir)`
- Defined: `mbl_analyzer.py:1188`
- Doc: Generate human-readable text report.

## menu.py

### clear_screen `def clear_screen()`
- Defined: `menu.py:304`

### print_header `def print_header(title, subtitle)`
- Defined: `menu.py:308`

### print_wrapped `def print_wrapped(text, indent)`
- Defined: `menu.py:318`

### wait_for_enter `def wait_for_enter()`
- Defined: `menu.py:323`

### run_script `def run_script(entry)`
- Defined: `menu.py:332`

### show_checkpoints `def show_checkpoints()`
- Defined: `menu.py:362`

### show_results `def show_results()`
- Defined: `menu.py:397`

### show_category `def show_category(cat)`
- Defined: `menu.py:433`

### main_menu `def main_menu()`
- Defined: `menu.py:473`

## percolation_analysis.py

### _safe_torch_load `def _safe_torch_load(path)`
- Defined: `percolation_analysis.py:165`
- Doc: Load a torch checkpoint that may contain unknown serialized classes

### main `def main()`
- Defined: `percolation_analysis.py:1264`

### get_effective_input_dim `def get_effective_input_dim(self)`
- Defined: `percolation_analysis.py:82`

### get_total_parameters `def get_total_parameters(self)`
- Defined: `percolation_analysis.py:85`

### get_percolation_thresholds `def get_percolation_thresholds(self)`
- Defined: `percolation_analysis.py:89`

### get_coefficients `def get_coefficients(self)`
- Defined: `percolation_analysis.py:98`

### __init__ `def __init__(self, weights)`
- Defined: `percolation_analysis.py:102`

### get_coefficients `def get_coefficients(self)`
- Defined: `percolation_analysis.py:105`

### get_flat_parameters `def get_flat_parameters(self)`
- Defined: `percolation_analysis.py:108`

### __init__ `def __init__(self)`
- Defined: `percolation_analysis.py:155`

### __repr__ `def __repr__(self)`
- Defined: `percolation_analysis.py:161`

### _patch_missing `def _patch_missing(exc)`
- Defined: `percolation_analysis.py:189`

### _cleanup `def _cleanup()`
- Defined: `percolation_analysis.py:230`

### migrate `def migrate(self, raw_data, config)`
- Defined: `percolation_analysis.py:265`

### _migrate_dict `def _migrate_dict(self, state_dict, config)`
- Defined: `percolation_analysis.py:285`

### _try_migrate_nested `def _try_migrate_nested(self, candidate, config)`
- Defined: `percolation_analysis.py:300`

### _migrate_custom `def _migrate_custom(self, sd, config)`
- Defined: `percolation_analysis.py:310`

### _migrate_coefs `def _migrate_coefs(self, sd)`
- Defined: `percolation_analysis.py:326`

### _migrate_standard `def _migrate_standard(self, sd)`
- Defined: `percolation_analysis.py:331`

### _np `def _np(tensor)`
- Defined: `percolation_analysis.py:340`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:351`

### construct_adjacency_from_weights `def construct_adjacency_from_weights(self, weights)`
- Defined: `percolation_analysis.py:354`

### construct_weight_correlation_graph `def construct_weight_correlation_graph(self, weights)`
- Defined: `percolation_analysis.py:375`

### construct_slot_interaction_graph `def construct_slot_interaction_graph(self, weights)`
- Defined: `percolation_analysis.py:386`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:403`

### analyze `def analyze(self, adjacency, thresholds)`
- Defined: `percolation_analysis.py:406`

### _susceptibility `def _susceptibility(self, sizes, n)`
- Defined: `percolation_analysis.py:449`

### _find_pc `def _find_pc(self, thresholds, res)`
- Defined: `percolation_analysis.py:456`

### _exponents `def _exponents(self, thresholds, res, pc)`
- Defined: `percolation_analysis.py:462`

### _fit_pl `def _fit_pl(x, y)`
- Defined: `percolation_analysis.py:478`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:489`

### analyze `def analyze(self, weights, thresholds)`
- Defined: `percolation_analysis.py:492`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:543`

### analyze `def analyze(self, weights)`
- Defined: `percolation_analysis.py:546`

### _d2a `def _d2a(self, delta)`
- Defined: `percolation_analysis.py:626`

### _kappa `def _kappa(self, wv)`
- Defined: `percolation_analysis.py:631`

### _hbar `def _hbar(self, wv)`
- Defined: `percolation_analysis.py:644`

### _teff `def _teff(self, wv)`
- Defined: `percolation_analysis.py:656`

### _lc `def _lc(self, wv, delta)`
- Defined: `percolation_analysis.py:659`

### _entropy `def _entropy(self, wv)`
- Defined: `percolation_analysis.py:669`

### _ipr `def _ipr(self, wv)`
- Defined: `percolation_analysis.py:680`

### _lsr `def _lsr(self, wv)`
- Defined: `percolation_analysis.py:687`

### _fractal `def _fractal(self, ipr, n)`
- Defined: `percolation_analysis.py:699`

### _phase `def _phase(self, alpha, temp, delta)`
- Defined: `percolation_analysis.py:704`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:726`

### analyze_at_threshold `def analyze_at_threshold(self, adjacency, threshold)`
- Defined: `percolation_analysis.py:729`

### _tau `def _tau(self, sizes)`
- Defined: `percolation_analysis.py:753`

### _critical `def _critical(self, sizes, n)`
- Defined: `percolation_analysis.py:767`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:776`

### classify_universality `def classify_universality(self, measured)`
- Defined: `percolation_analysis.py:779`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:809`

### should_save `def should_save(self)`
- Defined: `percolation_analysis.py:814`

### save `def save(self, results, output_dir)`
- Defined: `percolation_analysis.py:817`

### load `def load(self, output_dir)`
- Defined: `percolation_analysis.py:829`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:838`

### generate_all_figures `def generate_all_figures(self, results, output_dir)`
- Defined: `percolation_analysis.py:841`

### _plot_bond `def _plot_bond(self, data, out)`
- Defined: `percolation_analysis.py:856`

### _plot_pruning `def _plot_pruning(self, data, out)`
- Defined: `percolation_analysis.py:889`

### _plot_dashboard `def _plot_dashboard(self, data, out)`
- Defined: `percolation_analysis.py:952`

### _plot_site `def _plot_site(self, data, out)`
- Defined: `percolation_analysis.py:993`

### _plot_cluster `def _plot_cluster(self, data, out)`
- Defined: `percolation_analysis.py:1019`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:1050`

### generate_text_report `def generate_text_report(self, results, output_dir)`
- Defined: `percolation_analysis.py:1053`

### generate_json_report `def generate_json_report(self, results, output_dir)`
- Defined: `percolation_analysis.py:1131`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:1140`

### _load_weights `def _load_weights(self, checkpoint_path)`
- Defined: `percolation_analysis.py:1153`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `percolation_analysis.py:1164`

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `percolation_analysis.py:1212`

### _maybe_save `def _maybe_save(self, results, output_dir)`
- Defined: `percolation_analysis.py:1234`

### _comparative_summary `def _comparative_summary(self, all_res, output_dir)`
- Defined: `percolation_analysis.py:1238`

### __init__ `def __init__(self, config)`
- Defined: `percolation_analysis.py:118`

### _initialize_weights `def _initialize_weights(self)`
- Defined: `percolation_analysis.py:127`

### forward `def forward(self, a, b)`
- Defined: `percolation_analysis.py:132`

### get_coefficients `def get_coefficients(self)`
- Defined: `percolation_analysis.py:135`

### get_flat_parameters `def get_flat_parameters(self)`
- Defined: `percolation_analysis.py:141`

## plank.py

### set_random_seed `def set_random_seed(seed)`
- Defined: `plank.py:114`
- Doc: Set random seeds for reproducibility.

### parse_arguments `def parse_arguments()`
- Defined: `plank.py:1339`
- Doc: Parse command line arguments.

### create_config_from_args `def create_config_from_args(args)`
- Defined: `plank.py:1387`
- Doc: Create configuration from command line arguments.

### main `def main()`
- Defined: `plank.py:1399`
- Doc: Main execution entry point.

### __post_init__ `def __post_init__(self)`
- Defined: `plank.py:100`
- Doc: Validate configuration parameters.

### __init__ `def __init__(self, config)`
- Defined: `plank.py:133`

### _initialize_symmetric `def _initialize_symmetric(self)`
- Defined: `plank.py:143`
- Doc: Initialize with Xavier uniform, symmetric U and V.

### forward `def forward(self, matrix_a, matrix_b)`
- Defined: `plank.py:149`
- Doc: Forward pass computing approximate matrix product.

### get_coefficients `def get_coefficients(self)`
- Defined: `plank.py:166`
- Doc: Return current coefficient matrices.

### compute_lambda_effective `def compute_lambda_effective(self)`
- Defined: `plank.py:174`
- Doc: Compute effective lambda (confinement potential) from weight magnitudes.

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `plank.py:198`
- Doc: Check if this strategy can handle the given state dict.

### migrate `def migrate(self, state_dict)`
- Defined: `plank.py:203`
- Doc: Migrate state dict to standard format.

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `plank.py:211`

### migrate `def migrate(self, state_dict)`
- Defined: `plank.py:214`

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `plank.py:240`

### migrate `def migrate(self, state_dict)`
- Defined: `plank.py:243`

### can_migrate `def can_migrate(self, state_dict)`
- Defined: `plank.py:273`

### migrate `def migrate(self, state_dict)`
- Defined: `plank.py:276`

### __init__ `def __init__(self)`
- Defined: `plank.py:287`

### migrate_checkpoint `def migrate_checkpoint(self, path, device)`
- Defined: `plank.py:294`
- Doc: Attempt to migrate checkpoint using available strategies.

### generate_batch `def generate_batch(batch_size, config)`
- Defined: `plank.py:345`
- Doc: Generate a batch of random matrix pairs and their products.

### verify_structure `def verify_structure(coeffs, config)`
- Defined: `plank.py:366`
- Doc: Verify if coefficients represent valid Strassen structure.

### compute_kappa `def compute_kappa(model, num_batches, config)`
- Defined: `plank.py:388`
- Doc: Compute condition number of gradient covariance matrix.

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `plank.py:429`
- Doc: Compute maximum deviation from nearest integer values.

### compute_local_complexity `def compute_local_complexity(model, config)`
- Defined: `plank.py:445`
- Doc: Compute local complexity based on active parameters.

### compute_all_metrics `def compute_all_metrics(model, config)`
- Defined: `plank.py:464`
- Doc: Compute all crystallographic metrics at once.

### __init__ `def __init__(self, model, config)`
- Defined: `plank.py:486`

### test_gauge_invariance `def test_gauge_invariance(self)`
- Defined: `plank.py:490`
- Doc: Test if model exhibits true Strassen structure through permutation invariance.

### _compute_functional_error `def _compute_functional_error(self, test_coeffs)`
- Defined: `plank.py:531`
- Doc: Compute functional error between original and permuted coefficients.

### __init__ `def __init__(self, model, config)`
- Defined: `plank.py:563`

### measure_resilience_spectrum `def measure_resilience_spectrum(self)`
- Defined: `plank.py:570`
- Doc: Measure resilience across multiple noise levels.

### _test_noise_recovery `def _test_noise_recovery(self, sigma)`
- Defined: `plank.py:585`
- Doc: Test recovery from noise level sigma.

### _apply_noise `def _apply_noise(self, sigma)`
- Defined: `plank.py:614`
- Doc: Apply Gaussian noise to model parameters.

### _anneal_to_attractor `def _anneal_to_attractor(self)`
- Defined: `plank.py:621`
- Doc: Anneal model back to attractor using fine-tuning.

### _estimate_critical_noise `def _estimate_critical_noise(self, results)`
- Defined: `plank.py:654`
- Doc: Estimate critical noise level where success rate drops below 50%.

### __init__ `def __init__(self, metrics, diffraction_results, resilience_results, config)`
- Defined: `plank.py:680`

### compute `def compute(self)`
- Defined: `plank.py:692`
- Doc: Compute normalized purity index and grade.

### _assign_grade `def _assign_grade(self, index, delta)`
- Defined: `plank.py:745`
- Doc: Assign crystallographic grade based on delta (primary indicator).

### __init__ `def __init__(self, metrics, training_metrics, config)`
- Defined: `plank.py:770`

### calculate_all `def calculate_all(self)`
- Defined: `plank.py:789`
- Doc: Execute all Planck constant calculation methods.

### _determine_regime_and_weights `def _determine_regime_and_weights(self)`
- Defined: `plank.py:875`
- Doc: Determine confinement regime and corresponding weights.

### _compute_derived_constants `def _compute_derived_constants(self, h_bar)`
- Defined: `plank.py:886`
- Doc: Compute derived Planck-scale constants.

### _compute_universe_comparison `def _compute_universe_comparison(self, h_bar)`
- Defined: `plank.py:917`
- Doc: Compare calculated constants with physical universe.

### __init__ `def __init__(self, config)`
- Defined: `plank.py:939`

### load `def load(self, checkpoint_path, device)`
- Defined: `plank.py:943`
- Doc: Load checkpoint into model instance.

### extract_training_metrics `def extract_training_metrics(self, checkpoint_path)`
- Defined: `plank.py:986`
- Doc: Extract training metrics from checkpoint if available.

### __init__ `def __init__(self, config)`
- Defined: `plank.py:1021`

### analyze_checkpoint `def analyze_checkpoint(self, checkpoint_path, device)`
- Defined: `plank.py:1025`
- Doc: Perform complete analysis of a single checkpoint.

### analyze_directory `def analyze_directory(self, directory, device, pattern)`
- Defined: `plank.py:1104`
- Doc: Analyze all checkpoints in a directory.

### _print_summary `def _print_summary(self, report)`
- Defined: `plank.py:1146`
- Doc: Print formatted summary of analysis results.

### __init__ `def __init__(self, config)`
- Defined: `plank.py:1168`

### save_json_report `def save_json_report(self, report, suffix)`
- Defined: `plank.py:1173`
- Doc: Save individual report as JSON.

### save_aggregate_report `def save_aggregate_report(self, results)`
- Defined: `plank.py:1185`
- Doc: Save aggregate report from multiple analyses.

### _compute_statistics `def _compute_statistics(self, summaries)`
- Defined: `plank.py:1216`
- Doc: Compute aggregate statistics from summaries.

### _count_grades `def _count_grades(self, summaries)`
- Defined: `plank.py:1247`
- Doc: Count distribution of grades.

### generate_visualizations `def generate_visualizations(self, results)`
- Defined: `plank.py:1255`
- Doc: Generate visualization plots.

## purity_index.py

### main `def main()`
- Defined: `purity_index.py:611`

### get_coefficients `def get_coefficients(self)`
- Defined: `purity_index.py:45`

### calculate `def calculate(self, model)`
- Defined: `purity_index.py:50`

### calculate `def calculate(self, loss_history)`
- Defined: `purity_index.py:55`

### classify `def classify(self, alpha, temperature)`
- Defined: `purity_index.py:60`

### analyze_polycrystal `def analyze_polycrystal(self, model, pruning_level)`
- Defined: `purity_index.py:65`

### compare `def compare(self, original, polycrystal)`
- Defined: `purity_index.py:70`

### __init__ `def __init__(self, hidden_dim, matrix_size)`
- Defined: `purity_index.py:74`

### _initialize `def _initialize(self)`
- Defined: `purity_index.py:85`

### forward `def forward(self, a, b)`
- Defined: `purity_index.py:90`

### get_coefficients `def get_coefficients(self)`
- Defined: `purity_index.py:93`

### __init__ `def __init__(self, config)`
- Defined: `purity_index.py:102`

### calculate `def calculate(self, model)`
- Defined: `purity_index.py:105`

### _compute_layer_purity `def _compute_layer_purity(self, weights)`
- Defined: `purity_index.py:134`

### _delta_to_alpha `def _delta_to_alpha(self, delta)`
- Defined: `purity_index.py:140`

### _assess_purity_quality `def _assess_purity_quality(self, alpha, variance)`
- Defined: `purity_index.py:145`

### __init__ `def __init__(self, config)`
- Defined: `purity_index.py:157`

### calculate `def calculate(self, loss_history)`
- Defined: `purity_index.py:160`

### __init__ `def __init__(self, config)`
- Defined: `purity_index.py:200`

### classify `def classify(self, alpha, temperature)`
- Defined: `purity_index.py:203`

### classify_polycrystal_state `def classify_polycrystal_state(self, original_alpha, original_temp, poly_alpha, poly_temp)`
- Defined: `purity_index.py:219`

### __init__ `def __init__(self, config)`
- Defined: `purity_index.py:237`

### analyze_polycrystal `def analyze_polycrystal(self, model, pruning_level, loss_history)`
- Defined: `purity_index.py:243`

### _prune_model `def _prune_model(self, model, sparsity)`
- Defined: `purity_index.py:264`

### __init__ `def __init__(self, config)`
- Defined: `purity_index.py:275`

### compare `def compare(self, original, polycrystal)`
- Defined: `purity_index.py:279`

### migrate `def migrate(self, raw_data, device)`
- Defined: `purity_index.py:312`

### _migrate_dict `def _migrate_dict(self, state_dict, device)`
- Defined: `purity_index.py:322`

### _migrate_custom_format `def _migrate_custom_format(self, state_dict, device)`
- Defined: `purity_index.py:331`

### _migrate_coefs_format `def _migrate_coefs_format(self, state_dict)`
- Defined: `purity_index.py:350`

### _migrate_standard_format `def _migrate_standard_format(self, state_dict)`
- Defined: `purity_index.py:357`

### __init__ `def __init__(self, checkpoint_path, config)`
- Defined: `purity_index.py:362`

### _load_checkpoint `def _load_checkpoint(self)`
- Defined: `purity_index.py:375`

### analyze `def analyze(self)`
- Defined: `purity_index.py:399`

### _print_report `def _print_report(self, results)`
- Defined: `purity_index.py:445`

### __init__ `def __init__(self, config)`
- Defined: `purity_index.py:493`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `purity_index.py:496`

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `purity_index.py:510`

### generate_summary `def generate_summary(self, all_results, output_dir)`
- Defined: `purity_index.py:537`

### _generate_text_report `def _generate_text_report(self, summary, output_dir)`
- Defined: `purity_index.py:574`

## repor_experiments.py

### discretize_q `def discretize_q(w)`
- Defined: `repor_experiments.py:173`

### compute_delta `def compute_delta(U, V, W)`
- Defined: `repor_experiments.py:176`

### phase2 `def phase2(model)`
- Defined: `repor_experiments.py:181`
- Doc: Poda a target_rank slots, discretiza a {-1,0,1}, verifica.

### _verify_2x2 `def _verify_2x2(U, V, W, n)`
- Defined: `repor_experiments.py:204`

### zero_shot_verify `def zero_shot_verify(U, V, W, sizes)`
- Defined: `repor_experiments.py:214`

### _recursive_strassen `def _recursive_strassen(U, V, W, n, trials)`
- Defined: `repor_experiments.py:222`

### _strassen_rec `def _strassen_rec(A, B, U, V, W, n)`
- Defined: `repor_experiments.py:234`

### compute_kappa `def compute_kappa(model, num_batches, bs)`
- Defined: `repor_experiments.py:262`

### compute_alpha `def compute_alpha(delta)`
- Defined: `repor_experiments.py:279`

### compute_teff `def compute_teff(model, num_batches, bs)`
- Defined: `repor_experiments.py:284`

### classify_phase `def classify_phase(delta)`
- Defined: `repor_experiments.py:298`

### load_checkpoint `def load_checkpoint(path, device)`
- Defined: `repor_experiments.py:309`

### _extract_state `def _extract_state(d)`
- Defined: `repor_experiments.py:329`

### train_model `def train_model(cfg, model, epochs, bs, wd, lr, callback)`
- Defined: `repor_experiments.py:351`

### analyze_checkpoint `def analyze_checkpoint(path, device)`
- Defined: `repor_experiments.py:378`

### experiment1 `def experiment1(cfg)`
- Defined: `repor_experiments.py:428`

### experiment2 `def experiment2(cfg)`
- Defined: `repor_experiments.py:475`

### experiment3 `def experiment3(cfg)`
- Defined: `repor_experiments.py:507`

### experiment4 `def experiment4(cfg)`
- Defined: `repor_experiments.py:563`

### experiment5 `def experiment5(cfg)`
- Defined: `repor_experiments.py:607`

### _test_accuracy `def _test_accuracy(model, n, device)`
- Defined: `repor_experiments.py:633`

### _random_prune `def _random_prune(model, fraction)`
- Defined: `repor_experiments.py:644`

### _boundary_prune `def _boundary_prune(model, fraction)`
- Defined: `repor_experiments.py:655`

### analyze_checkpoints `def analyze_checkpoints(ckpt_dir, device)`
- Defined: `repor_experiments.py:665`

### main `def main()`
- Defined: `repor_experiments.py:705`

### _save `def _save(data, path)`
- Defined: `repor_experiments.py:783`

### __init__ `def __init__(self, cfg)`
- Defined: `repor_experiments.py:142`

### forward `def forward(self, A, B)`
- Defined: `repor_experiments.py:149`

### slot_importance `def slot_importance(self)`
- Defined: `repor_experiments.py:156`

### get_weights `def get_weights(self)`
- Defined: `repor_experiments.py:163`

### get_flat `def get_flat(self)`
- Defined: `repor_experiments.py:167`

### cb `def cb(ep, m, loss, acc)`
- Defined: `repor_experiments.py:434`

## scrodingger.py

### main `def main()`
- Defined: `scrodingger.py:779`

### get_coefficients `def get_coefficients(self)`
- Defined: `scrodingger.py:45`

### extract `def extract(self, model)`
- Defined: `scrodingger.py:50`

### calculate `def calculate(self, weights)`
- Defined: `scrodingger.py:55`

### construct `def construct(self, potential, mass)`
- Defined: `scrodingger.py:60`

### solve `def solve(self, hamiltonian, count)`
- Defined: `scrodingger.py:65`

### evolve `def evolve(self, initial_state, hamiltonian, time_steps, dt)`
- Defined: `scrodingger.py:70`

### calculate `def calculate(self, wave_function, operator)`
- Defined: `scrodingger.py:75`

### calculate `def calculate(self, wave_function, position_grid)`
- Defined: `scrodingger.py:80`

### load `def load(self, path, device)`
- Defined: `scrodingger.py:85`

### migrate `def migrate(self, raw_data)`
- Defined: `scrodingger.py:90`

### __init__ `def __init__(self, hidden_dim, matrix_size)`
- Defined: `scrodingger.py:94`

### _initialize `def _initialize(self)`
- Defined: `scrodingger.py:105`

### forward `def forward(self, a, b)`
- Defined: `scrodingger.py:110`

### get_coefficients `def get_coefficients(self)`
- Defined: `scrodingger.py:113`

### extract `def extract(self, model)`
- Defined: `scrodingger.py:122`

### __init__ `def __init__(self, config)`
- Defined: `scrodingger.py:129`

### calculate `def calculate(self, weights)`
- Defined: `scrodingger.py:132`

### __init__ `def __init__(self, config)`
- Defined: `scrodingger.py:151`

### construct `def construct(self, potential, mass)`
- Defined: `scrodingger.py:154`

### __init__ `def __init__(self, config)`
- Defined: `scrodingger.py:171`

### solve `def solve(self, hamiltonian, count)`
- Defined: `scrodingger.py:174`

### __init__ `def __init__(self, config)`
- Defined: `scrodingger.py:195`

### evolve `def evolve(self, initial_state, hamiltonian, time_steps, dt)`
- Defined: `scrodingger.py:198`

### calculate `def calculate(self, wave_function, operator)`
- Defined: `scrodingger.py:217`

### __init__ `def __init__(self, config)`
- Defined: `scrodingger.py:227`

### calculate `def calculate(self, wave_function, position_grid)`
- Defined: `scrodingger.py:230`

### load `def load(self, path, device)`
- Defined: `scrodingger.py:264`

### migrate `def migrate(self, raw_data)`
- Defined: `scrodingger.py:272`

### _migrate_dict `def _migrate_dict(self, state_dict)`
- Defined: `scrodingger.py:284`

### _migrate_custom_format `def _migrate_custom_format(self, state_dict)`
- Defined: `scrodingger.py:293`

### _migrate_coefs_format `def _migrate_coefs_format(self, state_dict)`
- Defined: `scrodingger.py:309`

### _migrate_standard_format `def _migrate_standard_format(self, state_dict)`
- Defined: `scrodingger.py:316`

### __init__ `def __init__(self, checkpoint_path, config)`
- Defined: `scrodingger.py:321`

### _load_checkpoint `def _load_checkpoint(self)`
- Defined: `scrodingger.py:335`

### analyze `def analyze(self)`
- Defined: `scrodingger.py:357`

### _calculate_tunneling_probability `def _calculate_tunneling_probability(self, potential, wave_function)`
- Defined: `scrodingger.py:451`

### _count_degeneracy `def _count_degeneracy(self, eigenvalues)`
- Defined: `scrodingger.py:465`

### _print_report `def _print_report(self, results)`
- Defined: `scrodingger.py:478`

### __init__ `def __init__(self, config)`
- Defined: `scrodingger.py:543`

### visualize `def visualize(self, data, output_path)`
- Defined: `scrodingger.py:546`

### __init__ `def __init__(self, config)`
- Defined: `scrodingger.py:610`

### process_checkpoint `def process_checkpoint(self, checkpoint_path, output_dir)`
- Defined: `scrodingger.py:614`

### process_directory `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- Defined: `scrodingger.py:628`

### generate_summary `def generate_summary(self, all_results, output_dir)`
- Defined: `scrodingger.py:655`

### _generate_text_report `def _generate_text_report(self, summary, output_dir)`
- Defined: `scrodingger.py:718`

## src/benchmarks/benchmark_final.py

### strassen_hybrid_multiply `def strassen_hybrid_multiply(A, B)`
- Defined: `src/benchmarks/benchmark_final.py:45`
- Doc: Multiply using our Strassen Hybrid implementation

### numpy_multiply `def numpy_multiply(A, B)`
- Defined: `src/benchmarks/benchmark_final.py:60`
- Doc: Standard NumPy BLAS multiplication

### benchmark `def benchmark(func, A, B, warmup, runs)`
- Defined: `src/benchmarks/benchmark_final.py:64`
- Doc: Run benchmark with warmup

### main `def main()`
- Defined: `src/benchmarks/benchmark_final.py:80`

## src/benchmarks/benchmark_scientific.py

### strassen_multiply `def strassen_multiply(A, B)`
- Defined: `src/benchmarks/benchmark_scientific.py:38`

### standard_avx512_multiply `def standard_avx512_multiply(A, B)`
- Defined: `src/benchmarks/benchmark_scientific.py:51`

### numpy_multiply `def numpy_multiply(A, B)`
- Defined: `src/benchmarks/benchmark_scientific.py:64`

### benchmark_function `def benchmark_function(func, A, B, runs, warmup)`
- Defined: `src/benchmarks/benchmark_scientific.py:67`
- Doc: Benchmark with statistical analysis

### main `def main()`
- Defined: `src/benchmarks/benchmark_scientific.py:91`

## src/benchmarks/benchmark_strassen.py

### load_config `def load_config(config_path)`
- Defined: `src/benchmarks/benchmark_strassen.py:61`
- Doc: Load configuration from TOML file.

### get_dtype `def get_dtype(dtype_str)`
- Defined: `src/benchmarks/benchmark_strassen.py:93`
- Doc: Convert dtype string to torch dtype.

### estimate_memory_mb `def estimate_memory_mb(n, dtype, batch_size)`
- Defined: `src/benchmarks/benchmark_strassen.py:104`
- Doc: Estimate memory usage for matrix multiplication.

### benchmark_resolution `def benchmark_resolution(n, cfg, dtype)`
- Defined: `src/benchmarks/benchmark_strassen.py:114`
- Doc: Benchmark Strassen vs standard matmul for given resolution.

### run_benchmark `def run_benchmark(cfg)`
- Defined: `src/benchmarks/benchmark_strassen.py:218`
- Doc: Run full benchmark suite.

### save_results `def save_results(results, filepath)`
- Defined: `src/benchmarks/benchmark_strassen.py:310`
- Doc: Save benchmark results to JSON.

### main `def main()`
- Defined: `src/benchmarks/benchmark_strassen.py:322`

## src/benchmarks/strassen_numpy.py

### _load_weights `def _load_weights()`
- Defined: `src/benchmarks/strassen_numpy.py:19`

### strassen_2x2_numpy `def strassen_2x2_numpy(A, B)`
- Defined: `src/benchmarks/strassen_numpy.py:29`
- Doc: Strassen 2x2 using grokked coefficients.

### strassen_numpy `def strassen_numpy(A, B)`
- Defined: `src/benchmarks/strassen_numpy.py:44`
- Doc: Recursive Strassen using NumPy.

### strassen_hybrid `def strassen_hybrid(A, B, threshold)`
- Defined: `src/benchmarks/strassen_numpy.py:78`
- Doc: Hybrid Strassen: use Strassen for large matrices, NumPy for small.

### multiplication_count `def multiplication_count(n)`
- Defined: `src/benchmarks/strassen_numpy.py:112`
- Doc: Count multiplications used by Strassen.

## src/discovery/auto_T_discovery.py

### verify_strassen_T `def verify_strassen_T(model_path, target_sizes)`
- Defined: `src/discovery/auto_T_discovery.py:306`
- Doc: Verify T discovery on Strassen model

### verify_expanded_correctness `def verify_expanded_correctness(U, V, W, target_size, expanded)`
- Defined: `src/discovery/auto_T_discovery.py:353`
- Doc: Verify that expanded operator correctly computes matrix multiplication

### recursive_strassen_multiply `def recursive_strassen_multiply(A, B, U, V, W, base_size)`
- Defined: `src/discovery/auto_T_discovery.py:389`
- Doc: Recursively apply learned Strassen decomposition

### __init__ `def __init__(self, tolerance, verbose)`
- Defined: `src/discovery/auto_T_discovery.py:42`

### analyze_structure `def analyze_structure(self, W)`
- Defined: `src/discovery/auto_T_discovery.py:46`
- Doc: Phase 1 & 2: Analyze weight matrix structure

### _detect_discrete_values `def _detect_discrete_values(self, W_flat)`
- Defined: `src/discovery/auto_T_discovery.py:89`
- Doc: Detect if weights cluster around discrete values

### _detect_block_structure `def _detect_block_structure(self, W)`
- Defined: `src/discovery/auto_T_discovery.py:105`
- Doc: Detect repeating block patterns

### _block_repetition_score `def _block_repetition_score(self, W, bm, bn)`
- Defined: `src/discovery/auto_T_discovery.py:127`
- Doc: Score how well blocks repeat (lower = more repetitive)

### _detect_symmetry_type `def _detect_symmetry_type(self, W)`
- Defined: `src/discovery/auto_T_discovery.py:144`
- Doc: Detect type of symmetry in weight matrix

### _is_permutation_symmetric `def _is_permutation_symmetric(self, W)`
- Defined: `src/discovery/auto_T_discovery.py:165`
- Doc: Check if rows are permutations of a base pattern

### _is_cyclic `def _is_cyclic(self, W)`
- Defined: `src/discovery/auto_T_discovery.py:171`
- Doc: Check for cyclic/Toeplitz structure

### _invariant_subspace_dim `def _invariant_subspace_dim(self, U, S, rank)`
- Defined: `src/discovery/auto_T_discovery.py:184`
- Doc: Estimate dimension of truly invariant subspace

### _discretization_error `def _discretization_error(self, W, values)`
- Defined: `src/discovery/auto_T_discovery.py:198`
- Doc: Compute error when discretizing to given values

### _print_analysis `def _print_analysis(self, W, S, structure)`
- Defined: `src/discovery/auto_T_discovery.py:213`
- Doc: Print analysis results

### construct_T `def construct_T(self, W_dict, target_size)`
- Defined: `src/discovery/auto_T_discovery.py:229`
- Doc: Phase 3: Construct expansion operator T

### _validate_expansion `def _validate_expansion(self, expanded, structure)`
- Defined: `src/discovery/auto_T_discovery.py:290`
- Doc: Validate that expansion preserves key invariants

## src/native/strassen_c.c

### alloc_matrix `static float* alloc_matrix(int n)`
- Defined: `src/native/strassen_c.c:15`
- Doc: Strassen Matrix Multiplication - C Implementation Author: grisun0  Compila: gcc -O3 -ffast-math -march=native -shared -f

### matmul_standard `static void matmul_standard(float* C, float* A, float* B, int n)`
- Defined: `src/native/strassen_c.c:20`
- Doc: #include <stdlib.h> #include <string.h> #include <stdio.h> #define THRESHOLD 64 /* Allocate matrix static float* alloc_m

### mat_add `static void mat_add(float* C, float* A, float* B, int n)`
- Defined: `src/native/strassen_c.c:33`
- Doc: /* Standard matrix multiplication for small matrices static void matmul_standard(float* C, float* A, float* B, int n) { 

### mat_sub `static void mat_sub(float* C, float* A, float* B, int n)`
- Defined: `src/native/strassen_c.c:41`
- Doc: } } } } /* Add matrices: C = A + B static void mat_add(float* C, float* A, float* B, int n) { int nn = n * n; for (int i

### extract_quadrant `static void extract_quadrant(float* Q, float* M, int n, int row, int col)`
- Defined: `src/native/strassen_c.c:49`
- Doc: for (int i = 0; i < nn; i++) { C[i] = A[i] + B[i]; } } /* Subtract matrices: C = A - B static void mat_sub(float* C, flo

### insert_quadrant `static void insert_quadrant(float* M, float* Q, int n, int row, int col)`
- Defined: `src/native/strassen_c.c:57`
- Doc: for (int i = 0; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Extract quadrant from matrix static void extract_quadrant(floa

### strassen_recursive `void strassen_recursive(float* C, float* A, float* B, int n)`
- Defined: `src/native/strassen_c.c:65`
- Doc: for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant into

### strassen_multiply `void strassen_multiply(float* C, float* A, float* B, int n)`
- Defined: `src/native/strassen_c.c:164`
- Doc: insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C,

### standard_multiply `void standard_multiply(float* C, float* A, float* B, int n)`
- Defined: `src/native/strassen_c.c:169`
- Doc: /* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2

## src/native/strassen_optimal.c

### strassen_level `static void strassen_level(float* C, float* A, float* B, int n, 
                           float...`
- Defined: `src/native/strassen_optimal.c:18`
- Doc: Uses in-place operations where possible and only applies Strassen for very large matrices where the asymptotic advantage

### strassen_optimal `void strassen_optimal(float* C, float* A, float* B, int n)`
- Defined: `src/native/strassen_optimal.c:130`

## src/native/strassen_turbo.c

### alloc_matrix `static inline float* alloc_matrix(int n)`
- Defined: `src/native/strassen_turbo.c:25`
- Doc: Compile: gcc -O3 -ffast-math -march=native -fopenmp -mavx2 -shared -fPIC -o libstrassen_turbo.so strassen_turbo.c  #incl

### mat_add_avx `static void mat_add_avx(float* __restrict C, const float* __restrict A, 
                        ...`
- Defined: `src/native/strassen_turbo.c:30`
- Doc: #include <stdio.h> #include <omp.h> #include <immintrin.h> #define THRESHOLD 128 #define BLOCK_SIZE 32 #define ALIGN 32 

### mat_sub_avx `static void mat_sub_avx(float* __restrict C, const float* __restrict A, 
                        ...`
- Defined: `src/native/strassen_turbo.c:50`
- Doc: for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_a

### matmul_blocked_avx `static void matmul_blocked_avx(float* __restrict C, const float* __restrict A, 
                 ...`
- Defined: `src/native/strassen_turbo.c:68`
- Doc: for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_s

### extract_quadrant `static void extract_quadrant(float* __restrict Q, const float* __restrict M, 
                   ...`
- Defined: `src/native/strassen_turbo.c:104`
- Doc: _mm256_storeu_ps(&C[i * n + j], vc); } for (; j < j_end; j++) { C[i * n + j] += a_ik * B[k * n + j]; } } } } } } } /* Ex

### insert_quadrant `static void insert_quadrant(float* __restrict M, const float* __restrict Q, 
                    ...`
- Defined: `src/native/strassen_turbo.c:114`
- Doc: } } /* Extract quadrant static void extract_quadrant(float* __restrict Q, const float* __restrict M, int n, int row, int

### strassen_turbo_recursive `void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)`
- Defined: `src/native/strassen_turbo.c:124`
- Doc: } } /* Insert quadrant static void insert_quadrant(float* __restrict M, const float* __restrict Q, int n, int row, int c

### strassen_turbo `void strassen_turbo(float* C, float* A, float* B, int n)`
- Defined: `src/native/strassen_turbo.c:261`
- Doc: insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C,

### get_num_threads `int get_num_threads(void)`
- Defined: `src/native/strassen_turbo.c:267`
- Doc: free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); fr

## src/training/convergence_theory.py

### convergence_theorem `def convergence_theorem()`
- Defined: `src/training/convergence_theory.py:193`
- Doc: THEOREM (Convergence to Algorithmic Invariance)

### verify_convergence_conditions `def verify_convergence_conditions(model, loss_fn, train_data, noise_threshold)`
- Defined: `src/training/convergence_theory.py:265`
- Doc: Verify that convergence conditions are satisfied for a trained model.

### __init__ `def __init__(self, model, loss_fn, n_samples, device)`
- Defined: `src/training/convergence_theory.py:40`

### estimate_trace `def estimate_trace(self, data)`
- Defined: `src/training/convergence_theory.py:47`
- Doc: Estimate tr(H) using Hutchinson's stochastic trace estimator.

### _rademacher_vector `def _rademacher_vector(self)`
- Defined: `src/training/convergence_theory.py:73`
- Doc: Generate Rademacher random vector (±1 with equal probability)

### _hessian_vector_product `def _hessian_vector_product(self, x, y, v)`
- Defined: `src/training/convergence_theory.py:90`
- Doc: Compute H @ v using the "double backward" trick.

### compute_kappa_eff `def compute_kappa_eff(self, data)`
- Defined: `src/training/convergence_theory.py:118`
- Doc: Compute κ_eff = -tr(H) / N

### __init__ `def __init__(self, model, loss_fn)`
- Defined: `src/training/convergence_theory.py:144`

### estimate_noise `def estimate_noise(self, data_loader, n_batches, n_threads)`
- Defined: `src/training/convergence_theory.py:148`
- Doc: Estimate hardware noise by measuring gradient variance across batches

### __init__ `def __init__(self, rank)`
- Defined: `src/training/convergence_theory.py:348`

### forward `def forward(self, x)`
- Defined: `src/training/convergence_theory.py:354`

## src/training/grokkit_physics.py

### strassen_multiply `def strassen_multiply(A, B)`
- Defined: `src/training/grokkit_physics.py:49`
- Doc: Wrapper for Strassen multiplication (uses float32).

### measure_physics `def measure_physics(N, num_samples)`
- Defined: `src/training/grokkit_physics.py:64`
- Doc: Measure the 'physical quantities' for a given matrix size.

### detect_phase_transition `def detect_phase_transition(results)`
- Defined: `src/training/grokkit_physics.py:126`
- Doc: Find the critical size N_c where the phase transition occurs.

### main `def main()`
- Defined: `src/training/grokkit_physics.py:148`

## src/training/main.py

### set_seed `def set_seed(seed)`
- Defined: `src/training/main.py:79`
- Doc: Fijar semilla para reproducibilidad.

### main `def main()`
- Defined: `src/training/main.py:345`

### __init__ `def __init__(self, num_slots)`
- Defined: `src/training/main.py:96`

### forward `def forward(self, A, B)`
- Defined: `src/training/main.py:108`
- Doc: Forward pass con multiplicación matemática pura.

### get_slot_norms `def get_slot_norms(self)`
- Defined: `src/training/main.py:155`
- Doc: Norma promedio de cada slot.

### get_active_slots `def get_active_slots(self)`
- Defined: `src/training/main.py:160`
- Doc: Número de slots activos.

### mask_slot `def mask_slot(self, slot_idx)`
- Defined: `src/training/main.py:164`
- Doc: Desactiva un slot.

### get_weakest_slot `def get_weakest_slot(self)`
- Defined: `src/training/main.py:169`
- Doc: Slot con menor norma entre los activos.

### print_coefficients `def print_coefficients(self)`
- Defined: `src/training/main.py:175`
- Doc: Muestra coeficientes descubiertos.

### __init__ `def __init__(self, num_samples, seed)`
- Defined: `src/training/main.py:202`

### __len__ `def __len__(self)`
- Defined: `src/training/main.py:210`

### __getitem__ `def __getitem__(self, idx)`
- Defined: `src/training/main.py:213`

### __init__ `def __init__(self, config)`
- Defined: `src/training/main.py:220`

### accuracy `def accuracy(self, pred, target)`
- Defined: `src/training/main.py:242`

### train_epoch `def train_epoch(self, optimizer)`
- Defined: `src/training/main.py:245`

### evaluate `def evaluate(self)`
- Defined: `src/training/main.py:265`

### train `def train(self)`
- Defined: `src/training/main.py:280`

## src/training/main_pure_math.py

### gen_data `def gen_data(n, scale)`
- Defined: `src/training/main_pure_math.py:58`

### train `def train(model, epochs, lr, l1, batch, verbose)`
- Defined: `src/training/main_pure_math.py:64`

### verify `def verify(model, n)`
- Defined: `src/training/main_pure_math.py:88`

### hard_prune `def hard_prune(model, keep)`
- Defined: `src/training/main_pure_math.py:106`
- Doc: Poda los slots más débiles, mantiene top-k.

### refine_pruned `def refine_pruned(model, active, epochs, lr)`
- Defined: `src/training/main_pure_math.py:122`
- Doc: Refina manteniendo slots podados en cero.

### show_coeffs `def show_coeffs(model, active)`
- Defined: `src/training/main_pure_math.py:159`

### main `def main()`
- Defined: `src/training/main_pure_math.py:186`

### __init__ `def __init__(self, rank)`
- Defined: `src/training/main_pure_math.py:28`

### forward `def forward(self, A, B)`
- Defined: `src/training/main_pure_math.py:35`

### slot_norms `def slot_norms(self)`
- Defined: `src/training/main_pure_math.py:47`
- Doc: Norma combinada de cada slot.

### active_count `def active_count(self, thresh)`
- Defined: `src/training/main_pure_math.py:54`

## src/training/strassen_core.py

### _load_weights `def _load_weights()`
- Defined: `src/training/strassen_core.py:14`

### strassen_2x2 `def strassen_2x2(A, B)`
- Defined: `src/training/strassen_core.py:21`

### strassen `def strassen(X, Y)`
- Defined: `src/training/strassen_core.py:44`

### get_coefficients `def get_coefficients()`
- Defined: `src/training/strassen_core.py:77`

### multiplication_count `def multiplication_count(n)`
- Defined: `src/training/strassen_core.py:82`

## src/training/strassen_grokkit.py

### generate_batch `def generate_batch(n, scale)`
- Defined: `src/training/strassen_grokkit.py:113`
- Doc: Genera batch de matrices aleatorias.

### train_grokkit `def train_grokkit(epochs, batch_size, lr, wd)`
- Defined: `src/training/strassen_grokkit.py:120`
- Doc: Entrena usando el framework Grokkit.

### verify_grokking `def verify_grokking(model, n_test)`
- Defined: `src/training/strassen_grokkit.py:203`
- Doc: Verifica que el operador ha grokkeado correctamente.

### progressive_sparsification `def progressive_sparsification(model, target_slots)`
- Defined: `src/training/strassen_grokkit.py:254`
- Doc: Fase 2: Esparsificación progresiva.

### main `def main()`
- Defined: `src/training/strassen_grokkit.py:351`
- Doc: Pipeline principal Grokkit para Strassen.

### __init__ `def __init__(self, rank)`
- Defined: `src/training/strassen_grokkit.py:40`

### forward `def forward(self, A, B)`
- Defined: `src/training/strassen_grokkit.py:49`
- Doc: Computa A @ B usando la descomposición tensorial.

### compute_LC `def compute_LC(self)`
- Defined: `src/training/strassen_grokkit.py:67`
- Doc: Linear Combination metric.

### compute_SP `def compute_SP(self)`
- Defined: `src/training/strassen_grokkit.py:83`
- Doc: Sparsity metric.

### slot_importance `def slot_importance(self)`
- Defined: `src/training/strassen_grokkit.py:101`
- Doc: Importancia de cada slot basada en normas.

### count_active `def count_active(self, threshold)`
- Defined: `src/training/strassen_grokkit.py:108`
- Doc: Cuenta slots activos.

## src/training/train_strassen.py

### generate_batch `def generate_batch(n, scale)`
- Defined: `src/training/train_strassen.py:60`

### train_phase1 `def train_phase1(epochs, batch_size, lr, wd)`
- Defined: `src/training/train_strassen.py:66`
- Doc: Phase 1: Grokking with Weight Decay as thermodynamic pressure.

### sparsify `def sparsify(model, target_slots)`
- Defined: `src/training/train_strassen.py:104`
- Doc: Phase 2: Progressive sparsification to target rank.

### discretize `def discretize(model, slots_to_prune)`
- Defined: `src/training/train_strassen.py:183`
- Doc: Phase 3: Discretize coefficients to {-1, 0, 1}.

### get_canonical_strassen `def get_canonical_strassen()`
- Defined: `src/training/train_strassen.py:211`
- Doc: Returns the canonical Strassen coefficients.

### verify `def verify(U, V, W, n_test)`
- Defined: `src/training/train_strassen.py:260`
- Doc: Verify the discretized operator.

### main `def main()`
- Defined: `src/training/train_strassen.py:299`
- Doc: Main training pipeline.

### __init__ `def __init__(self, rank)`
- Defined: `src/training/train_strassen.py:33`

### forward `def forward(self, A, B)`
- Defined: `src/training/train_strassen.py:40`

### slot_importance `def slot_importance(self)`
- Defined: `src/training/train_strassen.py:50`

### count_active `def count_active(self, threshold)`
- Defined: `src/training/train_strassen.py:56`

## superposition.py

### main `def main()`
- Defined: `superposition.py:765`

### __post_init__ `def __post_init__(self)`
- Defined: `superposition.py:54`

### load_checkpoint `def load_checkpoint(self, path, device)`
- Defined: `superposition.py:60`

### compute `def compute(self)`
- Defined: `superposition.py:64`

### analyze_checkpoint `def analyze_checkpoint(self, checkpoint_path)`
- Defined: `superposition.py:68`

### load_checkpoint `def load_checkpoint(self, path, device)`
- Defined: `superposition.py:78`

### detect_hidden_dim `def detect_hidden_dim(raw_data)`
- Defined: `superposition.py:89`
- Doc: Detect hidden dimension from checkpoint data structure by inspecting

### migrate_checkpoint `def migrate_checkpoint(raw_data)`
- Defined: `superposition.py:122`

### _migrate_dict `def _migrate_dict(state_dict)`
- Defined: `superposition.py:135`

### _migrate_custom_format `def _migrate_custom_format(state_dict)`
- Defined: `superposition.py:147`

### _migrate_coefs_format `def _migrate_coefs_format(state_dict)`
- Defined: `superposition.py:162`

### _migrate_encoder_format `def _migrate_encoder_format(state_dict)`
- Defined: `superposition.py:170`
- Doc: Handle encoder-based format from specific experimental architectures.

### _migrate_standard_format `def _migrate_standard_format(state_dict)`
- Defined: `superposition.py:215`

### __init__ `def __init__(self, config)`
- Defined: `superposition.py:223`

### generate_batch `def generate_batch(self, batch_size)`
- Defined: `superposition.py:226`
- Doc: Generate batch of matrix pairs and their product.

### generate_dataset `def generate_dataset(self, num_samples)`
- Defined: `superposition.py:240`
- Doc: Generate full dataset.

### __init__ `def __init__(self, config)`
- Defined: `superposition.py:261`

### _initialize_symmetric `def _initialize_symmetric(self)`
- Defined: `superposition.py:271`

### forward `def forward(self, a, b)`
- Defined: `superposition.py:276`
- Doc: Forward pass returning output and bottleneck activations.

### get_coefficients `def get_coefficients(self)`
- Defined: `superposition.py:284`

### __init__ `def __init__(self, config)`
- Defined: `superposition.py:298`

### encode `def encode(self, x)`
- Defined: `superposition.py:310`
- Doc: Encode bottleneck activations to sparse features.

### decode `def decode(self, z)`
- Defined: `superposition.py:319`
- Doc: Decode sparse features back to bottleneck.

### forward `def forward(self, x)`
- Defined: `superposition.py:328`

### __init__ `def __init__(self, config)`
- Defined: `superposition.py:339`

### compute_feature_probabilities `def compute_feature_probabilities(self, sae_activations)`
- Defined: `superposition.py:342`
- Doc: Calculate feature probabilities from SAE activations.

### compute_entropy `def compute_entropy(self, probabilities)`
- Defined: `superposition.py:352`
- Doc: Shannon entropy H(p) = -Σ p_i log p_i.

### compute_superposition `def compute_superposition(self, sae_activations)`
- Defined: `superposition.py:359`
- Doc: Main metric: ψ = F/N where F = e^{H(p)}.

### compute_frobenius_metric `def compute_frobenius_metric(self, weight_matrix)`
- Defined: `superposition.py:377`
- Doc: Baseline from Eq 2: ψ_Frob = ||W||_F^2 / N.

### compute_interference_matrix `def compute_interference_matrix(self, weight_matrix)`
- Defined: `superposition.py:385`
- Doc: Compute W^T @ W to analyze interference patterns.

### compute `def compute(self, sae_activations, weight_matrix)`
- Defined: `superposition.py:389`
- Doc: Unified interface.

### __init__ `def __init__(self, sae, config)`
- Defined: `superposition.py:410`

### train `def train(self, bottleneck_activations)`
- Defined: `superposition.py:421`
- Doc: Train SAE on extracted activations.

### __init__ `def __init__(self, config)`
- Defined: `superposition.py:480`

### load_model `def load_model(self, checkpoint_path)`
- Defined: `superposition.py:496`
- Doc: Load and migrate checkpoint to model.

### extract_bottleneck_activations `def extract_bottleneck_activations(self, model)`
- Defined: `superposition.py:554`
- Doc: Extract bottleneck activations (U(a) * V(b)) from model.

### analyze_checkpoint `def analyze_checkpoint(self, checkpoint_path)`
- Defined: `superposition.py:571`
- Doc: Full analysis pipeline for a single checkpoint:

### _save_intermediate_result `def _save_intermediate_result(self, result, name)`
- Defined: `superposition.py:633`
- Doc: Save result for individual checkpoint.

### analyze_directory `def analyze_directory(self, checkpoint_dir)`
- Defined: `superposition.py:641`
- Doc: Analyze all checkpoints in directory.

### _save_progress_checkpoint `def _save_progress_checkpoint(self, results)`
- Defined: `superposition.py:680`
- Doc: Save intermediate progress.

### _save_final_results `def _save_final_results(self, results)`
- Defined: `superposition.py:687`
- Doc: Save complete results.

### _generate_comparison_plots `def _generate_comparison_plots(self, results)`
- Defined: `superposition.py:694`
- Doc: Generate comparison plots across checkpoints.

### get_tensor `def get_tensor(key)`
- Defined: `superposition.py:148`

## train_batch_sweep.py

### train_for_batch_size `def train_for_batch_size(B, seed, output_dir)`
- Defined: `train_batch_sweep.py:11`

## unified_hidden_connections_suite.py

### main `def main()`
- Defined: `unified_hidden_connections_suite.py:1182`

### __post_init__ `def __post_init__(self)`
- Defined: `unified_hidden_connections_suite.py:76`

### __init__ `def __init__(self, config)`
- Defined: `unified_hidden_connections_suite.py:190`

### forward `def forward(self, A, B)`
- Defined: `unified_hidden_connections_suite.py:203`

### get_coefficients `def get_coefficients(self)`
- Defined: `unified_hidden_connections_suite.py:213`

### get_flat_parameters `def get_flat_parameters(self)`
- Defined: `unified_hidden_connections_suite.py:216`

### slot_importance `def slot_importance(self)`
- Defined: `unified_hidden_connections_suite.py:219`

### count_active_slots `def count_active_slots(self, threshold)`
- Defined: `unified_hidden_connections_suite.py:225`

### generate_batch `def generate_batch(self, batch_size)`
- Defined: `unified_hidden_connections_suite.py:232`

### __init__ `def __init__(self, config)`
- Defined: `unified_hidden_connections_suite.py:238`

### generate_batch `def generate_batch(self, batch_size)`
- Defined: `unified_hidden_connections_suite.py:241`

### save `def save(self, model, epoch, metrics, path)`
- Defined: `unified_hidden_connections_suite.py:265`

### load `def load(self, path, model)`
- Defined: `unified_hidden_connections_suite.py:266`

### save `def save(self, model, epoch, metrics, path)`
- Defined: `unified_hidden_connections_suite.py:272`

### load `def load(self, path, model)`
- Defined: `unified_hidden_connections_suite.py:284`

### train `def train(self, model, epochs, callback)`
- Defined: `unified_hidden_connections_suite.py:296`

### __init__ `def __init__(self, model_config, training_config, data_generator)`
- Defined: `unified_hidden_connections_suite.py:304`

### train `def train(self, model, epochs, callback)`
- Defined: `unified_hidden_connections_suite.py:314`

### calculate `def calculate(self, model)`
- Defined: `unified_hidden_connections_suite.py:361`

### __init__ `def __init__(self, config, tolerance)`
- Defined: `unified_hidden_connections_suite.py:368`

### _build_hessian_approximation `def _build_hessian_approximation(self, model)`
- Defined: `unified_hidden_connections_suite.py:372`

### calculate `def calculate(self, model)`
- Defined: `unified_hidden_connections_suite.py:384`

### __init__ `def __init__(self, config, regularization)`
- Defined: `unified_hidden_connections_suite.py:418`

### _compute_hessian `def _compute_hessian(self, model)`
- Defined: `unified_hidden_connections_suite.py:422`

### _generate_single_sample `def _generate_single_sample(self)`
- Defined: `unified_hidden_connections_suite.py:433`

### _loss_from_flat `def _loss_from_flat(self, flat_params, model, original_params, A, B, C_true)`
- Defined: `unified_hidden_connections_suite.py:437`

### _diagonal_hessian_approximation `def _diagonal_hessian_approximation(self, model, A, B, C_true)`
- Defined: `unified_hidden_connections_suite.py:456`

### calculate `def calculate(self, model)`
- Defined: `unified_hidden_connections_suite.py:470`

### __init__ `def __init__(self, config, noise_floor)`
- Defined: `unified_hidden_connections_suite.py:495`

### calculate `def calculate(self, model)`
- Defined: `unified_hidden_connections_suite.py:499`

### __init__ `def __init__(self, model_config, expansion_factor, l1_coefficient, sae_lr, sae_epochs, sae_batch_size, num_samples, epsilon)`
- Defined: `unified_hidden_connections_suite.py:555`

### _extract_activations `def _extract_activations(self, model)`
- Defined: `unified_hidden_connections_suite.py:576`

### _train_sae `def _train_sae(self, activations)`
- Defined: `unified_hidden_connections_suite.py:589`

### _sae_forward `def _sae_forward(self, x, W_enc, b_enc, b_dec)`
- Defined: `unified_hidden_connections_suite.py:616`

### calculate `def calculate(self, model)`
- Defined: `unified_hidden_connections_suite.py:623`

### run `def run(self, model)`
- Defined: `unified_hidden_connections_suite.py:649`

### get_name `def get_name(self)`
- Defined: `unified_hidden_connections_suite.py:653`

### __init__ `def __init__(self, config, model_config, training_config, data_generator, checkpoint_manager)`
- Defined: `unified_hidden_connections_suite.py:663`

### get_name `def get_name(self)`
- Defined: `unified_hidden_connections_suite.py:679`

### run `def run(self, model)`
- Defined: `unified_hidden_connections_suite.py:682`

### _analyze_temporal_correlation `def _analyze_temporal_correlation(self, results)`
- Defined: `unified_hidden_connections_suite.py:717`

### __init__ `def __init__(self, config, model_config, data_generator)`
- Defined: `unified_hidden_connections_suite.py:741`

### get_name `def get_name(self)`
- Defined: `unified_hidden_connections_suite.py:752`

### run `def run(self, model)`
- Defined: `unified_hidden_connections_suite.py:755`

### _create_gamma_model `def _create_gamma_model(self, base_model, gamma)`
- Defined: `unified_hidden_connections_suite.py:778`

### _train_gamma_model `def _train_gamma_model(self, model)`
- Defined: `unified_hidden_connections_suite.py:790`

### _detect_critical_transition `def _detect_critical_transition(self, results)`
- Defined: `unified_hidden_connections_suite.py:800`

### __init__ `def __init__(self, config, model_config, data_generator)`
- Defined: `unified_hidden_connections_suite.py:826`

### get_name `def get_name(self)`
- Defined: `unified_hidden_connections_suite.py:836`

### run `def run(self, model)`
- Defined: `unified_hidden_connections_suite.py:839`

### _apply_moebius `def _apply_moebius(self, A, B)`
- Defined: `unified_hidden_connections_suite.py:869`

### _apply_moebius_to_output `def _apply_moebius_to_output(self, C)`
- Defined: `unified_hidden_connections_suite.py:879`

### __init__ `def __init__(self, config, model_config, data_generator)`
- Defined: `unified_hidden_connections_suite.py:893`

### get_name `def get_name(self)`
- Defined: `unified_hidden_connections_suite.py:904`

### run `def run(self, model)`
- Defined: `unified_hidden_connections_suite.py:907`

### _test_uncertainty_bound `def _test_uncertainty_bound(self, results)`
- Defined: `unified_hidden_connections_suite.py:951`

### prune `def prune(self, model, fraction)`
- Defined: `unified_hidden_connections_suite.py:969`

### prune `def prune(self, model, fraction)`
- Defined: `unified_hidden_connections_suite.py:976`

### prune `def prune(self, model, fraction)`
- Defined: `unified_hidden_connections_suite.py:991`

### __init__ `def __init__(self, config, model_config, data_generator)`
- Defined: `unified_hidden_connections_suite.py:1010`

### get_name `def get_name(self)`
- Defined: `unified_hidden_connections_suite.py:1022`

### run `def run(self, model)`
- Defined: `unified_hidden_connections_suite.py:1025`

### _run_pruning_trials `def _run_pruning_trials(self, base_model, pruner, A, B, C_true)`
- Defined: `unified_hidden_connections_suite.py:1046`

### __init__ `def __init__(self, config)`
- Defined: `unified_hidden_connections_suite.py:1081`

### _build_experiments `def _build_experiments(self)`
- Defined: `unified_hidden_connections_suite.py:1088`

### run_all `def run_all(self)`
- Defined: `unified_hidden_connections_suite.py:1127`

### _aggregate_verdicts `def _aggregate_verdicts(self, all_results)`
- Defined: `unified_hidden_connections_suite.py:1150`

### _serialize_config `def _serialize_config(self)`
- Defined: `unified_hidden_connections_suite.py:1163`

### checkpoint_callback `def checkpoint_callback(epoch, m, loss, acc)`
- Defined: `unified_hidden_connections_suite.py:691`

## xray_tensor_diffractometer.py

### set_seed `def set_seed(seed)`
- Defined: `xray_tensor_diffractometer.py:64`

### setup_logger `def setup_logger(name, level)`
- Defined: `xray_tensor_diffractometer.py:72`

### run_epitaxy_from_best_crystal `def run_epitaxy_from_best_crystal(checkpoint_dir, target_sizes)`
- Defined: `xray_tensor_diffractometer.py:86`
- Doc: Pipeline automático: encuentra el mejor cristal y lo usa como semilla.

### main `def main()`
- Defined: `xray_tensor_diffractometer.py:2683`

### load_checkpoint `def load_checkpoint(self, path, device)`
- Defined: `xray_tensor_diffractometer.py:147`

### compute `def compute(self, model)`
- Defined: `xray_tensor_diffractometer.py:150`

### generate_batch `def generate_batch(self, batch_size)`
- Defined: `xray_tensor_diffractometer.py:153`

### generate_batch `def generate_batch(batch_size)`
- Defined: `xray_tensor_diffractometer.py:168`

### verify_structure `def verify_structure(coeffs)`
- Defined: `xray_tensor_diffractometer.py:179`

### __init__ `def __init__(self, hidden_dim, matrix_size)`
- Defined: `xray_tensor_diffractometer.py:188`

### _initialize_symmetric `def _initialize_symmetric(self)`
- Defined: `xray_tensor_diffractometer.py:199`

### forward `def forward(self, a, b)`
- Defined: `xray_tensor_diffractometer.py:204`

### get_coefficients `def get_coefficients(self)`
- Defined: `xray_tensor_diffractometer.py:207`

### __init__ `def __init__(self, seed_checkpoint_path, target_matrix_size, device)`
- Defined: `xray_tensor_diffractometer.py:224`

### _load_seed_crystal `def _load_seed_crystal(self)`
- Defined: `xray_tensor_diffractometer.py:242`
- Doc: Carga el cristal semilla verificando su pureza

### grow_epitaxial_crystal `def grow_epitaxial_crystal(self)`
- Defined: `xray_tensor_diffractometer.py:265`
- Doc: Crece un cristal epitaxial desde la semilla.

### _adjust_dimensions `def _adjust_dimensions(self, tensor, target_shape)`
- Defined: `xray_tensor_diffractometer.py:326`
- Doc: Ajusta dimensiones del tensor epitaxial para coincidir con el modelo objetivo.

### anneal_crystal `def anneal_crystal(self, model, max_epochs, early_stop_threshold)`
- Defined: `xray_tensor_diffractometer.py:360`
- Doc: Recocido térmico del cristal epitaxial.

### __init__ `def __init__(self, results_dir)`
- Defined: `xray_tensor_diffractometer.py:474`

### run_epitaxial_growth_experiment `def run_epitaxial_growth_experiment(self, seed_checkpoint, target_sizes)`
- Defined: `xray_tensor_diffractometer.py:479`
- Doc: Experimento completo: cultiva cristales de múltiples tamaños desde una semilla.

### _plot_epitaxial_evolution `def _plot_epitaxial_evolution(self, annealing_results, target_size, seed_name)`
- Defined: `xray_tensor_diffractometer.py:556`
- Doc: Visualiza la evolución del cristal durante el recocido

### _generate_comparative_report `def _generate_comparative_report(self, results)`
- Defined: `xray_tensor_diffractometer.py:604`
- Doc: Genera reporte comparativo de todos los experimentos epitaxiales

### helmholtz_free_energy `def helmholtz_free_energy(self)`
- Defined: `xray_tensor_diffractometer.py:680`
- Doc: F = U - T*S (a μ y N constantes)

### gibbs_free_energy `def gibbs_free_energy(self)`
- Defined: `xray_tensor_diffractometer.py:684`
- Doc: G = F + μ*N + P*V (presión algorítmica)

### is_stable `def is_stable(self)`
- Defined: `xray_tensor_diffractometer.py:689`
- Doc: Criterio de estabilidad: dG < 0

### compute_weight_diffraction `def compute_weight_diffraction(coeffs)`
- Defined: `xray_tensor_diffractometer.py:696`

### _compute_spectral_entropy `def _compute_spectral_entropy(power_spectrum)`
- Defined: `xray_tensor_diffractometer.py:719`

### extract_lattice_parameters `def extract_lattice_parameters(weight_tensor, rank)`
- Defined: `xray_tensor_diffractometer.py:726`
- Doc: Extrae parámetros de red preservando la geometría física del tensor.

### compute_gibbs_free_energy `def compute_gibbs_free_energy(loss, temp, entropy)`
- Defined: `xray_tensor_diffractometer.py:785`

### extract_canonical_decomposition `def extract_canonical_decomposition(coeffs, rank)`
- Defined: `xray_tensor_diffractometer.py:791`
- Doc: Descomposición Canónica del tensor tripartito (U, V, W).

### _discretize_to_integers `def _discretize_to_integers(factors)`
- Defined: `xray_tensor_diffractometer.py:847`
- Doc: Proyecta factores continuos a la red cristalina discreta {-1, 0, 1}.

### _check_strassen_equivalence `def _check_strassen_equivalence(discretized_factors)`
- Defined: `xray_tensor_diffractometer.py:868`
- Doc: Verifica si los factores discretizados corresponden a la estructura de Strassen.

### create_superlattice_seed `def create_superlattice_seed(base_tensor, scale_factor)`
- Defined: `xray_tensor_diffractometer.py:892`

### compute_effective_temperature `def compute_effective_temperature(gradient_buffer, learning_rate)`
- Defined: `xray_tensor_diffractometer.py:916`

### compute_critical_exponents `def compute_critical_exponents(temp_history, cv_history, alpha_history)`
- Defined: `xray_tensor_diffractometer.py:930`
- Doc: Calcula exponentes críticos cerca de transiciones de fase.

### compute_equation_of_state `def compute_equation_of_state(temp_eff, alpha, kappa)`
- Defined: `xray_tensor_diffractometer.py:1009`
- Doc: Ecuación de estado: T_c(α) = T_0 * exp(-c*α)

### compute_specific_heat `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- Defined: `xray_tensor_diffractometer.py:1048`

### estimate_hbar_algorithmic `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)`
- Defined: `xray_tensor_diffractometer.py:1061`

### compute_mutual_information `def compute_mutual_information(weights, gradients)`
- Defined: `xray_tensor_diffractometer.py:1069`

### check_extensivity `def check_extensivity(entropy_list, scale_factors)`
- Defined: `xray_tensor_diffractometer.py:1083`

### compute_fisher_information_matrix `def compute_fisher_information_matrix(model, samples)`
- Defined: `xray_tensor_diffractometer.py:1107`

### compute_ricci_curvature `def compute_ricci_curvature(fisher_matrix)`
- Defined: `xray_tensor_diffractometer.py:1126`

### calculate_carnot_efficiency `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)`
- Defined: `xray_tensor_diffractometer.py:1137`

### compute_kappa `def compute_kappa(model, dataloader, num_batches)`
- Defined: `xray_tensor_diffractometer.py:1161`

### compute_discretization_margin `def compute_discretization_margin(coeffs)`
- Defined: `xray_tensor_diffractometer.py:1187`

### compute_local_complexity `def compute_local_complexity(model)`
- Defined: `xray_tensor_diffractometer.py:1191`

### compute_alpha_purity `def compute_alpha_purity(coeffs)`
- Defined: `xray_tensor_diffractometer.py:1200`

### compute_kappa_quantum `def compute_kappa_quantum(coeffs, hbar)`
- Defined: `xray_tensor_diffractometer.py:1207`

### compute_poynting_vector `def compute_poynting_vector(coeffs)`
- Defined: `xray_tensor_diffractometer.py:1224`

### compute_all_metrics `def compute_all_metrics(model, dataloader)`
- Defined: `xray_tensor_diffractometer.py:1246`

### __init__ `def __init__(self, model, device)`
- Defined: `xray_tensor_diffractometer.py:1270`

### compute_boundary_gradient `def compute_boundary_gradient(self, weight)`
- Defined: `xray_tensor_diffractometer.py:1275`
- Doc: Approximate surface term: gradient concentrated on tensor boundaries.

### compute_bulk_gradient `def compute_bulk_gradient(self, weight)`
- Defined: `xray_tensor_diffractometer.py:1290`
- Doc: Interior (volume) term: everything except boundary.

### run_green_backprop_step `def run_green_backprop_step(self, A, B, C_true, lambda_boundary)`
- Defined: `xray_tensor_diffractometer.py:1296`
- Doc: Custom backward pass using Green-inspired decomposition.

### _get_boundary_mask `def _get_boundary_mask(self, weight)`
- Defined: `xray_tensor_diffractometer.py:1329`
- Doc: Returns a binary mask marking boundary elements of a tensor.

### train_with_green_cow `def train_with_green_cow(self, epochs, lr, lambda_boundary)`
- Defined: `xray_tensor_diffractometer.py:1341`

### load_checkpoint `def load_checkpoint(self, path, device)`
- Defined: `xray_tensor_diffractometer.py:1372`
- Doc: Load checkpoint with robust deserialization handling.

### migrate_checkpoint `def migrate_checkpoint(raw_data)`
- Defined: `xray_tensor_diffractometer.py:1407`
- Doc: Migrate checkpoint to standard format, extracting config if present.

### _migrate_dict `def _migrate_dict(state_dict)`
- Defined: `xray_tensor_diffractometer.py:1431`

### _migrate_custom_format `def _migrate_custom_format(state_dict)`
- Defined: `xray_tensor_diffractometer.py:1443`

### _migrate_coefs_format `def _migrate_coefs_format(state_dict)`
- Defined: `xray_tensor_diffractometer.py:1467`

### _migrate_encoder_format `def _migrate_encoder_format(state_dict)`
- Defined: `xray_tensor_diffractometer.py:1475`

### _migrate_standard_format `def _migrate_standard_format(state_dict)`
- Defined: `xray_tensor_diffractometer.py:1487`

### __init__ `def __init__(self, checkpoint_dir, results_dir)`
- Defined: `xray_tensor_diffractometer.py:1491`

### _load_all_checkpoints `def _load_all_checkpoints(self)`
- Defined: `xray_tensor_diffractometer.py:1503`

### run_full_boltzmann_program `def run_full_boltzmann_program(self)`
- Defined: `xray_tensor_diffractometer.py:1548`

### phase1_molecular_hypothesis `def phase1_molecular_hypothesis(self)`
- Defined: `xray_tensor_diffractometer.py:1575`

### phase2_entropy_production `def phase2_entropy_production(self)`
- Defined: `xray_tensor_diffractometer.py:1660`

### phase3_extensivity_law `def phase3_extensivity_law(self)`
- Defined: `xray_tensor_diffractometer.py:1742`

### phase4_quantum_basis_transform `def phase4_quantum_basis_transform(self)`
- Defined: `xray_tensor_diffractometer.py:1796`

### analyze_poynting_flow `def analyze_poynting_flow(self)`
- Defined: `xray_tensor_diffractometer.py:1849`

### phase5_thermodynamic_analysis `def phase5_thermodynamic_analysis(self)`
- Defined: `xray_tensor_diffractometer.py:1882`
- Doc: PHASE 5: THERMODYNAMIC ANALYSIS con exponentes críticos y ecuación de estado.

### phase6_spectroscopic_analysis `def phase6_spectroscopic_analysis(self)`
- Defined: `xray_tensor_diffractometer.py:2011`

### _plot_diffraction_pattern `def _plot_diffraction_pattern(self, diffraction_data, ckpt_name)`
- Defined: `xray_tensor_diffractometer.py:2097`

### _save_superlattice_seed `def _save_superlattice_seed(self, superlattice, ckpt_name)`
- Defined: `xray_tensor_diffractometer.py:2131`

### _classify_thermodynamic_phase `def _classify_thermodynamic_phase(self, t_eff, cv, alpha)`
- Defined: `xray_tensor_diffractometer.py:2150`

### _estimate_critical_temperature `def _estimate_critical_temperature(self, results)`
- Defined: `xray_tensor_diffractometer.py:2162`

### _verify_entropy_extensivity `def _verify_entropy_extensivity(self, results)`
- Defined: `xray_tensor_diffractometer.py:2174`

### _plot_phase_diagram `def _plot_phase_diagram(self, results)`
- Defined: `xray_tensor_diffractometer.py:2188`

### _plot_temperature_vs_purity `def _plot_temperature_vs_purity(self, results)`
- Defined: `xray_tensor_diffractometer.py:2213`

### _compute_entropy_simple `def _compute_entropy_simple(self, params)`
- Defined: `xray_tensor_diffractometer.py:2239`

### _compute_entropy `def _compute_entropy(self, params)`
- Defined: `xray_tensor_diffractometer.py:2265`

### _compute_effective_volume `def _compute_effective_volume(self, params)`
- Defined: `xray_tensor_diffractometer.py:2304`

### _plot_parameter_distribution `def _plot_parameter_distribution(self, params, group_name, kde)`
- Defined: `xray_tensor_diffractometer.py:2322`

### _simulate_training_trajectory `def _simulate_training_trajectory(self, final_params, final_delta)`
- Defined: `xray_tensor_diffractometer.py:2350`

### _compute_generalization_entropy `def _compute_generalization_entropy(self, params, successful_ckpts)`
- Defined: `xray_tensor_diffractometer.py:2361`

### _fit_timescale `def _fit_timescale(self, entropy_values)`
- Defined: `xray_tensor_diffractometer.py:2414`

### _plot_entropy_production `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- Defined: `xray_tensor_diffractometer.py:2424`

### _verify_scaling `def _verify_scaling(self, coeffs, N)`
- Defined: `xray_tensor_diffractometer.py:2442`

### _recursive_strassen `def _recursive_strassen(self, A, B, coeffs, N)`
- Defined: `xray_tensor_diffractometer.py:2453`

### _fit_extensivity `def _fit_extensivity(self, errors, sizes, purity)`
- Defined: `xray_tensor_diffractometer.py:2487`

### _verify_extensivity_universality `def _verify_extensivity_universality(self, results)`
- Defined: `xray_tensor_diffractometer.py:2501`

### _plot_extensivity `def _plot_extensivity(self, sizes, errors, purity, ckpt_name)`
- Defined: `xray_tensor_diffractometer.py:2505`

### _find_broken_symmetries `def _find_broken_symmetries(self, coeffs)`
- Defined: `xray_tensor_diffractometer.py:2518`

### _measure_uncertainty `def _measure_uncertainty(self, coeffs, basis)`
- Defined: `xray_tensor_diffractometer.py:2526`

### _plot_uncertainty_distribution `def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)`
- Defined: `xray_tensor_diffractometer.py:2537`

### _print_executive_summary `def _print_executive_summary(self, results)`
- Defined: `xray_tensor_diffractometer.py:2558`

### _save_results `def _save_results(self, results, filename)`
- Defined: `xray_tensor_diffractometer.py:2587`

### __init__ `def __init__(self, checkpoint_path, device)`
- Defined: `xray_tensor_diffractometer.py:2611`

### _load_model `def _load_model(self, path, device)`
- Defined: `xray_tensor_diffractometer.py:2617`

### run_full_analysis `def run_full_analysis(self)`
- Defined: `xray_tensor_diffractometer.py:2634`

### _assign_grade `def _assign_grade(self, delta, alpha)`
- Defined: `xray_tensor_diffractometer.py:2656`

### _save_report `def _save_report(self, report)`
- Defined: `xray_tensor_diffractometer.py:2668`

### generate_batch `def generate_batch(batch_size)`
- Defined: `xray_tensor_diffractometer.py:373`

### get_tensor `def get_tensor(key)`
- Defined: `xray_tensor_diffractometer.py:1444`

### model `def model(t, A, tau, C)`
- Defined: `xray_tensor_diffractometer.py:2415`

### model `def model(N, alpha, beta)`
- Defined: `xray_tensor_diffractometer.py:2488`

### convert_to_serializable `def convert_to_serializable(obj)`
- Defined: `xray_tensor_diffractometer.py:2590`

### dataloader `def dataloader()`
- Defined: `xray_tensor_diffractometer.py:2637`

### sample_dataloader `def sample_dataloader()`
- Defined: `xray_tensor_diffractometer.py:1906`

### sample_dataloader `def sample_dataloader()`
- Defined: `xray_tensor_diffractometer.py:2039`

### dataloader `def dataloader()`
- Defined: `xray_tensor_diffractometer.py:1522`
