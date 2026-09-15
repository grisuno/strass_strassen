# Symbols

| Symbol | Kind | File:Line | Signature |
|--------|------|-----------|-----------|
| `StrassenNet` | class | `app.py:24` | `class StrassenNet(Module)` |
| `__init__` | method | `app.py:25` | `def __init__(self, rank)` |
| `forward` | method | `app.py:31` | `def forward(self, A, B)` |
| `BatchSizeThermodynamics` | class | `batch_size.py:217` | `class BatchSizeThermodynamics` |
| `BilinearStrassenModel` | class | `batch_size.py:77` | `class BilinearStrassenModel(Module)` |
| `CheckpointMigrationManager` | class | `batch_size.py:131` | `class CheckpointMigrationManager` |
| `CheckpointMigrator` | class | `batch_size.py:100` | `class CheckpointMigrator(ABC)` |
| `Configuration` | class | `batch_size.py:31` | `class Configuration` |
| `CrystallographyMetrics` | class | `batch_size.py:156` | `class CrystallographyMetrics` |
| `CustomFormatMigrator` | class | `batch_size.py:110` | `class CustomFormatMigrator(CheckpointMigrator)` |
| `PlanckConstantCalculator` | class | `batch_size.py:186` | `class PlanckConstantCalculator` |
| `StandardFormatMigrator` | class | `batch_size.py:122` | `class StandardFormatMigrator(CheckpointMigrator)` |
| `StrassenCheckpointLoader` | class | `batch_size.py:262` | `class StrassenCheckpointLoader` |
| `StrassenDataGenerator` | class | `batch_size.py:147` | `class StrassenDataGenerator` |
| `StrassenPlanckAnalyzer` | class | `batch_size.py:292` | `class StrassenPlanckAnalyzer` |
| `__init__` | method | `batch_size.py:78` | `def __init__(self, config)` |
| `__init__` | method | `batch_size.py:132` | `def __init__(self)` |
| `__init__` | method | `batch_size.py:187` | `def __init__(self, metrics, training_metrics, config)` |
| `__init__` | method | `batch_size.py:218` | `def __init__(self, model, h_bar, delta_struct, config)` |
| `__init__` | method | `batch_size.py:263` | `def __init__(self, config)` |
| `__init__` | method | `batch_size.py:293` | `def __init__(self, config)` |
| `_measure_gradients` | method | `batch_size.py:246` | `def _measure_gradients(self, batch_size)` |
| `analyze_batch_size_spectrum` | method | `batch_size.py:224` | `def analyze_batch_size_spectrum(self)` |
| `analyze_checkpoint` | method | `batch_size.py:297` | `def analyze_checkpoint(self, path, device)` |
| `analyze_directory` | method | `batch_size.py:323` | `def analyze_directory(self, directory, device, pattern)` |
| `calculate_all` | method | `batch_size.py:196` | `def calculate_all(self)` |
| `can_migrate` | method | `batch_size.py:102` | `def can_migrate(self, state_dict)` |
| `can_migrate` | method | `batch_size.py:111` | `def can_migrate(self, state_dict)` |
| `can_migrate` | method | `batch_size.py:123` | `def can_migrate(self, state_dict)` |
| `compute_discretization_margin` | method | `batch_size.py:174` | `def compute_discretization_margin(coeffs)` |
| `compute_kappa` | method | `batch_size.py:158` | `def compute_kappa(model, num_batches, config)` |
| `compute_lambda_effective` | method | `batch_size.py:94` | `def compute_lambda_effective(self)` |
| `compute_local_complexity` | method | `batch_size.py:178` | `def compute_local_complexity(model, config)` |
| `extract_training_metrics` | method | `batch_size.py:281` | `def extract_training_metrics(self, path)` |
| `forward` | method | `batch_size.py:88` | `def forward(self, a, b)` |
| `generate_batch` | method | `batch_size.py:149` | `def generate_batch(batch_size, config)` |
| `get_coefficients` | method | `batch_size.py:91` | `def get_coefficients(self)` |
| `load` | method | `batch_size.py:267` | `def load(self, path, device)` |
| `main` | method | `batch_size.py:341` | `def main()` |
| `migrate` | method | `batch_size.py:106` | `def migrate(self, state_dict)` |
| `migrate` | method | `batch_size.py:114` | `def migrate(self, state_dict)` |
| `migrate` | method | `batch_size.py:126` | `def migrate(self, state_dict)` |
| `migrate_checkpoint` | method | `batch_size.py:135` | `def migrate_checkpoint(self, path, device)` |
| `set_random_seed` | method | `batch_size.py:71` | `def set_random_seed(seed)` |
| `BilinearStrassenModel` | class | `boltzmann_experiments.py:118` | `class BilinearStrassenModel(Module)` |
| `CheckpointLoader` | class | `boltzmann_experiments.py:45` | `class CheckpointLoader(ICheckpointLoader)` |
| `CheckpointLoadingError` | class | `boltzmann_experiments.py:37` | `class CheckpointLoadingError(Exception)` |
| `CheckpointMigrator` | class | `boltzmann_experiments.py:52` | `class CheckpointMigrator` |
| `Config` | class | `boltzmann_experiments.py:19` | `class Config` |
| `CrystallographyMetrics` | class | `boltzmann_experiments.py:137` | `class CrystallographyMetrics` |
| `DLProgram` | class | `boltzmann_experiments.py:199` | `class DLProgram` |
| `ICheckpointLoader` | class | `boltzmann_experiments.py:40` | `class ICheckpointLoader(ABC)` |
| `__init__` | method | `boltzmann_experiments.py:119` | `def __init__(self, n_slots)` |
| `__init__` | method | `boltzmann_experiments.py:200` | `def __init__(self, checkpoint_dir, results_dir)` |
| `_compute_effective_volume` | method | `boltzmann_experiments.py:466` | `def _compute_effective_volume(self, kde)` |
| `_compute_entropy` | method | `boltzmann_experiments.py:446` | `def _compute_entropy(self, params)` |
| `_compute_entropy_simple` | method | `boltzmann_experiments.py:435` | `def _compute_entropy_simple(self, params)` |
| `_compute_generalization_entropy` | method | `boltzmann_experiments.py:604` | `def _compute_generalization_entropy(self, params, successful_ckpts)` |
| `_compute_generalization_entropy` | method | `boltzmann_experiments.py:962` | `def _compute_generalization_entropy(self, params, successful_ckpts)` |
| `_find_broken_symmetries` | method | `boltzmann_experiments.py:895` | `def _find_broken_symmetries(self, coeffs)` |
| `_fit_extensivity` | method | `boltzmann_experiments.py:812` | `def _fit_extensivity(self, errors, sizes, purity)` |
| `_fit_timescale` | method | `boltzmann_experiments.py:691` | `def _fit_timescale(self, entropy_values)` |
| `_fit_timescale` | method | `boltzmann_experiments.py:1016` | `def _fit_timescale(self, entropy_values)` |
| `_format_direct_tensors` | method | `boltzmann_experiments.py:70` | `def _format_direct_tensors(tensor_dict)` |
| `_initialize_symmetric` | method | `boltzmann_experiments.py:126` | `def _initialize_symmetric(self)` |
| `_load_all_checkpoints` | method | `boltzmann_experiments.py:207` | `def _load_all_checkpoints(self)` |
| `_measure_uncertainty` | method | `boltzmann_experiments.py:903` | `def _measure_uncertainty(self, coeffs, basis)` |
| `_migrate_coefs_format` | method | `boltzmann_experiments.py:115` | `def _migrate_coefs_format(state_dict)` |
| `_migrate_dict` | method | `boltzmann_experiments.py:92` | `def _migrate_dict(state_dict)` |
| `_migrate_encoder_format` | method | `boltzmann_experiments.py:105` | `def _migrate_encoder_format(state_dict)` |
| `_plot_entropy_production` | method | `boltzmann_experiments.py:701` | `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)` |
| `_plot_entropy_production` | method | `boltzmann_experiments.py:1026` | `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)` |
| `_plot_extensivity` | method | `boltzmann_experiments.py:828` | `def _plot_extensivity(self, sizes, errors, purity, ckpt_name)` |
| `_plot_parameter_distribution` | method | `boltzmann_experiments.py:475` | `def _plot_parameter_distribution(self, params, group_name, kde)` |
| `_plot_uncertainty_distribution` | method | `boltzmann_experiments.py:914` | `def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)` |
| `_print_executive_summary` | method | `boltzmann_experiments.py:267` | `def _print_executive_summary(self, results)` |
| `_recursive_strassen` | method | `boltzmann_experiments.py:783` | `def _recursive_strassen(self, A, B, coeffs, N)` |
| `_save_results` | method | `boltzmann_experiments.py:307` | `def _save_results(self, results, filename)` |
| `_simulate_training_trajectory` | method | `boltzmann_experiments.py:592` | `def _simulate_training_trajectory(self, final_params, final_delta)` |
| `_simulate_training_trajectory` | method | `boltzmann_experiments.py:951` | `def _simulate_training_trajectory(self, final_params, final_delta)` |
| `_verify_extensivity_universality` | method | `boltzmann_experiments.py:824` | `def _verify_extensivity_universality(self, results)` |
| `_verify_scaling` | method | `boltzmann_experiments.py:773` | `def _verify_scaling(self, coeffs, N)` |
| `compute_alpha_purity` | method | `boltzmann_experiments.py:169` | `def compute_alpha_purity(coeffs)` |
| `compute_delta` | method | `boltzmann_experiments.py:156` | `def compute_delta(coeffs)` |
| `compute_kappa` | method | `boltzmann_experiments.py:139` | `def compute_kappa(coeffs)` |
| `compute_kappa_quantum` | method | `boltzmann_experiments.py:178` | `def compute_kappa_quantum(coeffs, hbar)` |
| `compute_local_complexity` | method | `boltzmann_experiments.py:161` | `def compute_local_complexity(coeffs)` |
| `convert_to_serializable` | method | `boltzmann_experiments.py:310` | `def convert_to_serializable(obj)` |
| `forward` | method | `boltzmann_experiments.py:131` | `def forward(self, a, b)` |
| `get_coefficients` | method | `boltzmann_experiments.py:134` | `def get_coefficients(self)` |
| `load_checkpoint` | method | `boltzmann_experiments.py:42` | `def load_checkpoint(self, path, device)` |
| `load_checkpoint` | method | `boltzmann_experiments.py:46` | `def load_checkpoint(self, path, device)` |
| `main` | method | `boltzmann_experiments.py:935` | `def main()` |
| `migrate_checkpoint` | method | `boltzmann_experiments.py:54` | `def migrate_checkpoint(raw_data)` |
| `model` | method | `boltzmann_experiments.py:692` | `def model(t, A, tau, C)` |
| `model` | method | `boltzmann_experiments.py:813` | `def model(N, alpha, beta)` |
| `model` | method | `boltzmann_experiments.py:1017` | `def model(t, A, tau, C)` |
| `phase1_molecular_hypothesis` | method | `boltzmann_experiments.py:328` | `def phase1_molecular_hypothesis(self)` |
| `phase2_entropy_production` | method | `boltzmann_experiments.py:506` | `def phase2_entropy_production(self)` |
| `phase3_extensivity_law` | method | `boltzmann_experiments.py:719` | `def phase3_extensivity_law(self)` |
| `phase3_extensivity_law` | method | `boltzmann_experiments.py:1044` | `def phase3_extensivity_law(self)` |
| `phase4_quantum_basis_transform` | method | `boltzmann_experiments.py:841` | `def phase4_quantum_basis_transform(self)` |
| `run_full_boltzmann_program` | method | `boltzmann_experiments.py:247` | `def run_full_boltzmann_program(self)` |
| `set_seed` | method | `boltzmann_experiments.py:30` | `def set_seed(seed)` |
| `estimate_gns` | function | `compute_gns_checkpoints.py:11` | `def estimate_gns(model, batch_size, num_batches)` |
| `main` | function | `compute_gns_checkpoints.py:37` | `def main()` |
| `BasinResilienceSpectrometer` | class | `crystallography.py:277` | `class BasinResilienceSpectrometer` |
| `BilinearStrassenModel` | class | `crystallography.py:44` | `class BilinearStrassenModel(Module)` |
| `CheckpointMigrator` | class | `crystallography.py:71` | `class CheckpointMigrator` |
| `Config` | class | `crystallography.py:25` | `class Config` |
| `CrystalPurityIndex` | class | `crystallography.py:345` | `class CrystalPurityIndex` |
| `CrystallographyMetrics` | class | `crystallography.py:206` | `class CrystallographyMetrics` |
| `LocalComplexity` | class | `crystallography.py:523` | `class LocalComplexity` |
| `SparsificationProtocol` | class | `crystallography.py:172` | `class SparsificationProtocol` |
| `StrassenCrystallographer` | class | `crystallography.py:416` | `class StrassenCrystallographer` |
| `StrassenDataGenerator` | class | `crystallography.py:155` | `class StrassenDataGenerator` |
| `StrassenDiffractionTest` | class | `crystallography.py:232` | `class StrassenDiffractionTest` |
| `__init__` | method | `crystallography.py:45` | `def __init__(self, n_slots)` |
| `__init__` | method | `crystallography.py:173` | `def __init__(self, model)` |
| `__init__` | method | `crystallography.py:233` | `def __init__(self, model)` |
| `__init__` | method | `crystallography.py:278` | `def __init__(self, model)` |
| `__init__` | method | `crystallography.py:346` | `def __init__(self, model, diffraction_results, resilience_results, metrics_results)` |
| `__init__` | method | `crystallography.py:417` | `def __init__(self, checkpoint_path, device)` |
| `_anneal_to_attractor` | method | `crystallography.py:317` | `def _anneal_to_attractor(self, max_epochs)` |
| `_apply_noise` | method | `crystallography.py:312` | `def _apply_noise(self, sigma)` |
| `_assign_grade` | method | `crystallography.py:399` | `def _assign_grade(self, index, delta)` |
| `_estimate_critical_noise` | method | `crystallography.py:329` | `def _estimate_critical_noise(self, results)` |
| `_functional_error` | method | `crystallography.py:262` | `def _functional_error(self, test_coeffs)` |
| `_initialize_symmetric` | method | `crystallography.py:52` | `def _initialize_symmetric(self)` |
| `_migrate_custom` | method | `crystallography.py:106` | `def _migrate_custom(state_dict)` |
| `_migrate_encoder` | method | `crystallography.py:123` | `def _migrate_encoder(state_dict)` |
| `_migrate_standard` | method | `crystallography.py:147` | `def _migrate_standard(state_dict)` |
| `_save_report` | method | `crystallography.py:506` | `def _save_report(self, report)` |
| `_test_noise_recovery` | method | `crystallography.py:293` | `def _test_noise_recovery(self, sigma, n_trials)` |
| `compute` | method | `crystallography.py:359` | `def compute(self)` |
| `compute` | method | `crystallography.py:525` | `def compute(model)` |
| `compute_discretization_margin` | method | `crystallography.py:225` | `def compute_discretization_margin(coeffs)` |
| `compute_kappa` | method | `crystallography.py:208` | `def compute_kappa(model, dataloader, num_batches)` |
| `dataloader_gen` | method | `crystallography.py:465` | `def dataloader_gen()` |
| `discretize_weights` | method | `crystallography.py:192` | `def discretize_weights(self, margin)` |
| `forward` | method | `crystallography.py:57` | `def forward(self, a, b)` |
| `generate_batch` | method | `crystallography.py:157` | `def generate_batch(batch_size)` |
| `get_coefficients` | method | `crystallography.py:60` | `def get_coefficients(self)` |
| `main` | method | `crystallography.py:542` | `def main()` |
| `measure_resilience_spectrum` | method | `crystallography.py:282` | `def measure_resilience_spectrum(self, noise_levels)` |
| `migrate_checkpoint` | method | `crystallography.py:73` | `def migrate_checkpoint(path, device)` |
| `prune_to_target` | method | `crystallography.py:176` | `def prune_to_target(self, target)` |
| `run_full_analysis` | method | `crystallography.py:445` | `def run_full_analysis(self)` |
| `set_seed` | method | `crystallography.py:35` | `def set_seed(seed)` |
| `test_gauge_invariance` | method | `crystallography.py:236` | `def test_gauge_invariance(self, n_samples)` |
| `verify_structure` | method | `crystallography.py:164` | `def verify_structure(coeffs)` |
| `AnalysisConfig` | class | `dirac_polos_zeros.py:24` | `class AnalysisConfig` |
| `AnalysisPipeline` | class | `dirac_polos_zeros.py:1288` | `class AnalysisPipeline` |
| `BilinearModel` | class | `dirac_polos_zeros.py:124` | `class BilinearModel(Module)` |
| `BodeVisualizer` | class | `dirac_polos_zeros.py:848` | `class BodeVisualizer` |
| `ChargeDistributionExtractor` | class | `dirac_polos_zeros.py:152` | `class ChargeDistributionExtractor` |
| `ChargeDistributionVisualizer` | class | `dirac_polos_zeros.py:710` | `class ChargeDistributionVisualizer` |
| `CheckpointLoader` | class | `dirac_polos_zeros.py:653` | `class CheckpointLoader` |
| `CheckpointMigrator` | class | `dirac_polos_zeros.py:661` | `class CheckpointMigrator` |
| `CombinedVisualizer` | class | `dirac_polos_zeros.py:966` | `class CombinedVisualizer` |
| `DiracDeltaAnalyzer` | class | `dirac_polos_zeros.py:160` | `class DiracDeltaAnalyzer` |
| `DivergenceCalculator` | class | `dirac_polos_zeros.py:248` | `class DivergenceCalculator` |
| `DivergenceVisualizer` | class | `dirac_polos_zeros.py:780` | `class DivergenceVisualizer` |
| `ElectricFieldCalculator` | class | `dirac_polos_zeros.py:192` | `class ElectricFieldCalculator` |
| `ElectricFieldVisualizer` | class | `dirac_polos_zeros.py:738` | `class ElectricFieldVisualizer` |
| `ElectricFluxCalculator` | class | `dirac_polos_zeros.py:225` | `class ElectricFluxCalculator` |
| `FrequencyResponseAnalyzer` | class | `dirac_polos_zeros.py:463` | `class FrequencyResponseAnalyzer` |
| `GaussLawVerifier` | class | `dirac_polos_zeros.py:253` | `class GaussLawVerifier` |
| `IChargeDistributionExtractor` | class | `dirac_polos_zeros.py:60` | `class IChargeDistributionExtractor(Protocol)` |
| `ICheckpointLoader` | class | `dirac_polos_zeros.py:110` | `class ICheckpointLoader(Protocol)` |
| `ICheckpointMigrator` | class | `dirac_polos_zeros.py:115` | `class ICheckpointMigrator(Protocol)` |
| `IDiracAnalyzer` | class | `dirac_polos_zeros.py:65` | `class IDiracAnalyzer(Protocol)` |
| `IFieldCalculator` | class | `dirac_polos_zeros.py:70` | `class IFieldCalculator(Protocol)` |
| `IFluxCalculator` | class | `dirac_polos_zeros.py:75` | `class IFluxCalculator(Protocol)` |
| `IFrequencyAnalyzer` | class | `dirac_polos_zeros.py:97` | `class IFrequencyAnalyzer(Protocol)` |
| `IModel` | class | `dirac_polos_zeros.py:54` | `class IModel(Protocol)` |
| `IPoleZeroAnalyzer` | class | `dirac_polos_zeros.py:90` | `class IPoleZeroAnalyzer(Protocol)` |
| `IStateSpaceExtractor` | class | `dirac_polos_zeros.py:80` | `class IStateSpaceExtractor(Protocol)` |
| `ITimeResponseAnalyzer` | class | `dirac_polos_zeros.py:104` | `class ITimeResponseAnalyzer(Protocol)` |
| `ITransferFunctionComputer` | class | `dirac_polos_zeros.py:85` | `class ITransferFunctionComputer(Protocol)` |
| `IVisualizer` | class | `dirac_polos_zeros.py:120` | `class IVisualizer(Protocol)` |
| `NyquistVisualizer` | class | `dirac_polos_zeros.py:899` | `class NyquistVisualizer` |
| `PoleZeroAnalyzer` | class | `dirac_polos_zeros.py:313` | `class PoleZeroAnalyzer` |
| `PoleZeroVisualizer` | class | `dirac_polos_zeros.py:809` | `class PoleZeroVisualizer` |
| `StateSpaceExtractor` | class | `dirac_polos_zeros.py:271` | `class StateSpaceExtractor` |
| `SystemAnalyzer` | class | `dirac_polos_zeros.py:1070` | `class SystemAnalyzer` |
| `TimeResponseAnalyzer` | class | `dirac_polos_zeros.py:566` | `class TimeResponseAnalyzer` |
| `TimeResponseVisualizer` | class | `dirac_polos_zeros.py:936` | `class TimeResponseVisualizer` |
| `TransferFunctionComputer` | class | `dirac_polos_zeros.py:298` | `class TransferFunctionComputer` |
| `__init__` | method | `dirac_polos_zeros.py:125` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `dirac_polos_zeros.py:161` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:193` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:226` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:254` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:314` | `def __init__(self, numerator, denominator, config)` |
| `__init__` | method | `dirac_polos_zeros.py:464` | `def __init__(self, numerator, denominator, config)` |
| `__init__` | method | `dirac_polos_zeros.py:567` | `def __init__(self, numerator, denominator, config)` |
| `__init__` | method | `dirac_polos_zeros.py:711` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:739` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:781` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:810` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:849` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:900` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:937` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:967` | `def __init__(self, config)` |
| `__init__` | method | `dirac_polos_zeros.py:1071` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `dirac_polos_zeros.py:1289` | `def __init__(self, config)` |
| `_compute` | method | `dirac_polos_zeros.py:323` | `def _compute(self)` |
| `_compute_aggregate_statistics` | method | `dirac_polos_zeros.py:1406` | `def _compute_aggregate_statistics(self, results)` |
| `_generate_text_report` | method | `dirac_polos_zeros.py:1473` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize` | method | `dirac_polos_zeros.py:136` | `def _initialize(self)` |
| `_load_model` | method | `dirac_polos_zeros.py:1088` | `def _load_model(self)` |
| `_migrate_coefs_format` | method | `dirac_polos_zeros.py:699` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `dirac_polos_zeros.py:683` | `def _migrate_custom_format(self, state_dict)` |
| `_migrate_dict` | method | `dirac_polos_zeros.py:674` | `def _migrate_dict(self, state_dict)` |
| `_migrate_standard_format` | method | `dirac_polos_zeros.py:706` | `def _migrate_standard_format(self, state_dict)` |
| `_print_report` | method | `dirac_polos_zeros.py:1204` | `def _print_report(self, results)` |
| `analyze` | method | `dirac_polos_zeros.py:66` | `def analyze(self, charge_density)` |
| `analyze` | method | `dirac_polos_zeros.py:164` | `def analyze(self, charge_density)` |
| `analyze` | method | `dirac_polos_zeros.py:1104` | `def analyze(self)` |
| `analyze_stability` | method | `dirac_polos_zeros.py:91` | `def analyze_stability(self)` |
| `analyze_stability` | method | `dirac_polos_zeros.py:340` | `def analyze_stability(self)` |
| `analyze_step_characteristics` | method | `dirac_polos_zeros.py:601` | `def analyze_step_characteristics(self, step_data)` |
| `calculate` | method | `dirac_polos_zeros.py:71` | `def calculate(self, dirac_data, eval_points)` |
| `calculate` | method | `dirac_polos_zeros.py:76` | `def calculate(self, electric_field, surface_points)` |
| `calculate` | method | `dirac_polos_zeros.py:196` | `def calculate(self, dirac_data, eval_points)` |
| `calculate` | method | `dirac_polos_zeros.py:229` | `def calculate(self, electric_field, surface_points)` |
| `calculate` | method | `dirac_polos_zeros.py:249` | `def calculate(self, electric_field)` |
| `classify_poles` | method | `dirac_polos_zeros.py:378` | `def classify_poles(self)` |
| `compute` | method | `dirac_polos_zeros.py:86` | `def compute(self, A, B, C, D)` |
| `compute` | method | `dirac_polos_zeros.py:299` | `def compute(self, A, B, C, D)` |
| `compute_bode` | method | `dirac_polos_zeros.py:98` | `def compute_bode(self)` |
| `compute_bode` | method | `dirac_polos_zeros.py:474` | `def compute_bode(self)` |
| `compute_damping` | method | `dirac_polos_zeros.py:406` | `def compute_damping(self)` |
| `compute_impulse` | method | `dirac_polos_zeros.py:106` | `def compute_impulse(self)` |
| `compute_impulse` | method | `dirac_polos_zeros.py:589` | `def compute_impulse(self)` |
| `compute_margins` | method | `dirac_polos_zeros.py:99` | `def compute_margins(self)` |
| `compute_margins` | method | `dirac_polos_zeros.py:490` | `def compute_margins(self)` |
| `compute_nyquist` | method | `dirac_polos_zeros.py:100` | `def compute_nyquist(self)` |
| `compute_nyquist` | method | `dirac_polos_zeros.py:516` | `def compute_nyquist(self)` |
| `compute_step` | method | `dirac_polos_zeros.py:105` | `def compute_step(self)` |
| `compute_step` | method | `dirac_polos_zeros.py:577` | `def compute_step(self)` |
| `compute_time_constants` | method | `dirac_polos_zeros.py:445` | `def compute_time_constants(self)` |
| `evaluate_nyquist_stability` | method | `dirac_polos_zeros.py:535` | `def evaluate_nyquist_stability(self, nyquist_data)` |
| `extract` | method | `dirac_polos_zeros.py:61` | `def extract(self, model)` |
| `extract` | method | `dirac_polos_zeros.py:81` | `def extract(self, model)` |
| `extract` | method | `dirac_polos_zeros.py:153` | `def extract(self, model)` |
| `extract` | method | `dirac_polos_zeros.py:272` | `def extract(self, model)` |
| `forward` | method | `dirac_polos_zeros.py:55` | `def forward(self, a, b)` |
| `forward` | method | `dirac_polos_zeros.py:141` | `def forward(self, a, b)` |
| `generate_summary` | method | `dirac_polos_zeros.py:1385` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `dirac_polos_zeros.py:56` | `def get_coefficients(self)` |
| `get_coefficients` | method | `dirac_polos_zeros.py:144` | `def get_coefficients(self)` |
| `get_poles` | method | `dirac_polos_zeros.py:92` | `def get_poles(self)` |
| `get_poles` | method | `dirac_polos_zeros.py:334` | `def get_poles(self)` |
| `get_zeros` | method | `dirac_polos_zeros.py:93` | `def get_zeros(self)` |
| `get_zeros` | method | `dirac_polos_zeros.py:337` | `def get_zeros(self)` |
| `load` | method | `dirac_polos_zeros.py:111` | `def load(self, path, device)` |
| `load` | method | `dirac_polos_zeros.py:654` | `def load(self, path, device)` |
| `main` | method | `dirac_polos_zeros.py:1545` | `def main()` |
| `migrate` | method | `dirac_polos_zeros.py:116` | `def migrate(self, raw_data)` |
| `migrate` | method | `dirac_polos_zeros.py:662` | `def migrate(self, raw_data)` |
| `process_checkpoint` | method | `dirac_polos_zeros.py:1300` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `dirac_polos_zeros.py:1357` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `verify` | method | `dirac_polos_zeros.py:257` | `def verify(self, dirac_data, flux_data)` |
| `visualize` | method | `dirac_polos_zeros.py:121` | `def visualize(self, data, output_path)` |
| `visualize` | method | `dirac_polos_zeros.py:714` | `def visualize(self, data, output_path)` |
| `visualize` | method | `dirac_polos_zeros.py:742` | `def visualize(self, data, output_path)` |
| `visualize` | method | `dirac_polos_zeros.py:784` | `def visualize(self, data, output_path)` |
| `visualize` | method | `dirac_polos_zeros.py:813` | `def visualize(self, data, output_path)` |
| `visualize` | method | `dirac_polos_zeros.py:852` | `def visualize(self, data, output_path)` |
| `visualize` | method | `dirac_polos_zeros.py:903` | `def visualize(self, data, output_path)` |
| `visualize` | method | `dirac_polos_zeros.py:940` | `def visualize(self, data, output_path)` |
| `visualize` | method | `dirac_polos_zeros.py:970` | `def visualize(self, data, output_path)` |
| `BenchmarkResult` | class | `experiments/ablation/ablation_study.py:27` | `class BenchmarkResult` |
| `analyze_results` | method | `experiments/ablation/ablation_study.py:233` | `def analyze_results(results)` |
| `benchmark_single` | method | `experiments/ablation/ablation_study.py:132` | `def benchmark_single(libs, algo_name, func_name, A, B, C, C_ref, n, n_runs, warmup)` |
| `load_libraries` | method | `experiments/ablation/ablation_study.py:54` | `def load_libraries()` |
| `main` | method | `experiments/ablation/ablation_study.py:273` | `def main()` |
| `max_time` | method | `experiments/ablation/ablation_study.py:47` | `def max_time(self)` |
| `mean_gflops` | method | `experiments/ablation/ablation_study.py:51` | `def mean_gflops(self)` |
| `mean_time` | method | `experiments/ablation/ablation_study.py:35` | `def mean_time(self)` |
| `min_time` | method | `experiments/ablation/ablation_study.py:43` | `def min_time(self)` |
| `run_ablation` | method | `experiments/ablation/ablation_study.py:180` | `def run_ablation(libs, sizes, n_runs, warmup)` |
| `run_openblas` | method | `experiments/ablation/ablation_study.py:111` | `def run_openblas(libs, A, B, C, n)` |
| `run_strassen` | method | `experiments/ablation/ablation_study.py:122` | `def run_strassen(libs, name, func_name, A, B, C, n)` |
| `std_time` | method | `experiments/ablation/ablation_study.py:39` | `def std_time(self)` |
| `StrassenOperator` | class | `experiments/apendix_experiments.py:51` | `class StrassenOperator(Module)` |
| `__init__` | method | `experiments/apendix_experiments.py:53` | `def __init__(self, rank, symmetric_init)` |
| `compute_S_theta` | method | `experiments/apendix_experiments.py:138` | `def compute_S_theta(model)` |
| `compute_delta` | method | `experiments/apendix_experiments.py:100` | `def compute_delta(model)` |
| `compute_gradient_covariance` | method | `experiments/apendix_experiments.py:154` | `def compute_gradient_covariance(model, batch_size, n_samples)` |
| `count_active` | method | `experiments/apendix_experiments.py:83` | `def count_active(self, threshold)` |
| `forward` | method | `experiments/apendix_experiments.py:67` | `def forward(self, A, B)` |
| `generate_batch` | method | `experiments/apendix_experiments.py:87` | `def generate_batch(n, device)` |
| `generate_test_set` | method | `experiments/apendix_experiments.py:93` | `def generate_test_set(n, device)` |
| `main` | method | `experiments/apendix_experiments.py:526` | `def main()` |
| `run_batch_size_effect` | method | `experiments/apendix_experiments.py:429` | `def run_batch_size_effect()` |
| `run_phase_diagram` | method | `experiments/apendix_experiments.py:327` | `def run_phase_diagram()` |
| `setup_matplotlib` | function | `experiments/apendix_experiments.py:34` | `def setup_matplotlib()` |
| `slot_importance` | method | `experiments/apendix_experiments.py:77` | `def slot_importance(self)` |
| `sparsify_and_discretize` | method | `experiments/apendix_experiments.py:274` | `def sparsify_and_discretize(model, batch_size)` |
| `train_with_logging` | method | `experiments/apendix_experiments.py:187` | `def train_with_logging(batch_size, total_epochs, lr, wd, symmetric_init, seed, log_interval)` |
| `verify_strassen_structure` | method | `experiments/apendix_experiments.py:117` | `def verify_strassen_structure(U_disc, V_disc, W_disc, tolerance)` |
| `cache_analysis` | function | `experiments/cache_analysis_v2.py:7` | `def cache_analysis()` |
| `ArithmeticDataset` | class | `experiments/extended_experiments/all_test_extended.py:190` | `class ArithmeticDataset(Dataset)` |
| `AttractorLandscapeProbe` | class | `experiments/extended_experiments/all_test_extended.py:488` | `class AttractorLandscapeProbe` |
| `BilinearModel` | class | `experiments/extended_experiments/all_test_extended.py:238` | `class BilinearModel(Module)` |
| `CheckpointManager` | class | `experiments/extended_experiments/all_test_extended.py:581` | `class CheckpointManager` |
| `Configuration` | class | `experiments/extended_experiments/all_test_extended.py:41` | `class Configuration` |
| `DiscretizationAnalyzer` | class | `experiments/extended_experiments/all_test_extended.py:809` | `class DiscretizationAnalyzer` |
| `ExpansionVerifier` | class | `experiments/extended_experiments/all_test_extended.py:851` | `class ExpansionVerifier` |
| `ExperimentPipeline` | class | `experiments/extended_experiments/all_test_extended.py:879` | `class ExperimentPipeline` |
| `ExperimentRunner` | class | `experiments/extended_experiments/all_test_extended.py:700` | `class ExperimentRunner` |
| `GradientCovarianceProbe` | class | `experiments/extended_experiments/all_test_extended.py:402` | `class GradientCovarianceProbe` |
| `MatrixMultiplicationTask` | class | `experiments/extended_experiments/all_test_extended.py:311` | `class MatrixMultiplicationTask(Task)` |
| `Narrator` | class | `experiments/extended_experiments/all_test_extended.py:95` | `class Narrator` |
| `ParityDataset` | class | `experiments/extended_experiments/all_test_extended.py:354` | `class ParityDataset(Dataset)` |
| `ParityTask` | class | `experiments/extended_experiments/all_test_extended.py:375` | `class ParityTask(Task)` |
| `RobustnessTest` | class | `experiments/extended_experiments/all_test_extended.py:524` | `class RobustnessTest` |
| `SpectralInterventionProbe` | class | `experiments/extended_experiments/all_test_extended.py:465` | `class SpectralInterventionProbe` |
| `SystemFingerprint` | class | `experiments/extended_experiments/all_test_extended.py:152` | `class SystemFingerprint` |
| `Task` | class | `experiments/extended_experiments/all_test_extended.py:293` | `class Task(ABC)` |
| `TrainingMetrics` | class | `experiments/extended_experiments/all_test_extended.py:627` | `class TrainingMetrics` |
| `VolumeEstimator` | class | `experiments/extended_experiments/all_test_extended.py:510` | `class VolumeEstimator` |
| `__getitem__` | method | `experiments/extended_experiments/all_test_extended.py:234` | `def __getitem__(self, idx)` |
| `__getitem__` | method | `experiments/extended_experiments/all_test_extended.py:371` | `def __getitem__(self, idx)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:89` | `def __init__(self)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:96` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:153` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:202` | `def __init__(self, size, modulus)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:253` | `def __init__(self, d_vocab, rank, scale)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:312` | `def __init__(self, modulus)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:357` | `def __init__(self, size, bit_length)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:376` | `def __init__(self, bit_length, modulus)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:405` | `def __init__(self, model)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:468` | `def __init__(self, model, target_kappa)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:491` | `def __init__(self, model)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:513` | `def __init__(self, success_radius)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:527` | `def __init__(self, model)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:584` | `def __init__(self, config, experiment_name)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:630` | `def __init__(self)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:703` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:812` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:854` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/all_test_extended.py:882` | `def __init__(self, config)` |
| `__len__` | method | `experiments/extended_experiments/all_test_extended.py:231` | `def __len__(self)` |
| `__len__` | method | `experiments/extended_experiments/all_test_extended.py:368` | `def __len__(self)` |
| `_generate_data` | method | `experiments/extended_experiments/all_test_extended.py:207` | `def _generate_data(self)` |
| `_generate_data` | method | `experiments/extended_experiments/all_test_extended.py:362` | `def _generate_data(self)` |
| `_generate_summary` | method | `experiments/extended_experiments/all_test_extended.py:1379` | `def _generate_summary(self)` |
| `_save_results` | method | `experiments/extended_experiments/all_test_extended.py:1371` | `def _save_results(self)` |
| `add_gaussian_noise` | method | `experiments/extended_experiments/all_test_extended.py:530` | `def add_gaussian_noise(self, sigma)` |
| `analyze` | method | `experiments/extended_experiments/all_test_extended.py:450` | `def analyze(self, dataloader, batch_size, learning_rate)` |
| `begin` | method | `experiments/extended_experiments/all_test_extended.py:101` | `def begin(self, experiment_name)` |
| `capture` | method | `experiments/extended_experiments/all_test_extended.py:156` | `def capture(self)` |
| `capture_gradients` | method | `experiments/extended_experiments/all_test_extended.py:409` | `def capture_gradients(self, dataloader, n_batches)` |
| `check_strassen_structure` | method | `experiments/extended_experiments/all_test_extended.py:832` | `def check_strassen_structure(self, model, modulus)` |
| `checker` | method | `experiments/extended_experiments/all_test_extended.py:1253` | `def checker()` |
| `checkpoint` | method | `experiments/extended_experiments/all_test_extended.py:116` | `def checkpoint(self, epoch, loss, accuracy)` |
| `claim` | method | `experiments/extended_experiments/all_test_extended.py:144` | `def claim(self, statement, confidence)` |
| `classify_failure_mode` | method | `experiments/extended_experiments/all_test_extended.py:505` | `def classify_failure_mode(self, final_weights, initial_weights)` |
| `complete` | method | `experiments/extended_experiments/all_test_extended.py:136` | `def complete(self, summary)` |
| `compute_condition_number` | method | `experiments/extended_experiments/all_test_extended.py:436` | `def compute_condition_number(self)` |
| `compute_covariance` | method | `experiments/extended_experiments/all_test_extended.py:429` | `def compute_covariance(self)` |
| `compute_discretization_margin` | method | `experiments/extended_experiments/all_test_extended.py:815` | `def compute_discretization_margin(self, model)` |
| `compute_fractal_dimension` | method | `experiments/extended_experiments/all_test_extended.py:520` | `def compute_fractal_dimension(self, trajectory)` |
| `compute_gradient_noise_scale` | method | `experiments/extended_experiments/all_test_extended.py:442` | `def compute_gradient_noise_scale(self, batch_size, learning_rate)` |
| `count_discretized_parameters` | method | `experiments/extended_experiments/all_test_extended.py:839` | `def count_discretized_parameters(self, model)` |
| `count_local_minima` | method | `experiments/extended_experiments/all_test_extended.py:494` | `def count_local_minima(self, directions, losses)` |
| `d_vocab` | method | `experiments/extended_experiments/all_test_extended.py:299` | `def d_vocab(self)` |
| `d_vocab` | method | `experiments/extended_experiments/all_test_extended.py:319` | `def d_vocab(self)` |
| `d_vocab` | method | `experiments/extended_experiments/all_test_extended.py:384` | `def d_vocab(self)` |
| `detect_grokking` | method | `experiments/extended_experiments/all_test_extended.py:660` | `def detect_grokking(self, loss_threshold, test_loss_threshold, min_duration)` |
| `discretize_weights` | method | `experiments/extended_experiments/all_test_extended.py:827` | `def discretize_weights(self, model)` |
| `estimate_volume_monte_carlo` | method | `experiments/extended_experiments/all_test_extended.py:516` | `def estimate_volume_monte_carlo(self, model_class, n_samples, success_checker)` |
| `evaluate` | method | `experiments/extended_experiments/all_test_extended.py:730` | `def evaluate(self, model, dataloader)` |
| `experiment_basin_volume` | method | `experiments/extended_experiments/all_test_extended.py:1145` | `def experiment_basin_volume(self)` |
| `experiment_batch_size_mechanism` | method | `experiments/extended_experiments/all_test_extended.py:891` | `def experiment_batch_size_mechanism(self)` |
| `experiment_failure_analysis` | method | `experiments/extended_experiments/all_test_extended.py:1026` | `def experiment_failure_analysis(self)` |
| `experiment_fragility` | method | `experiments/extended_experiments/all_test_extended.py:1217` | `def experiment_fragility(self)` |
| `experiment_generalization` | method | `experiments/extended_experiments/all_test_extended.py:1088` | `def experiment_generalization(self)` |
| `experiment_hardware_reproducibility` | method | `experiments/extended_experiments/all_test_extended.py:1158` | `def experiment_hardware_reproducibility(self)` |
| `experiment_kappa_intervention` | method | `experiments/extended_experiments/all_test_extended.py:962` | `def experiment_kappa_intervention(self)` |
| `failure` | method | `experiments/extended_experiments/all_test_extended.py:131` | `def failure(self, reason, details)` |
| `fgsm_attack` | method | `experiments/extended_experiments/all_test_extended.py:536` | `def fgsm_attack(self, x, y, epsilon)` |
| `forward` | method | `experiments/extended_experiments/all_test_extended.py:265` | `def forward(self, x)` |
| `generate_dataset` | method | `experiments/extended_experiments/all_test_extended.py:303` | `def generate_dataset(self, size)` |
| `generate_dataset` | method | `experiments/extended_experiments/all_test_extended.py:322` | `def generate_dataset(self, size)` |
| `generate_dataset` | method | `experiments/extended_experiments/all_test_extended.py:387` | `def generate_dataset(self, size)` |
| `get_U_weights` | method | `experiments/extended_experiments/all_test_extended.py:283` | `def get_U_weights(self)` |
| `get_V_weights` | method | `experiments/extended_experiments/all_test_extended.py:286` | `def get_V_weights(self)` |
| `get_W_weights` | method | `experiments/extended_experiments/all_test_extended.py:289` | `def get_W_weights(self)` |
| `get_weights` | method | `experiments/extended_experiments/all_test_extended.py:277` | `def get_weights(self)` |
| `load_checkpoint` | method | `experiments/extended_experiments/all_test_extended.py:619` | `def load_checkpoint(self, path, model, optimizer)` |
| `main` | method | `experiments/extended_experiments/all_test_extended.py:1403` | `def main()` |
| `measure_basin_width` | method | `experiments/extended_experiments/all_test_extended.py:501` | `def measure_basin_width(self, weights, direction, n_points)` |
| `name` | method | `experiments/extended_experiments/all_test_extended.py:295` | `def name(self)` |
| `name` | method | `experiments/extended_experiments/all_test_extended.py:316` | `def name(self)` |
| `name` | method | `experiments/extended_experiments/all_test_extended.py:381` | `def name(self)` |
| `note` | method | `experiments/extended_experiments/all_test_extended.py:148` | `def note(self, observation)` |
| `progress` | method | `experiments/extended_experiments/all_test_extended.py:109` | `def progress(self, current, total, metrics)` |
| `progress_bar_string` | method | `experiments/extended_experiments/all_test_extended.py:679` | `def progress_bar_string(self, epoch, total_epochs)` |
| `quantize_weights` | method | `experiments/extended_experiments/all_test_extended.py:544` | `def quantize_weights(self, bits)` |
| `report` | method | `experiments/extended_experiments/all_test_extended.py:174` | `def report(self)` |
| `result` | method | `experiments/extended_experiments/all_test_extended.py:121` | `def result(self, name, value, context)` |
| `run_all_experiments` | method | `experiments/extended_experiments/all_test_extended.py:1332` | `def run_all_experiments(self)` |
| `run_fragility_analysis` | method | `experiments/extended_experiments/all_test_extended.py:560` | `def run_fragility_analysis(self, sigma_values, checker)` |
| `run_training` | method | `experiments/extended_experiments/all_test_extended.py:748` | `def run_training(self, model, train_loader, test_loader, experiment_name, epochs, batch_size, lr, wd, verbose)` |
| `save_checkpoint` | method | `experiments/extended_experiments/all_test_extended.py:593` | `def save_checkpoint(self, model, optimizer, epoch, metrics)` |
| `spectral_regularizer` | method | `experiments/extended_experiments/all_test_extended.py:472` | `def spectral_regularizer(self)` |
| `test_discretization_with_noise` | method | `experiments/extended_experiments/all_test_extended.py:550` | `def test_discretization_with_noise(self, sigma, checker)` |
| `train_epoch` | method | `experiments/extended_experiments/all_test_extended.py:708` | `def train_epoch(self, model, dataloader, optimizer)` |
| `update` | method | `experiments/extended_experiments/all_test_extended.py:642` | `def update(self, train_loss, train_acc, test_loss, test_acc, kappa, grad_norm, weight_norm, disc_margin)` |
| `verdict` | method | `experiments/extended_experiments/all_test_extended.py:126` | `def verdict(self, hypothesis, evidence, conclusion)` |
| `verify` | method | `experiments/extended_experiments/all_test_extended.py:307` | `def verify(self, model, x, y)` |
| `verify` | method | `experiments/extended_experiments/all_test_extended.py:344` | `def verify(self, model, x, y)` |
| `verify` | method | `experiments/extended_experiments/all_test_extended.py:391` | `def verify(self, model, x, y)` |
| `verify_expansion` | method | `experiments/extended_experiments/all_test_extended.py:857` | `def verify_expansion(self, model, task, sizes)` |
| `StrassenOperator` | class | `experiments/extended_experiments/exp1_covariance_spectrometry.py:48` | `class StrassenOperator(Module)` |
| `__init__` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:54` | `def __init__(self, rank)` |
| `analyze_checkpoint` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:222` | `def analyze_checkpoint(checkpoint_path, batch_sizes, n_samples, n_runs)` |
| `compute_gradient_covariance` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:120` | `def compute_gradient_covariance(model, batch_size, n_samples)` |
| `compute_per_sample_gradients` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:78` | `def compute_per_sample_gradients(self, A, B, C_true)` |
| `forward` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:61` | `def forward(self, A, B)` |
| `generate_batch` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:112` | `def generate_batch(n, scale)` |
| `generate_visualization` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:380` | `def generate_visualization(results, output_dir)` |
| `get_all_parameters` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:71` | `def get_all_parameters(self)` |
| `load_checkpoint` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:196` | `def load_checkpoint(checkpoint_path)` |
| `main` | method | `experiments/extended_experiments/exp1_covariance_spectrometry.py:277` | `def main()` |
| `setup_matplotlib` | function | `experiments/extended_experiments/exp1_covariance_spectrometry.py:22` | `def setup_matplotlib()` |
| `StrassenOperator` | class | `experiments/extended_experiments/exp2_noise_ablation.py:49` | `class StrassenOperator(Module)` |
| `__init__` | method | `experiments/extended_experiments/exp2_noise_ablation.py:54` | `def __init__(self, rank)` |
| `compute_accuracy` | method | `experiments/extended_experiments/exp2_noise_ablation.py:91` | `def compute_accuracy(self, A, B, threshold)` |
| `compute_gradient_covariance_matrix` | method | `experiments/extended_experiments/exp2_noise_ablation.py:107` | `def compute_gradient_covariance_matrix(model, n_samples, batch_size)` |
| `compute_loss` | method | `experiments/extended_experiments/exp2_noise_ablation.py:85` | `def compute_loss(self, A, B)` |
| `experiment_treatment_a_gradient_noise` | method | `experiments/extended_experiments/exp2_noise_ablation.py:172` | `def experiment_treatment_a_gradient_noise(model, noise_std, n_test)` |
| `experiment_treatment_b_weight_noise` | method | `experiments/extended_experiments/exp2_noise_ablation.py:212` | `def experiment_treatment_b_weight_noise(model, noise_std, n_test)` |
| `experiment_treatment_c_structured_noise` | method | `experiments/extended_experiments/exp2_noise_ablation.py:246` | `def experiment_treatment_c_structured_noise(model, covariance, noise_std, n_test)` |
| `forward` | method | `experiments/extended_experiments/exp2_noise_ablation.py:61` | `def forward(self, A, B)` |
| `generate_batch` | method | `experiments/extended_experiments/exp2_noise_ablation.py:99` | `def generate_batch(n, scale)` |
| `generate_visualization` | method | `experiments/extended_experiments/exp2_noise_ablation.py:433` | `def generate_visualization(results, output_dir)` |
| `get_all_parameters` | method | `experiments/extended_experiments/exp2_noise_ablation.py:71` | `def get_all_parameters(self)` |
| `get_eigenbasis` | method | `experiments/extended_experiments/exp2_noise_ablation.py:139` | `def get_eigenbasis(covariance)` |
| `load_checkpoint` | method | `experiments/extended_experiments/exp2_noise_ablation.py:150` | `def load_checkpoint(checkpoint_path)` |
| `main` | method | `experiments/extended_experiments/exp2_noise_ablation.py:348` | `def main()` |
| `run_noise_ablation` | method | `experiments/extended_experiments/exp2_noise_ablation.py:315` | `def run_noise_ablation(checkpoint_path, noise_levels)` |
| `set_parameters` | method | `experiments/extended_experiments/exp2_noise_ablation.py:77` | `def set_parameters(self, new_params)` |
| `setup_matplotlib` | function | `experiments/extended_experiments/exp2_noise_ablation.py:25` | `def setup_matplotlib()` |
| `StrassenOperator` | class | `experiments/extended_experiments/exp3_prospective_prediction.py:56` | `class StrassenOperator(Module)` |
| `__init__` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:59` | `def __init__(self, rank)` |
| `compute_discretization_margin` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:97` | `def compute_discretization_margin(self)` |
| `compute_kappa` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:121` | `def compute_kappa(model, n_samples, batch_size)` |
| `compute_roc_analysis` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:252` | `def compute_roc_analysis(predictions)` |
| `count_active_slots` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:89` | `def count_active_slots(self, threshold)` |
| `forward` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:66` | `def forward(self, A, B)` |
| `generate_batch` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:114` | `def generate_batch(n, scale)` |
| `generate_visualization` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:410` | `def generate_visualization(results, predictions, output_dir)` |
| `get_all_parameters` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:76` | `def get_all_parameters(self)` |
| `is_grokked` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:107` | `def is_grokked(self, margin_threshold, active_slots_target)` |
| `load_checkpoint` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:166` | `def load_checkpoint(checkpoint_path)` |
| `main` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:313` | `def main()` |
| `run_prospective_prediction_experiment` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:236` | `def run_prospective_prediction_experiment(checkpoint_files)` |
| `set_parameters` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:82` | `def set_parameters(self, new_params)` |
| `setup_matplotlib` | function | `experiments/extended_experiments/exp3_prospective_prediction.py:32` | `def setup_matplotlib()` |
| `simulate_early_prediction` | method | `experiments/extended_experiments/exp3_prospective_prediction.py:188` | `def simulate_early_prediction(checkpoint_path, early_epoch_fraction)` |
| `StrassenOperator` | class | `experiments/extended_experiments/exp4_trajectory_perturbation.py:51` | `class StrassenOperator(Module)` |
| `__init__` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:54` | `def __init__(self, rank)` |
| `compute_gradient_norm` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:96` | `def compute_gradient_norm(self, A, B)` |
| `compute_metrics` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:194` | `def compute_metrics(model, name)` |
| `cosine_similarity` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:110` | `def cosine_similarity(self, other_params)` |
| `forward` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:61` | `def forward(self, A, B)` |
| `generate_batch` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:117` | `def generate_batch(n, scale)` |
| `generate_visualization` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:398` | `def generate_visualization(results, output_dir)` |
| `get_all_parameters` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:71` | `def get_all_parameters(self)` |
| `get_weight_direction` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:91` | `def get_weight_direction(self)` |
| `get_weight_norm` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:84` | `def get_weight_norm(self)` |
| `load_checkpoint` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:124` | `def load_checkpoint(checkpoint_path)` |
| `main` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:277` | `def main()` |
| `set_parameters` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:77` | `def set_parameters(self, new_params)` |
| `setup_matplotlib` | function | `experiments/extended_experiments/exp4_trajectory_perturbation.py:27` | `def setup_matplotlib()` |
| `simulate_trajectory_perturbation` | method | `experiments/extended_experiments/exp4_trajectory_perturbation.py:146` | `def simulate_trajectory_perturbation(checkpoint_path, perturbations)` |
| `StrassenOperator` | class | `experiments/extended_experiments/run_all_experiments.py:48` | `class StrassenOperator(Module)` |
| `__init__` | method | `experiments/extended_experiments/run_all_experiments.py:51` | `def __init__(self, rank)` |
| `compute_accuracy` | method | `experiments/extended_experiments/run_all_experiments.py:324` | `def compute_accuracy()` |
| `compute_discretization_margin` | method | `experiments/extended_experiments/run_all_experiments.py:88` | `def compute_discretization_margin(self)` |
| `compute_gradient_covariance_safe` | method | `experiments/extended_experiments/run_all_experiments.py:135` | `def compute_gradient_covariance_safe(model, batch_size, n_samples)` |
| `count_active_slots` | method | `experiments/extended_experiments/run_all_experiments.py:81` | `def count_active_slots(self, threshold)` |
| `forward` | method | `experiments/extended_experiments/run_all_experiments.py:58` | `def forward(self, A, B)` |
| `generate_batch` | method | `experiments/extended_experiments/run_all_experiments.py:95` | `def generate_batch(n, scale)` |
| `generate_summary_visualization` | method | `experiments/extended_experiments/run_all_experiments.py:496` | `def generate_summary_visualization(results, output_dir)` |
| `get_all_parameters` | method | `experiments/extended_experiments/run_all_experiments.py:68` | `def get_all_parameters(self)` |
| `load_checkpoint_robust` | method | `experiments/extended_experiments/run_all_experiments.py:101` | `def load_checkpoint_robust(checkpoint_path, model)` |
| `run_all_experiments` | method | `experiments/extended_experiments/run_all_experiments.py:205` | `def run_all_experiments()` |
| `set_parameters` | method | `experiments/extended_experiments/run_all_experiments.py:74` | `def set_parameters(self, new_params)` |
| `setup_matplotlib` | function | `experiments/extended_experiments/run_all_experiments.py:24` | `def setup_matplotlib()` |
| `BalancedRunsGenerator` | class | `experiments/extended_experiments/validate2.py:890` | `class BalancedRunsGenerator` |
| `BootstrapStatistics` | class | `experiments/extended_experiments/validate2.py:1135` | `class BootstrapStatistics` |
| `ExperimentConfig` | class | `experiments/extended_experiments/validate2.py:60` | `class ExperimentConfig` |
| `ExperimentOrchestrator` | class | `experiments/extended_experiments/validate2.py:1646` | `class ExperimentOrchestrator` |
| `GrokkingVerifier` | class | `experiments/extended_experiments/validate2.py:340` | `class GrokkingVerifier` |
| `IterativePruningEngine` | class | `experiments/extended_experiments/validate2.py:405` | `class IterativePruningEngine` |
| `LocalComplexityCalculator` | class | `experiments/extended_experiments/validate2.py:277` | `class LocalComplexityCalculator` |
| `LocalComplexityExperiment` | class | `experiments/extended_experiments/validate2.py:764` | `class LocalComplexityExperiment` |
| `StrassenDataGenerator` | class | `experiments/extended_experiments/validate2.py:225` | `class StrassenDataGenerator` |
| `StrassenOperator` | class | `experiments/extended_experiments/validate2.py:126` | `class StrassenOperator(Module)` |
| `VisualizationGenerator` | class | `experiments/extended_experiments/validate2.py:1281` | `class VisualizationGenerator` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:139` | `def __init__(self, rank)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:228` | `def __init__(self, num_samples, matrix_size, seed)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:287` | `def __init__(self, model, config)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:343` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:420` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:775` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:903` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:1145` | `def __init__(self, config)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:1284` | `def __init__(self, style)` |
| `__init__` | method | `experiments/extended_experiments/validate2.py:1659` | `def __init__(self, config)` |
| `__post_init__` | method | `experiments/extended_experiments/validate2.py:115` | `def __post_init__(self)` |
| `_compute_roc` | method | `experiments/extended_experiments/validate2.py:1099` | `def _compute_roc(self, y_true, y_scores)` |
| `_generate_batch` | method | `experiments/extended_experiments/validate2.py:393` | `def _generate_batch(self, n, scale)` |
| `_generate_batch` | method | `experiments/extended_experiments/validate2.py:510` | `def _generate_batch(self, n, scale)` |
| `_generate_batch` | method | `experiments/extended_experiments/validate2.py:882` | `def _generate_batch(self, n, scale)` |
| `_generate_batch` | method | `experiments/extended_experiments/validate2.py:1127` | `def _generate_batch(self, n, scale)` |
| `_generate_batch` | method | `experiments/extended_experiments/validate2.py:2117` | `def _generate_batch(self, n, scale)` |
| `_initialize_weights` | method | `experiments/extended_experiments/validate2.py:150` | `def _initialize_weights(self)` |
| `_train_single_run` | method | `experiments/extended_experiments/validate2.py:1021` | `def _train_single_run(self, run_idx, config)` |
| `analyze_checkpoints` | method | `experiments/extended_experiments/validate2.py:2353` | `def analyze_checkpoints()` |
| `compute_SP` | method | `experiments/extended_experiments/validate2.py:194` | `def compute_SP(self)` |
| `compute_accuracy_with_ci` | method | `experiments/extended_experiments/validate2.py:1260` | `def compute_accuracy_with_ci(self, correct)` |
| `compute_batch_diversity` | method | `experiments/extended_experiments/validate2.py:326` | `def compute_batch_diversity(self, batch_inputs)` |
| `compute_kappa_with_ci` | method | `experiments/extended_experiments/validate2.py:1236` | `def compute_kappa_with_ci(self, y_true, y_pred)` |
| `compute_lc` | method | `experiments/extended_experiments/validate2.py:292` | `def compute_lc(self, batch_inputs, batch_targets)` |
| `compute_roc_with_ci` | method | `experiments/extended_experiments/validate2.py:1149` | `def compute_roc_with_ci(self, y_true, y_scores)` |
| `compute_sparsity` | method | `experiments/extended_experiments/validate2.py:431` | `def compute_sparsity(self, model)` |
| `count_active` | method | `experiments/extended_experiments/validate2.py:190` | `def count_active(self, threshold)` |
| `find_grokked_checkpoint` | method | `experiments/extended_experiments/validate2.py:1689` | `def find_grokked_checkpoint(self)` |
| `find_grokked_checkpoint` | method | `experiments/extended_experiments/validate2.py:2313` | `def find_grokked_checkpoint()` |
| `fine_tune` | method | `experiments/extended_experiments/validate2.py:457` | `def fine_tune(self, model, train_data)` |
| `forward` | method | `experiments/extended_experiments/validate2.py:156` | `def forward(self, A, B)` |
| `generate_data` | method | `experiments/extended_experiments/validate2.py:242` | `def generate_data(self)` |
| `generate_matrix` | method | `experiments/extended_experiments/validate2.py:238` | `def generate_matrix(self)` |
| `generate_summary_report` | method | `experiments/extended_experiments/validate2.py:2124` | `def generate_summary_report(self)` |
| `get_state_dict` | method | `experiments/extended_experiments/validate2.py:202` | `def get_state_dict(self)` |
| `get_train_test` | method | `experiments/extended_experiments/validate2.py:260` | `def get_train_test(self, test_ratio)` |
| `get_weight_magnitudes` | method | `experiments/extended_experiments/validate2.py:426` | `def get_weight_magnitudes(self, model)` |
| `load_grokked_checkpoint` | method | `experiments/extended_experiments/validate2.py:1733` | `def load_grokked_checkpoint(self, checkpoint_path)` |
| `load_state_dict` | method | `experiments/extended_experiments/validate2.py:211` | `def load_state_dict(self, state_dict)` |
| `main` | method | `experiments/extended_experiments/validate2.py:2423` | `def main()` |
| `plot_balanced_runs_results` | method | `experiments/extended_experiments/validate2.py:1467` | `def plot_balanced_runs_results(self, balanced_data, save_path)` |
| `plot_discretization_results` | method | `experiments/extended_experiments/validate2.py:1556` | `def plot_discretization_results(self, pruning_data, save_path)` |
| `plot_local_complexity` | method | `experiments/extended_experiments/validate2.py:1298` | `def plot_local_complexity(self, epochs, lc_values, accuracy, save_path)` |
| `plot_pruning_results` | method | `experiments/extended_experiments/validate2.py:1342` | `def plot_pruning_results(self, pruning_data, save_path)` |
| `plot_roc_with_ci` | method | `experiments/extended_experiments/validate2.py:1404` | `def plot_roc_with_ci(self, roc_data, save_path)` |
| `prune_percent` | method | `experiments/extended_experiments/validate2.py:437` | `def prune_percent(self, model, percent)` |
| `run_all_experiments` | method | `experiments/extended_experiments/validate2.py:2235` | `def run_all_experiments(self, checkpoint_path)` |
| `run_balanced_experiments` | method | `experiments/extended_experiments/validate2.py:907` | `def run_balanced_experiments(self, n_runs)` |
| `run_balanced_runs_experiment` | method | `experiments/extended_experiments/validate2.py:1980` | `def run_balanced_runs_experiment(self, n_runs)` |
| `run_full_experiment` | method | `experiments/extended_experiments/validate2.py:779` | `def run_full_experiment(self, target_epochs)` |
| `run_lc_training_experiment` | method | `experiments/extended_experiments/validate2.py:1897` | `def run_lc_training_experiment(self, epochs)` |
| `run_local_complexity_experiment` | method | `experiments/extended_experiments/validate2.py:1795` | `def run_local_complexity_experiment(self, epochs)` |
| `run_protocol` | method | `experiments/extended_experiments/validate2.py:624` | `def run_protocol(self, model, train_data)` |
| `run_pruning_experiment` | method | `experiments/extended_experiments/validate2.py:1936` | `def run_pruning_experiment(self)` |
| `run_roc_analysis` | method | `experiments/extended_experiments/validate2.py:2043` | `def run_roc_analysis(self)` |
| `save_results` | method | `experiments/extended_experiments/validate2.py:2197` | `def save_results(self)` |
| `slot_importance` | method | `experiments/extended_experiments/validate2.py:183` | `def slot_importance(self)` |
| `verify` | method | `experiments/extended_experiments/validate2.py:347` | `def verify(self, model, n_test)` |
| `verify_checkpoint_is_grokked` | method | `experiments/extended_experiments/validate2.py:1768` | `def verify_checkpoint_is_grokked(self)` |
| `generate_ablation_figure` | function | `experiments/generate_figures.py:124` | `def generate_ablation_figure()` |
| `generate_benchmark_figure` | function | `experiments/generate_figures.py:63` | `def generate_benchmark_figure()` |
| `generate_coherence_figure` | function | `experiments/generate_figures.py:465` | `def generate_coherence_figure()` |
| `generate_crystallization_figure` | function | `experiments/generate_figures.py:534` | `def generate_crystallization_figure()` |
| `generate_phase_transition_figure` | function | `experiments/generate_figures.py:354` | `def generate_phase_transition_figure()` |
| `generate_weight_geometry_figure` | function | `experiments/generate_figures.py:258` | `def generate_weight_geometry_figure()` |
| `load_checkpoint_weights` | function | `experiments/generate_figures.py:219` | `def load_checkpoint_weights()` |
| `main` | function | `experiments/generate_figures.py:623` | `def main()` |
| `setup_matplotlib_for_plotting` | function | `experiments/generate_figures.py:15` | `def setup_matplotlib_for_plotting()` |
| `run_coherence_analysis` | function | `experiments/statistics/coherence_analysis.py:42` | `def run_coherence_analysis()` |
| `strassen_numpy` | function | `experiments/statistics/coherence_analysis.py:15` | `def strassen_numpy(A, B, threshold)` |
| `ExperimentConfig` | class | `experiments/statistics/rigorous_experiment.py:58` | `class ExperimentConfig` |
| `ExperimentResult` | class | `experiments/statistics/rigorous_experiment.py:84` | `class ExperimentResult` |
| `StrassenModel` | class | `experiments/statistics/rigorous_experiment.py:106` | `class StrassenModel(Module)` |
| `__init__` | method | `experiments/statistics/rigorous_experiment.py:108` | `def __init__(self, config)` |
| `cache_miss_proxy` | method | `experiments/statistics/rigorous_experiment.py:467` | `def cache_miss_proxy(B)` |
| `compute_discretization_error` | method | `experiments/statistics/rigorous_experiment.py:143` | `def compute_discretization_error(model, values)` |
| `compute_spectral_gap` | method | `experiments/statistics/rigorous_experiment.py:157` | `def compute_spectral_gap(model)` |
| `find_optimal_B` | method | `experiments/statistics/rigorous_experiment.py:519` | `def find_optimal_B(results, n_bootstrap)` |
| `fit_noise_model` | method | `experiments/statistics/rigorous_experiment.py:448` | `def fit_noise_model(results)` |
| `forward` | method | `experiments/statistics/rigorous_experiment.py:124` | `def forward(self, x)` |
| `full_model` | method | `experiments/statistics/rigorous_experiment.py:472` | `def full_model(B, alpha, beta, gamma)` |
| `generate_data` | method | `experiments/statistics/rigorous_experiment.py:131` | `def generate_data(n_samples, seed)` |
| `generate_report` | method | `experiments/statistics/rigorous_experiment.py:555` | `def generate_report(results, config)` |
| `get_mean_error` | method | `experiments/statistics/rigorous_experiment.py:526` | `def get_mean_error(data, B)` |
| `null_model` | method | `experiments/statistics/rigorous_experiment.py:476` | `def null_model(B, alpha, gamma)` |
| `perform_anova` | method | `experiments/statistics/rigorous_experiment.py:306` | `def perform_anova(results)` |
| `print_anova_table` | method | `experiments/statistics/rigorous_experiment.py:401` | `def print_anova_table(anova)` |
| `run_full_experiment` | method | `experiments/statistics/rigorous_experiment.py:269` | `def run_full_experiment(batch_sizes, n_seeds, n_runs_per_seed)` |
| `run_single_experiment` | method | `experiments/statistics/rigorous_experiment.py:169` | `def run_single_experiment(batch_size, seed, run_id, config)` |
| `measure_single_sgemm` | function | `experiments/validation/benchmark.py:49` | `def measure_single_sgemm(n, threads)` |
| `run_planck_analysis` | function | `experiments/validation/benchmark.py:59` | `def run_planck_analysis()` |
| `strassen_numpy` | function | `experiments/validation/benchmark.py:15` | `def strassen_numpy(A, B, threshold)` |
| `compute_cache_math` | function | `experiments/validation_experiments.py:250` | `def compute_cache_math()` |
| `convert_types` | function | `experiments/validation_experiments.py:313` | `def convert_types(obj)` |
| `main` | function | `experiments/validation_experiments.py:294` | `def main()` |
| `simulate_grokking_dynamics` | function | `experiments/validation_experiments.py:184` | `def simulate_grokking_dynamics()` |
| `strassen_2x2` | function | `experiments/validation_experiments.py:47` | `def strassen_2x2(A, B, U, V, W)` |
| `strassen_recursive` | function | `experiments/validation_experiments.py:56` | `def strassen_recursive(A, B, U, V, W, threshold)` |
| `test_expansion_sizes` | function | `experiments/validation_experiments.py:162` | `def test_expansion_sizes()` |
| `test_noise_stability` | function | `experiments/validation_experiments.py:125` | `def test_noise_stability()` |
| `test_uniqueness_via_permutation` | function | `experiments/validation_experiments.py:85` | `def test_uniqueness_via_permutation()` |
| `StrassenBilinear` | class | `experiments/verify_checkpoints.py:25` | `class StrassenBilinear(Module)` |
| `__init__` | method | `experiments/verify_checkpoints.py:27` | `def __init__(self, rank)` |
| `compute_S_theta` | method | `experiments/verify_checkpoints.py:152` | `def compute_S_theta(model)` |
| `compute_delta` | method | `experiments/verify_checkpoints.py:51` | `def compute_delta(model)` |
| `forward` | method | `experiments/verify_checkpoints.py:34` | `def forward(self, A, B)` |
| `get_discrete_coefficients` | method | `experiments/verify_checkpoints.py:44` | `def get_discrete_coefficients(self)` |
| `load_checkpoint` | method | `experiments/verify_checkpoints.py:166` | `def load_checkpoint(path)` |
| `main` | method | `experiments/verify_checkpoints.py:249` | `def main()` |
| `run_noise_stability_test` | method | `experiments/verify_checkpoints.py:226` | `def run_noise_stability_test(checkpoint_path, noise_levels)` |
| `strassen_expand` | method | `experiments/verify_checkpoints.py:89` | `def strassen_expand(A, B, U, V, W)` |
| `verify_2x2` | method | `experiments/verify_checkpoints.py:68` | `def verify_2x2(U, V, W, n_test)` |
| `verify_checkpoint` | method | `experiments/verify_checkpoints.py:183` | `def verify_checkpoint(checkpoint_path)` |
| `verify_expansion` | method | `experiments/verify_checkpoints.py:126` | `def verify_expansion(U, V, W, sizes)` |
| `CheckpointManager` | class | `experimetn2.py:190` | `class CheckpointManager` |
| `ComplexStrassStrassenModel` | class | `experimetn2.py:135` | `class ComplexStrassStrassenModel(Module)` |
| `ExactHessianCalculator` | class | `experimetn2.py:244` | `class ExactHessianCalculator` |
| `Experiment1RicciMBLDuality` | class | `experimetn2.py:393` | `class Experiment1RicciMBLDuality(IExperiment)` |
| `Experiment2AltlandZirnbauer` | class | `experimetn2.py:457` | `class Experiment2AltlandZirnbauer(IExperiment)` |
| `Experiment3ConformalIsomorphism` | class | `experimetn2.py:520` | `class Experiment3ConformalIsomorphism(IExperiment)` |
| `Experiment4CompressionFrontier` | class | `experimetn2.py:570` | `class Experiment4CompressionFrontier(IExperiment)` |
| `Experiment5HolographicPruning` | class | `experimetn2.py:624` | `class Experiment5HolographicPruning(IExperiment)` |
| `IExperiment` | class | `experimetn2.py:385` | `class IExperiment(ABC)` |
| `LevelSpacingRatioCalculator` | class | `experimetn2.py:207` | `class LevelSpacingRatioCalculator` |
| `StrassStrassenConfig` | class | `experimetn2.py:53` | `class StrassStrassenConfig` |
| `StrassStrassenModel` | class | `experimetn2.py:95` | `class StrassStrassenModel(Module)` |
| `StrassenDataGenerator` | class | `experimetn2.py:178` | `class StrassenDataGenerator` |
| `SuiteConfig` | class | `experimetn2.py:84` | `class SuiteConfig` |
| `SuperpositionMetricCalculator` | class | `experimetn2.py:328` | `class SuperpositionMetricCalculator` |
| `SyntheticPlanckCalculator` | class | `experimetn2.py:278` | `class SyntheticPlanckCalculator` |
| `TrainingConfig` | class | `experimetn2.py:71` | `class TrainingConfig` |
| `UnifiedSuite` | class | `experimetn2.py:699` | `class UnifiedSuite` |
| `__init__` | method | `experimetn2.py:101` | `def __init__(self, config)` |
| `__init__` | method | `experimetn2.py:140` | `def __init__(self, config, gamma)` |
| `__init__` | method | `experimetn2.py:180` | `def __init__(self, config)` |
| `__init__` | method | `experimetn2.py:209` | `def __init__(self, tolerance)` |
| `__init__` | method | `experimetn2.py:246` | `def __init__(self, config)` |
| `__init__` | method | `experimetn2.py:280` | `def __init__(self, noise_floor)` |
| `__init__` | method | `experimetn2.py:330` | `def __init__(self, config)` |
| `__init__` | method | `experimetn2.py:399` | `def __init__(self, suite_config, datagen)` |
| `__init__` | method | `experimetn2.py:463` | `def __init__(self, suite_config, datagen)` |
| `__init__` | method | `experimetn2.py:527` | `def __init__(self, suite_config, datagen)` |
| `__init__` | method | `experimetn2.py:576` | `def __init__(self, suite_config, datagen)` |
| `__init__` | method | `experimetn2.py:630` | `def __init__(self, suite_config, datagen)` |
| `__init__` | method | `experimetn2.py:701` | `def __init__(self, config)` |
| `__post_init__` | method | `experimetn2.py:63` | `def __post_init__(self)` |
| `calculate` | method | `experimetn2.py:283` | `def calculate(self, model, current_loss)` |
| `calculate` | method | `experimetn2.py:333` | `def calculate(self, model, datagen)` |
| `calculate_r_ratio` | method | `experimetn2.py:212` | `def calculate_r_ratio(self, eigenvalues)` |
| `compute_hessian` | method | `experimetn2.py:249` | `def compute_hessian(self, model, A, B, C_true)` |
| `execute_all` | method | `experimetn2.py:712` | `def execute_all(self)` |
| `forward` | method | `experimetn2.py:115` | `def forward(self, A, B)` |
| `forward` | method | `experimetn2.py:160` | `def forward(self, A, B)` |
| `generate_batch` | method | `experimetn2.py:183` | `def generate_batch(self, batch_size)` |
| `get_coefficients` | method | `experimetn2.py:125` | `def get_coefficients(self)` |
| `get_complex_tensors` | method | `experimetn2.py:153` | `def get_complex_tensors(self)` |
| `get_name` | method | `experimetn2.py:390` | `def get_name(self)` |
| `get_name` | method | `experimetn2.py:405` | `def get_name(self)` |
| `get_name` | method | `experimetn2.py:468` | `def get_name(self)` |
| `get_name` | method | `experimetn2.py:531` | `def get_name(self)` |
| `get_name` | method | `experimetn2.py:582` | `def get_name(self)` |
| `get_name` | method | `experimetn2.py:634` | `def get_name(self)` |
| `loss_fn` | method | `experimetn2.py:254` | `def loss_fn(flat_param_tensor)` |
| `main` | method | `experimetn2.py:745` | `def main()` |
| `run` | method | `experimetn2.py:388` | `def run(self, model)` |
| `run` | method | `experimetn2.py:408` | `def run(self, model)` |
| `run` | method | `experimetn2.py:471` | `def run(self, model)` |
| `run` | method | `experimetn2.py:534` | `def run(self, model)` |
| `run` | method | `experimetn2.py:585` | `def run(self, model)` |
| `run` | method | `experimetn2.py:637` | `def run(self, model)` |
| `save` | method | `experimetn2.py:192` | `def save(self, model, epoch, metrics, path)` |
| `slot_importance` | method | `experimetn2.py:128` | `def slot_importance(self)` |
| `BandStructureCalculator` | class | `fermi.py:135` | `class BandStructureCalculator` |
| `BilinearModel` | class | `fermi.py:83` | `class BilinearModel(Module)` |
| `BlochWaveConstructor` | class | `fermi.py:111` | `class BlochWaveConstructor` |
| `CheckpointMigrator` | class | `fermi.py:408` | `class CheckpointMigrator` |
| `DensityOfStatesCalculator` | class | `fermi.py:285` | `class DensityOfStatesCalculator` |
| `ElectronicPropertiesCalculator` | class | `fermi.py:306` | `class ElectronicPropertiesCalculator` |
| `FermiConfig` | class | `fermi.py:19` | `class FermiConfig` |
| `FermiLevelAnalyzer` | class | `fermi.py:458` | `class FermiLevelAnalyzer` |
| `FermiLevelCalculator` | class | `fermi.py:215` | `class FermiLevelCalculator` |
| `FermiPipeline` | class | `fermi.py:608` | `class FermiPipeline` |
| `IBandStructureCalculator` | class | `fermi.py:59` | `class IBandStructureCalculator(Protocol)` |
| `IBlochWaveConstructor` | class | `fermi.py:54` | `class IBlochWaveConstructor(Protocol)` |
| `IDensityOfStatesCalculator` | class | `fermi.py:69` | `class IDensityOfStatesCalculator(Protocol)` |
| `IElectronicPropertiesCalculator` | class | `fermi.py:74` | `class IElectronicPropertiesCalculator(Protocol)` |
| `IFermiLevelCalculator` | class | `fermi.py:64` | `class IFermiLevelCalculator(Protocol)` |
| `IMetalInsulatorClassifier` | class | `fermi.py:79` | `class IMetalInsulatorClassifier(Protocol)` |
| `IModel` | class | `fermi.py:49` | `class IModel(Protocol)` |
| `MetalInsulatorClassifier` | class | `fermi.py:368` | `class MetalInsulatorClassifier` |
| `__init__` | method | `fermi.py:84` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `fermi.py:112` | `def __init__(self, config)` |
| `__init__` | method | `fermi.py:136` | `def __init__(self, config)` |
| `__init__` | method | `fermi.py:216` | `def __init__(self, config)` |
| `__init__` | method | `fermi.py:286` | `def __init__(self, config)` |
| `__init__` | method | `fermi.py:307` | `def __init__(self, config)` |
| `__init__` | method | `fermi.py:369` | `def __init__(self, config)` |
| `__init__` | method | `fermi.py:459` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `fermi.py:609` | `def __init__(self, config)` |
| `_calculate_band_gap` | method | `fermi.py:171` | `def _calculate_band_gap(self, band_structure)` |
| `_calculate_chemical_potential` | method | `fermi.py:240` | `def _calculate_chemical_potential(self, eigenvalues, num_electrons)` |
| `_calculate_compressibility` | method | `fermi.py:357` | `def _calculate_compressibility(self, eigenvalues, fermi_level)` |
| `_calculate_effective_masses` | method | `fermi.py:188` | `def _calculate_effective_masses(self, k_points, band_structure, valence_idx, conduction_idx)` |
| `_calculate_electronic_pressure` | method | `fermi.py:347` | `def _calculate_electronic_pressure(self, eigenvalues, fermi_level)` |
| `_calculate_kinetic_energy` | method | `fermi.py:337` | `def _calculate_kinetic_energy(self, occupied_states)` |
| `_fermi_dirac` | method | `fermi.py:273` | `def _fermi_dirac(self, energy, mu, temperature)` |
| `_find_chemical_potential_iterative` | method | `fermi.py:253` | `def _find_chemical_potential_iterative(self, eigenvalues, num_electrons, temperature, max_iter)` |
| `_gaussian` | method | `fermi.py:302` | `def _gaussian(self, x, mu, sigma)` |
| `_generate_text_report` | method | `fermi.py:686` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize` | method | `fermi.py:95` | `def _initialize(self)` |
| `_is_direct_gap` | method | `fermi.py:208` | `def _is_direct_gap(self, band_structure, valence_idx, conduction_idx)` |
| `_load_checkpoint` | method | `fermi.py:472` | `def _load_checkpoint(self)` |
| `_migrate_coefs_format` | method | `fermi.py:447` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `fermi.py:428` | `def _migrate_custom_format(self, state_dict, device)` |
| `_migrate_dict` | method | `fermi.py:419` | `def _migrate_dict(self, state_dict, device)` |
| `_migrate_standard_format` | method | `fermi.py:454` | `def _migrate_standard_format(self, state_dict)` |
| `_print_report` | method | `fermi.py:550` | `def _print_report(self, results)` |
| `analyze` | method | `fermi.py:495` | `def analyze(self)` |
| `calculate` | method | `fermi.py:60` | `def calculate(self, model)` |
| `calculate` | method | `fermi.py:65` | `def calculate(self, eigenvalues, num_electrons)` |
| `calculate` | method | `fermi.py:70` | `def calculate(self, eigenvalues, energies)` |
| `calculate` | method | `fermi.py:75` | `def calculate(self, eigenvalues, eigenvectors, fermi_level)` |
| `calculate` | method | `fermi.py:140` | `def calculate(self, model)` |
| `calculate` | method | `fermi.py:219` | `def calculate(self, eigenvalues, num_electrons)` |
| `calculate` | method | `fermi.py:289` | `def calculate(self, eigenvalues, energies)` |
| `calculate` | method | `fermi.py:310` | `def calculate(self, eigenvalues, eigenvectors, fermi_level)` |
| `classify` | method | `fermi.py:80` | `def classify(self, band_gap, dos_at_fermi)` |
| `classify` | method | `fermi.py:372` | `def classify(self, band_gap, dos_at_fermi)` |
| `classify_transport` | method | `fermi.py:386` | `def classify_transport(self, effective_masses, band_gap)` |
| `construct` | method | `fermi.py:55` | `def construct(self, weights, k)` |
| `construct` | method | `fermi.py:115` | `def construct(self, weights, k)` |
| `forward` | method | `fermi.py:100` | `def forward(self, a, b)` |
| `generate_summary` | method | `fermi.py:653` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `fermi.py:50` | `def get_coefficients(self)` |
| `get_coefficients` | method | `fermi.py:103` | `def get_coefficients(self)` |
| `main` | method | `fermi.py:766` | `def main()` |
| `migrate` | method | `fermi.py:409` | `def migrate(self, raw_data, device)` |
| `plot_band_structures` | method | `fermi.py:724` | `def plot_band_structures(self, all_results, output_dir)` |
| `process_checkpoint` | method | `fermi.py:612` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `fermi.py:626` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `AccuracyCalculator` | class | `full_seed_prospector.py:333` | `class AccuracyCalculator(IMetricCalculator)` |
| `AdaptiveQuantizationLoss` | class | `full_seed_prospector.py:802` | `class AdaptiveQuantizationLoss(ILossComponent)` |
| `CanonicalStrassenProvider` | class | `full_seed_prospector.py:1588` | `class CanonicalStrassenProvider` |
| `CheckpointManager` | class | `full_seed_prospector.py:884` | `class CheckpointManager(ICheckpointManager)` |
| `CoefficientDiscretizationPhase` | class | `full_seed_prospector.py:1442` | `class CoefficientDiscretizationPhase(ITrainingPhase)` |
| `ComprehensiveMetricsAggregator` | class | `full_seed_prospector.py:694` | `class ComprehensiveMetricsAggregator` |
| `DeltaCalculator` | class | `full_seed_prospector.py:313` | `class DeltaCalculator(IMetricCalculator)` |
| `DynamicBatchSizeScheduler` | class | `full_seed_prospector.py:947` | `class DynamicBatchSizeScheduler` |
| `ExecutionMode` | class | `full_seed_prospector.py:46` | `class ExecutionMode(Enum)` |
| `GeometricLossAggregator` | class | `full_seed_prospector.py:868` | `class GeometricLossAggregator(ILossComponent)` |
| `GlassDetector` | class | `full_seed_prospector.py:969` | `class GlassDetector` |
| `GradientMetricsCalculator` | class | `full_seed_prospector.py:573` | `class GradientMetricsCalculator` |
| `ICheckpointManager` | class | `full_seed_prospector.py:179` | `class ICheckpointManager(ABC)` |
| `ILossComponent` | class | `full_seed_prospector.py:170` | `class ILossComponent(ABC)` |
| `IMetricCalculator` | class | `full_seed_prospector.py:162` | `class IMetricCalculator(ABC)` |
| `ITrainingPhase` | class | `full_seed_prospector.py:195` | `class ITrainingPhase(ABC)` |
| `KappaCalculator` | class | `full_seed_prospector.py:371` | `class KappaCalculator` |
| `LocalComplexityCalculator` | class | `full_seed_prospector.py:1849` | `class LocalComplexityCalculator(IMetricCalculator)` |
| `LongTrainingPhase` | class | `full_seed_prospector.py:1152` | `class LongTrainingPhase(ITrainingPhase)` |
| `LongTrainingPipeline` | class | `full_seed_prospector.py:1742` | `class LongTrainingPipeline` |
| `MatrixDataGenerator` | class | `full_seed_prospector.py:930` | `class MatrixDataGenerator` |
| `PerelmanEntropyCalculator` | class | `full_seed_prospector.py:455` | `class PerelmanEntropyCalculator(IMetricCalculator)` |
| `ProgressiveSparsificationPhase` | class | `full_seed_prospector.py:1335` | `class ProgressiveSparsificationPhase(ITrainingPhase)` |
| `ProspectorPhase` | class | `full_seed_prospector.py:1025` | `class ProspectorPhase(ITrainingPhase)` |
| `ResilienceSpectrometer` | class | `full_seed_prospector.py:599` | `class ResilienceSpectrometer` |
| `RicciCurvaturePenalty` | class | `full_seed_prospector.py:854` | `class RicciCurvaturePenalty(ILossComponent)` |
| `SeedProspector` | class | `full_seed_prospector.py:1623` | `class SeedProspector` |
| `SparsityCalculator` | class | `full_seed_prospector.py:553` | `class SparsityCalculator(IMetricCalculator)` |
| `StrassenOperator` | class | `full_seed_prospector.py:203` | `class StrassenOperator(Module)` |
| `StrassenVerifier` | class | `full_seed_prospector.py:1532` | `class StrassenVerifier` |
| `SuperpositionCalculator` | class | `full_seed_prospector.py:1915` | `class SuperpositionCalculator(IMetricCalculator)` |
| `ThermodynamicMetricsCalculator` | class | `full_seed_prospector.py:1964` | `class ThermodynamicMetricsCalculator(IMetricCalculator)` |
| `UnifiedConfig` | class | `full_seed_prospector.py:52` | `class UnifiedConfig` |
| `__init__` | method | `full_seed_prospector.py:206` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:316` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:336` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:374` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:458` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:556` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:602` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:697` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:805` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:857` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:871` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:887` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:933` | `def __init__(self, config, scale)` |
| `__init__` | method | `full_seed_prospector.py:950` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:972` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1028` | `def __init__(self, config, seed)` |
| `__init__` | method | `full_seed_prospector.py:1155` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1338` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1445` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1535` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1626` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1745` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1852` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1918` | `def __init__(self, config)` |
| `__init__` | method | `full_seed_prospector.py:1967` | `def __init__(self, config)` |
| `_calculate_log_W` | method | `full_seed_prospector.py:522` | `def _calculate_log_W(self, tau, R, grad_f_sq, f, n_params)` |
| `_final_refinement` | method | `full_seed_prospector.py:1407` | `def _final_refinement(self, model, optimizer, slots_to_prune)` |
| `_get_adaptive_weight` | method | `full_seed_prospector.py:835` | `def _get_adaptive_weight(self, epoch, kappa)` |
| `_initialize_sae` | method | `full_seed_prospector.py:1923` | `def _initialize_sae(self, input_dim, device)` |
| `_initialize_strassen_structure` | method | `full_seed_prospector.py:220` | `def _initialize_strassen_structure(self)` |
| `_prune_by_magnitude` | method | `full_seed_prospector.py:684` | `def _prune_by_magnitude(self, model, sparsity)` |
| `_set_seed` | method | `full_seed_prospector.py:1733` | `def _set_seed(self, seed)` |
| `_set_seed` | method | `full_seed_prospector.py:1763` | `def _set_seed(self, seed)` |
| `_signal_handler` | method | `full_seed_prospector.py:1758` | `def _signal_handler(self, signum, frame)` |
| `accumulate_gradient` | method | `full_seed_prospector.py:379` | `def accumulate_gradient(self, model)` |
| `accumulate_gradient` | method | `full_seed_prospector.py:787` | `def accumulate_gradient(self, model)` |
| `calculate` | method | `full_seed_prospector.py:166` | `def calculate(self)` |
| `calculate` | method | `full_seed_prospector.py:319` | `def calculate(self, model)` |
| `calculate` | method | `full_seed_prospector.py:339` | `def calculate(self, model, C_pred, C_true, n_test)` |
| `calculate` | method | `full_seed_prospector.py:465` | `def calculate(self, model, loss, epoch, gradient_norm)` |
| `calculate` | method | `full_seed_prospector.py:559` | `def calculate(self, model)` |
| `calculate` | method | `full_seed_prospector.py:577` | `def calculate(model)` |
| `calculate` | method | `full_seed_prospector.py:1855` | `def calculate(self, model)` |
| `calculate` | method | `full_seed_prospector.py:1930` | `def calculate(self, model)` |
| `calculate` | method | `full_seed_prospector.py:1970` | `def calculate(self, model, gradient_covariance)` |
| `calculate_kappa` | method | `full_seed_prospector.py:393` | `def calculate_kappa(self)` |
| `compute` | method | `full_seed_prospector.py:174` | `def compute(self, model, loss_mse, epoch)` |
| `compute` | method | `full_seed_prospector.py:808` | `def compute(self, model, loss_mse, epoch, kappa)` |
| `compute` | method | `full_seed_prospector.py:860` | `def compute(self, model, loss_mse, epoch)` |
| `compute` | method | `full_seed_prospector.py:876` | `def compute(self, model, loss_mse, epoch, kappa)` |
| `compute_all` | method | `full_seed_prospector.py:711` | `def compute_all(self, model, C_pred, C_true, loss, epoch, force_kappa, force_lc, force_sp)` |
| `count_active` | method | `full_seed_prospector.py:298` | `def count_active(self, threshold)` |
| `execute` | method | `full_seed_prospector.py:199` | `def execute(self, model)` |
| `execute` | method | `full_seed_prospector.py:1037` | `def execute(self, model)` |
| `execute` | method | `full_seed_prospector.py:1164` | `def execute(self, model)` |
| `execute` | method | `full_seed_prospector.py:1342` | `def execute(self, model)` |
| `execute` | method | `full_seed_prospector.py:1448` | `def execute(self, model)` |
| `forward` | method | `full_seed_prospector.py:276` | `def forward(self, A, B)` |
| `generate_batch` | method | `full_seed_prospector.py:938` | `def generate_batch(self, n)` |
| `get_batch_size` | method | `full_seed_prospector.py:953` | `def get_batch_size(self, epoch)` |
| `get_canonical` | method | `full_seed_prospector.py:1592` | `def get_canonical()` |
| `get_flat_parameters` | method | `full_seed_prospector.py:304` | `def get_flat_parameters(self)` |
| `get_kappa_trend` | method | `full_seed_prospector.py:425` | `def get_kappa_trend(self)` |
| `get_latest_checkpoint_path` | method | `full_seed_prospector.py:924` | `def get_latest_checkpoint_path(self)` |
| `get_parameter_count` | method | `full_seed_prospector.py:308` | `def get_parameter_count(self)` |
| `is_crystallizing` | method | `full_seed_prospector.py:439` | `def is_crystallizing(self)` |
| `load` | method | `full_seed_prospector.py:187` | `def load(self, path)` |
| `load` | method | `full_seed_prospector.py:911` | `def load(self, path)` |
| `main` | method | `full_seed_prospector.py:2022` | `def main()` |
| `measure` | method | `full_seed_prospector.py:605` | `def measure(self, model)` |
| `prospect` | method | `full_seed_prospector.py:1631` | `def prospect(self, total_attempts, start_seed)` |
| `reset` | method | `full_seed_prospector.py:449` | `def reset(self)` |
| `reset` | method | `full_seed_prospector.py:546` | `def reset(self)` |
| `reset` | method | `full_seed_prospector.py:795` | `def reset(self)` |
| `run` | method | `full_seed_prospector.py:1771` | `def run(self, resume_from, seed)` |
| `save` | method | `full_seed_prospector.py:183` | `def save(self, state, path)` |
| `save` | method | `full_seed_prospector.py:893` | `def save(self, state, path)` |
| `should_checkpoint` | method | `full_seed_prospector.py:191` | `def should_checkpoint(self)` |
| `should_checkpoint` | method | `full_seed_prospector.py:919` | `def should_checkpoint(self)` |
| `should_stop` | method | `full_seed_prospector.py:977` | `def should_stop(self, epoch, metrics)` |
| `slot_importance` | method | `full_seed_prospector.py:291` | `def slot_importance(self)` |
| `update_lr` | method | `full_seed_prospector.py:791` | `def update_lr(self, lr)` |
| `verify` | method | `full_seed_prospector.py:1538` | `def verify(self, U, V, W, n_test)` |
| `BilinearStrassenModel` | class | `grain.py:102` | `class BilinearStrassenModel(Module)` |
| `CheckpointManager` | class | `grain.py:309` | `class CheckpointManager` |
| `DomainFragmentationAnalyzer` | class | `grain.py:243` | `class DomainFragmentationAnalyzer` |
| `GrainBoundaryAnalyzer` | class | `grain.py:547` | `class GrainBoundaryAnalyzer` |
| `GrainBoundaryDetector` | class | `grain.py:153` | `class GrainBoundaryDetector` |
| `GrainBoundaryPipeline` | class | `grain.py:726` | `class GrainBoundaryPipeline` |
| `ICheckpointManager` | class | `grain.py:91` | `class ICheckpointManager(Protocol)` |
| `IDislocationCalculator` | class | `grain.py:81` | `class IDislocationCalculator(Protocol)` |
| `IDomainFragmentationAnalyzer` | class | `grain.py:86` | `class IDomainFragmentationAnalyzer(Protocol)` |
| `IGrainBoundaryDetector` | class | `grain.py:71` | `class IGrainBoundaryDetector(Protocol)` |
| `ILayerAnalyzer` | class | `grain.py:76` | `class ILayerAnalyzer(Protocol)` |
| `IModel` | class | `grain.py:65` | `class IModel(Protocol)` |
| `ITrainingMonitor` | class | `grain.py:97` | `class ITrainingMonitor(Protocol)` |
| `LayerAnalyzer` | class | `grain.py:130` | `class LayerAnalyzer` |
| `StrassenConfig` | class | `grain.py:25` | `class StrassenConfig` |
| `StrassenTrainer` | class | `grain.py:407` | `class StrassenTrainer` |
| `TrainingMetricsTracker` | class | `grain.py:340` | `class TrainingMetricsTracker` |
| `__init__` | method | `grain.py:103` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `grain.py:154` | `def __init__(self, config)` |
| `__init__` | method | `grain.py:244` | `def __init__(self, config)` |
| `__init__` | method | `grain.py:310` | `def __init__(self, config)` |
| `__init__` | method | `grain.py:341` | `def __init__(self, config)` |
| `__init__` | method | `grain.py:408` | `def __init__(self, config, seed)` |
| `__init__` | method | `grain.py:548` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `grain.py:727` | `def __init__(self, config)` |
| `_analyze_dislocation_evolution` | method | `grain.py:658` | `def _analyze_dislocation_evolution(self, grain_results)` |
| `_analyze_fragmentation` | method | `grain.py:227` | `def _analyze_fragmentation(self, layer_analysis)` |
| `_calculate_coherence_length` | method | `grain.py:297` | `def _calculate_coherence_length(self, layer_analysis)` |
| `_calculate_coordination_loss` | method | `grain.py:269` | `def _calculate_coordination_loss(self, layer_analysis)` |
| `_calculate_dislocation` | method | `grain.py:194` | `def _calculate_dislocation(self, layer_deltas)` |
| `_compute_accuracy` | method | `grain.py:468` | `def _compute_accuracy(self, pred, target)` |
| `_estimate_domain_count` | method | `grain.py:282` | `def _estimate_domain_count(self, layer_analysis)` |
| `_find_critical_pruning_level` | method | `grain.py:681` | `def _find_critical_pruning_level(self, grain_results)` |
| `_generate_batch` | method | `grain.py:447` | `def _generate_batch(self)` |
| `_generate_text_report` | method | `grain.py:797` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize_symmetric` | method | `grain.py:114` | `def _initialize_symmetric(self)` |
| `_load_checkpoint` | method | `grain.py:557` | `def _load_checkpoint(self)` |
| `_migrate_checkpoint` | method | `grain.py:580` | `def _migrate_checkpoint(self, raw_data)` |
| `_migrate_coefs_format` | method | `grain.py:618` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `grain.py:599` | `def _migrate_custom_format(self, state_dict)` |
| `_migrate_dict` | method | `grain.py:590` | `def _migrate_dict(self, state_dict)` |
| `_migrate_standard_format` | method | `grain.py:625` | `def _migrate_standard_format(self, state_dict)` |
| `_print_report` | method | `grain.py:687` | `def _print_report(self, results)` |
| `_prune_model` | method | `grain.py:185` | `def _prune_model(self, model, sparsity)` |
| `_save_checkpoint` | method | `grain.py:473` | `def _save_checkpoint(self, interrupted)` |
| `_setup_signal_handlers` | method | `grain.py:438` | `def _setup_signal_handlers(self)` |
| `_signal_handler` | method | `grain.py:442` | `def _signal_handler(self, signum, frame)` |
| `analyze` | method | `grain.py:87` | `def analyze(self, model, pruning_level)` |
| `analyze` | method | `grain.py:248` | `def analyze(self, model, pruning_level)` |
| `analyze` | method | `grain.py:628` | `def analyze(self)` |
| `analyze_layer` | method | `grain.py:77` | `def analyze_layer(self, weights, layer_name)` |
| `analyze_layer` | method | `grain.py:131` | `def analyze_layer(self, weights, layer_name)` |
| `calculate` | method | `grain.py:82` | `def calculate(self, layer_deltas)` |
| `detect` | method | `grain.py:72` | `def detect(self, model, pruning_level)` |
| `detect` | method | `grain.py:158` | `def detect(self, model, pruning_level)` |
| `forward` | method | `grain.py:66` | `def forward(self, a, b)` |
| `forward` | method | `grain.py:119` | `def forward(self, a, b)` |
| `generate_summary` | method | `grain.py:771` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `grain.py:67` | `def get_coefficients(self)` |
| `get_coefficients` | method | `grain.py:122` | `def get_coefficients(self)` |
| `get_current_metrics` | method | `grain.py:382` | `def get_current_metrics(self)` |
| `get_training_bar_string` | method | `grain.py:389` | `def get_training_bar_string(self, epoch, total_epochs)` |
| `load` | method | `grain.py:93` | `def load(self, path)` |
| `load` | method | `grain.py:332` | `def load(self, path)` |
| `main` | method | `grain.py:845` | `def main()` |
| `process_checkpoint` | method | `grain.py:730` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `grain.py:744` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `run_analysis` | method | `grain.py:838` | `def run_analysis(checkpoint_dir, output_dir, n_latest, config)` |
| `run_training` | method | `grain.py:833` | `def run_training(seed, config)` |
| `save` | method | `grain.py:92` | `def save(self, model, epoch, metrics, path)` |
| `save` | method | `grain.py:314` | `def save(self, model, epoch, metrics, path)` |
| `should_checkpoint` | method | `grain.py:99` | `def should_checkpoint(self)` |
| `should_save` | method | `grain.py:335` | `def should_save(self)` |
| `to_dict` | method | `grain.py:57` | `def to_dict(self)` |
| `train` | method | `grain.py:486` | `def train(self)` |
| `update` | method | `grain.py:98` | `def update(self, epoch, metrics)` |
| `update` | method | `grain.py:358` | `def update(self, epoch, loss, accuracy, model, grain_result)` |
| `BasinStabilityCalculator` | class | `gravity.py:327` | `class BasinStabilityCalculator` |
| `BilinearModel` | class | `gravity.py:114` | `class BilinearModel(Module)` |
| `ConditionNumberCalculator` | class | `gravity.py:398` | `class ConditionNumberCalculator` |
| `ConfigurationEntropyCalculator` | class | `gravity.py:154` | `class ConfigurationEntropyCalculator` |
| `GravitationalConstantCalculator` | class | `gravity.py:183` | `class GravitationalConstantCalculator` |
| `HeisenbergUncertaintyCalculator` | class | `gravity.py:271` | `class HeisenbergUncertaintyCalculator` |
| `IBasinStabilityCalculator` | class | `gravity.py:100` | `class IBasinStabilityCalculator(Protocol)` |
| `IConditionNumberCalculator` | class | `gravity.py:110` | `class IConditionNumberCalculator(Protocol)` |
| `IEntropyCalculator` | class | `gravity.py:70` | `class IEntropyCalculator(Protocol)` |
| `IGravitationalConstantCalculator` | class | `gravity.py:80` | `class IGravitationalConstantCalculator(Protocol)` |
| `IHeisenbergUncertaintyCalculator` | class | `gravity.py:90` | `class IHeisenbergUncertaintyCalculator(Protocol)` |
| `ILandauerConstantCalculator` | class | `gravity.py:85` | `class ILandauerConstantCalculator(Protocol)` |
| `ILocalComplexityCalculator` | class | `gravity.py:95` | `class ILocalComplexityCalculator(Protocol)` |
| `IModel` | class | `gravity.py:59` | `class IModel(Protocol)` |
| `IOrderParameterCalculator` | class | `gravity.py:65` | `class IOrderParameterCalculator(Protocol)` |
| `ISpecificHeatCalculator` | class | `gravity.py:75` | `class ISpecificHeatCalculator(Protocol)` |
| `IZeroShotTransferCalculator` | class | `gravity.py:105` | `class IZeroShotTransferCalculator(Protocol)` |
| `LandauerConstantCalculator` | class | `gravity.py:239` | `class LandauerConstantCalculator` |
| `LocalComplexityCalculator` | class | `gravity.py:310` | `class LocalComplexityCalculator` |
| `OrderParameterCalculator` | class | `gravity.py:142` | `class OrderParameterCalculator` |
| `PhaseTransitionDetector` | class | `gravity.py:454` | `class PhaseTransitionDetector` |
| `SpecificHeatCalculator` | class | `gravity.py:171` | `class SpecificHeatCalculator` |
| `ThermodynamicAnalyzer` | class | `gravity.py:505` | `class ThermodynamicAnalyzer` |
| `ThermodynamicConfig` | class | `gravity.py:24` | `class ThermodynamicConfig` |
| `ThermodynamicPipeline` | class | `gravity.py:837` | `class ThermodynamicPipeline` |
| `ZeroShotTransferCalculator` | class | `gravity.py:365` | `class ZeroShotTransferCalculator` |
| `__init__` | method | `gravity.py:115` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `gravity.py:143` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:155` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:172` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:184` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:240` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:272` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:311` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:328` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:366` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:399` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:455` | `def __init__(self, config)` |
| `__init__` | method | `gravity.py:506` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `gravity.py:838` | `def __init__(self, config)` |
| `_compute_emergent_constants` | method | `gravity.py:904` | `def _compute_emergent_constants(self, results)` |
| `_compute_kappa_correlation` | method | `gravity.py:993` | `def _compute_kappa_correlation(self, results)` |
| `_compute_static_gradient` | method | `gravity.py:598` | `def _compute_static_gradient(self)` |
| `_determine_failure_mode` | method | `gravity.py:733` | `def _determine_failure_mode(self, delta, basin, kappa, transition)` |
| `_generate_test_data` | method | `gravity.py:619` | `def _generate_test_data(self)` |
| `_generate_text_report` | method | `gravity.py:1023` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize` | method | `gravity.py:126` | `def _initialize(self)` |
| `_kronecker_recursive` | method | `gravity.py:391` | `def _kronecker_recursive(self, matrix, power)` |
| `_load_checkpoint` | method | `gravity.py:525` | `def _load_checkpoint(self)` |
| `_migrate_checkpoint` | method | `gravity.py:548` | `def _migrate_checkpoint(self, raw_data)` |
| `_migrate_coefs_format` | method | `gravity.py:588` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `gravity.py:569` | `def _migrate_custom_format(self, state_dict)` |
| `_migrate_dict` | method | `gravity.py:560` | `def _migrate_dict(self, state_dict)` |
| `_migrate_standard_format` | method | `gravity.py:595` | `def _migrate_standard_format(self, state_dict)` |
| `_print_report` | method | `gravity.py:744` | `def _print_report(self, results)` |
| `_prune_model` | method | `gravity.py:355` | `def _prune_model(self, model, sparsity)` |
| `_verify_universal_laws` | method | `gravity.py:959` | `def _verify_universal_laws(self, results)` |
| `analyze` | method | `gravity.py:630` | `def analyze(self)` |
| `calculate` | method | `gravity.py:66` | `def calculate(self, model)` |
| `calculate` | method | `gravity.py:71` | `def calculate(self, model)` |
| `calculate` | method | `gravity.py:76` | `def calculate(self, loss_history)` |
| `calculate` | method | `gravity.py:81` | `def calculate(self, model, gradient_history)` |
| `calculate` | method | `gravity.py:86` | `def calculate(self, entropy_change, energy_dissipated)` |
| `calculate` | method | `gravity.py:91` | `def calculate(self, model, temperature)` |
| `calculate` | method | `gravity.py:96` | `def calculate(self, model)` |
| `calculate` | method | `gravity.py:101` | `def calculate(self, model, test_data)` |
| `calculate` | method | `gravity.py:106` | `def calculate(self, model, target_size)` |
| `calculate` | method | `gravity.py:111` | `def calculate(self, gradient_covariance)` |
| `calculate` | method | `gravity.py:146` | `def calculate(self, model)` |
| `calculate` | method | `gravity.py:158` | `def calculate(self, model)` |
| `calculate` | method | `gravity.py:175` | `def calculate(self, loss_history)` |
| `calculate` | method | `gravity.py:187` | `def calculate(self, model, gradient_history, loss_history, static_gradient)` |
| `calculate` | method | `gravity.py:243` | `def calculate(self, entropy_change, energy_dissipated, has_transition, transition_window)` |
| `calculate` | method | `gravity.py:275` | `def calculate(self, model, temperature, static_gradient)` |
| `calculate` | method | `gravity.py:314` | `def calculate(self, model)` |
| `calculate` | method | `gravity.py:331` | `def calculate(self, model, test_data)` |
| `calculate` | method | `gravity.py:369` | `def calculate(self, model, target_size)` |
| `calculate` | method | `gravity.py:403` | `def calculate(self, gradient_history, static_gradient)` |
| `detect` | method | `gravity.py:458` | `def detect(self, loss_history, entropy_history)` |
| `extract_values` | method | `gravity.py:908` | `def extract_values(data_list, key_path)` |
| `forward` | method | `gravity.py:60` | `def forward(self, a, b)` |
| `forward` | method | `gravity.py:131` | `def forward(self, a, b)` |
| `generate_summary` | method | `gravity.py:882` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `gravity.py:61` | `def get_coefficients(self)` |
| `get_coefficients` | method | `gravity.py:134` | `def get_coefficients(self)` |
| `main` | method | `gravity.py:1114` | `def main()` |
| `process_checkpoint` | method | `gravity.py:841` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `gravity.py:855` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `BilinearStrassenModel` | class | `grigori_perelmans_ricci_flow.py:82` | `class BilinearStrassenModel(Module)` |
| `CheckpointMigrationManager` | class | `grigori_perelmans_ricci_flow.py:172` | `class CheckpointMigrationManager` |
| `CheckpointMigrator` | class | `grigori_perelmans_ricci_flow.py:145` | `class CheckpointMigrator(ABC)` |
| `CustomFormatMigrator` | class | `grigori_perelmans_ricci_flow.py:151` | `class CustomFormatMigrator(CheckpointMigrator)` |
| `GeometricPlanckCalculator` | class | `grigori_perelmans_ricci_flow.py:366` | `class GeometricPlanckCalculator` |
| `RicciConfig` | class | `grigori_perelmans_ricci_flow.py:34` | `class RicciConfig` |
| `RicciFlowAnalyzer` | class | `grigori_perelmans_ricci_flow.py:193` | `class RicciFlowAnalyzer` |
| `RicciFlowAnalyzerPipeline` | class | `grigori_perelmans_ricci_flow.py:430` | `class RicciFlowAnalyzerPipeline` |
| `SingularityEngine` | class | `grigori_perelmans_ricci_flow.py:310` | `class SingularityEngine` |
| `StandardFormatMigrator` | class | `grigori_perelmans_ricci_flow.py:166` | `class StandardFormatMigrator(CheckpointMigrator)` |
| `StrassenDataGenerator` | class | `grigori_perelmans_ricci_flow.py:128` | `class StrassenDataGenerator` |
| `__init__` | method | `grigori_perelmans_ricci_flow.py:86` | `def __init__(self, config)` |
| `__init__` | method | `grigori_perelmans_ricci_flow.py:173` | `def __init__(self)` |
| `__init__` | method | `grigori_perelmans_ricci_flow.py:199` | `def __init__(self, model, config)` |
| `__init__` | method | `grigori_perelmans_ricci_flow.py:316` | `def __init__(self, model, eigenvalues, config)` |
| `__init__` | method | `grigori_perelmans_ricci_flow.py:371` | `def __init__(self, eigenvalues, ricci_scalar, config)` |
| `__init__` | method | `grigori_perelmans_ricci_flow.py:435` | `def __init__(self, config)` |
| `_compute_diagonal_hessian` | method | `grigori_perelmans_ricci_flow.py:240` | `def _compute_diagonal_hessian(self, a, b, c)` |
| `_compute_spectral_entropy` | method | `grigori_perelmans_ricci_flow.py:420` | `def _compute_spectral_entropy(self)` |
| `_get_spectral_gap` | method | `grigori_perelmans_ricci_flow.py:412` | `def _get_spectral_gap(self)` |
| `_initialize` | method | `grigori_perelmans_ricci_flow.py:94` | `def _initialize(self)` |
| `_loss_wrapper` | method | `grigori_perelmans_ricci_flow.py:223` | `def _loss_wrapper(self, flat_params, original_params, a, b, c)` |
| `_print_summary` | method | `grigori_perelmans_ricci_flow.py:560` | `def _print_summary(self, report)` |
| `analyze_checkpoint` | method | `grigori_perelmans_ricci_flow.py:439` | `def analyze_checkpoint(self, checkpoint_path, device)` |
| `analyze_curvature` | method | `grigori_perelmans_ricci_flow.py:247` | `def analyze_curvature(self, hessian)` |
| `analyze_directory` | method | `grigori_perelmans_ricci_flow.py:527` | `def analyze_directory(self, directory, device, pattern)` |
| `calculate` | method | `grigori_perelmans_ricci_flow.py:376` | `def calculate(self)` |
| `can_migrate` | method | `grigori_perelmans_ricci_flow.py:147` | `def can_migrate(self, state_dict)` |
| `can_migrate` | method | `grigori_perelmans_ricci_flow.py:152` | `def can_migrate(self, state_dict)` |
| `can_migrate` | method | `grigori_perelmans_ricci_flow.py:167` | `def can_migrate(self, state_dict)` |
| `compute_heat_kernel_trace` | method | `grigori_perelmans_ricci_flow.py:285` | `def compute_heat_kernel_trace(self, eigenvalues, t)` |
| `compute_hessian` | method | `grigori_perelmans_ricci_flow.py:203` | `def compute_hessian(self, input_a, input_b, target_c)` |
| `compute_topological_entropy` | method | `grigori_perelmans_ricci_flow.py:296` | `def compute_topological_entropy(self, eigenvalues)` |
| `detect_necks` | method | `grigori_perelmans_ricci_flow.py:325` | `def detect_necks(self, curvature_analysis)` |
| `forward` | method | `grigori_perelmans_ricci_flow.py:99` | `def forward(self, a, b)` |
| `generate_batch` | method | `grigori_perelmans_ricci_flow.py:130` | `def generate_batch(batch_size, config)` |
| `get_coefficients` | method | `grigori_perelmans_ricci_flow.py:102` | `def get_coefficients(self)` |
| `get_flat_params` | method | `grigori_perelmans_ricci_flow.py:109` | `def get_flat_params(self)` |
| `main` | method | `grigori_perelmans_ricci_flow.py:593` | `def main()` |
| `migrate` | method | `grigori_perelmans_ricci_flow.py:149` | `def migrate(self, state_dict)` |
| `migrate` | method | `grigori_perelmans_ricci_flow.py:154` | `def migrate(self, state_dict)` |
| `migrate` | method | `grigori_perelmans_ricci_flow.py:169` | `def migrate(self, state_dict)` |
| `migrate_checkpoint` | method | `grigori_perelmans_ricci_flow.py:176` | `def migrate_checkpoint(self, path, device)` |
| `propose_surgery` | method | `grigori_perelmans_ricci_flow.py:336` | `def propose_surgery(self)` |
| `set_flat_params` | method | `grigori_perelmans_ricci_flow.py:114` | `def set_flat_params(self, flat_params)` |
| `set_random_seed` | method | `grigori_perelmans_ricci_flow.py:70` | `def set_random_seed(seed)` |
| `BilinearStrassenModel` | class | `hawking_radiation.py:197` | `class BilinearStrassenModel(Module)` |
| `BoltzmannConstantCalculator` | class | `hawking_radiation.py:768` | `class BoltzmannConstantCalculator` |
| `CustomUnpickler` | class | `hawking_radiation.py:36` | `class CustomUnpickler(Unpickler)` |
| `DummyClass` | class | `hawking_radiation.py:64` | `class DummyClass` |
| `GravitationalConstantCalculator` | class | `hawking_radiation.py:609` | `class GravitationalConstantCalculator` |
| `HawkingConfiguration` | class | `hawking_radiation.py:138` | `class HawkingConfiguration` |
| `HawkingRadiationCalculator` | class | `hawking_radiation.py:991` | `class HawkingRadiationCalculator` |
| `HorizonAreaCalculator` | class | `hawking_radiation.py:939` | `class HorizonAreaCalculator` |
| `IModel` | class | `hawking_radiation.py:188` | `class IModel(Protocol)` |
| `InformationalMassCalculator` | class | `hawking_radiation.py:883` | `class InformationalMassCalculator` |
| `MetadataExtractor` | class | `hawking_radiation.py:514` | `class MetadataExtractor` |
| `PlanckConstantCalculator` | class | `hawking_radiation.py:676` | `class PlanckConstantCalculator` |
| `RobustCheckpointMigrator` | class | `hawking_radiation.py:234` | `class RobustCheckpointMigrator` |
| `RobustHawkingAnalyzer` | class | `hawking_radiation.py:1166` | `class RobustHawkingAnalyzer` |
| `SpeedOfLightCalculator` | class | `hawking_radiation.py:823` | `class SpeedOfLightCalculator` |
| `__getitem__` | method | `hawking_radiation.py:75` | `def __getitem__(self, key)` |
| `__init__` | method | `hawking_radiation.py:65` | `def __init__(self)` |
| `__init__` | method | `hawking_radiation.py:200` | `def __init__(self, config)` |
| `__init__` | method | `hawking_radiation.py:612` | `def __init__(self, config)` |
| `__init__` | method | `hawking_radiation.py:679` | `def __init__(self, config)` |
| `__init__` | method | `hawking_radiation.py:771` | `def __init__(self, config)` |
| `__init__` | method | `hawking_radiation.py:826` | `def __init__(self, config)` |
| `__init__` | method | `hawking_radiation.py:886` | `def __init__(self, config)` |
| `__init__` | method | `hawking_radiation.py:942` | `def __init__(self, config)` |
| `__init__` | method | `hawking_radiation.py:994` | `def __init__(self, config)` |
| `__init__` | method | `hawking_radiation.py:1169` | `def __init__(self, config)` |
| `__repr__` | method | `hawking_radiation.py:72` | `def __repr__(self)` |
| `_classify_state` | method | `hawking_radiation.py:1151` | `def _classify_state(self, delta, T_hawking)` |
| `_compute_gradient` | method | `hawking_radiation.py:1227` | `def _compute_gradient(self, model)` |
| `_create_dummy_class` | method | `hawking_radiation.py:62` | `def _create_dummy_class(self, name)` |
| `_extract_delta` | method | `hawking_radiation.py:571` | `def _extract_delta(data, depth)` |
| `_generate_summary` | method | `hawking_radiation.py:1342` | `def _generate_summary(self, results, errors)` |
| `_initialize` | method | `hawking_radiation.py:211` | `def _initialize(self)` |
| `_is_state_dict` | method | `hawking_radiation.py:386` | `def _is_state_dict(self, data)` |
| `_migrate_coefs_format` | method | `hawking_radiation.py:449` | `def _migrate_coefs_format(self, state_dict, device)` |
| `_migrate_custom_format` | method | `hawking_radiation.py:424` | `def _migrate_custom_format(self, state_dict, device)` |
| `_migrate_dict` | method | `hawking_radiation.py:398` | `def _migrate_dict(self, state_dict, device)` |
| `_migrate_encoder_format` | method | `hawking_radiation.py:466` | `def _migrate_encoder_format(self, state_dict, device)` |
| `_migrate_prefixed_format` | method | `hawking_radiation.py:496` | `def _migrate_prefixed_format(self, state_dict, prefix, device)` |
| `_migrate_standard_format` | method | `hawking_radiation.py:456` | `def _migrate_standard_format(self, state_dict, device)` |
| `_print_report` | method | `hawking_radiation.py:1255` | `def _print_report(self, results)` |
| `_reconstruct_from_tensors` | method | `hawking_radiation.py:338` | `def _reconstruct_from_tensors(self, tensors, device)` |
| `_try_direct_tensor_extraction` | method | `hawking_radiation.py:313` | `def _try_direct_tensor_extraction(self, data, device)` |
| `_try_extract_state_dict` | method | `hawking_radiation.py:269` | `def _try_extract_state_dict(self, data, device)` |
| `_try_nested_extraction` | method | `hawking_radiation.py:288` | `def _try_nested_extraction(self, data, device)` |
| `analyze_checkpoint` | method | `hawking_radiation.py:1174` | `def analyze_checkpoint(self, checkpoint_path)` |
| `analyze_directory` | method | `hawking_radiation.py:1295` | `def analyze_directory(self, checkpoint_dir, output_dir, pattern)` |
| `calculate` | method | `hawking_radiation.py:615` | `def calculate(self, model, gradient, precomputed_delta)` |
| `calculate` | method | `hawking_radiation.py:682` | `def calculate(self, model, loss, precomputed_delta)` |
| `calculate` | method | `hawking_radiation.py:774` | `def calculate(self, model, loss, loss_history)` |
| `calculate` | method | `hawking_radiation.py:829` | `def calculate(self, model, h_bar, G_alg)` |
| `calculate` | method | `hawking_radiation.py:889` | `def calculate(self, model, G_alg, c_eff, h_bar)` |
| `calculate` | method | `hawking_radiation.py:945` | `def calculate(self, model, M_eff)` |
| `calculate_all` | method | `hawking_radiation.py:1003` | `def calculate_all(self, model, loss, loss_history, gradient, precomputed_delta)` |
| `extract` | method | `hawking_radiation.py:518` | `def extract(checkpoint)` |
| `extract_tensors` | method | `hawking_radiation.py:317` | `def extract_tensors(obj, prefix)` |
| `find_class` | method | `hawking_radiation.py:44` | `def find_class(self, module, name)` |
| `forward` | method | `hawking_radiation.py:190` | `def forward(self, a, b)` |
| `forward` | method | `hawking_radiation.py:216` | `def forward(self, a, b)` |
| `get` | method | `hawking_radiation.py:87` | `def get(self, key, default)` |
| `get_coefficients` | method | `hawking_radiation.py:189` | `def get_coefficients(self)` |
| `get_coefficients` | method | `hawking_radiation.py:219` | `def get_coefficients(self)` |
| `get_effective_input_dim` | method | `hawking_radiation.py:172` | `def get_effective_input_dim(self)` |
| `get_flat_parameters` | method | `hawking_radiation.py:226` | `def get_flat_parameters(self)` |
| `get_total_parameters` | method | `hawking_radiation.py:175` | `def get_total_parameters(self)` |
| `items` | method | `hawking_radiation.py:84` | `def items(self)` |
| `keys` | method | `hawking_radiation.py:78` | `def keys(self)` |
| `load_checkpoint_robust` | method | `hawking_radiation.py:93` | `def load_checkpoint_robust(path, device)` |
| `main` | method | `hawking_radiation.py:1406` | `def main()` |
| `migrate` | method | `hawking_radiation.py:245` | `def migrate(self, raw_data, device)` |
| `values` | method | `hawking_radiation.py:81` | `def values(self)` |
| `BandgapAnalyzer` | class | `maxwell_strassen_analysis.py:498` | `class BandgapAnalyzer` |
| `BilinearStrassenModel` | class | `maxwell_strassen_analysis.py:145` | `class BilinearStrassenModel(Module)` |
| `CheckpointManager` | class | `maxwell_strassen_analysis.py:605` | `class CheckpointManager` |
| `CheckpointMigrator` | class | `maxwell_strassen_analysis.py:171` | `class CheckpointMigrator` |
| `CrystalPhaseClassifier` | class | `maxwell_strassen_analysis.py:544` | `class CrystalPhaseClassifier` |
| `DielectricTensorAnalyzer` | class | `maxwell_strassen_analysis.py:288` | `class DielectricTensorAnalyzer` |
| `IDielectricAnalyzer` | class | `maxwell_strassen_analysis.py:132` | `class IDielectricAnalyzer(Protocol)` |
| `IGeometryMapper` | class | `maxwell_strassen_analysis.py:121` | `class IGeometryMapper(Protocol)` |
| `IMaxwellSolver` | class | `maxwell_strassen_analysis.py:126` | `class IMaxwellSolver(Protocol)` |
| `IModel` | class | `maxwell_strassen_analysis.py:116` | `class IModel(Protocol)` |
| `IPhaseClassifier` | class | `maxwell_strassen_analysis.py:137` | `class IPhaseClassifier(Protocol)` |
| `MaxwellAnalyzer` | class | `maxwell_strassen_analysis.py:677` | `class MaxwellAnalyzer` |
| `MaxwellConfiguration` | class | `maxwell_strassen_analysis.py:42` | `class MaxwellConfiguration` |
| `MaxwellScatteringSolver` | class | `maxwell_strassen_analysis.py:350` | `class MaxwellScatteringSolver` |
| `MaxwellVisualizer` | class | `maxwell_strassen_analysis.py:625` | `class MaxwellVisualizer` |
| `PhotonicEntropyCalculator` | class | `maxwell_strassen_analysis.py:466` | `class PhotonicEntropyCalculator` |
| `StrassenGeometryMapper` | class | `maxwell_strassen_analysis.py:227` | `class StrassenGeometryMapper` |
| `__init__` | method | `maxwell_strassen_analysis.py:146` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:241` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:295` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:360` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:474` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:505` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:548` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:606` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:626` | `def __init__(self, config)` |
| `__init__` | method | `maxwell_strassen_analysis.py:678` | `def __init__(self, config)` |
| `_calculate_purity_metrics` | method | `maxwell_strassen_analysis.py:690` | `def _calculate_purity_metrics(self, weights)` |
| `_find_peaks` | method | `maxwell_strassen_analysis.py:438` | `def _find_peaks(self, intensity)` |
| `_initialize_weights` | method | `maxwell_strassen_analysis.py:156` | `def _initialize_weights(self)` |
| `_migrate_coefs` | method | `maxwell_strassen_analysis.py:203` | `def _migrate_coefs(self, sd)` |
| `_migrate_custom` | method | `maxwell_strassen_analysis.py:194` | `def _migrate_custom(self, sd, config)` |
| `_migrate_dict` | method | `maxwell_strassen_analysis.py:185` | `def _migrate_dict(self, state_dict, config)` |
| `_migrate_standard` | method | `maxwell_strassen_analysis.py:208` | `def _migrate_standard(self, sd)` |
| `_np` | method | `maxwell_strassen_analysis.py:217` | `def _np(tensor)` |
| `analyze` | method | `maxwell_strassen_analysis.py:508` | `def analyze(self, fourier_coeffs)` |
| `analyze_checkpoint` | method | `maxwell_strassen_analysis.py:701` | `def analyze_checkpoint(self, checkpoint_path, output_dir)` |
| `analyze_permittivity_tensor` | method | `maxwell_strassen_analysis.py:133` | `def analyze_permittivity_tensor(self, permittivity)` |
| `analyze_permittivity_tensor` | method | `maxwell_strassen_analysis.py:298` | `def analyze_permittivity_tensor(self, permittivity)` |
| `calculate` | method | `maxwell_strassen_analysis.py:477` | `def calculate(self, potential, intensity)` |
| `classify` | method | `maxwell_strassen_analysis.py:138` | `def classify(self, metrics)` |
| `classify` | method | `maxwell_strassen_analysis.py:551` | `def classify(self, em_metrics, purity_metrics)` |
| `compute_scattering` | method | `maxwell_strassen_analysis.py:128` | `def compute_scattering(self, permittivity)` |
| `compute_scattering` | method | `maxwell_strassen_analysis.py:401` | `def compute_scattering(self, permittivity)` |
| `forward` | method | `maxwell_strassen_analysis.py:161` | `def forward(self, a, b)` |
| `generate_report` | method | `maxwell_strassen_analysis.py:765` | `def generate_report(self, results, output_dir)` |
| `get_coefficients` | method | `maxwell_strassen_analysis.py:117` | `def get_coefficients(self)` |
| `get_coefficients` | method | `maxwell_strassen_analysis.py:164` | `def get_coefficients(self)` |
| `get_effective_input_dim` | method | `maxwell_strassen_analysis.py:103` | `def get_effective_input_dim(self)` |
| `get_total_parameters` | method | `maxwell_strassen_analysis.py:106` | `def get_total_parameters(self)` |
| `main` | method | `maxwell_strassen_analysis.py:788` | `def main()` |
| `map_weights_to_lattice` | method | `maxwell_strassen_analysis.py:122` | `def map_weights_to_lattice(self, weights)` |
| `map_weights_to_lattice` | method | `maxwell_strassen_analysis.py:244` | `def map_weights_to_lattice(self, weights)` |
| `migrate` | method | `maxwell_strassen_analysis.py:172` | `def migrate(self, raw_data, config)` |
| `save` | method | `maxwell_strassen_analysis.py:613` | `def save(self, data, output_dir)` |
| `should_save` | method | `maxwell_strassen_analysis.py:610` | `def should_save(self)` |
| `solve_poisson` | method | `maxwell_strassen_analysis.py:127` | `def solve_poisson(self, charge_density, permittivity)` |
| `solve_poisson` | method | `maxwell_strassen_analysis.py:363` | `def solve_poisson(self, charge_density, permittivity)` |
| `visualize_lattice` | method | `maxwell_strassen_analysis.py:629` | `def visualize_lattice(self, permittivity, output_dir, name)` |
| `visualize_potential` | method | `maxwell_strassen_analysis.py:659` | `def visualize_potential(self, potential, output_dir, name)` |
| `visualize_scattering` | method | `maxwell_strassen_analysis.py:648` | `def visualize_scattering(self, scattering_slice, output_dir, name)` |
| `BilinearStrassenModel` | class | `mbl_analyzer.py:138` | `class BilinearStrassenModel(Module)` |
| `CheckpointMigrator` | class | `mbl_analyzer.py:746` | `class CheckpointMigrator` |
| `DiscretizationDialAnalyzer` | class | `mbl_analyzer.py:486` | `class DiscretizationDialAnalyzer` |
| `EffectiveTemperatureCalculator` | class | `mbl_analyzer.py:674` | `class EffectiveTemperatureCalculator` |
| `ICheckpointManager` | class | `mbl_analyzer.py:124` | `class ICheckpointManager(Protocol)` |
| `IDiscretizationDialAnalyzer` | class | `mbl_analyzer.py:118` | `class IDiscretizationDialAnalyzer(Protocol)` |
| `ILevelSpacingCalculator` | class | `mbl_analyzer.py:100` | `class ILevelSpacingCalculator(Protocol)` |
| `IModel` | class | `mbl_analyzer.py:93` | `class IModel(Protocol)` |
| `IParticipationRatioCalculator` | class | `mbl_analyzer.py:106` | `class IParticipationRatioCalculator(Protocol)` |
| `ISyntheticPlanckCalculator` | class | `mbl_analyzer.py:112` | `class ISyntheticPlanckCalculator(Protocol)` |
| `ITrainingMetricsCollector` | class | `mbl_analyzer.py:132` | `class ITrainingMetricsCollector(Protocol)` |
| `LevelSpacingRatioCalculator` | class | `mbl_analyzer.py:205` | `class LevelSpacingRatioCalculator` |
| `MBLAnalysisPipeline` | class | `mbl_analyzer.py:1102` | `class MBLAnalysisPipeline` |
| `MBLCheckpointAnalyzer` | class | `mbl_analyzer.py:962` | `class MBLCheckpointAnalyzer` |
| `MBLCheckpointManager` | class | `mbl_analyzer.py:800` | `class MBLCheckpointManager` |
| `MBLConfiguration` | class | `mbl_analyzer.py:24` | `class MBLConfiguration` |
| `MBLMetricsCollector` | class | `mbl_analyzer.py:855` | `class MBLMetricsCollector` |
| `ParticipationRatioCalculator` | class | `mbl_analyzer.py:321` | `class ParticipationRatioCalculator` |
| `PhaseClassifier` | class | `mbl_analyzer.py:721` | `class PhaseClassifier` |
| `PurityIndexCalculator` | class | `mbl_analyzer.py:614` | `class PurityIndexCalculator` |
| `SyntheticPlanckConstantCalculator` | class | `mbl_analyzer.py:420` | `class SyntheticPlanckConstantCalculator` |
| `__init__` | method | `mbl_analyzer.py:143` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:217` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:331` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:430` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:497` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:620` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:679` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:726` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:805` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:860` | `def __init__(self, config)` |
| `__init__` | method | `mbl_analyzer.py:967` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `mbl_analyzer.py:1107` | `def __init__(self, config)` |
| `_assess_purity_quality` | method | `mbl_analyzer.py:663` | `def _assess_purity_quality(self, alpha, variance)` |
| `_calculate_fractal_dimension` | method | `mbl_analyzer.py:409` | `def _calculate_fractal_dimension(self, ipr, n)` |
| `_calculate_ipr` | method | `mbl_analyzer.py:380` | `def _calculate_ipr(self, coefficients)` |
| `_calculate_renyi_ipr` | method | `mbl_analyzer.py:395` | `def _calculate_renyi_ipr(self, coefficients, q)` |
| `_calculate_spacing_ratios` | method | `mbl_analyzer.py:288` | `def _calculate_spacing_ratios(self, spacings)` |
| `_classify_phase` | method | `mbl_analyzer.py:303` | `def _classify_phase(self, mean_ratio)` |
| `_classify_quantum_phase` | method | `mbl_analyzer.py:946` | `def _classify_quantum_phase(self, level_spacing, hbar_results)` |
| `_compute_eigenvalues` | method | `mbl_analyzer.py:283` | `def _compute_eigenvalues(self, hessian)` |
| `_compute_layer_purity` | method | `mbl_analyzer.py:652` | `def _compute_layer_purity(self, weights)` |
| `_construct_hessian_from_weights` | method | `mbl_analyzer.py:267` | `def _construct_hessian_from_weights(self, model)` |
| `_delta_to_alpha` | method | `mbl_analyzer.py:607` | `def _delta_to_alpha(self, delta)` |
| `_delta_to_alpha` | method | `mbl_analyzer.py:658` | `def _delta_to_alpha(self, delta)` |
| `_generate_summary` | method | `mbl_analyzer.py:1026` | `def _generate_summary(self, metrics, robustness)` |
| `_generate_text_report` | method | `mbl_analyzer.py:1188` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize_weights` | method | `mbl_analyzer.py:154` | `def _initialize_weights(self)` |
| `_load_checkpoint` | method | `mbl_analyzer.py:975` | `def _load_checkpoint(self)` |
| `_migrate_coefs_format` | method | `mbl_analyzer.py:789` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `mbl_analyzer.py:770` | `def _migrate_custom_format(self, state_dict, device)` |
| `_migrate_dict` | method | `mbl_analyzer.py:761` | `def _migrate_dict(self, state_dict, device)` |
| `_migrate_standard_format` | method | `mbl_analyzer.py:796` | `def _migrate_standard_format(self, state_dict)` |
| `_perturb_and_measure` | method | `mbl_analyzer.py:584` | `def _perturb_and_measure(self, model, noise_level)` |
| `_print_report` | method | `mbl_analyzer.py:1043` | `def _print_report(self, results)` |
| `analyze` | method | `mbl_analyzer.py:998` | `def analyze(self)` |
| `analyze_robustness` | method | `mbl_analyzer.py:120` | `def analyze_robustness(self, model, noise_levels)` |
| `analyze_robustness` | method | `mbl_analyzer.py:528` | `def analyze_robustness(self, model, noise_levels)` |
| `calculate` | method | `mbl_analyzer.py:102` | `def calculate(self, model)` |
| `calculate` | method | `mbl_analyzer.py:108` | `def calculate(self, model)` |
| `calculate` | method | `mbl_analyzer.py:114` | `def calculate(self, participation_ratio, energy_gap)` |
| `calculate` | method | `mbl_analyzer.py:220` | `def calculate(self, model)` |
| `calculate` | method | `mbl_analyzer.py:334` | `def calculate(self, model)` |
| `calculate` | method | `mbl_analyzer.py:433` | `def calculate(self, participation_ratio, energy_gap)` |
| `calculate` | method | `mbl_analyzer.py:623` | `def calculate(self, model)` |
| `calculate` | method | `mbl_analyzer.py:682` | `def calculate(self, loss_history)` |
| `calculate_base_discretization` | method | `mbl_analyzer.py:501` | `def calculate_base_discretization(self, model)` |
| `calculate_from_model` | method | `mbl_analyzer.py:456` | `def calculate_from_model(self, model, level_spacing_results, pr_results)` |
| `classify` | method | `mbl_analyzer.py:729` | `def classify(self, alpha, temperature)` |
| `collect` | method | `mbl_analyzer.py:134` | `def collect(self, model, loss, epoch, loss_history)` |
| `collect` | method | `mbl_analyzer.py:870` | `def collect(self, model, loss, epoch, loss_history)` |
| `construct_hessian_approximation` | method | `mbl_analyzer.py:179` | `def construct_hessian_approximation(self)` |
| `forward` | method | `mbl_analyzer.py:96` | `def forward(self, a, b)` |
| `forward` | method | `mbl_analyzer.py:160` | `def forward(self, a, b)` |
| `generate_summary` | method | `mbl_analyzer.py:1155` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `mbl_analyzer.py:95` | `def get_coefficients(self)` |
| `get_coefficients` | method | `mbl_analyzer.py:164` | `def get_coefficients(self)` |
| `get_effective_input_dim` | method | `mbl_analyzer.py:84` | `def get_effective_input_dim(self)` |
| `get_flat_parameters` | method | `mbl_analyzer.py:172` | `def get_flat_parameters(self)` |
| `get_total_parameters` | method | `mbl_analyzer.py:87` | `def get_total_parameters(self)` |
| `load_checkpoint` | method | `mbl_analyzer.py:128` | `def load_checkpoint(self, path)` |
| `load_checkpoint` | method | `mbl_analyzer.py:850` | `def load_checkpoint(self, path)` |
| `main` | method | `mbl_analyzer.py:1227` | `def main()` |
| `migrate` | method | `mbl_analyzer.py:751` | `def migrate(self, raw_data, device)` |
| `process_checkpoint` | method | `mbl_analyzer.py:1110` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `mbl_analyzer.py:1125` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `save_checkpoint` | method | `mbl_analyzer.py:126` | `def save_checkpoint(self, model, epoch, metrics, loss_history, path)` |
| `save_checkpoint` | method | `mbl_analyzer.py:816` | `def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)` |
| `should_save_checkpoint` | method | `mbl_analyzer.py:810` | `def should_save_checkpoint(self)` |
| `clear_screen` | function | `menu.py:304` | `def clear_screen()` |
| `main_menu` | function | `menu.py:473` | `def main_menu()` |
| `print_header` | function | `menu.py:308` | `def print_header(title, subtitle)` |
| `print_wrapped` | function | `menu.py:318` | `def print_wrapped(text, indent)` |
| `run_script` | function | `menu.py:332` | `def run_script(entry)` |
| `show_category` | function | `menu.py:433` | `def show_category(cat)` |
| `show_checkpoints` | function | `menu.py:362` | `def show_checkpoints()` |
| `show_results` | function | `menu.py:397` | `def show_results()` |
| `wait_for_enter` | function | `menu.py:323` | `def wait_for_enter()` |
| `BilinearStrassenModel` | class | `percolation_analysis.py:117` | `class BilinearStrassenModel(Module)` |
| `BondPercolationAnalyzer` | class | `percolation_analysis.py:402` | `class BondPercolationAnalyzer` |
| `CheckpointMigrator` | class | `percolation_analysis.py:264` | `class CheckpointMigrator` |
| `ClusterSizeDistributionAnalyzer` | class | `percolation_analysis.py:725` | `class ClusterSizeDistributionAnalyzer` |
| `IModel` | class | `percolation_analysis.py:97` | `class IModel(Protocol)` |
| `NumpyModelWrapper` | class | `percolation_analysis.py:101` | `class NumpyModelWrapper` |
| `PercolationAnalysisPipeline` | class | `percolation_analysis.py:1139` | `class PercolationAnalysisPipeline` |
| `PercolationCheckpointManager` | class | `percolation_analysis.py:808` | `class PercolationCheckpointManager` |
| `PercolationConfiguration` | class | `percolation_analysis.py:31` | `class PercolationConfiguration` |
| `PercolationReportGenerator` | class | `percolation_analysis.py:1049` | `class PercolationReportGenerator` |
| `PercolationUniversalityAnalyzer` | class | `percolation_analysis.py:775` | `class PercolationUniversalityAnalyzer` |
| `PercolationVisualizationEngine` | class | `percolation_analysis.py:837` | `class PercolationVisualizationEngine` |
| `PruningPercolationAnalyzer` | class | `percolation_analysis.py:542` | `class PruningPercolationAnalyzer` |
| `SitePercolationAnalyzer` | class | `percolation_analysis.py:488` | `class SitePercolationAnalyzer` |
| `WeightGraphConstructor` | class | `percolation_analysis.py:350` | `class WeightGraphConstructor` |
| `_DummyObject` | class | `percolation_analysis.py:148` | `class _DummyObject` |
| `__init__` | method | `percolation_analysis.py:102` | `def __init__(self, weights)` |
| `__init__` | method | `percolation_analysis.py:118` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:155` | `def __init__(self)` |
| `__init__` | method | `percolation_analysis.py:351` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:403` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:489` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:543` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:726` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:776` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:809` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:838` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:1050` | `def __init__(self, config)` |
| `__init__` | method | `percolation_analysis.py:1140` | `def __init__(self, config)` |
| `__repr__` | method | `percolation_analysis.py:161` | `def __repr__(self)` |
| `_cleanup` | method | `percolation_analysis.py:230` | `def _cleanup()` |
| `_comparative_summary` | method | `percolation_analysis.py:1238` | `def _comparative_summary(self, all_res, output_dir)` |
| `_critical` | method | `percolation_analysis.py:767` | `def _critical(self, sizes, n)` |
| `_d2a` | method | `percolation_analysis.py:626` | `def _d2a(self, delta)` |
| `_entropy` | method | `percolation_analysis.py:669` | `def _entropy(self, wv)` |
| `_exponents` | method | `percolation_analysis.py:462` | `def _exponents(self, thresholds, res, pc)` |
| `_find_pc` | method | `percolation_analysis.py:456` | `def _find_pc(self, thresholds, res)` |
| `_fit_pl` | method | `percolation_analysis.py:478` | `def _fit_pl(x, y)` |
| `_fractal` | method | `percolation_analysis.py:699` | `def _fractal(self, ipr, n)` |
| `_hbar` | method | `percolation_analysis.py:644` | `def _hbar(self, wv)` |
| `_initialize_weights` | method | `percolation_analysis.py:127` | `def _initialize_weights(self)` |
| `_ipr` | method | `percolation_analysis.py:680` | `def _ipr(self, wv)` |
| `_kappa` | method | `percolation_analysis.py:631` | `def _kappa(self, wv)` |
| `_lc` | method | `percolation_analysis.py:659` | `def _lc(self, wv, delta)` |
| `_load_weights` | method | `percolation_analysis.py:1153` | `def _load_weights(self, checkpoint_path)` |
| `_lsr` | method | `percolation_analysis.py:687` | `def _lsr(self, wv)` |
| `_maybe_save` | method | `percolation_analysis.py:1234` | `def _maybe_save(self, results, output_dir)` |
| `_migrate_coefs` | method | `percolation_analysis.py:326` | `def _migrate_coefs(self, sd)` |
| `_migrate_custom` | method | `percolation_analysis.py:310` | `def _migrate_custom(self, sd, config)` |
| `_migrate_dict` | method | `percolation_analysis.py:285` | `def _migrate_dict(self, state_dict, config)` |
| `_migrate_standard` | method | `percolation_analysis.py:331` | `def _migrate_standard(self, sd)` |
| `_np` | method | `percolation_analysis.py:340` | `def _np(tensor)` |
| `_patch_missing` | method | `percolation_analysis.py:189` | `def _patch_missing(exc)` |
| `_phase` | method | `percolation_analysis.py:704` | `def _phase(self, alpha, temp, delta)` |
| `_plot_bond` | method | `percolation_analysis.py:856` | `def _plot_bond(self, data, out)` |
| `_plot_cluster` | method | `percolation_analysis.py:1019` | `def _plot_cluster(self, data, out)` |
| `_plot_dashboard` | method | `percolation_analysis.py:952` | `def _plot_dashboard(self, data, out)` |
| `_plot_pruning` | method | `percolation_analysis.py:889` | `def _plot_pruning(self, data, out)` |
| `_plot_site` | method | `percolation_analysis.py:993` | `def _plot_site(self, data, out)` |
| `_safe_torch_load` | method | `percolation_analysis.py:165` | `def _safe_torch_load(path)` |
| `_susceptibility` | method | `percolation_analysis.py:449` | `def _susceptibility(self, sizes, n)` |
| `_tau` | method | `percolation_analysis.py:753` | `def _tau(self, sizes)` |
| `_teff` | method | `percolation_analysis.py:656` | `def _teff(self, wv)` |
| `_try_migrate_nested` | method | `percolation_analysis.py:300` | `def _try_migrate_nested(self, candidate, config)` |
| `analyze` | method | `percolation_analysis.py:406` | `def analyze(self, adjacency, thresholds)` |
| `analyze` | method | `percolation_analysis.py:492` | `def analyze(self, weights, thresholds)` |
| `analyze` | method | `percolation_analysis.py:546` | `def analyze(self, weights)` |
| `analyze_at_threshold` | method | `percolation_analysis.py:729` | `def analyze_at_threshold(self, adjacency, threshold)` |
| `classify_universality` | method | `percolation_analysis.py:779` | `def classify_universality(self, measured)` |
| `construct_adjacency_from_weights` | method | `percolation_analysis.py:354` | `def construct_adjacency_from_weights(self, weights)` |
| `construct_slot_interaction_graph` | method | `percolation_analysis.py:386` | `def construct_slot_interaction_graph(self, weights)` |
| `construct_weight_correlation_graph` | method | `percolation_analysis.py:375` | `def construct_weight_correlation_graph(self, weights)` |
| `forward` | method | `percolation_analysis.py:132` | `def forward(self, a, b)` |
| `generate_all_figures` | method | `percolation_analysis.py:841` | `def generate_all_figures(self, results, output_dir)` |
| `generate_json_report` | method | `percolation_analysis.py:1131` | `def generate_json_report(self, results, output_dir)` |
| `generate_text_report` | method | `percolation_analysis.py:1053` | `def generate_text_report(self, results, output_dir)` |
| `get_coefficients` | method | `percolation_analysis.py:98` | `def get_coefficients(self)` |
| `get_coefficients` | method | `percolation_analysis.py:105` | `def get_coefficients(self)` |
| `get_coefficients` | method | `percolation_analysis.py:135` | `def get_coefficients(self)` |
| `get_effective_input_dim` | method | `percolation_analysis.py:82` | `def get_effective_input_dim(self)` |
| `get_flat_parameters` | method | `percolation_analysis.py:108` | `def get_flat_parameters(self)` |
| `get_flat_parameters` | method | `percolation_analysis.py:141` | `def get_flat_parameters(self)` |
| `get_percolation_thresholds` | method | `percolation_analysis.py:89` | `def get_percolation_thresholds(self)` |
| `get_total_parameters` | method | `percolation_analysis.py:85` | `def get_total_parameters(self)` |
| `load` | method | `percolation_analysis.py:829` | `def load(self, output_dir)` |
| `main` | method | `percolation_analysis.py:1264` | `def main()` |
| `migrate` | method | `percolation_analysis.py:265` | `def migrate(self, raw_data, config)` |
| `process_checkpoint` | method | `percolation_analysis.py:1164` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `percolation_analysis.py:1212` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `save` | method | `percolation_analysis.py:817` | `def save(self, results, output_dir)` |
| `should_save` | method | `percolation_analysis.py:814` | `def should_save(self)` |
| `BasinResilienceSpectrometer` | class | `plank.py:560` | `class BasinResilienceSpectrometer` |
| `BilinearStrassenModel` | class | `plank.py:127` | `class BilinearStrassenModel(Module)` |
| `CheckpointMigrationManager` | class | `plank.py:284` | `class CheckpointMigrationManager` |
| `CheckpointMigrator` | class | `plank.py:194` | `class CheckpointMigrator(ABC)` |
| `Configuration` | class | `plank.py:32` | `class Configuration` |
| `CrystalPurityIndex` | class | `plank.py:677` | `class CrystalPurityIndex` |
| `CrystallographyMetrics` | class | `plank.py:384` | `class CrystallographyMetrics` |
| `CustomFormatMigrator` | class | `plank.py:208` | `class CustomFormatMigrator(CheckpointMigrator)` |
| `EncoderFormatMigrator` | class | `plank.py:237` | `class EncoderFormatMigrator(CheckpointMigrator)` |
| `PlanckConstantCalculator` | class | `plank.py:763` | `class PlanckConstantCalculator` |
| `ReportGenerator` | class | `plank.py:1165` | `class ReportGenerator` |
| `StandardFormatMigrator` | class | `plank.py:270` | `class StandardFormatMigrator(CheckpointMigrator)` |
| `StrassenCheckpointLoader` | class | `plank.py:936` | `class StrassenCheckpointLoader` |
| `StrassenDataGenerator` | class | `plank.py:341` | `class StrassenDataGenerator` |
| `StrassenDiffractionTest` | class | `plank.py:483` | `class StrassenDiffractionTest` |
| `StrassenPlanckAnalyzer` | class | `plank.py:1014` | `class StrassenPlanckAnalyzer` |
| `__init__` | method | `plank.py:133` | `def __init__(self, config)` |
| `__init__` | method | `plank.py:287` | `def __init__(self)` |
| `__init__` | method | `plank.py:486` | `def __init__(self, model, config)` |
| `__init__` | method | `plank.py:563` | `def __init__(self, model, config)` |
| `__init__` | method | `plank.py:680` | `def __init__(self, metrics, diffraction_results, resilience_results, config)` |
| `__init__` | method | `plank.py:770` | `def __init__(self, metrics, training_metrics, config)` |
| `__init__` | method | `plank.py:939` | `def __init__(self, config)` |
| `__init__` | method | `plank.py:1021` | `def __init__(self, config)` |
| `__init__` | method | `plank.py:1168` | `def __init__(self, config)` |
| `__post_init__` | method | `plank.py:100` | `def __post_init__(self)` |
| `_anneal_to_attractor` | method | `plank.py:621` | `def _anneal_to_attractor(self)` |
| `_apply_noise` | method | `plank.py:614` | `def _apply_noise(self, sigma)` |
| `_assign_grade` | method | `plank.py:745` | `def _assign_grade(self, index, delta)` |
| `_compute_derived_constants` | method | `plank.py:886` | `def _compute_derived_constants(self, h_bar)` |
| `_compute_functional_error` | method | `plank.py:531` | `def _compute_functional_error(self, test_coeffs)` |
| `_compute_statistics` | method | `plank.py:1216` | `def _compute_statistics(self, summaries)` |
| `_compute_universe_comparison` | method | `plank.py:917` | `def _compute_universe_comparison(self, h_bar)` |
| `_count_grades` | method | `plank.py:1247` | `def _count_grades(self, summaries)` |
| `_determine_regime_and_weights` | method | `plank.py:875` | `def _determine_regime_and_weights(self)` |
| `_estimate_critical_noise` | method | `plank.py:654` | `def _estimate_critical_noise(self, results)` |
| `_initialize_symmetric` | method | `plank.py:143` | `def _initialize_symmetric(self)` |
| `_print_summary` | method | `plank.py:1146` | `def _print_summary(self, report)` |
| `_test_noise_recovery` | method | `plank.py:585` | `def _test_noise_recovery(self, sigma)` |
| `analyze_checkpoint` | method | `plank.py:1025` | `def analyze_checkpoint(self, checkpoint_path, device)` |
| `analyze_directory` | method | `plank.py:1104` | `def analyze_directory(self, directory, device, pattern)` |
| `calculate_all` | method | `plank.py:789` | `def calculate_all(self)` |
| `can_migrate` | method | `plank.py:198` | `def can_migrate(self, state_dict)` |
| `can_migrate` | method | `plank.py:211` | `def can_migrate(self, state_dict)` |
| `can_migrate` | method | `plank.py:240` | `def can_migrate(self, state_dict)` |
| `can_migrate` | method | `plank.py:273` | `def can_migrate(self, state_dict)` |
| `compute` | method | `plank.py:692` | `def compute(self)` |
| `compute_all_metrics` | method | `plank.py:464` | `def compute_all_metrics(model, config)` |
| `compute_discretization_margin` | method | `plank.py:429` | `def compute_discretization_margin(coeffs)` |
| `compute_kappa` | method | `plank.py:388` | `def compute_kappa(model, num_batches, config)` |
| `compute_lambda_effective` | method | `plank.py:174` | `def compute_lambda_effective(self)` |
| `compute_local_complexity` | method | `plank.py:445` | `def compute_local_complexity(model, config)` |
| `create_config_from_args` | method | `plank.py:1387` | `def create_config_from_args(args)` |
| `extract_training_metrics` | method | `plank.py:986` | `def extract_training_metrics(self, checkpoint_path)` |
| `forward` | method | `plank.py:149` | `def forward(self, matrix_a, matrix_b)` |
| `generate_batch` | method | `plank.py:345` | `def generate_batch(batch_size, config)` |
| `generate_visualizations` | method | `plank.py:1255` | `def generate_visualizations(self, results)` |
| `get_coefficients` | method | `plank.py:166` | `def get_coefficients(self)` |
| `load` | method | `plank.py:943` | `def load(self, checkpoint_path, device)` |
| `main` | method | `plank.py:1399` | `def main()` |
| `measure_resilience_spectrum` | method | `plank.py:570` | `def measure_resilience_spectrum(self)` |
| `migrate` | method | `plank.py:203` | `def migrate(self, state_dict)` |
| `migrate` | method | `plank.py:214` | `def migrate(self, state_dict)` |
| `migrate` | method | `plank.py:243` | `def migrate(self, state_dict)` |
| `migrate` | method | `plank.py:276` | `def migrate(self, state_dict)` |
| `migrate_checkpoint` | method | `plank.py:294` | `def migrate_checkpoint(self, path, device)` |
| `parse_arguments` | method | `plank.py:1339` | `def parse_arguments()` |
| `save_aggregate_report` | method | `plank.py:1185` | `def save_aggregate_report(self, results)` |
| `save_json_report` | method | `plank.py:1173` | `def save_json_report(self, report, suffix)` |
| `set_random_seed` | method | `plank.py:114` | `def set_random_seed(seed)` |
| `test_gauge_invariance` | method | `plank.py:490` | `def test_gauge_invariance(self)` |
| `verify_structure` | method | `plank.py:366` | `def verify_structure(coeffs, config)` |
| `BilinearModel` | class | `purity_index.py:73` | `class BilinearModel(Module)` |
| `CheckpointMigrator` | class | `purity_index.py:311` | `class CheckpointMigrator` |
| `EffectiveTemperatureCalculator` | class | `purity_index.py:156` | `class EffectiveTemperatureCalculator` |
| `IEffectiveTemperatureCalculator` | class | `purity_index.py:54` | `class IEffectiveTemperatureCalculator(Protocol)` |
| `IModel` | class | `purity_index.py:44` | `class IModel(Protocol)` |
| `IPhaseClassifier` | class | `purity_index.py:59` | `class IPhaseClassifier(Protocol)` |
| `IPolycrystalAnalyzer` | class | `purity_index.py:64` | `class IPolycrystalAnalyzer(Protocol)` |
| `IPurityComparator` | class | `purity_index.py:69` | `class IPurityComparator(Protocol)` |
| `IPurityIndexCalculator` | class | `purity_index.py:49` | `class IPurityIndexCalculator(Protocol)` |
| `PhaseClassifier` | class | `purity_index.py:199` | `class PhaseClassifier` |
| `PolycrystalAnalyzer` | class | `purity_index.py:236` | `class PolycrystalAnalyzer` |
| `PurityAnalyzer` | class | `purity_index.py:361` | `class PurityAnalyzer` |
| `PurityComparator` | class | `purity_index.py:274` | `class PurityComparator` |
| `PurityConfig` | class | `purity_index.py:19` | `class PurityConfig` |
| `PurityIndexCalculator` | class | `purity_index.py:101` | `class PurityIndexCalculator` |
| `PurityPipeline` | class | `purity_index.py:492` | `class PurityPipeline` |
| `__init__` | method | `purity_index.py:74` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `purity_index.py:102` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:157` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:200` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:237` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:275` | `def __init__(self, config)` |
| `__init__` | method | `purity_index.py:362` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `purity_index.py:493` | `def __init__(self, config)` |
| `_assess_purity_quality` | method | `purity_index.py:145` | `def _assess_purity_quality(self, alpha, variance)` |
| `_compute_layer_purity` | method | `purity_index.py:134` | `def _compute_layer_purity(self, weights)` |
| `_delta_to_alpha` | method | `purity_index.py:140` | `def _delta_to_alpha(self, delta)` |
| `_generate_text_report` | method | `purity_index.py:574` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize` | method | `purity_index.py:85` | `def _initialize(self)` |
| `_load_checkpoint` | method | `purity_index.py:375` | `def _load_checkpoint(self)` |
| `_migrate_coefs_format` | method | `purity_index.py:350` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `purity_index.py:331` | `def _migrate_custom_format(self, state_dict, device)` |
| `_migrate_dict` | method | `purity_index.py:322` | `def _migrate_dict(self, state_dict, device)` |
| `_migrate_standard_format` | method | `purity_index.py:357` | `def _migrate_standard_format(self, state_dict)` |
| `_print_report` | method | `purity_index.py:445` | `def _print_report(self, results)` |
| `_prune_model` | method | `purity_index.py:264` | `def _prune_model(self, model, sparsity)` |
| `analyze` | method | `purity_index.py:399` | `def analyze(self)` |
| `analyze_polycrystal` | method | `purity_index.py:65` | `def analyze_polycrystal(self, model, pruning_level)` |
| `analyze_polycrystal` | method | `purity_index.py:243` | `def analyze_polycrystal(self, model, pruning_level, loss_history)` |
| `calculate` | method | `purity_index.py:50` | `def calculate(self, model)` |
| `calculate` | method | `purity_index.py:55` | `def calculate(self, loss_history)` |
| `calculate` | method | `purity_index.py:105` | `def calculate(self, model)` |
| `calculate` | method | `purity_index.py:160` | `def calculate(self, loss_history)` |
| `classify` | method | `purity_index.py:60` | `def classify(self, alpha, temperature)` |
| `classify` | method | `purity_index.py:203` | `def classify(self, alpha, temperature)` |
| `classify_polycrystal_state` | method | `purity_index.py:219` | `def classify_polycrystal_state(self, original_alpha, original_temp, poly_alpha, poly_temp)` |
| `compare` | method | `purity_index.py:70` | `def compare(self, original, polycrystal)` |
| `compare` | method | `purity_index.py:279` | `def compare(self, original, polycrystal)` |
| `forward` | method | `purity_index.py:90` | `def forward(self, a, b)` |
| `generate_summary` | method | `purity_index.py:537` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `purity_index.py:45` | `def get_coefficients(self)` |
| `get_coefficients` | method | `purity_index.py:93` | `def get_coefficients(self)` |
| `main` | method | `purity_index.py:611` | `def main()` |
| `migrate` | method | `purity_index.py:312` | `def migrate(self, raw_data, device)` |
| `process_checkpoint` | method | `purity_index.py:496` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `purity_index.py:510` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `BilinearModel` | class | `repor_experiments.py:141` | `class BilinearModel(Module)` |
| `Exp1Config` | class | `repor_experiments.py:96` | `class Exp1Config` |
| `Exp2Config` | class | `repor_experiments.py:103` | `class Exp2Config` |
| `Exp3Config` | class | `repor_experiments.py:108` | `class Exp3Config` |
| `Exp4Config` | class | `repor_experiments.py:114` | `class Exp4Config` |
| `Exp5Config` | class | `repor_experiments.py:120` | `class Exp5Config` |
| `ModelConfig` | class | `repor_experiments.py:78` | `class ModelConfig` |
| `SuiteConfig` | class | `repor_experiments.py:126` | `class SuiteConfig` |
| `TrainConfig` | class | `repor_experiments.py:86` | `class TrainConfig` |
| `__init__` | method | `repor_experiments.py:142` | `def __init__(self, cfg)` |
| `_boundary_prune` | method | `repor_experiments.py:655` | `def _boundary_prune(model, fraction)` |
| `_extract_state` | method | `repor_experiments.py:329` | `def _extract_state(d)` |
| `_random_prune` | method | `repor_experiments.py:644` | `def _random_prune(model, fraction)` |
| `_recursive_strassen` | method | `repor_experiments.py:222` | `def _recursive_strassen(U, V, W, n, trials)` |
| `_save` | method | `repor_experiments.py:783` | `def _save(data, path)` |
| `_strassen_rec` | method | `repor_experiments.py:234` | `def _strassen_rec(A, B, U, V, W, n)` |
| `_test_accuracy` | method | `repor_experiments.py:633` | `def _test_accuracy(model, n, device)` |
| `_verify_2x2` | method | `repor_experiments.py:204` | `def _verify_2x2(U, V, W, n)` |
| `analyze_checkpoint` | method | `repor_experiments.py:378` | `def analyze_checkpoint(path, device)` |
| `analyze_checkpoints` | method | `repor_experiments.py:665` | `def analyze_checkpoints(ckpt_dir, device)` |
| `cb` | method | `repor_experiments.py:434` | `def cb(ep, m, loss, acc)` |
| `classify_phase` | method | `repor_experiments.py:298` | `def classify_phase(delta)` |
| `compute_alpha` | method | `repor_experiments.py:279` | `def compute_alpha(delta)` |
| `compute_delta` | method | `repor_experiments.py:176` | `def compute_delta(U, V, W)` |
| `compute_kappa` | method | `repor_experiments.py:262` | `def compute_kappa(model, num_batches, bs)` |
| `compute_teff` | method | `repor_experiments.py:284` | `def compute_teff(model, num_batches, bs)` |
| `discretize_q` | method | `repor_experiments.py:173` | `def discretize_q(w)` |
| `experiment1` | method | `repor_experiments.py:428` | `def experiment1(cfg)` |
| `experiment2` | method | `repor_experiments.py:475` | `def experiment2(cfg)` |
| `experiment3` | method | `repor_experiments.py:507` | `def experiment3(cfg)` |
| `experiment4` | method | `repor_experiments.py:563` | `def experiment4(cfg)` |
| `experiment5` | method | `repor_experiments.py:607` | `def experiment5(cfg)` |
| `forward` | method | `repor_experiments.py:149` | `def forward(self, A, B)` |
| `get_flat` | method | `repor_experiments.py:167` | `def get_flat(self)` |
| `get_weights` | method | `repor_experiments.py:163` | `def get_weights(self)` |
| `load_checkpoint` | method | `repor_experiments.py:309` | `def load_checkpoint(path, device)` |
| `main` | method | `repor_experiments.py:705` | `def main()` |
| `phase2` | method | `repor_experiments.py:181` | `def phase2(model)` |
| `slot_importance` | method | `repor_experiments.py:156` | `def slot_importance(self)` |
| `train_model` | method | `repor_experiments.py:351` | `def train_model(cfg, model, epochs, bs, wd, lr, callback)` |
| `zero_shot_verify` | method | `repor_experiments.py:214` | `def zero_shot_verify(U, V, W, sizes)` |
| `BilinearModel` | class | `scrodingger.py:93` | `class BilinearModel(Module)` |
| `CheckpointLoader` | class | `scrodingger.py:263` | `class CheckpointLoader` |
| `CheckpointMigrator` | class | `scrodingger.py:271` | `class CheckpointMigrator` |
| `EigenvalueSolver` | class | `scrodingger.py:170` | `class EigenvalueSolver` |
| `ExpectationValueCalculator` | class | `scrodingger.py:216` | `class ExpectationValueCalculator` |
| `HamiltonianConstructor` | class | `scrodingger.py:150` | `class HamiltonianConstructor` |
| `ICheckpointLoader` | class | `scrodingger.py:84` | `class ICheckpointLoader(Protocol)` |
| `ICheckpointMigrator` | class | `scrodingger.py:89` | `class ICheckpointMigrator(Protocol)` |
| `IEigenvalueSolver` | class | `scrodingger.py:64` | `class IEigenvalueSolver(Protocol)` |
| `IExpectationValueCalculator` | class | `scrodingger.py:74` | `class IExpectationValueCalculator(Protocol)` |
| `IHamiltonianConstructor` | class | `scrodingger.py:59` | `class IHamiltonianConstructor(Protocol)` |
| `IModel` | class | `scrodingger.py:44` | `class IModel(Protocol)` |
| `IPotentialCalculator` | class | `scrodingger.py:54` | `class IPotentialCalculator(Protocol)` |
| `ITimeEvolver` | class | `scrodingger.py:69` | `class ITimeEvolver(Protocol)` |
| `IUncertaintyCalculator` | class | `scrodingger.py:79` | `class IUncertaintyCalculator(Protocol)` |
| `IWaveFunctionExtractor` | class | `scrodingger.py:49` | `class IWaveFunctionExtractor(Protocol)` |
| `PotentialCalculator` | class | `scrodingger.py:128` | `class PotentialCalculator` |
| `SchrodingerAnalyzer` | class | `scrodingger.py:320` | `class SchrodingerAnalyzer` |
| `SchrodingerConfig` | class | `scrodingger.py:19` | `class SchrodingerConfig` |
| `SchrodingerPipeline` | class | `scrodingger.py:609` | `class SchrodingerPipeline` |
| `TimeEvolver` | class | `scrodingger.py:194` | `class TimeEvolver` |
| `UncertaintyCalculator` | class | `scrodingger.py:226` | `class UncertaintyCalculator` |
| `WaveFunctionExtractor` | class | `scrodingger.py:121` | `class WaveFunctionExtractor` |
| `WaveFunctionVisualizer` | class | `scrodingger.py:542` | `class WaveFunctionVisualizer` |
| `__init__` | method | `scrodingger.py:94` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `scrodingger.py:129` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:151` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:171` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:195` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:227` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:321` | `def __init__(self, checkpoint_path, config)` |
| `__init__` | method | `scrodingger.py:543` | `def __init__(self, config)` |
| `__init__` | method | `scrodingger.py:610` | `def __init__(self, config)` |
| `_calculate_tunneling_probability` | method | `scrodingger.py:451` | `def _calculate_tunneling_probability(self, potential, wave_function)` |
| `_count_degeneracy` | method | `scrodingger.py:465` | `def _count_degeneracy(self, eigenvalues)` |
| `_generate_text_report` | method | `scrodingger.py:718` | `def _generate_text_report(self, summary, output_dir)` |
| `_initialize` | method | `scrodingger.py:105` | `def _initialize(self)` |
| `_load_checkpoint` | method | `scrodingger.py:335` | `def _load_checkpoint(self)` |
| `_migrate_coefs_format` | method | `scrodingger.py:309` | `def _migrate_coefs_format(self, state_dict)` |
| `_migrate_custom_format` | method | `scrodingger.py:293` | `def _migrate_custom_format(self, state_dict)` |
| `_migrate_dict` | method | `scrodingger.py:284` | `def _migrate_dict(self, state_dict)` |
| `_migrate_standard_format` | method | `scrodingger.py:316` | `def _migrate_standard_format(self, state_dict)` |
| `_print_report` | method | `scrodingger.py:478` | `def _print_report(self, results)` |
| `analyze` | method | `scrodingger.py:357` | `def analyze(self)` |
| `calculate` | method | `scrodingger.py:55` | `def calculate(self, weights)` |
| `calculate` | method | `scrodingger.py:75` | `def calculate(self, wave_function, operator)` |
| `calculate` | method | `scrodingger.py:80` | `def calculate(self, wave_function, position_grid)` |
| `calculate` | method | `scrodingger.py:132` | `def calculate(self, weights)` |
| `calculate` | method | `scrodingger.py:217` | `def calculate(self, wave_function, operator)` |
| `calculate` | method | `scrodingger.py:230` | `def calculate(self, wave_function, position_grid)` |
| `construct` | method | `scrodingger.py:60` | `def construct(self, potential, mass)` |
| `construct` | method | `scrodingger.py:154` | `def construct(self, potential, mass)` |
| `evolve` | method | `scrodingger.py:70` | `def evolve(self, initial_state, hamiltonian, time_steps, dt)` |
| `evolve` | method | `scrodingger.py:198` | `def evolve(self, initial_state, hamiltonian, time_steps, dt)` |
| `extract` | method | `scrodingger.py:50` | `def extract(self, model)` |
| `extract` | method | `scrodingger.py:122` | `def extract(self, model)` |
| `forward` | method | `scrodingger.py:110` | `def forward(self, a, b)` |
| `generate_summary` | method | `scrodingger.py:655` | `def generate_summary(self, all_results, output_dir)` |
| `get_coefficients` | method | `scrodingger.py:45` | `def get_coefficients(self)` |
| `get_coefficients` | method | `scrodingger.py:113` | `def get_coefficients(self)` |
| `load` | method | `scrodingger.py:85` | `def load(self, path, device)` |
| `load` | method | `scrodingger.py:264` | `def load(self, path, device)` |
| `main` | method | `scrodingger.py:779` | `def main()` |
| `migrate` | method | `scrodingger.py:90` | `def migrate(self, raw_data)` |
| `migrate` | method | `scrodingger.py:272` | `def migrate(self, raw_data)` |
| `process_checkpoint` | method | `scrodingger.py:614` | `def process_checkpoint(self, checkpoint_path, output_dir)` |
| `process_directory` | method | `scrodingger.py:628` | `def process_directory(self, checkpoint_dir, n_latest, output_dir)` |
| `solve` | method | `scrodingger.py:65` | `def solve(self, hamiltonian, count)` |
| `solve` | method | `scrodingger.py:174` | `def solve(self, hamiltonian, count)` |
| `visualize` | method | `scrodingger.py:546` | `def visualize(self, data, output_path)` |
| `benchmark` | function | `src/benchmarks/benchmark_final.py:64` | `def benchmark(func, A, B, warmup, runs)` |
| `main` | function | `src/benchmarks/benchmark_final.py:80` | `def main()` |
| `numpy_multiply` | function | `src/benchmarks/benchmark_final.py:60` | `def numpy_multiply(A, B)` |
| `strassen_hybrid_multiply` | function | `src/benchmarks/benchmark_final.py:45` | `def strassen_hybrid_multiply(A, B)` |
| `benchmark_function` | function | `src/benchmarks/benchmark_scientific.py:67` | `def benchmark_function(func, A, B, runs, warmup)` |
| `main` | function | `src/benchmarks/benchmark_scientific.py:91` | `def main()` |
| `numpy_multiply` | function | `src/benchmarks/benchmark_scientific.py:64` | `def numpy_multiply(A, B)` |
| `standard_avx512_multiply` | function | `src/benchmarks/benchmark_scientific.py:51` | `def standard_avx512_multiply(A, B)` |
| `strassen_multiply` | function | `src/benchmarks/benchmark_scientific.py:38` | `def strassen_multiply(A, B)` |
| `BenchmarkConfig` | class | `src/benchmarks/benchmark_strassen.py:45` | `class BenchmarkConfig` |
| `BenchmarkResult` | class | `src/benchmarks/benchmark_strassen.py:29` | `class BenchmarkResult` |
| `benchmark_resolution` | method | `src/benchmarks/benchmark_strassen.py:114` | `def benchmark_resolution(n, cfg, dtype)` |
| `estimate_memory_mb` | method | `src/benchmarks/benchmark_strassen.py:104` | `def estimate_memory_mb(n, dtype, batch_size)` |
| `get_dtype` | method | `src/benchmarks/benchmark_strassen.py:93` | `def get_dtype(dtype_str)` |
| `load_config` | method | `src/benchmarks/benchmark_strassen.py:61` | `def load_config(config_path)` |
| `main` | method | `src/benchmarks/benchmark_strassen.py:322` | `def main()` |
| `run_benchmark` | method | `src/benchmarks/benchmark_strassen.py:218` | `def run_benchmark(cfg)` |
| `save_results` | method | `src/benchmarks/benchmark_strassen.py:310` | `def save_results(results, filepath)` |
| `_load_weights` | function | `src/benchmarks/strassen_numpy.py:19` | `def _load_weights()` |
| `multiplication_count` | function | `src/benchmarks/strassen_numpy.py:112` | `def multiplication_count(n)` |
| `strassen_2x2_numpy` | function | `src/benchmarks/strassen_numpy.py:29` | `def strassen_2x2_numpy(A, B)` |
| `strassen_hybrid` | function | `src/benchmarks/strassen_numpy.py:78` | `def strassen_hybrid(A, B, threshold)` |
| `strassen_numpy` | function | `src/benchmarks/strassen_numpy.py:44` | `def strassen_numpy(A, B)` |
| `AutoTDiscovery` | class | `src/discovery/auto_T_discovery.py:32` | `class AutoTDiscovery` |
| `SymmetryStructure` | class | `src/discovery/auto_T_discovery.py:22` | `class SymmetryStructure` |
| `__init__` | method | `src/discovery/auto_T_discovery.py:42` | `def __init__(self, tolerance, verbose)` |
| `_block_repetition_score` | method | `src/discovery/auto_T_discovery.py:127` | `def _block_repetition_score(self, W, bm, bn)` |
| `_detect_block_structure` | method | `src/discovery/auto_T_discovery.py:105` | `def _detect_block_structure(self, W)` |
| `_detect_discrete_values` | method | `src/discovery/auto_T_discovery.py:89` | `def _detect_discrete_values(self, W_flat)` |
| `_detect_symmetry_type` | method | `src/discovery/auto_T_discovery.py:144` | `def _detect_symmetry_type(self, W)` |
| `_discretization_error` | method | `src/discovery/auto_T_discovery.py:198` | `def _discretization_error(self, W, values)` |
| `_invariant_subspace_dim` | method | `src/discovery/auto_T_discovery.py:184` | `def _invariant_subspace_dim(self, U, S, rank)` |
| `_is_cyclic` | method | `src/discovery/auto_T_discovery.py:171` | `def _is_cyclic(self, W)` |
| `_is_permutation_symmetric` | method | `src/discovery/auto_T_discovery.py:165` | `def _is_permutation_symmetric(self, W)` |
| `_print_analysis` | method | `src/discovery/auto_T_discovery.py:213` | `def _print_analysis(self, W, S, structure)` |
| `_validate_expansion` | method | `src/discovery/auto_T_discovery.py:290` | `def _validate_expansion(self, expanded, structure)` |
| `analyze_structure` | method | `src/discovery/auto_T_discovery.py:46` | `def analyze_structure(self, W)` |
| `construct_T` | method | `src/discovery/auto_T_discovery.py:229` | `def construct_T(self, W_dict, target_size)` |
| `recursive_strassen_multiply` | method | `src/discovery/auto_T_discovery.py:389` | `def recursive_strassen_multiply(A, B, U, V, W, base_size)` |
| `verify_expanded_correctness` | method | `src/discovery/auto_T_discovery.py:353` | `def verify_expanded_correctness(U, V, W, target_size, expanded)` |
| `verify_strassen_T` | method | `src/discovery/auto_T_discovery.py:306` | `def verify_strassen_T(model_path, target_sizes)` |
| `THRESHOLD` | macro | `src/native/strassen_c.c:11` | `#define THRESHOLD` |
| `alloc_matrix` | function | `src/native/strassen_c.c:15` | `static float* alloc_matrix(int n)` |
| `extract_quadrant` | function | `src/native/strassen_c.c:49` | `static void extract_quadrant(float* Q, float* M, int n, int row, int col)` |
| `free` | function | `src/native/strassen_c.c:156` | `free(A11);` |
| `insert_quadrant` | function | `src/native/strassen_c.c:57` | `static void insert_quadrant(float* M, float* Q, int n, int row, int col)` |
| `mat_add` | function | `src/native/strassen_c.c:33` | `static void mat_add(float* C, float* A, float* B, int n)` |
| `mat_sub` | function | `src/native/strassen_c.c:41` | `static void mat_sub(float* C, float* A, float* B, int n)` |
| `matmul_standard` | function | `src/native/strassen_c.c:20` | `static void matmul_standard(float* C, float* A, float* B, int n)` |
| `memcpy` | function | `src/native/strassen_c.c:52` | `memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float));` |
| `memset` | function | `src/native/strassen_c.c:21` | `memset(C, 0, n * n * sizeof(float));` |
| `standard_multiply` | function | `src/native/strassen_c.c:169` | `void standard_multiply(float* C, float* A, float* B, int n)` |
| `strassen_multiply` | function | `src/native/strassen_c.c:164` | `void strassen_multiply(float* C, float* A, float* B, int n)` |
| `strassen_recursive` | function | `src/native/strassen_c.c:65` | `void strassen_recursive(float* C, float* A, float* B, int n)` |
| `STRASSEN_THRESHOLD` | macro | `src/native/strassen_optimal.c:15` | `#define STRASSEN_THRESHOLD` |
| `cblas_sgemm` | function | `src/native/strassen_optimal.c:21` | `cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n, n, n, 1.0f, A, n, B, n, 0.0f, C, n);` |
| `free` | function | `src/native/strassen_optimal.c:144` | `free(workspace);` |
| `memcpy` | function | `src/native/strassen_optimal.c:64` | `memcpy(qA11 + i*h, A + i*n, h*sizeof(float));` |
| `strassen_level` | function | `src/native/strassen_optimal.c:18` | `static void strassen_level(float* C, float* A, float* B, int n, 
                           float...` |
| `strassen_optimal` | function | `src/native/strassen_optimal.c:130` | `void strassen_optimal(float* C, float* A, float* B, int n)` |
| `ALIGN` | macro | `src/native/strassen_turbo.c:22` | `#define ALIGN` |
| `BLOCK_SIZE` | macro | `src/native/strassen_turbo.c:21` | `#define BLOCK_SIZE` |
| `THRESHOLD` | macro | `src/native/strassen_turbo.c:19` | `#define THRESHOLD` |
| `_mm256_store_ps` | function | `src/native/strassen_turbo.c:40` | `_mm256_store_ps(&C[i], vc);` |
| `_mm256_storeu_ps` | function | `src/native/strassen_turbo.c:90` | `_mm256_storeu_ps(&C[i * n + j], vc);` |
| `alloc_matrix` | function | `src/native/strassen_turbo.c:25` | `static inline float* alloc_matrix(int n)` |
| `extract_quadrant` | function | `src/native/strassen_turbo.c:104` | `static void extract_quadrant(float* __restrict Q, const float* __restrict M, 
                   ...` |
| `free` | function | `src/native/strassen_turbo.c:203` | `free(T1_1);` |
| `get_num_threads` | function | `src/native/strassen_turbo.c:267` | `int get_num_threads(void)` |
| `insert_quadrant` | function | `src/native/strassen_turbo.c:114` | `static void insert_quadrant(float* __restrict M, const float* __restrict Q, 
                    ...` |
| `mat_add_avx` | function | `src/native/strassen_turbo.c:30` | `static void mat_add_avx(float* __restrict C, const float* __restrict A, 
                        ...` |
| `mat_sub_avx` | function | `src/native/strassen_turbo.c:50` | `static void mat_sub_avx(float* __restrict C, const float* __restrict A, 
                        ...` |
| `matmul_blocked_avx` | function | `src/native/strassen_turbo.c:68` | `static void matmul_blocked_avx(float* __restrict C, const float* __restrict A, 
                 ...` |
| `memcpy` | function | `src/native/strassen_turbo.c:109` | `memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float));` |
| `memset` | function | `src/native/strassen_turbo.c:70` | `memset(C, 0, n * n * sizeof(float));` |
| `omp_get_max_threads` | function | `src/native/strassen_turbo.c:268` | `return omp_get_max_threads();` |
| `omp_set_num_threads` | function | `src/native/strassen_turbo.c:262` | `omp_set_num_threads(omp_get_max_threads());` |
| `strassen_turbo` | function | `src/native/strassen_turbo.c:261` | `void strassen_turbo(float* C, float* A, float* B, int n)` |
| `strassen_turbo_recursive` | function | `src/native/strassen_turbo.c:124` | `void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)` |
| `ConvergenceMetrics` | class | `src/training/convergence_theory.py:21` | `class ConvergenceMetrics` |
| `HardwareNoiseEstimator` | class | `src/training/convergence_theory.py:136` | `class HardwareNoiseEstimator` |
| `HutchinsonTraceEstimator` | class | `src/training/convergence_theory.py:31` | `class HutchinsonTraceEstimator` |
| `SimpleStrassenModel` | class | `src/training/convergence_theory.py:346` | `class SimpleStrassenModel(Module)` |
| `__init__` | method | `src/training/convergence_theory.py:40` | `def __init__(self, model, loss_fn, n_samples, device)` |
| `__init__` | method | `src/training/convergence_theory.py:144` | `def __init__(self, model, loss_fn)` |
| `__init__` | method | `src/training/convergence_theory.py:348` | `def __init__(self, rank)` |
| `_hessian_vector_product` | method | `src/training/convergence_theory.py:90` | `def _hessian_vector_product(self, x, y, v)` |
| `_rademacher_vector` | method | `src/training/convergence_theory.py:73` | `def _rademacher_vector(self)` |
| `compute_kappa_eff` | method | `src/training/convergence_theory.py:118` | `def compute_kappa_eff(self, data)` |
| `convergence_theorem` | method | `src/training/convergence_theory.py:193` | `def convergence_theorem()` |
| `estimate_noise` | method | `src/training/convergence_theory.py:148` | `def estimate_noise(self, data_loader, n_batches, n_threads)` |
| `estimate_trace` | method | `src/training/convergence_theory.py:47` | `def estimate_trace(self, data)` |
| `forward` | method | `src/training/convergence_theory.py:354` | `def forward(self, x)` |
| `verify_convergence_conditions` | method | `src/training/convergence_theory.py:265` | `def verify_convergence_conditions(model, loss_fn, train_data, noise_threshold)` |
| `detect_phase_transition` | function | `src/training/grokkit_physics.py:126` | `def detect_phase_transition(results)` |
| `main` | function | `src/training/grokkit_physics.py:148` | `def main()` |
| `measure_physics` | function | `src/training/grokkit_physics.py:64` | `def measure_physics(N, num_samples)` |
| `strassen_multiply` | function | `src/training/grokkit_physics.py:49` | `def strassen_multiply(A, B)` |
| `Config` | class | `src/training/main.py:44` | `class Config` |
| `Matrix4x4Dataset` | class | `src/training/main.py:199` | `class Matrix4x4Dataset(Dataset)` |
| `StrassenDiscovery` | class | `src/training/main.py:87` | `class StrassenDiscovery(Module)` |
| `Trainer` | class | `src/training/main.py:217` | `class Trainer` |
| `__getitem__` | method | `src/training/main.py:213` | `def __getitem__(self, idx)` |
| `__init__` | method | `src/training/main.py:96` | `def __init__(self, num_slots)` |
| `__init__` | method | `src/training/main.py:202` | `def __init__(self, num_samples, seed)` |
| `__init__` | method | `src/training/main.py:220` | `def __init__(self, config)` |
| `__len__` | method | `src/training/main.py:210` | `def __len__(self)` |
| `accuracy` | method | `src/training/main.py:242` | `def accuracy(self, pred, target)` |
| `evaluate` | method | `src/training/main.py:265` | `def evaluate(self)` |
| `forward` | method | `src/training/main.py:108` | `def forward(self, A, B)` |
| `get_active_slots` | method | `src/training/main.py:160` | `def get_active_slots(self)` |
| `get_slot_norms` | method | `src/training/main.py:155` | `def get_slot_norms(self)` |
| `get_weakest_slot` | method | `src/training/main.py:169` | `def get_weakest_slot(self)` |
| `main` | method | `src/training/main.py:345` | `def main()` |
| `mask_slot` | method | `src/training/main.py:164` | `def mask_slot(self, slot_idx)` |
| `print_coefficients` | method | `src/training/main.py:175` | `def print_coefficients(self)` |
| `set_seed` | method | `src/training/main.py:79` | `def set_seed(seed)` |
| `train` | method | `src/training/main.py:280` | `def train(self)` |
| `train_epoch` | method | `src/training/main.py:245` | `def train_epoch(self, optimizer)` |
| `StrassenModel` | class | `src/training/main_pure_math.py:22` | `class StrassenModel(Module)` |
| `__init__` | method | `src/training/main_pure_math.py:28` | `def __init__(self, rank)` |
| `active_count` | method | `src/training/main_pure_math.py:54` | `def active_count(self, thresh)` |
| `forward` | method | `src/training/main_pure_math.py:35` | `def forward(self, A, B)` |
| `gen_data` | method | `src/training/main_pure_math.py:58` | `def gen_data(n, scale)` |
| `hard_prune` | method | `src/training/main_pure_math.py:106` | `def hard_prune(model, keep)` |
| `main` | method | `src/training/main_pure_math.py:186` | `def main()` |
| `refine_pruned` | method | `src/training/main_pure_math.py:122` | `def refine_pruned(model, active, epochs, lr)` |
| `show_coeffs` | method | `src/training/main_pure_math.py:159` | `def show_coeffs(model, active)` |
| `slot_norms` | method | `src/training/main_pure_math.py:47` | `def slot_norms(self)` |
| `train` | method | `src/training/main_pure_math.py:64` | `def train(model, epochs, lr, l1, batch, verbose)` |
| `verify` | method | `src/training/main_pure_math.py:88` | `def verify(model, n)` |
| `_load_weights` | function | `src/training/strassen_core.py:14` | `def _load_weights()` |
| `get_coefficients` | function | `src/training/strassen_core.py:77` | `def get_coefficients()` |
| `multiplication_count` | function | `src/training/strassen_core.py:82` | `def multiplication_count(n)` |
| `strassen` | function | `src/training/strassen_core.py:44` | `def strassen(X, Y)` |
| `strassen_2x2` | function | `src/training/strassen_core.py:21` | `def strassen_2x2(A, B)` |
| `StrassenOperator` | class | `src/training/strassen_grokkit.py:27` | `class StrassenOperator(Module)` |
| `__init__` | method | `src/training/strassen_grokkit.py:40` | `def __init__(self, rank)` |
| `compute_LC` | method | `src/training/strassen_grokkit.py:67` | `def compute_LC(self)` |
| `compute_SP` | method | `src/training/strassen_grokkit.py:83` | `def compute_SP(self)` |
| `count_active` | method | `src/training/strassen_grokkit.py:108` | `def count_active(self, threshold)` |
| `forward` | method | `src/training/strassen_grokkit.py:49` | `def forward(self, A, B)` |
| `generate_batch` | method | `src/training/strassen_grokkit.py:113` | `def generate_batch(n, scale)` |
| `main` | method | `src/training/strassen_grokkit.py:351` | `def main()` |
| `progressive_sparsification` | method | `src/training/strassen_grokkit.py:254` | `def progressive_sparsification(model, target_slots)` |
| `slot_importance` | method | `src/training/strassen_grokkit.py:101` | `def slot_importance(self)` |
| `train_grokkit` | method | `src/training/strassen_grokkit.py:120` | `def train_grokkit(epochs, batch_size, lr, wd)` |
| `verify_grokking` | method | `src/training/strassen_grokkit.py:203` | `def verify_grokking(model, n_test)` |
| `StrassenOperator` | class | `src/training/train_strassen.py:26` | `class StrassenOperator(Module)` |
| `__init__` | method | `src/training/train_strassen.py:33` | `def __init__(self, rank)` |
| `count_active` | method | `src/training/train_strassen.py:56` | `def count_active(self, threshold)` |
| `discretize` | method | `src/training/train_strassen.py:183` | `def discretize(model, slots_to_prune)` |
| `forward` | method | `src/training/train_strassen.py:40` | `def forward(self, A, B)` |
| `generate_batch` | method | `src/training/train_strassen.py:60` | `def generate_batch(n, scale)` |
| `get_canonical_strassen` | method | `src/training/train_strassen.py:211` | `def get_canonical_strassen()` |
| `main` | method | `src/training/train_strassen.py:299` | `def main()` |
| `slot_importance` | method | `src/training/train_strassen.py:50` | `def slot_importance(self)` |
| `sparsify` | method | `src/training/train_strassen.py:104` | `def sparsify(model, target_slots)` |
| `train_phase1` | method | `src/training/train_strassen.py:66` | `def train_phase1(epochs, batch_size, lr, wd)` |
| `verify` | method | `src/training/train_strassen.py:260` | `def verify(U, V, W, n_test)` |
| `BilinearStrassenModel` | class | `superposition.py:258` | `class BilinearStrassenModel(Module)` |
| `CheckpointLoader` | class | `superposition.py:75` | `class CheckpointLoader` |
| `CheckpointLoadingError` | class | `superposition.py:71` | `class CheckpointLoadingError(Exception)` |
| `CheckpointMigrator` | class | `superposition.py:85` | `class CheckpointMigrator` |
| `Config` | class | `superposition.py:23` | `class Config` |
| `IAnalyzer` | class | `superposition.py:66` | `class IAnalyzer(ABC)` |
| `ICheckpointLoader` | class | `superposition.py:59` | `class ICheckpointLoader(Protocol)` |
| `IMetricsCalculator` | class | `superposition.py:62` | `class IMetricsCalculator(ABC)` |
| `SAETrainer` | class | `superposition.py:407` | `class SAETrainer` |
| `SparseAutoencoder` | class | `superposition.py:292` | `class SparseAutoencoder(Module)` |
| `StrassenCheckpointAnalyzer` | class | `superposition.py:475` | `class StrassenCheckpointAnalyzer(IAnalyzer)` |
| `StrassenDataGenerator` | class | `superposition.py:220` | `class StrassenDataGenerator` |
| `SuperpositionMetrics` | class | `superposition.py:334` | `class SuperpositionMetrics(IMetricsCalculator)` |
| `__init__` | method | `superposition.py:223` | `def __init__(self, config)` |
| `__init__` | method | `superposition.py:261` | `def __init__(self, config)` |
| `__init__` | method | `superposition.py:298` | `def __init__(self, config)` |
| `__init__` | method | `superposition.py:339` | `def __init__(self, config)` |
| `__init__` | method | `superposition.py:410` | `def __init__(self, sae, config)` |
| `__init__` | method | `superposition.py:480` | `def __init__(self, config)` |
| `__post_init__` | method | `superposition.py:54` | `def __post_init__(self)` |
| `_generate_comparison_plots` | method | `superposition.py:694` | `def _generate_comparison_plots(self, results)` |
| `_initialize_symmetric` | method | `superposition.py:271` | `def _initialize_symmetric(self)` |
| `_migrate_coefs_format` | method | `superposition.py:162` | `def _migrate_coefs_format(state_dict)` |
| `_migrate_custom_format` | method | `superposition.py:147` | `def _migrate_custom_format(state_dict)` |
| `_migrate_dict` | method | `superposition.py:135` | `def _migrate_dict(state_dict)` |
| `_migrate_encoder_format` | method | `superposition.py:170` | `def _migrate_encoder_format(state_dict)` |
| `_migrate_standard_format` | method | `superposition.py:215` | `def _migrate_standard_format(state_dict)` |
| `_save_final_results` | method | `superposition.py:687` | `def _save_final_results(self, results)` |
| `_save_intermediate_result` | method | `superposition.py:633` | `def _save_intermediate_result(self, result, name)` |
| `_save_progress_checkpoint` | method | `superposition.py:680` | `def _save_progress_checkpoint(self, results)` |
| `analyze_checkpoint` | method | `superposition.py:68` | `def analyze_checkpoint(self, checkpoint_path)` |
| `analyze_checkpoint` | method | `superposition.py:571` | `def analyze_checkpoint(self, checkpoint_path)` |
| `analyze_directory` | method | `superposition.py:641` | `def analyze_directory(self, checkpoint_dir)` |
| `compute` | method | `superposition.py:64` | `def compute(self)` |
| `compute` | method | `superposition.py:389` | `def compute(self, sae_activations, weight_matrix)` |
| `compute_entropy` | method | `superposition.py:352` | `def compute_entropy(self, probabilities)` |
| `compute_feature_probabilities` | method | `superposition.py:342` | `def compute_feature_probabilities(self, sae_activations)` |
| `compute_frobenius_metric` | method | `superposition.py:377` | `def compute_frobenius_metric(self, weight_matrix)` |
| `compute_interference_matrix` | method | `superposition.py:385` | `def compute_interference_matrix(self, weight_matrix)` |
| `compute_superposition` | method | `superposition.py:359` | `def compute_superposition(self, sae_activations)` |
| `decode` | method | `superposition.py:319` | `def decode(self, z)` |
| `detect_hidden_dim` | method | `superposition.py:89` | `def detect_hidden_dim(raw_data)` |
| `encode` | method | `superposition.py:310` | `def encode(self, x)` |
| `extract_bottleneck_activations` | method | `superposition.py:554` | `def extract_bottleneck_activations(self, model)` |
| `forward` | method | `superposition.py:276` | `def forward(self, a, b)` |
| `forward` | method | `superposition.py:328` | `def forward(self, x)` |
| `generate_batch` | method | `superposition.py:226` | `def generate_batch(self, batch_size)` |
| `generate_dataset` | method | `superposition.py:240` | `def generate_dataset(self, num_samples)` |
| `get_coefficients` | method | `superposition.py:284` | `def get_coefficients(self)` |
| `get_tensor` | method | `superposition.py:148` | `def get_tensor(key)` |
| `load_checkpoint` | method | `superposition.py:60` | `def load_checkpoint(self, path, device)` |
| `load_checkpoint` | method | `superposition.py:78` | `def load_checkpoint(self, path, device)` |
| `load_model` | method | `superposition.py:496` | `def load_model(self, checkpoint_path)` |
| `main` | method | `superposition.py:765` | `def main()` |
| `migrate_checkpoint` | method | `superposition.py:122` | `def migrate_checkpoint(raw_data)` |
| `train` | method | `superposition.py:421` | `def train(self, bottleneck_activations)` |
| `LocalConfig` | class | `train_batch_sweep.py:17` | `class LocalConfig` |
| `train_for_batch_size` | function | `train_batch_sweep.py:11` | `def train_for_batch_size(B, seed, output_dir)` |
| `AreaPruningStrategy` | class | `unified_hidden_connections_suite.py:988` | `class AreaPruningStrategy(IPruningStrategy)` |
| `CheckpointManager` | class | `unified_hidden_connections_suite.py:269` | `class CheckpointManager` |
| `Experiment1Config` | class | `unified_hidden_connections_suite.py:99` | `class Experiment1Config` |
| `Experiment1RicciMBLDuality` | class | `unified_hidden_connections_suite.py:657` | `class Experiment1RicciMBLDuality(IExperiment)` |
| `Experiment2AltlandZirnbauer` | class | `unified_hidden_connections_suite.py:735` | `class Experiment2AltlandZirnbauer(IExperiment)` |
| `Experiment2Config` | class | `unified_hidden_connections_suite.py:114` | `class Experiment2Config` |
| `Experiment3Config` | class | `unified_hidden_connections_suite.py:131` | `class Experiment3Config` |
| `Experiment3ConformalIsomorphism` | class | `unified_hidden_connections_suite.py:820` | `class Experiment3ConformalIsomorphism(IExperiment)` |
| `Experiment4CompressionFrontier` | class | `unified_hidden_connections_suite.py:886` | `class Experiment4CompressionFrontier(IExperiment)` |
| `Experiment4Config` | class | `unified_hidden_connections_suite.py:141` | `class Experiment4Config` |
| `Experiment5Config` | class | `unified_hidden_connections_suite.py:158` | `class Experiment5Config` |
| `Experiment5HolographicPruning` | class | `unified_hidden_connections_suite.py:1004` | `class Experiment5HolographicPruning(IExperiment)` |
| `ICheckpointManager` | class | `unified_hidden_connections_suite.py:262` | `class ICheckpointManager(Protocol)` |
| `IDataGenerator` | class | `unified_hidden_connections_suite.py:229` | `class IDataGenerator(Protocol)` |
| `IExperiment` | class | `unified_hidden_connections_suite.py:645` | `class IExperiment(ABC)` |
| `IMetricCalculator` | class | `unified_hidden_connections_suite.py:357` | `class IMetricCalculator(ABC)` |
| `IPruningStrategy` | class | `unified_hidden_connections_suite.py:965` | `class IPruningStrategy(ABC)` |
| `ITrainer` | class | `unified_hidden_connections_suite.py:293` | `class ITrainer(Protocol)` |
| `LevelSpacingRatioCalculator` | class | `unified_hidden_connections_suite.py:365` | `class LevelSpacingRatioCalculator` |
| `RicciScalarCalculator` | class | `unified_hidden_connections_suite.py:415` | `class RicciScalarCalculator` |
| `StrassStrassenConfig` | class | `unified_hidden_connections_suite.py:65` | `class StrassStrassenConfig` |
| `StrassStrassenModel` | class | `unified_hidden_connections_suite.py:183` | `class StrassStrassenModel(Module)` |
| `StrassenDataGenerator` | class | `unified_hidden_connections_suite.py:235` | `class StrassenDataGenerator` |
| `SuiteConfig` | class | `unified_hidden_connections_suite.py:169` | `class SuiteConfig` |
| `SuperpositionMetricCalculator` | class | `unified_hidden_connections_suite.py:552` | `class SuperpositionMetricCalculator` |
| `SyntheticPlanckCalculator` | class | `unified_hidden_connections_suite.py:492` | `class SyntheticPlanckCalculator` |
| `Trainer` | class | `unified_hidden_connections_suite.py:301` | `class Trainer` |
| `TrainingConfig` | class | `unified_hidden_connections_suite.py:84` | `class TrainingConfig` |
| `UnifiedSuite` | class | `unified_hidden_connections_suite.py:1078` | `class UnifiedSuite` |
| `VolumePruningStrategy` | class | `unified_hidden_connections_suite.py:973` | `class VolumePruningStrategy(IPruningStrategy)` |
| `__init__` | method | `unified_hidden_connections_suite.py:190` | `def __init__(self, config)` |
| `__init__` | method | `unified_hidden_connections_suite.py:238` | `def __init__(self, config)` |
| `__init__` | method | `unified_hidden_connections_suite.py:304` | `def __init__(self, model_config, training_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:368` | `def __init__(self, config, tolerance)` |
| `__init__` | method | `unified_hidden_connections_suite.py:418` | `def __init__(self, config, regularization)` |
| `__init__` | method | `unified_hidden_connections_suite.py:495` | `def __init__(self, config, noise_floor)` |
| `__init__` | method | `unified_hidden_connections_suite.py:555` | `def __init__(self, model_config, expansion_factor, l1_coefficient, sae_lr, sae_epochs, sae_batch_size, num_samples, epsi` |
| `__init__` | method | `unified_hidden_connections_suite.py:663` | `def __init__(self, config, model_config, training_config, data_generator, checkpoint_manager)` |
| `__init__` | method | `unified_hidden_connections_suite.py:741` | `def __init__(self, config, model_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:826` | `def __init__(self, config, model_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:893` | `def __init__(self, config, model_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:1010` | `def __init__(self, config, model_config, data_generator)` |
| `__init__` | method | `unified_hidden_connections_suite.py:1081` | `def __init__(self, config)` |
| `__post_init__` | method | `unified_hidden_connections_suite.py:76` | `def __post_init__(self)` |
| `_aggregate_verdicts` | method | `unified_hidden_connections_suite.py:1150` | `def _aggregate_verdicts(self, all_results)` |
| `_analyze_temporal_correlation` | method | `unified_hidden_connections_suite.py:717` | `def _analyze_temporal_correlation(self, results)` |
| `_apply_moebius` | method | `unified_hidden_connections_suite.py:869` | `def _apply_moebius(self, A, B)` |
| `_apply_moebius_to_output` | method | `unified_hidden_connections_suite.py:879` | `def _apply_moebius_to_output(self, C)` |
| `_build_experiments` | method | `unified_hidden_connections_suite.py:1088` | `def _build_experiments(self)` |
| `_build_hessian_approximation` | method | `unified_hidden_connections_suite.py:372` | `def _build_hessian_approximation(self, model)` |
| `_compute_hessian` | method | `unified_hidden_connections_suite.py:422` | `def _compute_hessian(self, model)` |
| `_create_gamma_model` | method | `unified_hidden_connections_suite.py:778` | `def _create_gamma_model(self, base_model, gamma)` |
| `_detect_critical_transition` | method | `unified_hidden_connections_suite.py:800` | `def _detect_critical_transition(self, results)` |
| `_diagonal_hessian_approximation` | method | `unified_hidden_connections_suite.py:456` | `def _diagonal_hessian_approximation(self, model, A, B, C_true)` |
| `_extract_activations` | method | `unified_hidden_connections_suite.py:576` | `def _extract_activations(self, model)` |
| `_generate_single_sample` | method | `unified_hidden_connections_suite.py:433` | `def _generate_single_sample(self)` |
| `_loss_from_flat` | method | `unified_hidden_connections_suite.py:437` | `def _loss_from_flat(self, flat_params, model, original_params, A, B, C_true)` |
| `_run_pruning_trials` | method | `unified_hidden_connections_suite.py:1046` | `def _run_pruning_trials(self, base_model, pruner, A, B, C_true)` |
| `_sae_forward` | method | `unified_hidden_connections_suite.py:616` | `def _sae_forward(self, x, W_enc, b_enc, b_dec)` |
| `_serialize_config` | method | `unified_hidden_connections_suite.py:1163` | `def _serialize_config(self)` |
| `_test_uncertainty_bound` | method | `unified_hidden_connections_suite.py:951` | `def _test_uncertainty_bound(self, results)` |
| `_train_gamma_model` | method | `unified_hidden_connections_suite.py:790` | `def _train_gamma_model(self, model)` |
| `_train_sae` | method | `unified_hidden_connections_suite.py:589` | `def _train_sae(self, activations)` |
| `calculate` | method | `unified_hidden_connections_suite.py:361` | `def calculate(self, model)` |
| `calculate` | method | `unified_hidden_connections_suite.py:384` | `def calculate(self, model)` |
| `calculate` | method | `unified_hidden_connections_suite.py:470` | `def calculate(self, model)` |
| `calculate` | method | `unified_hidden_connections_suite.py:499` | `def calculate(self, model)` |
| `calculate` | method | `unified_hidden_connections_suite.py:623` | `def calculate(self, model)` |
| `checkpoint_callback` | method | `unified_hidden_connections_suite.py:691` | `def checkpoint_callback(epoch, m, loss, acc)` |
| `count_active_slots` | method | `unified_hidden_connections_suite.py:225` | `def count_active_slots(self, threshold)` |
| `forward` | method | `unified_hidden_connections_suite.py:203` | `def forward(self, A, B)` |
| `generate_batch` | method | `unified_hidden_connections_suite.py:232` | `def generate_batch(self, batch_size)` |
| `generate_batch` | method | `unified_hidden_connections_suite.py:241` | `def generate_batch(self, batch_size)` |
| `get_coefficients` | method | `unified_hidden_connections_suite.py:213` | `def get_coefficients(self)` |
| `get_flat_parameters` | method | `unified_hidden_connections_suite.py:216` | `def get_flat_parameters(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:653` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:679` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:752` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:836` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:904` | `def get_name(self)` |
| `get_name` | method | `unified_hidden_connections_suite.py:1022` | `def get_name(self)` |
| `load` | method | `unified_hidden_connections_suite.py:266` | `def load(self, path, model)` |
| `load` | method | `unified_hidden_connections_suite.py:284` | `def load(self, path, model)` |
| `main` | method | `unified_hidden_connections_suite.py:1182` | `def main()` |
| `prune` | method | `unified_hidden_connections_suite.py:969` | `def prune(self, model, fraction)` |
| `prune` | method | `unified_hidden_connections_suite.py:976` | `def prune(self, model, fraction)` |
| `prune` | method | `unified_hidden_connections_suite.py:991` | `def prune(self, model, fraction)` |
| `run` | method | `unified_hidden_connections_suite.py:649` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:682` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:755` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:839` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:907` | `def run(self, model)` |
| `run` | method | `unified_hidden_connections_suite.py:1025` | `def run(self, model)` |
| `run_all` | method | `unified_hidden_connections_suite.py:1127` | `def run_all(self)` |
| `save` | method | `unified_hidden_connections_suite.py:265` | `def save(self, model, epoch, metrics, path)` |
| `save` | method | `unified_hidden_connections_suite.py:272` | `def save(self, model, epoch, metrics, path)` |
| `slot_importance` | method | `unified_hidden_connections_suite.py:219` | `def slot_importance(self)` |
| `train` | method | `unified_hidden_connections_suite.py:296` | `def train(self, model, epochs, callback)` |
| `train` | method | `unified_hidden_connections_suite.py:314` | `def train(self, model, epochs, callback)` |
| `BilinearStrassenModel` | class | `xray_tensor_diffractometer.py:187` | `class BilinearStrassenModel(Module)` |
| `BoltzmannAnalysisProgram` | class | `xray_tensor_diffractometer.py:1490` | `class BoltzmannAnalysisProgram` |
| `CheckpointLoader` | class | `xray_tensor_diffractometer.py:1371` | `class CheckpointLoader` |
| `CheckpointLoadingError` | class | `xray_tensor_diffractometer.py:155` | `class CheckpointLoadingError(Exception)` |
| `CheckpointMigrator` | class | `xray_tensor_diffractometer.py:1405` | `class CheckpointMigrator` |
| `Config` | class | `xray_tensor_diffractometer.py:27` | `class Config` |
| `CrystallographyMetrics` | class | `xray_tensor_diffractometer.py:1158` | `class CrystallographyMetrics` |
| `EpitaxialGrowthEngine` | class | `xray_tensor_diffractometer.py:216` | `class EpitaxialGrowthEngine` |
| `EpitaxyExperiment` | class | `xray_tensor_diffractometer.py:469` | `class EpitaxyExperiment` |
| `GreenCowExperiment` | class | `xray_tensor_diffractometer.py:1264` | `class GreenCowExperiment` |
| `ICheckpointLoader` | class | `xray_tensor_diffractometer.py:146` | `class ICheckpointLoader(Protocol)` |
| `IDataGenerator` | class | `xray_tensor_diffractometer.py:152` | `class IDataGenerator(Protocol)` |
| `IMetricsCalculator` | class | `xray_tensor_diffractometer.py:149` | `class IMetricsCalculator(Protocol)` |
| `MetricsComputationError` | class | `xray_tensor_diffractometer.py:158` | `class MetricsComputationError(Exception)` |
| `SpectroscopyMetrics` | class | `xray_tensor_diffractometer.py:693` | `class SpectroscopyMetrics` |
| `StrassenCrystallographer` | class | `xray_tensor_diffractometer.py:2610` | `class StrassenCrystallographer` |
| `StrassenDataGenerator` | class | `xray_tensor_diffractometer.py:166` | `class StrassenDataGenerator` |
| `ThermodynamicMetrics` | class | `xray_tensor_diffractometer.py:913` | `class ThermodynamicMetrics` |
| `ThermodynamicPotential` | class | `xray_tensor_diffractometer.py:671` | `class ThermodynamicPotential` |
| `TrainingError` | class | `xray_tensor_diffractometer.py:161` | `class TrainingError(Exception)` |
| `__init__` | method | `xray_tensor_diffractometer.py:188` | `def __init__(self, hidden_dim, matrix_size)` |
| `__init__` | method | `xray_tensor_diffractometer.py:224` | `def __init__(self, seed_checkpoint_path, target_matrix_size, device)` |
| `__init__` | method | `xray_tensor_diffractometer.py:474` | `def __init__(self, results_dir)` |
| `__init__` | method | `xray_tensor_diffractometer.py:1270` | `def __init__(self, model, device)` |
| `__init__` | method | `xray_tensor_diffractometer.py:1491` | `def __init__(self, checkpoint_dir, results_dir)` |
| `__init__` | method | `xray_tensor_diffractometer.py:2611` | `def __init__(self, checkpoint_path, device)` |
| `_adjust_dimensions` | method | `xray_tensor_diffractometer.py:326` | `def _adjust_dimensions(self, tensor, target_shape)` |
| `_assign_grade` | method | `xray_tensor_diffractometer.py:2656` | `def _assign_grade(self, delta, alpha)` |
| `_check_strassen_equivalence` | method | `xray_tensor_diffractometer.py:868` | `def _check_strassen_equivalence(discretized_factors)` |
| `_classify_thermodynamic_phase` | method | `xray_tensor_diffractometer.py:2150` | `def _classify_thermodynamic_phase(self, t_eff, cv, alpha)` |
| `_compute_effective_volume` | method | `xray_tensor_diffractometer.py:2304` | `def _compute_effective_volume(self, params)` |
| `_compute_entropy` | method | `xray_tensor_diffractometer.py:2265` | `def _compute_entropy(self, params)` |
| `_compute_entropy_simple` | method | `xray_tensor_diffractometer.py:2239` | `def _compute_entropy_simple(self, params)` |
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
