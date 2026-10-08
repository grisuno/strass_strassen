# Symbols (page 1 of 5)
Pages: [SYMBOLS.md](SYMBOLS.md), [SYMBOLS_p2.md](SYMBOLS_p2.md), [SYMBOLS_p3.md](SYMBOLS_p3.md), [SYMBOLS_p4.md](SYMBOLS_p4.md), [SYMBOLS_p5.md](SYMBOLS_p5.md)

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

Next: [SYMBOLS_p2.md](SYMBOLS_p2.md)
