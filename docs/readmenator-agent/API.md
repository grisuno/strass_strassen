# API (page 1 of 3)
Pages: [API.md](API.md), [API_p2.md](API_p2.md), [API_p3.md](API_p3.md)

## app.py
- `StrassenNet.__init__` (method) `app.py:25` `def __init__(self, rank)`
- `StrassenNet.forward` (method) `app.py:31` `def forward(self, A, B)`

## batch_size.py
- `Configuration.set_random_seed` (method) `batch_size.py:71` `def set_random_seed(seed)`
- `BilinearStrassenModel.__init__` (method) `batch_size.py:78` `def __init__(self, config)`
- `BilinearStrassenModel.forward` (method) `batch_size.py:88` `def forward(self, a, b)`
- `BilinearStrassenModel.get_coefficients` (method) `batch_size.py:91` `def get_coefficients(self)`
- `BilinearStrassenModel.compute_lambda_effective` (method) `batch_size.py:94` `def compute_lambda_effective(self)`
- `CheckpointMigrator.can_migrate` (method) `batch_size.py:102` `def can_migrate(self, state_dict)`
- `CheckpointMigrator.migrate` (method) `batch_size.py:106` `def migrate(self, state_dict)`
- `CustomFormatMigrator.can_migrate` (method) `batch_size.py:111` `def can_migrate(self, state_dict)`
- `CustomFormatMigrator.migrate` (method) `batch_size.py:114` `def migrate(self, state_dict)`
- `StandardFormatMigrator.can_migrate` (method) `batch_size.py:123` `def can_migrate(self, state_dict)`
- `StandardFormatMigrator.migrate` (method) `batch_size.py:126` `def migrate(self, state_dict)`
- `CheckpointMigrationManager.__init__` (method) `batch_size.py:132` `def __init__(self)`
- `CheckpointMigrationManager.migrate_checkpoint` (method) `batch_size.py:135` `def migrate_checkpoint(self, path, device)`
- `StrassenDataGenerator.generate_batch` (method) `batch_size.py:149` `def generate_batch(batch_size, config)`
- `CrystallographyMetrics.compute_kappa` (method) `batch_size.py:158` `def compute_kappa(model, num_batches, config)`
- `CrystallographyMetrics.compute_discretization_margin` (method) `batch_size.py:174` `def compute_discretization_margin(coeffs)`
- `CrystallographyMetrics.compute_local_complexity` (method) `batch_size.py:178` `def compute_local_complexity(model, config)`
- `PlanckConstantCalculator.__init__` (method) `batch_size.py:187` `def __init__(self, metrics, training_metrics, config)`
- `PlanckConstantCalculator.calculate_all` (method) `batch_size.py:196` `def calculate_all(self)`
- `BatchSizeThermodynamics.__init__` (method) `batch_size.py:218` `def __init__(self, model, h_bar, delta_struct, config)`
- `BatchSizeThermodynamics.analyze_batch_size_spectrum` (method) `batch_size.py:224` `def analyze_batch_size_spectrum(self)`
- `StrassenCheckpointLoader.__init__` (method) `batch_size.py:263` `def __init__(self, config)`
- `StrassenCheckpointLoader.load` (method) `batch_size.py:267` `def load(self, path, device)`
- `StrassenCheckpointLoader.extract_training_metrics` (method) `batch_size.py:281` `def extract_training_metrics(self, path)`
- `StrassenPlanckAnalyzer.__init__` (method) `batch_size.py:293` `def __init__(self, config)`
- `StrassenPlanckAnalyzer.analyze_checkpoint` (method) `batch_size.py:297` `def analyze_checkpoint(self, path, device)`
- `StrassenPlanckAnalyzer.analyze_directory` (method) `batch_size.py:323` `def analyze_directory(self, directory, device, pattern)`
- `StrassenPlanckAnalyzer.main` (method) `batch_size.py:341` `def main()`

## boltzmann_experiments.py
- `Config.set_seed` (method) `boltzmann_experiments.py:30` `def set_seed(seed)`
- `ICheckpointLoader.load_checkpoint` (method) `boltzmann_experiments.py:42` `def load_checkpoint(self, path, device)`
- `CheckpointLoader.load_checkpoint` (method) `boltzmann_experiments.py:46` `def load_checkpoint(self, path, device)`
- `CheckpointMigrator.migrate_checkpoint` (method) `boltzmann_experiments.py:54` `def migrate_checkpoint(raw_data)`
- `BilinearStrassenModel.__init__` (method) `boltzmann_experiments.py:119` `def __init__(self, n_slots)`
- `BilinearStrassenModel.forward` (method) `boltzmann_experiments.py:131` `def forward(self, a, b)`
- `BilinearStrassenModel.get_coefficients` (method) `boltzmann_experiments.py:134` `def get_coefficients(self)`
- `CrystallographyMetrics.compute_kappa` (method) `boltzmann_experiments.py:139` `def compute_kappa(coeffs)` -- Classical kappa - will be inf for discrete states
- `CrystallographyMetrics.compute_delta` (method) `boltzmann_experiments.py:156` `def compute_delta(coeffs)` -- Discretization error δ
- `CrystallographyMetrics.compute_local_complexity` (method) `boltzmann_experiments.py:161` `def compute_local_complexity(coeffs)`
- `CrystallographyMetrics.compute_alpha_purity` (method) `boltzmann_experiments.py:169` `def compute_alpha_purity(coeffs)` -- Alpha purity: α = -log(δ), inverse temperature metric for discrete states
- `CrystallographyMetrics.compute_kappa_quantum` (method) `boltzmann_experiments.py:178` `def compute_kappa_quantum(coeffs, hbar)` -- Quantum-regularized kappa for singular covariance states
- `DLProgram.__init__` (method) `boltzmann_experiments.py:200` `def __init__(self, checkpoint_dir, results_dir)`
- `DLProgram.run_full_boltzmann_program` (method) `boltzmann_experiments.py:247` `def run_full_boltzmann_program(self)`
- `DLProgram.convert_to_serializable` (method) `boltzmann_experiments.py:310` `def convert_to_serializable(obj)`
- `DLProgram.phase1_molecular_hypothesis` (method) `boltzmann_experiments.py:328` `def phase1_molecular_hypothesis(self)`
- `DLProgram.phase2_entropy_production` (method) `boltzmann_experiments.py:506` `def phase2_entropy_production(self)`
- `DLProgram.model` (method) `boltzmann_experiments.py:692` `def model(t, A, tau, C)`
- `DLProgram.phase3_extensivity_law` (method) `boltzmann_experiments.py:719` `def phase3_extensivity_law(self)`
- `DLProgram.model` (method) `boltzmann_experiments.py:813` `def model(N, alpha, beta)`
- `DLProgram.phase4_quantum_basis_transform` (method) `boltzmann_experiments.py:841` `def phase4_quantum_basis_transform(self)`
- `DLProgram.main` (method) `boltzmann_experiments.py:935` `def main()`
- `DLProgram.model` (method) `boltzmann_experiments.py:1017` `def model(t, A, tau, C)`
- `DLProgram.phase3_extensivity_law` (method) `boltzmann_experiments.py:1044` `def phase3_extensivity_law(self)`

## compute_gns_checkpoints.py
- `estimate_gns` (function) `compute_gns_checkpoints.py:11` `def estimate_gns(model, batch_size, num_batches)`
- `main` (function) `compute_gns_checkpoints.py:37` `def main()`

## crystallography.py
- `Config.set_seed` (method) `crystallography.py:35` `def set_seed(seed)`
- `BilinearStrassenModel.__init__` (method) `crystallography.py:45` `def __init__(self, n_slots)`
- `BilinearStrassenModel.forward` (method) `crystallography.py:57` `def forward(self, a, b)`
- `BilinearStrassenModel.get_coefficients` (method) `crystallography.py:60` `def get_coefficients(self)`
- `CheckpointMigrator.migrate_checkpoint` (method) `crystallography.py:73` `def migrate_checkpoint(path, device)`
- `StrassenDataGenerator.generate_batch` (method) `crystallography.py:157` `def generate_batch(batch_size)`
- `StrassenDataGenerator.verify_structure` (method) `crystallography.py:164` `def verify_structure(coeffs)`
- `SparsificationProtocol.__init__` (method) `crystallography.py:173` `def __init__(self, model)`
- `SparsificationProtocol.prune_to_target` (method) `crystallography.py:176` `def prune_to_target(self, target)`
- `SparsificationProtocol.discretize_weights` (method) `crystallography.py:192` `def discretize_weights(self, margin)`
- `CrystallographyMetrics.compute_kappa` (method) `crystallography.py:208` `def compute_kappa(model, dataloader, num_batches)`
- `CrystallographyMetrics.compute_discretization_margin` (method) `crystallography.py:225` `def compute_discretization_margin(coeffs)`
- `StrassenDiffractionTest.__init__` (method) `crystallography.py:233` `def __init__(self, model)`
- `StrassenDiffractionTest.test_gauge_invariance` (method) `crystallography.py:236` `def test_gauge_invariance(self, n_samples)`
- `BasinResilienceSpectrometer.__init__` (method) `crystallography.py:278` `def __init__(self, model)`
- `BasinResilienceSpectrometer.measure_resilience_spectrum` (method) `crystallography.py:282` `def measure_resilience_spectrum(self, noise_levels)`
- `CrystalPurityIndex.__init__` (method) `crystallography.py:346` `def __init__(self, model, diffraction_results, resilience_results, metrics_results)`
- `CrystalPurityIndex.compute` (method) `crystallography.py:359` `def compute(self)`
- `StrassenCrystallographer.__init__` (method) `crystallography.py:417` `def __init__(self, checkpoint_path, device)`
- `StrassenCrystallographer.run_full_analysis` (method) `crystallography.py:445` `def run_full_analysis(self)`
- `StrassenCrystallographer.dataloader_gen` (method) `crystallography.py:465` `def dataloader_gen()`
- `LocalComplexity.compute` (method) `crystallography.py:525` `def compute(model)` -- Computa LC basado en Can't Stop Won't Stop paper
- `LocalComplexity.main` (method) `crystallography.py:542` `def main()`

## dirac_polos_zeros.py
- `IModel.forward` (method) `dirac_polos_zeros.py:55` `def forward(self, a, b)`
- `IModel.get_coefficients` (method) `dirac_polos_zeros.py:56` `def get_coefficients(self)`
- `IChargeDistributionExtractor.extract` (method) `dirac_polos_zeros.py:61` `def extract(self, model)`
- `IDiracAnalyzer.analyze` (method) `dirac_polos_zeros.py:66` `def analyze(self, charge_density)`
- `IFieldCalculator.calculate` (method) `dirac_polos_zeros.py:71` `def calculate(self, dirac_data, eval_points)`
- `IFluxCalculator.calculate` (method) `dirac_polos_zeros.py:76` `def calculate(self, electric_field, surface_points)`
- `IStateSpaceExtractor.extract` (method) `dirac_polos_zeros.py:81` `def extract(self, model)`
- `ITransferFunctionComputer.compute` (method) `dirac_polos_zeros.py:86` `def compute(self, A, B, C, D)`
- `IPoleZeroAnalyzer.analyze_stability` (method) `dirac_polos_zeros.py:91` `def analyze_stability(self)`
- `IPoleZeroAnalyzer.get_poles` (method) `dirac_polos_zeros.py:92` `def get_poles(self)`
- `IPoleZeroAnalyzer.get_zeros` (method) `dirac_polos_zeros.py:93` `def get_zeros(self)`
- `IFrequencyAnalyzer.compute_bode` (method) `dirac_polos_zeros.py:98` `def compute_bode(self)`
- `IFrequencyAnalyzer.compute_margins` (method) `dirac_polos_zeros.py:99` `def compute_margins(self)`
- `IFrequencyAnalyzer.compute_nyquist` (method) `dirac_polos_zeros.py:100` `def compute_nyquist(self)`
- `ITimeResponseAnalyzer.compute_step` (method) `dirac_polos_zeros.py:105` `def compute_step(self)`
- `ITimeResponseAnalyzer.compute_impulse` (method) `dirac_polos_zeros.py:106` `def compute_impulse(self)`
- `ICheckpointLoader.load` (method) `dirac_polos_zeros.py:111` `def load(self, path, device)`
- `ICheckpointMigrator.migrate` (method) `dirac_polos_zeros.py:116` `def migrate(self, raw_data)`
- `IVisualizer.visualize` (method) `dirac_polos_zeros.py:121` `def visualize(self, data, output_path)`
- `BilinearModel.__init__` (method) `dirac_polos_zeros.py:125` `def __init__(self, hidden_dim, matrix_size)`
- `BilinearModel.forward` (method) `dirac_polos_zeros.py:141` `def forward(self, a, b)`
- `BilinearModel.get_coefficients` (method) `dirac_polos_zeros.py:144` `def get_coefficients(self)`
- `ChargeDistributionExtractor.extract` (method) `dirac_polos_zeros.py:153` `def extract(self, model)`
- `DiracDeltaAnalyzer.__init__` (method) `dirac_polos_zeros.py:161` `def __init__(self, config)`
- `DiracDeltaAnalyzer.analyze` (method) `dirac_polos_zeros.py:164` `def analyze(self, charge_density)`
- `ElectricFieldCalculator.__init__` (method) `dirac_polos_zeros.py:193` `def __init__(self, config)`
- `ElectricFieldCalculator.calculate` (method) `dirac_polos_zeros.py:196` `def calculate(self, dirac_data, eval_points)`
- `ElectricFluxCalculator.__init__` (method) `dirac_polos_zeros.py:226` `def __init__(self, config)`
- `ElectricFluxCalculator.calculate` (method) `dirac_polos_zeros.py:229` `def calculate(self, electric_field, surface_points)`
- `DivergenceCalculator.calculate` (method) `dirac_polos_zeros.py:249` `def calculate(self, electric_field)`
- `GaussLawVerifier.__init__` (method) `dirac_polos_zeros.py:254` `def __init__(self, config)`
- `GaussLawVerifier.verify` (method) `dirac_polos_zeros.py:257` `def verify(self, dirac_data, flux_data)`
- `StateSpaceExtractor.extract` (method) `dirac_polos_zeros.py:272` `def extract(self, model)`
- `TransferFunctionComputer.compute` (method) `dirac_polos_zeros.py:299` `def compute(self, A, B, C, D)`
- `PoleZeroAnalyzer.__init__` (method) `dirac_polos_zeros.py:314` `def __init__(self, numerator, denominator, config)`
- `PoleZeroAnalyzer.get_poles` (method) `dirac_polos_zeros.py:334` `def get_poles(self)`
- `PoleZeroAnalyzer.get_zeros` (method) `dirac_polos_zeros.py:337` `def get_zeros(self)`
- `PoleZeroAnalyzer.analyze_stability` (method) `dirac_polos_zeros.py:340` `def analyze_stability(self)`
- `PoleZeroAnalyzer.classify_poles` (method) `dirac_polos_zeros.py:378` `def classify_poles(self)`
- `PoleZeroAnalyzer.compute_damping` (method) `dirac_polos_zeros.py:406` `def compute_damping(self)`
- `PoleZeroAnalyzer.compute_time_constants` (method) `dirac_polos_zeros.py:445` `def compute_time_constants(self)`
- `FrequencyResponseAnalyzer.__init__` (method) `dirac_polos_zeros.py:464` `def __init__(self, numerator, denominator, config)`
- `FrequencyResponseAnalyzer.compute_bode` (method) `dirac_polos_zeros.py:474` `def compute_bode(self)`
- `FrequencyResponseAnalyzer.compute_margins` (method) `dirac_polos_zeros.py:490` `def compute_margins(self)`
- `FrequencyResponseAnalyzer.compute_nyquist` (method) `dirac_polos_zeros.py:516` `def compute_nyquist(self)`
- `FrequencyResponseAnalyzer.evaluate_nyquist_stability` (method) `dirac_polos_zeros.py:535` `def evaluate_nyquist_stability(self, nyquist_data)`
- `TimeResponseAnalyzer.__init__` (method) `dirac_polos_zeros.py:567` `def __init__(self, numerator, denominator, config)`
- `TimeResponseAnalyzer.compute_step` (method) `dirac_polos_zeros.py:577` `def compute_step(self)`
- `TimeResponseAnalyzer.compute_impulse` (method) `dirac_polos_zeros.py:589` `def compute_impulse(self)`
- `TimeResponseAnalyzer.analyze_step_characteristics` (method) `dirac_polos_zeros.py:601` `def analyze_step_characteristics(self, step_data)`
- `CheckpointLoader.load` (method) `dirac_polos_zeros.py:654` `def load(self, path, device)`
- `CheckpointMigrator.migrate` (method) `dirac_polos_zeros.py:662` `def migrate(self, raw_data)`
- `ChargeDistributionVisualizer.__init__` (method) `dirac_polos_zeros.py:711` `def __init__(self, config)`
- `ChargeDistributionVisualizer.visualize` (method) `dirac_polos_zeros.py:714` `def visualize(self, data, output_path)`
- `ElectricFieldVisualizer.__init__` (method) `dirac_polos_zeros.py:739` `def __init__(self, config)`
- `ElectricFieldVisualizer.visualize` (method) `dirac_polos_zeros.py:742` `def visualize(self, data, output_path)`
- `DivergenceVisualizer.__init__` (method) `dirac_polos_zeros.py:781` `def __init__(self, config)`
- `DivergenceVisualizer.visualize` (method) `dirac_polos_zeros.py:784` `def visualize(self, data, output_path)`
- `PoleZeroVisualizer.__init__` (method) `dirac_polos_zeros.py:810` `def __init__(self, config)`
- `PoleZeroVisualizer.visualize` (method) `dirac_polos_zeros.py:813` `def visualize(self, data, output_path)`
- `BodeVisualizer.__init__` (method) `dirac_polos_zeros.py:849` `def __init__(self, config)`
- `BodeVisualizer.visualize` (method) `dirac_polos_zeros.py:852` `def visualize(self, data, output_path)`
- `NyquistVisualizer.__init__` (method) `dirac_polos_zeros.py:900` `def __init__(self, config)`
- `NyquistVisualizer.visualize` (method) `dirac_polos_zeros.py:903` `def visualize(self, data, output_path)`
- `TimeResponseVisualizer.__init__` (method) `dirac_polos_zeros.py:937` `def __init__(self, config)`
- `TimeResponseVisualizer.visualize` (method) `dirac_polos_zeros.py:940` `def visualize(self, data, output_path)`
- `CombinedVisualizer.__init__` (method) `dirac_polos_zeros.py:967` `def __init__(self, config)`
- `CombinedVisualizer.visualize` (method) `dirac_polos_zeros.py:970` `def visualize(self, data, output_path)`
- `SystemAnalyzer.__init__` (method) `dirac_polos_zeros.py:1071` `def __init__(self, checkpoint_path, config)`
- `SystemAnalyzer.analyze` (method) `dirac_polos_zeros.py:1104` `def analyze(self)`
- `AnalysisPipeline.__init__` (method) `dirac_polos_zeros.py:1289` `def __init__(self, config)`
- `AnalysisPipeline.process_checkpoint` (method) `dirac_polos_zeros.py:1300` `def process_checkpoint(self, checkpoint_path, output_dir)`
- `AnalysisPipeline.process_directory` (method) `dirac_polos_zeros.py:1357` `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `AnalysisPipeline.generate_summary` (method) `dirac_polos_zeros.py:1385` `def generate_summary(self, all_results, output_dir)`
- `AnalysisPipeline.main` (method) `dirac_polos_zeros.py:1545` `def main()`

## experiments/ablation/ablation_study.py
- `BenchmarkResult.mean_time` (method) `experiments/ablation/ablation_study.py:35` `def mean_time(self)`
- `BenchmarkResult.std_time` (method) `experiments/ablation/ablation_study.py:39` `def std_time(self)`
- `BenchmarkResult.min_time` (method) `experiments/ablation/ablation_study.py:43` `def min_time(self)`
- `BenchmarkResult.max_time` (method) `experiments/ablation/ablation_study.py:47` `def max_time(self)`
- `BenchmarkResult.mean_gflops` (method) `experiments/ablation/ablation_study.py:51` `def mean_gflops(self)`
- `BenchmarkResult.load_libraries` (method) `experiments/ablation/ablation_study.py:54` `def load_libraries()` -- Cargar bibliotecas con manejo de errores
- `BenchmarkResult.run_openblas` (method) `experiments/ablation/ablation_study.py:111` `def run_openblas(libs, A, B, C, n)` -- Ejecutar multiplicación con OpenBLAS
- `BenchmarkResult.run_strassen` (method) `experiments/ablation/ablation_study.py:122` `def run_strassen(libs, name, func_name, A, B, C, n)` -- Ejecutar multiplicación con Strassen
- `BenchmarkResult.benchmark_single` (method) `experiments/ablation/ablation_study.py:132` `def benchmark_single(libs, algo_name, func_name, A, B, C, C_ref, n, n_runs, warmup)` -- Benchmark una implementación
- `BenchmarkResult.run_ablation` (method) `experiments/ablation/ablation_study.py:180` `def run_ablation(libs, sizes, n_runs, warmup)` -- Ejecutar ablación completa
- `BenchmarkResult.analyze_results` (method) `experiments/ablation/ablation_study.py:233` `def analyze_results(results)` -- Analizar y presentar resultados
- `BenchmarkResult.main` (method) `experiments/ablation/ablation_study.py:273` `def main()`

## experiments/apendix_experiments.py
- `setup_matplotlib` (function) `experiments/apendix_experiments.py:34` `def setup_matplotlib()`
- `StrassenOperator.__init__` (method) `experiments/apendix_experiments.py:53` `def __init__(self, rank, symmetric_init)`
- `StrassenOperator.forward` (method) `experiments/apendix_experiments.py:67` `def forward(self, A, B)`
- `StrassenOperator.slot_importance` (method) `experiments/apendix_experiments.py:77` `def slot_importance(self)`
- `StrassenOperator.count_active` (method) `experiments/apendix_experiments.py:83` `def count_active(self, threshold)`
- `StrassenOperator.generate_batch` (method) `experiments/apendix_experiments.py:87` `def generate_batch(n, device)`
- `StrassenOperator.generate_test_set` (method) `experiments/apendix_experiments.py:93` `def generate_test_set(n, device)`
- `StrassenOperator.compute_delta` (method) `experiments/apendix_experiments.py:100` `def compute_delta(model)`
- `StrassenOperator.verify_strassen_structure` (method) `experiments/apendix_experiments.py:117` `def verify_strassen_structure(U_disc, V_disc, W_disc, tolerance)`
- `StrassenOperator.compute_S_theta` (method) `experiments/apendix_experiments.py:138` `def compute_S_theta(model)`
- `StrassenOperator.compute_gradient_covariance` (method) `experiments/apendix_experiments.py:154` `def compute_gradient_covariance(model, batch_size, n_samples)`
- `StrassenOperator.train_with_logging` (method) `experiments/apendix_experiments.py:187` `def train_with_logging(batch_size, total_epochs, lr, wd, symmetric_init, seed, log_interval)`
- `StrassenOperator.sparsify_and_discretize` (method) `experiments/apendix_experiments.py:274` `def sparsify_and_discretize(model, batch_size)`
- `StrassenOperator.run_phase_diagram` (method) `experiments/apendix_experiments.py:327` `def run_phase_diagram()`
- `StrassenOperator.run_batch_size_effect` (method) `experiments/apendix_experiments.py:429` `def run_batch_size_effect()`
- `StrassenOperator.main` (method) `experiments/apendix_experiments.py:526` `def main()`

## experiments/cache_analysis_v2.py
- `cache_analysis` (function) `experiments/cache_analysis_v2.py:7` `def cache_analysis()` -- Full memory analysis for training.

## experiments/extended_experiments/exp2_noise_ablation.py
- `setup_matplotlib` (function) `experiments/extended_experiments/exp2_noise_ablation.py:25` `def setup_matplotlib()`
- `StrassenOperator.__init__` (method) `experiments/extended_experiments/exp2_noise_ablation.py:54` `def __init__(self, rank)`
- `StrassenOperator.forward` (method) `experiments/extended_experiments/exp2_noise_ablation.py:61` `def forward(self, A, B)`
- `StrassenOperator.get_all_parameters` (method) `experiments/extended_experiments/exp2_noise_ablation.py:71` `def get_all_parameters(self)`
- `StrassenOperator.set_parameters` (method) `experiments/extended_experiments/exp2_noise_ablation.py:77` `def set_parameters(self, new_params)` -- Set parameters from a flattened tensor.
- `StrassenOperator.compute_loss` (method) `experiments/extended_experiments/exp2_noise_ablation.py:85` `def compute_loss(self, A, B)` -- Compute MSE loss.
- `StrassenOperator.compute_accuracy` (method) `experiments/extended_experiments/exp2_noise_ablation.py:91` `def compute_accuracy(self, A, B, threshold)` -- Compute accuracy (proportion of predictions within threshold).
- `StrassenOperator.generate_batch` (method) `experiments/extended_experiments/exp2_noise_ablation.py:99` `def generate_batch(n, scale)` -- Generate batch of matrices.
- `StrassenOperator.compute_gradient_covariance_matrix` (method) `experiments/extended_experiments/exp2_noise_ablation.py:107` `def compute_gradient_covariance_matrix(model, n_samples, batch_size)` -- Compute the gradient covariance matrix Σ.
- `StrassenOperator.get_eigenbasis` (method) `experiments/extended_experiments/exp2_noise_ablation.py:139` `def get_eigenbasis(covariance)` -- Get eigenvectors and eigenvalues of covariance matrix.
- `StrassenOperator.load_checkpoint` (method) `experiments/extended_experiments/exp2_noise_ablation.py:150` `def load_checkpoint(checkpoint_path)` -- Load model from checkpoint file.
- `StrassenOperator.experiment_treatment_a_gradient_noise` (method) `experiments/extended_experiments/exp2_noise_ablation.py:172` `def experiment_treatment_a_gradient_noise(model, noise_std, n_test)` -- Treatment A: Add noise to gradients DURING forward/backward pass.
- `StrassenOperator.experiment_treatment_b_weight_noise` (method) `experiments/extended_experiments/exp2_noise_ablation.py:212` `def experiment_treatment_b_weight_noise(model, noise_std, n_test)` -- Treatment B: Noise on weights BEFORE evaluation (already done in paper).
- `StrassenOperator.experiment_treatment_c_structured_noise` (method) `experiments/extended_experiments/exp2_noise_ablation.py:246` `def experiment_treatment_c_structured_noise(model, covariance, noise_std, n_test)` -- Treatment C: Structured noise by eigenvectors of Σ.
- `StrassenOperator.run_noise_ablation` (method) `experiments/extended_experiments/exp2_noise_ablation.py:315` `def run_noise_ablation(checkpoint_path, noise_levels)` -- Run complete noise ablation experiment on a checkpoint.
- `StrassenOperator.main` (method) `experiments/extended_experiments/exp2_noise_ablation.py:348` `def main()` -- Main execution for Experiment 2.
- `StrassenOperator.generate_visualization` (method) `experiments/extended_experiments/exp2_noise_ablation.py:433` `def generate_visualization(results, output_dir)` -- Generate publication-quality figures.

## experiments/extended_experiments/exp3_prospective_prediction.py
- `setup_matplotlib` (function) `experiments/extended_experiments/exp3_prospective_prediction.py:32` `def setup_matplotlib()`
- `StrassenOperator.__init__` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:59` `def __init__(self, rank)`
- `StrassenOperator.forward` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:66` `def forward(self, A, B)`
- `StrassenOperator.get_all_parameters` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:76` `def get_all_parameters(self)`
- `StrassenOperator.set_parameters` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:82` `def set_parameters(self, new_params)`
- `StrassenOperator.count_active_slots` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:89` `def count_active_slots(self, threshold)` -- Count active slots based on weight norms.
- `StrassenOperator.compute_discretization_margin` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:97` `def compute_discretization_margin(self)` -- Compute how close weights are to discrete values {-1, 0, 1}. δ(θ) = mean(|w - round(w)|)
- `StrassenOperator.is_grokked` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:107` `def is_grokked(self, margin_threshold, active_slots_target)` -- Check if model has grokked (discretized with low error).
- `StrassenOperator.generate_batch` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:114` `def generate_batch(n, scale)` -- Generate batch of matrices.
- `StrassenOperator.compute_kappa` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:121` `def compute_kappa(model, n_samples, batch_size)` -- Compute condition number κ(Σ) of gradient covariance matrix.
- `StrassenOperator.load_checkpoint` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:166` `def load_checkpoint(checkpoint_path)` -- Load model from checkpoint file.
- `StrassenOperator.simulate_early_prediction` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:188` `def simulate_early_prediction(checkpoint_path, early_epoch_fraction)` -- Simulate the prospective prediction experiment.
- `StrassenOperator.run_prospective_prediction_experiment` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:236` `def run_prospective_prediction_experiment(checkpoint_files)` -- Run the full prospective prediction experiment across all checkpoints.
- `StrassenOperator.compute_roc_analysis` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:252` `def compute_roc_analysis(predictions)` -- Compute ROC curve and AUC for κ as predictor of success.
- `StrassenOperator.main` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:313` `def main()` -- Main execution for Experiment 3.
- `StrassenOperator.generate_visualization` (method) `experiments/extended_experiments/exp3_prospective_prediction.py:410` `def generate_visualization(results, predictions, output_dir)` -- Generate publication-quality figures.

## experiments/extended_experiments/exp4_trajectory_perturbation.py
- `setup_matplotlib` (function) `experiments/extended_experiments/exp4_trajectory_perturbation.py:27` `def setup_matplotlib()`
- `StrassenOperator.__init__` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:54` `def __init__(self, rank)`
- `StrassenOperator.forward` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:61` `def forward(self, A, B)`
- `StrassenOperator.get_all_parameters` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:71` `def get_all_parameters(self)`
- `StrassenOperator.set_parameters` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:77` `def set_parameters(self, new_params)`
- `StrassenOperator.get_weight_norm` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:84` `def get_weight_norm(self)` -- Get total L2 norm of all parameters.
- `StrassenOperator.get_weight_direction` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:91` `def get_weight_direction(self)` -- Get normalized weight vector direction.
- `StrassenOperator.compute_gradient_norm` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:96` `def compute_gradient_norm(self, A, B)` -- Compute norm of gradients.
- `StrassenOperator.cosine_similarity` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:110` `def cosine_similarity(self, other_params)` -- Compute cosine similarity between current weights and target weights.
- `StrassenOperator.generate_batch` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:117` `def generate_batch(n, scale)` -- Generate batch of matrices.
- `StrassenOperator.load_checkpoint` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:124` `def load_checkpoint(checkpoint_path)` -- Load model from checkpoint file.
- `StrassenOperator.simulate_trajectory_perturbation` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:146` `def simulate_trajectory_perturbation(checkpoint_path, perturbations)` -- Simulate trajectory perturbation effects using available checkpoints.
- `StrassenOperator.compute_metrics` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:194` `def compute_metrics(model, name)` -- Compute evaluation metrics.
- `StrassenOperator.main` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:277` `def main()` -- Main execution for Experiment 4.
- `StrassenOperator.generate_visualization` (method) `experiments/extended_experiments/exp4_trajectory_perturbation.py:398` `def generate_visualization(results, output_dir)` -- Generate publication-quality figures.

## experiments/extended_experiments/run_all_experiments.py
- `setup_matplotlib` (function) `experiments/extended_experiments/run_all_experiments.py:24` `def setup_matplotlib()`
- `StrassenOperator.__init__` (method) `experiments/extended_experiments/run_all_experiments.py:51` `def __init__(self, rank)`
- `StrassenOperator.forward` (method) `experiments/extended_experiments/run_all_experiments.py:58` `def forward(self, A, B)`
- `StrassenOperator.get_all_parameters` (method) `experiments/extended_experiments/run_all_experiments.py:68` `def get_all_parameters(self)`
- `StrassenOperator.set_parameters` (method) `experiments/extended_experiments/run_all_experiments.py:74` `def set_parameters(self, new_params)`
- `StrassenOperator.count_active_slots` (method) `experiments/extended_experiments/run_all_experiments.py:81` `def count_active_slots(self, threshold)`
- `StrassenOperator.compute_discretization_margin` (method) `experiments/extended_experiments/run_all_experiments.py:88` `def compute_discretization_margin(self)`
- `StrassenOperator.generate_batch` (method) `experiments/extended_experiments/run_all_experiments.py:95` `def generate_batch(n, scale)`
- `StrassenOperator.load_checkpoint_robust` (method) `experiments/extended_experiments/run_all_experiments.py:101` `def load_checkpoint_robust(checkpoint_path, model)` -- Load checkpoint with multiple format fallback strategies.
- `StrassenOperator.compute_gradient_covariance_safe` (method) `experiments/extended_experiments/run_all_experiments.py:135` `def compute_gradient_covariance_safe(model, batch_size, n_samples)` -- Compute κ(Σₜ) with numerical safety.
- `StrassenOperator.run_all_experiments` (method) `experiments/extended_experiments/run_all_experiments.py:205` `def run_all_experiments()` -- Run all experiments.
- `StrassenOperator.compute_accuracy` (method) `experiments/extended_experiments/run_all_experiments.py:324` `def compute_accuracy()`
- `StrassenOperator.generate_summary_visualization` (method) `experiments/extended_experiments/run_all_experiments.py:496` `def generate_summary_visualization(results, output_dir)` -- Generate summary visualization.

## experiments/extended_experiments/validate2.py
- `StrassenOperator.__init__` (method) `experiments/extended_experiments/validate2.py:139` `def __init__(self, rank)`
- `StrassenOperator.forward` (method) `experiments/extended_experiments/validate2.py:156` `def forward(self, A, B)` -- Computar A @ B usando descomposición tensorial.
- `StrassenOperator.slot_importance` (method) `experiments/extended_experiments/validate2.py:183` `def slot_importance(self)` -- Importancia de cada slot basada en normas.
- `StrassenOperator.count_active` (method) `experiments/extended_experiments/validate2.py:190` `def count_active(self, threshold)` -- Contar slots activos.
- `StrassenOperator.compute_SP` (method) `experiments/extended_experiments/validate2.py:194` `def compute_SP(self)` -- Métrica de Sparsity.
- `StrassenOperator.get_state_dict` (method) `experiments/extended_experiments/validate2.py:202` `def get_state_dict(self)` -- Obtener estado completo para checkpointing.
- `StrassenOperator.load_state_dict` (method) `experiments/extended_experiments/validate2.py:211` `def load_state_dict(self, state_dict)` -- Cargar estado completo desde checkpoint.
- `StrassenDataGenerator.__init__` (method) `experiments/extended_experiments/validate2.py:228` `def __init__(self, num_samples, matrix_size, seed)`
- `StrassenDataGenerator.generate_matrix` (method) `experiments/extended_experiments/validate2.py:238` `def generate_matrix(self)` -- Generar matriz aleatoria con valores enteros.
- `StrassenDataGenerator.generate_data` (method) `experiments/extended_experiments/validate2.py:242` `def generate_data(self)` -- Generar pares de matrices y sus productos.
- `StrassenDataGenerator.get_train_test` (method) `experiments/extended_experiments/validate2.py:260` `def get_train_test(self, test_ratio)` -- Dividir en conjuntos de entrenamiento y prueba.
- `LocalComplexityCalculator.__init__` (method) `experiments/extended_experiments/validate2.py:287` `def __init__(self, model, config)`
- `LocalComplexityCalculator.compute_lc` (method) `experiments/extended_experiments/validate2.py:292` `def compute_lc(self, batch_inputs, batch_targets)` -- Calcular LC para un batch específico.
- `LocalComplexityCalculator.compute_batch_diversity` (method) `experiments/extended_experiments/validate2.py:326` `def compute_batch_diversity(self, batch_inputs)` -- Calcular diversidad del batch basada en varianza de activaciones.
- `GrokkingVerifier.__init__` (method) `experiments/extended_experiments/validate2.py:343` `def __init__(self, config)`
- `GrokkingVerifier.verify` (method) `experiments/extended_experiments/validate2.py:347` `def verify(self, model, n_test)` -- Verificar que el operador ha grokkeado correctamente.
- `IterativePruningEngine.__init__` (method) `experiments/extended_experiments/validate2.py:420` `def __init__(self, config)`
- `IterativePruningEngine.get_weight_magnitudes` (method) `experiments/extended_experiments/validate2.py:426` `def get_weight_magnitudes(self, model)` -- Obtener magnitud absoluta de todos los pesos.
- `IterativePruningEngine.compute_sparsity` (method) `experiments/extended_experiments/validate2.py:431` `def compute_sparsity(self, model)` -- Calcular porcentaje de pesos en cero.
- `IterativePruningEngine.prune_percent` (method) `experiments/extended_experiments/validate2.py:437` `def prune_percent(self, model, percent)` -- Podar el porcentaje especificado de pesos menos importantes.
- `IterativePruningEngine.fine_tune` (method) `experiments/extended_experiments/validate2.py:457` `def fine_tune(self, model, train_data)` -- Fine-tune del modelo podado con métricas completas.
- `IterativePruningEngine.run_protocol` (method) `experiments/extended_experiments/validate2.py:624` `def run_protocol(self, model, train_data)` -- Ejecutar protocolo completo de poda iterativa.
- `LocalComplexityExperiment.__init__` (method) `experiments/extended_experiments/validate2.py:775` `def __init__(self, config)`
- `LocalComplexityExperiment.run_full_experiment` (method) `experiments/extended_experiments/validate2.py:779` `def run_full_experiment(self, target_epochs)` -- Ejecutar experimento completo de LC entrenando desde cero.
- `BalancedRunsGenerator.__init__` (method) `experiments/extended_experiments/validate2.py:903` `def __init__(self, config)`
- `BalancedRunsGenerator.run_balanced_experiments` (method) `experiments/extended_experiments/validate2.py:907` `def run_balanced_experiments(self, n_runs)` -- Ejecutar multiples runs con condiciones disenhadas para producir mix.
- `BootstrapStatistics.__init__` (method) `experiments/extended_experiments/validate2.py:1145` `def __init__(self, config)`
- `BootstrapStatistics.compute_roc_with_ci` (method) `experiments/extended_experiments/validate2.py:1149` `def compute_roc_with_ci(self, y_true, y_scores)` -- Calcular curva ROC con intervalos de confianza bootstrap.
- `BootstrapStatistics.compute_kappa_with_ci` (method) `experiments/extended_experiments/validate2.py:1236` `def compute_kappa_with_ci(self, y_true, y_pred)` -- Calcular Kappa de Cohen con IC bootstrap.
- `BootstrapStatistics.compute_accuracy_with_ci` (method) `experiments/extended_experiments/validate2.py:1260` `def compute_accuracy_with_ci(self, correct)` -- Calcular accuracy con IC binomial.
- `VisualizationGenerator.__init__` (method) `experiments/extended_experiments/validate2.py:1284` `def __init__(self, style)`
- `VisualizationGenerator.plot_local_complexity` (method) `experiments/extended_experiments/validate2.py:1298` `def plot_local_complexity(self, epochs, lc_values, accuracy, save_path)` -- Graficar evolución de Local Complexity y Accuracy.
- `VisualizationGenerator.plot_pruning_results` (method) `experiments/extended_experiments/validate2.py:1342` `def plot_pruning_results(self, pruning_data, save_path)` -- Graficar resultados de poda iterativa.
- `VisualizationGenerator.plot_roc_with_ci` (method) `experiments/extended_experiments/validate2.py:1404` `def plot_roc_with_ci(self, roc_data, save_path)` -- Graficar curva ROC con intervalos de confianza.
- `VisualizationGenerator.plot_balanced_runs_results` (method) `experiments/extended_experiments/validate2.py:1467` `def plot_balanced_runs_results(self, balanced_data, save_path)` -- Graficar resultados del experimento de runs balanceados.
- `VisualizationGenerator.plot_discretization_results` (method) `experiments/extended_experiments/validate2.py:1556` `def plot_discretization_results(self, pruning_data, save_path)` -- Graficar resultados de discretizacion.
- `ExperimentOrchestrator.__init__` (method) `experiments/extended_experiments/validate2.py:1659` `def __init__(self, config)`
- `ExperimentOrchestrator.find_grokked_checkpoint` (method) `experiments/extended_experiments/validate2.py:1689` `def find_grokked_checkpoint(self)` -- Buscar checkpoint grokkeado en múltiples ubicaciones.
- `ExperimentOrchestrator.load_grokked_checkpoint` (method) `experiments/extended_experiments/validate2.py:1733` `def load_grokked_checkpoint(self, checkpoint_path)` -- Cargar checkpoint grokkeado y verificar que grokkeó.
- `ExperimentOrchestrator.verify_checkpoint_is_grokked` (method) `experiments/extended_experiments/validate2.py:1768` `def verify_checkpoint_is_grokked(self)` -- Verificar que el checkpoint cargado realmente grokkeó.
- `ExperimentOrchestrator.run_local_complexity_experiment` (method) `experiments/extended_experiments/validate2.py:1795` `def run_local_complexity_experiment(self, epochs)` -- Ejecutar experimento de Local Complexity vs Época.
- `ExperimentOrchestrator.run_lc_training_experiment` (method) `experiments/extended_experiments/validate2.py:1897` `def run_lc_training_experiment(self, epochs)` -- Ejecutar experimento de Local Complexity .
- `ExperimentOrchestrator.run_pruning_experiment` (method) `experiments/extended_experiments/validate2.py:1936` `def run_pruning_experiment(self)` -- Ejecutar protocolo de poda iterativa + fine-tuning.
- `ExperimentOrchestrator.run_balanced_runs_experiment` (method) `experiments/extended_experiments/validate2.py:1980` `def run_balanced_runs_experiment(self, n_runs)` -- Ejecutar experimento de runs balanceados (PUNTO C DEL REVISOR).
- `ExperimentOrchestrator.run_roc_analysis` (method) `experiments/extended_experiments/validate2.py:2043` `def run_roc_analysis(self)` -- Ejecutar análisis ROC/AUC con bootstrap.
- `ExperimentOrchestrator.generate_summary_report` (method) `experiments/extended_experiments/validate2.py:2124` `def generate_summary_report(self)` -- Generar reporte de resumen en markdown.
- `ExperimentOrchestrator.save_results` (method) `experiments/extended_experiments/validate2.py:2197` `def save_results(self)` -- Guardar todos los resultados.
- `ExperimentOrchestrator.run_all_experiments` (method) `experiments/extended_experiments/validate2.py:2235` `def run_all_experiments(self, checkpoint_path)` -- Ejecutar suite completa de experimentos.
- `ExperimentOrchestrator.find_grokked_checkpoint` (method) `experiments/extended_experiments/validate2.py:2313` `def find_grokked_checkpoint()` -- Buscar checkpoint grokkeado en múltiples ubicaciones.
- `ExperimentOrchestrator.analyze_checkpoints` (method) `experiments/extended_experiments/validate2.py:2353` `def analyze_checkpoints()` -- Analizar todos los checkpoints disponibles para encontrar el grokkeado.
- `ExperimentOrchestrator.main` (method) `experiments/extended_experiments/validate2.py:2423` `def main()` -- Punto de entrada principal.

## experiments/generate_figures.py
- `setup_matplotlib_for_plotting` (function) `experiments/generate_figures.py:15` `def setup_matplotlib_for_plotting()` -- Configure matplotlib and seaborn for proper rendering.
- `generate_benchmark_figure` (function) `experiments/generate_figures.py:63` `def generate_benchmark_figure()` -- Generate benchmark performance comparison plot.
- `generate_ablation_figure` (function) `experiments/generate_figures.py:124` `def generate_ablation_figure()` -- Generate ablation study visualization.
- `load_checkpoint_weights` (function) `experiments/generate_figures.py:219` `def load_checkpoint_weights()` -- Load all checkpoint files and extract weight tensors.
- `generate_weight_geometry_figure` (function) `experiments/generate_figures.py:258` `def generate_weight_geometry_figure()` -- Generate weight space geometry visualization.
- `generate_phase_transition_figure` (function) `experiments/generate_figures.py:354` `def generate_phase_transition_figure()` -- Generate phase transition analysis from checkpoint evolution.
- `generate_coherence_figure` (function) `experiments/generate_figures.py:465` `def generate_coherence_figure()` -- Generate cache coherence analysis visualization.
- `generate_crystallization_figure` (function) `experiments/generate_figures.py:534` `def generate_crystallization_figure()` -- Visualize the crystallization of Strassen coefficients.
- `main` (function) `experiments/generate_figures.py:623` `def main()`

## experiments/statistics/coherence_analysis.py
- `strassen_numpy` (function) `experiments/statistics/coherence_analysis.py:15` `def strassen_numpy(A, B, threshold)`
- `run_coherence_analysis` (function) `experiments/statistics/coherence_analysis.py:42` `def run_coherence_analysis()`

## experiments/statistics/rigorous_experiment.py
- `StrassenModel.__init__` (method) `experiments/statistics/rigorous_experiment.py:108` `def __init__(self, config)`
- `StrassenModel.forward` (method) `experiments/statistics/rigorous_experiment.py:124` `def forward(self, x)`
- `StrassenModel.generate_data` (method) `experiments/statistics/rigorous_experiment.py:131` `def generate_data(n_samples, seed)` -- Generate matrix multiplication dataset
- `StrassenModel.compute_discretization_error` (method) `experiments/statistics/rigorous_experiment.py:143` `def compute_discretization_error(model, values)` -- Compute mean distance to nearest discrete value
- `StrassenModel.compute_spectral_gap` (method) `experiments/statistics/rigorous_experiment.py:157` `def compute_spectral_gap(model)` -- Compute maximum spectral gap ratio
- `StrassenModel.run_single_experiment` (method) `experiments/statistics/rigorous_experiment.py:169` `def run_single_experiment(batch_size, seed, run_id, config)` -- Run a single controlled experiment
- `StrassenModel.run_full_experiment` (method) `experiments/statistics/rigorous_experiment.py:269` `def run_full_experiment(batch_sizes, n_seeds, n_runs_per_seed)` -- Run complete factorial experiment
- `StrassenModel.perform_anova` (method) `experiments/statistics/rigorous_experiment.py:306` `def perform_anova(results)` -- Perform full factorial ANOVA
- `StrassenModel.print_anova_table` (method) `experiments/statistics/rigorous_experiment.py:401` `def print_anova_table(anova)` -- Print formatted ANOVA table
- `StrassenModel.fit_noise_model` (method) `experiments/statistics/rigorous_experiment.py:448` `def fit_noise_model(results)` -- Fit theoretical noise model: Var(loss) = α/B + β·cache_miss(B) + γ
- `StrassenModel.cache_miss_proxy` (method) `experiments/statistics/rigorous_experiment.py:467` `def cache_miss_proxy(B)`
- `StrassenModel.full_model` (method) `experiments/statistics/rigorous_experiment.py:472` `def full_model(B, alpha, beta, gamma)`
- `StrassenModel.null_model` (method) `experiments/statistics/rigorous_experiment.py:476` `def null_model(B, alpha, gamma)`
- `StrassenModel.find_optimal_B` (method) `experiments/statistics/rigorous_experiment.py:519` `def find_optimal_B(results, n_bootstrap)` -- Find optimal batch size with bootstrap confidence interval
- `StrassenModel.get_mean_error` (method) `experiments/statistics/rigorous_experiment.py:526` `def get_mean_error(data, B)`
- `StrassenModel.generate_report` (method) `experiments/statistics/rigorous_experiment.py:555` `def generate_report(results, config)` -- Generate complete statistical report

## experiments/validation/benchmark.py
- `strassen_numpy` (function) `experiments/validation/benchmark.py:15` `def strassen_numpy(A, B, threshold)` -- Strassen recursivo con NumPy para productos base.
- `measure_single_sgemm` (function) `experiments/validation/benchmark.py:49` `def measure_single_sgemm(n, threads)` -- Mide tiempo de un solo sgemm de tamaño n.
- `run_planck_analysis` (function) `experiments/validation/benchmark.py:59` `def run_planck_analysis()` -- Ejecuta el análisis del Límite de Planck.

## experiments/validation_experiments.py
- `strassen_2x2` (function) `experiments/validation_experiments.py:47` `def strassen_2x2(A, B, U, V, W)` -- Compute 2x2 matrix multiplication using Strassen coefficients.
- `strassen_recursive` (function) `experiments/validation_experiments.py:56` `def strassen_recursive(A, B, U, V, W, threshold)` -- Recursive Strassen for NxN matrices.
- `test_uniqueness_via_permutation` (function) `experiments/validation_experiments.py:85` `def test_uniqueness_via_permutation()` -- Test that permuting slots produces equivalent computation.
- `test_noise_stability` (function) `experiments/validation_experiments.py:125` `def test_noise_stability()` -- Test stability under Gaussian noise.
- `test_expansion_sizes` (function) `experiments/validation_experiments.py:162` `def test_expansion_sizes()` -- Test expansion to larger sizes.
- `simulate_grokking_dynamics` (function) `experiments/validation_experiments.py:184` `def simulate_grokking_dynamics()` -- Simulate grokking dynamics for visualization.
- `compute_cache_math` (function) `experiments/validation_experiments.py:250` `def compute_cache_math()` -- Compute L3 cache requirements for different batch sizes.
- `main` (function) `experiments/validation_experiments.py:294` `def main()` -- Run all validation experiments.
- `convert_types` (function) `experiments/validation_experiments.py:313` `def convert_types(obj)`

## experiments/verify_checkpoints.py
- `StrassenBilinear.__init__` (method) `experiments/verify_checkpoints.py:27` `def __init__(self, rank)`
- `StrassenBilinear.forward` (method) `experiments/verify_checkpoints.py:34` `def forward(self, A, B)`
- `StrassenBilinear.get_discrete_coefficients` (method) `experiments/verify_checkpoints.py:44` `def get_discrete_coefficients(self)`
- `StrassenBilinear.compute_delta` (method) `experiments/verify_checkpoints.py:51` `def compute_delta(model)`
- `StrassenBilinear.verify_2x2` (method) `experiments/verify_checkpoints.py:68` `def verify_2x2(U, V, W, n_test)`
- `StrassenBilinear.strassen_expand` (method) `experiments/verify_checkpoints.py:89` `def strassen_expand(A, B, U, V, W)`
- `StrassenBilinear.verify_expansion` (method) `experiments/verify_checkpoints.py:126` `def verify_expansion(U, V, W, sizes)`
- `StrassenBilinear.compute_S_theta` (method) `experiments/verify_checkpoints.py:152` `def compute_S_theta(model)`
- `StrassenBilinear.load_checkpoint` (method) `experiments/verify_checkpoints.py:166` `def load_checkpoint(path)`
- `StrassenBilinear.verify_checkpoint` (method) `experiments/verify_checkpoints.py:183` `def verify_checkpoint(checkpoint_path)`
- `StrassenBilinear.run_noise_stability_test` (method) `experiments/verify_checkpoints.py:226` `def run_noise_stability_test(checkpoint_path, noise_levels)`
- `StrassenBilinear.main` (method) `experiments/verify_checkpoints.py:249` `def main()`

## experimetn2.py
- `StrassStrassenModel.__init__` (method) `experimetn2.py:101` `def __init__(self, config)`
- `StrassStrassenModel.forward` (method) `experimetn2.py:115` `def forward(self, A, B)`
- `StrassStrassenModel.get_coefficients` (method) `experimetn2.py:125` `def get_coefficients(self)`
- `StrassStrassenModel.slot_importance` (method) `experimetn2.py:128` `def slot_importance(self)`
- `ComplexStrassStrassenModel.__init__` (method) `experimetn2.py:140` `def __init__(self, config, gamma)`
- `ComplexStrassStrassenModel.get_complex_tensors` (method) `experimetn2.py:153` `def get_complex_tensors(self)`
- `ComplexStrassStrassenModel.forward` (method) `experimetn2.py:160` `def forward(self, A, B)`
- `StrassenDataGenerator.__init__` (method) `experimetn2.py:180` `def __init__(self, config)`
- `StrassenDataGenerator.generate_batch` (method) `experimetn2.py:183` `def generate_batch(self, batch_size)`
- `CheckpointManager.save` (method) `experimetn2.py:192` `def save(self, model, epoch, metrics, path)`
- `LevelSpacingRatioCalculator.__init__` (method) `experimetn2.py:209` `def __init__(self, tolerance)`
- `LevelSpacingRatioCalculator.calculate_r_ratio` (method) `experimetn2.py:212` `def calculate_r_ratio(self, eigenvalues)`
- `ExactHessianCalculator.__init__` (method) `experimetn2.py:246` `def __init__(self, config)`
- `ExactHessianCalculator.compute_hessian` (method) `experimetn2.py:249` `def compute_hessian(self, model, A, B, C_true)`
- `ExactHessianCalculator.loss_fn` (method) `experimetn2.py:254` `def loss_fn(flat_param_tensor)`
- `SyntheticPlanckCalculator.__init__` (method) `experimetn2.py:280` `def __init__(self, noise_floor)`
- `SyntheticPlanckCalculator.calculate` (method) `experimetn2.py:283` `def calculate(self, model, current_loss)`
- `SuperpositionMetricCalculator.__init__` (method) `experimetn2.py:330` `def __init__(self, config)`
- `SuperpositionMetricCalculator.calculate` (method) `experimetn2.py:333` `def calculate(self, model, datagen)`
- `IExperiment.run` (method) `experimetn2.py:388` `def run(self, model)`
- `IExperiment.get_name` (method) `experimetn2.py:390` `def get_name(self)`
- `Experiment1RicciMBLDuality.__init__` (method) `experimetn2.py:399` `def __init__(self, suite_config, datagen)`
- `Experiment1RicciMBLDuality.get_name` (method) `experimetn2.py:405` `def get_name(self)`
- `Experiment1RicciMBLDuality.run` (method) `experimetn2.py:408` `def run(self, model)`
- `Experiment2AltlandZirnbauer.__init__` (method) `experimetn2.py:463` `def __init__(self, suite_config, datagen)`
- `Experiment2AltlandZirnbauer.get_name` (method) `experimetn2.py:468` `def get_name(self)`
- `Experiment2AltlandZirnbauer.run` (method) `experimetn2.py:471` `def run(self, model)`
- `Experiment3ConformalIsomorphism.__init__` (method) `experimetn2.py:527` `def __init__(self, suite_config, datagen)`
- `Experiment3ConformalIsomorphism.get_name` (method) `experimetn2.py:531` `def get_name(self)`
- `Experiment3ConformalIsomorphism.run` (method) `experimetn2.py:534` `def run(self, model)`
- `Experiment4CompressionFrontier.__init__` (method) `experimetn2.py:576` `def __init__(self, suite_config, datagen)`
- `Experiment4CompressionFrontier.get_name` (method) `experimetn2.py:582` `def get_name(self)`
- `Experiment4CompressionFrontier.run` (method) `experimetn2.py:585` `def run(self, model)`
- `Experiment5HolographicPruning.__init__` (method) `experimetn2.py:630` `def __init__(self, suite_config, datagen)`
- `Experiment5HolographicPruning.get_name` (method) `experimetn2.py:634` `def get_name(self)`
- `Experiment5HolographicPruning.run` (method) `experimetn2.py:637` `def run(self, model)`
- `UnifiedSuite.__init__` (method) `experimetn2.py:701` `def __init__(self, config)`
- `UnifiedSuite.execute_all` (method) `experimetn2.py:712` `def execute_all(self)`
- `UnifiedSuite.main` (method) `experimetn2.py:745` `def main()`

## fermi.py
- `IModel.get_coefficients` (method) `fermi.py:50` `def get_coefficients(self)`
- `IBlochWaveConstructor.construct` (method) `fermi.py:55` `def construct(self, weights, k)`
- `IBandStructureCalculator.calculate` (method) `fermi.py:60` `def calculate(self, model)`
- `IFermiLevelCalculator.calculate` (method) `fermi.py:65` `def calculate(self, eigenvalues, num_electrons)`
- `IDensityOfStatesCalculator.calculate` (method) `fermi.py:70` `def calculate(self, eigenvalues, energies)`
- `IElectronicPropertiesCalculator.calculate` (method) `fermi.py:75` `def calculate(self, eigenvalues, eigenvectors, fermi_level)`
- `IMetalInsulatorClassifier.classify` (method) `fermi.py:80` `def classify(self, band_gap, dos_at_fermi)`
- `BilinearModel.__init__` (method) `fermi.py:84` `def __init__(self, hidden_dim, matrix_size)`
- `BilinearModel.forward` (method) `fermi.py:100` `def forward(self, a, b)`
- `BilinearModel.get_coefficients` (method) `fermi.py:103` `def get_coefficients(self)`
- `BlochWaveConstructor.__init__` (method) `fermi.py:112` `def __init__(self, config)`
- `BlochWaveConstructor.construct` (method) `fermi.py:115` `def construct(self, weights, k)`
- `BandStructureCalculator.__init__` (method) `fermi.py:136` `def __init__(self, config)`
- `BandStructureCalculator.calculate` (method) `fermi.py:140` `def calculate(self, model)`
- `FermiLevelCalculator.__init__` (method) `fermi.py:216` `def __init__(self, config)`
- `FermiLevelCalculator.calculate` (method) `fermi.py:219` `def calculate(self, eigenvalues, num_electrons)`
- `DensityOfStatesCalculator.__init__` (method) `fermi.py:286` `def __init__(self, config)`
- `DensityOfStatesCalculator.calculate` (method) `fermi.py:289` `def calculate(self, eigenvalues, energies)`
- `ElectronicPropertiesCalculator.__init__` (method) `fermi.py:307` `def __init__(self, config)`
- `ElectronicPropertiesCalculator.calculate` (method) `fermi.py:310` `def calculate(self, eigenvalues, eigenvectors, fermi_level)`
- `MetalInsulatorClassifier.__init__` (method) `fermi.py:369` `def __init__(self, config)`
- `MetalInsulatorClassifier.classify` (method) `fermi.py:372` `def classify(self, band_gap, dos_at_fermi)`
- `MetalInsulatorClassifier.classify_transport` (method) `fermi.py:386` `def classify_transport(self, effective_masses, band_gap)`
- `CheckpointMigrator.migrate` (method) `fermi.py:409` `def migrate(self, raw_data, device)`
- `FermiLevelAnalyzer.__init__` (method) `fermi.py:459` `def __init__(self, checkpoint_path, config)`
- `FermiLevelAnalyzer.analyze` (method) `fermi.py:495` `def analyze(self)`
- `FermiPipeline.__init__` (method) `fermi.py:609` `def __init__(self, config)`
- `FermiPipeline.process_checkpoint` (method) `fermi.py:612` `def process_checkpoint(self, checkpoint_path, output_dir)`
- `FermiPipeline.process_directory` (method) `fermi.py:626` `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `FermiPipeline.generate_summary` (method) `fermi.py:653` `def generate_summary(self, all_results, output_dir)`
- `FermiPipeline.plot_band_structures` (method) `fermi.py:724` `def plot_band_structures(self, all_results, output_dir)` -- Generate comparison plots of band structures across checkpoints.
- `FermiPipeline.main` (method) `fermi.py:766` `def main()`


Next: [API_p2.md](API_p2.md)
