# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis.

**Total Files Parsed:** 60 | **Total Symbols Extracted:** 2071 | **Total Imports:** 638

## Structural Knowledge Map
> **Note:** The visual graph below has been intelligently pruned to the top 300 most relevant nodes to prevent rendering crashes. Full details of all 60 files are documented below.

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray: 5 5,color:#aaa;
    xray_tensor_diffractometer_py["xray_tensor_diffractometer.py (py)"]
    class xray_tensor_diffractometer_py mod;
    xray_tensor_diffractometer_py_Config["Config"]
    class xray_tensor_diffractometer_py_Config cls;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_Config
    xray_tensor_diffractometer_py_set_seed["set_seed"]
    class xray_tensor_diffractometer_py_set_seed fn;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_set_seed
    xray_tensor_diffractometer_py_setup_logger["setup_logger"]
    class xray_tensor_diffractometer_py_setup_logger fn;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_setup_logger
    xray_tensor_diffractometer_py_run_epitaxy_from_best_crystal["run_epitaxy_from_best_crystal"]
    class xray_tensor_diffractometer_py_run_epitaxy_from_best_crystal fn;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_run_epitaxy_from_best_crystal
    xray_tensor_diffractometer_py_ICheckpointLoader["ICheckpointLoader"]
    class xray_tensor_diffractometer_py_ICheckpointLoader cls;
    xray_tensor_diffractometer_py --> xray_tensor_diffractometer_py_ICheckpointLoader
    percolation_analysis_py["percolation_analysis.py (py)"]
    class percolation_analysis_py mod;
    percolation_analysis_py_PercolationConfiguration["PercolationConfiguration"]
    class percolation_analysis_py_PercolationConfiguration cls;
    percolation_analysis_py --> percolation_analysis_py_PercolationConfiguration
    percolation_analysis_py_IModel["IModel"]
    class percolation_analysis_py_IModel cls;
    percolation_analysis_py --> percolation_analysis_py_IModel
    percolation_analysis_py_NumpyModelWrapper["NumpyModelWrapper"]
    class percolation_analysis_py_NumpyModelWrapper cls;
    percolation_analysis_py --> percolation_analysis_py_NumpyModelWrapper
    percolation_analysis_py__DummyObject["_DummyObject"]
    class percolation_analysis_py__DummyObject cls;
    percolation_analysis_py --> percolation_analysis_py__DummyObject
    percolation_analysis_py__safe_torch_load["_safe_torch_load"]
    class percolation_analysis_py__safe_torch_load fn;
    percolation_analysis_py --> percolation_analysis_py__safe_torch_load
    experiments_extended_experiments_all_test_extended_py["all_test_extended.py (py)"]
    class experiments_extended_experiments_all_test_extended_py mod;
    experiments_extended_experiments_all_test_extended_py_Configuration["Configuration"]
    class experiments_extended_experiments_all_test_extended_py_Configuration cls;
    experiments_extended_experiments_all_test_extended_py --> experiments_extended_experiments_all_test_extended_py_Configuration
    experiments_extended_experiments_all_test_extended_py_Narrator["Narrator"]
    class experiments_extended_experiments_all_test_extended_py_Narrator cls;
    experiments_extended_experiments_all_test_extended_py --> experiments_extended_experiments_all_test_extended_py_Narrator
    experiments_extended_experiments_all_test_extended_py_SystemFingerprint["SystemFingerprint"]
    class experiments_extended_experiments_all_test_extended_py_SystemFingerprint cls;
    experiments_extended_experiments_all_test_extended_py --> experiments_extended_experiments_all_test_extended_py_SystemFingerprint
    experiments_extended_experiments_all_test_extended_py_ArithmeticDataset["ArithmeticDataset"]
    class experiments_extended_experiments_all_test_extended_py_ArithmeticDataset cls;
    experiments_extended_experiments_all_test_extended_py --> experiments_extended_experiments_all_test_extended_py_ArithmeticDataset
    experiments_extended_experiments_all_test_extended_py_BilinearModel["BilinearModel"]
    class experiments_extended_experiments_all_test_extended_py_BilinearModel cls;
    experiments_extended_experiments_all_test_extended_py --> experiments_extended_experiments_all_test_extended_py_BilinearModel
    experiments_extended_experiments_validate2_py["validate2.py (py)"]
    class experiments_extended_experiments_validate2_py mod;
    experiments_extended_experiments_validate2_py_ExperimentConfig["ExperimentConfig"]
    class experiments_extended_experiments_validate2_py_ExperimentConfig cls;
    experiments_extended_experiments_validate2_py --> experiments_extended_experiments_validate2_py_ExperimentConfig
    experiments_extended_experiments_validate2_py_StrassenOperator["StrassenOperator"]
    class experiments_extended_experiments_validate2_py_StrassenOperator cls;
    experiments_extended_experiments_validate2_py --> experiments_extended_experiments_validate2_py_StrassenOperator
    experiments_extended_experiments_validate2_py_StrassenDataGenerator["StrassenDataGenerator"]
    class experiments_extended_experiments_validate2_py_StrassenDataGenerator cls;
    experiments_extended_experiments_validate2_py --> experiments_extended_experiments_validate2_py_StrassenDataGenerator
    experiments_extended_experiments_validate2_py_LocalComplexityCalculator["LocalComplexityCalculator"]
    class experiments_extended_experiments_validate2_py_LocalComplexityCalculator cls;
    experiments_extended_experiments_validate2_py --> experiments_extended_experiments_validate2_py_LocalComplexityCalculator
    experiments_extended_experiments_validate2_py_GrokkingVerifier["GrokkingVerifier"]
    class experiments_extended_experiments_validate2_py_GrokkingVerifier cls;
    experiments_extended_experiments_validate2_py --> experiments_extended_experiments_validate2_py_GrokkingVerifier
    repor_experiments_py["repor_experiments.py (py)"]
    class repor_experiments_py mod;
    repor_experiments_py_ModelConfig["ModelConfig"]
    class repor_experiments_py_ModelConfig cls;
    repor_experiments_py --> repor_experiments_py_ModelConfig
    repor_experiments_py_TrainConfig["TrainConfig"]
    class repor_experiments_py_TrainConfig cls;
    repor_experiments_py --> repor_experiments_py_TrainConfig
    repor_experiments_py_Exp1Config["Exp1Config"]
    class repor_experiments_py_Exp1Config cls;
    repor_experiments_py --> repor_experiments_py_Exp1Config
    repor_experiments_py_Exp2Config["Exp2Config"]
    class repor_experiments_py_Exp2Config cls;
    repor_experiments_py --> repor_experiments_py_Exp2Config
    repor_experiments_py_Exp3Config["Exp3Config"]
    class repor_experiments_py_Exp3Config cls;
    repor_experiments_py --> repor_experiments_py_Exp3Config
    full_seed_prospector_py["full_seed_prospector.py (py)"]
    class full_seed_prospector_py mod;
    full_seed_prospector_py_ExecutionMode["ExecutionMode"]
    class full_seed_prospector_py_ExecutionMode cls;
    full_seed_prospector_py --> full_seed_prospector_py_ExecutionMode
    full_seed_prospector_py_UnifiedConfig["UnifiedConfig"]
    class full_seed_prospector_py_UnifiedConfig cls;
    full_seed_prospector_py --> full_seed_prospector_py_UnifiedConfig
    full_seed_prospector_py_IMetricCalculator["IMetricCalculator"]
    class full_seed_prospector_py_IMetricCalculator cls;
    full_seed_prospector_py --> full_seed_prospector_py_IMetricCalculator
    full_seed_prospector_py_ILossComponent["ILossComponent"]
    class full_seed_prospector_py_ILossComponent cls;
    full_seed_prospector_py --> full_seed_prospector_py_ILossComponent
    full_seed_prospector_py_ICheckpointManager["ICheckpointManager"]
    class full_seed_prospector_py_ICheckpointManager cls;
    full_seed_prospector_py --> full_seed_prospector_py_ICheckpointManager
    gravity_py["gravity.py (py)"]
    class gravity_py mod;
    gravity_py_ThermodynamicConfig["ThermodynamicConfig"]
    class gravity_py_ThermodynamicConfig cls;
    gravity_py --> gravity_py_ThermodynamicConfig
    gravity_py_IModel["IModel"]
    class gravity_py_IModel cls;
    gravity_py --> gravity_py_IModel
    gravity_py_IOrderParameterCalculator["IOrderParameterCalculator"]
    class gravity_py_IOrderParameterCalculator cls;
    gravity_py --> gravity_py_IOrderParameterCalculator
    gravity_py_IEntropyCalculator["IEntropyCalculator"]
    class gravity_py_IEntropyCalculator cls;
    gravity_py --> gravity_py_IEntropyCalculator
    gravity_py_ISpecificHeatCalculator["ISpecificHeatCalculator"]
    class gravity_py_ISpecificHeatCalculator cls;
    gravity_py --> gravity_py_ISpecificHeatCalculator
    grain_py["grain.py (py)"]
    class grain_py mod;
    grain_py_StrassenConfig["StrassenConfig"]
    class grain_py_StrassenConfig cls;
    grain_py --> grain_py_StrassenConfig
    grain_py_IModel["IModel"]
    class grain_py_IModel cls;
    grain_py --> grain_py_IModel
    grain_py_IGrainBoundaryDetector["IGrainBoundaryDetector"]
    class grain_py_IGrainBoundaryDetector cls;
    grain_py --> grain_py_IGrainBoundaryDetector
    grain_py_ILayerAnalyzer["ILayerAnalyzer"]
    class grain_py_ILayerAnalyzer cls;
    grain_py --> grain_py_ILayerAnalyzer
    grain_py_IDislocationCalculator["IDislocationCalculator"]
    class grain_py_IDislocationCalculator cls;
    grain_py --> grain_py_IDislocationCalculator
    dirac_polos_zeros_py["dirac_polos_zeros.py (py)"]
    class dirac_polos_zeros_py mod;
    dirac_polos_zeros_py_AnalysisConfig["AnalysisConfig"]
    class dirac_polos_zeros_py_AnalysisConfig cls;
    dirac_polos_zeros_py --> dirac_polos_zeros_py_AnalysisConfig
    dirac_polos_zeros_py_IModel["IModel"]
    class dirac_polos_zeros_py_IModel cls;
    dirac_polos_zeros_py --> dirac_polos_zeros_py_IModel
    dirac_polos_zeros_py_IChargeDistributionExtractor["IChargeDistributionExtractor"]
    class dirac_polos_zeros_py_IChargeDistributionExtractor cls;
    dirac_polos_zeros_py --> dirac_polos_zeros_py_IChargeDistributionExtractor
    dirac_polos_zeros_py_IDiracAnalyzer["IDiracAnalyzer"]
    class dirac_polos_zeros_py_IDiracAnalyzer cls;
    dirac_polos_zeros_py --> dirac_polos_zeros_py_IDiracAnalyzer
    dirac_polos_zeros_py_IFieldCalculator["IFieldCalculator"]
    class dirac_polos_zeros_py_IFieldCalculator cls;
    dirac_polos_zeros_py --> dirac_polos_zeros_py_IFieldCalculator
    unified_hidden_connections_suite_py["unified_hidden_connections_suite.py (py)"]
    class unified_hidden_connections_suite_py mod;
    unified_hidden_connections_suite_py_StrassStrassenConfig["StrassStrassenConfig"]
    class unified_hidden_connections_suite_py_StrassStrassenConfig cls;
    unified_hidden_connections_suite_py --> unified_hidden_connections_suite_py_StrassStrassenConfig
    unified_hidden_connections_suite_py_TrainingConfig["TrainingConfig"]
    class unified_hidden_connections_suite_py_TrainingConfig cls;
    unified_hidden_connections_suite_py --> unified_hidden_connections_suite_py_TrainingConfig
    unified_hidden_connections_suite_py_Experiment1Config["Experiment1Config"]
    class unified_hidden_connections_suite_py_Experiment1Config cls;
    unified_hidden_connections_suite_py --> unified_hidden_connections_suite_py_Experiment1Config
    unified_hidden_connections_suite_py_Experiment2Config["Experiment2Config"]
    class unified_hidden_connections_suite_py_Experiment2Config cls;
    unified_hidden_connections_suite_py --> unified_hidden_connections_suite_py_Experiment2Config
    unified_hidden_connections_suite_py_Experiment3Config["Experiment3Config"]
    class unified_hidden_connections_suite_py_Experiment3Config cls;
    unified_hidden_connections_suite_py --> unified_hidden_connections_suite_py_Experiment3Config
    experimetn2_py["experimetn2.py (py)"]
    class experimetn2_py mod;
    experimetn2_py_StrassStrassenConfig["StrassStrassenConfig"]
    class experimetn2_py_StrassStrassenConfig cls;
    experimetn2_py --> experimetn2_py_StrassStrassenConfig
    experimetn2_py_TrainingConfig["TrainingConfig"]
    class experimetn2_py_TrainingConfig cls;
    experimetn2_py --> experimetn2_py_TrainingConfig
    experimetn2_py_SuiteConfig["SuiteConfig"]
    class experimetn2_py_SuiteConfig cls;
    experimetn2_py --> experimetn2_py_SuiteConfig
    experimetn2_py_StrassStrassenModel["StrassStrassenModel"]
    class experimetn2_py_StrassStrassenModel cls;
    experimetn2_py --> experimetn2_py_StrassStrassenModel
    experimetn2_py_ComplexStrassStrassenModel["ComplexStrassStrassenModel"]
    class experimetn2_py_ComplexStrassStrassenModel cls;
    experimetn2_py --> experimetn2_py_ComplexStrassStrassenModel
    superposition_py["superposition.py (py)"]
    class superposition_py mod;
    superposition_py_Config["Config"]
    class superposition_py_Config cls;
    superposition_py --> superposition_py_Config
    superposition_py_ICheckpointLoader["ICheckpointLoader"]
    class superposition_py_ICheckpointLoader cls;
    superposition_py --> superposition_py_ICheckpointLoader
    superposition_py_IMetricsCalculator["IMetricsCalculator"]
    class superposition_py_IMetricsCalculator cls;
    superposition_py --> superposition_py_IMetricsCalculator
    superposition_py_IAnalyzer["IAnalyzer"]
    class superposition_py_IAnalyzer cls;
    superposition_py --> superposition_py_IAnalyzer
    superposition_py_CheckpointLoadingError["CheckpointLoadingError"]
    class superposition_py_CheckpointLoadingError cls;
    superposition_py --> superposition_py_CheckpointLoadingError
    mbl_analyzer_py["mbl_analyzer.py (py)"]
    class mbl_analyzer_py mod;
    mbl_analyzer_py_MBLConfiguration["MBLConfiguration"]
    class mbl_analyzer_py_MBLConfiguration cls;
    mbl_analyzer_py --> mbl_analyzer_py_MBLConfiguration
    mbl_analyzer_py_IModel["IModel"]
    class mbl_analyzer_py_IModel cls;
    mbl_analyzer_py --> mbl_analyzer_py_IModel
    mbl_analyzer_py_ILevelSpacingCalculator["ILevelSpacingCalculator"]
    class mbl_analyzer_py_ILevelSpacingCalculator cls;
    mbl_analyzer_py --> mbl_analyzer_py_ILevelSpacingCalculator
    mbl_analyzer_py_IParticipationRatioCalculator["IParticipationRatioCalculator"]
    class mbl_analyzer_py_IParticipationRatioCalculator cls;
    mbl_analyzer_py --> mbl_analyzer_py_IParticipationRatioCalculator
    mbl_analyzer_py_ISyntheticPlanckCalculator["ISyntheticPlanckCalculator"]
    class mbl_analyzer_py_ISyntheticPlanckCalculator cls;
    mbl_analyzer_py --> mbl_analyzer_py_ISyntheticPlanckCalculator
    maxwell_strassen_analysis_py["maxwell_strassen_analysis.py (py)"]
    class maxwell_strassen_analysis_py mod;
    maxwell_strassen_analysis_py_MaxwellConfiguration["MaxwellConfiguration"]
    class maxwell_strassen_analysis_py_MaxwellConfiguration cls;
    maxwell_strassen_analysis_py --> maxwell_strassen_analysis_py_MaxwellConfiguration
    maxwell_strassen_analysis_py_IModel["IModel"]
    class maxwell_strassen_analysis_py_IModel cls;
    maxwell_strassen_analysis_py --> maxwell_strassen_analysis_py_IModel
    maxwell_strassen_analysis_py_IGeometryMapper["IGeometryMapper"]
    class maxwell_strassen_analysis_py_IGeometryMapper cls;
    maxwell_strassen_analysis_py --> maxwell_strassen_analysis_py_IGeometryMapper
    maxwell_strassen_analysis_py_IMaxwellSolver["IMaxwellSolver"]
    class maxwell_strassen_analysis_py_IMaxwellSolver cls;
    maxwell_strassen_analysis_py --> maxwell_strassen_analysis_py_IMaxwellSolver
    maxwell_strassen_analysis_py_IDielectricAnalyzer["IDielectricAnalyzer"]
    class maxwell_strassen_analysis_py_IDielectricAnalyzer cls;
    maxwell_strassen_analysis_py --> maxwell_strassen_analysis_py_IDielectricAnalyzer
    boltzmann_experiments_py["boltzmann_experiments.py (py)"]
    class boltzmann_experiments_py mod;
    boltzmann_experiments_py_Config["Config"]
    class boltzmann_experiments_py_Config cls;
    boltzmann_experiments_py --> boltzmann_experiments_py_Config
    boltzmann_experiments_py_set_seed["set_seed"]
    class boltzmann_experiments_py_set_seed fn;
    boltzmann_experiments_py --> boltzmann_experiments_py_set_seed
    boltzmann_experiments_py_CheckpointLoadingError["CheckpointLoadingError"]
    class boltzmann_experiments_py_CheckpointLoadingError cls;
    boltzmann_experiments_py --> boltzmann_experiments_py_CheckpointLoadingError
    boltzmann_experiments_py_ICheckpointLoader["ICheckpointLoader"]
    class boltzmann_experiments_py_ICheckpointLoader cls;
    boltzmann_experiments_py --> boltzmann_experiments_py_ICheckpointLoader
    boltzmann_experiments_py_CheckpointLoader["CheckpointLoader"]
    class boltzmann_experiments_py_CheckpointLoader cls;
    boltzmann_experiments_py --> boltzmann_experiments_py_CheckpointLoader
    grigori_perelmans_ricci_flow_py["grigori_perelmans_ricci_flow.py (py)"]
    class grigori_perelmans_ricci_flow_py mod;
    grigori_perelmans_ricci_flow_py_RicciConfig["RicciConfig"]
    class grigori_perelmans_ricci_flow_py_RicciConfig cls;
    grigori_perelmans_ricci_flow_py --> grigori_perelmans_ricci_flow_py_RicciConfig
    grigori_perelmans_ricci_flow_py_set_random_seed["set_random_seed"]
    class grigori_perelmans_ricci_flow_py_set_random_seed fn;
    grigori_perelmans_ricci_flow_py --> grigori_perelmans_ricci_flow_py_set_random_seed
    grigori_perelmans_ricci_flow_py_BilinearStrassenModel["BilinearStrassenModel"]
    class grigori_perelmans_ricci_flow_py_BilinearStrassenModel cls;
    grigori_perelmans_ricci_flow_py --> grigori_perelmans_ricci_flow_py_BilinearStrassenModel
    grigori_perelmans_ricci_flow_py_StrassenDataGenerator["StrassenDataGenerator"]
    class grigori_perelmans_ricci_flow_py_StrassenDataGenerator cls;
    grigori_perelmans_ricci_flow_py --> grigori_perelmans_ricci_flow_py_StrassenDataGenerator
    grigori_perelmans_ricci_flow_py_CheckpointMigrator["CheckpointMigrator"]
    class grigori_perelmans_ricci_flow_py_CheckpointMigrator cls;
    grigori_perelmans_ricci_flow_py --> grigori_perelmans_ricci_flow_py_CheckpointMigrator
    hawking_radiation_py["hawking_radiation.py (py)"]
    class hawking_radiation_py mod;
    hawking_radiation_py_CustomUnpickler["CustomUnpickler"]
    class hawking_radiation_py_CustomUnpickler cls;
    hawking_radiation_py --> hawking_radiation_py_CustomUnpickler
    hawking_radiation_py_load_checkpoint_robust["load_checkpoint_robust"]
    class hawking_radiation_py_load_checkpoint_robust fn;
    hawking_radiation_py --> hawking_radiation_py_load_checkpoint_robust
    hawking_radiation_py_HawkingConfiguration["HawkingConfiguration"]
    class hawking_radiation_py_HawkingConfiguration cls;
    hawking_radiation_py --> hawking_radiation_py_HawkingConfiguration
    hawking_radiation_py_IModel["IModel"]
    class hawking_radiation_py_IModel cls;
    hawking_radiation_py --> hawking_radiation_py_IModel
    hawking_radiation_py_BilinearStrassenModel["BilinearStrassenModel"]
    class hawking_radiation_py_BilinearStrassenModel cls;
    hawking_radiation_py --> hawking_radiation_py_BilinearStrassenModel
    experiments_generate_figures_py["generate_figures.py (py)"]
    class experiments_generate_figures_py mod;
    experiments_generate_figures_py_setup_matplotlib_for_plotting["setup_matplotlib_for_plotting"]
    class experiments_generate_figures_py_setup_matplotlib_for_plotting fn;
    experiments_generate_figures_py --> experiments_generate_figures_py_setup_matplotlib_for_plotting
    experiments_generate_figures_py_generate_benchmark_figure["generate_benchmark_figure"]
    class experiments_generate_figures_py_generate_benchmark_figure fn;
    experiments_generate_figures_py --> experiments_generate_figures_py_generate_benchmark_figure
    experiments_generate_figures_py_generate_ablation_figure["generate_ablation_figure"]
    class experiments_generate_figures_py_generate_ablation_figure fn;
    experiments_generate_figures_py --> experiments_generate_figures_py_generate_ablation_figure
    experiments_generate_figures_py_load_checkpoint_weights["load_checkpoint_weights"]
    class experiments_generate_figures_py_load_checkpoint_weights fn;
    experiments_generate_figures_py --> experiments_generate_figures_py_load_checkpoint_weights
    experiments_generate_figures_py_generate_weight_geometry_figure["generate_weight_geometry_figure"]
    class experiments_generate_figures_py_generate_weight_geometry_figure fn;
    experiments_generate_figures_py --> experiments_generate_figures_py_generate_weight_geometry_figure
    plank_py["plank.py (py)"]
    class plank_py mod;
    plank_py_Configuration["Configuration"]
    class plank_py_Configuration cls;
    plank_py --> plank_py_Configuration
    plank_py_set_random_seed["set_random_seed"]
    class plank_py_set_random_seed fn;
    plank_py --> plank_py_set_random_seed
    plank_py_BilinearStrassenModel["BilinearStrassenModel"]
    class plank_py_BilinearStrassenModel cls;
    plank_py --> plank_py_BilinearStrassenModel
    plank_py_CheckpointMigrator["CheckpointMigrator"]
    class plank_py_CheckpointMigrator cls;
    plank_py --> plank_py_CheckpointMigrator
    plank_py_CustomFormatMigrator["CustomFormatMigrator"]
    class plank_py_CustomFormatMigrator cls;
    plank_py --> plank_py_CustomFormatMigrator
    scrodingger_py["scrodingger.py (py)"]
    class scrodingger_py mod;
    scrodingger_py_SchrodingerConfig["SchrodingerConfig"]
    class scrodingger_py_SchrodingerConfig cls;
    scrodingger_py --> scrodingger_py_SchrodingerConfig
    scrodingger_py_IModel["IModel"]
    class scrodingger_py_IModel cls;
    scrodingger_py --> scrodingger_py_IModel
    scrodingger_py_IWaveFunctionExtractor["IWaveFunctionExtractor"]
    class scrodingger_py_IWaveFunctionExtractor cls;
    scrodingger_py --> scrodingger_py_IWaveFunctionExtractor
    scrodingger_py_IPotentialCalculator["IPotentialCalculator"]
    class scrodingger_py_IPotentialCalculator cls;
    scrodingger_py --> scrodingger_py_IPotentialCalculator
    scrodingger_py_IHamiltonianConstructor["IHamiltonianConstructor"]
    class scrodingger_py_IHamiltonianConstructor cls;
    scrodingger_py --> scrodingger_py_IHamiltonianConstructor
    fermi_py["fermi.py (py)"]
    class fermi_py mod;
    fermi_py_FermiConfig["FermiConfig"]
    class fermi_py_FermiConfig cls;
    fermi_py --> fermi_py_FermiConfig
    fermi_py_IModel["IModel"]
    class fermi_py_IModel cls;
    fermi_py --> fermi_py_IModel
    fermi_py_IBlochWaveConstructor["IBlochWaveConstructor"]
    class fermi_py_IBlochWaveConstructor cls;
    fermi_py --> fermi_py_IBlochWaveConstructor
    fermi_py_IBandStructureCalculator["IBandStructureCalculator"]
    class fermi_py_IBandStructureCalculator cls;
    fermi_py --> fermi_py_IBandStructureCalculator
    fermi_py_IFermiLevelCalculator["IFermiLevelCalculator"]
    class fermi_py_IFermiLevelCalculator cls;
    fermi_py --> fermi_py_IFermiLevelCalculator
    purity_index_py["purity_index.py (py)"]
    class purity_index_py mod;
    purity_index_py_PurityConfig["PurityConfig"]
    class purity_index_py_PurityConfig cls;
    purity_index_py --> purity_index_py_PurityConfig
    purity_index_py_IModel["IModel"]
    class purity_index_py_IModel cls;
    purity_index_py --> purity_index_py_IModel
    purity_index_py_IPurityIndexCalculator["IPurityIndexCalculator"]
    class purity_index_py_IPurityIndexCalculator cls;
    purity_index_py --> purity_index_py_IPurityIndexCalculator
    purity_index_py_IEffectiveTemperatureCalculator["IEffectiveTemperatureCalculator"]
    class purity_index_py_IEffectiveTemperatureCalculator cls;
    purity_index_py --> purity_index_py_IEffectiveTemperatureCalculator
    purity_index_py_IPhaseClassifier["IPhaseClassifier"]
    class purity_index_py_IPhaseClassifier cls;
    purity_index_py --> purity_index_py_IPhaseClassifier
    batch_size_py["batch_size.py (py)"]
    class batch_size_py mod;
    batch_size_py_Configuration["Configuration"]
    class batch_size_py_Configuration cls;
    batch_size_py --> batch_size_py_Configuration
    batch_size_py_set_random_seed["set_random_seed"]
    class batch_size_py_set_random_seed fn;
    batch_size_py --> batch_size_py_set_random_seed
    batch_size_py_BilinearStrassenModel["BilinearStrassenModel"]
    class batch_size_py_BilinearStrassenModel cls;
    batch_size_py --> batch_size_py_BilinearStrassenModel
    batch_size_py_CheckpointMigrator["CheckpointMigrator"]
    class batch_size_py_CheckpointMigrator cls;
    batch_size_py --> batch_size_py_CheckpointMigrator
    batch_size_py_CustomFormatMigrator["CustomFormatMigrator"]
    class batch_size_py_CustomFormatMigrator cls;
    batch_size_py --> batch_size_py_CustomFormatMigrator
    src_training_main_py["main.py (py)"]
    class src_training_main_py mod;
    src_training_main_py_Config["Config"]
    class src_training_main_py_Config cls;
    src_training_main_py --> src_training_main_py_Config
    src_training_main_py_set_seed["set_seed"]
    class src_training_main_py_set_seed fn;
    src_training_main_py --> src_training_main_py_set_seed
    src_training_main_py_StrassenDiscovery["StrassenDiscovery"]
    class src_training_main_py_StrassenDiscovery cls;
    src_training_main_py --> src_training_main_py_StrassenDiscovery
    src_training_main_py_Matrix4x4Dataset["Matrix4x4Dataset"]
    class src_training_main_py_Matrix4x4Dataset cls;
    src_training_main_py --> src_training_main_py_Matrix4x4Dataset
    src_training_main_py_Trainer["Trainer"]
    class src_training_main_py_Trainer cls;
    src_training_main_py --> src_training_main_py_Trainer
    experiments_extended_experiments_run_all_experiments_py["run_all_experiments.py (py)"]
    class experiments_extended_experiments_run_all_experiments_py mod;
    experiments_extended_experiments_run_all_experiments_py_setup_matplotlib["setup_matplotlib"]
    class experiments_extended_experiments_run_all_experiments_py_setup_matplotlib fn;
    experiments_extended_experiments_run_all_experiments_py --> experiments_extended_experiments_run_all_experiments_py_setup_matplotlib
    experiments_extended_experiments_run_all_experiments_py_StrassenOperator["StrassenOperator"]
    class experiments_extended_experiments_run_all_experiments_py_StrassenOperator cls;
    experiments_extended_experiments_run_all_experiments_py --> experiments_extended_experiments_run_all_experiments_py_StrassenOperator
    experiments_extended_experiments_run_all_experiments_py_generate_batch["generate_batch"]
    class experiments_extended_experiments_run_all_experiments_py_generate_batch fn;
    experiments_extended_experiments_run_all_experiments_py --> experiments_extended_experiments_run_all_experiments_py_generate_batch
    experiments_extended_experiments_run_all_experiments_py_load_checkpoint_robust["load_checkpoint_robust"]
    class experiments_extended_experiments_run_all_experiments_py_load_checkpoint_robust fn;
    experiments_extended_experiments_run_all_experiments_py --> experiments_extended_experiments_run_all_experiments_py_load_checkpoint_robust
    experiments_extended_experiments_run_all_experiments_py_compute_gradient_covariance_safe["compute_gradient_covariance_safe"]
    class experiments_extended_experiments_run_all_experiments_py_compute_gradient_covariance_safe fn;
    experiments_extended_experiments_run_all_experiments_py --> experiments_extended_experiments_run_all_experiments_py_compute_gradient_covariance_safe
    experiments_apendix_experiments_py["apendix_experiments.py (py)"]
    class experiments_apendix_experiments_py mod;
    experiments_apendix_experiments_py_setup_matplotlib["setup_matplotlib"]
    class experiments_apendix_experiments_py_setup_matplotlib fn;
    experiments_apendix_experiments_py --> experiments_apendix_experiments_py_setup_matplotlib
    experiments_apendix_experiments_py_StrassenOperator["StrassenOperator"]
    class experiments_apendix_experiments_py_StrassenOperator cls;
    experiments_apendix_experiments_py --> experiments_apendix_experiments_py_StrassenOperator
    experiments_apendix_experiments_py_generate_batch["generate_batch"]
    class experiments_apendix_experiments_py_generate_batch fn;
    experiments_apendix_experiments_py --> experiments_apendix_experiments_py_generate_batch
    experiments_apendix_experiments_py_generate_test_set["generate_test_set"]
    class experiments_apendix_experiments_py_generate_test_set fn;
    experiments_apendix_experiments_py --> experiments_apendix_experiments_py_generate_test_set
    experiments_apendix_experiments_py_compute_delta["compute_delta"]
    class experiments_apendix_experiments_py_compute_delta fn;
    experiments_apendix_experiments_py --> experiments_apendix_experiments_py_compute_delta
    experiments_extended_experiments_exp3_prospective_prediction_py["exp3_prospective_prediction.py (py)"]
    class experiments_extended_experiments_exp3_prospective_prediction_py mod;
    experiments_extended_experiments_exp3_prospective_prediction_py_setup_matplotlib["setup_matplotlib"]
    class experiments_extended_experiments_exp3_prospective_prediction_py_setup_matplotlib fn;
    experiments_extended_experiments_exp3_prospective_prediction_py --> experiments_extended_experiments_exp3_prospective_prediction_py_setup_matplotlib
    experiments_extended_experiments_exp3_prospective_prediction_py_StrassenOperator["StrassenOperator"]
    class experiments_extended_experiments_exp3_prospective_prediction_py_StrassenOperator cls;
    experiments_extended_experiments_exp3_prospective_prediction_py --> experiments_extended_experiments_exp3_prospective_prediction_py_StrassenOperator
    experiments_extended_experiments_exp3_prospective_prediction_py_generate_batch["generate_batch"]
    class experiments_extended_experiments_exp3_prospective_prediction_py_generate_batch fn;
    experiments_extended_experiments_exp3_prospective_prediction_py --> experiments_extended_experiments_exp3_prospective_prediction_py_generate_batch
    experiments_extended_experiments_exp3_prospective_prediction_py_compute_kappa["compute_kappa"]
    class experiments_extended_experiments_exp3_prospective_prediction_py_compute_kappa fn;
    experiments_extended_experiments_exp3_prospective_prediction_py --> experiments_extended_experiments_exp3_prospective_prediction_py_compute_kappa
    experiments_extended_experiments_exp3_prospective_prediction_py_load_checkpoint["load_checkpoint"]
    class experiments_extended_experiments_exp3_prospective_prediction_py_load_checkpoint fn;
    experiments_extended_experiments_exp3_prospective_prediction_py --> experiments_extended_experiments_exp3_prospective_prediction_py_load_checkpoint
    experiments_extended_experiments_exp1_covariance_spectrometry_py["exp1_covariance_spectrometry.py (py)"]
    class experiments_extended_experiments_exp1_covariance_spectrometry_py mod;
    experiments_extended_experiments_exp1_covariance_spectrometry_py_setup_matplotlib["setup_matplotlib"]
    class experiments_extended_experiments_exp1_covariance_spectrometry_py_setup_matplotlib fn;
    experiments_extended_experiments_exp1_covariance_spectrometry_py --> experiments_extended_experiments_exp1_covariance_spectrometry_py_setup_matplotlib
    experiments_extended_experiments_exp1_covariance_spectrometry_py_StrassenOperator["StrassenOperator"]
    class experiments_extended_experiments_exp1_covariance_spectrometry_py_StrassenOperator cls;
    experiments_extended_experiments_exp1_covariance_spectrometry_py --> experiments_extended_experiments_exp1_covariance_spectrometry_py_StrassenOperator
    experiments_extended_experiments_exp1_covariance_spectrometry_py_generate_batch["generate_batch"]
    class experiments_extended_experiments_exp1_covariance_spectrometry_py_generate_batch fn;
    experiments_extended_experiments_exp1_covariance_spectrometry_py --> experiments_extended_experiments_exp1_covariance_spectrometry_py_generate_batch
    experiments_extended_experiments_exp1_covariance_spectrometry_py_compute_gradient_covariance["compute_gradient_covariance"]
    class experiments_extended_experiments_exp1_covariance_spectrometry_py_compute_gradient_covariance fn;
    experiments_extended_experiments_exp1_covariance_spectrometry_py --> experiments_extended_experiments_exp1_covariance_spectrometry_py_compute_gradient_covariance
    experiments_extended_experiments_exp1_covariance_spectrometry_py_load_checkpoint["load_checkpoint"]
    class experiments_extended_experiments_exp1_covariance_spectrometry_py_load_checkpoint fn;
    experiments_extended_experiments_exp1_covariance_spectrometry_py --> experiments_extended_experiments_exp1_covariance_spectrometry_py_load_checkpoint
    src_benchmarks_benchmark_strassen_py["benchmark_strassen.py (py)"]
    class src_benchmarks_benchmark_strassen_py mod;
    src_benchmarks_benchmark_strassen_py_BenchmarkResult["BenchmarkResult"]
    class src_benchmarks_benchmark_strassen_py_BenchmarkResult cls;
    src_benchmarks_benchmark_strassen_py --> src_benchmarks_benchmark_strassen_py_BenchmarkResult
    src_benchmarks_benchmark_strassen_py_BenchmarkConfig["BenchmarkConfig"]
    class src_benchmarks_benchmark_strassen_py_BenchmarkConfig cls;
    src_benchmarks_benchmark_strassen_py --> src_benchmarks_benchmark_strassen_py_BenchmarkConfig
    src_benchmarks_benchmark_strassen_py_load_config["load_config"]
    class src_benchmarks_benchmark_strassen_py_load_config fn;
    src_benchmarks_benchmark_strassen_py --> src_benchmarks_benchmark_strassen_py_load_config
    src_benchmarks_benchmark_strassen_py_get_dtype["get_dtype"]
    class src_benchmarks_benchmark_strassen_py_get_dtype fn;
    src_benchmarks_benchmark_strassen_py --> src_benchmarks_benchmark_strassen_py_get_dtype
    src_benchmarks_benchmark_strassen_py_estimate_memory_mb["estimate_memory_mb"]
    class src_benchmarks_benchmark_strassen_py_estimate_memory_mb fn;
    src_benchmarks_benchmark_strassen_py --> src_benchmarks_benchmark_strassen_py_estimate_memory_mb
    crystallography_py["crystallography.py (py)"]
    class crystallography_py mod;
    crystallography_py_Config["Config"]
    class crystallography_py_Config cls;
    crystallography_py --> crystallography_py_Config
    crystallography_py_set_seed["set_seed"]
    class crystallography_py_set_seed fn;
    crystallography_py --> crystallography_py_set_seed
    crystallography_py_BilinearStrassenModel["BilinearStrassenModel"]
    class crystallography_py_BilinearStrassenModel cls;
    crystallography_py --> crystallography_py_BilinearStrassenModel
    crystallography_py_CheckpointMigrator["CheckpointMigrator"]
    class crystallography_py_CheckpointMigrator cls;
    crystallography_py --> crystallography_py_CheckpointMigrator
    crystallography_py_StrassenDataGenerator["StrassenDataGenerator"]
    class crystallography_py_StrassenDataGenerator cls;
    crystallography_py --> crystallography_py_StrassenDataGenerator
    experiments_statistics_rigorous_experiment_py["rigorous_experiment.py (py)"]
    class experiments_statistics_rigorous_experiment_py mod;
    experiments_statistics_rigorous_experiment_py_ExperimentConfig["ExperimentConfig"]
    class experiments_statistics_rigorous_experiment_py_ExperimentConfig cls;
    experiments_statistics_rigorous_experiment_py --> experiments_statistics_rigorous_experiment_py_ExperimentConfig
    experiments_statistics_rigorous_experiment_py_ExperimentResult["ExperimentResult"]
    class experiments_statistics_rigorous_experiment_py_ExperimentResult cls;
    experiments_statistics_rigorous_experiment_py --> experiments_statistics_rigorous_experiment_py_ExperimentResult
    experiments_statistics_rigorous_experiment_py_StrassenModel["StrassenModel"]
    class experiments_statistics_rigorous_experiment_py_StrassenModel cls;
    experiments_statistics_rigorous_experiment_py --> experiments_statistics_rigorous_experiment_py_StrassenModel
    experiments_statistics_rigorous_experiment_py_generate_data["generate_data"]
    class experiments_statistics_rigorous_experiment_py_generate_data fn;
    experiments_statistics_rigorous_experiment_py --> experiments_statistics_rigorous_experiment_py_generate_data
    experiments_statistics_rigorous_experiment_py_compute_discretization_error["compute_discretization_error"]
    class experiments_statistics_rigorous_experiment_py_compute_discretization_error fn;
    experiments_statistics_rigorous_experiment_py --> experiments_statistics_rigorous_experiment_py_compute_discretization_error
    experiments_extended_experiments_exp2_noise_ablation_py["exp2_noise_ablation.py (py)"]
    class experiments_extended_experiments_exp2_noise_ablation_py mod;
    experiments_extended_experiments_exp2_noise_ablation_py_setup_matplotlib["setup_matplotlib"]
    class experiments_extended_experiments_exp2_noise_ablation_py_setup_matplotlib fn;
    experiments_extended_experiments_exp2_noise_ablation_py --> experiments_extended_experiments_exp2_noise_ablation_py_setup_matplotlib
    experiments_extended_experiments_exp2_noise_ablation_py_StrassenOperator["StrassenOperator"]
    class experiments_extended_experiments_exp2_noise_ablation_py_StrassenOperator cls;
    experiments_extended_experiments_exp2_noise_ablation_py --> experiments_extended_experiments_exp2_noise_ablation_py_StrassenOperator
    experiments_extended_experiments_exp2_noise_ablation_py_generate_batch["generate_batch"]
    class experiments_extended_experiments_exp2_noise_ablation_py_generate_batch fn;
    experiments_extended_experiments_exp2_noise_ablation_py --> experiments_extended_experiments_exp2_noise_ablation_py_generate_batch
    experiments_extended_experiments_exp2_noise_ablation_py_compute_gradient_covariance_matrix["compute_gradient_covariance_matrix"]
    class experiments_extended_experiments_exp2_noise_ablation_py_compute_gradient_covariance_matrix fn;
    experiments_extended_experiments_exp2_noise_ablation_py --> experiments_extended_experiments_exp2_noise_ablation_py_compute_gradient_covariance_matrix
    experiments_extended_experiments_exp2_noise_ablation_py_get_eigenbasis["get_eigenbasis"]
    class experiments_extended_experiments_exp2_noise_ablation_py_get_eigenbasis fn;
    experiments_extended_experiments_exp2_noise_ablation_py --> experiments_extended_experiments_exp2_noise_ablation_py_get_eigenbasis
    experiments_extended_experiments_exp4_trajectory_perturbation_py["exp4_trajectory_perturbation.py (py)"]
    class experiments_extended_experiments_exp4_trajectory_perturbation_py mod;
    experiments_extended_experiments_exp4_trajectory_perturbation_py_setup_matplotlib["setup_matplotlib"]
    class experiments_extended_experiments_exp4_trajectory_perturbation_py_setup_matplotlib fn;
    experiments_extended_experiments_exp4_trajectory_perturbation_py --> experiments_extended_experiments_exp4_trajectory_perturbation_py_setup_matplotlib
    experiments_extended_experiments_exp4_trajectory_perturbation_py_StrassenOperator["StrassenOperator"]
    class experiments_extended_experiments_exp4_trajectory_perturbation_py_StrassenOperator cls;
    experiments_extended_experiments_exp4_trajectory_perturbation_py --> experiments_extended_experiments_exp4_trajectory_perturbation_py_StrassenOperator
    experiments_extended_experiments_exp4_trajectory_perturbation_py_generate_batch["generate_batch"]
    class experiments_extended_experiments_exp4_trajectory_perturbation_py_generate_batch fn;
    experiments_extended_experiments_exp4_trajectory_perturbation_py --> experiments_extended_experiments_exp4_trajectory_perturbation_py_generate_batch
    experiments_extended_experiments_exp4_trajectory_perturbation_py_load_checkpoint["load_checkpoint"]
    class experiments_extended_experiments_exp4_trajectory_perturbation_py_load_checkpoint fn;
    experiments_extended_experiments_exp4_trajectory_perturbation_py --> experiments_extended_experiments_exp4_trajectory_perturbation_py_load_checkpoint
    experiments_extended_experiments_exp4_trajectory_perturbation_py_simulate_trajectory_perturbation["simulate_trajectory_perturbation"]
    class experiments_extended_experiments_exp4_trajectory_perturbation_py_simulate_trajectory_perturbation fn;
    experiments_extended_experiments_exp4_trajectory_perturbation_py --> experiments_extended_experiments_exp4_trajectory_perturbation_py_simulate_trajectory_perturbation
    experiments_ablation_ablation_study_py["ablation_study.py (py)"]
    class experiments_ablation_ablation_study_py mod;
    experiments_ablation_ablation_study_py_BenchmarkResult["BenchmarkResult"]
    class experiments_ablation_ablation_study_py_BenchmarkResult cls;
    experiments_ablation_ablation_study_py --> experiments_ablation_ablation_study_py_BenchmarkResult
    experiments_ablation_ablation_study_py_load_libraries["load_libraries"]
    class experiments_ablation_ablation_study_py_load_libraries fn;
    experiments_ablation_ablation_study_py --> experiments_ablation_ablation_study_py_load_libraries
    experiments_ablation_ablation_study_py_run_openblas["run_openblas"]
    class experiments_ablation_ablation_study_py_run_openblas fn;
    experiments_ablation_ablation_study_py --> experiments_ablation_ablation_study_py_run_openblas
    experiments_ablation_ablation_study_py_run_strassen["run_strassen"]
    class experiments_ablation_ablation_study_py_run_strassen fn;
    experiments_ablation_ablation_study_py --> experiments_ablation_ablation_study_py_run_strassen
    experiments_ablation_ablation_study_py_benchmark_single["benchmark_single"]
    class experiments_ablation_ablation_study_py_benchmark_single fn;
    experiments_ablation_ablation_study_py --> experiments_ablation_ablation_study_py_benchmark_single
    src_benchmarks_benchmark_scientific_py["benchmark_scientific.py (py)"]
    class src_benchmarks_benchmark_scientific_py mod;
    src_benchmarks_benchmark_scientific_py_strassen_multiply["strassen_multiply"]
    class src_benchmarks_benchmark_scientific_py_strassen_multiply fn;
    src_benchmarks_benchmark_scientific_py --> src_benchmarks_benchmark_scientific_py_strassen_multiply
    src_benchmarks_benchmark_scientific_py_standard_avx512_multiply["standard_avx512_multiply"]
    class src_benchmarks_benchmark_scientific_py_standard_avx512_multiply fn;
    src_benchmarks_benchmark_scientific_py --> src_benchmarks_benchmark_scientific_py_standard_avx512_multiply
    src_benchmarks_benchmark_scientific_py_numpy_multiply["numpy_multiply"]
    class src_benchmarks_benchmark_scientific_py_numpy_multiply fn;
    src_benchmarks_benchmark_scientific_py --> src_benchmarks_benchmark_scientific_py_numpy_multiply
    src_benchmarks_benchmark_scientific_py_benchmark_function["benchmark_function"]
    class src_benchmarks_benchmark_scientific_py_benchmark_function fn;
    src_benchmarks_benchmark_scientific_py --> src_benchmarks_benchmark_scientific_py_benchmark_function
    src_benchmarks_benchmark_scientific_py_main["main"]
    class src_benchmarks_benchmark_scientific_py_main fn;
    src_benchmarks_benchmark_scientific_py --> src_benchmarks_benchmark_scientific_py_main
    app_py["app.py (py)"]
    class app_py mod;
    app_py_StrassenNet["StrassenNet"]
    class app_py_StrassenNet cls;
    app_py --> app_py_StrassenNet
    app_py___init__["__init__"]
    class app_py___init__ fn;
    app_py --> app_py___init__
    app_py_forward["forward"]
    class app_py_forward fn;
    app_py --> app_py_forward
    src_discovery_auto_T_discovery_py["auto_T_discovery.py (py)"]
    class src_discovery_auto_T_discovery_py mod;
    src_discovery_auto_T_discovery_py_SymmetryStructure["SymmetryStructure"]
    class src_discovery_auto_T_discovery_py_SymmetryStructure cls;
    src_discovery_auto_T_discovery_py --> src_discovery_auto_T_discovery_py_SymmetryStructure
    src_discovery_auto_T_discovery_py_AutoTDiscovery["AutoTDiscovery"]
    class src_discovery_auto_T_discovery_py_AutoTDiscovery cls;
    src_discovery_auto_T_discovery_py --> src_discovery_auto_T_discovery_py_AutoTDiscovery
    src_discovery_auto_T_discovery_py_verify_strassen_T["verify_strassen_T"]
    class src_discovery_auto_T_discovery_py_verify_strassen_T fn;
    src_discovery_auto_T_discovery_py --> src_discovery_auto_T_discovery_py_verify_strassen_T
    src_discovery_auto_T_discovery_py_verify_expanded_correctness["verify_expanded_correctness"]
    class src_discovery_auto_T_discovery_py_verify_expanded_correctness fn;
    src_discovery_auto_T_discovery_py --> src_discovery_auto_T_discovery_py_verify_expanded_correctness
    src_discovery_auto_T_discovery_py_recursive_strassen_multiply["recursive_strassen_multiply"]
    class src_discovery_auto_T_discovery_py_recursive_strassen_multiply fn;
    src_discovery_auto_T_discovery_py --> src_discovery_auto_T_discovery_py_recursive_strassen_multiply
    src_training_convergence_theory_py["convergence_theory.py (py)"]
    class src_training_convergence_theory_py mod;
    src_training_convergence_theory_py_ConvergenceMetrics["ConvergenceMetrics"]
    class src_training_convergence_theory_py_ConvergenceMetrics cls;
    src_training_convergence_theory_py --> src_training_convergence_theory_py_ConvergenceMetrics
    src_training_convergence_theory_py_HutchinsonTraceEstimator["HutchinsonTraceEstimator"]
    class src_training_convergence_theory_py_HutchinsonTraceEstimator cls;
    src_training_convergence_theory_py --> src_training_convergence_theory_py_HutchinsonTraceEstimator
    src_training_convergence_theory_py_HardwareNoiseEstimator["HardwareNoiseEstimator"]
    class src_training_convergence_theory_py_HardwareNoiseEstimator cls;
    src_training_convergence_theory_py --> src_training_convergence_theory_py_HardwareNoiseEstimator
    src_training_convergence_theory_py_convergence_theorem["convergence_theorem"]
    class src_training_convergence_theory_py_convergence_theorem fn;
    src_training_convergence_theory_py --> src_training_convergence_theory_py_convergence_theorem
    src_training_convergence_theory_py_verify_convergence_conditions["verify_convergence_conditions"]
    class src_training_convergence_theory_py_verify_convergence_conditions fn;
    src_training_convergence_theory_py --> src_training_convergence_theory_py_verify_convergence_conditions
    experiments_validation_experiments_py["validation_experiments.py (py)"]
    class experiments_validation_experiments_py mod;
    experiments_validation_experiments_py_strassen_2x2["strassen_2x2"]
    class experiments_validation_experiments_py_strassen_2x2 fn;
    experiments_validation_experiments_py --> experiments_validation_experiments_py_strassen_2x2
    experiments_validation_experiments_py_strassen_recursive["strassen_recursive"]
    class experiments_validation_experiments_py_strassen_recursive fn;
    experiments_validation_experiments_py --> experiments_validation_experiments_py_strassen_recursive
    experiments_validation_experiments_py_test_uniqueness_via_permutation["test_uniqueness_via_permutation"]
    class experiments_validation_experiments_py_test_uniqueness_via_permutation fn;
    experiments_validation_experiments_py --> experiments_validation_experiments_py_test_uniqueness_via_permutation
    experiments_validation_experiments_py_test_noise_stability["test_noise_stability"]
    class experiments_validation_experiments_py_test_noise_stability fn;
    experiments_validation_experiments_py --> experiments_validation_experiments_py_test_noise_stability
    experiments_validation_experiments_py_test_expansion_sizes["test_expansion_sizes"]
    class experiments_validation_experiments_py_test_expansion_sizes fn;
    experiments_validation_experiments_py --> experiments_validation_experiments_py_test_expansion_sizes
    train_batch_sweep_py["train_batch_sweep.py (py)"]
    class train_batch_sweep_py mod;
    train_batch_sweep_py_train_for_batch_size["train_for_batch_size"]
    class train_batch_sweep_py_train_for_batch_size fn;
    train_batch_sweep_py --> train_batch_sweep_py_train_for_batch_size
    train_batch_sweep_py_LocalConfig["LocalConfig"]
    class train_batch_sweep_py_LocalConfig cls;
    train_batch_sweep_py --> train_batch_sweep_py_LocalConfig
    experiments_ablation_ablation_8192_py["ablation_8192.py (py)"]
    class experiments_ablation_ablation_8192_py mod;
    experiments_verify_checkpoints_py["verify_checkpoints.py (py)"]
    class experiments_verify_checkpoints_py mod;
    experiments_verify_checkpoints_py_StrassenBilinear["StrassenBilinear"]
    class experiments_verify_checkpoints_py_StrassenBilinear cls;
    experiments_verify_checkpoints_py --> experiments_verify_checkpoints_py_StrassenBilinear
    experiments_verify_checkpoints_py_compute_delta["compute_delta"]
    class experiments_verify_checkpoints_py_compute_delta fn;
    experiments_verify_checkpoints_py --> experiments_verify_checkpoints_py_compute_delta
    experiments_verify_checkpoints_py_verify_2x2["verify_2x2"]
    class experiments_verify_checkpoints_py_verify_2x2 fn;
    experiments_verify_checkpoints_py --> experiments_verify_checkpoints_py_verify_2x2
    experiments_verify_checkpoints_py_strassen_expand["strassen_expand"]
    class experiments_verify_checkpoints_py_strassen_expand fn;
    experiments_verify_checkpoints_py --> experiments_verify_checkpoints_py_strassen_expand
    experiments_verify_checkpoints_py_verify_expansion["verify_expansion"]
    class experiments_verify_checkpoints_py_verify_expansion fn;
    experiments_verify_checkpoints_py --> experiments_verify_checkpoints_py_verify_expansion
    src_native_strassen_turbo_c["strassen_turbo.c (c)"]
    class src_native_strassen_turbo_c mod;
    src_native_strassen_turbo_c_alloc_matrix["alloc_matrix"]
    class src_native_strassen_turbo_c_alloc_matrix fn;
    src_native_strassen_turbo_c --> src_native_strassen_turbo_c_alloc_matrix
    src_native_strassen_turbo_c_mat_add_avx["mat_add_avx"]
    class src_native_strassen_turbo_c_mat_add_avx fn;
    src_native_strassen_turbo_c --> src_native_strassen_turbo_c_mat_add_avx
    src_native_strassen_turbo_c_mat_sub_avx["mat_sub_avx"]
    class src_native_strassen_turbo_c_mat_sub_avx fn;
    src_native_strassen_turbo_c --> src_native_strassen_turbo_c_mat_sub_avx
    src_native_strassen_turbo_c_matmul_blocked_avx["matmul_blocked_avx"]
    class src_native_strassen_turbo_c_matmul_blocked_avx fn;
    src_native_strassen_turbo_c --> src_native_strassen_turbo_c_matmul_blocked_avx
    src_native_strassen_turbo_c_extract_quadrant["extract_quadrant"]
    class src_native_strassen_turbo_c_extract_quadrant fn;
    src_native_strassen_turbo_c --> src_native_strassen_turbo_c_extract_quadrant
    src_training_main_pure_math_py["main_pure_math.py (py)"]
    class src_training_main_pure_math_py mod;
    src_training_main_pure_math_py_StrassenModel["StrassenModel"]
    class src_training_main_pure_math_py_StrassenModel cls;
    src_training_main_pure_math_py --> src_training_main_pure_math_py_StrassenModel
    src_training_main_pure_math_py_gen_data["gen_data"]
    class src_training_main_pure_math_py_gen_data fn;
    src_training_main_pure_math_py --> src_training_main_pure_math_py_gen_data
    src_training_main_pure_math_py_train["train"]
    class src_training_main_pure_math_py_train fn;
    src_training_main_pure_math_py --> src_training_main_pure_math_py_train
    src_training_main_pure_math_py_verify["verify"]
    class src_training_main_pure_math_py_verify fn;
    src_training_main_pure_math_py --> src_training_main_pure_math_py_verify
    src_training_main_pure_math_py_hard_prune["hard_prune"]
    class src_training_main_pure_math_py_hard_prune fn;
    src_training_main_pure_math_py --> src_training_main_pure_math_py_hard_prune
    src_training_grokkit_physics_py["grokkit_physics.py (py)"]
    class src_training_grokkit_physics_py mod;
    src_training_grokkit_physics_py_strassen_multiply["strassen_multiply"]
    class src_training_grokkit_physics_py_strassen_multiply fn;
    src_training_grokkit_physics_py --> src_training_grokkit_physics_py_strassen_multiply
    src_training_grokkit_physics_py_measure_physics["measure_physics"]
    class src_training_grokkit_physics_py_measure_physics fn;
    src_training_grokkit_physics_py --> src_training_grokkit_physics_py_measure_physics
    src_training_grokkit_physics_py_detect_phase_transition["detect_phase_transition"]
    class src_training_grokkit_physics_py_detect_phase_transition fn;
    src_training_grokkit_physics_py --> src_training_grokkit_physics_py_detect_phase_transition
    src_training_grokkit_physics_py_main["main"]
    class src_training_grokkit_physics_py_main fn;
    src_training_grokkit_physics_py --> src_training_grokkit_physics_py_main
    compute_gns_checkpoints_py["compute_gns_checkpoints.py (py)"]
    class compute_gns_checkpoints_py mod;
    compute_gns_checkpoints_py_estimate_gns["estimate_gns"]
    class compute_gns_checkpoints_py_estimate_gns fn;
    compute_gns_checkpoints_py --> compute_gns_checkpoints_py_estimate_gns
    compute_gns_checkpoints_py_main["main"]
    class compute_gns_checkpoints_py_main fn;
    compute_gns_checkpoints_py --> compute_gns_checkpoints_py_main
    src_training_strassen_grokkit_py["strassen_grokkit.py (py)"]
    class src_training_strassen_grokkit_py mod;
    src_training_strassen_grokkit_py_StrassenOperator["StrassenOperator"]
    class src_training_strassen_grokkit_py_StrassenOperator cls;
    src_training_strassen_grokkit_py --> src_training_strassen_grokkit_py_StrassenOperator
    src_training_strassen_grokkit_py_generate_batch["generate_batch"]
    class src_training_strassen_grokkit_py_generate_batch fn;
    src_training_strassen_grokkit_py --> src_training_strassen_grokkit_py_generate_batch
    src_training_strassen_grokkit_py_train_grokkit["train_grokkit"]
    class src_training_strassen_grokkit_py_train_grokkit fn;
    src_training_strassen_grokkit_py --> src_training_strassen_grokkit_py_train_grokkit
    src_training_strassen_grokkit_py_verify_grokking["verify_grokking"]
    class src_training_strassen_grokkit_py_verify_grokking fn;
    src_training_strassen_grokkit_py --> src_training_strassen_grokkit_py_verify_grokking
    src_training_strassen_grokkit_py_progressive_sparsification["progressive_sparsification"]
    class src_training_strassen_grokkit_py_progressive_sparsification fn;
    src_training_strassen_grokkit_py --> src_training_strassen_grokkit_py_progressive_sparsification
    src_training_train_strassen_py["train_strassen.py (py)"]
    class src_training_train_strassen_py mod;
    src_training_train_strassen_py_StrassenOperator["StrassenOperator"]
    class src_training_train_strassen_py_StrassenOperator cls;
    src_training_train_strassen_py --> src_training_train_strassen_py_StrassenOperator
    src_training_train_strassen_py_generate_batch["generate_batch"]
    class src_training_train_strassen_py_generate_batch fn;
    src_training_train_strassen_py --> src_training_train_strassen_py_generate_batch
    src_training_train_strassen_py_train_phase1["train_phase1"]
    class src_training_train_strassen_py_train_phase1 fn;
    src_training_train_strassen_py --> src_training_train_strassen_py_train_phase1
    src_training_train_strassen_py_sparsify["sparsify"]
    class src_training_train_strassen_py_sparsify fn;
    src_training_train_strassen_py --> src_training_train_strassen_py_sparsify
    src_training_train_strassen_py_discretize["discretize"]
    class src_training_train_strassen_py_discretize fn;
    src_training_train_strassen_py --> src_training_train_strassen_py_discretize
    menu_py["menu.py (py)"]
    class menu_py mod;
    menu_py_clear_screen["clear_screen"]
    class menu_py_clear_screen fn;
    menu_py --> menu_py_clear_screen
    menu_py_print_header["print_header"]
    class menu_py_print_header fn;
    menu_py --> menu_py_print_header
    menu_py_print_wrapped["print_wrapped"]
    class menu_py_print_wrapped fn;
    menu_py --> menu_py_print_wrapped
    menu_py_wait_for_enter["wait_for_enter"]
    class menu_py_wait_for_enter fn;
    menu_py --> menu_py_wait_for_enter
    menu_py_run_script["run_script"]
    class menu_py_run_script fn;
    menu_py --> menu_py_run_script
    src_benchmarks_strassen_numpy_py["strassen_numpy.py (py)"]
    class src_benchmarks_strassen_numpy_py mod;
    src_benchmarks_strassen_numpy_py__load_weights["_load_weights"]
    class src_benchmarks_strassen_numpy_py__load_weights fn;
    src_benchmarks_strassen_numpy_py --> src_benchmarks_strassen_numpy_py__load_weights
    src_benchmarks_strassen_numpy_py_strassen_2x2_numpy["strassen_2x2_numpy"]
    class src_benchmarks_strassen_numpy_py_strassen_2x2_numpy fn;
    src_benchmarks_strassen_numpy_py --> src_benchmarks_strassen_numpy_py_strassen_2x2_numpy
    src_benchmarks_strassen_numpy_py_strassen_numpy["strassen_numpy"]
    class src_benchmarks_strassen_numpy_py_strassen_numpy fn;
    src_benchmarks_strassen_numpy_py --> src_benchmarks_strassen_numpy_py_strassen_numpy
    src_benchmarks_strassen_numpy_py_strassen_hybrid["strassen_hybrid"]
    class src_benchmarks_strassen_numpy_py_strassen_hybrid fn;
    src_benchmarks_strassen_numpy_py --> src_benchmarks_strassen_numpy_py_strassen_hybrid
    src_benchmarks_strassen_numpy_py_multiplication_count["multiplication_count"]
    class src_benchmarks_strassen_numpy_py_multiplication_count fn;
    src_benchmarks_strassen_numpy_py --> src_benchmarks_strassen_numpy_py_multiplication_count
    src_benchmarks_benchmark_final_py["benchmark_final.py (py)"]
    class src_benchmarks_benchmark_final_py mod;
    src_benchmarks_benchmark_final_py_strassen_hybrid_multiply["strassen_hybrid_multiply"]
    class src_benchmarks_benchmark_final_py_strassen_hybrid_multiply fn;
    src_benchmarks_benchmark_final_py --> src_benchmarks_benchmark_final_py_strassen_hybrid_multiply
    src_benchmarks_benchmark_final_py_numpy_multiply["numpy_multiply"]
    class src_benchmarks_benchmark_final_py_numpy_multiply fn;
    src_benchmarks_benchmark_final_py --> src_benchmarks_benchmark_final_py_numpy_multiply
    src_benchmarks_benchmark_final_py_benchmark["benchmark"]
    class src_benchmarks_benchmark_final_py_benchmark fn;
    src_benchmarks_benchmark_final_py --> src_benchmarks_benchmark_final_py_benchmark
    src_benchmarks_benchmark_final_py_main["main"]
    class src_benchmarks_benchmark_final_py_main fn;
    src_benchmarks_benchmark_final_py --> src_benchmarks_benchmark_final_py_main
    experiments_validation_benchmark_py["benchmark.py (py)"]
    class experiments_validation_benchmark_py mod;
    experiments_validation_benchmark_py_strassen_numpy["strassen_numpy"]
    class experiments_validation_benchmark_py_strassen_numpy fn;
    experiments_validation_benchmark_py --> experiments_validation_benchmark_py_strassen_numpy
    experiments_validation_benchmark_py_measure_single_sgemm["measure_single_sgemm"]
    class experiments_validation_benchmark_py_measure_single_sgemm fn;
    experiments_validation_benchmark_py --> experiments_validation_benchmark_py_measure_single_sgemm
    experiments_validation_benchmark_py_run_planck_analysis["run_planck_analysis"]
    class experiments_validation_benchmark_py_run_planck_analysis fn;
    experiments_validation_benchmark_py --> experiments_validation_benchmark_py_run_planck_analysis
    src_native_strassen_optimal_c["strassen_optimal.c (c)"]
    class src_native_strassen_optimal_c mod;
    src_native_strassen_optimal_c_strassen_level["strassen_level"]
    class src_native_strassen_optimal_c_strassen_level fn;
    src_native_strassen_optimal_c --> src_native_strassen_optimal_c_strassen_level
    src_native_strassen_optimal_c_strassen_optimal["strassen_optimal"]
    class src_native_strassen_optimal_c_strassen_optimal fn;
    src_native_strassen_optimal_c --> src_native_strassen_optimal_c_strassen_optimal
    src_native_strassen_optimal_c_STRASSEN_THRESHOLD["STRASSEN_THRESHOLD"]
    class src_native_strassen_optimal_c_STRASSEN_THRESHOLD fn;
    src_native_strassen_optimal_c --> src_native_strassen_optimal_c_STRASSEN_THRESHOLD
    experiments_statistics_coherence_analysis_py["coherence_analysis.py (py)"]
    class experiments_statistics_coherence_analysis_py mod;
```

---

## Architecture Reference

### C (3 files)

#### `strassen_c.c`
**Path:** `src/native/strassen_c.c`

**Functions:**
- `alloc_matrix` (line 15) - *Strassen Matrix Multiplication - C Implementation Author: grisun0  Compila: gcc -O3 -ffast-math -march=native -shared -fPIC -o libstrassen.so stras...*
- `matmul_standard` (line 20) - *#include <stdlib.h> #include <string.h> #include <stdio.h>  #define THRESHOLD 64  /* Allocate matrix static float* alloc_matrix(int n) { return (fl...*
- `mat_add` (line 33) - */* Standard matrix multiplication for small matrices static void matmul_standard(float* C, float* A, float* B, int n) { memset(C, 0, n * n * sizeof...*
- `mat_sub` (line 41) - *} } } }  /* Add matrices: C = A + B static void mat_add(float* C, float* A, float* B, int n) { int nn = n * n; for (int i = 0; i < nn; i++) { C[i] ...*
- `extract_quadrant` (line 49) - *for (int i = 0; i < nn; i++) { C[i] = A[i] + B[i]; } }  /* Subtract matrices: C = A - B static void mat_sub(float* C, float* A, float* B, int n) { ...*
- `insert_quadrant` (line 57) - *for (int i = 0; i < nn; i++) { C[i] = A[i] - B[i]; } }  /* Extract quadrant from matrix static void extract_quadrant(float* Q, float* M, int n, int...*
- `strassen_recursive` (line 65) - *for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } }  /* Insert quadrant into matrix static void insert...*
- `strassen_multiply` (line 164) - *insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C, C22, n, h, h);  /* Free me...*
- `standard_multiply` (line 169) - */* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free...*

**Macros:**
- `THRESHOLD` (line 11)

#### `strassen_optimal.c`
**Path:** `src/native/strassen_optimal.c`

**Functions:**
- `strassen_level` (line 18) - *Uses in-place operations where possible and only applies Strassen for very large matrices where the asymptotic advantage overcomes overhead.   #inc...*
- `strassen_optimal` (line 130)

**Macros:**
- `STRASSEN_THRESHOLD` (line 15)

#### `strassen_turbo.c`
**Path:** `src/native/strassen_turbo.c`

**Functions:**
- `alloc_matrix` (line 25) - *Compile: gcc -O3 -ffast-math -march=native -fopenmp -mavx2 -shared -fPIC -o libstrassen_turbo.so strassen_turbo.c   #include <stdlib.h> #include <s...*
- `mat_add_avx` (line 30) - *#include <stdio.h> #include <omp.h> #include <immintrin.h>  #define THRESHOLD 128 #define BLOCK_SIZE 32 #define ALIGN 32  /* Aligned allocation sta...*
- `mat_sub_avx` (line 50) - *for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_add_ps(va, vb); _mm256_store...*
- `matmul_blocked_avx` (line 68) - *for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_sub_ps(va, vb); _mm256_store...*
- `extract_quadrant` (line 104) - *_mm256_storeu_ps(&C[i * n + j], vc); }  for (; j < j_end; j++) { C[i * n + j] += a_ik * B[k * n + j]; } } } } } } }  /* Extract quadrant*
- `insert_quadrant` (line 114) - *} }  /* Extract quadrant static void extract_quadrant(float* __restrict Q, const float* __restrict M, int n, int row, int col) { int h = n / 2; #pr...*
- `strassen_turbo_recursive` (line 124) - *} }  /* Insert quadrant static void insert_quadrant(float* __restrict M, const float* __restrict Q, int n, int row, int col) { int h = n / 2; #prag...*
- `strassen_turbo` (line 261) - *insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C, C22, n, h, h);  /* Free me...*
- `get_num_threads` (line 267) - *free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6);...*

**Macros:**
- `THRESHOLD` (line 19)
- `BLOCK_SIZE` (line 21)
- `ALIGN` (line 22)

### PY (56 files)

#### `app.py`
**Path:** `app.py`

**Classs:**
- `StrassenNet` (line 24)

**Functions:**
- `__init__` (line 25)
- `forward` (line 31)

#### `batch_size.py`
**Path:** `batch_size.py`

**Classs:**
- `Configuration` (line 31)
- `BilinearStrassenModel` (line 77)
- `CheckpointMigrator` (line 100)
- `CustomFormatMigrator` (line 110)
- `StandardFormatMigrator` (line 122)
- `CheckpointMigrationManager` (line 131)
- `StrassenDataGenerator` (line 147)
- `CrystallographyMetrics` (line 156)
- `PlanckConstantCalculator` (line 186)
- `BatchSizeThermodynamics` (line 217)
- `StrassenCheckpointLoader` (line 262)
- `StrassenPlanckAnalyzer` (line 292)

**Functions:**
- `set_random_seed` (line 71)
- `main` (line 341)
- `__init__` (line 78)
- `forward` (line 88)
- `get_coefficients` (line 91)
- `compute_lambda_effective` (line 94)
- `can_migrate` (line 102)
- `migrate` (line 106)
- `can_migrate` (line 111)
- `migrate` (line 114)
- `can_migrate` (line 123)
- `migrate` (line 126)
- `__init__` (line 132)
- `migrate_checkpoint` (line 135)
- `generate_batch` (line 149)
- `compute_kappa` (line 158)
- `compute_discretization_margin` (line 174)
- `compute_local_complexity` (line 178)
- `__init__` (line 187)
- `calculate_all` (line 196)
- `__init__` (line 218)
- `analyze_batch_size_spectrum` (line 224)
- `_measure_gradients` (line 246)
- `__init__` (line 263)
- `load` (line 267)
- `extract_training_metrics` (line 281)
- `__init__` (line 293)
- `analyze_checkpoint` (line 297)
- `analyze_directory` (line 323)

#### `boltzmann_experiments.py`
**Path:** `boltzmann_experiments.py`

**Classs:**
- `Config` (line 19)
- `CheckpointLoadingError` (line 37)
- `ICheckpointLoader` (line 40)
- `CheckpointLoader` (line 45)
- `CheckpointMigrator` (line 52)
- `BilinearStrassenModel` (line 118)
- `CrystallographyMetrics` (line 137)
- `DLProgram` (line 199)

**Functions:**
- `set_seed` (line 30)
- `main` (line 935)
- `_simulate_training_trajectory` (line 951)
- `_compute_generalization_entropy` (line 962) - *Entropía de generalización con manejo robusto de dimensionalidad*
- `_fit_timescale` (line 1016)
- `_plot_entropy_production` (line 1026)
- `phase3_extensivity_law` (line 1044)
- `load_checkpoint` (line 42)
- `load_checkpoint` (line 46)
- `migrate_checkpoint` (line 54)
- `_format_direct_tensors` (line 70)
- `_migrate_dict` (line 92)
- `_migrate_encoder_format` (line 105)
- `_migrate_coefs_format` (line 115)
- `__init__` (line 119)
- `_initialize_symmetric` (line 126)
- `forward` (line 131)
- `get_coefficients` (line 134)
- `compute_kappa` (line 139) - *Classical kappa - will be inf for discrete states*
- `compute_delta` (line 156) - *Discretization error δ*
- `compute_local_complexity` (line 161)
- `compute_alpha_purity` (line 169) - *Alpha purity: α = -log(δ), inverse temperature metric for discrete states*
- `compute_kappa_quantum` (line 178) - *Quantum-regularized kappa for singular covariance states*
- `__init__` (line 200)
- `_load_all_checkpoints` (line 207)
- `run_full_boltzmann_program` (line 247)
- `_print_executive_summary` (line 267)
- `_save_results` (line 307)
- `phase1_molecular_hypothesis` (line 328)
- `_compute_entropy_simple` (line 435) - *Entropía simple sin KDE para datos de baja varianza*
- `_compute_entropy` (line 446) - *Entropía con manejo robusto de covarianza*
- `_compute_effective_volume` (line 466)
- `_plot_parameter_distribution` (line 475)
- `phase2_entropy_production` (line 506)
- `_simulate_training_trajectory` (line 592)
- `_compute_generalization_entropy` (line 604) - *Entropía de generalización con manejo robusto de datos idénticos*
- `_fit_timescale` (line 691)
- `_plot_entropy_production` (line 701)
- `phase3_extensivity_law` (line 719)
- `_verify_scaling` (line 773)
- `_recursive_strassen` (line 783)
- `_fit_extensivity` (line 812)
- `_verify_extensivity_universality` (line 824)
- `_plot_extensivity` (line 828)
- `phase4_quantum_basis_transform` (line 841)
- `_find_broken_symmetries` (line 895)
- `_measure_uncertainty` (line 903)
- `_plot_uncertainty_distribution` (line 914)
- `model` (line 1017)
- `convert_to_serializable` (line 310)
- `model` (line 692)
- `model` (line 813)

#### `compute_gns_checkpoints.py`
**Path:** `compute_gns_checkpoints.py`

**Functions:**
- `estimate_gns` (line 11)
- `main` (line 37)

#### `crystallography.py`
**Path:** `crystallography.py`

**Classs:**
- `Config` (line 25)
- `BilinearStrassenModel` (line 44)
- `CheckpointMigrator` (line 71)
- `StrassenDataGenerator` (line 155)
- `SparsificationProtocol` (line 172)
- `CrystallographyMetrics` (line 206)
- `StrassenDiffractionTest` (line 232)
- `BasinResilienceSpectrometer` (line 277)
- `CrystalPurityIndex` (line 345)
- `StrassenCrystallographer` (line 416)
- `LocalComplexity` (line 523)

**Functions:**
- `set_seed` (line 35)
- `main` (line 542)
- `__init__` (line 45)
- `_initialize_symmetric` (line 52)
- `forward` (line 57)
- `get_coefficients` (line 60)
- `migrate_checkpoint` (line 73)
- `_migrate_custom` (line 106) - *Maneja formatos custom U,V,W directos*
- `_migrate_encoder` (line 123) - *Extracción de encoder.layers*
- `_migrate_standard` (line 147) - *Formato estándar U.weight, V.weight, W.weight*
- `generate_batch` (line 157)
- `verify_structure` (line 164)
- `__init__` (line 173)
- `prune_to_target` (line 176)
- `discretize_weights` (line 192)
- `compute_kappa` (line 208)
- `compute_discretization_margin` (line 225)
- `__init__` (line 233)
- `test_gauge_invariance` (line 236)
- `_functional_error` (line 262)
- `__init__` (line 278)
- `measure_resilience_spectrum` (line 282)
- `_test_noise_recovery` (line 293)
- `_apply_noise` (line 312)
- `_anneal_to_attractor` (line 317)
- `_estimate_critical_noise` (line 329)
- `__init__` (line 346)
- `compute` (line 359)
- `_assign_grade` (line 399)
- `__init__` (line 417)
- `run_full_analysis` (line 445)
- `_save_report` (line 506)
- `compute` (line 525) - *Computa LC basado en Can't Stop Won't Stop paper*
- `dataloader_gen` (line 465)

#### `dirac_polos_zeros.py`
**Path:** `dirac_polos_zeros.py`

**Classs:**
- `AnalysisConfig` (line 24)
- `IModel` (line 54)
- `IChargeDistributionExtractor` (line 60)
- `IDiracAnalyzer` (line 65)
- `IFieldCalculator` (line 70)
- `IFluxCalculator` (line 75)
- `IStateSpaceExtractor` (line 80)
- `ITransferFunctionComputer` (line 85)
- `IPoleZeroAnalyzer` (line 90)
- `IFrequencyAnalyzer` (line 97)
- `ITimeResponseAnalyzer` (line 104)
- `ICheckpointLoader` (line 110)
- `ICheckpointMigrator` (line 115)
- `IVisualizer` (line 120)
- `BilinearModel` (line 124)
- `ChargeDistributionExtractor` (line 152)
- `DiracDeltaAnalyzer` (line 160)
- `ElectricFieldCalculator` (line 192)
- `ElectricFluxCalculator` (line 225)
- `DivergenceCalculator` (line 248)
- `GaussLawVerifier` (line 253)
- `StateSpaceExtractor` (line 271)
- `TransferFunctionComputer` (line 298)
- `PoleZeroAnalyzer` (line 313)
- `FrequencyResponseAnalyzer` (line 463)
- `TimeResponseAnalyzer` (line 566)
- `CheckpointLoader` (line 653)
- `CheckpointMigrator` (line 661)
- `ChargeDistributionVisualizer` (line 710)
- `ElectricFieldVisualizer` (line 738)
- `DivergenceVisualizer` (line 780)
- `PoleZeroVisualizer` (line 809)
- `BodeVisualizer` (line 848)
- `NyquistVisualizer` (line 899)
- `TimeResponseVisualizer` (line 936)
- `CombinedVisualizer` (line 966)
- `SystemAnalyzer` (line 1070)
- `AnalysisPipeline` (line 1288)

**Functions:**
- `main` (line 1545)
- `forward` (line 55)
- `get_coefficients` (line 56)
- `extract` (line 61)
- `analyze` (line 66)
- `calculate` (line 71)
- `calculate` (line 76)
- `extract` (line 81)
- `compute` (line 86)
- `analyze_stability` (line 91)
- `get_poles` (line 92)
- `get_zeros` (line 93)
- `compute_bode` (line 98)
- `compute_margins` (line 99)
- `compute_nyquist` (line 100)
- `compute_step` (line 105)
- `compute_impulse` (line 106)
- `load` (line 111)
- `migrate` (line 116)
- `visualize` (line 121)
- `__init__` (line 125)
- `_initialize` (line 136)
- `forward` (line 141)
- `get_coefficients` (line 144)
- `extract` (line 153)
- `__init__` (line 161)
- `analyze` (line 164)
- `__init__` (line 193)
- `calculate` (line 196)
- `__init__` (line 226)
- `calculate` (line 229)
- `calculate` (line 249)
- `__init__` (line 254)
- `verify` (line 257)
- `extract` (line 272)
- `compute` (line 299)
- `__init__` (line 314)
- `_compute` (line 323)
- `get_poles` (line 334)
- `get_zeros` (line 337)
- `analyze_stability` (line 340)
- `classify_poles` (line 378)
- `compute_damping` (line 406)
- `compute_time_constants` (line 445)
- `__init__` (line 464)
- `compute_bode` (line 474)
- `compute_margins` (line 490)
- `compute_nyquist` (line 516)
- `evaluate_nyquist_stability` (line 535)
- `__init__` (line 567)
- `compute_step` (line 577)
- `compute_impulse` (line 589)
- `analyze_step_characteristics` (line 601)
- `load` (line 654)
- `migrate` (line 662)
- `_migrate_dict` (line 674)
- `_migrate_custom_format` (line 683)
- `_migrate_coefs_format` (line 699)
- `_migrate_standard_format` (line 706)
- `__init__` (line 711)
- `visualize` (line 714)
- `__init__` (line 739)
- `visualize` (line 742)
- `__init__` (line 781)
- `visualize` (line 784)
- `__init__` (line 810)
- `visualize` (line 813)
- `__init__` (line 849)
- `visualize` (line 852)
- `__init__` (line 900)
- `visualize` (line 903)
- `__init__` (line 937)
- `visualize` (line 940)
- `__init__` (line 967)
- `visualize` (line 970)
- `__init__` (line 1071)
- `_load_model` (line 1088)
- `analyze` (line 1104)
- `_print_report` (line 1204)
- `__init__` (line 1289)
- `process_checkpoint` (line 1300)
- `process_directory` (line 1357)
- `generate_summary` (line 1385)
- `_compute_aggregate_statistics` (line 1406)
- `_generate_text_report` (line 1473)

#### `ablation_8192.py`
**Path:** `experiments/ablation/ablation_8192.py`

*No symbols extracted*

#### `ablation_study.py`
**Path:** `experiments/ablation/ablation_study.py`

**Classs:**
- `BenchmarkResult` (line 27)

**Functions:**
- `load_libraries` (line 54) - *Cargar bibliotecas con manejo de errores*
- `run_openblas` (line 111) - *Ejecutar multiplicación con OpenBLAS*
- `run_strassen` (line 122) - *Ejecutar multiplicación con Strassen*
- `benchmark_single` (line 132) - *Benchmark una implementación*
- `run_ablation` (line 180) - *Ejecutar ablación completa*
- `analyze_results` (line 233) - *Analizar y presentar resultados*
- `main` (line 273)
- `mean_time` (line 35)
- `std_time` (line 39)
- `min_time` (line 43)
- `max_time` (line 47)
- `mean_gflops` (line 51)

#### `apendix_experiments.py`
**Path:** `experiments/apendix_experiments.py`

**Classs:**
- `StrassenOperator` (line 51)

**Functions:**
- `setup_matplotlib` (line 34)
- `generate_batch` (line 87)
- `generate_test_set` (line 93)
- `compute_delta` (line 100)
- `verify_strassen_structure` (line 117)
- `compute_S_theta` (line 138)
- `compute_gradient_covariance` (line 154)
- `train_with_logging` (line 187)
- `sparsify_and_discretize` (line 274)
- `run_phase_diagram` (line 327)
- `run_batch_size_effect` (line 429)
- `main` (line 526)
- `__init__` (line 53)
- `forward` (line 67)
- `slot_importance` (line 77)
- `count_active` (line 83)

#### `cache_analysis_v2.py`
**Path:** `experiments/cache_analysis_v2.py`

**Functions:**
- `cache_analysis` (line 7) - *Full memory analysis for training.

Model: Bilinear with rank-8 (target 7 active)
- U: 8 x 4 = 32 floats
- V: 8 x 4 = 32 floats  
- W: 4 x 8 = 32 floats
Total params: 96 floats = 384 bytes

AdamW optimizer state (per param):
- m (momentum): 96 floats
- v (variance): 96 floats
Total optimizer: 192 floats = 768 bytes

Per-batch memory:
- Input batch: B x 8 floats (A, B flattened)
- Hidden: B x 8 floats (intermediate M)
- Output: B x 4 floats (C)
- Gradients: same as forward
Total per-sample: ~40 floats x 2 (forward+backward) = 80 floats = 320 bytes

Total training memory = model + optimizer + batch*

#### `all_test_extended.py`
**Path:** `experiments/extended_experiments/all_test_extended.py`

**Classs:**
- `Configuration` (line 41)
- `Narrator` (line 95)
- `SystemFingerprint` (line 152)
- `ArithmeticDataset` (line 190) - *Dataset for arithmetic operations based on original implementation.

Generates (a, b) pairs and their product c = (a * b) mod MODULUS.
Each a, b, c is a one-hot vector of size MODULUS.

The BilinearModel learns: c = W @ ((U @ a) * (V @ b))
This is a lookup table for modular multiplication.*
- `BilinearModel` (line 238) - *Bilinear model from original implementation.

Architecture:
- U: Linear(d_vocab, rank) -> [67, 8] weights
- V: Linear(d_vocab, rank) -> [67, 8] weights  
- W: Linear(rank, d_vocab) -> [8, 67] weights

Forward:
- a = x[:, 0] -> (batch, d_vocab)
- b = x[:, 1] -> (batch, d_vocab)
- m = U(a) * V(b) -> (batch, rank)
- logits = W(m) -> (batch, d_vocab)*
- `Task` (line 293)
- `MatrixMultiplicationTask` (line 311)
- `ParityDataset` (line 354) - *Dataset for parity task.*
- `ParityTask` (line 375)
- `GradientCovarianceProbe` (line 402) - *Analyzes gradient covariance to understand batch size effects.*
- `SpectralInterventionProbe` (line 465) - *Actively intervenes on condition number during training.*
- `AttractorLandscapeProbe` (line 488) - *Analyzes attractor landscapes to understand failure modes.*
- `VolumeEstimator` (line 510) - *Estimates volume of success basin in weight space.*
- `RobustnessTest` (line 524) - *Tests robustness against various perturbations.*
- `CheckpointManager` (line 581) - *Manages training checkpoints with configurable intervals.*
- `TrainingMetrics` (line 627) - *Comprehensive metrics collection for training progress.*
- `ExperimentRunner` (line 700) - *Main orchestration engine for training and evaluation.*
- `DiscretizationAnalyzer` (line 809) - *Analyzes discretization quality and success probability.*
- `ExpansionVerifier` (line 851) - *Verifies zero-shot transfer to larger problem sizes.*
- `ExperimentPipeline` (line 879) - *Executes the complete battery of open-question experiments.*

**Functions:**
- `main` (line 1403)
- `__init__` (line 89)
- `__init__` (line 96)
- `begin` (line 101)
- `progress` (line 109)
- `checkpoint` (line 116)
- `result` (line 121)
- `verdict` (line 126)
- `failure` (line 131)
- `complete` (line 136)
- `claim` (line 144)
- `note` (line 148)
- `__init__` (line 153)
- `capture` (line 156)
- `report` (line 174)
- `__init__` (line 202)
- `_generate_data` (line 207)
- `__len__` (line 231)
- `__getitem__` (line 234)
- `__init__` (line 253)
- `forward` (line 265)
- `get_weights` (line 277)
- `get_U_weights` (line 283)
- `get_V_weights` (line 286)
- `get_W_weights` (line 289)
- `name` (line 295)
- `d_vocab` (line 299)
- `generate_dataset` (line 303)
- `verify` (line 307)
- `__init__` (line 312)
- `name` (line 316)
- `d_vocab` (line 319)
- `generate_dataset` (line 322)
- `verify` (line 344)
- `__init__` (line 357)
- `_generate_data` (line 362)
- `__len__` (line 368)
- `__getitem__` (line 371)
- `__init__` (line 376)
- `name` (line 381)
- `d_vocab` (line 384)
- `generate_dataset` (line 387)
- `verify` (line 391)
- `__init__` (line 405)
- `capture_gradients` (line 409)
- `compute_covariance` (line 429)
- `compute_condition_number` (line 436)
- `compute_gradient_noise_scale` (line 442)
- `analyze` (line 450)
- `__init__` (line 468)
- `spectral_regularizer` (line 472)
- `__init__` (line 491)
- `count_local_minima` (line 494)
- `measure_basin_width` (line 501)
- `classify_failure_mode` (line 505)
- `__init__` (line 513)
- `estimate_volume_monte_carlo` (line 516)
- `compute_fractal_dimension` (line 520)
- `__init__` (line 527)
- `add_gaussian_noise` (line 530)
- `fgsm_attack` (line 536)
- `quantize_weights` (line 544)
- `test_discretization_with_noise` (line 550)
- `run_fragility_analysis` (line 560)
- `__init__` (line 584)
- `save_checkpoint` (line 593)
- `load_checkpoint` (line 619)
- `__init__` (line 630)
- `update` (line 642)
- `detect_grokking` (line 660)
- `progress_bar_string` (line 679)
- `__init__` (line 703)
- `train_epoch` (line 708)
- `evaluate` (line 730)
- `run_training` (line 748)
- `__init__` (line 812)
- `compute_discretization_margin` (line 815)
- `discretize_weights` (line 827)
- `check_strassen_structure` (line 832)
- `count_discretized_parameters` (line 839)
- `__init__` (line 854)
- `verify_expansion` (line 857)
- `__init__` (line 882)
- `experiment_batch_size_mechanism` (line 891) - *Experiment 1: Why batch size [24,128] works.*
- `experiment_kappa_intervention` (line 962) - *Experiment 2: Active intervention on κ.*
- `experiment_failure_analysis` (line 1026) - *Experiment 3: Why 32% of runs fail.*
- `experiment_generalization` (line 1088) - *Experiment 4: Generalization to other tasks.*
- `experiment_basin_volume` (line 1145) - *Experiment 5: Basin volume estimation.*
- `experiment_hardware_reproducibility` (line 1158) - *Experiment 6: Hardware reproducibility testing.*
- `experiment_fragility` (line 1217) - *Experiment 7: Discretization fragility testing.*
- `run_all_experiments` (line 1332)
- `_save_results` (line 1371)
- `_generate_summary` (line 1379)
- `checker` (line 1253)

#### `exp1_covariance_spectrometry.py`
**Path:** `experiments/extended_experiments/exp1_covariance_spectrometry.py`

**Classs:**
- `StrassenOperator` (line 48) - *Spectral operator for 2x2 matrix multiplication.
Tensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)*

**Functions:**
- `setup_matplotlib` (line 22)
- `generate_batch` (line 112) - *Generate batch of matrices.*
- `compute_gradient_covariance` (line 120) - *Compute the gradient covariance matrix Σₜ and its eigenvalues.

Returns:
    kappa: condition number λ_max / λ_min
    lambda_max: maximum eigenvalue
    lambda_min: minimum eigenvalue  
    trace: trace of covariance (total gradient energy)
    frobenius_norm: Frobenius norm of covariance*
- `load_checkpoint` (line 196) - *Load model from checkpoint file.*
- `analyze_checkpoint` (line 222) - *Analyze a single checkpoint with multiple batch sizes.*
- `main` (line 277) - *Main execution for Experiment 1.*
- `generate_visualization` (line 380) - *Generate publication-quality figures.*
- `__init__` (line 54)
- `forward` (line 61)
- `get_all_parameters` (line 71) - *Get all parameters as a single flattened vector.*
- `compute_per_sample_gradients` (line 78) - *Compute per-sample gradients for covariance estimation.
Returns: gradients shape [batch_size, num_parameters]*

#### `exp2_noise_ablation.py`
**Path:** `experiments/extended_experiments/exp2_noise_ablation.py`

**Classs:**
- `StrassenOperator` (line 49) - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 25)
- `generate_batch` (line 99) - *Generate batch of matrices.*
- `compute_gradient_covariance_matrix` (line 107) - *Compute the gradient covariance matrix Σ.*
- `get_eigenbasis` (line 139) - *Get eigenvectors and eigenvalues of covariance matrix.*
- `load_checkpoint` (line 150) - *Load model from checkpoint file.*
- `experiment_treatment_a_gradient_noise` (line 172) - *Treatment A: Add noise to gradients DURING forward/backward pass.

This simulates the effect of gradient noise during training without actual retraining.*
- `experiment_treatment_b_weight_noise` (line 212) - *Treatment B: Noise on weights BEFORE evaluation (already done in paper).
This is the fallback mechanism test.*
- `experiment_treatment_c_structured_noise` (line 246) - *Treatment C: Structured noise by eigenvectors of Σ.

Tests whether damage is isotropic or aligned with gradient covariance directions.*
- `run_noise_ablation` (line 315) - *Run complete noise ablation experiment on a checkpoint.*
- `main` (line 348) - *Main execution for Experiment 2.*
- `generate_visualization` (line 433) - *Generate publication-quality figures.*
- `__init__` (line 54)
- `forward` (line 61)
- `get_all_parameters` (line 71)
- `set_parameters` (line 77) - *Set parameters from a flattened tensor.*
- `compute_loss` (line 85) - *Compute MSE loss.*
- `compute_accuracy` (line 91) - *Compute accuracy (proportion of predictions within threshold).*

#### `exp3_prospective_prediction.py`
**Path:** `experiments/extended_experiments/exp3_prospective_prediction.py`

**Classs:**
- `StrassenOperator` (line 56) - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 32)
- `generate_batch` (line 114) - *Generate batch of matrices.*
- `compute_kappa` (line 121) - *Compute condition number κ(Σ) of gradient covariance matrix.*
- `load_checkpoint` (line 166) - *Load model from checkpoint file.*
- `simulate_early_prediction` (line 188) - *Simulate the prospective prediction experiment.

Since we can't actually retrain, we use available checkpoints to simulate:
- "Early" checkpoint = first checkpoint in sequence
- "Final" checkpoint = last checkpoint in sequence

This tests if early-stage κ predicts final-stage success.*
- `run_prospective_prediction_experiment` (line 236) - *Run the full prospective prediction experiment across all checkpoints.*
- `compute_roc_analysis` (line 252) - *Compute ROC curve and AUC for κ as predictor of success.*
- `main` (line 313) - *Main execution for Experiment 3.*
- `generate_visualization` (line 410) - *Generate publication-quality figures.*
- `__init__` (line 59)
- `forward` (line 66)
- `get_all_parameters` (line 76)
- `set_parameters` (line 82)
- `count_active_slots` (line 89) - *Count active slots based on weight norms.*
- `compute_discretization_margin` (line 97) - *Compute how close weights are to discrete values {-1, 0, 1}.
δ(θ) = mean(|w - round(w)|)*
- `is_grokked` (line 107) - *Check if model has grokked (discretized with low error).*

#### `exp4_trajectory_perturbation.py`
**Path:** `experiments/extended_experiments/exp4_trajectory_perturbation.py`

**Classs:**
- `StrassenOperator` (line 51) - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 27)
- `generate_batch` (line 117) - *Generate batch of matrices.*
- `load_checkpoint` (line 124) - *Load model from checkpoint file.*
- `simulate_trajectory_perturbation` (line 146) - *Simulate trajectory perturbation effects using available checkpoints.

Since we can't retrain, we simulate the effect of perturbations by:
1. Taking a "final" checkpoint as the target
2. Using earlier checkpoints to simulate "early training state"
3. Applying perturbations and measuring their effect*
- `main` (line 277) - *Main execution for Experiment 4.*
- `generate_visualization` (line 398) - *Generate publication-quality figures.*
- `__init__` (line 54)
- `forward` (line 61)
- `get_all_parameters` (line 71)
- `set_parameters` (line 77)
- `get_weight_norm` (line 84) - *Get total L2 norm of all parameters.*
- `get_weight_direction` (line 91) - *Get normalized weight vector direction.*
- `compute_gradient_norm` (line 96) - *Compute norm of gradients.*
- `cosine_similarity` (line 110) - *Compute cosine similarity between current weights and target weights.*
- `compute_metrics` (line 194) - *Compute evaluation metrics.*

#### `exp5_discreteness_attractors.py`
**Path:** `experiments/extended_experiments/exp5_discreteness_attractors.py`

*No symbols extracted*

#### `run_all_experiments.py`
**Path:** `experiments/extended_experiments/run_all_experiments.py`

**Classs:**
- `StrassenOperator` (line 48) - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 24)
- `generate_batch` (line 95)
- `load_checkpoint_robust` (line 101) - *Load checkpoint with multiple format fallback strategies.*
- `compute_gradient_covariance_safe` (line 135) - *Compute κ(Σₜ) with numerical safety.*
- `run_all_experiments` (line 205) - *Run all experiments.*
- `generate_summary_visualization` (line 496) - *Generate summary visualization.*
- `__init__` (line 51)
- `forward` (line 58)
- `get_all_parameters` (line 68)
- `set_parameters` (line 74)
- `count_active_slots` (line 81)
- `compute_discretization_margin` (line 88)
- `compute_accuracy` (line 324)

#### `validate2.py`
**Path:** `experiments/extended_experiments/validate2.py`

**Classs:**
- `ExperimentConfig` (line 60) - *Configuración centralizada para todos los experimentos.

Sigue el principio de responsabilidad única - solo gestiona parámetros.
No usa magic numbers; todos los valores están definidos aquí.*
- `StrassenOperator` (line 126) - *Operador Strassen para multiplicación de matrices 2x2 vía descomposición tensorial.

El modelo representa el tensor de rango R:
C_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)

Donde:
- U, V: Coeficientes de combinación lineal (LC)
- W: Coeficientes de reconstrucción
- Sparsity (SP): Cuántos slots están activos*
- `StrassenDataGenerator` (line 225) - *Generador de datos para multiplicación de matrices 2x2.*
- `LocalComplexityCalculator` (line 277) - *Calculadora de Complejidad Local basada en la varianza del gradiente.

LC = ||grad||^2 / N (Noise Scale normalizada)

Esta métrica captura la "dificultad" del batch actual y su relación
con el aprendizaje del modelo.*
- `GrokkingVerifier` (line 340) - *Verifica que un modelo ha grokkeado correctamente.*
- `IterativePruningEngine` (line 405) - *Motor para poda iterativa con fine-tuning completo.

Protocolo:
1. Calcular importancia de pesos (magnitud L1)
2. Podar p% de pesos menos importantes
3. Fine-tune por épocas especificadas
4. Chequear degradación δ
5. Si δ < threshold, continuar; si no, detener
6. Verificar discretización con δ < 0.1 (PUNTO A DEL REVISOR)

Este protocolo es CRUCIAL para verificar la hipótesis de cuenca discreta.*
- `LocalComplexityExperiment` (line 764) - *Experimento de Local Complexity entrenando desde cero.

Esto es CRUCIAL para responder al revisor (PUNTO B):
- Se entrena un modelo desde cero hasta grokking
- Se mide LC en cada época para capturar la transición de fase
- Si LC muestra un cambio alrededor del grokking, la métrica es útil
- Si LC permanece constante, la métrica NO captura la transición*
- `BalancedRunsGenerator` (line 890) - *Generador de runs balanceados para calcular AUC válido.

Esto es CRUCIAL para responder al revisor (PUNTO C):
- Entrenar múltiples modelos con diferentes hiperparámetros
- Algunos grokkean, otros no (condiciones variadas)
- Generar dataset balanceado para ROC/AUC

Si todos los samples son de una sola clase, AUC es indefinido.
Necesitamos mix de grokked + no-grokked para calcularlo.*
- `BootstrapStatistics` (line 1135) - *Generador de estadísticas con intervalos de confianza bootstrap.

Calcula:
- Curvas ROC con IC del 95%
- AUC con IC del 95%
- Kappa de Cohen con IC del 95%*
- `VisualizationGenerator` (line 1281) - *Generador de visualizaciones con estilo académico.*
- `ExperimentOrchestrator` (line 1646) - *Orquestador principal para todos los experimentos.

Coordina:
1. Carga de checkpoint grokkeado
2. Verificación de grokking
3. Experimento de Local Complexity
4. Protocolo de poda iterativa
5. Análisis ROC/AUC
6. Generación de visualizaciones*

**Functions:**
- `find_grokked_checkpoint` (line 2313) - *Buscar checkpoint grokkeado en múltiples ubicaciones.

Returns:
    Path al archivo de checkpoint grokkeado*
- `analyze_checkpoints` (line 2353) - *Analizar todos los checkpoints disponibles para encontrar el grokkeado.

Returns:
    Diccionario con métricas de cada checkpoint*
- `main` (line 2423) - *Punto de entrada principal.*
- `__post_init__` (line 115)
- `__init__` (line 139)
- `_initialize_weights` (line 150) - *Inicializar pesos con consideración para grokking.*
- `forward` (line 156) - *Computar A @ B usando descomposición tensorial.

Args:
    A: Tensor de entrada de forma (batch, 2, 2)
    B: Tensor de entrada de forma (batch, 2, 2)
    
Returns:
    Tensor de salida de forma (batch, 2, 2)*
- `slot_importance` (line 183) - *Importancia de cada slot basada en normas.*
- `count_active` (line 190) - *Contar slots activos.*
- `compute_SP` (line 194) - *Métrica de Sparsity. SP -> 0 significa máxima sparsity.*
- `get_state_dict` (line 202) - *Obtener estado completo para checkpointing.*
- `load_state_dict` (line 211) - *Cargar estado completo desde checkpoint.*
- `__init__` (line 228)
- `generate_matrix` (line 238) - *Generar matriz aleatoria con valores enteros.*
- `generate_data` (line 242) - *Generar pares de matrices y sus productos.*
- `get_train_test` (line 260) - *Dividir en conjuntos de entrenamiento y prueba.*
- `__init__` (line 287)
- `compute_lc` (line 292) - *Calcular LC para un batch específico.

LC = ||g||^2 / N_batch

Donde g es el gradiente de la pérdida respecto a los pesos.*
- `compute_batch_diversity` (line 326) - *Calcular diversidad del batch basada en varianza de activaciones.*
- `__init__` (line 343)
- `verify` (line 347) - *Verificar que el operador ha grokkeado correctamente.

Returns:
    Tupla de (éxito, métricas)*
- `_generate_batch` (line 393) - *Generar batch de matrices aleatorias.*
- `__init__` (line 420)
- `get_weight_magnitudes` (line 426) - *Obtener magnitud absoluta de todos los pesos.*
- `compute_sparsity` (line 431) - *Calcular porcentaje de pesos en cero.*
- `prune_percent` (line 437) - *Podar el porcentaje especificado de pesos menos importantes.

Returns: (num_pruned, current_sparsity)*
- `fine_tune` (line 457) - *Fine-tune del modelo podado con métricas completas.*
- `_generate_batch` (line 510) - *Generar batch de matrices aleatorias.*
- `run_protocol` (line 624) - *Ejecutar protocolo completo de poda iterativa.

Returns:
    Diccionario con resultados completos*
- `__init__` (line 775)
- `run_full_experiment` (line 779) - *Ejecutar experimento completo de LC entrenando desde cero.*
- `_generate_batch` (line 882)
- `__init__` (line 903)
- `run_balanced_experiments` (line 907) - *Ejecutar multiples runs con condiciones disenhadas para producir mix.

Returns:
    Diccionario con resultados de todos los runs*
- `_train_single_run` (line 1021) - *Entrenar un solo modelo con configuración específica.*
- `_compute_roc` (line 1099) - *Calcular ROC/AUC básico.*
- `_generate_batch` (line 1127)
- `__init__` (line 1145)
- `compute_roc_with_ci` (line 1149) - *Calcular curva ROC con intervalos de confianza bootstrap.

Returns:
    Diccionario con resultados ROC*
- `compute_kappa_with_ci` (line 1236) - *Calcular Kappa de Cohen con IC bootstrap.*
- `compute_accuracy_with_ci` (line 1260) - *Calcular accuracy con IC binomial.*
- `__init__` (line 1284)
- `plot_local_complexity` (line 1298) - *Graficar evolución de Local Complexity y Accuracy.*
- `plot_pruning_results` (line 1342) - *Graficar resultados de poda iterativa.*
- `plot_roc_with_ci` (line 1404) - *Graficar curva ROC con intervalos de confianza.*
- `plot_balanced_runs_results` (line 1467) - *Graficar resultados del experimento de runs balanceados.*
- `plot_discretization_results` (line 1556) - *Graficar resultados de discretizacion.*
- `__init__` (line 1659)
- `find_grokked_checkpoint` (line 1689) - *Buscar checkpoint grokkeado en múltiples ubicaciones.

Returns:
    Path al archivo de checkpoint o None si no se encuentra*
- `load_grokked_checkpoint` (line 1733) - *Cargar checkpoint grokkeado y verificar que grokkeó.

Returns:
    Modelo cargado y verificado*
- `verify_checkpoint_is_grokked` (line 1768) - *Verificar que el checkpoint cargado realmente grokkeó.

Returns:
    Tupla de (es_grokked, métricas)*
- `run_local_complexity_experiment` (line 1795) - *Ejecutar experimento de Local Complexity vs Época.

Nota: Como usamos un checkpoint ya grokked, esto mide la LC
durante el fine-tuning post-poda, no durante el grokking inicial.

Returns:
    Diccionario con historial de LC y accuracy*
- `run_lc_training_experiment` (line 1897) - *Ejecutar experimento de Local Complexity .

- Entrenar un modelo desde cero hasta grokking
- Medir LC en cada época para capturar la transición de fase
- Si LC muestra un cambio alrededor del grokking, la métrica es útil
- Si LC permanece constante, la métrica NO captura la transición

Returns:
    Diccionario con historial completo del experimento*
- `run_pruning_experiment` (line 1936) - *Ejecutar protocolo de poda iterativa + fine-tuning.

Returns:
    Diccionario con resultados de poda*
- `run_balanced_runs_experiment` (line 1980) - *Ejecutar experimento de runs balanceados (PUNTO C DEL REVISOR).

Esto es CRUCIAL para obtener un AUC valido:
- Entrenar multiples modelos con diferentes hiperparametros
- Algunos grokkean, otros no
- Generar dataset balanceado para ROC/AUC

Returns:
    Diccionario con resultados de todos los runs*
- `run_roc_analysis` (line 2043) - *Ejecutar análisis ROC/AUC con bootstrap.

Returns:
    Diccionario con resultados ROC*
- `_generate_batch` (line 2117) - *Generar batch de matrices aleatorias.*
- `generate_summary_report` (line 2124) - *Generar reporte de resumen en markdown.*
- `save_results` (line 2197) - *Guardar todos los resultados.*
- `run_all_experiments` (line 2235) - *Ejecutar suite completa de experimentos.*

#### `generate_figures.py`
**Path:** `experiments/generate_figures.py`

**Functions:**
- `setup_matplotlib_for_plotting` (line 15) - *Configure matplotlib and seaborn for proper rendering.*
- `generate_benchmark_figure` (line 63) - *Generate benchmark performance comparison plot.*
- `generate_ablation_figure` (line 124) - *Generate ablation study visualization.*
- `load_checkpoint_weights` (line 219) - *Load all checkpoint files and extract weight tensors.*
- `generate_weight_geometry_figure` (line 258) - *Generate weight space geometry visualization.*
- `generate_phase_transition_figure` (line 354) - *Generate phase transition analysis from checkpoint evolution.*
- `generate_coherence_figure` (line 465) - *Generate cache coherence analysis visualization.*
- `generate_crystallization_figure` (line 534) - *Visualize the crystallization of Strassen coefficients.*
- `main` (line 623)

#### `coherence_analysis.py`
**Path:** `experiments/statistics/coherence_analysis.py`

**Functions:**
- `strassen_numpy` (line 15)
- `run_coherence_analysis` (line 42)

#### `rigorous_experiment.py`
**Path:** `experiments/statistics/rigorous_experiment.py`

**Classs:**
- `ExperimentConfig` (line 58) - *Complete hyperparameter specification for reproducibility*
- `ExperimentResult` (line 84) - *Single experiment result*
- `StrassenModel` (line 106) - *Strassen-like bilinear model*

**Functions:**
- `generate_data` (line 131) - *Generate matrix multiplication dataset*
- `compute_discretization_error` (line 143) - *Compute mean distance to nearest discrete value*
- `compute_spectral_gap` (line 157) - *Compute maximum spectral gap ratio*
- `run_single_experiment` (line 169) - *Run a single controlled experiment*
- `run_full_experiment` (line 269) - *Run complete factorial experiment*
- `perform_anova` (line 306) - *Perform full factorial ANOVA

Returns complete ANOVA table with SS, df, MS, F, p, η²*
- `print_anova_table` (line 401) - *Print formatted ANOVA table*
- `fit_noise_model` (line 448) - *Fit theoretical noise model:
Var(loss) = α/B + β·cache_miss(B) + γ

Compare to null model: Var(loss) = α/B + γ*
- `find_optimal_B` (line 519) - *Find optimal batch size with bootstrap confidence interval*
- `generate_report` (line 555) - *Generate complete statistical report*
- `__init__` (line 108)
- `forward` (line 124)
- `cache_miss_proxy` (line 467)
- `full_model` (line 472)
- `null_model` (line 476)
- `get_mean_error` (line 526)

#### `benchmark.py`
**Path:** `experiments/validation/benchmark.py`

**Functions:**
- `strassen_numpy` (line 15) - *Strassen recursivo con NumPy para productos base.*
- `measure_single_sgemm` (line 49) - *Mide tiempo de un solo sgemm de tamaño n.*
- `run_planck_analysis` (line 59) - *Ejecuta el análisis del Límite de Planck.*

#### `validation_experiments.py`
**Path:** `experiments/validation_experiments.py`

**Functions:**
- `strassen_2x2` (line 47) - *Compute 2x2 matrix multiplication using Strassen coefficients.*
- `strassen_recursive` (line 56) - *Recursive Strassen for NxN matrices.*
- `test_uniqueness_via_permutation` (line 85) - *Test that permuting slots produces equivalent computation.*
- `test_noise_stability` (line 125) - *Test stability under Gaussian noise.*
- `test_expansion_sizes` (line 162) - *Test expansion to larger sizes.*
- `simulate_grokking_dynamics` (line 184) - *Simulate grokking dynamics for visualization.*
- `compute_cache_math` (line 250) - *Compute L3 cache requirements for different batch sizes.*
- `main` (line 294) - *Run all validation experiments.*
- `convert_types` (line 313)

#### `verify_checkpoints.py`
**Path:** `experiments/verify_checkpoints.py`

**Classs:**
- `StrassenBilinear` (line 25)

**Functions:**
- `compute_delta` (line 51)
- `verify_2x2` (line 68)
- `strassen_expand` (line 89)
- `verify_expansion` (line 126)
- `compute_S_theta` (line 152)
- `load_checkpoint` (line 166)
- `verify_checkpoint` (line 183)
- `run_noise_stability_test` (line 226)
- `main` (line 249)
- `__init__` (line 27)
- `forward` (line 34)
- `get_discrete_coefficients` (line 44)

#### `experimetn2.py`
**Path:** `experimetn2.py`

**Classs:**
- `StrassStrassenConfig` (line 53) - *Immutable canonical configuration for the Strassen bilinear model.*
- `TrainingConfig` (line 71) - *Immutable training hyperparameters for crystallization runs.*
- `SuiteConfig` (line 84) - *Top-level suite configuration orchestrator.*
- `StrassStrassenModel` (line 95) - *Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.
Implements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.
U, V in R^{rank x input_dim}, W in R^{output_dim x rank}.*
- `ComplexStrassStrassenModel` (line 135) - *Genuinely complex-valued bilinear model for Altland-Zirnbauer spectral testing.
Uses complex parameters and arithmetic to break time-reversal symmetry.*
- `StrassenDataGenerator` (line 178) - *Generates random 2x2 matrix pairs and their exact products.*
- `CheckpointManager` (line 190) - *Handles serialization and metadata tracking of model checkpoints.*
- `LevelSpacingRatioCalculator` (line 207) - *Computes adjacent gap ratio 'r' of eigenvalues to determine spectral class.*
- `ExactHessianCalculator` (line 244) - *Computes the mathematically exact full-rank Hessian of the model loss.*
- `SyntheticPlanckCalculator` (line 278) - *Computes the synthetic Planck constant from parameter and loss statistics.*
- `SuperpositionMetricCalculator` (line 328) - *Measures representation density via Sparse Autoencoder (SAE) bottleneck analysis.*
- `IExperiment` (line 385) - *Abstract interface defining the execution protocol for experiments.*
- `Experiment1RicciMBLDuality` (line 393) - *Experiment 1: Ricci-MBL Duality.
Tracks geometric curvature (Hessian Ricci scalar) against adjacent gap ratio
during a long-duration optimization trajectory to capture crystallization.*
- `Experiment2AltlandZirnbauer` (line 457) - *Experiment 2: Altland-Zirnbauer Symmetry Dial.
Uses complex parameters and arithmetic to trigger a true symmetry crossover
from real symmetric ensembles (GOE) to complex Hermitian (GUE).*
- `Experiment3ConformalIsomorphism` (line 520) - *Experiment 3: Conformal Isomorphism.
Conducts mathematical stress-testing of fractional-linear Möbius
transformations vs scale-conformal transformations to map the physical
limits of learned tensor models.*
- `Experiment4CompressionFrontier` (line 570) - *Experiment 4: Compression Frontier.
Tests the thermodynamic bound between parameter uncertainty (hbar) and
representation superposition (psi) under varied weight decay levels.*
- `Experiment5HolographicPruning` (line 624) - *Experiment 5: Holographic Pruning.
Distinguishes volumetric representation structures from structured boundary-state
mechanisms via element-wise versus slot-wise pruning.*
- `UnifiedSuite` (line 699) - *Orchestrates and structures the execution of the five experiments.*

**Functions:**
- `main` (line 745)
- `__post_init__` (line 63)
- `__init__` (line 101)
- `forward` (line 115)
- `get_coefficients` (line 125)
- `slot_importance` (line 128)
- `__init__` (line 140)
- `get_complex_tensors` (line 153)
- `forward` (line 160)
- `__init__` (line 180)
- `generate_batch` (line 183)
- `save` (line 192)
- `__init__` (line 209)
- `calculate_r_ratio` (line 212)
- `__init__` (line 246)
- `compute_hessian` (line 249)
- `__init__` (line 280)
- `calculate` (line 283)
- `__init__` (line 330)
- `calculate` (line 333)
- `run` (line 388)
- `get_name` (line 390)
- `__init__` (line 399)
- `get_name` (line 405)
- `run` (line 408)
- `__init__` (line 463)
- `get_name` (line 468)
- `run` (line 471)
- `__init__` (line 527)
- `get_name` (line 531)
- `run` (line 534)
- `__init__` (line 576)
- `get_name` (line 582)
- `run` (line 585)
- `__init__` (line 630)
- `get_name` (line 634)
- `run` (line 637)
- `__init__` (line 701)
- `execute_all` (line 712)
- `loss_fn` (line 254)

#### `fermi.py`
**Path:** `fermi.py`

**Classs:**
- `FermiConfig` (line 19)
- `IModel` (line 49)
- `IBlochWaveConstructor` (line 54)
- `IBandStructureCalculator` (line 59)
- `IFermiLevelCalculator` (line 64)
- `IDensityOfStatesCalculator` (line 69)
- `IElectronicPropertiesCalculator` (line 74)
- `IMetalInsulatorClassifier` (line 79)
- `BilinearModel` (line 83)
- `BlochWaveConstructor` (line 111)
- `BandStructureCalculator` (line 135)
- `FermiLevelCalculator` (line 215)
- `DensityOfStatesCalculator` (line 285)
- `ElectronicPropertiesCalculator` (line 306)
- `MetalInsulatorClassifier` (line 368)
- `CheckpointMigrator` (line 408)
- `FermiLevelAnalyzer` (line 458)
- `FermiPipeline` (line 608)

**Functions:**
- `main` (line 766)
- `get_coefficients` (line 50)
- `construct` (line 55)
- `calculate` (line 60)
- `calculate` (line 65)
- `calculate` (line 70)
- `calculate` (line 75)
- `classify` (line 80)
- `__init__` (line 84)
- `_initialize` (line 95)
- `forward` (line 100)
- `get_coefficients` (line 103)
- `__init__` (line 112)
- `construct` (line 115)
- `__init__` (line 136)
- `calculate` (line 140)
- `_calculate_band_gap` (line 171)
- `_calculate_effective_masses` (line 188)
- `_is_direct_gap` (line 208)
- `__init__` (line 216)
- `calculate` (line 219)
- `_calculate_chemical_potential` (line 240)
- `_find_chemical_potential_iterative` (line 253)
- `_fermi_dirac` (line 273)
- `__init__` (line 286)
- `calculate` (line 289)
- `_gaussian` (line 302)
- `__init__` (line 307)
- `calculate` (line 310)
- `_calculate_kinetic_energy` (line 337)
- `_calculate_electronic_pressure` (line 347)
- `_calculate_compressibility` (line 357)
- `__init__` (line 369)
- `classify` (line 372)
- `classify_transport` (line 386)
- `migrate` (line 409)
- `_migrate_dict` (line 419)
- `_migrate_custom_format` (line 428)
- `_migrate_coefs_format` (line 447)
- `_migrate_standard_format` (line 454)
- `__init__` (line 459)
- `_load_checkpoint` (line 472)
- `analyze` (line 495)
- `_print_report` (line 550)
- `__init__` (line 609)
- `process_checkpoint` (line 612)
- `process_directory` (line 626)
- `generate_summary` (line 653)
- `_generate_text_report` (line 686)
- `plot_band_structures` (line 724) - *Generate comparison plots of band structures across checkpoints.*

#### `full_seed_prospector.py`
**Path:** `full_seed_prospector.py`

**Classs:**
- `ExecutionMode` (line 46)
- `UnifiedConfig` (line 52) - *Immutable unified configuration for all execution modes.*
- `IMetricCalculator` (line 162) - *Interface for metric calculation strategies.*
- `ILossComponent` (line 170) - *Interface for loss function components.*
- `ICheckpointManager` (line 179) - *Interface for checkpoint management.*
- `ITrainingPhase` (line 195) - *Interface for training phase execution.*
- `StrassenOperator` (line 203) - *Spectral operator for 2x2 matrix multiplication with Strassen structure.*
- `DeltaCalculator` (line 313) - *Calculate delta (discretization margin) metric.*
- `AccuracyCalculator` (line 333) - *Calculate matrix multiplication accuracy.*
- `KappaCalculator` (line 371) - *Calculate kappa (gradient covariance condition number).*
- `PerelmanEntropyCalculator` (line 455) - *Calculate Perelman W-entropy with adaptive tau.*
- `SparsityCalculator` (line 553) - *Calculate sparsity metrics.*
- `GradientMetricsCalculator` (line 573) - *Calculate gradient statistics.*
- `ResilienceSpectrometer` (line 599) - *Measure structural stability under progressive pruning.*
- `ComprehensiveMetricsAggregator` (line 694) - *Aggregate all metrics including LC, SP, kappa, delta, h_bar_eff, T_eff.*
- `AdaptiveQuantizationLoss` (line 802) - *Adaptive quantization loss pushing towards {-1, 0, 1}.*
- `RicciCurvaturePenalty` (line 854) - *Ricci scalar curvature penalty.*
- `GeometricLossAggregator` (line 868) - *Aggregate geometric loss components.*
- `CheckpointManager` (line 884) - *Manage model checkpointing.*
- `MatrixDataGenerator` (line 930) - *Generate random matrix data for training.*
- `DynamicBatchSizeScheduler` (line 947) - *Schedule batch size with cosine annealing.*
- `GlassDetector` (line 969) - *Detect glass state (non-crystallizing seeds).*
- `ProspectorPhase` (line 1025) - *Fast prospecting phase to identify crystal seeds.*
- `LongTrainingPhase` (line 1152) - *Long training phase with full thermodynamic metrics.*
- `ProgressiveSparsificationPhase` (line 1335) - *Progressive sparsification guided by slot importance.*
- `CoefficientDiscretizationPhase` (line 1442) - *Discretize coefficients to {-1, 0, 1}.*
- `StrassenVerifier` (line 1532) - *Verify Strassen algorithm correctness.*
- `CanonicalStrassenProvider` (line 1588) - *Provide canonical Strassen algorithm coefficients.*
- `SeedProspector` (line 1623) - *Prospect seeds for crystallization candidates.*
- `LongTrainingPipeline` (line 1742) - *Long training pipeline with all phases.*
- `LocalComplexityCalculator` (line 1849) - *Calculate Local Complexity as defined in the paper.*
- `SuperpositionCalculator` (line 1915) - *Calculate Superposition metrics using sparse autoencoder.*
- `ThermodynamicMetricsCalculator` (line 1964) - *Calculate thermodynamic metrics: h_bar_eff and T_eff.*

**Functions:**
- `main` (line 2022) - *Main entry point.*
- `calculate` (line 166)
- `compute` (line 174)
- `save` (line 183)
- `load` (line 187)
- `should_checkpoint` (line 191)
- `execute` (line 199)
- `__init__` (line 206)
- `_initialize_strassen_structure` (line 220) - *Initialize with bias towards canonical Strassen structure.*
- `forward` (line 276) - *Forward pass computing C = A @ B via low-rank factorization.*
- `slot_importance` (line 291) - *Calculate importance of each rank slot.*
- `count_active` (line 298) - *Count active slots above threshold.*
- `get_flat_parameters` (line 304) - *Get flattened parameter vector.*
- `get_parameter_count` (line 308) - *Get total parameter count.*
- `__init__` (line 316)
- `calculate` (line 319) - *Calculate delta: mean squared distance to {-1, 0, 1}.*
- `__init__` (line 336)
- `calculate` (line 339) - *Calculate accuracy percentage.*
- `__init__` (line 374)
- `accumulate_gradient` (line 379) - *Accumulate current gradient vector.*
- `calculate_kappa` (line 393) - *Calculate condition number of gradient covariance matrix.*
- `get_kappa_trend` (line 425) - *Determine kappa trend direction.*
- `is_crystallizing` (line 439) - *Detect if system is in crystallization phase.*
- `reset` (line 449) - *Reset calculator state.*
- `__init__` (line 458)
- `calculate` (line 465) - *Calculate W-entropy with adaptive tau coupling to GNS.*
- `_calculate_log_W` (line 522) - *Calculate log(W) with numerical stability.*
- `reset` (line 546) - *Reset calculator state.*
- `__init__` (line 556)
- `calculate` (line 559) - *Calculate active slots and sparsity.*
- `calculate` (line 577) - *Calculate gradient norm statistics.*
- `__init__` (line 602)
- `measure` (line 605) - *Measure resilience via progressive magnitude pruning.*
- `_prune_by_magnitude` (line 684) - *Prune parameters by magnitude threshold.*
- `__init__` (line 697)
- `compute_all` (line 711) - *Compute all available metrics.*
- `accumulate_gradient` (line 787) - *Accumulate gradient for kappa calculation.*
- `update_lr` (line 791) - *Update current learning rate.*
- `reset` (line 795) - *Reset all calculators.*
- `__init__` (line 805)
- `compute` (line 808) - *Compute quantization loss with adaptive weighting.*
- `_get_adaptive_weight` (line 835) - *Calculate adaptive weight based on epoch and kappa.*
- `__init__` (line 857)
- `compute` (line 860) - *Compute Ricci curvature penalty.*
- `__init__` (line 871)
- `compute` (line 876) - *Compute total geometric loss.*
- `__init__` (line 887)
- `save` (line 893) - *Save checkpoint to disk.*
- `load` (line 911) - *Load checkpoint from disk.*
- `should_checkpoint` (line 919) - *Check if checkpoint interval has elapsed.*
- `get_latest_checkpoint_path` (line 924) - *Get path to latest checkpoint if exists.*
- `__init__` (line 933)
- `generate_batch` (line 938) - *Generate batch of random matrices.*
- `__init__` (line 950)
- `get_batch_size` (line 953) - *Get batch size for current epoch.*
- `__init__` (line 972)
- `should_stop` (line 977) - *Determine if training should stop due to glass state.*
- `__init__` (line 1028)
- `execute` (line 1037) - *Execute prospecting phase with all metrics visible.*
- `__init__` (line 1155)
- `execute` (line 1164) - *Execute long training phase.*
- `__init__` (line 1338)
- `execute` (line 1342) - *Execute sparsification phase.*
- `_final_refinement` (line 1407) - *Final refinement after pruning.*
- `__init__` (line 1445)
- `execute` (line 1448) - *Execute discretization phase.*
- `__init__` (line 1535)
- `verify` (line 1538) - *Verify algorithm on random test matrices.*
- `get_canonical` (line 1592) - *Return canonical Strassen algorithm matrices.*
- `__init__` (line 1626)
- `prospect` (line 1631) - *Prospect multiple seeds for crystals.*
- `_set_seed` (line 1733) - *Set random seed for reproducibility.*
- `__init__` (line 1745)
- `_signal_handler` (line 1758) - *Handle interrupt signal.*
- `_set_seed` (line 1763) - *Set random seed for reproducibility.*
- `run` (line 1771) - *Run complete training pipeline.*
- `__init__` (line 1852)
- `calculate` (line 1855) - *Calculate LC as effective local dimensionality (paper definition).
Compatible fraction approach → log(volume) can be negative.*
- `__init__` (line 1918)
- `_initialize_sae` (line 1923) - *Initialize sparse autoencoder if not already done.*
- `calculate` (line 1930) - *Calculate superposition coefficient psi and effective features F.*
- `__init__` (line 1967)
- `calculate` (line 1970) - *Calculate effective Planck constant and temperature.*

#### `grain.py`
**Path:** `grain.py`

**Classs:**
- `StrassenConfig` (line 25)
- `IModel` (line 65)
- `IGrainBoundaryDetector` (line 71)
- `ILayerAnalyzer` (line 76)
- `IDislocationCalculator` (line 81)
- `IDomainFragmentationAnalyzer` (line 86)
- `ICheckpointManager` (line 91)
- `ITrainingMonitor` (line 97)
- `BilinearStrassenModel` (line 102)
- `LayerAnalyzer` (line 130)
- `GrainBoundaryDetector` (line 153)
- `DomainFragmentationAnalyzer` (line 243)
- `CheckpointManager` (line 309)
- `TrainingMetricsTracker` (line 340)
- `StrassenTrainer` (line 407)
- `GrainBoundaryAnalyzer` (line 547)
- `GrainBoundaryPipeline` (line 726)

**Functions:**
- `run_training` (line 833)
- `run_analysis` (line 838)
- `main` (line 845)
- `to_dict` (line 57)
- `forward` (line 66)
- `get_coefficients` (line 67)
- `detect` (line 72)
- `analyze_layer` (line 77)
- `calculate` (line 82)
- `analyze` (line 87)
- `save` (line 92)
- `load` (line 93)
- `update` (line 98)
- `should_checkpoint` (line 99)
- `__init__` (line 103)
- `_initialize_symmetric` (line 114)
- `forward` (line 119)
- `get_coefficients` (line 122)
- `analyze_layer` (line 131)
- `__init__` (line 154)
- `detect` (line 158)
- `_prune_model` (line 185)
- `_calculate_dislocation` (line 194)
- `_analyze_fragmentation` (line 227)
- `__init__` (line 244)
- `analyze` (line 248)
- `_calculate_coordination_loss` (line 269)
- `_estimate_domain_count` (line 282)
- `_calculate_coherence_length` (line 297)
- `__init__` (line 310)
- `save` (line 314)
- `load` (line 332)
- `should_save` (line 335)
- `__init__` (line 341)
- `update` (line 358)
- `get_current_metrics` (line 382)
- `get_training_bar_string` (line 389)
- `__init__` (line 408)
- `_setup_signal_handlers` (line 438)
- `_signal_handler` (line 442)
- `_generate_batch` (line 447)
- `_compute_accuracy` (line 468)
- `_save_checkpoint` (line 473)
- `train` (line 486)
- `__init__` (line 548)
- `_load_checkpoint` (line 557)
- `_migrate_checkpoint` (line 580)
- `_migrate_dict` (line 590)
- `_migrate_custom_format` (line 599)
- `_migrate_coefs_format` (line 618)
- `_migrate_standard_format` (line 625)
- `analyze` (line 628)
- `_analyze_dislocation_evolution` (line 658)
- `_find_critical_pruning_level` (line 681)
- `_print_report` (line 687)
- `__init__` (line 727)
- `process_checkpoint` (line 730)
- `process_directory` (line 744)
- `generate_summary` (line 771)
- `_generate_text_report` (line 797)

#### `gravity.py`
**Path:** `gravity.py`

**Classs:**
- `ThermodynamicConfig` (line 24)
- `IModel` (line 59)
- `IOrderParameterCalculator` (line 65)
- `IEntropyCalculator` (line 70)
- `ISpecificHeatCalculator` (line 75)
- `IGravitationalConstantCalculator` (line 80)
- `ILandauerConstantCalculator` (line 85)
- `IHeisenbergUncertaintyCalculator` (line 90)
- `ILocalComplexityCalculator` (line 95)
- `IBasinStabilityCalculator` (line 100)
- `IZeroShotTransferCalculator` (line 105)
- `IConditionNumberCalculator` (line 110)
- `BilinearModel` (line 114)
- `OrderParameterCalculator` (line 142)
- `ConfigurationEntropyCalculator` (line 154)
- `SpecificHeatCalculator` (line 171)
- `GravitationalConstantCalculator` (line 183)
- `LandauerConstantCalculator` (line 239)
- `HeisenbergUncertaintyCalculator` (line 271)
- `LocalComplexityCalculator` (line 310)
- `BasinStabilityCalculator` (line 327)
- `ZeroShotTransferCalculator` (line 365)
- `ConditionNumberCalculator` (line 398)
- `PhaseTransitionDetector` (line 454)
- `ThermodynamicAnalyzer` (line 505)
- `ThermodynamicPipeline` (line 837)

**Functions:**
- `main` (line 1114)
- `forward` (line 60)
- `get_coefficients` (line 61)
- `calculate` (line 66)
- `calculate` (line 71)
- `calculate` (line 76)
- `calculate` (line 81)
- `calculate` (line 86)
- `calculate` (line 91)
- `calculate` (line 96)
- `calculate` (line 101)
- `calculate` (line 106)
- `calculate` (line 111)
- `__init__` (line 115)
- `_initialize` (line 126)
- `forward` (line 131)
- `get_coefficients` (line 134)
- `__init__` (line 143)
- `calculate` (line 146)
- `__init__` (line 155)
- `calculate` (line 158)
- `__init__` (line 172)
- `calculate` (line 175)
- `__init__` (line 184)
- `calculate` (line 187)
- `__init__` (line 240)
- `calculate` (line 243)
- `__init__` (line 272)
- `calculate` (line 275)
- `__init__` (line 311)
- `calculate` (line 314)
- `__init__` (line 328)
- `calculate` (line 331)
- `_prune_model` (line 355)
- `__init__` (line 366)
- `calculate` (line 369)
- `_kronecker_recursive` (line 391)
- `__init__` (line 399)
- `calculate` (line 403)
- `__init__` (line 455)
- `detect` (line 458)
- `__init__` (line 506)
- `_load_checkpoint` (line 525)
- `_migrate_checkpoint` (line 548)
- `_migrate_dict` (line 560)
- `_migrate_custom_format` (line 569)
- `_migrate_coefs_format` (line 588)
- `_migrate_standard_format` (line 595)
- `_compute_static_gradient` (line 598)
- `_generate_test_data` (line 619)
- `analyze` (line 630)
- `_determine_failure_mode` (line 733)
- `_print_report` (line 744)
- `__init__` (line 838)
- `process_checkpoint` (line 841)
- `process_directory` (line 855)
- `generate_summary` (line 882)
- `_compute_emergent_constants` (line 904)
- `_verify_universal_laws` (line 959)
- `_compute_kappa_correlation` (line 993)
- `_generate_text_report` (line 1023)
- `extract_values` (line 908)

#### `grigori_perelmans_ricci_flow.py`
**Path:** `grigori_perelmans_ricci_flow.py`

**Classs:**
- `RicciConfig` (line 34) - *Immutable configuration for Ricci Flow analysis.*
- `BilinearStrassenModel` (line 82) - *Bilinear model f(A,B) = W((U*A) ⊙ (V*B)).*
- `StrassenDataGenerator` (line 128)
- `CheckpointMigrator` (line 145)
- `CustomFormatMigrator` (line 151)
- `StandardFormatMigrator` (line 166)
- `CheckpointMigrationManager` (line 172)
- `RicciFlowAnalyzer` (line 193) - *Calculates Ricci curvature metrics using Hessian as Metric Tensor proxy.
In Perelman's flow, dg/dt = -2Ric. 
Here we analyze instantaneous state of Metric (Hessian).*
- `SingularityEngine` (line 310) - *Identifies 'necks' (singularities) in the geometry and proposes 'surgery' (pruning).
A 'neck' is a parameter direction with extreme curvature (Hessian eigenvalue).*
- `GeometricPlanckCalculator` (line 366) - *Estimates effective Planck constant from Spectral Geometry.
Uses the Hessian eigenvalues to define an energy spectrum.*
- `RicciFlowAnalyzerPipeline` (line 430) - *Complete analysis pipeline for Ricci Flow and Planck Estimation.
Orchestrates Hessian computation, Curvature Analysis, and Physics metrics.*

**Functions:**
- `set_random_seed` (line 70)
- `main` (line 593)
- `__init__` (line 86)
- `_initialize` (line 94)
- `forward` (line 99)
- `get_coefficients` (line 102)
- `get_flat_params` (line 109) - *Return all parameters as a single flattened vector.*
- `set_flat_params` (line 114) - *Set model parameters from a flattened vector.*
- `generate_batch` (line 130)
- `can_migrate` (line 147)
- `migrate` (line 149)
- `can_migrate` (line 152)
- `migrate` (line 154)
- `can_migrate` (line 167)
- `migrate` (line 169)
- `__init__` (line 173)
- `migrate_checkpoint` (line 176)
- `__init__` (line 199)
- `compute_hessian` (line 203) - *Computes exact Hessian of loss w.r.t parameters.
H = d^2L / dtheta^2*
- `_loss_wrapper` (line 223) - *Wrapper to compute loss from flat param vector.*
- `_compute_diagonal_hessian` (line 240) - *Approximation: Diagonal of Hessian (Gauss-Newton).*
- `analyze_curvature` (line 247) - *Analyze Hessian spectrum to derive Ricci Scalar and Topological invariants.*
- `compute_heat_kernel_trace` (line 285) - *Trace of Heat Kernel: Z(t) = Sum( exp(-lambda_i * t) ).
Relates to Partition Function in Quantum Mechanics.*
- `compute_topological_entropy` (line 296) - *von Neumann Entropy / Spectral Entropy.
S = - Sum( p_i * log(p_i) ) where p_i are normalized eigenvalue weights.*
- `__init__` (line 316)
- `detect_necks` (line 325) - *Identify if the system is in a 'bottleneck' state.*
- `propose_surgery` (line 336) - *Propose parameters to 'cut' (prune) based on curvature heuristics.
In Strassen, the 'bias' slot (8th) often carries the 'noise' or singular connection.*
- `__init__` (line 371)
- `calculate` (line 376)
- `_get_spectral_gap` (line 412)
- `_compute_spectral_entropy` (line 420)
- `__init__` (line 435)
- `analyze_checkpoint` (line 439) - *Perform complete analysis of a single checkpoint.*
- `analyze_directory` (line 527)
- `_print_summary` (line 560)

#### `hawking_radiation.py`
**Path:** `hawking_radiation.py`

**Classs:**
- `CustomUnpickler` (line 36) - *Custom unpickler that handles unknown classes by creating dummy objects.

This solves the "Can't get attribute 'UnifiedConfig'" error by providing
fallback objects for any class that can't be found.*
- `HawkingConfiguration` (line 138) - *Immutable configuration for Hawking radiation analysis.*
- `IModel` (line 188)
- `BilinearStrassenModel` (line 197) - *Bilinear model for Strassen matrix multiplication.*
- `RobustCheckpointMigrator` (line 234) - *Enhanced checkpoint migrator that handles:
- Direct U, V, W tensors
- U_coefs format
- Standard .weight format
- Nested structures with config
- Encoder format
- State dicts within state dicts*
- `MetadataExtractor` (line 514) - *Extracts metadata from various checkpoint formats.*
- `GravitationalConstantCalculator` (line 609) - *Calculates effective gravitational constant G_alg.*
- `PlanckConstantCalculator` (line 676) - *Calculates effective Planck constant h_bar_eff.*
- `BoltzmannConstantCalculator` (line 768) - *Calculates effective Boltzmann constant k_B_eff.*
- `SpeedOfLightCalculator` (line 823) - *Calculates effective speed of light c_eff.*
- `InformationalMassCalculator` (line 883) - *Calculates effective mass M_eff.*
- `HorizonAreaCalculator` (line 939) - *Calculates effective area A_eff.*
- `HawkingRadiationCalculator` (line 991) - *Calculates Hawking radiation metrics.*
- `RobustHawkingAnalyzer` (line 1166) - *Robust analyzer that handles exotic checkpoint formats.*
- `DummyClass` (line 64)

**Functions:**
- `load_checkpoint_robust` (line 93) - *Load checkpoint with robust handling of custom classes.

Tries multiple loading strategies in order:
1. Standard torch.load
2. Custom unpickler
3. Weights_only mode with manual extraction*
- `main` (line 1406)
- `find_class` (line 44) - *Override find_class to handle missing classes gracefully.*
- `_create_dummy_class` (line 62) - *Create a dummy class that can hold attributes.*
- `get_effective_input_dim` (line 172)
- `get_total_parameters` (line 175)
- `get_coefficients` (line 189)
- `forward` (line 190)
- `__init__` (line 200)
- `_initialize` (line 211)
- `forward` (line 216)
- `get_coefficients` (line 219)
- `get_flat_parameters` (line 226)
- `migrate` (line 245) - *Main migration entry point with multiple strategies.*
- `_try_extract_state_dict` (line 269) - *Try standard state dict extraction methods.*
- `_try_nested_extraction` (line 288) - *Try to extract from nested structures.*
- `_try_direct_tensor_extraction` (line 313) - *Try to extract tensors directly from any structure.*
- `_reconstruct_from_tensors` (line 338) - *Reconstruct U, V, W from found tensors.*
- `_is_state_dict` (line 386) - *Check if dict looks like a state dict.*
- `_migrate_dict` (line 398) - *Migrate a state dict to the expected format.*
- `_migrate_custom_format` (line 424)
- `_migrate_coefs_format` (line 449)
- `_migrate_standard_format` (line 456)
- `_migrate_encoder_format` (line 466) - *Handle encoder.layers style checkpoints.*
- `_migrate_prefixed_format` (line 496) - *Handle prefixed state dict keys.*
- `extract` (line 518) - *Extract all relevant metadata from checkpoint.*
- `_extract_delta` (line 571) - *Recursively search for delta in nested structures.*
- `__init__` (line 612)
- `calculate` (line 615)
- `__init__` (line 679)
- `calculate` (line 682)
- `__init__` (line 771)
- `calculate` (line 774)
- `__init__` (line 826)
- `calculate` (line 829)
- `__init__` (line 886)
- `calculate` (line 889)
- `__init__` (line 942)
- `calculate` (line 945)
- `__init__` (line 994)
- `calculate_all` (line 1003) - *Calculate all Hawking radiation metrics.*
- `_classify_state` (line 1151)
- `__init__` (line 1169)
- `analyze_checkpoint` (line 1174) - *Analyze a single checkpoint with robust error handling.*
- `_compute_gradient` (line 1227) - *Compute gradient on random batch.*
- `_print_report` (line 1255) - *Print formatted report.*
- `analyze_directory` (line 1295) - *Analyze all checkpoints in directory.*
- `_generate_summary` (line 1342) - *Generate aggregate summary.*
- `extract_tensors` (line 317)
- `__init__` (line 65)
- `__repr__` (line 72)
- `__getitem__` (line 75)
- `keys` (line 78)
- `values` (line 81)
- `items` (line 84)
- `get` (line 87)

#### `maxwell_strassen_analysis.py`
**Path:** `maxwell_strassen_analysis.py`

**Classs:**
- `MaxwellConfiguration` (line 42) - *Configuration for Maxwellian analysis of Strassen crystals.

Physics Parameters:
-------------------
LATTICE_CONSTANT: Spacing between weight nodes (arbitrary units, e.g., nm scale).
PERMITTIVITY_VACUUM: ε_0 (scaled for numerical stability).
PERMITTIVITY_WEIGHT_SCALE: Factor to convert weight magnitude to dielectric contrast.

Simulation Parameters:
----------------------
GRID_DIMENSION: Size of the cubic lattice for weight embedding (N x N x N).
FREQUENCY_SAMPLES: Number of frequency points for spectral analysis.
WAVEVECTOR_RANGE: Range of k-vectors for scattering simulation.

Analysis Thresholds:
--------------------
CRYSTALLINITY_THRESHOLD: Entropy threshold to classify as Crystal vs Glass.
BRAGG_PEAK_SHARPNESS: Minimum prominence for a peak to be considered Bragg scattering.*
- `IModel` (line 116)
- `IGeometryMapper` (line 121)
- `IMaxwellSolver` (line 126)
- `IDielectricAnalyzer` (line 132)
- `IPhaseClassifier` (line 137)
- `BilinearStrassenModel` (line 145)
- `CheckpointMigrator` (line 171)
- `StrassenGeometryMapper` (line 227) - *Maps abstract weight vectors to a 3D Dielectric Lattice.

Strategy:
1. Flatten U, V, W into a single vector.
2. Populate a 3D grid (NxNxN) using a space-filling curve (Z-order) 
   or layer-wise assignment.
3. Compute Effective Charge Density ρ and Permittivity ε.

Crystal Hypothesis:
- Crystallized weights (discrete values) form ordered arrays.
- Glass weights (random) form noise.*
- `DielectricTensorAnalyzer` (line 288) - *Analyzes the anisotropy of the dielectric medium.

Glass: Isotropic (ε ~ scalar)
Crystal: Anisotropic (ε ~ tensor with specific principal axes)*
- `MaxwellScatteringSolver` (line 350) - *Solves Maxwell's equations in the frequency domain via FFT.

Key Analysis:
1. Electrostatics: Solve ∇·(ε∇φ) = -ρ.
2. Scattering: Calculate Fourier Transform of Dielectric contrast.
   - Crystal: Sharp Bragg Peaks at k-vectors determined by lattice periodicity.
   - Glass: Broad diffuse scattering (Rayleigh).*
- `PhotonicEntropyCalculator` (line 466) - *Calculates the entropy of the electromagnetic field distribution.
S = -∑ p(E) log p(E)

Glass: High Entropy (Disordered field).
Crystal: Low Entropy (Ordered, localized modes).*
- `BandgapAnalyzer` (line 498) - *Estimates if a photonic bandgap exists.
A bandgap implies certain frequencies cannot propagate.

We use the Fourier coefficients to estimate the gap.*
- `CrystalPhaseClassifier` (line 544) - *Classifies the material phase (Crystal vs Glass) based on EM metrics.*
- `CheckpointManager` (line 605)
- `MaxwellVisualizer` (line 625)
- `MaxwellAnalyzer` (line 677)

**Functions:**
- `main` (line 788)
- `get_effective_input_dim` (line 103)
- `get_total_parameters` (line 106)
- `get_coefficients` (line 117)
- `map_weights_to_lattice` (line 122)
- `solve_poisson` (line 127)
- `compute_scattering` (line 128)
- `analyze_permittivity_tensor` (line 133)
- `classify` (line 138)
- `__init__` (line 146)
- `_initialize_weights` (line 156)
- `forward` (line 161)
- `get_coefficients` (line 164)
- `migrate` (line 172)
- `_migrate_dict` (line 185)
- `_migrate_custom` (line 194)
- `_migrate_coefs` (line 203)
- `_migrate_standard` (line 208)
- `_np` (line 217)
- `__init__` (line 241)
- `map_weights_to_lattice` (line 244) - *Returns:
    charge_density: 3D array of charge distribution.
    permittivity: 3D array of dielectric constants.*
- `__init__` (line 295)
- `analyze_permittivity_tensor` (line 298) - *Analyze the effective permittivity tensor of the medium.*
- `__init__` (line 360)
- `solve_poisson` (line 363) - *Solves Poisson equation for Electric Potential φ.
Using Spectral Method (FFT):
∇²φ = -ρ / ε_0

For variable ε (heterogeneous medium), this is approximate.
We use the convolution theorem for the Green's function.*
- `compute_scattering` (line 401) - *Compute the Scattering Amplitude S(k).
S(k) ∝ |FT(Δε)|²

Δε = ε(r) - ε_avg (Dielectric contrast)

Returns:
    Dict containing scattering intensity map and peak analysis.*
- `_find_peaks` (line 438) - *Detect sharp peaks indicative of crystallinity.*
- `__init__` (line 474)
- `calculate` (line 477)
- `__init__` (line 505)
- `analyze` (line 508)
- `__init__` (line 548)
- `classify` (line 551) - *Decision Logic:
1. If Purity Alpha > Threshold AND Discretization is high -> Pre-cursor.
2. If Anisotropy is High -> Crystal Order.
3. If Scattering shows Bragg Peaks -> Long Range Order.
4. If Entropy is Low -> Ordered System.*
- `__init__` (line 606)
- `should_save` (line 610)
- `save` (line 613)
- `__init__` (line 626)
- `visualize_lattice` (line 629)
- `visualize_scattering` (line 648)
- `visualize_potential` (line 659)
- `__init__` (line 678)
- `_calculate_purity_metrics` (line 690)
- `analyze_checkpoint` (line 701) - *Executes the full Maxwellian analysis pipeline on a single checkpoint.*
- `generate_report` (line 765)

#### `mbl_analyzer.py`
**Path:** `mbl_analyzer.py`

**Classs:**
- `MBLConfiguration` (line 24) - *Comprehensive configuration for MBL analysis of Strassen algorithm crystallization.
All parameters are centralized here following SOLID principles.*
- `IModel` (line 93) - *Protocol for models compatible with MBL analysis.*
- `ILevelSpacingCalculator` (line 100) - *Protocol for level spacing ratio calculation.*
- `IParticipationRatioCalculator` (line 106) - *Protocol for participation ratio calculation.*
- `ISyntheticPlanckCalculator` (line 112) - *Protocol for synthetic Planck's constant calculation.*
- `IDiscretizationDialAnalyzer` (line 118) - *Protocol for discretization dial analysis.*
- `ICheckpointManager` (line 124) - *Protocol for checkpoint management.*
- `ITrainingMetricsCollector` (line 132) - *Protocol for collecting all training metrics.*
- `BilinearStrassenModel` (line 138) - *Bilinear model for Strassen algorithm implementation.
Represents the 2x2 matrix multiplication with hidden dimension expansion.*
- `LevelSpacingRatioCalculator` (line 205) - *Calculates the level spacing ratio r for MBL phase detection.

The ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})
where delta_n = E_{n+1} - E_n (energy level spacing).

References:
- Oganesyan & Huse (2008): r_WD ≈ 0.53 (Wigner-Dyson, thermal)
- Poisson statistics: r_P ≈ 0.386 (MBL/localized phase)*
- `ParticipationRatioCalculator` (line 321) - *Calculates Inverse Participation Ratio (IPR) for localization analysis.

IPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.
IPR = 1 for fully localized state, IPR = 1/N for fully delocalized state.

Used to quantify the 'crystallinity' of the weight distribution.*
- `SyntheticPlanckConstantCalculator` (line 420) - *Calculates effective synthetic Planck's constant (hbar_eff) from model properties.

Based on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)
where PR is the Participation Ratio and Energy_Gap is the spectral gap.

This represents the quantum of action in the synthetic quantum system.*
- `DiscretizationDialAnalyzer` (line 486) - *Analyzes the discretization parameter delta as a phase transition control.

The discretization delta measures how close weights are to discrete values.
It acts as a "dial" that controls the quantum-classical transition.

This implements the noise robustness test: applying Gaussian perturbations
and measuring when the energy gap collapses (loss of quantum protection).*
- `PurityIndexCalculator` (line 614) - *Original purity calculation preserved exactly as in user's code.
Calculates the 'crystallinity' of the weight distribution.*
- `EffectiveTemperatureCalculator` (line 674) - *Original temperature calculation preserved exactly.*
- `PhaseClassifier` (line 721) - *Original phase classification preserved exactly.*
- `CheckpointMigrator` (line 746) - *Original checkpoint migration logic preserved exactly.*
- `MBLCheckpointManager` (line 800) - *Manages checkpoint saving with 5-minute intervals and latest file maintenance.*
- `MBLMetricsCollector` (line 855) - *Collects all MBL metrics for comprehensive training monitoring.*
- `MBLCheckpointAnalyzer` (line 962) - *Comprehensive analyzer for MBL metrics from checkpoints.*
- `MBLAnalysisPipeline` (line 1102) - *Main pipeline for processing checkpoints and generating reports.*

**Functions:**
- `main` (line 1227)
- `get_effective_input_dim` (line 84)
- `get_total_parameters` (line 87)
- `get_coefficients` (line 95)
- `forward` (line 96)
- `calculate` (line 102)
- `calculate` (line 108)
- `calculate` (line 114)
- `analyze_robustness` (line 120)
- `save_checkpoint` (line 126)
- `load_checkpoint` (line 128)
- `collect` (line 134)
- `__init__` (line 143)
- `_initialize_weights` (line 154) - *Xavier initialization with symmetry constraint for U and V.*
- `forward` (line 160) - *Forward pass implementing bilinear multiplication.*
- `get_coefficients` (line 164) - *Returns weight matrices for analysis.*
- `get_flat_parameters` (line 172) - *Returns all parameters flattened for Hamiltonian construction.*
- `construct_hessian_approximation` (line 179) - *Constructs approximate Hessian matrix from weight correlations.
This serves as the 'Hamiltonian' for MBL analysis.*
- `__init__` (line 217)
- `calculate` (line 220) - *Calculate level spacing statistics from model weights.

Returns:
    Dictionary containing mean ratio, variance, and phase classification.*
- `_construct_hessian_from_weights` (line 267) - *Alternative Hessian construction for generic models.*
- `_compute_eigenvalues` (line 283) - *Compute sorted eigenvalues of the Hamiltonian.*
- `_calculate_spacing_ratios` (line 288) - *Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).*
- `_classify_phase` (line 303) - *Classify the quantum phase based on level spacing ratio.*
- `__init__` (line 331)
- `calculate` (line 334) - *Calculate participation ratios for all weight layers.

Returns:
    Dictionary containing global and layer-wise IPR metrics.*
- `_calculate_ipr` (line 380) - *Calculate standard Inverse Participation Ratio.
IPR = sum_i |c_i|^4 / (sum_i |c_i|^2)^2*
- `_calculate_renyi_ipr` (line 395) - *Calculate q-th order Rényi IPR.
I_q = sum_i |c_i|^{2q} / (sum_i |c_i|^2)^q*
- `_calculate_fractal_dimension` (line 409) - *Calculate fractal dimension D_q from IPR.
IPR ~ N^{-D_q} => D_q = -log(IPR) / log(N)*
- `__init__` (line 430)
- `calculate` (line 433) - *Calculate synthetic Planck's constant.

Args:
    participation_ratio: Inverse participation ratio (measure of localization)
    energy_gap: Energy gap from spectrum (measure of quantum discreteness)

Returns:
    Synthetic hbar value representing the quantum scale of the system.*
- `calculate_from_model` (line 456) - *Comprehensive calculation from model and previous analyses.*
- `__init__` (line 497)
- `calculate_base_discretization` (line 501) - *Calculate the base discretization level from weight rounding error.*
- `analyze_robustness` (line 528) - *Test robustness by applying noise and measuring gap collapse.

Args:
    model: The neural network model
    noise_levels: Tuple of noise magnitudes to test

Returns:
    Dictionary containing robustness metrics and phase transition points.*
- `_perturb_and_measure` (line 584) - *Apply noise to model and measure resulting metrics.*
- `_delta_to_alpha` (line 607) - *Convert discretization error to purity alpha.*
- `__init__` (line 620)
- `calculate` (line 623)
- `_compute_layer_purity` (line 652)
- `_delta_to_alpha` (line 658)
- `_assess_purity_quality` (line 663)
- `__init__` (line 679)
- `calculate` (line 682)
- `__init__` (line 726)
- `classify` (line 729)
- `migrate` (line 751)
- `_migrate_dict` (line 761)
- `_migrate_custom_format` (line 770)
- `_migrate_coefs_format` (line 789)
- `_migrate_standard_format` (line 796)
- `__init__` (line 805)
- `should_save_checkpoint` (line 810) - *Check if 5 minutes have elapsed since last checkpoint.*
- `save_checkpoint` (line 816) - *Save checkpoint with all MBL metrics.*
- `load_checkpoint` (line 850) - *Load checkpoint with automatic device placement.*
- `__init__` (line 860)
- `collect` (line 870) - *Collect all metrics for the current training state.*
- `_classify_quantum_phase` (line 946) - *Classify the combined quantum phase.*
- `__init__` (line 967)
- `_load_checkpoint` (line 975) - *Load and migrate checkpoint to model.*
- `analyze` (line 998) - *Perform complete MBL analysis on checkpoint.*
- `_generate_summary` (line 1026) - *Generate executive summary of analysis.*
- `_print_report` (line 1043) - *Print formatted analysis report.*
- `__init__` (line 1107)
- `process_checkpoint` (line 1110) - *Process single checkpoint and save results.*
- `process_directory` (line 1125) - *Process multiple checkpoints from directory.*
- `generate_summary` (line 1155) - *Generate aggregate summary report.*
- `_generate_text_report` (line 1188) - *Generate human-readable text report.*

#### `measure_strassen.py`
**Path:** `measure_strassen.py`

*No symbols extracted*

#### `menu.py`
**Path:** `menu.py`

**Functions:**
- `clear_screen` (line 304)
- `print_header` (line 308)
- `print_wrapped` (line 318)
- `wait_for_enter` (line 323)
- `run_script` (line 332)
- `show_checkpoints` (line 362)
- `show_results` (line 397)
- `show_category` (line 433)
- `main_menu` (line 473)

#### `percolation_analysis.py`
**Path:** `percolation_analysis.py`

**Classs:**
- `PercolationConfiguration` (line 31)
- `IModel` (line 97)
- `NumpyModelWrapper` (line 101)
- `_DummyObject` (line 148) - *Transparent stand-in for any unpicklable class found inside a checkpoint.
Stores all keyword and positional constructor arguments as attributes so
that downstream dict-key lookups still work on objects that behave like
namespaces (e.g. UnifiedConfig, TrainingConfig, etc.).*
- `CheckpointMigrator` (line 264)
- `WeightGraphConstructor` (line 350)
- `BondPercolationAnalyzer` (line 402)
- `SitePercolationAnalyzer` (line 488)
- `PruningPercolationAnalyzer` (line 542)
- `ClusterSizeDistributionAnalyzer` (line 725)
- `PercolationUniversalityAnalyzer` (line 775)
- `PercolationCheckpointManager` (line 808)
- `PercolationVisualizationEngine` (line 837)
- `PercolationReportGenerator` (line 1049)
- `PercolationAnalysisPipeline` (line 1139)
- `BilinearStrassenModel` (line 117)

**Functions:**
- `_safe_torch_load` (line 165) - *Load a torch checkpoint that may contain unknown serialized classes
(UnifiedConfig, TrainingConfig, custom dataclasses, etc.).

Strategy:
  1. Try normal torch.load.  If it works, return immediately.
  2. On AttributeError / ModuleNotFoundError, extract the missing
     class names from the exception, inject _DummyObject into
     sys.modules / __main__ under those names, and retry.
  3. Repeat up to MAX_RETRIES times (each retry may reveal a
     new missing class that was nested deeper in the pickle stream).
  4. Clean up injected names after loading.

This keeps torch's own deserialization path (including correct
persistent_load for tensor storage) fully intact.*
- `main` (line 1264)
- `get_effective_input_dim` (line 82)
- `get_total_parameters` (line 85)
- `get_percolation_thresholds` (line 89)
- `get_coefficients` (line 98)
- `__init__` (line 102)
- `get_coefficients` (line 105)
- `get_flat_parameters` (line 108)
- `__init__` (line 155)
- `__repr__` (line 161)
- `_patch_missing` (line 189)
- `_cleanup` (line 230)
- `migrate` (line 265)
- `_migrate_dict` (line 285)
- `_try_migrate_nested` (line 300)
- `_migrate_custom` (line 310)
- `_migrate_coefs` (line 326)
- `_migrate_standard` (line 331)
- `_np` (line 340)
- `__init__` (line 351)
- `construct_adjacency_from_weights` (line 354)
- `construct_weight_correlation_graph` (line 375)
- `construct_slot_interaction_graph` (line 386)
- `__init__` (line 403)
- `analyze` (line 406)
- `_susceptibility` (line 449)
- `_find_pc` (line 456)
- `_exponents` (line 462)
- `_fit_pl` (line 478)
- `__init__` (line 489)
- `analyze` (line 492)
- `__init__` (line 543)
- `analyze` (line 546)
- `_d2a` (line 626)
- `_kappa` (line 631)
- `_hbar` (line 644)
- `_teff` (line 656)
- `_lc` (line 659)
- `_entropy` (line 669)
- `_ipr` (line 680)
- `_lsr` (line 687)
- `_fractal` (line 699)
- `_phase` (line 704)
- `__init__` (line 726)
- `analyze_at_threshold` (line 729)
- `_tau` (line 753)
- `_critical` (line 767)
- `__init__` (line 776)
- `classify_universality` (line 779)
- `__init__` (line 809)
- `should_save` (line 814)
- `save` (line 817)
- `load` (line 829)
- `__init__` (line 838)
- `generate_all_figures` (line 841)
- `_plot_bond` (line 856)
- `_plot_pruning` (line 889)
- `_plot_dashboard` (line 952)
- `_plot_site` (line 993)
- `_plot_cluster` (line 1019)
- `__init__` (line 1050)
- `generate_text_report` (line 1053)
- `generate_json_report` (line 1131)
- `__init__` (line 1140)
- `_load_weights` (line 1153)
- `process_checkpoint` (line 1164)
- `process_directory` (line 1212)
- `_maybe_save` (line 1234)
- `_comparative_summary` (line 1238)
- `__init__` (line 118)
- `_initialize_weights` (line 127)
- `forward` (line 132)
- `get_coefficients` (line 135)
- `get_flat_parameters` (line 141)

#### `plank.py`
**Path:** `plank.py`

**Classs:**
- `Configuration` (line 32) - *Immutable configuration container following Single Responsibility Principle.*
- `BilinearStrassenModel` (line 127) - *Bilinear model for Strassen matrix multiplication.
Implements f(A,B) = W((U*A) ⊙ (V*B)) where ⊙ is element-wise product.*
- `CheckpointMigrator` (line 194) - *Abstract base for checkpoint migration strategies.*
- `CustomFormatMigrator` (line 208) - *Handles custom U,V,W direct formats.*
- `EncoderFormatMigrator` (line 237) - *Handles encoder.layers format.*
- `StandardFormatMigrator` (line 270) - *Handles standard U.weight, V.weight, W.weight format.*
- `CheckpointMigrationManager` (line 284) - *Manages multiple migration strategies.*
- `StrassenDataGenerator` (line 341) - *Generates training data for 2x2 matrix multiplication.*
- `CrystallographyMetrics` (line 384) - *Computes crystallographic quality metrics for Strassen models.*
- `StrassenDiffractionTest` (line 483) - *Tests gauge invariance through permutation symmetry.*
- `BasinResilienceSpectrometer` (line 560) - *Measures basin of attraction through noise injection and recovery.*
- `CrystalPurityIndex` (line 677) - *Computes normalized purity index from component metrics.*
- `PlanckConstantCalculator` (line 763) - *Calculates effective Planck constant from Strassen model parameters.

Maps crystallographic metrics to quantum thermodynamic quantities.*
- `StrassenCheckpointLoader` (line 936) - *Loads and migrates Strassen checkpoints with fallback strategies.*
- `StrassenPlanckAnalyzer` (line 1014) - *Complete analysis pipeline for Strassen checkpoints.

Orchestrates crystallographic analysis and Planck constant calculation.*
- `ReportGenerator` (line 1165) - *Generates reports and visualizations from analysis results.*

**Functions:**
- `set_random_seed` (line 114) - *Set random seeds for reproducibility.*
- `parse_arguments` (line 1339) - *Parse command line arguments.*
- `create_config_from_args` (line 1387) - *Create configuration from command line arguments.*
- `main` (line 1399) - *Main execution entry point.*
- `__post_init__` (line 100) - *Validate configuration parameters.*
- `__init__` (line 133)
- `_initialize_symmetric` (line 143) - *Initialize with Xavier uniform, symmetric U and V.*
- `forward` (line 149) - *Forward pass computing approximate matrix product.

Args:
    matrix_a: Flattened input matrix A [batch, INPUT_DIM]
    matrix_b: Flattened input matrix B [batch, INPUT_DIM]

Returns:
    Approximate product C = A @ B [batch, OUTPUT_DIM]*
- `get_coefficients` (line 166) - *Return current coefficient matrices.*
- `compute_lambda_effective` (line 174) - *Compute effective lambda (confinement potential) from weight magnitudes.
Derived from weight decay interpretation as harmonic confinement.*
- `can_migrate` (line 198) - *Check if this strategy can handle the given state dict.*
- `migrate` (line 203) - *Migrate state dict to standard format.*
- `can_migrate` (line 211)
- `migrate` (line 214)
- `can_migrate` (line 240)
- `migrate` (line 243)
- `can_migrate` (line 273)
- `migrate` (line 276)
- `__init__` (line 287)
- `migrate_checkpoint` (line 294) - *Attempt to migrate checkpoint using available strategies.

Args:
    path: Path to checkpoint file
    device: Device to load tensors to

Returns:
    Migrated state dict or None if migration fails*
- `generate_batch` (line 345) - *Generate a batch of random matrix pairs and their products.

Returns:
    Tuple of (A_flat, B_flat, C_flat) where C = A @ B*
- `verify_structure` (line 366) - *Verify if coefficients represent valid Strassen structure.*
- `compute_kappa` (line 388) - *Compute condition number of gradient covariance matrix.

High kappa indicates ill-conditioned optimization landscape.
Low kappa (approaching 1.0) indicates well-conditioned, crystalline structure.*
- `compute_discretization_margin` (line 429) - *Compute maximum deviation from nearest integer values.

Delta measures how close coefficients are to discrete (integer) values,
indicating crystalline structure formation.*
- `compute_local_complexity` (line 445) - *Compute local complexity based on active parameters.

From "Can't Stop Won't Stop" paper - measures effective parameter count.*
- `compute_all_metrics` (line 464) - *Compute all crystallographic metrics at once.*
- `__init__` (line 486)
- `test_gauge_invariance` (line 490) - *Test if model exhibits true Strassen structure through permutation invariance.

Genuine Strassen algorithm should have exactly one valid permutation (identity).*
- `_compute_functional_error` (line 531) - *Compute functional error between original and permuted coefficients.*
- `__init__` (line 563)
- `measure_resilience_spectrum` (line 570) - *Measure resilience across multiple noise levels.

Returns spectrum showing critical noise level where recovery fails.*
- `_test_noise_recovery` (line 585) - *Test recovery from noise level sigma.*
- `_apply_noise` (line 614) - *Apply Gaussian noise to model parameters.*
- `_anneal_to_attractor` (line 621) - *Anneal model back to attractor using fine-tuning.

Returns number of epochs needed for recovery.*
- `_estimate_critical_noise` (line 654) - *Estimate critical noise level where success rate drops below 50%.*
- `__init__` (line 680)
- `compute` (line 692) - *Compute normalized purity index and grade.*
- `_assign_grade` (line 745) - *Assign crystallographic grade based on delta (primary indicator).*
- `__init__` (line 770)
- `calculate_all` (line 789) - *Execute all Planck constant calculation methods.*
- `_determine_regime_and_weights` (line 875) - *Determine confinement regime and corresponding weights.*
- `_compute_derived_constants` (line 886) - *Compute derived Planck-scale constants.*
- `_compute_universe_comparison` (line 917) - *Compare calculated constants with physical universe.*
- `__init__` (line 939)
- `load` (line 943) - *Load checkpoint into model instance.

Args:
    checkpoint_path: Path to checkpoint file
    device: Target device

Returns:
    Loaded model or None if loading fails*
- `extract_training_metrics` (line 986) - *Extract training metrics from checkpoint if available.*
- `__init__` (line 1021)
- `analyze_checkpoint` (line 1025) - *Perform complete analysis of a single checkpoint.

Args:
    checkpoint_path: Path to checkpoint file
    device: Computation device

Returns:
    Complete analysis report*
- `analyze_directory` (line 1104) - *Analyze all checkpoints in a directory.

Args:
    directory: Directory containing checkpoints
    device: Computation device
    pattern: File pattern to match

Returns:
    List of analysis reports*
- `_print_summary` (line 1146) - *Print formatted summary of analysis results.*
- `__init__` (line 1168)
- `save_json_report` (line 1173) - *Save individual report as JSON.*
- `save_aggregate_report` (line 1185) - *Save aggregate report from multiple analyses.*
- `_compute_statistics` (line 1216) - *Compute aggregate statistics from summaries.*
- `_count_grades` (line 1247) - *Count distribution of grades.*
- `generate_visualizations` (line 1255) - *Generate visualization plots.*

#### `purity_index.py`
**Path:** `purity_index.py`

**Classs:**
- `PurityConfig` (line 19)
- `IModel` (line 44)
- `IPurityIndexCalculator` (line 49)
- `IEffectiveTemperatureCalculator` (line 54)
- `IPhaseClassifier` (line 59)
- `IPolycrystalAnalyzer` (line 64)
- `IPurityComparator` (line 69)
- `BilinearModel` (line 73)
- `PurityIndexCalculator` (line 101)
- `EffectiveTemperatureCalculator` (line 156)
- `PhaseClassifier` (line 199)
- `PolycrystalAnalyzer` (line 236)
- `PurityComparator` (line 274)
- `CheckpointMigrator` (line 311)
- `PurityAnalyzer` (line 361)
- `PurityPipeline` (line 492)

**Functions:**
- `main` (line 611)
- `get_coefficients` (line 45)
- `calculate` (line 50)
- `calculate` (line 55)
- `classify` (line 60)
- `analyze_polycrystal` (line 65)
- `compare` (line 70)
- `__init__` (line 74)
- `_initialize` (line 85)
- `forward` (line 90)
- `get_coefficients` (line 93)
- `__init__` (line 102)
- `calculate` (line 105)
- `_compute_layer_purity` (line 134)
- `_delta_to_alpha` (line 140)
- `_assess_purity_quality` (line 145)
- `__init__` (line 157)
- `calculate` (line 160)
- `__init__` (line 200)
- `classify` (line 203)
- `classify_polycrystal_state` (line 219)
- `__init__` (line 237)
- `analyze_polycrystal` (line 243)
- `_prune_model` (line 264)
- `__init__` (line 275)
- `compare` (line 279)
- `migrate` (line 312)
- `_migrate_dict` (line 322)
- `_migrate_custom_format` (line 331)
- `_migrate_coefs_format` (line 350)
- `_migrate_standard_format` (line 357)
- `__init__` (line 362)
- `_load_checkpoint` (line 375)
- `analyze` (line 399)
- `_print_report` (line 445)
- `__init__` (line 493)
- `process_checkpoint` (line 496)
- `process_directory` (line 510)
- `generate_summary` (line 537)
- `_generate_text_report` (line 574)

#### `repor_experiments.py`
**Path:** `repor_experiments.py`

**Classs:**
- `ModelConfig` (line 78)
- `TrainConfig` (line 86)
- `Exp1Config` (line 96)
- `Exp2Config` (line 103)
- `Exp3Config` (line 108)
- `Exp4Config` (line 114)
- `Exp5Config` (line 120)
- `SuiteConfig` (line 126)
- `BilinearModel` (line 141)

**Functions:**
- `discretize_q` (line 173)
- `compute_delta` (line 176)
- `phase2` (line 181) - *Poda a target_rank slots, discretiza a {-1,0,1}, verifica.*
- `_verify_2x2` (line 204)
- `zero_shot_verify` (line 214)
- `_recursive_strassen` (line 222)
- `_strassen_rec` (line 234)
- `compute_kappa` (line 262)
- `compute_alpha` (line 279)
- `compute_teff` (line 284)
- `classify_phase` (line 298)
- `load_checkpoint` (line 309)
- `_extract_state` (line 329)
- `train_model` (line 351)
- `analyze_checkpoint` (line 378)
- `experiment1` (line 428)
- `experiment2` (line 475)
- `experiment3` (line 507)
- `experiment4` (line 563)
- `experiment5` (line 607)
- `_test_accuracy` (line 633)
- `_random_prune` (line 644)
- `_boundary_prune` (line 655)
- `analyze_checkpoints` (line 665)
- `main` (line 705)
- `_save` (line 783)
- `__init__` (line 142)
- `forward` (line 149)
- `slot_importance` (line 156)
- `get_weights` (line 163)
- `get_flat` (line 167)
- `cb` (line 434)

#### `scrodingger.py`
**Path:** `scrodingger.py`

**Classs:**
- `SchrodingerConfig` (line 19)
- `IModel` (line 44)
- `IWaveFunctionExtractor` (line 49)
- `IPotentialCalculator` (line 54)
- `IHamiltonianConstructor` (line 59)
- `IEigenvalueSolver` (line 64)
- `ITimeEvolver` (line 69)
- `IExpectationValueCalculator` (line 74)
- `IUncertaintyCalculator` (line 79)
- `ICheckpointLoader` (line 84)
- `ICheckpointMigrator` (line 89)
- `BilinearModel` (line 93)
- `WaveFunctionExtractor` (line 121)
- `PotentialCalculator` (line 128)
- `HamiltonianConstructor` (line 150)
- `EigenvalueSolver` (line 170)
- `TimeEvolver` (line 194)
- `ExpectationValueCalculator` (line 216)
- `UncertaintyCalculator` (line 226)
- `CheckpointLoader` (line 263)
- `CheckpointMigrator` (line 271)
- `SchrodingerAnalyzer` (line 320)
- `WaveFunctionVisualizer` (line 542)
- `SchrodingerPipeline` (line 609)

**Functions:**
- `main` (line 779)
- `get_coefficients` (line 45)
- `extract` (line 50)
- `calculate` (line 55)
- `construct` (line 60)
- `solve` (line 65)
- `evolve` (line 70)
- `calculate` (line 75)
- `calculate` (line 80)
- `load` (line 85)
- `migrate` (line 90)
- `__init__` (line 94)
- `_initialize` (line 105)
- `forward` (line 110)
- `get_coefficients` (line 113)
- `extract` (line 122)
- `__init__` (line 129)
- `calculate` (line 132)
- `__init__` (line 151)
- `construct` (line 154)
- `__init__` (line 171)
- `solve` (line 174)
- `__init__` (line 195)
- `evolve` (line 198)
- `calculate` (line 217)
- `__init__` (line 227)
- `calculate` (line 230)
- `load` (line 264)
- `migrate` (line 272)
- `_migrate_dict` (line 284)
- `_migrate_custom_format` (line 293)
- `_migrate_coefs_format` (line 309)
- `_migrate_standard_format` (line 316)
- `__init__` (line 321)
- `_load_checkpoint` (line 335)
- `analyze` (line 357)
- `_calculate_tunneling_probability` (line 451)
- `_count_degeneracy` (line 465)
- `_print_report` (line 478)
- `__init__` (line 543)
- `visualize` (line 546)
- `__init__` (line 610)
- `process_checkpoint` (line 614)
- `process_directory` (line 628)
- `generate_summary` (line 655)
- `_generate_text_report` (line 718)

#### `benchmark_final.py`
**Path:** `src/benchmarks/benchmark_final.py`

**Functions:**
- `strassen_hybrid_multiply` (line 45) - *Multiply using our Strassen Hybrid implementation*
- `numpy_multiply` (line 60) - *Standard NumPy BLAS multiplication*
- `benchmark` (line 64) - *Run benchmark with warmup*
- `main` (line 80)

#### `benchmark_scientific.py`
**Path:** `src/benchmarks/benchmark_scientific.py`

**Functions:**
- `strassen_multiply` (line 38)
- `standard_avx512_multiply` (line 51)
- `numpy_multiply` (line 64)
- `benchmark_function` (line 67) - *Benchmark with statistical analysis*
- `main` (line 91)

#### `benchmark_strassen.py`
**Path:** `src/benchmarks/benchmark_strassen.py`

**Classs:**
- `BenchmarkResult` (line 29)
- `BenchmarkConfig` (line 45)

**Functions:**
- `load_config` (line 61) - *Load configuration from TOML file.*
- `get_dtype` (line 93) - *Convert dtype string to torch dtype.*
- `estimate_memory_mb` (line 104) - *Estimate memory usage for matrix multiplication.*
- `benchmark_resolution` (line 114) - *Benchmark Strassen vs standard matmul for given resolution.*
- `run_benchmark` (line 218) - *Run full benchmark suite.*
- `save_results` (line 310) - *Save benchmark results to JSON.*
- `main` (line 322)

#### `strassen_numpy.py`
**Path:** `src/benchmarks/strassen_numpy.py`

**Functions:**
- `_load_weights` (line 19)
- `strassen_2x2_numpy` (line 29) - *Strassen 2x2 using grokked coefficients.*
- `strassen_numpy` (line 44) - *Recursive Strassen using NumPy.*
- `strassen_hybrid` (line 78) - *Hybrid Strassen: use Strassen for large matrices, NumPy for small.
This is faster because NumPy matmul is highly optimized for small matrices.*
- `multiplication_count` (line 112) - *Count multiplications used by Strassen.*

#### `auto_T_discovery.py`
**Path:** `src/discovery/auto_T_discovery.py`

**Classs:**
- `SymmetryStructure` (line 22) - *Discovered symmetries in weight matrix*
- `AutoTDiscovery` (line 32) - *Automatic discovery of expansion operator T from converged weights.

The algorithm works in three phases:
1. Spectral Analysis: Extract dominant singular subspace
2. Symmetry Detection: Find block/permutation structure
3. T Construction: Build expansion operator preserving invariants*

**Functions:**
- `verify_strassen_T` (line 306) - *Verify T discovery on Strassen model*
- `verify_expanded_correctness` (line 353) - *Verify that expanded operator correctly computes matrix multiplication*
- `recursive_strassen_multiply` (line 389) - *Recursively apply learned Strassen decomposition

This is the IMPLEMENTATION of T: it shows how the base 2x2 decomposition
extends to arbitrary sizes via recursive block application.*
- `__init__` (line 42)
- `analyze_structure` (line 46) - *Phase 1 & 2: Analyze weight matrix structure*
- `_detect_discrete_values` (line 89) - *Detect if weights cluster around discrete values*
- `_detect_block_structure` (line 105) - *Detect repeating block patterns*
- `_block_repetition_score` (line 127) - *Score how well blocks repeat (lower = more repetitive)*
- `_detect_symmetry_type` (line 144) - *Detect type of symmetry in weight matrix*
- `_is_permutation_symmetric` (line 165) - *Check if rows are permutations of a base pattern*
- `_is_cyclic` (line 171) - *Check for cyclic/Toeplitz structure*
- `_invariant_subspace_dim` (line 184) - *Estimate dimension of truly invariant subspace*
- `_discretization_error` (line 198) - *Compute error when discretizing to given values*
- `_print_analysis` (line 213) - *Print analysis results*
- `construct_T` (line 229) - *Phase 3: Construct expansion operator T

For matrix multiplication (U, V, W tensors), T expands via
recursive block structure discovered from the base case.

Args:
    W_dict: Dictionary with 'U', 'V', 'W' tensors
    target_size: Target matrix dimension (n' in the paper)

Returns:
    Expanded weight dictionary*
- `_validate_expansion` (line 290) - *Validate that expansion preserves key invariants*

#### `convergence_theory.py`
**Path:** `src/training/convergence_theory.py`

**Classs:**
- `ConvergenceMetrics` (line 21) - *Metrics for tracking convergence to algorithmic invariance*
- `HutchinsonTraceEstimator` (line 31) - *Efficient Hessian trace estimation using Hutchinson's method.

tr(H) ≈ E[v^T H v] where v ~ Rademacher(±1)

Complexity: O(n_samples * forward_backward_pass) instead of O(n²)*
- `HardwareNoiseEstimator` (line 136) - *Estimate ε_hw(B, T) - hardware-induced variance

Extended model addressing reviewer's criticism:
ε_hw(B, T) = α/B + β*cache_miss_rate(B) + γ*thread_contention(T)*
- `SimpleStrassenModel` (line 346) - *Simple model for testing convergence verification*

**Functions:**
- `convergence_theorem` (line 193) - *THEOREM (Convergence to Algorithmic Invariance)

Let W_t denote the weights at step t under SGD with:
    W_{t+1} = W_t - η∇L(W_t) + ξ_t

where ξ_t is stochastic gradient noise with Var(ξ_t) = σ²/B + ε_hw(B,T).

ASSUMPTIONS:
A1. The target function f admits a rank-r tensor decomposition
A2. The loss L is twice differentiable with Lipschitz Hessian
A3. The effective curvature satisfies κ_eff < 0 after step t₀

THEOREM: Under A1-A3, if
    
    Var(ξ_t) < σ_min(W*)² / (η · condition(H))

where σ_min(W*) is the smallest non-zero singular value of the optimal
decomposition, then with probability 1-δ:

    lim_{t->∞} d(W_t, W*) = 0

where d is the subspace distance and W* is the algorithmically invariant solution.

PROOF SKETCH:

1. By A3 (κ_eff < 0), the loss landscape is locally convex near convergence

2. The noise condition ensures gradient updates don't overshoot the
   invariant subspace defined by σ_min(W*)

3. By spectral gap analysis, W_t projects onto the dominant singular
   subspace with increasing precision as t -> ∞

4. The discretization {-1, 0, 1} emerges because integer solutions
   are fixed points of the projection when noise is controlled

IMPLICATIONS FOR T:

The expansion operator T is constructible because:
- T preserves the dominant singular subspace (by definition)
- The rank-r structure is independent of problem scale (by A1)
- Therefore T = block_embed(W_r) where W_r is the converged rank-r solution*
- `verify_convergence_conditions` (line 265) - *Verify that convergence conditions are satisfied for a trained model.*
- `__init__` (line 40)
- `estimate_trace` (line 47) - *Estimate tr(H) using Hutchinson's stochastic trace estimator.

Returns:
    (mean_trace, std_trace)*
- `_rademacher_vector` (line 73) - *Generate Rademacher random vector (±1 with equal probability)*
- `_hessian_vector_product` (line 90) - *Compute H @ v using the "double backward" trick.

H @ v = ∂/∂θ (∇L · v)*
- `compute_kappa_eff` (line 118) - *Compute κ_eff = -tr(H) / N

Interpretation:
- κ_eff < 0 and stable -> grokking likely
- κ_eff > 0 or oscillating -> grokking unlikely*
- `__init__` (line 144)
- `estimate_noise` (line 148) - *Estimate hardware noise by measuring gradient variance across batches*
- `__init__` (line 348)
- `forward` (line 354)

#### `grokkit_physics.py`
**Path:** `src/training/grokkit_physics.py`

**Functions:**
- `strassen_multiply` (line 49) - *Wrapper for Strassen multiplication (uses float32).*
- `measure_physics` (line 64) - *Measure the 'physical quantities' for a given matrix size.

Returns:
    dict with: speedup, hbar_strassen, hbar_numpy, coherence, error*
- `detect_phase_transition` (line 126) - *Find the critical size N_c where the phase transition occurs.
Uses the maximum derivative of speedup curve.*
- `main` (line 148)

#### `main.py`
**Path:** `src/training/main.py`

**Classs:**
- `Config` (line 44) - *Configuración del experimento.*
- `StrassenDiscovery` (line 87) - *Modelo para descubrir Strassen mediante coeficientes aprendibles.

Arquitectura:
- U_coefs[i]: Combinación de bloques de A para producto M_i
- V_coefs[i]: Combinación de bloques de B para producto M_i  
- W_coefs[j,i]: Contribución de M_i al bloque C_j del resultado*
- `Matrix4x4Dataset` (line 199) - *Dataset de multiplicación de matrices 4x4.*
- `Trainer` (line 217) - *Entrenador con enmascaramiento progresivo.*

**Functions:**
- `set_seed` (line 79) - *Fijar semilla para reproducibilidad.*
- `main` (line 345)
- `__init__` (line 96)
- `forward` (line 108) - *Forward pass con multiplicación matemática pura.
A, B: (batch, 4, 4) -> C: (batch, 4, 4)*
- `get_slot_norms` (line 155) - *Norma promedio de cada slot.*
- `get_active_slots` (line 160) - *Número de slots activos.*
- `mask_slot` (line 164) - *Desactiva un slot.*
- `get_weakest_slot` (line 169) - *Slot con menor norma entre los activos.*
- `print_coefficients` (line 175) - *Muestra coeficientes descubiertos.*
- `__init__` (line 202)
- `__len__` (line 210)
- `__getitem__` (line 213)
- `__init__` (line 220)
- `accuracy` (line 242)
- `train_epoch` (line 245)
- `evaluate` (line 265)
- `train` (line 280)

#### `main_pure_math.py`
**Path:** `src/training/main_pure_math.py`

**Classs:**
- `StrassenModel` (line 22) - *Descomposición de tensor para multiplicación de matrices 2x2.
C_ij = sum_r W[ij,r] * (U[r,:] @ a) * (V[r,:] @ b)*

**Functions:**
- `gen_data` (line 58)
- `train` (line 64)
- `verify` (line 88)
- `hard_prune` (line 106) - *Poda los slots más débiles, mantiene top-k.*
- `refine_pruned` (line 122) - *Refina manteniendo slots podados en cero.*
- `show_coeffs` (line 159)
- `main` (line 186)
- `__init__` (line 28)
- `forward` (line 35)
- `slot_norms` (line 47) - *Norma combinada de cada slot.*
- `active_count` (line 54)

#### `strassen_core.py`
**Path:** `src/training/strassen_core.py`

**Functions:**
- `_load_weights` (line 14)
- `strassen_2x2` (line 21)
- `strassen` (line 44)
- `get_coefficients` (line 77)
- `multiplication_count` (line 82)

#### `strassen_grokkit.py`
**Path:** `src/training/strassen_grokkit.py`

**Classs:**
- `StrassenOperator` (line 27) - *Operador espectral para multiplicación de matrices 2x2.

Representa el tensor de rango R:
C_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)

Donde:
- U, V: Coeficientes de combinación lineal (LC)
- W: Coeficientes de reconstrucción
- Esparsidad (SP): Cuántos slots están activos*

**Functions:**
- `generate_batch` (line 113) - *Genera batch de matrices aleatorias.*
- `train_grokkit` (line 120) - *Entrena usando el framework Grokkit.

WD (Weight Decay) actúa como presión termodinámica que:
1. Empuja hacia soluciones de menor norma
2. Promueve esparsidad natural (slots débiles -> 0)
3. Cristaliza el operador en el mínimo de energía (rango 7)*
- `verify_grokking` (line 203) - *Verifica que el operador ha grokkeado correctamente.*
- `progressive_sparsification` (line 254) - *Fase 2: Esparsificación progresiva.
Reduce gradualmente a 7 slots manteniendo accuracy.*
- `main` (line 351) - *Pipeline principal Grokkit para Strassen.*
- `__init__` (line 40)
- `forward` (line 49) - *Computa A @ B usando la descomposición tensorial.*
- `compute_LC` (line 67) - *Linear Combination metric.
Mide qué tan bien los coeficientes forman combinaciones válidas.
LC -> 1 significa combinaciones perfectas.*
- `compute_SP` (line 83) - *Sparsity metric.
SP -> 0 significa máxima esparsidad (menos slots activos).
SP = (slots_activos - 7) / rank para normalizar*
- `slot_importance` (line 101) - *Importancia de cada slot basada en normas.*
- `count_active` (line 108) - *Cuenta slots activos.*

#### `train_strassen.py`
**Path:** `src/training/train_strassen.py`

**Classs:**
- `StrassenOperator` (line 26) - *Spectral operator for 2x2 matrix multiplication.

Tensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)*

**Functions:**
- `generate_batch` (line 60)
- `train_phase1` (line 66) - *Phase 1: Grokking with Weight Decay as thermodynamic pressure.*
- `sparsify` (line 104) - *Phase 2: Progressive sparsification to target rank.*
- `discretize` (line 183) - *Phase 3: Discretize coefficients to {-1, 0, 1}.*
- `get_canonical_strassen` (line 211) - *Returns the canonical Strassen coefficients.
Exact discrete coefficients for rank-7 tensor decomposition.

Strassen's 7 products:
M1 = (a11 + a22)(b11 + b22)
M2 = (a21 + a22) * b11
M3 = a11 * (b12 - b22)
M4 = a22 * (b21 - b11)
M5 = (a11 + a12) * b22
M6 = (a21 - a11)(b11 + b12)
M7 = (a12 - a22)(b21 + b22)

Result reconstruction:
c11 = M1 + M4 - M5 + M7
c12 = M3 + M5
c21 = M2 + M4
c22 = M1 - M2 + M3 + M6*
- `verify` (line 260) - *Verify the discretized operator.*
- `main` (line 299) - *Main training pipeline.*
- `__init__` (line 33)
- `forward` (line 40)
- `slot_importance` (line 50)
- `count_active` (line 56)

#### `superposition.py`
**Path:** `superposition.py`

**Classs:**
- `Config` (line 23) - *Configuration for Superposition Analysis on Strassen Checkpoints.*
- `ICheckpointLoader` (line 59)
- `IMetricsCalculator` (line 62)
- `IAnalyzer` (line 66)
- `CheckpointLoadingError` (line 71)
- `CheckpointLoader` (line 75) - *Loads raw checkpoint files.*
- `CheckpointMigrator` (line 85) - *Migrates various checkpoint formats to standard state_dict.*
- `StrassenDataGenerator` (line 220) - *Generates matrix multiplication data for activation extraction.*
- `BilinearStrassenModel` (line 258) - *Your existing Strassen model architecture.*
- `SparseAutoencoder` (line 292) - *SAE with tied weights (W_dec = W_enc^T).
Corrected dimensions: W_enc: [D, N], encode uses W_enc^T, decode uses W_enc.*
- `SuperpositionMetrics` (line 334) - *Calculates superposition metrics from Section 4 of the paper.*
- `SAETrainer` (line 407) - *Trains SAE on bottleneck activations extracted from Strassen model.*
- `StrassenCheckpointAnalyzer` (line 475) - *Analyzes existing Strassen checkpoints for superposition metrics.*

**Functions:**
- `main` (line 765)
- `__post_init__` (line 54)
- `load_checkpoint` (line 60)
- `compute` (line 64)
- `analyze_checkpoint` (line 68)
- `load_checkpoint` (line 78)
- `detect_hidden_dim` (line 89) - *Detect hidden dimension from checkpoint data structure by inspecting
tensor shapes in various known formats.
Returns None if cannot be determined unambiguously.*
- `migrate_checkpoint` (line 122)
- `_migrate_dict` (line 135)
- `_migrate_custom_format` (line 147)
- `_migrate_coefs_format` (line 162)
- `_migrate_encoder_format` (line 170) - *Handle encoder-based format from specific experimental architectures.
Extracts U, V, W from sequential encoder layers assuming specific indexing.*
- `_migrate_standard_format` (line 215)
- `__init__` (line 223)
- `generate_batch` (line 226) - *Generate batch of matrix pairs and their product.*
- `generate_dataset` (line 240) - *Generate full dataset.*
- `__init__` (line 261)
- `_initialize_symmetric` (line 271)
- `forward` (line 276) - *Forward pass returning output and bottleneck activations.*
- `get_coefficients` (line 284)
- `__init__` (line 298)
- `encode` (line 310) - *Encode bottleneck activations to sparse features.
x: [batch, N]
W_enc: [D, N]
Returns: [batch, D]*
- `decode` (line 319) - *Decode sparse features back to bottleneck.
z: [batch, D]
W_enc: [D, N]
Returns: [batch, N]*
- `forward` (line 328)
- `__init__` (line 339)
- `compute_feature_probabilities` (line 342) - *Calculate feature probabilities from SAE activations.
p_i = Σ_s |z_i,s| / Σ_j Σ_s |z_j,s|*
- `compute_entropy` (line 352) - *Shannon entropy H(p) = -Σ p_i log p_i.*
- `compute_superposition` (line 359) - *Main metric: ψ = F/N where F = e^{H(p)}.*
- `compute_frobenius_metric` (line 377) - *Baseline from Eq 2: ψ_Frob = ||W||_F^2 / N.
Applied to the bottleneck transformation (W matrix of Strassen).*
- `compute_interference_matrix` (line 385) - *Compute W^T @ W to analyze interference patterns.*
- `compute` (line 389) - *Unified interface.*
- `__init__` (line 410)
- `train` (line 421) - *Train SAE on extracted activations.*
- `__init__` (line 480)
- `load_model` (line 496) - *Load and migrate checkpoint to model.
Returns tuple of (model, effective_config) where effective_config has 
the correct HIDDEN_DIM for this specific checkpoint to avoid dimension mismatches.*
- `extract_bottleneck_activations` (line 554) - *Extract bottleneck activations (U(a) * V(b)) from model.*
- `analyze_checkpoint` (line 571) - *Full analysis pipeline for a single checkpoint:
1. Load model with correct dimensions (detecting hidden_dim from checkpoint)
2. Extract bottleneck activations
3. Train SAE with matching dimensions (using effective_config)
4. Calculate superposition metrics with correct normalization (N=hidden_dim)
5. Calculate baseline Frobenius metric on W weights*
- `_save_intermediate_result` (line 633) - *Save result for individual checkpoint.*
- `analyze_directory` (line 641) - *Analyze all checkpoints in directory.*
- `_save_progress_checkpoint` (line 680) - *Save intermediate progress.*
- `_save_final_results` (line 687) - *Save complete results.*
- `_generate_comparison_plots` (line 694) - *Generate comparison plots across checkpoints.*
- `get_tensor` (line 148)

#### `train_batch_sweep.py`
**Path:** `train_batch_sweep.py`

**Classs:**
- `LocalConfig` (line 17)

**Functions:**
- `train_for_batch_size` (line 11)

#### `unified_hidden_connections_suite.py`
**Path:** `unified_hidden_connections_suite.py`

**Classs:**
- `StrassStrassenConfig` (line 65) - *Immutable canonical configuration for the Strassen bilinear model.*
- `TrainingConfig` (line 84) - *Immutable training hyperparameters.*
- `Experiment1Config` (line 99) - *Configuration for Experiment 1: Ricci-MBL Duality.*
- `Experiment2Config` (line 114) - *Configuration for Experiment 2: Altland-Zirnbauer Symmetry Dial.*
- `Experiment3Config` (line 131) - *Configuration for Experiment 3: Conformal Isomorphism.*
- `Experiment4Config` (line 141) - *Configuration for Experiment 4: Compression Frontier.*
- `Experiment5Config` (line 158) - *Configuration for Experiment 5: Holographic Pruning.*
- `SuiteConfig` (line 169) - *Top-level suite orchestration configuration.*
- `StrassStrassenModel` (line 183) - *Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.
Implements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.
U, V ∈ R^{rank x input_dim}, W ∈ R^{output_dim x rank}.*
- `IDataGenerator` (line 229) - *Protocol for deterministic data generation.*
- `StrassenDataGenerator` (line 235) - *Generates random 2x2 matrix pairs and their exact products.*
- `ICheckpointManager` (line 262) - *Protocol for checkpoint persistence.*
- `CheckpointManager` (line 269) - *Handles safe serialization and deserialization of model checkpoints.*
- `ITrainer` (line 293) - *Protocol for training routines.*
- `Trainer` (line 301) - *Standard trainer with AdamW, cosine annealing, and gradient clipping.*
- `IMetricCalculator` (line 357) - *Abstract base for all metric calculators.*
- `LevelSpacingRatioCalculator` (line 365) - *Computes the adjacent gap ratio r for Hessian eigenvalue spectra.*
- `RicciScalarCalculator` (line 415) - *Computes Ricci scalar and geometric curvature metrics from Hessian.*
- `SyntheticPlanckCalculator` (line 492) - *Estimates the synthetic Planck constant from model weight statistics.*
- `SuperpositionMetricCalculator` (line 552) - *Measures superposition via sparse autoencoder bottleneck analysis.*
- `IExperiment` (line 645) - *Abstract base for all experiments in the suite.*
- `Experiment1RicciMBLDuality` (line 657) - *Tests the claim that Ricci curvature smoothing is the geometric mechanism
driving the Wigner-Dyson to Poisson/MBL spectral transition.*
- `Experiment2AltlandZirnbauer` (line 735) - *Tests the claim that the imaginary-weight control parameter gamma drives
the system between GOE (orthogonal) and GUE (unitary) random matrix classes.*
- `Experiment3ConformalIsomorphism` (line 820) - *Tests the claim that the network learns the underlying conformal operator
by applying Moebius transformations to inputs and measuring equivariance.*
- `Experiment4CompressionFrontier` (line 886) - *Tests the thermodynamic uncertainty relation between synthetic Planck
constant and superposition metric across a sweep of batch sizes and
weight decay values.*
- `IPruningStrategy` (line 965) - *Abstract base for structured pruning strategies.*
- `VolumePruningStrategy` (line 973) - *Prunes weights uniformly at random across all layers.*
- `AreaPruningStrategy` (line 988) - *Prunes weights only from tensor boundaries (slot edges).*
- `Experiment5HolographicPruning` (line 1004) - *Tests whether the crystal phase encodes information on boundaries
(area law) versus the glass phase encoding it volumetrically.*
- `UnifiedSuite` (line 1078) - *Orchestrates the execution of all five hidden-connection experiments.*

**Functions:**
- `main` (line 1182)
- `__post_init__` (line 76)
- `__init__` (line 190)
- `forward` (line 203)
- `get_coefficients` (line 213)
- `get_flat_parameters` (line 216)
- `slot_importance` (line 219)
- `count_active_slots` (line 225)
- `generate_batch` (line 232)
- `__init__` (line 238)
- `generate_batch` (line 241)
- `save` (line 265)
- `load` (line 266)
- `save` (line 272)
- `load` (line 284)
- `train` (line 296)
- `__init__` (line 304)
- `train` (line 314)
- `calculate` (line 361)
- `__init__` (line 368)
- `_build_hessian_approximation` (line 372)
- `calculate` (line 384)
- `__init__` (line 418)
- `_compute_hessian` (line 422)
- `_generate_single_sample` (line 433)
- `_loss_from_flat` (line 437)
- `_diagonal_hessian_approximation` (line 456)
- `calculate` (line 470)
- `__init__` (line 495)
- `calculate` (line 499)
- `__init__` (line 555)
- `_extract_activations` (line 576)
- `_train_sae` (line 589)
- `_sae_forward` (line 616)
- `calculate` (line 623)
- `run` (line 649)
- `get_name` (line 653)
- `__init__` (line 663)
- `get_name` (line 679)
- `run` (line 682)
- `_analyze_temporal_correlation` (line 717)
- `__init__` (line 741)
- `get_name` (line 752)
- `run` (line 755)
- `_create_gamma_model` (line 778)
- `_train_gamma_model` (line 790)
- `_detect_critical_transition` (line 800)
- `__init__` (line 826)
- `get_name` (line 836)
- `run` (line 839)
- `_apply_moebius` (line 869)
- `_apply_moebius_to_output` (line 879)
- `__init__` (line 893)
- `get_name` (line 904)
- `run` (line 907)
- `_test_uncertainty_bound` (line 951)
- `prune` (line 969)
- `prune` (line 976)
- `prune` (line 991)
- `__init__` (line 1010)
- `get_name` (line 1022)
- `run` (line 1025)
- `_run_pruning_trials` (line 1046)
- `__init__` (line 1081)
- `_build_experiments` (line 1088)
- `run_all` (line 1127)
- `_aggregate_verdicts` (line 1150)
- `_serialize_config` (line 1163)
- `checkpoint_callback` (line 691)

#### `xray_tensor_diffractometer.py`
**Path:** `xray_tensor_diffractometer.py`

**Classs:**
- `Config` (line 27)
- `ICheckpointLoader` (line 146)
- `IMetricsCalculator` (line 149)
- `IDataGenerator` (line 152)
- `CheckpointLoadingError` (line 155)
- `MetricsComputationError` (line 158)
- `TrainingError` (line 161)
- `StrassenDataGenerator` (line 166)
- `BilinearStrassenModel` (line 187)
- `EpitaxialGrowthEngine` (line 216) - *Motor de crecimiento epitaxial para cristales algorítmicos.

FÍSICA: Imita el crecimiento de cristales en sustratos donde la estructura
atómica del sustrato guía la formación del nuevo cristal.*
- `EpitaxyExperiment` (line 469) - *Experimento completo de epitaxia: sembrar, crecer, analizar.*
- `ThermodynamicPotential` (line 671) - *Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C*
- `SpectroscopyMetrics` (line 693)
- `ThermodynamicMetrics` (line 913)
- `CrystallographyMetrics` (line 1158)
- `GreenCowExperiment` (line 1264) - *🐄 Green's Cow: Uses integration-by-parts analogy to split gradient into bulk and boundary terms.
Inspired by Green's identities: ∫_Ω ∇u·v = ∫_∂Ω u(v·n) - ∫_Ω u∇·v
Applied to weight tensors as discrete manifolds.*
- `CheckpointLoader` (line 1371)
- `CheckpointMigrator` (line 1405)
- `BoltzmannAnalysisProgram` (line 1490)
- `StrassenCrystallographer` (line 2610)

**Functions:**
- `set_seed` (line 64)
- `setup_logger` (line 72)
- `run_epitaxy_from_best_crystal` (line 86) - *Pipeline automático: encuentra el mejor cristal y lo usa como semilla.*
- `main` (line 2683)
- `load_checkpoint` (line 147)
- `compute` (line 150)
- `generate_batch` (line 153)
- `generate_batch` (line 168)
- `verify_structure` (line 179)
- `__init__` (line 188)
- `_initialize_symmetric` (line 199)
- `forward` (line 204)
- `get_coefficients` (line 207)
- `__init__` (line 224)
- `_load_seed_crystal` (line 242) - *Carga el cristal semilla verificando su pureza*
- `grow_epitaxial_crystal` (line 265) - *Crece un cristal epitaxial desde la semilla.

MÉTODO: Kronecker product preserva la estructura periódica:
Si A es cristal Strassen de 2x2, entonces A ⊗ I_n es cristal de (2n)x(2n)*
- `_adjust_dimensions` (line 326) - *Ajusta dimensiones del tensor epitaxial para coincidir con el modelo objetivo.
Rellena con ruido térmico pequeño o trunca según sea necesario.*
- `anneal_crystal` (line 360) - *Recocido térmico del cristal epitaxial.

FÍSICA: En lugar de "entrenar desde cero", aplicamos temperatura decreciente
para que el cristal se auto-organice alrededor de la semilla.*
- `__init__` (line 474)
- `run_epitaxial_growth_experiment` (line 479) - *Experimento completo: cultiva cristales de múltiples tamaños desde una semilla.*
- `_plot_epitaxial_evolution` (line 556) - *Visualiza la evolución del cristal durante el recocido*
- `_generate_comparative_report` (line 604) - *Genera reporte comparativo de todos los experimentos epitaxiales*
- `helmholtz_free_energy` (line 680) - *F = U - T*S (a μ y N constantes)*
- `gibbs_free_energy` (line 684) - *G = F + μ*N + P*V (presión algorítmica)*
- `is_stable` (line 689) - *Criterio de estabilidad: dG < 0*
- `compute_weight_diffraction` (line 696)
- `_compute_spectral_entropy` (line 719)
- `extract_lattice_parameters` (line 726) - *Extrae parámetros de red preservando la geometría física del tensor.

FIX: En lugar de reshape arbitrario, aplicamos SVD sobre la matriz 
de covarianza que preserva la estructura de correlaciones.*
- `compute_gibbs_free_energy` (line 785)
- `extract_canonical_decomposition` (line 791) - *Descomposición Canónica del tensor tripartito (U, V, W).

FIX: Preserva la estructura bilineal en lugar de tratar como matriz plana.
Aplicamos HOSVD (Higher-Order SVD) para tensores de orden 3.*
- `_discretize_to_integers` (line 847) - *Proyecta factores continuos a la red cristalina discreta {-1, 0, 1}.*
- `_check_strassen_equivalence` (line 868) - *Verifica si los factores discretizados corresponden a la estructura de Strassen.*
- `create_superlattice_seed` (line 892)
- `compute_effective_temperature` (line 916)
- `compute_critical_exponents` (line 930) - *Calcula exponentes críticos cerca de transiciones de fase.

Leyes de escala:
- C_v ~ |T - T_c|^{-α_exp}  (calor específico)
- ξ ~ |T - T_c|^{-ν}        (longitud de correlación)
- τ ~ |T - T_c|^{-z}        (tiempo de grokking)*
- `compute_equation_of_state` (line 1009) - *Ecuación de estado: T_c(α) = T_0 * exp(-c*α)

FIX: Relación constitutiva que describe la curva de coexistencia cristal-vidrio.*
- `compute_specific_heat` (line 1048)
- `estimate_hbar_algorithmic` (line 1061)
- `compute_mutual_information` (line 1069)
- `check_extensivity` (line 1083)
- `compute_fisher_information_matrix` (line 1107)
- `compute_ricci_curvature` (line 1126)
- `calculate_carnot_efficiency` (line 1137)
- `compute_kappa` (line 1161)
- `compute_discretization_margin` (line 1187)
- `compute_local_complexity` (line 1191)
- `compute_alpha_purity` (line 1200)
- `compute_kappa_quantum` (line 1207)
- `compute_poynting_vector` (line 1224)
- `compute_all_metrics` (line 1246)
- `__init__` (line 1270)
- `compute_boundary_gradient` (line 1275) - *Approximate surface term: gradient concentrated on tensor boundaries.
For a matrix W ∈ ℝ^{m×n}, boundary = first/last row + first/last column.*
- `compute_bulk_gradient` (line 1290) - *Interior (volume) term: everything except boundary.*
- `run_green_backprop_step` (line 1296) - *Custom backward pass using Green-inspired decomposition.
Loss = MSE + λ_boundary * ||boundary_grad||²*
- `_get_boundary_mask` (line 1329) - *Returns a binary mask marking boundary elements of a tensor.*
- `train_with_green_cow` (line 1341)
- `load_checkpoint` (line 1372) - *Load checkpoint with robust deserialization handling.
Injects Config as UnifiedConfig alias to handle cross-script compatibility.*
- `migrate_checkpoint` (line 1407) - *Migrate checkpoint to standard format, extracting config if present.
Returns migrated state_dict compatible with BilinearStrassenModel.*
- `_migrate_dict` (line 1431)
- `_migrate_custom_format` (line 1443)
- `_migrate_coefs_format` (line 1467)
- `_migrate_encoder_format` (line 1475)
- `_migrate_standard_format` (line 1487)
- `__init__` (line 1491)
- `_load_all_checkpoints` (line 1503)
- `run_full_boltzmann_program` (line 1548)
- `phase1_molecular_hypothesis` (line 1575)
- `phase2_entropy_production` (line 1660)
- `phase3_extensivity_law` (line 1742)
- `phase4_quantum_basis_transform` (line 1796)
- `analyze_poynting_flow` (line 1849)
- `phase5_thermodynamic_analysis` (line 1882) - *PHASE 5: THERMODYNAMIC ANALYSIS con exponentes críticos y ecuación de estado.

FIX: Ahora calcula exponentes críticos y ecuación de estado para cada checkpoint.*
- `phase6_spectroscopic_analysis` (line 2011)
- `_plot_diffraction_pattern` (line 2097)
- `_save_superlattice_seed` (line 2131)
- `_classify_thermodynamic_phase` (line 2150)
- `_estimate_critical_temperature` (line 2162)
- `_verify_entropy_extensivity` (line 2174)
- `_plot_phase_diagram` (line 2188)
- `_plot_temperature_vs_purity` (line 2213)
- `_compute_entropy_simple` (line 2239)
- `_compute_entropy` (line 2265)
- `_compute_effective_volume` (line 2304)
- `_plot_parameter_distribution` (line 2322)
- `_simulate_training_trajectory` (line 2350)
- `_compute_generalization_entropy` (line 2361)
- `_fit_timescale` (line 2414)
- `_plot_entropy_production` (line 2424)
- `_verify_scaling` (line 2442)
- `_recursive_strassen` (line 2453)
- `_fit_extensivity` (line 2487)
- `_verify_extensivity_universality` (line 2501)
- `_plot_extensivity` (line 2505)
- `_find_broken_symmetries` (line 2518)
- `_measure_uncertainty` (line 2526)
- `_plot_uncertainty_distribution` (line 2537)
- `_print_executive_summary` (line 2558)
- `_save_results` (line 2587)
- `__init__` (line 2611)
- `_load_model` (line 2617)
- `run_full_analysis` (line 2634)
- `_assign_grade` (line 2656)
- `_save_report` (line 2668)
- `generate_batch` (line 373)
- `get_tensor` (line 1444)
- `model` (line 2415)
- `model` (line 2488)
- `convert_to_serializable` (line 2590)
- `dataloader` (line 2637)
- `sample_dataloader` (line 1906)
- `sample_dataloader` (line 2039)
- `dataloader` (line 1522)

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
