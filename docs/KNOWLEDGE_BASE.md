# Polyglot Codebase Knowledge Graph

> Generated offline by **readmenator**. Supports C, C++, Python, Go, Rust, JS/TS, Java, C#, Shell, PHP, Dart, GDScript, Nim, ASM.
> No LLMs. No tokens. Pure static analysis. See more [here](https://github.com/grisuno/ReadMenator)

**Total Files Parsed:** 60 | **Total Symbols Extracted:** 2071 | **Total Imports:** 638

## Structural Knowledge Map
> **Note:** The visual graph below has been intelligently pruned to the top 300 most relevant nodes to prevent rendering crashes. Full details of all 60 files are documented below.

```mermaid
graph TD
    classDef mod fill:#1e1e1e,stroke:#ff6666,stroke-width:2px,color:#fff;
    classDef cls fill:#2d2d2d,stroke:#4ec9b0,stroke-width:2px,color:#fff;
    classDef fn fill:#333,stroke:#dcdcaa,stroke-width:1px,color:#dcdcaa;
    classDef ext fill:#111,stroke:#666,stroke-dasharray:5 5,color:#aaa;
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
- `alloc_matrix` (line 15) `static float* alloc_matrix(int n)` - *Strassen Matrix Multiplication - C Implementation Author: grisun0  Compila: gcc -O3 -ffast-math -march=native -shared -fPIC -o libstrassen.so stras...*
- `matmul_standard` (line 20) `static void matmul_standard(float* C, float* A, float* B, int n)` - *#include <stdlib.h> #include <string.h> #include <stdio.h> #define THRESHOLD 64 /* Allocate matrix static float* alloc_matrix(int n) { return (floa...*
- `mat_add` (line 33) `static void mat_add(float* C, float* A, float* B, int n)` - */* Standard matrix multiplication for small matrices static void matmul_standard(float* C, float* A, float* B, int n) { memset(C, 0, n * n * sizeof...*
- `mat_sub` (line 41) `static void mat_sub(float* C, float* A, float* B, int n)` - *} } } } /* Add matrices: C = A + B static void mat_add(float* C, float* A, float* B, int n) { int nn = n * n; for (int i = 0; i < nn; i++) { C[i] =...*
- `extract_quadrant` (line 49) `static void extract_quadrant(float* Q, float* M, int n, int row, int col)` - *for (int i = 0; i < nn; i++) { C[i] = A[i] + B[i]; } } /* Subtract matrices: C = A - B static void mat_sub(float* C, float* A, float* B, int n) { i...*
- `insert_quadrant` (line 57) `static void insert_quadrant(float* M, float* Q, int n, int row, int col)` - *for (int i = 0; i < nn; i++) { C[i] = A[i] - B[i]; } } /* Extract quadrant from matrix static void extract_quadrant(float* Q, float* M, int n, int ...*
- `strassen_recursive` (line 65) `void strassen_recursive(float* C, float* A, float* B, int n)` - *for (int i = 0; i < h; i++) { memcpy(&Q[i * h], &M[(row + i) * n + col], h * sizeof(float)); } } /* Insert quadrant into matrix static void insert_...*
- `strassen_multiply` (line 164) `void strassen_multiply(float* C, float* A, float* B, int n)` - *insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C, C22, n, h, h); /* Free mem...*
- `standard_multiply` (line 169) `void standard_multiply(float* C, float* A, float* B, int n)` - */* Free memory free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free...*

**Macros:**
- `THRESHOLD` (line 11)

#### `strassen_optimal.c`
**Path:** `src/native/strassen_optimal.c`

**Functions:**
- `strassen_level` (line 18) `static void strassen_level(float* C, float* A, float* B, int n, 
                           float...` - *Uses in-place operations where possible and only applies Strassen for very large matrices where the asymptotic advantage overcomes overhead.  #incl...*
- `strassen_optimal` (line 130) `void strassen_optimal(float* C, float* A, float* B, int n)`

**Macros:**
- `STRASSEN_THRESHOLD` (line 15)

#### `strassen_turbo.c`
**Path:** `src/native/strassen_turbo.c`

**Functions:**
- `alloc_matrix` (line 25) `static inline float* alloc_matrix(int n)` - *Compile: gcc -O3 -ffast-math -march=native -fopenmp -mavx2 -shared -fPIC -o libstrassen_turbo.so strassen_turbo.c  #include <stdlib.h> #include <st...*
- `mat_add_avx` (line 30) `static void mat_add_avx(float* __restrict C, const float* __restrict A, 
                        ...` - *#include <stdio.h> #include <omp.h> #include <immintrin.h> #define THRESHOLD 128 #define BLOCK_SIZE 32 #define ALIGN 32 /* Aligned allocation stati...*
- `mat_sub_avx` (line 50) `static void mat_sub_avx(float* __restrict C, const float* __restrict A, 
                        ...` - *for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_add_ps(va, vb); _mm256_store...*
- `matmul_blocked_avx` (line 68) `static void matmul_blocked_avx(float* __restrict C, const float* __restrict A, 
                 ...` - *for (; i <= nn - 8; i += 8) { __m256 va = _mm256_load_ps(&A[i]); __m256 vb = _mm256_load_ps(&B[i]); __m256 vc = _mm256_sub_ps(va, vb); _mm256_store...*
- `extract_quadrant` (line 104) `static void extract_quadrant(float* __restrict Q, const float* __restrict M, 
                   ...` - *_mm256_storeu_ps(&C[i * n + j], vc); } for (; j < j_end; j++) { C[i * n + j] += a_ik * B[k * n + j]; } } } } } } } /* Extract quadrant*
- `insert_quadrant` (line 114) `static void insert_quadrant(float* __restrict M, const float* __restrict Q, 
                    ...` - *} } /* Extract quadrant static void extract_quadrant(float* __restrict Q, const float* __restrict M, int n, int row, int col) { int h = n / 2; #pra...*
- `strassen_turbo_recursive` (line 124) `void strassen_turbo_recursive(float* C, float* A, float* B, int n, int depth)` - *} } /* Insert quadrant static void insert_quadrant(float* __restrict M, const float* __restrict Q, int n, int row, int col) { int h = n / 2; #pragm...*
- `strassen_turbo` (line 261) `void strassen_turbo(float* C, float* A, float* B, int n)` - *insert_quadrant(C, C11, n, 0, 0); insert_quadrant(C, C12, n, 0, h); insert_quadrant(C, C21, n, h, 0); insert_quadrant(C, C22, n, h, h); /* Free mem...*
- `get_num_threads` (line 267) `int get_num_threads(void)` - *free(A11); free(A12); free(A21); free(A22); free(B11); free(B12); free(B21); free(B22); free(M1); free(M2); free(M3); free(M4); free(M5); free(M6);...*

**Macros:**
- `THRESHOLD` (line 19)
- `BLOCK_SIZE` (line 21)
- `ALIGN` (line 22)

### PY (56 files)

#### `app.py`
**Path:** `app.py`

**Classes:**
- `StrassenNet` (line 24) `class StrassenNet`

**Functions:**
- `__init__` (line 25) `def __init__(self, rank)`
- `forward` (line 31) `def forward(self, A, B)`

#### `batch_size.py`
**Path:** `batch_size.py`

**Classes:**
- `Configuration` (line 31) `class Configuration`
- `BilinearStrassenModel` (line 77) `class BilinearStrassenModel`
- `CheckpointMigrator` (line 100) `class CheckpointMigrator(ABC)`
- `CustomFormatMigrator` (line 110) `class CustomFormatMigrator(CheckpointMigrator)`
- `StandardFormatMigrator` (line 122) `class StandardFormatMigrator(CheckpointMigrator)`
- `CheckpointMigrationManager` (line 131) `class CheckpointMigrationManager`
- `StrassenDataGenerator` (line 147) `class StrassenDataGenerator`
- `CrystallographyMetrics` (line 156) `class CrystallographyMetrics`
- `PlanckConstantCalculator` (line 186) `class PlanckConstantCalculator`
- `BatchSizeThermodynamics` (line 217) `class BatchSizeThermodynamics`
- `StrassenCheckpointLoader` (line 262) `class StrassenCheckpointLoader`
- `StrassenPlanckAnalyzer` (line 292) `class StrassenPlanckAnalyzer`

**Functions:**
- `set_random_seed` (line 71) `def set_random_seed(seed)`
- `main` (line 341) `def main()`
- `__init__` (line 78) `def __init__(self, config)`
- `forward` (line 88) `def forward(self, a, b)`
- `get_coefficients` (line 91) `def get_coefficients(self)`
- `compute_lambda_effective` (line 94) `def compute_lambda_effective(self)`
- `can_migrate` (line 102) `def can_migrate(self, state_dict)`
- `migrate` (line 106) `def migrate(self, state_dict)`
- `can_migrate` (line 111) `def can_migrate(self, state_dict)`
- `migrate` (line 114) `def migrate(self, state_dict)`
- `can_migrate` (line 123) `def can_migrate(self, state_dict)`
- `migrate` (line 126) `def migrate(self, state_dict)`
- `__init__` (line 132) `def __init__(self)`
- `migrate_checkpoint` (line 135) `def migrate_checkpoint(self, path, device)`
- `generate_batch` (line 149) `def generate_batch(batch_size, config)`
- `compute_kappa` (line 158) `def compute_kappa(model, num_batches, config)`
- `compute_discretization_margin` (line 174) `def compute_discretization_margin(coeffs)`
- `compute_local_complexity` (line 178) `def compute_local_complexity(model, config)`
- `__init__` (line 187) `def __init__(self, metrics, training_metrics, config)`
- `calculate_all` (line 196) `def calculate_all(self)`
- `__init__` (line 218) `def __init__(self, model, h_bar, delta_struct, config)`
- `analyze_batch_size_spectrum` (line 224) `def analyze_batch_size_spectrum(self)`
- `_measure_gradients` (line 246) `def _measure_gradients(self, batch_size)`
- `__init__` (line 263) `def __init__(self, config)`
- `load` (line 267) `def load(self, path, device)`
- `extract_training_metrics` (line 281) `def extract_training_metrics(self, path)`
- `__init__` (line 293) `def __init__(self, config)`
- `analyze_checkpoint` (line 297) `def analyze_checkpoint(self, path, device)`
- `analyze_directory` (line 323) `def analyze_directory(self, directory, device, pattern)`

#### `boltzmann_experiments.py`
**Path:** `boltzmann_experiments.py`

**Classes:**
- `Config` (line 19) `class Config`
- `CheckpointLoadingError` (line 37) `class CheckpointLoadingError(Exception)`
- `ICheckpointLoader` (line 40) `class ICheckpointLoader(ABC)`
- `CheckpointLoader` (line 45) `class CheckpointLoader(ICheckpointLoader)`
- `CheckpointMigrator` (line 52) `class CheckpointMigrator`
- `BilinearStrassenModel` (line 118) `class BilinearStrassenModel`
- `CrystallographyMetrics` (line 137) `class CrystallographyMetrics`
- `DLProgram` (line 199) `class DLProgram`

**Functions:**
- `set_seed` (line 30) `def set_seed(seed)`
- `main` (line 935) `def main()`
- `_simulate_training_trajectory` (line 951) `def _simulate_training_trajectory(self, final_params, final_delta)`
- `_compute_generalization_entropy` (line 962) `def _compute_generalization_entropy(self, params, successful_ckpts)` - *Entropía de generalización con manejo robusto de dimensionalidad*
- `_fit_timescale` (line 1016) `def _fit_timescale(self, entropy_values)`
- `_plot_entropy_production` (line 1026) `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- `phase3_extensivity_law` (line 1044) `def phase3_extensivity_law(self)`
- `load_checkpoint` (line 42) `def load_checkpoint(self, path, device)`
- `load_checkpoint` (line 46) `def load_checkpoint(self, path, device)`
- `migrate_checkpoint` (line 54) `def migrate_checkpoint(raw_data)`
- `_format_direct_tensors` (line 70) `def _format_direct_tensors(tensor_dict)`
- `_migrate_dict` (line 92) `def _migrate_dict(state_dict)`
- `_migrate_encoder_format` (line 105) `def _migrate_encoder_format(state_dict)`
- `_migrate_coefs_format` (line 115) `def _migrate_coefs_format(state_dict)`
- `__init__` (line 119) `def __init__(self, n_slots)`
- `_initialize_symmetric` (line 126) `def _initialize_symmetric(self)`
- `forward` (line 131) `def forward(self, a, b)`
- `get_coefficients` (line 134) `def get_coefficients(self)`
- `compute_kappa` (line 139) `def compute_kappa(coeffs)` - *Classical kappa - will be inf for discrete states*
- `compute_delta` (line 156) `def compute_delta(coeffs)` - *Discretization error δ*
- `compute_local_complexity` (line 161) `def compute_local_complexity(coeffs)`
- `compute_alpha_purity` (line 169) `def compute_alpha_purity(coeffs)` - *Alpha purity: α = -log(δ), inverse temperature metric for discrete states*
- `compute_kappa_quantum` (line 178) `def compute_kappa_quantum(coeffs, hbar)` - *Quantum-regularized kappa for singular covariance states*
- `__init__` (line 200) `def __init__(self, checkpoint_dir, results_dir)`
- `_load_all_checkpoints` (line 207) `def _load_all_checkpoints(self)`
- `run_full_boltzmann_program` (line 247) `def run_full_boltzmann_program(self)`
- `_print_executive_summary` (line 267) `def _print_executive_summary(self, results)`
- `_save_results` (line 307) `def _save_results(self, results, filename)`
- `phase1_molecular_hypothesis` (line 328) `def phase1_molecular_hypothesis(self)`
- `_compute_entropy_simple` (line 435) `def _compute_entropy_simple(self, params)` - *Entropía simple sin KDE para datos de baja varianza*
- `_compute_entropy` (line 446) `def _compute_entropy(self, params)` - *Entropía con manejo robusto de covarianza*
- `_compute_effective_volume` (line 466) `def _compute_effective_volume(self, kde)`
- `_plot_parameter_distribution` (line 475) `def _plot_parameter_distribution(self, params, group_name, kde)`
- `phase2_entropy_production` (line 506) `def phase2_entropy_production(self)`
- `_simulate_training_trajectory` (line 592) `def _simulate_training_trajectory(self, final_params, final_delta)`
- `_compute_generalization_entropy` (line 604) `def _compute_generalization_entropy(self, params, successful_ckpts)` - *Entropía de generalización con manejo robusto de datos idénticos*
- `_fit_timescale` (line 691) `def _fit_timescale(self, entropy_values)`
- `_plot_entropy_production` (line 701) `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- `phase3_extensivity_law` (line 719) `def phase3_extensivity_law(self)`
- `_verify_scaling` (line 773) `def _verify_scaling(self, coeffs, N)`
- `_recursive_strassen` (line 783) `def _recursive_strassen(self, A, B, coeffs, N)`
- `_fit_extensivity` (line 812) `def _fit_extensivity(self, errors, sizes, purity)`
- `_verify_extensivity_universality` (line 824) `def _verify_extensivity_universality(self, results)`
- `_plot_extensivity` (line 828) `def _plot_extensivity(self, sizes, errors, purity, ckpt_name)`
- `phase4_quantum_basis_transform` (line 841) `def phase4_quantum_basis_transform(self)`
- `_find_broken_symmetries` (line 895) `def _find_broken_symmetries(self, coeffs)`
- `_measure_uncertainty` (line 903) `def _measure_uncertainty(self, coeffs, basis)`
- `_plot_uncertainty_distribution` (line 914) `def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)`
- `model` (line 1017) `def model(t, A, tau, C)`
- `convert_to_serializable` (line 310) `def convert_to_serializable(obj)`
- `model` (line 692) `def model(t, A, tau, C)`
- `model` (line 813) `def model(N, alpha, beta)`

#### `compute_gns_checkpoints.py`
**Path:** `compute_gns_checkpoints.py`

**Functions:**
- `estimate_gns` (line 11) `def estimate_gns(model, batch_size, num_batches)`
- `main` (line 37) `def main()`

#### `crystallography.py`
**Path:** `crystallography.py`

**Classes:**
- `Config` (line 25) `class Config`
- `BilinearStrassenModel` (line 44) `class BilinearStrassenModel`
- `CheckpointMigrator` (line 71) `class CheckpointMigrator`
- `StrassenDataGenerator` (line 155) `class StrassenDataGenerator`
- `SparsificationProtocol` (line 172) `class SparsificationProtocol`
- `CrystallographyMetrics` (line 206) `class CrystallographyMetrics`
- `StrassenDiffractionTest` (line 232) `class StrassenDiffractionTest`
- `BasinResilienceSpectrometer` (line 277) `class BasinResilienceSpectrometer`
- `CrystalPurityIndex` (line 345) `class CrystalPurityIndex`
- `StrassenCrystallographer` (line 416) `class StrassenCrystallographer`
- `LocalComplexity` (line 523) `class LocalComplexity`

**Functions:**
- `set_seed` (line 35) `def set_seed(seed)`
- `main` (line 542) `def main()`
- `__init__` (line 45) `def __init__(self, n_slots)`
- `_initialize_symmetric` (line 52) `def _initialize_symmetric(self)`
- `forward` (line 57) `def forward(self, a, b)`
- `get_coefficients` (line 60) `def get_coefficients(self)`
- `migrate_checkpoint` (line 73) `def migrate_checkpoint(path, device)`
- `_migrate_custom` (line 106) `def _migrate_custom(state_dict)` - *Maneja formatos custom U,V,W directos*
- `_migrate_encoder` (line 123) `def _migrate_encoder(state_dict)` - *Extracción de encoder.layers*
- `_migrate_standard` (line 147) `def _migrate_standard(state_dict)` - *Formato estándar U.weight, V.weight, W.weight*
- `generate_batch` (line 157) `def generate_batch(batch_size)`
- `verify_structure` (line 164) `def verify_structure(coeffs)`
- `__init__` (line 173) `def __init__(self, model)`
- `prune_to_target` (line 176) `def prune_to_target(self, target)`
- `discretize_weights` (line 192) `def discretize_weights(self, margin)`
- `compute_kappa` (line 208) `def compute_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin` (line 225) `def compute_discretization_margin(coeffs)`
- `__init__` (line 233) `def __init__(self, model)`
- `test_gauge_invariance` (line 236) `def test_gauge_invariance(self, n_samples)`
- `_functional_error` (line 262) `def _functional_error(self, test_coeffs)`
- `__init__` (line 278) `def __init__(self, model)`
- `measure_resilience_spectrum` (line 282) `def measure_resilience_spectrum(self, noise_levels)`
- `_test_noise_recovery` (line 293) `def _test_noise_recovery(self, sigma, n_trials)`
- `_apply_noise` (line 312) `def _apply_noise(self, sigma)`
- `_anneal_to_attractor` (line 317) `def _anneal_to_attractor(self, max_epochs)`
- `_estimate_critical_noise` (line 329) `def _estimate_critical_noise(self, results)`
- `__init__` (line 346) `def __init__(self, model, diffraction_results, resilience_results, metrics_results)`
- `compute` (line 359) `def compute(self)`
- `_assign_grade` (line 399) `def _assign_grade(self, index, delta)`
- `__init__` (line 417) `def __init__(self, checkpoint_path, device)`
- `run_full_analysis` (line 445) `def run_full_analysis(self)`
- `_save_report` (line 506) `def _save_report(self, report)`
- `compute` (line 525) `def compute(model)` - *Computa LC basado en Can't Stop Won't Stop paper*
- `dataloader_gen` (line 465) `def dataloader_gen()`

#### `dirac_polos_zeros.py`
**Path:** `dirac_polos_zeros.py`

**Classes:**
- `AnalysisConfig` (line 24) `class AnalysisConfig`
- `IModel` (line 54) `class IModel(Protocol)`
- `IChargeDistributionExtractor` (line 60) `class IChargeDistributionExtractor(Protocol)`
- `IDiracAnalyzer` (line 65) `class IDiracAnalyzer(Protocol)`
- `IFieldCalculator` (line 70) `class IFieldCalculator(Protocol)`
- `IFluxCalculator` (line 75) `class IFluxCalculator(Protocol)`
- `IStateSpaceExtractor` (line 80) `class IStateSpaceExtractor(Protocol)`
- `ITransferFunctionComputer` (line 85) `class ITransferFunctionComputer(Protocol)`
- `IPoleZeroAnalyzer` (line 90) `class IPoleZeroAnalyzer(Protocol)`
- `IFrequencyAnalyzer` (line 97) `class IFrequencyAnalyzer(Protocol)`
- `ITimeResponseAnalyzer` (line 104) `class ITimeResponseAnalyzer(Protocol)`
- `ICheckpointLoader` (line 110) `class ICheckpointLoader(Protocol)`
- `ICheckpointMigrator` (line 115) `class ICheckpointMigrator(Protocol)`
- `IVisualizer` (line 120) `class IVisualizer(Protocol)`
- `BilinearModel` (line 124) `class BilinearModel`
- `ChargeDistributionExtractor` (line 152) `class ChargeDistributionExtractor`
- `DiracDeltaAnalyzer` (line 160) `class DiracDeltaAnalyzer`
- `ElectricFieldCalculator` (line 192) `class ElectricFieldCalculator`
- `ElectricFluxCalculator` (line 225) `class ElectricFluxCalculator`
- `DivergenceCalculator` (line 248) `class DivergenceCalculator`
- `GaussLawVerifier` (line 253) `class GaussLawVerifier`
- `StateSpaceExtractor` (line 271) `class StateSpaceExtractor`
- `TransferFunctionComputer` (line 298) `class TransferFunctionComputer`
- `PoleZeroAnalyzer` (line 313) `class PoleZeroAnalyzer`
- `FrequencyResponseAnalyzer` (line 463) `class FrequencyResponseAnalyzer`
- `TimeResponseAnalyzer` (line 566) `class TimeResponseAnalyzer`
- `CheckpointLoader` (line 653) `class CheckpointLoader`
- `CheckpointMigrator` (line 661) `class CheckpointMigrator`
- `ChargeDistributionVisualizer` (line 710) `class ChargeDistributionVisualizer`
- `ElectricFieldVisualizer` (line 738) `class ElectricFieldVisualizer`
- `DivergenceVisualizer` (line 780) `class DivergenceVisualizer`
- `PoleZeroVisualizer` (line 809) `class PoleZeroVisualizer`
- `BodeVisualizer` (line 848) `class BodeVisualizer`
- `NyquistVisualizer` (line 899) `class NyquistVisualizer`
- `TimeResponseVisualizer` (line 936) `class TimeResponseVisualizer`
- `CombinedVisualizer` (line 966) `class CombinedVisualizer`
- `SystemAnalyzer` (line 1070) `class SystemAnalyzer`
- `AnalysisPipeline` (line 1288) `class AnalysisPipeline`

**Functions:**
- `main` (line 1545) `def main()`
- `forward` (line 55) `def forward(self, a, b)`
- `get_coefficients` (line 56) `def get_coefficients(self)`
- `extract` (line 61) `def extract(self, model)`
- `analyze` (line 66) `def analyze(self, charge_density)`
- `calculate` (line 71) `def calculate(self, dirac_data, eval_points)`
- `calculate` (line 76) `def calculate(self, electric_field, surface_points)`
- `extract` (line 81) `def extract(self, model)`
- `compute` (line 86) `def compute(self, A, B, C, D)`
- `analyze_stability` (line 91) `def analyze_stability(self)`
- `get_poles` (line 92) `def get_poles(self)`
- `get_zeros` (line 93) `def get_zeros(self)`
- `compute_bode` (line 98) `def compute_bode(self)`
- `compute_margins` (line 99) `def compute_margins(self)`
- `compute_nyquist` (line 100) `def compute_nyquist(self)`
- `compute_step` (line 105) `def compute_step(self)`
- `compute_impulse` (line 106) `def compute_impulse(self)`
- `load` (line 111) `def load(self, path, device)`
- `migrate` (line 116) `def migrate(self, raw_data)`
- `visualize` (line 121) `def visualize(self, data, output_path)`
- `__init__` (line 125) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 136) `def _initialize(self)`
- `forward` (line 141) `def forward(self, a, b)`
- `get_coefficients` (line 144) `def get_coefficients(self)`
- `extract` (line 153) `def extract(self, model)`
- `__init__` (line 161) `def __init__(self, config)`
- `analyze` (line 164) `def analyze(self, charge_density)`
- `__init__` (line 193) `def __init__(self, config)`
- `calculate` (line 196) `def calculate(self, dirac_data, eval_points)`
- `__init__` (line 226) `def __init__(self, config)`
- `calculate` (line 229) `def calculate(self, electric_field, surface_points)`
- `calculate` (line 249) `def calculate(self, electric_field)`
- `__init__` (line 254) `def __init__(self, config)`
- `verify` (line 257) `def verify(self, dirac_data, flux_data)`
- `extract` (line 272) `def extract(self, model)`
- `compute` (line 299) `def compute(self, A, B, C, D)`
- `__init__` (line 314) `def __init__(self, numerator, denominator, config)`
- `_compute` (line 323) `def _compute(self)`
- `get_poles` (line 334) `def get_poles(self)`
- `get_zeros` (line 337) `def get_zeros(self)`
- `analyze_stability` (line 340) `def analyze_stability(self)`
- `classify_poles` (line 378) `def classify_poles(self)`
- `compute_damping` (line 406) `def compute_damping(self)`
- `compute_time_constants` (line 445) `def compute_time_constants(self)`
- `__init__` (line 464) `def __init__(self, numerator, denominator, config)`
- `compute_bode` (line 474) `def compute_bode(self)`
- `compute_margins` (line 490) `def compute_margins(self)`
- `compute_nyquist` (line 516) `def compute_nyquist(self)`
- `evaluate_nyquist_stability` (line 535) `def evaluate_nyquist_stability(self, nyquist_data)`
- `__init__` (line 567) `def __init__(self, numerator, denominator, config)`
- `compute_step` (line 577) `def compute_step(self)`
- `compute_impulse` (line 589) `def compute_impulse(self)`
- `analyze_step_characteristics` (line 601) `def analyze_step_characteristics(self, step_data)`
- `load` (line 654) `def load(self, path, device)`
- `migrate` (line 662) `def migrate(self, raw_data)`
- `_migrate_dict` (line 674) `def _migrate_dict(self, state_dict)`
- `_migrate_custom_format` (line 683) `def _migrate_custom_format(self, state_dict)`
- `_migrate_coefs_format` (line 699) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 706) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 711) `def __init__(self, config)`
- `visualize` (line 714) `def visualize(self, data, output_path)`
- `__init__` (line 739) `def __init__(self, config)`
- `visualize` (line 742) `def visualize(self, data, output_path)`
- `__init__` (line 781) `def __init__(self, config)`
- `visualize` (line 784) `def visualize(self, data, output_path)`
- `__init__` (line 810) `def __init__(self, config)`
- `visualize` (line 813) `def visualize(self, data, output_path)`
- `__init__` (line 849) `def __init__(self, config)`
- `visualize` (line 852) `def visualize(self, data, output_path)`
- `__init__` (line 900) `def __init__(self, config)`
- `visualize` (line 903) `def visualize(self, data, output_path)`
- `__init__` (line 937) `def __init__(self, config)`
- `visualize` (line 940) `def visualize(self, data, output_path)`
- `__init__` (line 967) `def __init__(self, config)`
- `visualize` (line 970) `def visualize(self, data, output_path)`
- `__init__` (line 1071) `def __init__(self, checkpoint_path, config)`
- `_load_model` (line 1088) `def _load_model(self)`
- `analyze` (line 1104) `def analyze(self)`
- `_print_report` (line 1204) `def _print_report(self, results)`
- `__init__` (line 1289) `def __init__(self, config)`
- `process_checkpoint` (line 1300) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 1357) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 1385) `def generate_summary(self, all_results, output_dir)`
- `_compute_aggregate_statistics` (line 1406) `def _compute_aggregate_statistics(self, results)`
- `_generate_text_report` (line 1473) `def _generate_text_report(self, summary, output_dir)`

#### `ablation_8192.py`
**Path:** `experiments/ablation/ablation_8192.py`

*No symbols extracted*

#### `ablation_study.py`
**Path:** `experiments/ablation/ablation_study.py`

**Classes:**
- `BenchmarkResult` (line 27) `class BenchmarkResult`

**Functions:**
- `load_libraries` (line 54) `def load_libraries()` - *Cargar bibliotecas con manejo de errores*
- `run_openblas` (line 111) `def run_openblas(libs, A, B, C, n)` - *Ejecutar multiplicación con OpenBLAS*
- `run_strassen` (line 122) `def run_strassen(libs, name, func_name, A, B, C, n)` - *Ejecutar multiplicación con Strassen*
- `benchmark_single` (line 132) `def benchmark_single(libs, algo_name, func_name, A, B, C, C_ref, n, n_runs, warmup)` - *Benchmark una implementación*
- `run_ablation` (line 180) `def run_ablation(libs, sizes, n_runs, warmup)` - *Ejecutar ablación completa*
- `analyze_results` (line 233) `def analyze_results(results)` - *Analizar y presentar resultados*
- `main` (line 273) `def main()`
- `mean_time` (line 35) `def mean_time(self)`
- `std_time` (line 39) `def std_time(self)`
- `min_time` (line 43) `def min_time(self)`
- `max_time` (line 47) `def max_time(self)`
- `mean_gflops` (line 51) `def mean_gflops(self)`

#### `apendix_experiments.py`
**Path:** `experiments/apendix_experiments.py`

**Classes:**
- `StrassenOperator` (line 51) `class StrassenOperator`

**Functions:**
- `setup_matplotlib` (line 34) `def setup_matplotlib()`
- `generate_batch` (line 87) `def generate_batch(n, device)`
- `generate_test_set` (line 93) `def generate_test_set(n, device)`
- `compute_delta` (line 100) `def compute_delta(model)`
- `verify_strassen_structure` (line 117) `def verify_strassen_structure(U_disc, V_disc, W_disc, tolerance)`
- `compute_S_theta` (line 138) `def compute_S_theta(model)`
- `compute_gradient_covariance` (line 154) `def compute_gradient_covariance(model, batch_size, n_samples)`
- `train_with_logging` (line 187) `def train_with_logging(batch_size, total_epochs, lr, wd, symmetric_init, seed, log_interval)`
- `sparsify_and_discretize` (line 274) `def sparsify_and_discretize(model, batch_size)`
- `run_phase_diagram` (line 327) `def run_phase_diagram()`
- `run_batch_size_effect` (line 429) `def run_batch_size_effect()`
- `main` (line 526) `def main()`
- `__init__` (line 53) `def __init__(self, rank, symmetric_init)`
- `forward` (line 67) `def forward(self, A, B)`
- `slot_importance` (line 77) `def slot_importance(self)`
- `count_active` (line 83) `def count_active(self, threshold)`

#### `cache_analysis_v2.py`
**Path:** `experiments/cache_analysis_v2.py`

**Functions:**
- `cache_analysis` (line 7) `def cache_analysis()` - *Full memory analysis for training.

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

**Classes:**
- `Configuration` (line 41) `class Configuration`
- `Narrator` (line 95) `class Narrator`
- `SystemFingerprint` (line 152) `class SystemFingerprint`
- `ArithmeticDataset` (line 190) `class ArithmeticDataset(Dataset)` - *Dataset for arithmetic operations based on original implementation.

Generates (a, b) pairs and their product c = (a * b) mod MODULUS.
Each a, b, c is a one-hot vector of size MODULUS.

The BilinearModel learns: c = W @ ((U @ a) * (V @ b))
This is a lookup table for modular multiplication.*
- `BilinearModel` (line 238) `class BilinearModel` - *Bilinear model from original implementation.

Architecture:
- U: Linear(d_vocab, rank) -> [67, 8] weights
- V: Linear(d_vocab, rank) -> [67, 8] weights  
- W: Linear(rank, d_vocab) -> [8, 67] weights

Forward:
- a = x[:, 0] -> (batch, d_vocab)
- b = x[:, 1] -> (batch, d_vocab)
- m = U(a) * V(b) -> (batch, rank)
- logits = W(m) -> (batch, d_vocab)*
- `Task` (line 293) `class Task(ABC)`
- `MatrixMultiplicationTask` (line 311) `class MatrixMultiplicationTask(Task)`
- `ParityDataset` (line 354) `class ParityDataset(Dataset)` - *Dataset for parity task.*
- `ParityTask` (line 375) `class ParityTask(Task)`
- `GradientCovarianceProbe` (line 402) `class GradientCovarianceProbe` - *Analyzes gradient covariance to understand batch size effects.*
- `SpectralInterventionProbe` (line 465) `class SpectralInterventionProbe` - *Actively intervenes on condition number during training.*
- `AttractorLandscapeProbe` (line 488) `class AttractorLandscapeProbe` - *Analyzes attractor landscapes to understand failure modes.*
- `VolumeEstimator` (line 510) `class VolumeEstimator` - *Estimates volume of success basin in weight space.*
- `RobustnessTest` (line 524) `class RobustnessTest` - *Tests robustness against various perturbations.*
- `CheckpointManager` (line 581) `class CheckpointManager` - *Manages training checkpoints with configurable intervals.*
- `TrainingMetrics` (line 627) `class TrainingMetrics` - *Comprehensive metrics collection for training progress.*
- `ExperimentRunner` (line 700) `class ExperimentRunner` - *Main orchestration engine for training and evaluation.*
- `DiscretizationAnalyzer` (line 809) `class DiscretizationAnalyzer` - *Analyzes discretization quality and success probability.*
- `ExpansionVerifier` (line 851) `class ExpansionVerifier` - *Verifies zero-shot transfer to larger problem sizes.*
- `ExperimentPipeline` (line 879) `class ExperimentPipeline` - *Executes the complete battery of open-question experiments.*

**Functions:**
- `main` (line 1403) `def main()`
- `__init__` (line 89) `def __init__(self)`
- `__init__` (line 96) `def __init__(self, config)`
- `begin` (line 101) `def begin(self, experiment_name)`
- `progress` (line 109) `def progress(self, current, total, metrics)`
- `checkpoint` (line 116) `def checkpoint(self, epoch, loss, accuracy)`
- `result` (line 121) `def result(self, name, value, context)`
- `verdict` (line 126) `def verdict(self, hypothesis, evidence, conclusion)`
- `failure` (line 131) `def failure(self, reason, details)`
- `complete` (line 136) `def complete(self, summary)`
- `claim` (line 144) `def claim(self, statement, confidence)`
- `note` (line 148) `def note(self, observation)`
- `__init__` (line 153) `def __init__(self, config)`
- `capture` (line 156) `def capture(self)`
- `report` (line 174) `def report(self)`
- `__init__` (line 202) `def __init__(self, size, modulus)`
- `_generate_data` (line 207) `def _generate_data(self)`
- `__len__` (line 231) `def __len__(self)`
- `__getitem__` (line 234) `def __getitem__(self, idx)`
- `__init__` (line 253) `def __init__(self, d_vocab, rank, scale)`
- `forward` (line 265) `def forward(self, x)`
- `get_weights` (line 277) `def get_weights(self)`
- `get_U_weights` (line 283) `def get_U_weights(self)`
- `get_V_weights` (line 286) `def get_V_weights(self)`
- `get_W_weights` (line 289) `def get_W_weights(self)`
- `name` (line 295) `def name(self)`
- `d_vocab` (line 299) `def d_vocab(self)`
- `generate_dataset` (line 303) `def generate_dataset(self, size)`
- `verify` (line 307) `def verify(self, model, x, y)`
- `__init__` (line 312) `def __init__(self, modulus)`
- `name` (line 316) `def name(self)`
- `d_vocab` (line 319) `def d_vocab(self)`
- `generate_dataset` (line 322) `def generate_dataset(self, size)`
- `verify` (line 344) `def verify(self, model, x, y)`
- `__init__` (line 357) `def __init__(self, size, bit_length)`
- `_generate_data` (line 362) `def _generate_data(self)`
- `__len__` (line 368) `def __len__(self)`
- `__getitem__` (line 371) `def __getitem__(self, idx)`
- `__init__` (line 376) `def __init__(self, bit_length, modulus)`
- `name` (line 381) `def name(self)`
- `d_vocab` (line 384) `def d_vocab(self)`
- `generate_dataset` (line 387) `def generate_dataset(self, size)`
- `verify` (line 391) `def verify(self, model, x, y)`
- `__init__` (line 405) `def __init__(self, model)`
- `capture_gradients` (line 409) `def capture_gradients(self, dataloader, n_batches)`
- `compute_covariance` (line 429) `def compute_covariance(self)`
- `compute_condition_number` (line 436) `def compute_condition_number(self)`
- `compute_gradient_noise_scale` (line 442) `def compute_gradient_noise_scale(self, batch_size, learning_rate)`
- `analyze` (line 450) `def analyze(self, dataloader, batch_size, learning_rate)`
- `__init__` (line 468) `def __init__(self, model, target_kappa)`
- `spectral_regularizer` (line 472) `def spectral_regularizer(self)`
- `__init__` (line 491) `def __init__(self, model)`
- `count_local_minima` (line 494) `def count_local_minima(self, directions, losses)`
- `measure_basin_width` (line 501) `def measure_basin_width(self, weights, direction, n_points)`
- `classify_failure_mode` (line 505) `def classify_failure_mode(self, final_weights, initial_weights)`
- `__init__` (line 513) `def __init__(self, success_radius)`
- `estimate_volume_monte_carlo` (line 516) `def estimate_volume_monte_carlo(self, model_class, n_samples, success_checker)`
- `compute_fractal_dimension` (line 520) `def compute_fractal_dimension(self, trajectory)`
- `__init__` (line 527) `def __init__(self, model)`
- `add_gaussian_noise` (line 530) `def add_gaussian_noise(self, sigma)`
- `fgsm_attack` (line 536) `def fgsm_attack(self, x, y, epsilon)`
- `quantize_weights` (line 544) `def quantize_weights(self, bits)`
- `test_discretization_with_noise` (line 550) `def test_discretization_with_noise(self, sigma, checker)`
- `run_fragility_analysis` (line 560) `def run_fragility_analysis(self, sigma_values, checker)`
- `__init__` (line 584) `def __init__(self, config, experiment_name)`
- `save_checkpoint` (line 593) `def save_checkpoint(self, model, optimizer, epoch, metrics)`
- `load_checkpoint` (line 619) `def load_checkpoint(self, path, model, optimizer)`
- `__init__` (line 630) `def __init__(self)`
- `update` (line 642) `def update(self, train_loss, train_acc, test_loss, test_acc, kappa, grad_norm, weight_norm, disc_margin)`
- `detect_grokking` (line 660) `def detect_grokking(self, loss_threshold, test_loss_threshold, min_duration)`
- `progress_bar_string` (line 679) `def progress_bar_string(self, epoch, total_epochs)`
- `__init__` (line 703) `def __init__(self, config)`
- `train_epoch` (line 708) `def train_epoch(self, model, dataloader, optimizer)`
- `evaluate` (line 730) `def evaluate(self, model, dataloader)`
- `run_training` (line 748) `def run_training(self, model, train_loader, test_loader, experiment_name, epochs, batch_size, lr, wd, verbose)`
- `__init__` (line 812) `def __init__(self, config)`
- `compute_discretization_margin` (line 815) `def compute_discretization_margin(self, model)`
- `discretize_weights` (line 827) `def discretize_weights(self, model)`
- `check_strassen_structure` (line 832) `def check_strassen_structure(self, model, modulus)`
- `count_discretized_parameters` (line 839) `def count_discretized_parameters(self, model)`
- `__init__` (line 854) `def __init__(self, config)`
- `verify_expansion` (line 857) `def verify_expansion(self, model, task, sizes)`
- `__init__` (line 882) `def __init__(self, config)`
- `experiment_batch_size_mechanism` (line 891) `def experiment_batch_size_mechanism(self)` - *Experiment 1: Why batch size [24,128] works.*
- `experiment_kappa_intervention` (line 962) `def experiment_kappa_intervention(self)` - *Experiment 2: Active intervention on κ.*
- `experiment_failure_analysis` (line 1026) `def experiment_failure_analysis(self)` - *Experiment 3: Why 32% of runs fail.*
- `experiment_generalization` (line 1088) `def experiment_generalization(self)` - *Experiment 4: Generalization to other tasks.*
- `experiment_basin_volume` (line 1145) `def experiment_basin_volume(self)` - *Experiment 5: Basin volume estimation.*
- `experiment_hardware_reproducibility` (line 1158) `def experiment_hardware_reproducibility(self)` - *Experiment 6: Hardware reproducibility testing.*
- `experiment_fragility` (line 1217) `def experiment_fragility(self)` - *Experiment 7: Discretization fragility testing.*
- `run_all_experiments` (line 1332) `def run_all_experiments(self)`
- `_save_results` (line 1371) `def _save_results(self)`
- `_generate_summary` (line 1379) `def _generate_summary(self)`
- `checker` (line 1253) `def checker()`

#### `exp1_covariance_spectrometry.py`
**Path:** `experiments/extended_experiments/exp1_covariance_spectrometry.py`

**Classes:**
- `StrassenOperator` (line 48) `class StrassenOperator` - *Spectral operator for 2x2 matrix multiplication.
Tensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)*

**Functions:**
- `setup_matplotlib` (line 22) `def setup_matplotlib()`
- `generate_batch` (line 112) `def generate_batch(n, scale)` - *Generate batch of matrices.*
- `compute_gradient_covariance` (line 120) `def compute_gradient_covariance(model, batch_size, n_samples)` - *Compute the gradient covariance matrix Σₜ and its eigenvalues.

Returns:
    kappa: condition number λ_max / λ_min
    lambda_max: maximum eigenvalue
    lambda_min: minimum eigenvalue  
    trace: trace of covariance (total gradient energy)
    frobenius_norm: Frobenius norm of covariance*
- `load_checkpoint` (line 196) `def load_checkpoint(checkpoint_path)` - *Load model from checkpoint file.*
- `analyze_checkpoint` (line 222) `def analyze_checkpoint(checkpoint_path, batch_sizes, n_samples, n_runs)` - *Analyze a single checkpoint with multiple batch sizes.*
- `main` (line 277) `def main()` - *Main execution for Experiment 1.*
- `generate_visualization` (line 380) `def generate_visualization(results, output_dir)` - *Generate publication-quality figures.*
- `__init__` (line 54) `def __init__(self, rank)`
- `forward` (line 61) `def forward(self, A, B)`
- `get_all_parameters` (line 71) `def get_all_parameters(self)` - *Get all parameters as a single flattened vector.*
- `compute_per_sample_gradients` (line 78) `def compute_per_sample_gradients(self, A, B, C_true)` - *Compute per-sample gradients for covariance estimation.
Returns: gradients shape [batch_size, num_parameters]*

#### `exp2_noise_ablation.py`
**Path:** `experiments/extended_experiments/exp2_noise_ablation.py`

**Classes:**
- `StrassenOperator` (line 49) `class StrassenOperator` - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 25) `def setup_matplotlib()`
- `generate_batch` (line 99) `def generate_batch(n, scale)` - *Generate batch of matrices.*
- `compute_gradient_covariance_matrix` (line 107) `def compute_gradient_covariance_matrix(model, n_samples, batch_size)` - *Compute the gradient covariance matrix Σ.*
- `get_eigenbasis` (line 139) `def get_eigenbasis(covariance)` - *Get eigenvectors and eigenvalues of covariance matrix.*
- `load_checkpoint` (line 150) `def load_checkpoint(checkpoint_path)` - *Load model from checkpoint file.*
- `experiment_treatment_a_gradient_noise` (line 172) `def experiment_treatment_a_gradient_noise(model, noise_std, n_test)` - *Treatment A: Add noise to gradients DURING forward/backward pass.

This simulates the effect of gradient noise during training without actual retraining.*
- `experiment_treatment_b_weight_noise` (line 212) `def experiment_treatment_b_weight_noise(model, noise_std, n_test)` - *Treatment B: Noise on weights BEFORE evaluation (already done in paper).
This is the fallback mechanism test.*
- `experiment_treatment_c_structured_noise` (line 246) `def experiment_treatment_c_structured_noise(model, covariance, noise_std, n_test)` - *Treatment C: Structured noise by eigenvectors of Σ.

Tests whether damage is isotropic or aligned with gradient covariance directions.*
- `run_noise_ablation` (line 315) `def run_noise_ablation(checkpoint_path, noise_levels)` - *Run complete noise ablation experiment on a checkpoint.*
- `main` (line 348) `def main()` - *Main execution for Experiment 2.*
- `generate_visualization` (line 433) `def generate_visualization(results, output_dir)` - *Generate publication-quality figures.*
- `__init__` (line 54) `def __init__(self, rank)`
- `forward` (line 61) `def forward(self, A, B)`
- `get_all_parameters` (line 71) `def get_all_parameters(self)`
- `set_parameters` (line 77) `def set_parameters(self, new_params)` - *Set parameters from a flattened tensor.*
- `compute_loss` (line 85) `def compute_loss(self, A, B)` - *Compute MSE loss.*
- `compute_accuracy` (line 91) `def compute_accuracy(self, A, B, threshold)` - *Compute accuracy (proportion of predictions within threshold).*

#### `exp3_prospective_prediction.py`
**Path:** `experiments/extended_experiments/exp3_prospective_prediction.py`

**Classes:**
- `StrassenOperator` (line 56) `class StrassenOperator` - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 32) `def setup_matplotlib()`
- `generate_batch` (line 114) `def generate_batch(n, scale)` - *Generate batch of matrices.*
- `compute_kappa` (line 121) `def compute_kappa(model, n_samples, batch_size)` - *Compute condition number κ(Σ) of gradient covariance matrix.*
- `load_checkpoint` (line 166) `def load_checkpoint(checkpoint_path)` - *Load model from checkpoint file.*
- `simulate_early_prediction` (line 188) `def simulate_early_prediction(checkpoint_path, early_epoch_fraction)` - *Simulate the prospective prediction experiment.

Since we can't actually retrain, we use available checkpoints to simulate:
- "Early" checkpoint = first checkpoint in sequence
- "Final" checkpoint = last checkpoint in sequence

This tests if early-stage κ predicts final-stage success.*
- `run_prospective_prediction_experiment` (line 236) `def run_prospective_prediction_experiment(checkpoint_files)` - *Run the full prospective prediction experiment across all checkpoints.*
- `compute_roc_analysis` (line 252) `def compute_roc_analysis(predictions)` - *Compute ROC curve and AUC for κ as predictor of success.*
- `main` (line 313) `def main()` - *Main execution for Experiment 3.*
- `generate_visualization` (line 410) `def generate_visualization(results, predictions, output_dir)` - *Generate publication-quality figures.*
- `__init__` (line 59) `def __init__(self, rank)`
- `forward` (line 66) `def forward(self, A, B)`
- `get_all_parameters` (line 76) `def get_all_parameters(self)`
- `set_parameters` (line 82) `def set_parameters(self, new_params)`
- `count_active_slots` (line 89) `def count_active_slots(self, threshold)` - *Count active slots based on weight norms.*
- `compute_discretization_margin` (line 97) `def compute_discretization_margin(self)` - *Compute how close weights are to discrete values {-1, 0, 1}.
δ(θ) = mean(|w - round(w)|)*
- `is_grokked` (line 107) `def is_grokked(self, margin_threshold, active_slots_target)` - *Check if model has grokked (discretized with low error).*

#### `exp4_trajectory_perturbation.py`
**Path:** `experiments/extended_experiments/exp4_trajectory_perturbation.py`

**Classes:**
- `StrassenOperator` (line 51) `class StrassenOperator` - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 27) `def setup_matplotlib()`
- `generate_batch` (line 117) `def generate_batch(n, scale)` - *Generate batch of matrices.*
- `load_checkpoint` (line 124) `def load_checkpoint(checkpoint_path)` - *Load model from checkpoint file.*
- `simulate_trajectory_perturbation` (line 146) `def simulate_trajectory_perturbation(checkpoint_path, perturbations)` - *Simulate trajectory perturbation effects using available checkpoints.

Since we can't retrain, we simulate the effect of perturbations by:
1. Taking a "final" checkpoint as the target
2. Using earlier checkpoints to simulate "early training state"
3. Applying perturbations and measuring their effect*
- `main` (line 277) `def main()` - *Main execution for Experiment 4.*
- `generate_visualization` (line 398) `def generate_visualization(results, output_dir)` - *Generate publication-quality figures.*
- `__init__` (line 54) `def __init__(self, rank)`
- `forward` (line 61) `def forward(self, A, B)`
- `get_all_parameters` (line 71) `def get_all_parameters(self)`
- `set_parameters` (line 77) `def set_parameters(self, new_params)`
- `get_weight_norm` (line 84) `def get_weight_norm(self)` - *Get total L2 norm of all parameters.*
- `get_weight_direction` (line 91) `def get_weight_direction(self)` - *Get normalized weight vector direction.*
- `compute_gradient_norm` (line 96) `def compute_gradient_norm(self, A, B)` - *Compute norm of gradients.*
- `cosine_similarity` (line 110) `def cosine_similarity(self, other_params)` - *Compute cosine similarity between current weights and target weights.*
- `compute_metrics` (line 194) `def compute_metrics(model, name)` - *Compute evaluation metrics.*

#### `exp5_discreteness_attractors.py`
**Path:** `experiments/extended_experiments/exp5_discreteness_attractors.py`

*No symbols extracted*

#### `run_all_experiments.py`
**Path:** `experiments/extended_experiments/run_all_experiments.py`

**Classes:**
- `StrassenOperator` (line 48) `class StrassenOperator` - *Spectral operator for 2x2 matrix multiplication.*

**Functions:**
- `setup_matplotlib` (line 24) `def setup_matplotlib()`
- `generate_batch` (line 95) `def generate_batch(n, scale)`
- `load_checkpoint_robust` (line 101) `def load_checkpoint_robust(checkpoint_path, model)` - *Load checkpoint with multiple format fallback strategies.*
- `compute_gradient_covariance_safe` (line 135) `def compute_gradient_covariance_safe(model, batch_size, n_samples)` - *Compute κ(Σₜ) with numerical safety.*
- `run_all_experiments` (line 205) `def run_all_experiments()` - *Run all experiments.*
- `generate_summary_visualization` (line 496) `def generate_summary_visualization(results, output_dir)` - *Generate summary visualization.*
- `__init__` (line 51) `def __init__(self, rank)`
- `forward` (line 58) `def forward(self, A, B)`
- `get_all_parameters` (line 68) `def get_all_parameters(self)`
- `set_parameters` (line 74) `def set_parameters(self, new_params)`
- `count_active_slots` (line 81) `def count_active_slots(self, threshold)`
- `compute_discretization_margin` (line 88) `def compute_discretization_margin(self)`
- `compute_accuracy` (line 324) `def compute_accuracy()`

#### `validate2.py`
**Path:** `experiments/extended_experiments/validate2.py`

**Classes:**
- `ExperimentConfig` (line 60) `class ExperimentConfig` - *Configuración centralizada para todos los experimentos.

Sigue el principio de responsabilidad única - solo gestiona parámetros.
No usa magic numbers; todos los valores están definidos aquí.*
- `StrassenOperator` (line 126) `class StrassenOperator` - *Operador Strassen para multiplicación de matrices 2x2 vía descomposición tensorial.

El modelo representa el tensor de rango R:
C_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)

Donde:
- U, V: Coeficientes de combinación lineal (LC)
- W: Coeficientes de reconstrucción
- Sparsity (SP): Cuántos slots están activos*
- `StrassenDataGenerator` (line 225) `class StrassenDataGenerator` - *Generador de datos para multiplicación de matrices 2x2.*
- `LocalComplexityCalculator` (line 277) `class LocalComplexityCalculator` - *Calculadora de Complejidad Local basada en la varianza del gradiente.

LC = ||grad||^2 / N (Noise Scale normalizada)

Esta métrica captura la "dificultad" del batch actual y su relación
con el aprendizaje del modelo.*
- `GrokkingVerifier` (line 340) `class GrokkingVerifier` - *Verifica que un modelo ha grokkeado correctamente.*
- `IterativePruningEngine` (line 405) `class IterativePruningEngine` - *Motor para poda iterativa con fine-tuning completo.

Protocolo:
1. Calcular importancia de pesos (magnitud L1)
2. Podar p% de pesos menos importantes
3. Fine-tune por épocas especificadas
4. Chequear degradación δ
5. Si δ < threshold, continuar; si no, detener
6. Verificar discretización con δ < 0.1 (PUNTO A DEL REVISOR)

Este protocolo es CRUCIAL para verificar la hipótesis de cuenca discreta.*
- `LocalComplexityExperiment` (line 764) `class LocalComplexityExperiment` - *Experimento de Local Complexity entrenando desde cero.

Esto es CRUCIAL para responder al revisor (PUNTO B):
- Se entrena un modelo desde cero hasta grokking
- Se mide LC en cada época para capturar la transición de fase
- Si LC muestra un cambio alrededor del grokking, la métrica es útil
- Si LC permanece constante, la métrica NO captura la transición*
- `BalancedRunsGenerator` (line 890) `class BalancedRunsGenerator` - *Generador de runs balanceados para calcular AUC válido.

Esto es CRUCIAL para responder al revisor (PUNTO C):
- Entrenar múltiples modelos con diferentes hiperparámetros
- Algunos grokkean, otros no (condiciones variadas)
- Generar dataset balanceado para ROC/AUC

Si todos los samples son de una sola clase, AUC es indefinido.
Necesitamos mix de grokked + no-grokked para calcularlo.*
- `BootstrapStatistics` (line 1135) `class BootstrapStatistics` - *Generador de estadísticas con intervalos de confianza bootstrap.

Calcula:
- Curvas ROC con IC del 95%
- AUC con IC del 95%
- Kappa de Cohen con IC del 95%*
- `VisualizationGenerator` (line 1281) `class VisualizationGenerator` - *Generador de visualizaciones con estilo académico.*
- `ExperimentOrchestrator` (line 1646) `class ExperimentOrchestrator` - *Orquestador principal para todos los experimentos.

Coordina:
1. Carga de checkpoint grokkeado
2. Verificación de grokking
3. Experimento de Local Complexity
4. Protocolo de poda iterativa
5. Análisis ROC/AUC
6. Generación de visualizaciones*

**Functions:**
- `find_grokked_checkpoint` (line 2313) `def find_grokked_checkpoint()` - *Buscar checkpoint grokkeado en múltiples ubicaciones.

Returns:
    Path al archivo de checkpoint grokkeado*
- `analyze_checkpoints` (line 2353) `def analyze_checkpoints()` - *Analizar todos los checkpoints disponibles para encontrar el grokkeado.

Returns:
    Diccionario con métricas de cada checkpoint*
- `main` (line 2423) `def main()` - *Punto de entrada principal.*
- `__post_init__` (line 115) `def __post_init__(self)`
- `__init__` (line 139) `def __init__(self, rank)`
- `_initialize_weights` (line 150) `def _initialize_weights(self)` - *Inicializar pesos con consideración para grokking.*
- `forward` (line 156) `def forward(self, A, B)` - *Computar A @ B usando descomposición tensorial.

Args:
    A: Tensor de entrada de forma (batch, 2, 2)
    B: Tensor de entrada de forma (batch, 2, 2)
    
Returns:
    Tensor de salida de forma (batch, 2, 2)*
- `slot_importance` (line 183) `def slot_importance(self)` - *Importancia de cada slot basada en normas.*
- `count_active` (line 190) `def count_active(self, threshold)` - *Contar slots activos.*
- `compute_SP` (line 194) `def compute_SP(self)` - *Métrica de Sparsity. SP -> 0 significa máxima sparsity.*
- `get_state_dict` (line 202) `def get_state_dict(self)` - *Obtener estado completo para checkpointing.*
- `load_state_dict` (line 211) `def load_state_dict(self, state_dict)` - *Cargar estado completo desde checkpoint.*
- `__init__` (line 228) `def __init__(self, num_samples, matrix_size, seed)`
- `generate_matrix` (line 238) `def generate_matrix(self)` - *Generar matriz aleatoria con valores enteros.*
- `generate_data` (line 242) `def generate_data(self)` - *Generar pares de matrices y sus productos.*
- `get_train_test` (line 260) `def get_train_test(self, test_ratio)` - *Dividir en conjuntos de entrenamiento y prueba.*
- `__init__` (line 287) `def __init__(self, model, config)`
- `compute_lc` (line 292) `def compute_lc(self, batch_inputs, batch_targets)` - *Calcular LC para un batch específico.

LC = ||g||^2 / N_batch

Donde g es el gradiente de la pérdida respecto a los pesos.*
- `compute_batch_diversity` (line 326) `def compute_batch_diversity(self, batch_inputs)` - *Calcular diversidad del batch basada en varianza de activaciones.*
- `__init__` (line 343) `def __init__(self, config)`
- `verify` (line 347) `def verify(self, model, n_test)` - *Verificar que el operador ha grokkeado correctamente.

Returns:
    Tupla de (éxito, métricas)*
- `_generate_batch` (line 393) `def _generate_batch(self, n, scale)` - *Generar batch de matrices aleatorias.*
- `__init__` (line 420) `def __init__(self, config)`
- `get_weight_magnitudes` (line 426) `def get_weight_magnitudes(self, model)` - *Obtener magnitud absoluta de todos los pesos.*
- `compute_sparsity` (line 431) `def compute_sparsity(self, model)` - *Calcular porcentaje de pesos en cero.*
- `prune_percent` (line 437) `def prune_percent(self, model, percent)` - *Podar el porcentaje especificado de pesos menos importantes.

Returns: (num_pruned, current_sparsity)*
- `fine_tune` (line 457) `def fine_tune(self, model, train_data)` - *Fine-tune del modelo podado con métricas completas.*
- `_generate_batch` (line 510) `def _generate_batch(self, n, scale)` - *Generar batch de matrices aleatorias.*
- `run_protocol` (line 624) `def run_protocol(self, model, train_data)` - *Ejecutar protocolo completo de poda iterativa.

Returns:
    Diccionario con resultados completos*
- `__init__` (line 775) `def __init__(self, config)`
- `run_full_experiment` (line 779) `def run_full_experiment(self, target_epochs)` - *Ejecutar experimento completo de LC entrenando desde cero.*
- `_generate_batch` (line 882) `def _generate_batch(self, n, scale)`
- `__init__` (line 903) `def __init__(self, config)`
- `run_balanced_experiments` (line 907) `def run_balanced_experiments(self, n_runs)` - *Ejecutar multiples runs con condiciones disenhadas para producir mix.

Returns:
    Diccionario con resultados de todos los runs*
- `_train_single_run` (line 1021) `def _train_single_run(self, run_idx, config)` - *Entrenar un solo modelo con configuración específica.*
- `_compute_roc` (line 1099) `def _compute_roc(self, y_true, y_scores)` - *Calcular ROC/AUC básico.*
- `_generate_batch` (line 1127) `def _generate_batch(self, n, scale)`
- `__init__` (line 1145) `def __init__(self, config)`
- `compute_roc_with_ci` (line 1149) `def compute_roc_with_ci(self, y_true, y_scores)` - *Calcular curva ROC con intervalos de confianza bootstrap.

Returns:
    Diccionario con resultados ROC*
- `compute_kappa_with_ci` (line 1236) `def compute_kappa_with_ci(self, y_true, y_pred)` - *Calcular Kappa de Cohen con IC bootstrap.*
- `compute_accuracy_with_ci` (line 1260) `def compute_accuracy_with_ci(self, correct)` - *Calcular accuracy con IC binomial.*
- `__init__` (line 1284) `def __init__(self, style)`
- `plot_local_complexity` (line 1298) `def plot_local_complexity(self, epochs, lc_values, accuracy, save_path)` - *Graficar evolución de Local Complexity y Accuracy.*
- `plot_pruning_results` (line 1342) `def plot_pruning_results(self, pruning_data, save_path)` - *Graficar resultados de poda iterativa.*
- `plot_roc_with_ci` (line 1404) `def plot_roc_with_ci(self, roc_data, save_path)` - *Graficar curva ROC con intervalos de confianza.*
- `plot_balanced_runs_results` (line 1467) `def plot_balanced_runs_results(self, balanced_data, save_path)` - *Graficar resultados del experimento de runs balanceados.*
- `plot_discretization_results` (line 1556) `def plot_discretization_results(self, pruning_data, save_path)` - *Graficar resultados de discretizacion.*
- `__init__` (line 1659) `def __init__(self, config)`
- `find_grokked_checkpoint` (line 1689) `def find_grokked_checkpoint(self)` - *Buscar checkpoint grokkeado en múltiples ubicaciones.

Returns:
    Path al archivo de checkpoint o None si no se encuentra*
- `load_grokked_checkpoint` (line 1733) `def load_grokked_checkpoint(self, checkpoint_path)` - *Cargar checkpoint grokkeado y verificar que grokkeó.

Returns:
    Modelo cargado y verificado*
- `verify_checkpoint_is_grokked` (line 1768) `def verify_checkpoint_is_grokked(self)` - *Verificar que el checkpoint cargado realmente grokkeó.

Returns:
    Tupla de (es_grokked, métricas)*
- `run_local_complexity_experiment` (line 1795) `def run_local_complexity_experiment(self, epochs)` - *Ejecutar experimento de Local Complexity vs Época.

Nota: Como usamos un checkpoint ya grokked, esto mide la LC
durante el fine-tuning post-poda, no durante el grokking inicial.

Returns:
    Diccionario con historial de LC y accuracy*
- `run_lc_training_experiment` (line 1897) `def run_lc_training_experiment(self, epochs)` - *Ejecutar experimento de Local Complexity .

- Entrenar un modelo desde cero hasta grokking
- Medir LC en cada época para capturar la transición de fase
- Si LC muestra un cambio alrededor del grokking, la métrica es útil
- Si LC permanece constante, la métrica NO captura la transición

Returns:
    Diccionario con historial completo del experimento*
- `run_pruning_experiment` (line 1936) `def run_pruning_experiment(self)` - *Ejecutar protocolo de poda iterativa + fine-tuning.

Returns:
    Diccionario con resultados de poda*
- `run_balanced_runs_experiment` (line 1980) `def run_balanced_runs_experiment(self, n_runs)` - *Ejecutar experimento de runs balanceados (PUNTO C DEL REVISOR).

Esto es CRUCIAL para obtener un AUC valido:
- Entrenar multiples modelos con diferentes hiperparametros
- Algunos grokkean, otros no
- Generar dataset balanceado para ROC/AUC

Returns:
    Diccionario con resultados de todos los runs*
- `run_roc_analysis` (line 2043) `def run_roc_analysis(self)` - *Ejecutar análisis ROC/AUC con bootstrap.

Returns:
    Diccionario con resultados ROC*
- `_generate_batch` (line 2117) `def _generate_batch(self, n, scale)` - *Generar batch de matrices aleatorias.*
- `generate_summary_report` (line 2124) `def generate_summary_report(self)` - *Generar reporte de resumen en markdown.*
- `save_results` (line 2197) `def save_results(self)` - *Guardar todos los resultados.*
- `run_all_experiments` (line 2235) `def run_all_experiments(self, checkpoint_path)` - *Ejecutar suite completa de experimentos.*

#### `generate_figures.py`
**Path:** `experiments/generate_figures.py`

**Functions:**
- `setup_matplotlib_for_plotting` (line 15) `def setup_matplotlib_for_plotting()` - *Configure matplotlib and seaborn for proper rendering.*
- `generate_benchmark_figure` (line 63) `def generate_benchmark_figure()` - *Generate benchmark performance comparison plot.*
- `generate_ablation_figure` (line 124) `def generate_ablation_figure()` - *Generate ablation study visualization.*
- `load_checkpoint_weights` (line 219) `def load_checkpoint_weights()` - *Load all checkpoint files and extract weight tensors.*
- `generate_weight_geometry_figure` (line 258) `def generate_weight_geometry_figure()` - *Generate weight space geometry visualization.*
- `generate_phase_transition_figure` (line 354) `def generate_phase_transition_figure()` - *Generate phase transition analysis from checkpoint evolution.*
- `generate_coherence_figure` (line 465) `def generate_coherence_figure()` - *Generate cache coherence analysis visualization.*
- `generate_crystallization_figure` (line 534) `def generate_crystallization_figure()` - *Visualize the crystallization of Strassen coefficients.*
- `main` (line 623) `def main()`

#### `coherence_analysis.py`
**Path:** `experiments/statistics/coherence_analysis.py`

**Functions:**
- `strassen_numpy` (line 15) `def strassen_numpy(A, B, threshold)`
- `run_coherence_analysis` (line 42) `def run_coherence_analysis()`

#### `rigorous_experiment.py`
**Path:** `experiments/statistics/rigorous_experiment.py`

**Classes:**
- `ExperimentConfig` (line 58) `class ExperimentConfig` - *Complete hyperparameter specification for reproducibility*
- `ExperimentResult` (line 84) `class ExperimentResult` - *Single experiment result*
- `StrassenModel` (line 106) `class StrassenModel` - *Strassen-like bilinear model*

**Functions:**
- `generate_data` (line 131) `def generate_data(n_samples, seed)` - *Generate matrix multiplication dataset*
- `compute_discretization_error` (line 143) `def compute_discretization_error(model, values)` - *Compute mean distance to nearest discrete value*
- `compute_spectral_gap` (line 157) `def compute_spectral_gap(model)` - *Compute maximum spectral gap ratio*
- `run_single_experiment` (line 169) `def run_single_experiment(batch_size, seed, run_id, config)` - *Run a single controlled experiment*
- `run_full_experiment` (line 269) `def run_full_experiment(batch_sizes, n_seeds, n_runs_per_seed)` - *Run complete factorial experiment*
- `perform_anova` (line 306) `def perform_anova(results)` - *Perform full factorial ANOVA

Returns complete ANOVA table with SS, df, MS, F, p, η²*
- `print_anova_table` (line 401) `def print_anova_table(anova)` - *Print formatted ANOVA table*
- `fit_noise_model` (line 448) `def fit_noise_model(results)` - *Fit theoretical noise model:
Var(loss) = α/B + β·cache_miss(B) + γ

Compare to null model: Var(loss) = α/B + γ*
- `find_optimal_B` (line 519) `def find_optimal_B(results, n_bootstrap)` - *Find optimal batch size with bootstrap confidence interval*
- `generate_report` (line 555) `def generate_report(results, config)` - *Generate complete statistical report*
- `__init__` (line 108) `def __init__(self, config)`
- `forward` (line 124) `def forward(self, x)`
- `cache_miss_proxy` (line 467) `def cache_miss_proxy(B)`
- `full_model` (line 472) `def full_model(B, alpha, beta, gamma)`
- `null_model` (line 476) `def null_model(B, alpha, gamma)`
- `get_mean_error` (line 526) `def get_mean_error(data, B)`

#### `benchmark.py`
**Path:** `experiments/validation/benchmark.py`

**Functions:**
- `strassen_numpy` (line 15) `def strassen_numpy(A, B, threshold)` - *Strassen recursivo con NumPy para productos base.*
- `measure_single_sgemm` (line 49) `def measure_single_sgemm(n, threads)` - *Mide tiempo de un solo sgemm de tamaño n.*
- `run_planck_analysis` (line 59) `def run_planck_analysis()` - *Ejecuta el análisis del Límite de Planck.*

#### `validation_experiments.py`
**Path:** `experiments/validation_experiments.py`

**Functions:**
- `strassen_2x2` (line 47) `def strassen_2x2(A, B, U, V, W)` - *Compute 2x2 matrix multiplication using Strassen coefficients.*
- `strassen_recursive` (line 56) `def strassen_recursive(A, B, U, V, W, threshold)` - *Recursive Strassen for NxN matrices.*
- `test_uniqueness_via_permutation` (line 85) `def test_uniqueness_via_permutation()` - *Test that permuting slots produces equivalent computation.*
- `test_noise_stability` (line 125) `def test_noise_stability()` - *Test stability under Gaussian noise.*
- `test_expansion_sizes` (line 162) `def test_expansion_sizes()` - *Test expansion to larger sizes.*
- `simulate_grokking_dynamics` (line 184) `def simulate_grokking_dynamics()` - *Simulate grokking dynamics for visualization.*
- `compute_cache_math` (line 250) `def compute_cache_math()` - *Compute L3 cache requirements for different batch sizes.*
- `main` (line 294) `def main()` - *Run all validation experiments.*
- `convert_types` (line 313) `def convert_types(obj)`

#### `verify_checkpoints.py`
**Path:** `experiments/verify_checkpoints.py`

**Classes:**
- `StrassenBilinear` (line 25) `class StrassenBilinear`

**Functions:**
- `compute_delta` (line 51) `def compute_delta(model)`
- `verify_2x2` (line 68) `def verify_2x2(U, V, W, n_test)`
- `strassen_expand` (line 89) `def strassen_expand(A, B, U, V, W)`
- `verify_expansion` (line 126) `def verify_expansion(U, V, W, sizes)`
- `compute_S_theta` (line 152) `def compute_S_theta(model)`
- `load_checkpoint` (line 166) `def load_checkpoint(path)`
- `verify_checkpoint` (line 183) `def verify_checkpoint(checkpoint_path)`
- `run_noise_stability_test` (line 226) `def run_noise_stability_test(checkpoint_path, noise_levels)`
- `main` (line 249) `def main()`
- `__init__` (line 27) `def __init__(self, rank)`
- `forward` (line 34) `def forward(self, A, B)`
- `get_discrete_coefficients` (line 44) `def get_discrete_coefficients(self)`

#### `experimetn2.py`
**Path:** `experimetn2.py`

**Classes:**
- `StrassStrassenConfig` (line 53) `class StrassStrassenConfig` - *Immutable canonical configuration for the Strassen bilinear model.*
- `TrainingConfig` (line 71) `class TrainingConfig` - *Immutable training hyperparameters for crystallization runs.*
- `SuiteConfig` (line 84) `class SuiteConfig` - *Top-level suite configuration orchestrator.*
- `StrassStrassenModel` (line 95) `class StrassStrassenModel` - *Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.
Implements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.
U, V in R^{rank x input_dim}, W in R^{output_dim x rank}.*
- `ComplexStrassStrassenModel` (line 135) `class ComplexStrassStrassenModel` - *Genuinely complex-valued bilinear model for Altland-Zirnbauer spectral testing.
Uses complex parameters and arithmetic to break time-reversal symmetry.*
- `StrassenDataGenerator` (line 178) `class StrassenDataGenerator` - *Generates random 2x2 matrix pairs and their exact products.*
- `CheckpointManager` (line 190) `class CheckpointManager` - *Handles serialization and metadata tracking of model checkpoints.*
- `LevelSpacingRatioCalculator` (line 207) `class LevelSpacingRatioCalculator` - *Computes adjacent gap ratio 'r' of eigenvalues to determine spectral class.*
- `ExactHessianCalculator` (line 244) `class ExactHessianCalculator` - *Computes the mathematically exact full-rank Hessian of the model loss.*
- `SyntheticPlanckCalculator` (line 278) `class SyntheticPlanckCalculator` - *Computes the synthetic Planck constant from parameter and loss statistics.*
- `SuperpositionMetricCalculator` (line 328) `class SuperpositionMetricCalculator` - *Measures representation density via Sparse Autoencoder (SAE) bottleneck analysis.*
- `IExperiment` (line 385) `class IExperiment(ABC)` - *Abstract interface defining the execution protocol for experiments.*
- `Experiment1RicciMBLDuality` (line 393) `class Experiment1RicciMBLDuality(IExperiment)` - *Experiment 1: Ricci-MBL Duality.
Tracks geometric curvature (Hessian Ricci scalar) against adjacent gap ratio
during a long-duration optimization trajectory to capture crystallization.*
- `Experiment2AltlandZirnbauer` (line 457) `class Experiment2AltlandZirnbauer(IExperiment)` - *Experiment 2: Altland-Zirnbauer Symmetry Dial.
Uses complex parameters and arithmetic to trigger a true symmetry crossover
from real symmetric ensembles (GOE) to complex Hermitian (GUE).*
- `Experiment3ConformalIsomorphism` (line 520) `class Experiment3ConformalIsomorphism(IExperiment)` - *Experiment 3: Conformal Isomorphism.
Conducts mathematical stress-testing of fractional-linear Möbius
transformations vs scale-conformal transformations to map the physical
limits of learned tensor models.*
- `Experiment4CompressionFrontier` (line 570) `class Experiment4CompressionFrontier(IExperiment)` - *Experiment 4: Compression Frontier.
Tests the thermodynamic bound between parameter uncertainty (hbar) and
representation superposition (psi) under varied weight decay levels.*
- `Experiment5HolographicPruning` (line 624) `class Experiment5HolographicPruning(IExperiment)` - *Experiment 5: Holographic Pruning.
Distinguishes volumetric representation structures from structured boundary-state
mechanisms via element-wise versus slot-wise pruning.*
- `UnifiedSuite` (line 699) `class UnifiedSuite` - *Orchestrates and structures the execution of the five experiments.*

**Functions:**
- `main` (line 745) `def main()`
- `__post_init__` (line 63) `def __post_init__(self)`
- `__init__` (line 101) `def __init__(self, config)`
- `forward` (line 115) `def forward(self, A, B)`
- `get_coefficients` (line 125) `def get_coefficients(self)`
- `slot_importance` (line 128) `def slot_importance(self)`
- `__init__` (line 140) `def __init__(self, config, gamma)`
- `get_complex_tensors` (line 153) `def get_complex_tensors(self)`
- `forward` (line 160) `def forward(self, A, B)`
- `__init__` (line 180) `def __init__(self, config)`
- `generate_batch` (line 183) `def generate_batch(self, batch_size)`
- `save` (line 192) `def save(self, model, epoch, metrics, path)`
- `__init__` (line 209) `def __init__(self, tolerance)`
- `calculate_r_ratio` (line 212) `def calculate_r_ratio(self, eigenvalues)`
- `__init__` (line 246) `def __init__(self, config)`
- `compute_hessian` (line 249) `def compute_hessian(self, model, A, B, C_true)`
- `__init__` (line 280) `def __init__(self, noise_floor)`
- `calculate` (line 283) `def calculate(self, model, current_loss)`
- `__init__` (line 330) `def __init__(self, config)`
- `calculate` (line 333) `def calculate(self, model, datagen)`
- `run` (line 388) `def run(self, model)`
- `get_name` (line 390) `def get_name(self)`
- `__init__` (line 399) `def __init__(self, suite_config, datagen)`
- `get_name` (line 405) `def get_name(self)`
- `run` (line 408) `def run(self, model)`
- `__init__` (line 463) `def __init__(self, suite_config, datagen)`
- `get_name` (line 468) `def get_name(self)`
- `run` (line 471) `def run(self, model)`
- `__init__` (line 527) `def __init__(self, suite_config, datagen)`
- `get_name` (line 531) `def get_name(self)`
- `run` (line 534) `def run(self, model)`
- `__init__` (line 576) `def __init__(self, suite_config, datagen)`
- `get_name` (line 582) `def get_name(self)`
- `run` (line 585) `def run(self, model)`
- `__init__` (line 630) `def __init__(self, suite_config, datagen)`
- `get_name` (line 634) `def get_name(self)`
- `run` (line 637) `def run(self, model)`
- `__init__` (line 701) `def __init__(self, config)`
- `execute_all` (line 712) `def execute_all(self)`
- `loss_fn` (line 254) `def loss_fn(flat_param_tensor)`

#### `fermi.py`
**Path:** `fermi.py`

**Classes:**
- `FermiConfig` (line 19) `class FermiConfig`
- `IModel` (line 49) `class IModel(Protocol)`
- `IBlochWaveConstructor` (line 54) `class IBlochWaveConstructor(Protocol)`
- `IBandStructureCalculator` (line 59) `class IBandStructureCalculator(Protocol)`
- `IFermiLevelCalculator` (line 64) `class IFermiLevelCalculator(Protocol)`
- `IDensityOfStatesCalculator` (line 69) `class IDensityOfStatesCalculator(Protocol)`
- `IElectronicPropertiesCalculator` (line 74) `class IElectronicPropertiesCalculator(Protocol)`
- `IMetalInsulatorClassifier` (line 79) `class IMetalInsulatorClassifier(Protocol)`
- `BilinearModel` (line 83) `class BilinearModel`
- `BlochWaveConstructor` (line 111) `class BlochWaveConstructor`
- `BandStructureCalculator` (line 135) `class BandStructureCalculator`
- `FermiLevelCalculator` (line 215) `class FermiLevelCalculator`
- `DensityOfStatesCalculator` (line 285) `class DensityOfStatesCalculator`
- `ElectronicPropertiesCalculator` (line 306) `class ElectronicPropertiesCalculator`
- `MetalInsulatorClassifier` (line 368) `class MetalInsulatorClassifier`
- `CheckpointMigrator` (line 408) `class CheckpointMigrator`
- `FermiLevelAnalyzer` (line 458) `class FermiLevelAnalyzer`
- `FermiPipeline` (line 608) `class FermiPipeline`

**Functions:**
- `main` (line 766) `def main()`
- `get_coefficients` (line 50) `def get_coefficients(self)`
- `construct` (line 55) `def construct(self, weights, k)`
- `calculate` (line 60) `def calculate(self, model)`
- `calculate` (line 65) `def calculate(self, eigenvalues, num_electrons)`
- `calculate` (line 70) `def calculate(self, eigenvalues, energies)`
- `calculate` (line 75) `def calculate(self, eigenvalues, eigenvectors, fermi_level)`
- `classify` (line 80) `def classify(self, band_gap, dos_at_fermi)`
- `__init__` (line 84) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 95) `def _initialize(self)`
- `forward` (line 100) `def forward(self, a, b)`
- `get_coefficients` (line 103) `def get_coefficients(self)`
- `__init__` (line 112) `def __init__(self, config)`
- `construct` (line 115) `def construct(self, weights, k)`
- `__init__` (line 136) `def __init__(self, config)`
- `calculate` (line 140) `def calculate(self, model)`
- `_calculate_band_gap` (line 171) `def _calculate_band_gap(self, band_structure)`
- `_calculate_effective_masses` (line 188) `def _calculate_effective_masses(self, k_points, band_structure, valence_idx, conduction_idx)`
- `_is_direct_gap` (line 208) `def _is_direct_gap(self, band_structure, valence_idx, conduction_idx)`
- `__init__` (line 216) `def __init__(self, config)`
- `calculate` (line 219) `def calculate(self, eigenvalues, num_electrons)`
- `_calculate_chemical_potential` (line 240) `def _calculate_chemical_potential(self, eigenvalues, num_electrons)`
- `_find_chemical_potential_iterative` (line 253) `def _find_chemical_potential_iterative(self, eigenvalues, num_electrons, temperature, max_iter)`
- `_fermi_dirac` (line 273) `def _fermi_dirac(self, energy, mu, temperature)`
- `__init__` (line 286) `def __init__(self, config)`
- `calculate` (line 289) `def calculate(self, eigenvalues, energies)`
- `_gaussian` (line 302) `def _gaussian(self, x, mu, sigma)`
- `__init__` (line 307) `def __init__(self, config)`
- `calculate` (line 310) `def calculate(self, eigenvalues, eigenvectors, fermi_level)`
- `_calculate_kinetic_energy` (line 337) `def _calculate_kinetic_energy(self, occupied_states)`
- `_calculate_electronic_pressure` (line 347) `def _calculate_electronic_pressure(self, eigenvalues, fermi_level)`
- `_calculate_compressibility` (line 357) `def _calculate_compressibility(self, eigenvalues, fermi_level)`
- `__init__` (line 369) `def __init__(self, config)`
- `classify` (line 372) `def classify(self, band_gap, dos_at_fermi)`
- `classify_transport` (line 386) `def classify_transport(self, effective_masses, band_gap)`
- `migrate` (line 409) `def migrate(self, raw_data, device)`
- `_migrate_dict` (line 419) `def _migrate_dict(self, state_dict, device)`
- `_migrate_custom_format` (line 428) `def _migrate_custom_format(self, state_dict, device)`
- `_migrate_coefs_format` (line 447) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 454) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 459) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 472) `def _load_checkpoint(self)`
- `analyze` (line 495) `def analyze(self)`
- `_print_report` (line 550) `def _print_report(self, results)`
- `__init__` (line 609) `def __init__(self, config)`
- `process_checkpoint` (line 612) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 626) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 653) `def generate_summary(self, all_results, output_dir)`
- `_generate_text_report` (line 686) `def _generate_text_report(self, summary, output_dir)`
- `plot_band_structures` (line 724) `def plot_band_structures(self, all_results, output_dir)` - *Generate comparison plots of band structures across checkpoints.*

#### `full_seed_prospector.py`
**Path:** `full_seed_prospector.py`

**Classes:**
- `ExecutionMode` (line 46) `class ExecutionMode(Enum)`
- `UnifiedConfig` (line 52) `class UnifiedConfig` - *Immutable unified configuration for all execution modes.*
- `IMetricCalculator` (line 162) `class IMetricCalculator(ABC)` - *Interface for metric calculation strategies.*
- `ILossComponent` (line 170) `class ILossComponent(ABC)` - *Interface for loss function components.*
- `ICheckpointManager` (line 179) `class ICheckpointManager(ABC)` - *Interface for checkpoint management.*
- `ITrainingPhase` (line 195) `class ITrainingPhase(ABC)` - *Interface for training phase execution.*
- `StrassenOperator` (line 203) `class StrassenOperator` - *Spectral operator for 2x2 matrix multiplication with Strassen structure.*
- `DeltaCalculator` (line 313) `class DeltaCalculator(IMetricCalculator)` - *Calculate delta (discretization margin) metric.*
- `AccuracyCalculator` (line 333) `class AccuracyCalculator(IMetricCalculator)` - *Calculate matrix multiplication accuracy.*
- `KappaCalculator` (line 371) `class KappaCalculator` - *Calculate kappa (gradient covariance condition number).*
- `PerelmanEntropyCalculator` (line 455) `class PerelmanEntropyCalculator(IMetricCalculator)` - *Calculate Perelman W-entropy with adaptive tau.*
- `SparsityCalculator` (line 553) `class SparsityCalculator(IMetricCalculator)` - *Calculate sparsity metrics.*
- `GradientMetricsCalculator` (line 573) `class GradientMetricsCalculator` - *Calculate gradient statistics.*
- `ResilienceSpectrometer` (line 599) `class ResilienceSpectrometer` - *Measure structural stability under progressive pruning.*
- `ComprehensiveMetricsAggregator` (line 694) `class ComprehensiveMetricsAggregator` - *Aggregate all metrics including LC, SP, kappa, delta, h_bar_eff, T_eff.*
- `AdaptiveQuantizationLoss` (line 802) `class AdaptiveQuantizationLoss(ILossComponent)` - *Adaptive quantization loss pushing towards {-1, 0, 1}.*
- `RicciCurvaturePenalty` (line 854) `class RicciCurvaturePenalty(ILossComponent)` - *Ricci scalar curvature penalty.*
- `GeometricLossAggregator` (line 868) `class GeometricLossAggregator(ILossComponent)` - *Aggregate geometric loss components.*
- `CheckpointManager` (line 884) `class CheckpointManager(ICheckpointManager)` - *Manage model checkpointing.*
- `MatrixDataGenerator` (line 930) `class MatrixDataGenerator` - *Generate random matrix data for training.*
- `DynamicBatchSizeScheduler` (line 947) `class DynamicBatchSizeScheduler` - *Schedule batch size with cosine annealing.*
- `GlassDetector` (line 969) `class GlassDetector` - *Detect glass state (non-crystallizing seeds).*
- `ProspectorPhase` (line 1025) `class ProspectorPhase(ITrainingPhase)` - *Fast prospecting phase to identify crystal seeds.*
- `LongTrainingPhase` (line 1152) `class LongTrainingPhase(ITrainingPhase)` - *Long training phase with full thermodynamic metrics.*
- `ProgressiveSparsificationPhase` (line 1335) `class ProgressiveSparsificationPhase(ITrainingPhase)` - *Progressive sparsification guided by slot importance.*
- `CoefficientDiscretizationPhase` (line 1442) `class CoefficientDiscretizationPhase(ITrainingPhase)` - *Discretize coefficients to {-1, 0, 1}.*
- `StrassenVerifier` (line 1532) `class StrassenVerifier` - *Verify Strassen algorithm correctness.*
- `CanonicalStrassenProvider` (line 1588) `class CanonicalStrassenProvider` - *Provide canonical Strassen algorithm coefficients.*
- `SeedProspector` (line 1623) `class SeedProspector` - *Prospect seeds for crystallization candidates.*
- `LongTrainingPipeline` (line 1742) `class LongTrainingPipeline` - *Long training pipeline with all phases.*
- `LocalComplexityCalculator` (line 1849) `class LocalComplexityCalculator(IMetricCalculator)` - *Calculate Local Complexity as defined in the paper.*
- `SuperpositionCalculator` (line 1915) `class SuperpositionCalculator(IMetricCalculator)` - *Calculate Superposition metrics using sparse autoencoder.*
- `ThermodynamicMetricsCalculator` (line 1964) `class ThermodynamicMetricsCalculator(IMetricCalculator)` - *Calculate thermodynamic metrics: h_bar_eff and T_eff.*

**Functions:**
- `main` (line 2022) `def main()` - *Main entry point.*
- `calculate` (line 166) `def calculate(self)`
- `compute` (line 174) `def compute(self, model, loss_mse, epoch)`
- `save` (line 183) `def save(self, state, path)`
- `load` (line 187) `def load(self, path)`
- `should_checkpoint` (line 191) `def should_checkpoint(self)`
- `execute` (line 199) `def execute(self, model)`
- `__init__` (line 206) `def __init__(self, config)`
- `_initialize_strassen_structure` (line 220) `def _initialize_strassen_structure(self)` - *Initialize with bias towards canonical Strassen structure.*
- `forward` (line 276) `def forward(self, A, B)` - *Forward pass computing C = A @ B via low-rank factorization.*
- `slot_importance` (line 291) `def slot_importance(self)` - *Calculate importance of each rank slot.*
- `count_active` (line 298) `def count_active(self, threshold)` - *Count active slots above threshold.*
- `get_flat_parameters` (line 304) `def get_flat_parameters(self)` - *Get flattened parameter vector.*
- `get_parameter_count` (line 308) `def get_parameter_count(self)` - *Get total parameter count.*
- `__init__` (line 316) `def __init__(self, config)`
- `calculate` (line 319) `def calculate(self, model)` - *Calculate delta: mean squared distance to {-1, 0, 1}.*
- `__init__` (line 336) `def __init__(self, config)`
- `calculate` (line 339) `def calculate(self, model, C_pred, C_true, n_test)` - *Calculate accuracy percentage.*
- `__init__` (line 374) `def __init__(self, config)`
- `accumulate_gradient` (line 379) `def accumulate_gradient(self, model)` - *Accumulate current gradient vector.*
- `calculate_kappa` (line 393) `def calculate_kappa(self)` - *Calculate condition number of gradient covariance matrix.*
- `get_kappa_trend` (line 425) `def get_kappa_trend(self)` - *Determine kappa trend direction.*
- `is_crystallizing` (line 439) `def is_crystallizing(self)` - *Detect if system is in crystallization phase.*
- `reset` (line 449) `def reset(self)` - *Reset calculator state.*
- `__init__` (line 458) `def __init__(self, config)`
- `calculate` (line 465) `def calculate(self, model, loss, epoch, gradient_norm)` - *Calculate W-entropy with adaptive tau coupling to GNS.*
- `_calculate_log_W` (line 522) `def _calculate_log_W(self, tau, R, grad_f_sq, f, n_params)` - *Calculate log(W) with numerical stability.*
- `reset` (line 546) `def reset(self)` - *Reset calculator state.*
- `__init__` (line 556) `def __init__(self, config)`
- `calculate` (line 559) `def calculate(self, model)` - *Calculate active slots and sparsity.*
- `calculate` (line 577) `def calculate(model)` - *Calculate gradient norm statistics.*
- `__init__` (line 602) `def __init__(self, config)`
- `measure` (line 605) `def measure(self, model)` - *Measure resilience via progressive magnitude pruning.*
- `_prune_by_magnitude` (line 684) `def _prune_by_magnitude(self, model, sparsity)` - *Prune parameters by magnitude threshold.*
- `__init__` (line 697) `def __init__(self, config)`
- `compute_all` (line 711) `def compute_all(self, model, C_pred, C_true, loss, epoch, force_kappa, force_lc, force_sp)` - *Compute all available metrics.*
- `accumulate_gradient` (line 787) `def accumulate_gradient(self, model)` - *Accumulate gradient for kappa calculation.*
- `update_lr` (line 791) `def update_lr(self, lr)` - *Update current learning rate.*
- `reset` (line 795) `def reset(self)` - *Reset all calculators.*
- `__init__` (line 805) `def __init__(self, config)`
- `compute` (line 808) `def compute(self, model, loss_mse, epoch, kappa)` - *Compute quantization loss with adaptive weighting.*
- `_get_adaptive_weight` (line 835) `def _get_adaptive_weight(self, epoch, kappa)` - *Calculate adaptive weight based on epoch and kappa.*
- `__init__` (line 857) `def __init__(self, config)`
- `compute` (line 860) `def compute(self, model, loss_mse, epoch)` - *Compute Ricci curvature penalty.*
- `__init__` (line 871) `def __init__(self, config)`
- `compute` (line 876) `def compute(self, model, loss_mse, epoch, kappa)` - *Compute total geometric loss.*
- `__init__` (line 887) `def __init__(self, config)`
- `save` (line 893) `def save(self, state, path)` - *Save checkpoint to disk.*
- `load` (line 911) `def load(self, path)` - *Load checkpoint from disk.*
- `should_checkpoint` (line 919) `def should_checkpoint(self)` - *Check if checkpoint interval has elapsed.*
- `get_latest_checkpoint_path` (line 924) `def get_latest_checkpoint_path(self)` - *Get path to latest checkpoint if exists.*
- `__init__` (line 933) `def __init__(self, config, scale)`
- `generate_batch` (line 938) `def generate_batch(self, n)` - *Generate batch of random matrices.*
- `__init__` (line 950) `def __init__(self, config)`
- `get_batch_size` (line 953) `def get_batch_size(self, epoch)` - *Get batch size for current epoch.*
- `__init__` (line 972) `def __init__(self, config)`
- `should_stop` (line 977) `def should_stop(self, epoch, metrics)` - *Determine if training should stop due to glass state.*
- `__init__` (line 1028) `def __init__(self, config, seed)`
- `execute` (line 1037) `def execute(self, model)` - *Execute prospecting phase with all metrics visible.*
- `__init__` (line 1155) `def __init__(self, config)`
- `execute` (line 1164) `def execute(self, model)` - *Execute long training phase.*
- `__init__` (line 1338) `def __init__(self, config)`
- `execute` (line 1342) `def execute(self, model)` - *Execute sparsification phase.*
- `_final_refinement` (line 1407) `def _final_refinement(self, model, optimizer, slots_to_prune)` - *Final refinement after pruning.*
- `__init__` (line 1445) `def __init__(self, config)`
- `execute` (line 1448) `def execute(self, model)` - *Execute discretization phase.*
- `__init__` (line 1535) `def __init__(self, config)`
- `verify` (line 1538) `def verify(self, U, V, W, n_test)` - *Verify algorithm on random test matrices.*
- `get_canonical` (line 1592) `def get_canonical()` - *Return canonical Strassen algorithm matrices.*
- `__init__` (line 1626) `def __init__(self, config)`
- `prospect` (line 1631) `def prospect(self, total_attempts, start_seed)` - *Prospect multiple seeds for crystals.*
- `_set_seed` (line 1733) `def _set_seed(self, seed)` - *Set random seed for reproducibility.*
- `__init__` (line 1745) `def __init__(self, config)`
- `_signal_handler` (line 1758) `def _signal_handler(self, signum, frame)` - *Handle interrupt signal.*
- `_set_seed` (line 1763) `def _set_seed(self, seed)` - *Set random seed for reproducibility.*
- `run` (line 1771) `def run(self, resume_from, seed)` - *Run complete training pipeline.*
- `__init__` (line 1852) `def __init__(self, config)`
- `calculate` (line 1855) `def calculate(self, model)` - *Calculate LC as effective local dimensionality (paper definition).
Compatible fraction approach → log(volume) can be negative.*
- `__init__` (line 1918) `def __init__(self, config)`
- `_initialize_sae` (line 1923) `def _initialize_sae(self, input_dim, device)` - *Initialize sparse autoencoder if not already done.*
- `calculate` (line 1930) `def calculate(self, model)` - *Calculate superposition coefficient psi and effective features F.*
- `__init__` (line 1967) `def __init__(self, config)`
- `calculate` (line 1970) `def calculate(self, model, gradient_covariance)` - *Calculate effective Planck constant and temperature.*

#### `grain.py`
**Path:** `grain.py`

**Classes:**
- `StrassenConfig` (line 25) `class StrassenConfig`
- `IModel` (line 65) `class IModel(Protocol)`
- `IGrainBoundaryDetector` (line 71) `class IGrainBoundaryDetector(Protocol)`
- `ILayerAnalyzer` (line 76) `class ILayerAnalyzer(Protocol)`
- `IDislocationCalculator` (line 81) `class IDislocationCalculator(Protocol)`
- `IDomainFragmentationAnalyzer` (line 86) `class IDomainFragmentationAnalyzer(Protocol)`
- `ICheckpointManager` (line 91) `class ICheckpointManager(Protocol)`
- `ITrainingMonitor` (line 97) `class ITrainingMonitor(Protocol)`
- `BilinearStrassenModel` (line 102) `class BilinearStrassenModel`
- `LayerAnalyzer` (line 130) `class LayerAnalyzer`
- `GrainBoundaryDetector` (line 153) `class GrainBoundaryDetector`
- `DomainFragmentationAnalyzer` (line 243) `class DomainFragmentationAnalyzer`
- `CheckpointManager` (line 309) `class CheckpointManager`
- `TrainingMetricsTracker` (line 340) `class TrainingMetricsTracker`
- `StrassenTrainer` (line 407) `class StrassenTrainer`
- `GrainBoundaryAnalyzer` (line 547) `class GrainBoundaryAnalyzer`
- `GrainBoundaryPipeline` (line 726) `class GrainBoundaryPipeline`

**Functions:**
- `run_training` (line 833) `def run_training(seed, config)`
- `run_analysis` (line 838) `def run_analysis(checkpoint_dir, output_dir, n_latest, config)`
- `main` (line 845) `def main()`
- `to_dict` (line 57) `def to_dict(self)`
- `forward` (line 66) `def forward(self, a, b)`
- `get_coefficients` (line 67) `def get_coefficients(self)`
- `detect` (line 72) `def detect(self, model, pruning_level)`
- `analyze_layer` (line 77) `def analyze_layer(self, weights, layer_name)`
- `calculate` (line 82) `def calculate(self, layer_deltas)`
- `analyze` (line 87) `def analyze(self, model, pruning_level)`
- `save` (line 92) `def save(self, model, epoch, metrics, path)`
- `load` (line 93) `def load(self, path)`
- `update` (line 98) `def update(self, epoch, metrics)`
- `should_checkpoint` (line 99) `def should_checkpoint(self)`
- `__init__` (line 103) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize_symmetric` (line 114) `def _initialize_symmetric(self)`
- `forward` (line 119) `def forward(self, a, b)`
- `get_coefficients` (line 122) `def get_coefficients(self)`
- `analyze_layer` (line 131) `def analyze_layer(self, weights, layer_name)`
- `__init__` (line 154) `def __init__(self, config)`
- `detect` (line 158) `def detect(self, model, pruning_level)`
- `_prune_model` (line 185) `def _prune_model(self, model, sparsity)`
- `_calculate_dislocation` (line 194) `def _calculate_dislocation(self, layer_deltas)`
- `_analyze_fragmentation` (line 227) `def _analyze_fragmentation(self, layer_analysis)`
- `__init__` (line 244) `def __init__(self, config)`
- `analyze` (line 248) `def analyze(self, model, pruning_level)`
- `_calculate_coordination_loss` (line 269) `def _calculate_coordination_loss(self, layer_analysis)`
- `_estimate_domain_count` (line 282) `def _estimate_domain_count(self, layer_analysis)`
- `_calculate_coherence_length` (line 297) `def _calculate_coherence_length(self, layer_analysis)`
- `__init__` (line 310) `def __init__(self, config)`
- `save` (line 314) `def save(self, model, epoch, metrics, path)`
- `load` (line 332) `def load(self, path)`
- `should_save` (line 335) `def should_save(self)`
- `__init__` (line 341) `def __init__(self, config)`
- `update` (line 358) `def update(self, epoch, loss, accuracy, model, grain_result)`
- `get_current_metrics` (line 382) `def get_current_metrics(self)`
- `get_training_bar_string` (line 389) `def get_training_bar_string(self, epoch, total_epochs)`
- `__init__` (line 408) `def __init__(self, config, seed)`
- `_setup_signal_handlers` (line 438) `def _setup_signal_handlers(self)`
- `_signal_handler` (line 442) `def _signal_handler(self, signum, frame)`
- `_generate_batch` (line 447) `def _generate_batch(self)`
- `_compute_accuracy` (line 468) `def _compute_accuracy(self, pred, target)`
- `_save_checkpoint` (line 473) `def _save_checkpoint(self, interrupted)`
- `train` (line 486) `def train(self)`
- `__init__` (line 548) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 557) `def _load_checkpoint(self)`
- `_migrate_checkpoint` (line 580) `def _migrate_checkpoint(self, raw_data)`
- `_migrate_dict` (line 590) `def _migrate_dict(self, state_dict)`
- `_migrate_custom_format` (line 599) `def _migrate_custom_format(self, state_dict)`
- `_migrate_coefs_format` (line 618) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 625) `def _migrate_standard_format(self, state_dict)`
- `analyze` (line 628) `def analyze(self)`
- `_analyze_dislocation_evolution` (line 658) `def _analyze_dislocation_evolution(self, grain_results)`
- `_find_critical_pruning_level` (line 681) `def _find_critical_pruning_level(self, grain_results)`
- `_print_report` (line 687) `def _print_report(self, results)`
- `__init__` (line 727) `def __init__(self, config)`
- `process_checkpoint` (line 730) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 744) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 771) `def generate_summary(self, all_results, output_dir)`
- `_generate_text_report` (line 797) `def _generate_text_report(self, summary, output_dir)`

#### `gravity.py`
**Path:** `gravity.py`

**Classes:**
- `ThermodynamicConfig` (line 24) `class ThermodynamicConfig`
- `IModel` (line 59) `class IModel(Protocol)`
- `IOrderParameterCalculator` (line 65) `class IOrderParameterCalculator(Protocol)`
- `IEntropyCalculator` (line 70) `class IEntropyCalculator(Protocol)`
- `ISpecificHeatCalculator` (line 75) `class ISpecificHeatCalculator(Protocol)`
- `IGravitationalConstantCalculator` (line 80) `class IGravitationalConstantCalculator(Protocol)`
- `ILandauerConstantCalculator` (line 85) `class ILandauerConstantCalculator(Protocol)`
- `IHeisenbergUncertaintyCalculator` (line 90) `class IHeisenbergUncertaintyCalculator(Protocol)`
- `ILocalComplexityCalculator` (line 95) `class ILocalComplexityCalculator(Protocol)`
- `IBasinStabilityCalculator` (line 100) `class IBasinStabilityCalculator(Protocol)`
- `IZeroShotTransferCalculator` (line 105) `class IZeroShotTransferCalculator(Protocol)`
- `IConditionNumberCalculator` (line 110) `class IConditionNumberCalculator(Protocol)`
- `BilinearModel` (line 114) `class BilinearModel`
- `OrderParameterCalculator` (line 142) `class OrderParameterCalculator`
- `ConfigurationEntropyCalculator` (line 154) `class ConfigurationEntropyCalculator`
- `SpecificHeatCalculator` (line 171) `class SpecificHeatCalculator`
- `GravitationalConstantCalculator` (line 183) `class GravitationalConstantCalculator`
- `LandauerConstantCalculator` (line 239) `class LandauerConstantCalculator`
- `HeisenbergUncertaintyCalculator` (line 271) `class HeisenbergUncertaintyCalculator`
- `LocalComplexityCalculator` (line 310) `class LocalComplexityCalculator`
- `BasinStabilityCalculator` (line 327) `class BasinStabilityCalculator`
- `ZeroShotTransferCalculator` (line 365) `class ZeroShotTransferCalculator`
- `ConditionNumberCalculator` (line 398) `class ConditionNumberCalculator`
- `PhaseTransitionDetector` (line 454) `class PhaseTransitionDetector`
- `ThermodynamicAnalyzer` (line 505) `class ThermodynamicAnalyzer`
- `ThermodynamicPipeline` (line 837) `class ThermodynamicPipeline`

**Functions:**
- `main` (line 1114) `def main()`
- `forward` (line 60) `def forward(self, a, b)`
- `get_coefficients` (line 61) `def get_coefficients(self)`
- `calculate` (line 66) `def calculate(self, model)`
- `calculate` (line 71) `def calculate(self, model)`
- `calculate` (line 76) `def calculate(self, loss_history)`
- `calculate` (line 81) `def calculate(self, model, gradient_history)`
- `calculate` (line 86) `def calculate(self, entropy_change, energy_dissipated)`
- `calculate` (line 91) `def calculate(self, model, temperature)`
- `calculate` (line 96) `def calculate(self, model)`
- `calculate` (line 101) `def calculate(self, model, test_data)`
- `calculate` (line 106) `def calculate(self, model, target_size)`
- `calculate` (line 111) `def calculate(self, gradient_covariance)`
- `__init__` (line 115) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 126) `def _initialize(self)`
- `forward` (line 131) `def forward(self, a, b)`
- `get_coefficients` (line 134) `def get_coefficients(self)`
- `__init__` (line 143) `def __init__(self, config)`
- `calculate` (line 146) `def calculate(self, model)`
- `__init__` (line 155) `def __init__(self, config)`
- `calculate` (line 158) `def calculate(self, model)`
- `__init__` (line 172) `def __init__(self, config)`
- `calculate` (line 175) `def calculate(self, loss_history)`
- `__init__` (line 184) `def __init__(self, config)`
- `calculate` (line 187) `def calculate(self, model, gradient_history, loss_history, static_gradient)`
- `__init__` (line 240) `def __init__(self, config)`
- `calculate` (line 243) `def calculate(self, entropy_change, energy_dissipated, has_transition, transition_window)`
- `__init__` (line 272) `def __init__(self, config)`
- `calculate` (line 275) `def calculate(self, model, temperature, static_gradient)`
- `__init__` (line 311) `def __init__(self, config)`
- `calculate` (line 314) `def calculate(self, model)`
- `__init__` (line 328) `def __init__(self, config)`
- `calculate` (line 331) `def calculate(self, model, test_data)`
- `_prune_model` (line 355) `def _prune_model(self, model, sparsity)`
- `__init__` (line 366) `def __init__(self, config)`
- `calculate` (line 369) `def calculate(self, model, target_size)`
- `_kronecker_recursive` (line 391) `def _kronecker_recursive(self, matrix, power)`
- `__init__` (line 399) `def __init__(self, config)`
- `calculate` (line 403) `def calculate(self, gradient_history, static_gradient)`
- `__init__` (line 455) `def __init__(self, config)`
- `detect` (line 458) `def detect(self, loss_history, entropy_history)`
- `__init__` (line 506) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 525) `def _load_checkpoint(self)`
- `_migrate_checkpoint` (line 548) `def _migrate_checkpoint(self, raw_data)`
- `_migrate_dict` (line 560) `def _migrate_dict(self, state_dict)`
- `_migrate_custom_format` (line 569) `def _migrate_custom_format(self, state_dict)`
- `_migrate_coefs_format` (line 588) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 595) `def _migrate_standard_format(self, state_dict)`
- `_compute_static_gradient` (line 598) `def _compute_static_gradient(self)`
- `_generate_test_data` (line 619) `def _generate_test_data(self)`
- `analyze` (line 630) `def analyze(self)`
- `_determine_failure_mode` (line 733) `def _determine_failure_mode(self, delta, basin, kappa, transition)`
- `_print_report` (line 744) `def _print_report(self, results)`
- `__init__` (line 838) `def __init__(self, config)`
- `process_checkpoint` (line 841) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 855) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 882) `def generate_summary(self, all_results, output_dir)`
- `_compute_emergent_constants` (line 904) `def _compute_emergent_constants(self, results)`
- `_verify_universal_laws` (line 959) `def _verify_universal_laws(self, results)`
- `_compute_kappa_correlation` (line 993) `def _compute_kappa_correlation(self, results)`
- `_generate_text_report` (line 1023) `def _generate_text_report(self, summary, output_dir)`
- `extract_values` (line 908) `def extract_values(data_list, key_path)`

#### `grigori_perelmans_ricci_flow.py`
**Path:** `grigori_perelmans_ricci_flow.py`

**Classes:**
- `RicciConfig` (line 34) `class RicciConfig` - *Immutable configuration for Ricci Flow analysis.*
- `BilinearStrassenModel` (line 82) `class BilinearStrassenModel` - *Bilinear model f(A,B) = W((U*A) ⊙ (V*B)).*
- `StrassenDataGenerator` (line 128) `class StrassenDataGenerator`
- `CheckpointMigrator` (line 145) `class CheckpointMigrator(ABC)`
- `CustomFormatMigrator` (line 151) `class CustomFormatMigrator(CheckpointMigrator)`
- `StandardFormatMigrator` (line 166) `class StandardFormatMigrator(CheckpointMigrator)`
- `CheckpointMigrationManager` (line 172) `class CheckpointMigrationManager`
- `RicciFlowAnalyzer` (line 193) `class RicciFlowAnalyzer` - *Calculates Ricci curvature metrics using Hessian as Metric Tensor proxy.
In Perelman's flow, dg/dt = -2Ric. 
Here we analyze instantaneous state of Metric (Hessian).*
- `SingularityEngine` (line 310) `class SingularityEngine` - *Identifies 'necks' (singularities) in the geometry and proposes 'surgery' (pruning).
A 'neck' is a parameter direction with extreme curvature (Hessian eigenvalue).*
- `GeometricPlanckCalculator` (line 366) `class GeometricPlanckCalculator` - *Estimates effective Planck constant from Spectral Geometry.
Uses the Hessian eigenvalues to define an energy spectrum.*
- `RicciFlowAnalyzerPipeline` (line 430) `class RicciFlowAnalyzerPipeline` - *Complete analysis pipeline for Ricci Flow and Planck Estimation.
Orchestrates Hessian computation, Curvature Analysis, and Physics metrics.*

**Functions:**
- `set_random_seed` (line 70) `def set_random_seed(seed)`
- `main` (line 593) `def main()`
- `__init__` (line 86) `def __init__(self, config)`
- `_initialize` (line 94) `def _initialize(self)`
- `forward` (line 99) `def forward(self, a, b)`
- `get_coefficients` (line 102) `def get_coefficients(self)`
- `get_flat_params` (line 109) `def get_flat_params(self)` - *Return all parameters as a single flattened vector.*
- `set_flat_params` (line 114) `def set_flat_params(self, flat_params)` - *Set model parameters from a flattened vector.*
- `generate_batch` (line 130) `def generate_batch(batch_size, config)`
- `can_migrate` (line 147) `def can_migrate(self, state_dict)`
- `migrate` (line 149) `def migrate(self, state_dict)`
- `can_migrate` (line 152) `def can_migrate(self, state_dict)`
- `migrate` (line 154) `def migrate(self, state_dict)`
- `can_migrate` (line 167) `def can_migrate(self, state_dict)`
- `migrate` (line 169) `def migrate(self, state_dict)`
- `__init__` (line 173) `def __init__(self)`
- `migrate_checkpoint` (line 176) `def migrate_checkpoint(self, path, device)`
- `__init__` (line 199) `def __init__(self, model, config)`
- `compute_hessian` (line 203) `def compute_hessian(self, input_a, input_b, target_c)` - *Computes exact Hessian of loss w.r.t parameters.
H = d^2L / dtheta^2*
- `_loss_wrapper` (line 223) `def _loss_wrapper(self, flat_params, original_params, a, b, c)` - *Wrapper to compute loss from flat param vector.*
- `_compute_diagonal_hessian` (line 240) `def _compute_diagonal_hessian(self, a, b, c)` - *Approximation: Diagonal of Hessian (Gauss-Newton).*
- `analyze_curvature` (line 247) `def analyze_curvature(self, hessian)` - *Analyze Hessian spectrum to derive Ricci Scalar and Topological invariants.*
- `compute_heat_kernel_trace` (line 285) `def compute_heat_kernel_trace(self, eigenvalues, t)` - *Trace of Heat Kernel: Z(t) = Sum( exp(-lambda_i * t) ).
Relates to Partition Function in Quantum Mechanics.*
- `compute_topological_entropy` (line 296) `def compute_topological_entropy(self, eigenvalues)` - *von Neumann Entropy / Spectral Entropy.
S = - Sum( p_i * log(p_i) ) where p_i are normalized eigenvalue weights.*
- `__init__` (line 316) `def __init__(self, model, eigenvalues, config)`
- `detect_necks` (line 325) `def detect_necks(self, curvature_analysis)` - *Identify if the system is in a 'bottleneck' state.*
- `propose_surgery` (line 336) `def propose_surgery(self)` - *Propose parameters to 'cut' (prune) based on curvature heuristics.
In Strassen, the 'bias' slot (8th) often carries the 'noise' or singular connection.*
- `__init__` (line 371) `def __init__(self, eigenvalues, ricci_scalar, config)`
- `calculate` (line 376) `def calculate(self)`
- `_get_spectral_gap` (line 412) `def _get_spectral_gap(self)`
- `_compute_spectral_entropy` (line 420) `def _compute_spectral_entropy(self)`
- `__init__` (line 435) `def __init__(self, config)`
- `analyze_checkpoint` (line 439) `def analyze_checkpoint(self, checkpoint_path, device)` - *Perform complete analysis of a single checkpoint.*
- `analyze_directory` (line 527) `def analyze_directory(self, directory, device, pattern)`
- `_print_summary` (line 560) `def _print_summary(self, report)`

#### `hawking_radiation.py`
**Path:** `hawking_radiation.py`

**Classes:**
- `CustomUnpickler` (line 36) `class CustomUnpickler` - *Custom unpickler that handles unknown classes by creating dummy objects.

This solves the "Can't get attribute 'UnifiedConfig'" error by providing
fallback objects for any class that can't be found.*
- `HawkingConfiguration` (line 138) `class HawkingConfiguration` - *Immutable configuration for Hawking radiation analysis.*
- `IModel` (line 188) `class IModel(Protocol)`
- `BilinearStrassenModel` (line 197) `class BilinearStrassenModel` - *Bilinear model for Strassen matrix multiplication.*
- `RobustCheckpointMigrator` (line 234) `class RobustCheckpointMigrator` - *Enhanced checkpoint migrator that handles:
- Direct U, V, W tensors
- U_coefs format
- Standard .weight format
- Nested structures with config
- Encoder format
- State dicts within state dicts*
- `MetadataExtractor` (line 514) `class MetadataExtractor` - *Extracts metadata from various checkpoint formats.*
- `GravitationalConstantCalculator` (line 609) `class GravitationalConstantCalculator` - *Calculates effective gravitational constant G_alg.*
- `PlanckConstantCalculator` (line 676) `class PlanckConstantCalculator` - *Calculates effective Planck constant h_bar_eff.*
- `BoltzmannConstantCalculator` (line 768) `class BoltzmannConstantCalculator` - *Calculates effective Boltzmann constant k_B_eff.*
- `SpeedOfLightCalculator` (line 823) `class SpeedOfLightCalculator` - *Calculates effective speed of light c_eff.*
- `InformationalMassCalculator` (line 883) `class InformationalMassCalculator` - *Calculates effective mass M_eff.*
- `HorizonAreaCalculator` (line 939) `class HorizonAreaCalculator` - *Calculates effective area A_eff.*
- `HawkingRadiationCalculator` (line 991) `class HawkingRadiationCalculator` - *Calculates Hawking radiation metrics.*
- `RobustHawkingAnalyzer` (line 1166) `class RobustHawkingAnalyzer` - *Robust analyzer that handles exotic checkpoint formats.*
- `DummyClass` (line 64) `class DummyClass`

**Functions:**
- `load_checkpoint_robust` (line 93) `def load_checkpoint_robust(path, device)` - *Load checkpoint with robust handling of custom classes.

Tries multiple loading strategies in order:
1. Standard torch.load
2. Custom unpickler
3. Weights_only mode with manual extraction*
- `main` (line 1406) `def main()`
- `find_class` (line 44) `def find_class(self, module, name)` - *Override find_class to handle missing classes gracefully.*
- `_create_dummy_class` (line 62) `def _create_dummy_class(self, name)` - *Create a dummy class that can hold attributes.*
- `get_effective_input_dim` (line 172) `def get_effective_input_dim(self)`
- `get_total_parameters` (line 175) `def get_total_parameters(self)`
- `get_coefficients` (line 189) `def get_coefficients(self)`
- `forward` (line 190) `def forward(self, a, b)`
- `__init__` (line 200) `def __init__(self, config)`
- `_initialize` (line 211) `def _initialize(self)`
- `forward` (line 216) `def forward(self, a, b)`
- `get_coefficients` (line 219) `def get_coefficients(self)`
- `get_flat_parameters` (line 226) `def get_flat_parameters(self)`
- `migrate` (line 245) `def migrate(self, raw_data, device)` - *Main migration entry point with multiple strategies.*
- `_try_extract_state_dict` (line 269) `def _try_extract_state_dict(self, data, device)` - *Try standard state dict extraction methods.*
- `_try_nested_extraction` (line 288) `def _try_nested_extraction(self, data, device)` - *Try to extract from nested structures.*
- `_try_direct_tensor_extraction` (line 313) `def _try_direct_tensor_extraction(self, data, device)` - *Try to extract tensors directly from any structure.*
- `_reconstruct_from_tensors` (line 338) `def _reconstruct_from_tensors(self, tensors, device)` - *Reconstruct U, V, W from found tensors.*
- `_is_state_dict` (line 386) `def _is_state_dict(self, data)` - *Check if dict looks like a state dict.*
- `_migrate_dict` (line 398) `def _migrate_dict(self, state_dict, device)` - *Migrate a state dict to the expected format.*
- `_migrate_custom_format` (line 424) `def _migrate_custom_format(self, state_dict, device)`
- `_migrate_coefs_format` (line 449) `def _migrate_coefs_format(self, state_dict, device)`
- `_migrate_standard_format` (line 456) `def _migrate_standard_format(self, state_dict, device)`
- `_migrate_encoder_format` (line 466) `def _migrate_encoder_format(self, state_dict, device)` - *Handle encoder.layers style checkpoints.*
- `_migrate_prefixed_format` (line 496) `def _migrate_prefixed_format(self, state_dict, prefix, device)` - *Handle prefixed state dict keys.*
- `extract` (line 518) `def extract(checkpoint)` - *Extract all relevant metadata from checkpoint.*
- `_extract_delta` (line 571) `def _extract_delta(data, depth)` - *Recursively search for delta in nested structures.*
- `__init__` (line 612) `def __init__(self, config)`
- `calculate` (line 615) `def calculate(self, model, gradient, precomputed_delta)`
- `__init__` (line 679) `def __init__(self, config)`
- `calculate` (line 682) `def calculate(self, model, loss, precomputed_delta)`
- `__init__` (line 771) `def __init__(self, config)`
- `calculate` (line 774) `def calculate(self, model, loss, loss_history)`
- `__init__` (line 826) `def __init__(self, config)`
- `calculate` (line 829) `def calculate(self, model, h_bar, G_alg)`
- `__init__` (line 886) `def __init__(self, config)`
- `calculate` (line 889) `def calculate(self, model, G_alg, c_eff, h_bar)`
- `__init__` (line 942) `def __init__(self, config)`
- `calculate` (line 945) `def calculate(self, model, M_eff)`
- `__init__` (line 994) `def __init__(self, config)`
- `calculate_all` (line 1003) `def calculate_all(self, model, loss, loss_history, gradient, precomputed_delta)` - *Calculate all Hawking radiation metrics.*
- `_classify_state` (line 1151) `def _classify_state(self, delta, T_hawking)`
- `__init__` (line 1169) `def __init__(self, config)`
- `analyze_checkpoint` (line 1174) `def analyze_checkpoint(self, checkpoint_path)` - *Analyze a single checkpoint with robust error handling.*
- `_compute_gradient` (line 1227) `def _compute_gradient(self, model)` - *Compute gradient on random batch.*
- `_print_report` (line 1255) `def _print_report(self, results)` - *Print formatted report.*
- `analyze_directory` (line 1295) `def analyze_directory(self, checkpoint_dir, output_dir, pattern)` - *Analyze all checkpoints in directory.*
- `_generate_summary` (line 1342) `def _generate_summary(self, results, errors)` - *Generate aggregate summary.*
- `extract_tensors` (line 317) `def extract_tensors(obj, prefix)`
- `__init__` (line 65) `def __init__(self)`
- `__repr__` (line 72) `def __repr__(self)`
- `__getitem__` (line 75) `def __getitem__(self, key)`
- `keys` (line 78) `def keys(self)`
- `values` (line 81) `def values(self)`
- `items` (line 84) `def items(self)`
- `get` (line 87) `def get(self, key, default)`

#### `maxwell_strassen_analysis.py`
**Path:** `maxwell_strassen_analysis.py`

**Classes:**
- `MaxwellConfiguration` (line 42) `class MaxwellConfiguration` - *Configuration for Maxwellian analysis of Strassen crystals.

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
- `IModel` (line 116) `class IModel(Protocol)`
- `IGeometryMapper` (line 121) `class IGeometryMapper(Protocol)`
- `IMaxwellSolver` (line 126) `class IMaxwellSolver(Protocol)`
- `IDielectricAnalyzer` (line 132) `class IDielectricAnalyzer(Protocol)`
- `IPhaseClassifier` (line 137) `class IPhaseClassifier(Protocol)`
- `BilinearStrassenModel` (line 145) `class BilinearStrassenModel`
- `CheckpointMigrator` (line 171) `class CheckpointMigrator`
- `StrassenGeometryMapper` (line 227) `class StrassenGeometryMapper` - *Maps abstract weight vectors to a 3D Dielectric Lattice.

Strategy:
1. Flatten U, V, W into a single vector.
2. Populate a 3D grid (NxNxN) using a space-filling curve (Z-order) 
   or layer-wise assignment.
3. Compute Effective Charge Density ρ and Permittivity ε.

Crystal Hypothesis:
- Crystallized weights (discrete values) form ordered arrays.
- Glass weights (random) form noise.*
- `DielectricTensorAnalyzer` (line 288) `class DielectricTensorAnalyzer` - *Analyzes the anisotropy of the dielectric medium.

Glass: Isotropic (ε ~ scalar)
Crystal: Anisotropic (ε ~ tensor with specific principal axes)*
- `MaxwellScatteringSolver` (line 350) `class MaxwellScatteringSolver` - *Solves Maxwell's equations in the frequency domain via FFT.

Key Analysis:
1. Electrostatics: Solve ∇·(ε∇φ) = -ρ.
2. Scattering: Calculate Fourier Transform of Dielectric contrast.
   - Crystal: Sharp Bragg Peaks at k-vectors determined by lattice periodicity.
   - Glass: Broad diffuse scattering (Rayleigh).*
- `PhotonicEntropyCalculator` (line 466) `class PhotonicEntropyCalculator` - *Calculates the entropy of the electromagnetic field distribution.
S = -∑ p(E) log p(E)

Glass: High Entropy (Disordered field).
Crystal: Low Entropy (Ordered, localized modes).*
- `BandgapAnalyzer` (line 498) `class BandgapAnalyzer` - *Estimates if a photonic bandgap exists.
A bandgap implies certain frequencies cannot propagate.

We use the Fourier coefficients to estimate the gap.*
- `CrystalPhaseClassifier` (line 544) `class CrystalPhaseClassifier` - *Classifies the material phase (Crystal vs Glass) based on EM metrics.*
- `CheckpointManager` (line 605) `class CheckpointManager`
- `MaxwellVisualizer` (line 625) `class MaxwellVisualizer`
- `MaxwellAnalyzer` (line 677) `class MaxwellAnalyzer`

**Functions:**
- `main` (line 788) `def main()`
- `get_effective_input_dim` (line 103) `def get_effective_input_dim(self)`
- `get_total_parameters` (line 106) `def get_total_parameters(self)`
- `get_coefficients` (line 117) `def get_coefficients(self)`
- `map_weights_to_lattice` (line 122) `def map_weights_to_lattice(self, weights)`
- `solve_poisson` (line 127) `def solve_poisson(self, charge_density, permittivity)`
- `compute_scattering` (line 128) `def compute_scattering(self, permittivity)`
- `analyze_permittivity_tensor` (line 133) `def analyze_permittivity_tensor(self, permittivity)`
- `classify` (line 138) `def classify(self, metrics)`
- `__init__` (line 146) `def __init__(self, config)`
- `_initialize_weights` (line 156) `def _initialize_weights(self)`
- `forward` (line 161) `def forward(self, a, b)`
- `get_coefficients` (line 164) `def get_coefficients(self)`
- `migrate` (line 172) `def migrate(self, raw_data, config)`
- `_migrate_dict` (line 185) `def _migrate_dict(self, state_dict, config)`
- `_migrate_custom` (line 194) `def _migrate_custom(self, sd, config)`
- `_migrate_coefs` (line 203) `def _migrate_coefs(self, sd)`
- `_migrate_standard` (line 208) `def _migrate_standard(self, sd)`
- `_np` (line 217) `def _np(tensor)`
- `__init__` (line 241) `def __init__(self, config)`
- `map_weights_to_lattice` (line 244) `def map_weights_to_lattice(self, weights)` - *Returns:
    charge_density: 3D array of charge distribution.
    permittivity: 3D array of dielectric constants.*
- `__init__` (line 295) `def __init__(self, config)`
- `analyze_permittivity_tensor` (line 298) `def analyze_permittivity_tensor(self, permittivity)` - *Analyze the effective permittivity tensor of the medium.*
- `__init__` (line 360) `def __init__(self, config)`
- `solve_poisson` (line 363) `def solve_poisson(self, charge_density, permittivity)` - *Solves Poisson equation for Electric Potential φ.
Using Spectral Method (FFT):
∇²φ = -ρ / ε_0

For variable ε (heterogeneous medium), this is approximate.
We use the convolution theorem for the Green's function.*
- `compute_scattering` (line 401) `def compute_scattering(self, permittivity)` - *Compute the Scattering Amplitude S(k).
S(k) ∝ |FT(Δε)|²

Δε = ε(r) - ε_avg (Dielectric contrast)

Returns:
    Dict containing scattering intensity map and peak analysis.*
- `_find_peaks` (line 438) `def _find_peaks(self, intensity)` - *Detect sharp peaks indicative of crystallinity.*
- `__init__` (line 474) `def __init__(self, config)`
- `calculate` (line 477) `def calculate(self, potential, intensity)`
- `__init__` (line 505) `def __init__(self, config)`
- `analyze` (line 508) `def analyze(self, fourier_coeffs)`
- `__init__` (line 548) `def __init__(self, config)`
- `classify` (line 551) `def classify(self, em_metrics, purity_metrics)` - *Decision Logic:
1. If Purity Alpha > Threshold AND Discretization is high -> Pre-cursor.
2. If Anisotropy is High -> Crystal Order.
3. If Scattering shows Bragg Peaks -> Long Range Order.
4. If Entropy is Low -> Ordered System.*
- `__init__` (line 606) `def __init__(self, config)`
- `should_save` (line 610) `def should_save(self)`
- `save` (line 613) `def save(self, data, output_dir)`
- `__init__` (line 626) `def __init__(self, config)`
- `visualize_lattice` (line 629) `def visualize_lattice(self, permittivity, output_dir, name)`
- `visualize_scattering` (line 648) `def visualize_scattering(self, scattering_slice, output_dir, name)`
- `visualize_potential` (line 659) `def visualize_potential(self, potential, output_dir, name)`
- `__init__` (line 678) `def __init__(self, config)`
- `_calculate_purity_metrics` (line 690) `def _calculate_purity_metrics(self, weights)`
- `analyze_checkpoint` (line 701) `def analyze_checkpoint(self, checkpoint_path, output_dir)` - *Executes the full Maxwellian analysis pipeline on a single checkpoint.*
- `generate_report` (line 765) `def generate_report(self, results, output_dir)`

#### `mbl_analyzer.py`
**Path:** `mbl_analyzer.py`

**Classes:**
- `MBLConfiguration` (line 24) `class MBLConfiguration` - *Comprehensive configuration for MBL analysis of Strassen algorithm crystallization.
All parameters are centralized here following SOLID principles.*
- `IModel` (line 93) `class IModel(Protocol)` - *Protocol for models compatible with MBL analysis.*
- `ILevelSpacingCalculator` (line 100) `class ILevelSpacingCalculator(Protocol)` - *Protocol for level spacing ratio calculation.*
- `IParticipationRatioCalculator` (line 106) `class IParticipationRatioCalculator(Protocol)` - *Protocol for participation ratio calculation.*
- `ISyntheticPlanckCalculator` (line 112) `class ISyntheticPlanckCalculator(Protocol)` - *Protocol for synthetic Planck's constant calculation.*
- `IDiscretizationDialAnalyzer` (line 118) `class IDiscretizationDialAnalyzer(Protocol)` - *Protocol for discretization dial analysis.*
- `ICheckpointManager` (line 124) `class ICheckpointManager(Protocol)` - *Protocol for checkpoint management.*
- `ITrainingMetricsCollector` (line 132) `class ITrainingMetricsCollector(Protocol)` - *Protocol for collecting all training metrics.*
- `BilinearStrassenModel` (line 138) `class BilinearStrassenModel` - *Bilinear model for Strassen algorithm implementation.
Represents the 2x2 matrix multiplication with hidden dimension expansion.*
- `LevelSpacingRatioCalculator` (line 205) `class LevelSpacingRatioCalculator` - *Calculates the level spacing ratio r for MBL phase detection.

The ratio r_n = min(delta_n, delta_{n+1}) / max(delta_n, delta_{n+1})
where delta_n = E_{n+1} - E_n (energy level spacing).

References:
- Oganesyan & Huse (2008): r_WD ≈ 0.53 (Wigner-Dyson, thermal)
- Poisson statistics: r_P ≈ 0.386 (MBL/localized phase)*
- `ParticipationRatioCalculator` (line 321) `class ParticipationRatioCalculator` - *Calculates Inverse Participation Ratio (IPR) for localization analysis.

IPR = sum_i |c_i|^4 where c_i are coefficients in the chosen basis.
IPR = 1 for fully localized state, IPR = 1/N for fully delocalized state.

Used to quantify the 'crystallinity' of the weight distribution.*
- `SyntheticPlanckConstantCalculator` (line 420) `class SyntheticPlanckConstantCalculator` - *Calculates effective synthetic Planck's constant (hbar_eff) from model properties.

Based on the relation: hbar_eff ∝ 1 / sqrt(PR * Energy_Gap)
where PR is the Participation Ratio and Energy_Gap is the spectral gap.

This represents the quantum of action in the synthetic quantum system.*
- `DiscretizationDialAnalyzer` (line 486) `class DiscretizationDialAnalyzer` - *Analyzes the discretization parameter delta as a phase transition control.

The discretization delta measures how close weights are to discrete values.
It acts as a "dial" that controls the quantum-classical transition.

This implements the noise robustness test: applying Gaussian perturbations
and measuring when the energy gap collapses (loss of quantum protection).*
- `PurityIndexCalculator` (line 614) `class PurityIndexCalculator` - *Original purity calculation preserved exactly as in user's code.
Calculates the 'crystallinity' of the weight distribution.*
- `EffectiveTemperatureCalculator` (line 674) `class EffectiveTemperatureCalculator` - *Original temperature calculation preserved exactly.*
- `PhaseClassifier` (line 721) `class PhaseClassifier` - *Original phase classification preserved exactly.*
- `CheckpointMigrator` (line 746) `class CheckpointMigrator` - *Original checkpoint migration logic preserved exactly.*
- `MBLCheckpointManager` (line 800) `class MBLCheckpointManager` - *Manages checkpoint saving with 5-minute intervals and latest file maintenance.*
- `MBLMetricsCollector` (line 855) `class MBLMetricsCollector` - *Collects all MBL metrics for comprehensive training monitoring.*
- `MBLCheckpointAnalyzer` (line 962) `class MBLCheckpointAnalyzer` - *Comprehensive analyzer for MBL metrics from checkpoints.*
- `MBLAnalysisPipeline` (line 1102) `class MBLAnalysisPipeline` - *Main pipeline for processing checkpoints and generating reports.*

**Functions:**
- `main` (line 1227) `def main()`
- `get_effective_input_dim` (line 84) `def get_effective_input_dim(self)`
- `get_total_parameters` (line 87) `def get_total_parameters(self)`
- `get_coefficients` (line 95) `def get_coefficients(self)`
- `forward` (line 96) `def forward(self, a, b)`
- `calculate` (line 102) `def calculate(self, model)`
- `calculate` (line 108) `def calculate(self, model)`
- `calculate` (line 114) `def calculate(self, participation_ratio, energy_gap)`
- `analyze_robustness` (line 120) `def analyze_robustness(self, model, noise_levels)`
- `save_checkpoint` (line 126) `def save_checkpoint(self, model, epoch, metrics, loss_history, path)`
- `load_checkpoint` (line 128) `def load_checkpoint(self, path)`
- `collect` (line 134) `def collect(self, model, loss, epoch, loss_history)`
- `__init__` (line 143) `def __init__(self, config)`
- `_initialize_weights` (line 154) `def _initialize_weights(self)` - *Xavier initialization with symmetry constraint for U and V.*
- `forward` (line 160) `def forward(self, a, b)` - *Forward pass implementing bilinear multiplication.*
- `get_coefficients` (line 164) `def get_coefficients(self)` - *Returns weight matrices for analysis.*
- `get_flat_parameters` (line 172) `def get_flat_parameters(self)` - *Returns all parameters flattened for Hamiltonian construction.*
- `construct_hessian_approximation` (line 179) `def construct_hessian_approximation(self)` - *Constructs approximate Hessian matrix from weight correlations.
This serves as the 'Hamiltonian' for MBL analysis.*
- `__init__` (line 217) `def __init__(self, config)`
- `calculate` (line 220) `def calculate(self, model)` - *Calculate level spacing statistics from model weights.

Returns:
    Dictionary containing mean ratio, variance, and phase classification.*
- `_construct_hessian_from_weights` (line 267) `def _construct_hessian_from_weights(self, model)` - *Alternative Hessian construction for generic models.*
- `_compute_eigenvalues` (line 283) `def _compute_eigenvalues(self, hessian)` - *Compute sorted eigenvalues of the Hamiltonian.*
- `_calculate_spacing_ratios` (line 288) `def _calculate_spacing_ratios(self, spacings)` - *Calculate adjacent gap ratios r_n = min(s_n, s_{n+1}) / max(s_n, s_{n+1}).*
- `_classify_phase` (line 303) `def _classify_phase(self, mean_ratio)` - *Classify the quantum phase based on level spacing ratio.*
- `__init__` (line 331) `def __init__(self, config)`
- `calculate` (line 334) `def calculate(self, model)` - *Calculate participation ratios for all weight layers.

Returns:
    Dictionary containing global and layer-wise IPR metrics.*
- `_calculate_ipr` (line 380) `def _calculate_ipr(self, coefficients)` - *Calculate standard Inverse Participation Ratio.
IPR = sum_i |c_i|^4 / (sum_i |c_i|^2)^2*
- `_calculate_renyi_ipr` (line 395) `def _calculate_renyi_ipr(self, coefficients, q)` - *Calculate q-th order Rényi IPR.
I_q = sum_i |c_i|^{2q} / (sum_i |c_i|^2)^q*
- `_calculate_fractal_dimension` (line 409) `def _calculate_fractal_dimension(self, ipr, n)` - *Calculate fractal dimension D_q from IPR.
IPR ~ N^{-D_q} => D_q = -log(IPR) / log(N)*
- `__init__` (line 430) `def __init__(self, config)`
- `calculate` (line 433) `def calculate(self, participation_ratio, energy_gap)` - *Calculate synthetic Planck's constant.

Args:
    participation_ratio: Inverse participation ratio (measure of localization)
    energy_gap: Energy gap from spectrum (measure of quantum discreteness)

Returns:
    Synthetic hbar value representing the quantum scale of the system.*
- `calculate_from_model` (line 456) `def calculate_from_model(self, model, level_spacing_results, pr_results)` - *Comprehensive calculation from model and previous analyses.*
- `__init__` (line 497) `def __init__(self, config)`
- `calculate_base_discretization` (line 501) `def calculate_base_discretization(self, model)` - *Calculate the base discretization level from weight rounding error.*
- `analyze_robustness` (line 528) `def analyze_robustness(self, model, noise_levels)` - *Test robustness by applying noise and measuring gap collapse.

Args:
    model: The neural network model
    noise_levels: Tuple of noise magnitudes to test

Returns:
    Dictionary containing robustness metrics and phase transition points.*
- `_perturb_and_measure` (line 584) `def _perturb_and_measure(self, model, noise_level)` - *Apply noise to model and measure resulting metrics.*
- `_delta_to_alpha` (line 607) `def _delta_to_alpha(self, delta)` - *Convert discretization error to purity alpha.*
- `__init__` (line 620) `def __init__(self, config)`
- `calculate` (line 623) `def calculate(self, model)`
- `_compute_layer_purity` (line 652) `def _compute_layer_purity(self, weights)`
- `_delta_to_alpha` (line 658) `def _delta_to_alpha(self, delta)`
- `_assess_purity_quality` (line 663) `def _assess_purity_quality(self, alpha, variance)`
- `__init__` (line 679) `def __init__(self, config)`
- `calculate` (line 682) `def calculate(self, loss_history)`
- `__init__` (line 726) `def __init__(self, config)`
- `classify` (line 729) `def classify(self, alpha, temperature)`
- `migrate` (line 751) `def migrate(self, raw_data, device)`
- `_migrate_dict` (line 761) `def _migrate_dict(self, state_dict, device)`
- `_migrate_custom_format` (line 770) `def _migrate_custom_format(self, state_dict, device)`
- `_migrate_coefs_format` (line 789) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 796) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 805) `def __init__(self, config)`
- `should_save_checkpoint` (line 810) `def should_save_checkpoint(self)` - *Check if 5 minutes have elapsed since last checkpoint.*
- `save_checkpoint` (line 816) `def save_checkpoint(self, model, epoch, metrics, loss_history, checkpoint_dir)` - *Save checkpoint with all MBL metrics.*
- `load_checkpoint` (line 850) `def load_checkpoint(self, path)` - *Load checkpoint with automatic device placement.*
- `__init__` (line 860) `def __init__(self, config)`
- `collect` (line 870) `def collect(self, model, loss, epoch, loss_history)` - *Collect all metrics for the current training state.*
- `_classify_quantum_phase` (line 946) `def _classify_quantum_phase(self, level_spacing, hbar_results)` - *Classify the combined quantum phase.*
- `__init__` (line 967) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 975) `def _load_checkpoint(self)` - *Load and migrate checkpoint to model.*
- `analyze` (line 998) `def analyze(self)` - *Perform complete MBL analysis on checkpoint.*
- `_generate_summary` (line 1026) `def _generate_summary(self, metrics, robustness)` - *Generate executive summary of analysis.*
- `_print_report` (line 1043) `def _print_report(self, results)` - *Print formatted analysis report.*
- `__init__` (line 1107) `def __init__(self, config)`
- `process_checkpoint` (line 1110) `def process_checkpoint(self, checkpoint_path, output_dir)` - *Process single checkpoint and save results.*
- `process_directory` (line 1125) `def process_directory(self, checkpoint_dir, n_latest, output_dir)` - *Process multiple checkpoints from directory.*
- `generate_summary` (line 1155) `def generate_summary(self, all_results, output_dir)` - *Generate aggregate summary report.*
- `_generate_text_report` (line 1188) `def _generate_text_report(self, summary, output_dir)` - *Generate human-readable text report.*

#### `measure_strassen.py`
**Path:** `measure_strassen.py`

*No symbols extracted*

#### `menu.py`
**Path:** `menu.py`

**Functions:**
- `clear_screen` (line 304) `def clear_screen()`
- `print_header` (line 308) `def print_header(title, subtitle)`
- `print_wrapped` (line 318) `def print_wrapped(text, indent)`
- `wait_for_enter` (line 323) `def wait_for_enter()`
- `run_script` (line 332) `def run_script(entry)`
- `show_checkpoints` (line 362) `def show_checkpoints()`
- `show_results` (line 397) `def show_results()`
- `show_category` (line 433) `def show_category(cat)`
- `main_menu` (line 473) `def main_menu()`

#### `percolation_analysis.py`
**Path:** `percolation_analysis.py`

**Classes:**
- `PercolationConfiguration` (line 31) `class PercolationConfiguration`
- `IModel` (line 97) `class IModel(Protocol)`
- `NumpyModelWrapper` (line 101) `class NumpyModelWrapper`
- `_DummyObject` (line 148) `class _DummyObject` - *Transparent stand-in for any unpicklable class found inside a checkpoint.
Stores all keyword and positional constructor arguments as attributes so
that downstream dict-key lookups still work on objects that behave like
namespaces (e.g. UnifiedConfig, TrainingConfig, etc.).*
- `CheckpointMigrator` (line 264) `class CheckpointMigrator`
- `WeightGraphConstructor` (line 350) `class WeightGraphConstructor`
- `BondPercolationAnalyzer` (line 402) `class BondPercolationAnalyzer`
- `SitePercolationAnalyzer` (line 488) `class SitePercolationAnalyzer`
- `PruningPercolationAnalyzer` (line 542) `class PruningPercolationAnalyzer`
- `ClusterSizeDistributionAnalyzer` (line 725) `class ClusterSizeDistributionAnalyzer`
- `PercolationUniversalityAnalyzer` (line 775) `class PercolationUniversalityAnalyzer`
- `PercolationCheckpointManager` (line 808) `class PercolationCheckpointManager`
- `PercolationVisualizationEngine` (line 837) `class PercolationVisualizationEngine`
- `PercolationReportGenerator` (line 1049) `class PercolationReportGenerator`
- `PercolationAnalysisPipeline` (line 1139) `class PercolationAnalysisPipeline`
- `BilinearStrassenModel` (line 117) `class BilinearStrassenModel`

**Functions:**
- `_safe_torch_load` (line 165) `def _safe_torch_load(path)` - *Load a torch checkpoint that may contain unknown serialized classes
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
- `main` (line 1264) `def main()`
- `get_effective_input_dim` (line 82) `def get_effective_input_dim(self)`
- `get_total_parameters` (line 85) `def get_total_parameters(self)`
- `get_percolation_thresholds` (line 89) `def get_percolation_thresholds(self)`
- `get_coefficients` (line 98) `def get_coefficients(self)`
- `__init__` (line 102) `def __init__(self, weights)`
- `get_coefficients` (line 105) `def get_coefficients(self)`
- `get_flat_parameters` (line 108) `def get_flat_parameters(self)`
- `__init__` (line 155) `def __init__(self)`
- `__repr__` (line 161) `def __repr__(self)`
- `_patch_missing` (line 189) `def _patch_missing(exc)`
- `_cleanup` (line 230) `def _cleanup()`
- `migrate` (line 265) `def migrate(self, raw_data, config)`
- `_migrate_dict` (line 285) `def _migrate_dict(self, state_dict, config)`
- `_try_migrate_nested` (line 300) `def _try_migrate_nested(self, candidate, config)`
- `_migrate_custom` (line 310) `def _migrate_custom(self, sd, config)`
- `_migrate_coefs` (line 326) `def _migrate_coefs(self, sd)`
- `_migrate_standard` (line 331) `def _migrate_standard(self, sd)`
- `_np` (line 340) `def _np(tensor)`
- `__init__` (line 351) `def __init__(self, config)`
- `construct_adjacency_from_weights` (line 354) `def construct_adjacency_from_weights(self, weights)`
- `construct_weight_correlation_graph` (line 375) `def construct_weight_correlation_graph(self, weights)`
- `construct_slot_interaction_graph` (line 386) `def construct_slot_interaction_graph(self, weights)`
- `__init__` (line 403) `def __init__(self, config)`
- `analyze` (line 406) `def analyze(self, adjacency, thresholds)`
- `_susceptibility` (line 449) `def _susceptibility(self, sizes, n)`
- `_find_pc` (line 456) `def _find_pc(self, thresholds, res)`
- `_exponents` (line 462) `def _exponents(self, thresholds, res, pc)`
- `_fit_pl` (line 478) `def _fit_pl(x, y)`
- `__init__` (line 489) `def __init__(self, config)`
- `analyze` (line 492) `def analyze(self, weights, thresholds)`
- `__init__` (line 543) `def __init__(self, config)`
- `analyze` (line 546) `def analyze(self, weights)`
- `_d2a` (line 626) `def _d2a(self, delta)`
- `_kappa` (line 631) `def _kappa(self, wv)`
- `_hbar` (line 644) `def _hbar(self, wv)`
- `_teff` (line 656) `def _teff(self, wv)`
- `_lc` (line 659) `def _lc(self, wv, delta)`
- `_entropy` (line 669) `def _entropy(self, wv)`
- `_ipr` (line 680) `def _ipr(self, wv)`
- `_lsr` (line 687) `def _lsr(self, wv)`
- `_fractal` (line 699) `def _fractal(self, ipr, n)`
- `_phase` (line 704) `def _phase(self, alpha, temp, delta)`
- `__init__` (line 726) `def __init__(self, config)`
- `analyze_at_threshold` (line 729) `def analyze_at_threshold(self, adjacency, threshold)`
- `_tau` (line 753) `def _tau(self, sizes)`
- `_critical` (line 767) `def _critical(self, sizes, n)`
- `__init__` (line 776) `def __init__(self, config)`
- `classify_universality` (line 779) `def classify_universality(self, measured)`
- `__init__` (line 809) `def __init__(self, config)`
- `should_save` (line 814) `def should_save(self)`
- `save` (line 817) `def save(self, results, output_dir)`
- `load` (line 829) `def load(self, output_dir)`
- `__init__` (line 838) `def __init__(self, config)`
- `generate_all_figures` (line 841) `def generate_all_figures(self, results, output_dir)`
- `_plot_bond` (line 856) `def _plot_bond(self, data, out)`
- `_plot_pruning` (line 889) `def _plot_pruning(self, data, out)`
- `_plot_dashboard` (line 952) `def _plot_dashboard(self, data, out)`
- `_plot_site` (line 993) `def _plot_site(self, data, out)`
- `_plot_cluster` (line 1019) `def _plot_cluster(self, data, out)`
- `__init__` (line 1050) `def __init__(self, config)`
- `generate_text_report` (line 1053) `def generate_text_report(self, results, output_dir)`
- `generate_json_report` (line 1131) `def generate_json_report(self, results, output_dir)`
- `__init__` (line 1140) `def __init__(self, config)`
- `_load_weights` (line 1153) `def _load_weights(self, checkpoint_path)`
- `process_checkpoint` (line 1164) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 1212) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `_maybe_save` (line 1234) `def _maybe_save(self, results, output_dir)`
- `_comparative_summary` (line 1238) `def _comparative_summary(self, all_res, output_dir)`
- `__init__` (line 118) `def __init__(self, config)`
- `_initialize_weights` (line 127) `def _initialize_weights(self)`
- `forward` (line 132) `def forward(self, a, b)`
- `get_coefficients` (line 135) `def get_coefficients(self)`
- `get_flat_parameters` (line 141) `def get_flat_parameters(self)`

#### `plank.py`
**Path:** `plank.py`

**Classes:**
- `Configuration` (line 32) `class Configuration` - *Immutable configuration container following Single Responsibility Principle.*
- `BilinearStrassenModel` (line 127) `class BilinearStrassenModel` - *Bilinear model for Strassen matrix multiplication.
Implements f(A,B) = W((U*A) ⊙ (V*B)) where ⊙ is element-wise product.*
- `CheckpointMigrator` (line 194) `class CheckpointMigrator(ABC)` - *Abstract base for checkpoint migration strategies.*
- `CustomFormatMigrator` (line 208) `class CustomFormatMigrator(CheckpointMigrator)` - *Handles custom U,V,W direct formats.*
- `EncoderFormatMigrator` (line 237) `class EncoderFormatMigrator(CheckpointMigrator)` - *Handles encoder.layers format.*
- `StandardFormatMigrator` (line 270) `class StandardFormatMigrator(CheckpointMigrator)` - *Handles standard U.weight, V.weight, W.weight format.*
- `CheckpointMigrationManager` (line 284) `class CheckpointMigrationManager` - *Manages multiple migration strategies.*
- `StrassenDataGenerator` (line 341) `class StrassenDataGenerator` - *Generates training data for 2x2 matrix multiplication.*
- `CrystallographyMetrics` (line 384) `class CrystallographyMetrics` - *Computes crystallographic quality metrics for Strassen models.*
- `StrassenDiffractionTest` (line 483) `class StrassenDiffractionTest` - *Tests gauge invariance through permutation symmetry.*
- `BasinResilienceSpectrometer` (line 560) `class BasinResilienceSpectrometer` - *Measures basin of attraction through noise injection and recovery.*
- `CrystalPurityIndex` (line 677) `class CrystalPurityIndex` - *Computes normalized purity index from component metrics.*
- `PlanckConstantCalculator` (line 763) `class PlanckConstantCalculator` - *Calculates effective Planck constant from Strassen model parameters.

Maps crystallographic metrics to quantum thermodynamic quantities.*
- `StrassenCheckpointLoader` (line 936) `class StrassenCheckpointLoader` - *Loads and migrates Strassen checkpoints with fallback strategies.*
- `StrassenPlanckAnalyzer` (line 1014) `class StrassenPlanckAnalyzer` - *Complete analysis pipeline for Strassen checkpoints.

Orchestrates crystallographic analysis and Planck constant calculation.*
- `ReportGenerator` (line 1165) `class ReportGenerator` - *Generates reports and visualizations from analysis results.*

**Functions:**
- `set_random_seed` (line 114) `def set_random_seed(seed)` - *Set random seeds for reproducibility.*
- `parse_arguments` (line 1339) `def parse_arguments()` - *Parse command line arguments.*
- `create_config_from_args` (line 1387) `def create_config_from_args(args)` - *Create configuration from command line arguments.*
- `main` (line 1399) `def main()` - *Main execution entry point.*
- `__post_init__` (line 100) `def __post_init__(self)` - *Validate configuration parameters.*
- `__init__` (line 133) `def __init__(self, config)`
- `_initialize_symmetric` (line 143) `def _initialize_symmetric(self)` - *Initialize with Xavier uniform, symmetric U and V.*
- `forward` (line 149) `def forward(self, matrix_a, matrix_b)` - *Forward pass computing approximate matrix product.

Args:
    matrix_a: Flattened input matrix A [batch, INPUT_DIM]
    matrix_b: Flattened input matrix B [batch, INPUT_DIM]

Returns:
    Approximate product C = A @ B [batch, OUTPUT_DIM]*
- `get_coefficients` (line 166) `def get_coefficients(self)` - *Return current coefficient matrices.*
- `compute_lambda_effective` (line 174) `def compute_lambda_effective(self)` - *Compute effective lambda (confinement potential) from weight magnitudes.
Derived from weight decay interpretation as harmonic confinement.*
- `can_migrate` (line 198) `def can_migrate(self, state_dict)` - *Check if this strategy can handle the given state dict.*
- `migrate` (line 203) `def migrate(self, state_dict)` - *Migrate state dict to standard format.*
- `can_migrate` (line 211) `def can_migrate(self, state_dict)`
- `migrate` (line 214) `def migrate(self, state_dict)`
- `can_migrate` (line 240) `def can_migrate(self, state_dict)`
- `migrate` (line 243) `def migrate(self, state_dict)`
- `can_migrate` (line 273) `def can_migrate(self, state_dict)`
- `migrate` (line 276) `def migrate(self, state_dict)`
- `__init__` (line 287) `def __init__(self)`
- `migrate_checkpoint` (line 294) `def migrate_checkpoint(self, path, device)` - *Attempt to migrate checkpoint using available strategies.

Args:
    path: Path to checkpoint file
    device: Device to load tensors to

Returns:
    Migrated state dict or None if migration fails*
- `generate_batch` (line 345) `def generate_batch(batch_size, config)` - *Generate a batch of random matrix pairs and their products.

Returns:
    Tuple of (A_flat, B_flat, C_flat) where C = A @ B*
- `verify_structure` (line 366) `def verify_structure(coeffs, config)` - *Verify if coefficients represent valid Strassen structure.*
- `compute_kappa` (line 388) `def compute_kappa(model, num_batches, config)` - *Compute condition number of gradient covariance matrix.

High kappa indicates ill-conditioned optimization landscape.
Low kappa (approaching 1.0) indicates well-conditioned, crystalline structure.*
- `compute_discretization_margin` (line 429) `def compute_discretization_margin(coeffs)` - *Compute maximum deviation from nearest integer values.

Delta measures how close coefficients are to discrete (integer) values,
indicating crystalline structure formation.*
- `compute_local_complexity` (line 445) `def compute_local_complexity(model, config)` - *Compute local complexity based on active parameters.

From "Can't Stop Won't Stop" paper - measures effective parameter count.*
- `compute_all_metrics` (line 464) `def compute_all_metrics(model, config)` - *Compute all crystallographic metrics at once.*
- `__init__` (line 486) `def __init__(self, model, config)`
- `test_gauge_invariance` (line 490) `def test_gauge_invariance(self)` - *Test if model exhibits true Strassen structure through permutation invariance.

Genuine Strassen algorithm should have exactly one valid permutation (identity).*
- `_compute_functional_error` (line 531) `def _compute_functional_error(self, test_coeffs)` - *Compute functional error between original and permuted coefficients.*
- `__init__` (line 563) `def __init__(self, model, config)`
- `measure_resilience_spectrum` (line 570) `def measure_resilience_spectrum(self)` - *Measure resilience across multiple noise levels.

Returns spectrum showing critical noise level where recovery fails.*
- `_test_noise_recovery` (line 585) `def _test_noise_recovery(self, sigma)` - *Test recovery from noise level sigma.*
- `_apply_noise` (line 614) `def _apply_noise(self, sigma)` - *Apply Gaussian noise to model parameters.*
- `_anneal_to_attractor` (line 621) `def _anneal_to_attractor(self)` - *Anneal model back to attractor using fine-tuning.

Returns number of epochs needed for recovery.*
- `_estimate_critical_noise` (line 654) `def _estimate_critical_noise(self, results)` - *Estimate critical noise level where success rate drops below 50%.*
- `__init__` (line 680) `def __init__(self, metrics, diffraction_results, resilience_results, config)`
- `compute` (line 692) `def compute(self)` - *Compute normalized purity index and grade.*
- `_assign_grade` (line 745) `def _assign_grade(self, index, delta)` - *Assign crystallographic grade based on delta (primary indicator).*
- `__init__` (line 770) `def __init__(self, metrics, training_metrics, config)`
- `calculate_all` (line 789) `def calculate_all(self)` - *Execute all Planck constant calculation methods.*
- `_determine_regime_and_weights` (line 875) `def _determine_regime_and_weights(self)` - *Determine confinement regime and corresponding weights.*
- `_compute_derived_constants` (line 886) `def _compute_derived_constants(self, h_bar)` - *Compute derived Planck-scale constants.*
- `_compute_universe_comparison` (line 917) `def _compute_universe_comparison(self, h_bar)` - *Compare calculated constants with physical universe.*
- `__init__` (line 939) `def __init__(self, config)`
- `load` (line 943) `def load(self, checkpoint_path, device)` - *Load checkpoint into model instance.

Args:
    checkpoint_path: Path to checkpoint file
    device: Target device

Returns:
    Loaded model or None if loading fails*
- `extract_training_metrics` (line 986) `def extract_training_metrics(self, checkpoint_path)` - *Extract training metrics from checkpoint if available.*
- `__init__` (line 1021) `def __init__(self, config)`
- `analyze_checkpoint` (line 1025) `def analyze_checkpoint(self, checkpoint_path, device)` - *Perform complete analysis of a single checkpoint.

Args:
    checkpoint_path: Path to checkpoint file
    device: Computation device

Returns:
    Complete analysis report*
- `analyze_directory` (line 1104) `def analyze_directory(self, directory, device, pattern)` - *Analyze all checkpoints in a directory.

Args:
    directory: Directory containing checkpoints
    device: Computation device
    pattern: File pattern to match

Returns:
    List of analysis reports*
- `_print_summary` (line 1146) `def _print_summary(self, report)` - *Print formatted summary of analysis results.*
- `__init__` (line 1168) `def __init__(self, config)`
- `save_json_report` (line 1173) `def save_json_report(self, report, suffix)` - *Save individual report as JSON.*
- `save_aggregate_report` (line 1185) `def save_aggregate_report(self, results)` - *Save aggregate report from multiple analyses.*
- `_compute_statistics` (line 1216) `def _compute_statistics(self, summaries)` - *Compute aggregate statistics from summaries.*
- `_count_grades` (line 1247) `def _count_grades(self, summaries)` - *Count distribution of grades.*
- `generate_visualizations` (line 1255) `def generate_visualizations(self, results)` - *Generate visualization plots.*

#### `purity_index.py`
**Path:** `purity_index.py`

**Classes:**
- `PurityConfig` (line 19) `class PurityConfig`
- `IModel` (line 44) `class IModel(Protocol)`
- `IPurityIndexCalculator` (line 49) `class IPurityIndexCalculator(Protocol)`
- `IEffectiveTemperatureCalculator` (line 54) `class IEffectiveTemperatureCalculator(Protocol)`
- `IPhaseClassifier` (line 59) `class IPhaseClassifier(Protocol)`
- `IPolycrystalAnalyzer` (line 64) `class IPolycrystalAnalyzer(Protocol)`
- `IPurityComparator` (line 69) `class IPurityComparator(Protocol)`
- `BilinearModel` (line 73) `class BilinearModel`
- `PurityIndexCalculator` (line 101) `class PurityIndexCalculator`
- `EffectiveTemperatureCalculator` (line 156) `class EffectiveTemperatureCalculator`
- `PhaseClassifier` (line 199) `class PhaseClassifier`
- `PolycrystalAnalyzer` (line 236) `class PolycrystalAnalyzer`
- `PurityComparator` (line 274) `class PurityComparator`
- `CheckpointMigrator` (line 311) `class CheckpointMigrator`
- `PurityAnalyzer` (line 361) `class PurityAnalyzer`
- `PurityPipeline` (line 492) `class PurityPipeline`

**Functions:**
- `main` (line 611) `def main()`
- `get_coefficients` (line 45) `def get_coefficients(self)`
- `calculate` (line 50) `def calculate(self, model)`
- `calculate` (line 55) `def calculate(self, loss_history)`
- `classify` (line 60) `def classify(self, alpha, temperature)`
- `analyze_polycrystal` (line 65) `def analyze_polycrystal(self, model, pruning_level)`
- `compare` (line 70) `def compare(self, original, polycrystal)`
- `__init__` (line 74) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 85) `def _initialize(self)`
- `forward` (line 90) `def forward(self, a, b)`
- `get_coefficients` (line 93) `def get_coefficients(self)`
- `__init__` (line 102) `def __init__(self, config)`
- `calculate` (line 105) `def calculate(self, model)`
- `_compute_layer_purity` (line 134) `def _compute_layer_purity(self, weights)`
- `_delta_to_alpha` (line 140) `def _delta_to_alpha(self, delta)`
- `_assess_purity_quality` (line 145) `def _assess_purity_quality(self, alpha, variance)`
- `__init__` (line 157) `def __init__(self, config)`
- `calculate` (line 160) `def calculate(self, loss_history)`
- `__init__` (line 200) `def __init__(self, config)`
- `classify` (line 203) `def classify(self, alpha, temperature)`
- `classify_polycrystal_state` (line 219) `def classify_polycrystal_state(self, original_alpha, original_temp, poly_alpha, poly_temp)`
- `__init__` (line 237) `def __init__(self, config)`
- `analyze_polycrystal` (line 243) `def analyze_polycrystal(self, model, pruning_level, loss_history)`
- `_prune_model` (line 264) `def _prune_model(self, model, sparsity)`
- `__init__` (line 275) `def __init__(self, config)`
- `compare` (line 279) `def compare(self, original, polycrystal)`
- `migrate` (line 312) `def migrate(self, raw_data, device)`
- `_migrate_dict` (line 322) `def _migrate_dict(self, state_dict, device)`
- `_migrate_custom_format` (line 331) `def _migrate_custom_format(self, state_dict, device)`
- `_migrate_coefs_format` (line 350) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 357) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 362) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 375) `def _load_checkpoint(self)`
- `analyze` (line 399) `def analyze(self)`
- `_print_report` (line 445) `def _print_report(self, results)`
- `__init__` (line 493) `def __init__(self, config)`
- `process_checkpoint` (line 496) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 510) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 537) `def generate_summary(self, all_results, output_dir)`
- `_generate_text_report` (line 574) `def _generate_text_report(self, summary, output_dir)`

#### `repor_experiments.py`
**Path:** `repor_experiments.py`

**Classes:**
- `ModelConfig` (line 78) `class ModelConfig`
- `TrainConfig` (line 86) `class TrainConfig`
- `Exp1Config` (line 96) `class Exp1Config`
- `Exp2Config` (line 103) `class Exp2Config`
- `Exp3Config` (line 108) `class Exp3Config`
- `Exp4Config` (line 114) `class Exp4Config`
- `Exp5Config` (line 120) `class Exp5Config`
- `SuiteConfig` (line 126) `class SuiteConfig`
- `BilinearModel` (line 141) `class BilinearModel`

**Functions:**
- `discretize_q` (line 173) `def discretize_q(w)`
- `compute_delta` (line 176) `def compute_delta(U, V, W)`
- `phase2` (line 181) `def phase2(model)` - *Poda a target_rank slots, discretiza a {-1,0,1}, verifica.*
- `_verify_2x2` (line 204) `def _verify_2x2(U, V, W, n)`
- `zero_shot_verify` (line 214) `def zero_shot_verify(U, V, W, sizes)`
- `_recursive_strassen` (line 222) `def _recursive_strassen(U, V, W, n, trials)`
- `_strassen_rec` (line 234) `def _strassen_rec(A, B, U, V, W, n)`
- `compute_kappa` (line 262) `def compute_kappa(model, num_batches, bs)`
- `compute_alpha` (line 279) `def compute_alpha(delta)`
- `compute_teff` (line 284) `def compute_teff(model, num_batches, bs)`
- `classify_phase` (line 298) `def classify_phase(delta)`
- `load_checkpoint` (line 309) `def load_checkpoint(path, device)`
- `_extract_state` (line 329) `def _extract_state(d)`
- `train_model` (line 351) `def train_model(cfg, model, epochs, bs, wd, lr, callback)`
- `analyze_checkpoint` (line 378) `def analyze_checkpoint(path, device)`
- `experiment1` (line 428) `def experiment1(cfg)`
- `experiment2` (line 475) `def experiment2(cfg)`
- `experiment3` (line 507) `def experiment3(cfg)`
- `experiment4` (line 563) `def experiment4(cfg)`
- `experiment5` (line 607) `def experiment5(cfg)`
- `_test_accuracy` (line 633) `def _test_accuracy(model, n, device)`
- `_random_prune` (line 644) `def _random_prune(model, fraction)`
- `_boundary_prune` (line 655) `def _boundary_prune(model, fraction)`
- `analyze_checkpoints` (line 665) `def analyze_checkpoints(ckpt_dir, device)`
- `main` (line 705) `def main()`
- `_save` (line 783) `def _save(data, path)`
- `__init__` (line 142) `def __init__(self, cfg)`
- `forward` (line 149) `def forward(self, A, B)`
- `slot_importance` (line 156) `def slot_importance(self)`
- `get_weights` (line 163) `def get_weights(self)`
- `get_flat` (line 167) `def get_flat(self)`
- `cb` (line 434) `def cb(ep, m, loss, acc)`

#### `scrodingger.py`
**Path:** `scrodingger.py`

**Classes:**
- `SchrodingerConfig` (line 19) `class SchrodingerConfig`
- `IModel` (line 44) `class IModel(Protocol)`
- `IWaveFunctionExtractor` (line 49) `class IWaveFunctionExtractor(Protocol)`
- `IPotentialCalculator` (line 54) `class IPotentialCalculator(Protocol)`
- `IHamiltonianConstructor` (line 59) `class IHamiltonianConstructor(Protocol)`
- `IEigenvalueSolver` (line 64) `class IEigenvalueSolver(Protocol)`
- `ITimeEvolver` (line 69) `class ITimeEvolver(Protocol)`
- `IExpectationValueCalculator` (line 74) `class IExpectationValueCalculator(Protocol)`
- `IUncertaintyCalculator` (line 79) `class IUncertaintyCalculator(Protocol)`
- `ICheckpointLoader` (line 84) `class ICheckpointLoader(Protocol)`
- `ICheckpointMigrator` (line 89) `class ICheckpointMigrator(Protocol)`
- `BilinearModel` (line 93) `class BilinearModel`
- `WaveFunctionExtractor` (line 121) `class WaveFunctionExtractor`
- `PotentialCalculator` (line 128) `class PotentialCalculator`
- `HamiltonianConstructor` (line 150) `class HamiltonianConstructor`
- `EigenvalueSolver` (line 170) `class EigenvalueSolver`
- `TimeEvolver` (line 194) `class TimeEvolver`
- `ExpectationValueCalculator` (line 216) `class ExpectationValueCalculator`
- `UncertaintyCalculator` (line 226) `class UncertaintyCalculator`
- `CheckpointLoader` (line 263) `class CheckpointLoader`
- `CheckpointMigrator` (line 271) `class CheckpointMigrator`
- `SchrodingerAnalyzer` (line 320) `class SchrodingerAnalyzer`
- `WaveFunctionVisualizer` (line 542) `class WaveFunctionVisualizer`
- `SchrodingerPipeline` (line 609) `class SchrodingerPipeline`

**Functions:**
- `main` (line 779) `def main()`
- `get_coefficients` (line 45) `def get_coefficients(self)`
- `extract` (line 50) `def extract(self, model)`
- `calculate` (line 55) `def calculate(self, weights)`
- `construct` (line 60) `def construct(self, potential, mass)`
- `solve` (line 65) `def solve(self, hamiltonian, count)`
- `evolve` (line 70) `def evolve(self, initial_state, hamiltonian, time_steps, dt)`
- `calculate` (line 75) `def calculate(self, wave_function, operator)`
- `calculate` (line 80) `def calculate(self, wave_function, position_grid)`
- `load` (line 85) `def load(self, path, device)`
- `migrate` (line 90) `def migrate(self, raw_data)`
- `__init__` (line 94) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize` (line 105) `def _initialize(self)`
- `forward` (line 110) `def forward(self, a, b)`
- `get_coefficients` (line 113) `def get_coefficients(self)`
- `extract` (line 122) `def extract(self, model)`
- `__init__` (line 129) `def __init__(self, config)`
- `calculate` (line 132) `def calculate(self, weights)`
- `__init__` (line 151) `def __init__(self, config)`
- `construct` (line 154) `def construct(self, potential, mass)`
- `__init__` (line 171) `def __init__(self, config)`
- `solve` (line 174) `def solve(self, hamiltonian, count)`
- `__init__` (line 195) `def __init__(self, config)`
- `evolve` (line 198) `def evolve(self, initial_state, hamiltonian, time_steps, dt)`
- `calculate` (line 217) `def calculate(self, wave_function, operator)`
- `__init__` (line 227) `def __init__(self, config)`
- `calculate` (line 230) `def calculate(self, wave_function, position_grid)`
- `load` (line 264) `def load(self, path, device)`
- `migrate` (line 272) `def migrate(self, raw_data)`
- `_migrate_dict` (line 284) `def _migrate_dict(self, state_dict)`
- `_migrate_custom_format` (line 293) `def _migrate_custom_format(self, state_dict)`
- `_migrate_coefs_format` (line 309) `def _migrate_coefs_format(self, state_dict)`
- `_migrate_standard_format` (line 316) `def _migrate_standard_format(self, state_dict)`
- `__init__` (line 321) `def __init__(self, checkpoint_path, config)`
- `_load_checkpoint` (line 335) `def _load_checkpoint(self)`
- `analyze` (line 357) `def analyze(self)`
- `_calculate_tunneling_probability` (line 451) `def _calculate_tunneling_probability(self, potential, wave_function)`
- `_count_degeneracy` (line 465) `def _count_degeneracy(self, eigenvalues)`
- `_print_report` (line 478) `def _print_report(self, results)`
- `__init__` (line 543) `def __init__(self, config)`
- `visualize` (line 546) `def visualize(self, data, output_path)`
- `__init__` (line 610) `def __init__(self, config)`
- `process_checkpoint` (line 614) `def process_checkpoint(self, checkpoint_path, output_dir)`
- `process_directory` (line 628) `def process_directory(self, checkpoint_dir, n_latest, output_dir)`
- `generate_summary` (line 655) `def generate_summary(self, all_results, output_dir)`
- `_generate_text_report` (line 718) `def _generate_text_report(self, summary, output_dir)`

#### `benchmark_final.py`
**Path:** `src/benchmarks/benchmark_final.py`

**Functions:**
- `strassen_hybrid_multiply` (line 45) `def strassen_hybrid_multiply(A, B)` - *Multiply using our Strassen Hybrid implementation*
- `numpy_multiply` (line 60) `def numpy_multiply(A, B)` - *Standard NumPy BLAS multiplication*
- `benchmark` (line 64) `def benchmark(func, A, B, warmup, runs)` - *Run benchmark with warmup*
- `main` (line 80) `def main()`

#### `benchmark_scientific.py`
**Path:** `src/benchmarks/benchmark_scientific.py`

**Functions:**
- `strassen_multiply` (line 38) `def strassen_multiply(A, B)`
- `standard_avx512_multiply` (line 51) `def standard_avx512_multiply(A, B)`
- `numpy_multiply` (line 64) `def numpy_multiply(A, B)`
- `benchmark_function` (line 67) `def benchmark_function(func, A, B, runs, warmup)` - *Benchmark with statistical analysis*
- `main` (line 91) `def main()`

#### `benchmark_strassen.py`
**Path:** `src/benchmarks/benchmark_strassen.py`

**Classes:**
- `BenchmarkResult` (line 29) `class BenchmarkResult`
- `BenchmarkConfig` (line 45) `class BenchmarkConfig`

**Functions:**
- `load_config` (line 61) `def load_config(config_path)` - *Load configuration from TOML file.*
- `get_dtype` (line 93) `def get_dtype(dtype_str)` - *Convert dtype string to torch dtype.*
- `estimate_memory_mb` (line 104) `def estimate_memory_mb(n, dtype, batch_size)` - *Estimate memory usage for matrix multiplication.*
- `benchmark_resolution` (line 114) `def benchmark_resolution(n, cfg, dtype)` - *Benchmark Strassen vs standard matmul for given resolution.*
- `run_benchmark` (line 218) `def run_benchmark(cfg)` - *Run full benchmark suite.*
- `save_results` (line 310) `def save_results(results, filepath)` - *Save benchmark results to JSON.*
- `main` (line 322) `def main()`

#### `strassen_numpy.py`
**Path:** `src/benchmarks/strassen_numpy.py`

**Functions:**
- `_load_weights` (line 19) `def _load_weights()`
- `strassen_2x2_numpy` (line 29) `def strassen_2x2_numpy(A, B)` - *Strassen 2x2 using grokked coefficients.*
- `strassen_numpy` (line 44) `def strassen_numpy(A, B)` - *Recursive Strassen using NumPy.*
- `strassen_hybrid` (line 78) `def strassen_hybrid(A, B, threshold)` - *Hybrid Strassen: use Strassen for large matrices, NumPy for small.
This is faster because NumPy matmul is highly optimized for small matrices.*
- `multiplication_count` (line 112) `def multiplication_count(n)` - *Count multiplications used by Strassen.*

#### `auto_T_discovery.py`
**Path:** `src/discovery/auto_T_discovery.py`

**Classes:**
- `SymmetryStructure` (line 22) `class SymmetryStructure` - *Discovered symmetries in weight matrix*
- `AutoTDiscovery` (line 32) `class AutoTDiscovery` - *Automatic discovery of expansion operator T from converged weights.

The algorithm works in three phases:
1. Spectral Analysis: Extract dominant singular subspace
2. Symmetry Detection: Find block/permutation structure
3. T Construction: Build expansion operator preserving invariants*

**Functions:**
- `verify_strassen_T` (line 306) `def verify_strassen_T(model_path, target_sizes)` - *Verify T discovery on Strassen model*
- `verify_expanded_correctness` (line 353) `def verify_expanded_correctness(U, V, W, target_size, expanded)` - *Verify that expanded operator correctly computes matrix multiplication*
- `recursive_strassen_multiply` (line 389) `def recursive_strassen_multiply(A, B, U, V, W, base_size)` - *Recursively apply learned Strassen decomposition

This is the IMPLEMENTATION of T: it shows how the base 2x2 decomposition
extends to arbitrary sizes via recursive block application.*
- `__init__` (line 42) `def __init__(self, tolerance, verbose)`
- `analyze_structure` (line 46) `def analyze_structure(self, W)` - *Phase 1 & 2: Analyze weight matrix structure*
- `_detect_discrete_values` (line 89) `def _detect_discrete_values(self, W_flat)` - *Detect if weights cluster around discrete values*
- `_detect_block_structure` (line 105) `def _detect_block_structure(self, W)` - *Detect repeating block patterns*
- `_block_repetition_score` (line 127) `def _block_repetition_score(self, W, bm, bn)` - *Score how well blocks repeat (lower = more repetitive)*
- `_detect_symmetry_type` (line 144) `def _detect_symmetry_type(self, W)` - *Detect type of symmetry in weight matrix*
- `_is_permutation_symmetric` (line 165) `def _is_permutation_symmetric(self, W)` - *Check if rows are permutations of a base pattern*
- `_is_cyclic` (line 171) `def _is_cyclic(self, W)` - *Check for cyclic/Toeplitz structure*
- `_invariant_subspace_dim` (line 184) `def _invariant_subspace_dim(self, U, S, rank)` - *Estimate dimension of truly invariant subspace*
- `_discretization_error` (line 198) `def _discretization_error(self, W, values)` - *Compute error when discretizing to given values*
- `_print_analysis` (line 213) `def _print_analysis(self, W, S, structure)` - *Print analysis results*
- `construct_T` (line 229) `def construct_T(self, W_dict, target_size)` - *Phase 3: Construct expansion operator T

For matrix multiplication (U, V, W tensors), T expands via
recursive block structure discovered from the base case.

Args:
    W_dict: Dictionary with 'U', 'V', 'W' tensors
    target_size: Target matrix dimension (n' in the paper)

Returns:
    Expanded weight dictionary*
- `_validate_expansion` (line 290) `def _validate_expansion(self, expanded, structure)` - *Validate that expansion preserves key invariants*

#### `convergence_theory.py`
**Path:** `src/training/convergence_theory.py`

**Classes:**
- `ConvergenceMetrics` (line 21) `class ConvergenceMetrics` - *Metrics for tracking convergence to algorithmic invariance*
- `HutchinsonTraceEstimator` (line 31) `class HutchinsonTraceEstimator` - *Efficient Hessian trace estimation using Hutchinson's method.

tr(H) ≈ E[v^T H v] where v ~ Rademacher(±1)

Complexity: O(n_samples * forward_backward_pass) instead of O(n²)*
- `HardwareNoiseEstimator` (line 136) `class HardwareNoiseEstimator` - *Estimate ε_hw(B, T) - hardware-induced variance

Extended model addressing reviewer's criticism:
ε_hw(B, T) = α/B + β*cache_miss_rate(B) + γ*thread_contention(T)*
- `SimpleStrassenModel` (line 346) `class SimpleStrassenModel` - *Simple model for testing convergence verification*

**Functions:**
- `convergence_theorem` (line 193) `def convergence_theorem()` - *THEOREM (Convergence to Algorithmic Invariance)

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
- `verify_convergence_conditions` (line 265) `def verify_convergence_conditions(model, loss_fn, train_data, noise_threshold)` - *Verify that convergence conditions are satisfied for a trained model.*
- `__init__` (line 40) `def __init__(self, model, loss_fn, n_samples, device)`
- `estimate_trace` (line 47) `def estimate_trace(self, data)` - *Estimate tr(H) using Hutchinson's stochastic trace estimator.

Returns:
    (mean_trace, std_trace)*
- `_rademacher_vector` (line 73) `def _rademacher_vector(self)` - *Generate Rademacher random vector (±1 with equal probability)*
- `_hessian_vector_product` (line 90) `def _hessian_vector_product(self, x, y, v)` - *Compute H @ v using the "double backward" trick.

H @ v = ∂/∂θ (∇L · v)*
- `compute_kappa_eff` (line 118) `def compute_kappa_eff(self, data)` - *Compute κ_eff = -tr(H) / N

Interpretation:
- κ_eff < 0 and stable -> grokking likely
- κ_eff > 0 or oscillating -> grokking unlikely*
- `__init__` (line 144) `def __init__(self, model, loss_fn)`
- `estimate_noise` (line 148) `def estimate_noise(self, data_loader, n_batches, n_threads)` - *Estimate hardware noise by measuring gradient variance across batches*
- `__init__` (line 348) `def __init__(self, rank)`
- `forward` (line 354) `def forward(self, x)`

#### `grokkit_physics.py`
**Path:** `src/training/grokkit_physics.py`

**Functions:**
- `strassen_multiply` (line 49) `def strassen_multiply(A, B)` - *Wrapper for Strassen multiplication (uses float32).*
- `measure_physics` (line 64) `def measure_physics(N, num_samples)` - *Measure the 'physical quantities' for a given matrix size.

Returns:
    dict with: speedup, hbar_strassen, hbar_numpy, coherence, error*
- `detect_phase_transition` (line 126) `def detect_phase_transition(results)` - *Find the critical size N_c where the phase transition occurs.
Uses the maximum derivative of speedup curve.*
- `main` (line 148) `def main()`

#### `main.py`
**Path:** `src/training/main.py`

**Classes:**
- `Config` (line 44) `class Config` - *Configuración del experimento.*
- `StrassenDiscovery` (line 87) `class StrassenDiscovery` - *Modelo para descubrir Strassen mediante coeficientes aprendibles.

Arquitectura:
- U_coefs[i]: Combinación de bloques de A para producto M_i
- V_coefs[i]: Combinación de bloques de B para producto M_i  
- W_coefs[j,i]: Contribución de M_i al bloque C_j del resultado*
- `Matrix4x4Dataset` (line 199) `class Matrix4x4Dataset(Dataset)` - *Dataset de multiplicación de matrices 4x4.*
- `Trainer` (line 217) `class Trainer` - *Entrenador con enmascaramiento progresivo.*

**Functions:**
- `set_seed` (line 79) `def set_seed(seed)` - *Fijar semilla para reproducibilidad.*
- `main` (line 345) `def main()`
- `__init__` (line 96) `def __init__(self, num_slots)`
- `forward` (line 108) `def forward(self, A, B)` - *Forward pass con multiplicación matemática pura.
A, B: (batch, 4, 4) -> C: (batch, 4, 4)*
- `get_slot_norms` (line 155) `def get_slot_norms(self)` - *Norma promedio de cada slot.*
- `get_active_slots` (line 160) `def get_active_slots(self)` - *Número de slots activos.*
- `mask_slot` (line 164) `def mask_slot(self, slot_idx)` - *Desactiva un slot.*
- `get_weakest_slot` (line 169) `def get_weakest_slot(self)` - *Slot con menor norma entre los activos.*
- `print_coefficients` (line 175) `def print_coefficients(self)` - *Muestra coeficientes descubiertos.*
- `__init__` (line 202) `def __init__(self, num_samples, seed)`
- `__len__` (line 210) `def __len__(self)`
- `__getitem__` (line 213) `def __getitem__(self, idx)`
- `__init__` (line 220) `def __init__(self, config)`
- `accuracy` (line 242) `def accuracy(self, pred, target)`
- `train_epoch` (line 245) `def train_epoch(self, optimizer)`
- `evaluate` (line 265) `def evaluate(self)`
- `train` (line 280) `def train(self)`

#### `main_pure_math.py`
**Path:** `src/training/main_pure_math.py`

**Classes:**
- `StrassenModel` (line 22) `class StrassenModel` - *Descomposición de tensor para multiplicación de matrices 2x2.
C_ij = sum_r W[ij,r] * (U[r,:] @ a) * (V[r,:] @ b)*

**Functions:**
- `gen_data` (line 58) `def gen_data(n, scale)`
- `train` (line 64) `def train(model, epochs, lr, l1, batch, verbose)`
- `verify` (line 88) `def verify(model, n)`
- `hard_prune` (line 106) `def hard_prune(model, keep)` - *Poda los slots más débiles, mantiene top-k.*
- `refine_pruned` (line 122) `def refine_pruned(model, active, epochs, lr)` - *Refina manteniendo slots podados en cero.*
- `show_coeffs` (line 159) `def show_coeffs(model, active)`
- `main` (line 186) `def main()`
- `__init__` (line 28) `def __init__(self, rank)`
- `forward` (line 35) `def forward(self, A, B)`
- `slot_norms` (line 47) `def slot_norms(self)` - *Norma combinada de cada slot.*
- `active_count` (line 54) `def active_count(self, thresh)`

#### `strassen_core.py`
**Path:** `src/training/strassen_core.py`

**Functions:**
- `_load_weights` (line 14) `def _load_weights()`
- `strassen_2x2` (line 21) `def strassen_2x2(A, B)`
- `strassen` (line 44) `def strassen(X, Y)`
- `get_coefficients` (line 77) `def get_coefficients()`
- `multiplication_count` (line 82) `def multiplication_count(n)`

#### `strassen_grokkit.py`
**Path:** `src/training/strassen_grokkit.py`

**Classes:**
- `StrassenOperator` (line 27) `class StrassenOperator` - *Operador espectral para multiplicación de matrices 2x2.

Representa el tensor de rango R:
C_ij = Σ_r W[ij,r] * (U[r,:] · a) * (V[r,:] · b)

Donde:
- U, V: Coeficientes de combinación lineal (LC)
- W: Coeficientes de reconstrucción
- Esparsidad (SP): Cuántos slots están activos*

**Functions:**
- `generate_batch` (line 113) `def generate_batch(n, scale)` - *Genera batch de matrices aleatorias.*
- `train_grokkit` (line 120) `def train_grokkit(epochs, batch_size, lr, wd)` - *Entrena usando el framework Grokkit.

WD (Weight Decay) actúa como presión termodinámica que:
1. Empuja hacia soluciones de menor norma
2. Promueve esparsidad natural (slots débiles -> 0)
3. Cristaliza el operador en el mínimo de energía (rango 7)*
- `verify_grokking` (line 203) `def verify_grokking(model, n_test)` - *Verifica que el operador ha grokkeado correctamente.*
- `progressive_sparsification` (line 254) `def progressive_sparsification(model, target_slots)` - *Fase 2: Esparsificación progresiva.
Reduce gradualmente a 7 slots manteniendo accuracy.*
- `main` (line 351) `def main()` - *Pipeline principal Grokkit para Strassen.*
- `__init__` (line 40) `def __init__(self, rank)`
- `forward` (line 49) `def forward(self, A, B)` - *Computa A @ B usando la descomposición tensorial.*
- `compute_LC` (line 67) `def compute_LC(self)` - *Linear Combination metric.
Mide qué tan bien los coeficientes forman combinaciones válidas.
LC -> 1 significa combinaciones perfectas.*
- `compute_SP` (line 83) `def compute_SP(self)` - *Sparsity metric.
SP -> 0 significa máxima esparsidad (menos slots activos).
SP = (slots_activos - 7) / rank para normalizar*
- `slot_importance` (line 101) `def slot_importance(self)` - *Importancia de cada slot basada en normas.*
- `count_active` (line 108) `def count_active(self, threshold)` - *Cuenta slots activos.*

#### `train_strassen.py`
**Path:** `src/training/train_strassen.py`

**Classes:**
- `StrassenOperator` (line 26) `class StrassenOperator` - *Spectral operator for 2x2 matrix multiplication.

Tensor decomposition: C_ij = sum_r W[ij,r] * (U[r,:] . a) * (V[r,:] . b)*

**Functions:**
- `generate_batch` (line 60) `def generate_batch(n, scale)`
- `train_phase1` (line 66) `def train_phase1(epochs, batch_size, lr, wd)` - *Phase 1: Grokking with Weight Decay as thermodynamic pressure.*
- `sparsify` (line 104) `def sparsify(model, target_slots)` - *Phase 2: Progressive sparsification to target rank.*
- `discretize` (line 183) `def discretize(model, slots_to_prune)` - *Phase 3: Discretize coefficients to {-1, 0, 1}.*
- `get_canonical_strassen` (line 211) `def get_canonical_strassen()` - *Returns the canonical Strassen coefficients.
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
- `verify` (line 260) `def verify(U, V, W, n_test)` - *Verify the discretized operator.*
- `main` (line 299) `def main()` - *Main training pipeline.*
- `__init__` (line 33) `def __init__(self, rank)`
- `forward` (line 40) `def forward(self, A, B)`
- `slot_importance` (line 50) `def slot_importance(self)`
- `count_active` (line 56) `def count_active(self, threshold)`

#### `superposition.py`
**Path:** `superposition.py`

**Classes:**
- `Config` (line 23) `class Config` - *Configuration for Superposition Analysis on Strassen Checkpoints.*
- `ICheckpointLoader` (line 59) `class ICheckpointLoader(Protocol)`
- `IMetricsCalculator` (line 62) `class IMetricsCalculator(ABC)`
- `IAnalyzer` (line 66) `class IAnalyzer(ABC)`
- `CheckpointLoadingError` (line 71) `class CheckpointLoadingError(Exception)`
- `CheckpointLoader` (line 75) `class CheckpointLoader` - *Loads raw checkpoint files.*
- `CheckpointMigrator` (line 85) `class CheckpointMigrator` - *Migrates various checkpoint formats to standard state_dict.*
- `StrassenDataGenerator` (line 220) `class StrassenDataGenerator` - *Generates matrix multiplication data for activation extraction.*
- `BilinearStrassenModel` (line 258) `class BilinearStrassenModel` - *Your existing Strassen model architecture.*
- `SparseAutoencoder` (line 292) `class SparseAutoencoder` - *SAE with tied weights (W_dec = W_enc^T).
Corrected dimensions: W_enc: [D, N], encode uses W_enc^T, decode uses W_enc.*
- `SuperpositionMetrics` (line 334) `class SuperpositionMetrics(IMetricsCalculator)` - *Calculates superposition metrics from Section 4 of the paper.*
- `SAETrainer` (line 407) `class SAETrainer` - *Trains SAE on bottleneck activations extracted from Strassen model.*
- `StrassenCheckpointAnalyzer` (line 475) `class StrassenCheckpointAnalyzer(IAnalyzer)` - *Analyzes existing Strassen checkpoints for superposition metrics.*

**Functions:**
- `main` (line 765) `def main()`
- `__post_init__` (line 54) `def __post_init__(self)`
- `load_checkpoint` (line 60) `def load_checkpoint(self, path, device)`
- `compute` (line 64) `def compute(self)`
- `analyze_checkpoint` (line 68) `def analyze_checkpoint(self, checkpoint_path)`
- `load_checkpoint` (line 78) `def load_checkpoint(self, path, device)`
- `detect_hidden_dim` (line 89) `def detect_hidden_dim(raw_data)` - *Detect hidden dimension from checkpoint data structure by inspecting
tensor shapes in various known formats.
Returns None if cannot be determined unambiguously.*
- `migrate_checkpoint` (line 122) `def migrate_checkpoint(raw_data)`
- `_migrate_dict` (line 135) `def _migrate_dict(state_dict)`
- `_migrate_custom_format` (line 147) `def _migrate_custom_format(state_dict)`
- `_migrate_coefs_format` (line 162) `def _migrate_coefs_format(state_dict)`
- `_migrate_encoder_format` (line 170) `def _migrate_encoder_format(state_dict)` - *Handle encoder-based format from specific experimental architectures.
Extracts U, V, W from sequential encoder layers assuming specific indexing.*
- `_migrate_standard_format` (line 215) `def _migrate_standard_format(state_dict)`
- `__init__` (line 223) `def __init__(self, config)`
- `generate_batch` (line 226) `def generate_batch(self, batch_size)` - *Generate batch of matrix pairs and their product.*
- `generate_dataset` (line 240) `def generate_dataset(self, num_samples)` - *Generate full dataset.*
- `__init__` (line 261) `def __init__(self, config)`
- `_initialize_symmetric` (line 271) `def _initialize_symmetric(self)`
- `forward` (line 276) `def forward(self, a, b)` - *Forward pass returning output and bottleneck activations.*
- `get_coefficients` (line 284) `def get_coefficients(self)`
- `__init__` (line 298) `def __init__(self, config)`
- `encode` (line 310) `def encode(self, x)` - *Encode bottleneck activations to sparse features.
x: [batch, N]
W_enc: [D, N]
Returns: [batch, D]*
- `decode` (line 319) `def decode(self, z)` - *Decode sparse features back to bottleneck.
z: [batch, D]
W_enc: [D, N]
Returns: [batch, N]*
- `forward` (line 328) `def forward(self, x)`
- `__init__` (line 339) `def __init__(self, config)`
- `compute_feature_probabilities` (line 342) `def compute_feature_probabilities(self, sae_activations)` - *Calculate feature probabilities from SAE activations.
p_i = Σ_s |z_i,s| / Σ_j Σ_s |z_j,s|*
- `compute_entropy` (line 352) `def compute_entropy(self, probabilities)` - *Shannon entropy H(p) = -Σ p_i log p_i.*
- `compute_superposition` (line 359) `def compute_superposition(self, sae_activations)` - *Main metric: ψ = F/N where F = e^{H(p)}.*
- `compute_frobenius_metric` (line 377) `def compute_frobenius_metric(self, weight_matrix)` - *Baseline from Eq 2: ψ_Frob = ||W||_F^2 / N.
Applied to the bottleneck transformation (W matrix of Strassen).*
- `compute_interference_matrix` (line 385) `def compute_interference_matrix(self, weight_matrix)` - *Compute W^T @ W to analyze interference patterns.*
- `compute` (line 389) `def compute(self, sae_activations, weight_matrix)` - *Unified interface.*
- `__init__` (line 410) `def __init__(self, sae, config)`
- `train` (line 421) `def train(self, bottleneck_activations)` - *Train SAE on extracted activations.*
- `__init__` (line 480) `def __init__(self, config)`
- `load_model` (line 496) `def load_model(self, checkpoint_path)` - *Load and migrate checkpoint to model.
Returns tuple of (model, effective_config) where effective_config has 
the correct HIDDEN_DIM for this specific checkpoint to avoid dimension mismatches.*
- `extract_bottleneck_activations` (line 554) `def extract_bottleneck_activations(self, model)` - *Extract bottleneck activations (U(a) * V(b)) from model.*
- `analyze_checkpoint` (line 571) `def analyze_checkpoint(self, checkpoint_path)` - *Full analysis pipeline for a single checkpoint:
1. Load model with correct dimensions (detecting hidden_dim from checkpoint)
2. Extract bottleneck activations
3. Train SAE with matching dimensions (using effective_config)
4. Calculate superposition metrics with correct normalization (N=hidden_dim)
5. Calculate baseline Frobenius metric on W weights*
- `_save_intermediate_result` (line 633) `def _save_intermediate_result(self, result, name)` - *Save result for individual checkpoint.*
- `analyze_directory` (line 641) `def analyze_directory(self, checkpoint_dir)` - *Analyze all checkpoints in directory.*
- `_save_progress_checkpoint` (line 680) `def _save_progress_checkpoint(self, results)` - *Save intermediate progress.*
- `_save_final_results` (line 687) `def _save_final_results(self, results)` - *Save complete results.*
- `_generate_comparison_plots` (line 694) `def _generate_comparison_plots(self, results)` - *Generate comparison plots across checkpoints.*
- `get_tensor` (line 148) `def get_tensor(key)`

#### `train_batch_sweep.py`
**Path:** `train_batch_sweep.py`

**Classes:**
- `LocalConfig` (line 17) `class LocalConfig`

**Functions:**
- `train_for_batch_size` (line 11) `def train_for_batch_size(B, seed, output_dir)`

#### `unified_hidden_connections_suite.py`
**Path:** `unified_hidden_connections_suite.py`

**Classes:**
- `StrassStrassenConfig` (line 65) `class StrassStrassenConfig` - *Immutable canonical configuration for the Strassen bilinear model.*
- `TrainingConfig` (line 84) `class TrainingConfig` - *Immutable training hyperparameters.*
- `Experiment1Config` (line 99) `class Experiment1Config` - *Configuration for Experiment 1: Ricci-MBL Duality.*
- `Experiment2Config` (line 114) `class Experiment2Config` - *Configuration for Experiment 2: Altland-Zirnbauer Symmetry Dial.*
- `Experiment3Config` (line 131) `class Experiment3Config` - *Configuration for Experiment 3: Conformal Isomorphism.*
- `Experiment4Config` (line 141) `class Experiment4Config` - *Configuration for Experiment 4: Compression Frontier.*
- `Experiment5Config` (line 158) `class Experiment5Config` - *Configuration for Experiment 5: Holographic Pruning.*
- `SuiteConfig` (line 169) `class SuiteConfig` - *Top-level suite orchestration configuration.*
- `StrassStrassenModel` (line 183) `class StrassStrassenModel` - *Exact bilinear tensor-decomposition model for 2x2 matrix multiplication.
Implements C = W((U * A) ⊙ (V * B)) where ⊙ denotes element-wise product.
U, V ∈ R^{rank x input_dim}, W ∈ R^{output_dim x rank}.*
- `IDataGenerator` (line 229) `class IDataGenerator(Protocol)` - *Protocol for deterministic data generation.*
- `StrassenDataGenerator` (line 235) `class StrassenDataGenerator` - *Generates random 2x2 matrix pairs and their exact products.*
- `ICheckpointManager` (line 262) `class ICheckpointManager(Protocol)` - *Protocol for checkpoint persistence.*
- `CheckpointManager` (line 269) `class CheckpointManager` - *Handles safe serialization and deserialization of model checkpoints.*
- `ITrainer` (line 293) `class ITrainer(Protocol)` - *Protocol for training routines.*
- `Trainer` (line 301) `class Trainer` - *Standard trainer with AdamW, cosine annealing, and gradient clipping.*
- `IMetricCalculator` (line 357) `class IMetricCalculator(ABC)` - *Abstract base for all metric calculators.*
- `LevelSpacingRatioCalculator` (line 365) `class LevelSpacingRatioCalculator` - *Computes the adjacent gap ratio r for Hessian eigenvalue spectra.*
- `RicciScalarCalculator` (line 415) `class RicciScalarCalculator` - *Computes Ricci scalar and geometric curvature metrics from Hessian.*
- `SyntheticPlanckCalculator` (line 492) `class SyntheticPlanckCalculator` - *Estimates the synthetic Planck constant from model weight statistics.*
- `SuperpositionMetricCalculator` (line 552) `class SuperpositionMetricCalculator` - *Measures superposition via sparse autoencoder bottleneck analysis.*
- `IExperiment` (line 645) `class IExperiment(ABC)` - *Abstract base for all experiments in the suite.*
- `Experiment1RicciMBLDuality` (line 657) `class Experiment1RicciMBLDuality(IExperiment)` - *Tests the claim that Ricci curvature smoothing is the geometric mechanism
driving the Wigner-Dyson to Poisson/MBL spectral transition.*
- `Experiment2AltlandZirnbauer` (line 735) `class Experiment2AltlandZirnbauer(IExperiment)` - *Tests the claim that the imaginary-weight control parameter gamma drives
the system between GOE (orthogonal) and GUE (unitary) random matrix classes.*
- `Experiment3ConformalIsomorphism` (line 820) `class Experiment3ConformalIsomorphism(IExperiment)` - *Tests the claim that the network learns the underlying conformal operator
by applying Moebius transformations to inputs and measuring equivariance.*
- `Experiment4CompressionFrontier` (line 886) `class Experiment4CompressionFrontier(IExperiment)` - *Tests the thermodynamic uncertainty relation between synthetic Planck
constant and superposition metric across a sweep of batch sizes and
weight decay values.*
- `IPruningStrategy` (line 965) `class IPruningStrategy(ABC)` - *Abstract base for structured pruning strategies.*
- `VolumePruningStrategy` (line 973) `class VolumePruningStrategy(IPruningStrategy)` - *Prunes weights uniformly at random across all layers.*
- `AreaPruningStrategy` (line 988) `class AreaPruningStrategy(IPruningStrategy)` - *Prunes weights only from tensor boundaries (slot edges).*
- `Experiment5HolographicPruning` (line 1004) `class Experiment5HolographicPruning(IExperiment)` - *Tests whether the crystal phase encodes information on boundaries
(area law) versus the glass phase encoding it volumetrically.*
- `UnifiedSuite` (line 1078) `class UnifiedSuite` - *Orchestrates the execution of all five hidden-connection experiments.*

**Functions:**
- `main` (line 1182) `def main()`
- `__post_init__` (line 76) `def __post_init__(self)`
- `__init__` (line 190) `def __init__(self, config)`
- `forward` (line 203) `def forward(self, A, B)`
- `get_coefficients` (line 213) `def get_coefficients(self)`
- `get_flat_parameters` (line 216) `def get_flat_parameters(self)`
- `slot_importance` (line 219) `def slot_importance(self)`
- `count_active_slots` (line 225) `def count_active_slots(self, threshold)`
- `generate_batch` (line 232) `def generate_batch(self, batch_size)`
- `__init__` (line 238) `def __init__(self, config)`
- `generate_batch` (line 241) `def generate_batch(self, batch_size)`
- `save` (line 265) `def save(self, model, epoch, metrics, path)`
- `load` (line 266) `def load(self, path, model)`
- `save` (line 272) `def save(self, model, epoch, metrics, path)`
- `load` (line 284) `def load(self, path, model)`
- `train` (line 296) `def train(self, model, epochs, callback)`
- `__init__` (line 304) `def __init__(self, model_config, training_config, data_generator)`
- `train` (line 314) `def train(self, model, epochs, callback)`
- `calculate` (line 361) `def calculate(self, model)`
- `__init__` (line 368) `def __init__(self, config, tolerance)`
- `_build_hessian_approximation` (line 372) `def _build_hessian_approximation(self, model)`
- `calculate` (line 384) `def calculate(self, model)`
- `__init__` (line 418) `def __init__(self, config, regularization)`
- `_compute_hessian` (line 422) `def _compute_hessian(self, model)`
- `_generate_single_sample` (line 433) `def _generate_single_sample(self)`
- `_loss_from_flat` (line 437) `def _loss_from_flat(self, flat_params, model, original_params, A, B, C_true)`
- `_diagonal_hessian_approximation` (line 456) `def _diagonal_hessian_approximation(self, model, A, B, C_true)`
- `calculate` (line 470) `def calculate(self, model)`
- `__init__` (line 495) `def __init__(self, config, noise_floor)`
- `calculate` (line 499) `def calculate(self, model)`
- `__init__` (line 555) `def __init__(self, model_config, expansion_factor, l1_coefficient, sae_lr, sae_epochs, sae_batch_size, num_samples, epsilon)`
- `_extract_activations` (line 576) `def _extract_activations(self, model)`
- `_train_sae` (line 589) `def _train_sae(self, activations)`
- `_sae_forward` (line 616) `def _sae_forward(self, x, W_enc, b_enc, b_dec)`
- `calculate` (line 623) `def calculate(self, model)`
- `run` (line 649) `def run(self, model)`
- `get_name` (line 653) `def get_name(self)`
- `__init__` (line 663) `def __init__(self, config, model_config, training_config, data_generator, checkpoint_manager)`
- `get_name` (line 679) `def get_name(self)`
- `run` (line 682) `def run(self, model)`
- `_analyze_temporal_correlation` (line 717) `def _analyze_temporal_correlation(self, results)`
- `__init__` (line 741) `def __init__(self, config, model_config, data_generator)`
- `get_name` (line 752) `def get_name(self)`
- `run` (line 755) `def run(self, model)`
- `_create_gamma_model` (line 778) `def _create_gamma_model(self, base_model, gamma)`
- `_train_gamma_model` (line 790) `def _train_gamma_model(self, model)`
- `_detect_critical_transition` (line 800) `def _detect_critical_transition(self, results)`
- `__init__` (line 826) `def __init__(self, config, model_config, data_generator)`
- `get_name` (line 836) `def get_name(self)`
- `run` (line 839) `def run(self, model)`
- `_apply_moebius` (line 869) `def _apply_moebius(self, A, B)`
- `_apply_moebius_to_output` (line 879) `def _apply_moebius_to_output(self, C)`
- `__init__` (line 893) `def __init__(self, config, model_config, data_generator)`
- `get_name` (line 904) `def get_name(self)`
- `run` (line 907) `def run(self, model)`
- `_test_uncertainty_bound` (line 951) `def _test_uncertainty_bound(self, results)`
- `prune` (line 969) `def prune(self, model, fraction)`
- `prune` (line 976) `def prune(self, model, fraction)`
- `prune` (line 991) `def prune(self, model, fraction)`
- `__init__` (line 1010) `def __init__(self, config, model_config, data_generator)`
- `get_name` (line 1022) `def get_name(self)`
- `run` (line 1025) `def run(self, model)`
- `_run_pruning_trials` (line 1046) `def _run_pruning_trials(self, base_model, pruner, A, B, C_true)`
- `__init__` (line 1081) `def __init__(self, config)`
- `_build_experiments` (line 1088) `def _build_experiments(self)`
- `run_all` (line 1127) `def run_all(self)`
- `_aggregate_verdicts` (line 1150) `def _aggregate_verdicts(self, all_results)`
- `_serialize_config` (line 1163) `def _serialize_config(self)`
- `checkpoint_callback` (line 691) `def checkpoint_callback(epoch, m, loss, acc)`

#### `xray_tensor_diffractometer.py`
**Path:** `xray_tensor_diffractometer.py`

**Classes:**
- `Config` (line 27) `class Config`
- `ICheckpointLoader` (line 146) `class ICheckpointLoader(Protocol)`
- `IMetricsCalculator` (line 149) `class IMetricsCalculator(Protocol)`
- `IDataGenerator` (line 152) `class IDataGenerator(Protocol)`
- `CheckpointLoadingError` (line 155) `class CheckpointLoadingError(Exception)`
- `MetricsComputationError` (line 158) `class MetricsComputationError(Exception)`
- `TrainingError` (line 161) `class TrainingError(Exception)`
- `StrassenDataGenerator` (line 166) `class StrassenDataGenerator`
- `BilinearStrassenModel` (line 187) `class BilinearStrassenModel`
- `EpitaxialGrowthEngine` (line 216) `class EpitaxialGrowthEngine` - *Motor de crecimiento epitaxial para cristales algorítmicos.

FÍSICA: Imita el crecimiento de cristales en sustratos donde la estructura
atómica del sustrato guía la formación del nuevo cristal.*
- `EpitaxyExperiment` (line 469) `class EpitaxyExperiment` - *Experimento completo de epitaxia: sembrar, crecer, analizar.*
- `ThermodynamicPotential` (line 671) `class ThermodynamicPotential` - *Potencial de Helmholtz: F = U - T*S + μ*N + α_term*C*
- `SpectroscopyMetrics` (line 693) `class SpectroscopyMetrics`
- `ThermodynamicMetrics` (line 913) `class ThermodynamicMetrics`
- `CrystallographyMetrics` (line 1158) `class CrystallographyMetrics`
- `GreenCowExperiment` (line 1264) `class GreenCowExperiment` - *🐄 Green's Cow: Uses integration-by-parts analogy to split gradient into bulk and boundary terms.
Inspired by Green's identities: ∫_Ω ∇u·v = ∫_∂Ω u(v·n) - ∫_Ω u∇·v
Applied to weight tensors as discrete manifolds.*
- `CheckpointLoader` (line 1371) `class CheckpointLoader`
- `CheckpointMigrator` (line 1405) `class CheckpointMigrator`
- `BoltzmannAnalysisProgram` (line 1490) `class BoltzmannAnalysisProgram`
- `StrassenCrystallographer` (line 2610) `class StrassenCrystallographer`

**Functions:**
- `set_seed` (line 64) `def set_seed(seed)`
- `setup_logger` (line 72) `def setup_logger(name, level)`
- `run_epitaxy_from_best_crystal` (line 86) `def run_epitaxy_from_best_crystal(checkpoint_dir, target_sizes)` - *Pipeline automático: encuentra el mejor cristal y lo usa como semilla.*
- `main` (line 2683) `def main()`
- `load_checkpoint` (line 147) `def load_checkpoint(self, path, device)`
- `compute` (line 150) `def compute(self, model)`
- `generate_batch` (line 153) `def generate_batch(self, batch_size)`
- `generate_batch` (line 168) `def generate_batch(batch_size)`
- `verify_structure` (line 179) `def verify_structure(coeffs)`
- `__init__` (line 188) `def __init__(self, hidden_dim, matrix_size)`
- `_initialize_symmetric` (line 199) `def _initialize_symmetric(self)`
- `forward` (line 204) `def forward(self, a, b)`
- `get_coefficients` (line 207) `def get_coefficients(self)`
- `__init__` (line 224) `def __init__(self, seed_checkpoint_path, target_matrix_size, device)`
- `_load_seed_crystal` (line 242) `def _load_seed_crystal(self)` - *Carga el cristal semilla verificando su pureza*
- `grow_epitaxial_crystal` (line 265) `def grow_epitaxial_crystal(self)` - *Crece un cristal epitaxial desde la semilla.

MÉTODO: Kronecker product preserva la estructura periódica:
Si A es cristal Strassen de 2x2, entonces A ⊗ I_n es cristal de (2n)x(2n)*
- `_adjust_dimensions` (line 326) `def _adjust_dimensions(self, tensor, target_shape)` - *Ajusta dimensiones del tensor epitaxial para coincidir con el modelo objetivo.
Rellena con ruido térmico pequeño o trunca según sea necesario.*
- `anneal_crystal` (line 360) `def anneal_crystal(self, model, max_epochs, early_stop_threshold)` - *Recocido térmico del cristal epitaxial.

FÍSICA: En lugar de "entrenar desde cero", aplicamos temperatura decreciente
para que el cristal se auto-organice alrededor de la semilla.*
- `__init__` (line 474) `def __init__(self, results_dir)`
- `run_epitaxial_growth_experiment` (line 479) `def run_epitaxial_growth_experiment(self, seed_checkpoint, target_sizes)` - *Experimento completo: cultiva cristales de múltiples tamaños desde una semilla.*
- `_plot_epitaxial_evolution` (line 556) `def _plot_epitaxial_evolution(self, annealing_results, target_size, seed_name)` - *Visualiza la evolución del cristal durante el recocido*
- `_generate_comparative_report` (line 604) `def _generate_comparative_report(self, results)` - *Genera reporte comparativo de todos los experimentos epitaxiales*
- `helmholtz_free_energy` (line 680) `def helmholtz_free_energy(self)` - *F = U - T*S (a μ y N constantes)*
- `gibbs_free_energy` (line 684) `def gibbs_free_energy(self)` - *G = F + μ*N + P*V (presión algorítmica)*
- `is_stable` (line 689) `def is_stable(self)` - *Criterio de estabilidad: dG < 0*
- `compute_weight_diffraction` (line 696) `def compute_weight_diffraction(coeffs)`
- `_compute_spectral_entropy` (line 719) `def _compute_spectral_entropy(power_spectrum)`
- `extract_lattice_parameters` (line 726) `def extract_lattice_parameters(weight_tensor, rank)` - *Extrae parámetros de red preservando la geometría física del tensor.

FIX: En lugar de reshape arbitrario, aplicamos SVD sobre la matriz 
de covarianza que preserva la estructura de correlaciones.*
- `compute_gibbs_free_energy` (line 785) `def compute_gibbs_free_energy(loss, temp, entropy)`
- `extract_canonical_decomposition` (line 791) `def extract_canonical_decomposition(coeffs, rank)` - *Descomposición Canónica del tensor tripartito (U, V, W).

FIX: Preserva la estructura bilineal en lugar de tratar como matriz plana.
Aplicamos HOSVD (Higher-Order SVD) para tensores de orden 3.*
- `_discretize_to_integers` (line 847) `def _discretize_to_integers(factors)` - *Proyecta factores continuos a la red cristalina discreta {-1, 0, 1}.*
- `_check_strassen_equivalence` (line 868) `def _check_strassen_equivalence(discretized_factors)` - *Verifica si los factores discretizados corresponden a la estructura de Strassen.*
- `create_superlattice_seed` (line 892) `def create_superlattice_seed(base_tensor, scale_factor)`
- `compute_effective_temperature` (line 916) `def compute_effective_temperature(gradient_buffer, learning_rate)`
- `compute_critical_exponents` (line 930) `def compute_critical_exponents(temp_history, cv_history, alpha_history)` - *Calcula exponentes críticos cerca de transiciones de fase.

Leyes de escala:
- C_v ~ |T - T_c|^{-α_exp}  (calor específico)
- ξ ~ |T - T_c|^{-ν}        (longitud de correlación)
- τ ~ |T - T_c|^{-z}        (tiempo de grokking)*
- `compute_equation_of_state` (line 1009) `def compute_equation_of_state(temp_eff, alpha, kappa)` - *Ecuación de estado: T_c(α) = T_0 * exp(-c*α)

FIX: Relación constitutiva que describe la curva de coexistencia cristal-vidrio.*
- `compute_specific_heat` (line 1048) `def compute_specific_heat(loss_history, temp_history, cv_threshold)`
- `estimate_hbar_algorithmic` (line 1061) `def estimate_hbar_algorithmic(model_complexity, weight_dim, mutual_information)`
- `compute_mutual_information` (line 1069) `def compute_mutual_information(weights, gradients)`
- `check_extensivity` (line 1083) `def check_extensivity(entropy_list, scale_factors)`
- `compute_fisher_information_matrix` (line 1107) `def compute_fisher_information_matrix(model, samples)`
- `compute_ricci_curvature` (line 1126) `def compute_ricci_curvature(fisher_matrix)`
- `calculate_carnot_efficiency` (line 1137) `def calculate_carnot_efficiency(delta_alpha, total_flops, initial_alpha)`
- `compute_kappa` (line 1161) `def compute_kappa(model, dataloader, num_batches)`
- `compute_discretization_margin` (line 1187) `def compute_discretization_margin(coeffs)`
- `compute_local_complexity` (line 1191) `def compute_local_complexity(model)`
- `compute_alpha_purity` (line 1200) `def compute_alpha_purity(coeffs)`
- `compute_kappa_quantum` (line 1207) `def compute_kappa_quantum(coeffs, hbar)`
- `compute_poynting_vector` (line 1224) `def compute_poynting_vector(coeffs)`
- `compute_all_metrics` (line 1246) `def compute_all_metrics(model, dataloader)`
- `__init__` (line 1270) `def __init__(self, model, device)`
- `compute_boundary_gradient` (line 1275) `def compute_boundary_gradient(self, weight)` - *Approximate surface term: gradient concentrated on tensor boundaries.
For a matrix W ∈ ℝ^{m×n}, boundary = first/last row + first/last column.*
- `compute_bulk_gradient` (line 1290) `def compute_bulk_gradient(self, weight)` - *Interior (volume) term: everything except boundary.*
- `run_green_backprop_step` (line 1296) `def run_green_backprop_step(self, A, B, C_true, lambda_boundary)` - *Custom backward pass using Green-inspired decomposition.
Loss = MSE + λ_boundary * ||boundary_grad||²*
- `_get_boundary_mask` (line 1329) `def _get_boundary_mask(self, weight)` - *Returns a binary mask marking boundary elements of a tensor.*
- `train_with_green_cow` (line 1341) `def train_with_green_cow(self, epochs, lr, lambda_boundary)`
- `load_checkpoint` (line 1372) `def load_checkpoint(self, path, device)` - *Load checkpoint with robust deserialization handling.
Injects Config as UnifiedConfig alias to handle cross-script compatibility.*
- `migrate_checkpoint` (line 1407) `def migrate_checkpoint(raw_data)` - *Migrate checkpoint to standard format, extracting config if present.
Returns migrated state_dict compatible with BilinearStrassenModel.*
- `_migrate_dict` (line 1431) `def _migrate_dict(state_dict)`
- `_migrate_custom_format` (line 1443) `def _migrate_custom_format(state_dict)`
- `_migrate_coefs_format` (line 1467) `def _migrate_coefs_format(state_dict)`
- `_migrate_encoder_format` (line 1475) `def _migrate_encoder_format(state_dict)`
- `_migrate_standard_format` (line 1487) `def _migrate_standard_format(state_dict)`
- `__init__` (line 1491) `def __init__(self, checkpoint_dir, results_dir)`
- `_load_all_checkpoints` (line 1503) `def _load_all_checkpoints(self)`
- `run_full_boltzmann_program` (line 1548) `def run_full_boltzmann_program(self)`
- `phase1_molecular_hypothesis` (line 1575) `def phase1_molecular_hypothesis(self)`
- `phase2_entropy_production` (line 1660) `def phase2_entropy_production(self)`
- `phase3_extensivity_law` (line 1742) `def phase3_extensivity_law(self)`
- `phase4_quantum_basis_transform` (line 1796) `def phase4_quantum_basis_transform(self)`
- `analyze_poynting_flow` (line 1849) `def analyze_poynting_flow(self)`
- `phase5_thermodynamic_analysis` (line 1882) `def phase5_thermodynamic_analysis(self)` - *PHASE 5: THERMODYNAMIC ANALYSIS con exponentes críticos y ecuación de estado.

FIX: Ahora calcula exponentes críticos y ecuación de estado para cada checkpoint.*
- `phase6_spectroscopic_analysis` (line 2011) `def phase6_spectroscopic_analysis(self)`
- `_plot_diffraction_pattern` (line 2097) `def _plot_diffraction_pattern(self, diffraction_data, ckpt_name)`
- `_save_superlattice_seed` (line 2131) `def _save_superlattice_seed(self, superlattice, ckpt_name)`
- `_classify_thermodynamic_phase` (line 2150) `def _classify_thermodynamic_phase(self, t_eff, cv, alpha)`
- `_estimate_critical_temperature` (line 2162) `def _estimate_critical_temperature(self, results)`
- `_verify_entropy_extensivity` (line 2174) `def _verify_entropy_extensivity(self, results)`
- `_plot_phase_diagram` (line 2188) `def _plot_phase_diagram(self, results)`
- `_plot_temperature_vs_purity` (line 2213) `def _plot_temperature_vs_purity(self, results)`
- `_compute_entropy_simple` (line 2239) `def _compute_entropy_simple(self, params)`
- `_compute_entropy` (line 2265) `def _compute_entropy(self, params)`
- `_compute_effective_volume` (line 2304) `def _compute_effective_volume(self, params)`
- `_plot_parameter_distribution` (line 2322) `def _plot_parameter_distribution(self, params, group_name, kde)`
- `_simulate_training_trajectory` (line 2350) `def _simulate_training_trajectory(self, final_params, final_delta)`
- `_compute_generalization_entropy` (line 2361) `def _compute_generalization_entropy(self, params, successful_ckpts)`
- `_fit_timescale` (line 2414) `def _fit_timescale(self, entropy_values)`
- `_plot_entropy_production` (line 2424) `def _plot_entropy_production(self, t, S, dS_dt, ckpt_name)`
- `_verify_scaling` (line 2442) `def _verify_scaling(self, coeffs, N)`
- `_recursive_strassen` (line 2453) `def _recursive_strassen(self, A, B, coeffs, N)`
- `_fit_extensivity` (line 2487) `def _fit_extensivity(self, errors, sizes, purity)`
- `_verify_extensivity_universality` (line 2501) `def _verify_extensivity_universality(self, results)`
- `_plot_extensivity` (line 2505) `def _plot_extensivity(self, sizes, errors, purity, ckpt_name)`
- `_find_broken_symmetries` (line 2518) `def _find_broken_symmetries(self, coeffs)`
- `_measure_uncertainty` (line 2526) `def _measure_uncertainty(self, coeffs, basis)`
- `_plot_uncertainty_distribution` (line 2537) `def _plot_uncertainty_distribution(self, coeffs, symmetry_basis, ckpt_name)`
- `_print_executive_summary` (line 2558) `def _print_executive_summary(self, results)`
- `_save_results` (line 2587) `def _save_results(self, results, filename)`
- `__init__` (line 2611) `def __init__(self, checkpoint_path, device)`
- `_load_model` (line 2617) `def _load_model(self, path, device)`
- `run_full_analysis` (line 2634) `def run_full_analysis(self)`
- `_assign_grade` (line 2656) `def _assign_grade(self, delta, alpha)`
- `_save_report` (line 2668) `def _save_report(self, report)`
- `generate_batch` (line 373) `def generate_batch(batch_size)`
- `get_tensor` (line 1444) `def get_tensor(key)`
- `model` (line 2415) `def model(t, A, tau, C)`
- `model` (line 2488) `def model(N, alpha, beta)`
- `convert_to_serializable` (line 2590) `def convert_to_serializable(obj)`
- `dataloader` (line 2637) `def dataloader()`
- `sample_dataloader` (line 1906) `def sample_dataloader()`
- `sample_dataloader` (line 2039) `def sample_dataloader()`
- `dataloader` (line 1522) `def dataloader()`

### SH (1 files)

#### `install.sh`
**Path:** `install.sh`

*No symbols extracted*
