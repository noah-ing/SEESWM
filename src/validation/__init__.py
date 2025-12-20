"""
Experimental Validation Framework for SEESWM Hypothesis Testing.

This module provides research-grade tools for validating the core hypothesis:
"Collective intelligence from many small specialized agents, embedded in
simulated worlds, will exhibit emergent capabilities that equivalent-parameter
monolithic models cannot."

Components:
- Ablation studies
- Scaling law experiments
- Synergy measurement (information-theoretic)
- Generalization tests
- Emergent behavior detection
- Comparison baselines
- Statistical rigor utilities
- Interpretability suite
"""

from .ablations import (
    AblationConfig,
    AblationStudy,
    run_ablation_suite,
)

from .scaling import (
    ScalingExperiment,
    plot_scaling_laws,
    find_phase_transitions,
)

from .synergy import (
    SynergyMeasurer,
    PartialInformationDecomposition,
    compute_true_synergy,
)

from .generalization import (
    GeneralizationTest,
    ZeroShotTransfer,
    CompositionalTest,
    OODTest,
)

from .emergence import (
    EmergenceDetector,
    BehaviorProbe,
    CommunicationAnalyzer,
)

from .baselines import (
    BaselineComparison,
    SingleAgentBaseline,
    EnsembleBaseline,
    CentralizedBaseline,
)

from .statistics import (
    ExperimentStats,
    paired_significance_test,
    compute_effect_size,
    confidence_interval,
    report_results,
)

from .interpretability import (
    RepresentationProbe,
    MessageAnalyzer,
    CausalIntervention,
    ActivationPatcher,
)

__all__ = [
    # Ablations
    'AblationConfig', 'AblationStudy', 'run_ablation_suite',
    # Scaling
    'ScalingExperiment', 'plot_scaling_laws', 'find_phase_transitions',
    # Synergy
    'SynergyMeasurer', 'PartialInformationDecomposition', 'compute_true_synergy',
    # Generalization
    'GeneralizationTest', 'ZeroShotTransfer', 'CompositionalTest', 'OODTest',
    # Emergence
    'EmergenceDetector', 'BehaviorProbe', 'CommunicationAnalyzer',
    # Baselines
    'BaselineComparison', 'SingleAgentBaseline', 'EnsembleBaseline', 'CentralizedBaseline',
    # Statistics
    'ExperimentStats', 'paired_significance_test', 'compute_effect_size',
    'confidence_interval', 'report_results',
    # Interpretability
    'RepresentationProbe', 'MessageAnalyzer', 'CausalIntervention', 'ActivationPatcher',
]
