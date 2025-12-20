"""
Self-Model module - Meta-cognition and uncertainty quantification.

Phase 4 components:
- Uncertainty: Ensemble, MC Dropout, Evidential methods
- Calibration: Temperature scaling, Platt scaling, Focal loss
- Meta-learning: MAML, Meta-SGD, Reptile
- Theory of Mind: Belief modeling, intent prediction
- Integration: MetaCognitiveSwarm combining all capabilities
"""

from .capabilities import SelfModel, CapabilityBelief, TaskOutcome

from .uncertainty import (
    UncertaintyEstimate,
    EnsembleUncertainty,
    MCDropoutUncertainty,
    EvidentialNetwork,
    UncertaintyAggregator,
    UncertaintyHead,
)

from .calibration import (
    CalibrationMetrics,
    compute_calibration_metrics,
    TemperatureScaling,
    PlattScaling,
    FocalLoss,
    LabelSmoothing,
    CalibrationLoss,
    AdaptiveCalibrator,
    CalibrationTracker,
)

from .meta_learning import (
    Task,
    MetaLearningConfig,
    MAML,
    MetaSGD,
    Reptile,
    TaskEmbedding,
    AdaptiveMetaLearner,
)

from .theory_of_mind import (
    IntentType,
    BeliefState,
    AgentModel,
    BeliefEncoder,
    IntentPredictor,
    TheoryOfMind,
    CollectiveBeliefAggregator,
)

from .swarm_integration import (
    AgentMetaCognition,
    MetaCognitiveSwarm,
    MetaCognitionTrainer,
)

__all__ = [
    # Capabilities (Phase 1)
    "SelfModel",
    "CapabilityBelief",
    "TaskOutcome",
    # Uncertainty
    "UncertaintyEstimate",
    "EnsembleUncertainty",
    "MCDropoutUncertainty",
    "EvidentialNetwork",
    "UncertaintyAggregator",
    "UncertaintyHead",
    # Calibration
    "CalibrationMetrics",
    "compute_calibration_metrics",
    "TemperatureScaling",
    "PlattScaling",
    "FocalLoss",
    "LabelSmoothing",
    "CalibrationLoss",
    "AdaptiveCalibrator",
    "CalibrationTracker",
    # Meta-learning
    "Task",
    "MetaLearningConfig",
    "MAML",
    "MetaSGD",
    "Reptile",
    "TaskEmbedding",
    "AdaptiveMetaLearner",
    # Theory of Mind
    "IntentType",
    "BeliefState",
    "AgentModel",
    "BeliefEncoder",
    "IntentPredictor",
    "TheoryOfMind",
    "CollectiveBeliefAggregator",
    # Swarm Integration
    "AgentMetaCognition",
    "MetaCognitiveSwarm",
    "MetaCognitionTrainer",
]
