"""
Noosphere v1.6.0

New in this version:
- WorldModelBundle: export/import transferable world dynamics for community sharing
  Only person-independent components (RSSM, physics, consequence model) are included.
  S4 encoder, apparatus calibration, and personal data are explicitly excluded.
- Bug fixes: removed dead shlex import; BCIApparatusEnv now handles both
  callable and .next_segment() EEG sources correctly.
"""

from noosphere.agent      import NoosphereAgent, AgentConfig
from noosphere.perception import HybridPerceptionModel
from noosphere.rssm       import (
    RSSM, ConsequenceModel, ObservationDecoder,
    DigitalConsequenceHead, EnhancedConsequenceModel,
)
from noosphere.physics    import PhysicsAugmentedRSSM
from noosphere.planner    import Actor, Critic, ActionEncoder, MCTSPlanner
from noosphere.memory     import SequenceReplayBuffer, EpisodicMemory, WorkingMemory
from noosphere.s4_eeg     import S4EEGEncoder
from noosphere.gnn        import KinematicGNN
from noosphere.tokenizer  import UnifiedTokenizer, build_tokenizer
from noosphere.apparatus  import (
    MovementExecutor, KinematicSolver, ObstacleSphere,
    IntentionFilter, AnomalyDetector,
    CoordinatePredictor, SparseGPPredictor, NeuralCoordinatePredictor,
    TemporalSmoother, CalibrationSession, PositionErrorFeedback,
    ArmConfig, JointState,
)
from noosphere.hardware   import ServoController
from noosphere.proto      import NCPEncoder, NCPDecoder, Channel, MsgType, NCPTransport
from noosphere.learning   import (
    LearningManager, LearningConfig, LearningSignal,
    SupervisedCoordinateLoss, S4XYZSupervisionLoss, PositionErrorLoss,
    StepNFTPolicy, StepNFTLoss, EEGAugment,
)
from noosphere.actions    import (
    Action, ActionSpace, ActBridge, Executor, Tier,
    NullExecutor, ShellExecutor, ApparatusExecutor,
    DigitalStateObserver, ShellOutputEncoder,
    make_apparatus_space, make_shell_space, make_binary_space,
)
from noosphere.trainer    import (
    Trainer, TrainerConfig, Env,
    BCIApparatusEnv, SyntheticBCIEnv,
    reach_reward, save_checkpoint, load_checkpoint,
)
from noosphere.monitor    import Monitor, MonitorConfig, Alert, Level
from noosphere.bundle     import (
    export_bundle, load_bundle, inspect_bundle, check_compatibility,
    BundleMetadata, ALL_BUNDLE_KEYS,
)

__version__ = "1.6.0"
__all__ = [
    # Core agent
    "NoosphereAgent", "AgentConfig",
    # Perception
    "HybridPerceptionModel", "S4EEGEncoder", "KinematicGNN",
    "UnifiedTokenizer", "build_tokenizer",
    # World model
    "RSSM", "PhysicsAugmentedRSSM",
    "ConsequenceModel", "ObservationDecoder",
    "DigitalConsequenceHead", "EnhancedConsequenceModel",
    # Planning
    "Actor", "Critic", "ActionEncoder", "MCTSPlanner",
    # Memory
    "SequenceReplayBuffer", "EpisodicMemory", "WorkingMemory",
    # Apparatus
    "MovementExecutor", "KinematicSolver", "ObstacleSphere",
    "IntentionFilter", "AnomalyDetector",
    "CoordinatePredictor", "SparseGPPredictor", "NeuralCoordinatePredictor",
    "TemporalSmoother", "CalibrationSession", "PositionErrorFeedback",
    "ArmConfig", "JointState",
    # Hardware
    "ServoController",
    # Protocol
    "NCPEncoder", "NCPDecoder", "Channel", "MsgType", "NCPTransport",
    # Learning
    "LearningManager", "LearningConfig", "LearningSignal",
    "SupervisedCoordinateLoss", "S4XYZSupervisionLoss", "PositionErrorLoss",
    "StepNFTPolicy", "StepNFTLoss", "EEGAugment",
    # Actions
    "Action", "ActionSpace", "ActBridge", "Executor", "Tier",
    "NullExecutor", "ShellExecutor", "ApparatusExecutor",
    "DigitalStateObserver", "ShellOutputEncoder",
    "make_apparatus_space", "make_shell_space", "make_binary_space",
    # Training
    "Trainer", "TrainerConfig", "Env",
    "BCIApparatusEnv", "SyntheticBCIEnv", "reach_reward",
    "save_checkpoint", "load_checkpoint",
    # Monitoring
    "Monitor", "MonitorConfig", "Alert", "Level",
    # Bundle
    "export_bundle", "load_bundle", "inspect_bundle", "check_compatibility",
    "BundleMetadata", "ALL_BUNDLE_KEYS",
]
