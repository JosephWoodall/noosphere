"""
Noosphere — Physics-Informed World Model Agent with BCI Apparatus Control

Quick start
-----------
    from noosphere import NoosphereAgent, AgentConfig

    cfg   = AgentConfig(n_actions=6, n_eeg_ch=3, n_nodes=6)
    agent = NoosphereAgent(cfg, device=torch.device("cpu"))

    obs    = {"eeg": eeg_array, "rgb": rgb_array, "kinematics": joints_array}
    action, info = agent.step(obs)

Adding a new sensor modality
-----------------------------
    from noosphere.tokenizer import ImagePatchTokenizer
    agent.perception.tokenizer.register_modality(
        "thermal", ImagePatchTokenizer(1, cfg.d_model, patch_size=8)
    )

BCI + apparatus control
-----------------------
    from noosphere.apparatus import MovementExecutor, IntentionFilter, AnomalyDetector
    from noosphere.hardware  import ServoController

    executor = MovementExecutor()
    servo    = ServoController(backend="sim")   # swap "rpi_pca9685" for real hardware
    commands = executor.plan_and_execute(target_xyz)
    for angles_deg in commands:
        servo.set_all_angles(angles_deg)

Communication protocol
----------------------
    from noosphere.proto import NCPEncoder, NCPDecoder, Channel
    enc = NCPEncoder()
    frame = enc.eeg_segment(raw_uv, probs, root_label, intent, xyz, v, f, ts)
    # publish frame to Channel.EEG_SOURCE via Redis or any transport
"""

from noosphere.agent      import NoosphereAgent, AgentConfig
from noosphere.perception import HybridPerceptionModel
from noosphere.rssm       import RSSM, ConsequenceModel, ObservationDecoder
from noosphere.physics    import PhysicsAugmentedRSSM
from noosphere.planner    import Actor, Critic, ActionEncoder, MCTSPlanner
from noosphere.memory     import SequenceReplayBuffer, EpisodicMemory, WorkingMemory
from noosphere.s4_eeg     import S4EEGEncoder
from noosphere.gnn        import KinematicGNN
from noosphere.tokenizer  import UnifiedTokenizer, build_tokenizer
from noosphere.apparatus  import (
    MovementExecutor, KinematicSolver, ObstacleSphere,
    IntentionFilter, AnomalyDetector, CoordinatePredictor,
    ArmConfig, JointState,
)
from noosphere.hardware   import ServoController
from noosphere.proto      import NCPEncoder, NCPDecoder, Channel, MsgType
from noosphere.learning   import (
    LearningManager, LearningConfig, LearningSignal,
    StepNFTPolicy, StepNFTLoss,
)

__version__ = "1.1.0"
__all__ = [
    # Agent
    "NoosphereAgent", "AgentConfig",
    # Perception
    "HybridPerceptionModel", "S4EEGEncoder", "KinematicGNN",
    "UnifiedTokenizer", "build_tokenizer",
    # World model
    "RSSM", "PhysicsAugmentedRSSM",
    "ConsequenceModel", "ObservationDecoder",
    # Planning
    "Actor", "Critic", "ActionEncoder", "MCTSPlanner",
    # Memory
    "SequenceReplayBuffer", "EpisodicMemory", "WorkingMemory",
    # Apparatus
    "MovementExecutor", "KinematicSolver", "ObstacleSphere",
    "IntentionFilter", "AnomalyDetector", "CoordinatePredictor",
    "ArmConfig", "JointState",
    # Hardware
    "ServoController",
    # Protocol
    "NCPEncoder", "NCPDecoder", "Channel", "MsgType",
    # Learning
    "LearningManager", "LearningConfig", "LearningSignal",
    "StepNFTPolicy", "StepNFTLoss",
]
