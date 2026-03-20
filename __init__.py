"""
Noosphere v1.2.0

Quick start — any sensor subset works (missing streams are masked, not zeroed):
    obs = {"eeg": eeg_array}                        # EEG-only
    obs = {"rgb": rgb, "depth": depth}              # vision-only
    obs = {"eeg": eeg, "rgb": rgb, "kinematics": k} # all streams
    action, info = agent.step(obs)

Attaching a digital executor:
    from noosphere.actions import make_shell_space, ShellExecutor, ActBridge
    space  = make_shell_space(working_dir=".")
    bridge = ActBridge(space, ShellExecutor(allow_list=["ls","pwd","git"]))
    cfg    = AgentConfig(n_actions=space.n_actions)
    agent  = NoosphereAgent(cfg, device)
    agent.act_bridge = bridge

Continuous training:
    from noosphere.trainer import Trainer, TrainerConfig
    trainer = Trainer(agent, env, TrainerConfig())
    trainer.run()
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
from noosphere.actions    import (
    Action, ActionSpace, ActBridge, Executor,
    NullExecutor, ShellExecutor, ApparatusExecutor,
    make_apparatus_space, make_shell_space, make_binary_space,
)
from noosphere.trainer    import Trainer, TrainerConfig, Env, save_checkpoint, load_checkpoint

__version__ = "1.2.0"
__all__ = [
    "NoosphereAgent","AgentConfig",
    "HybridPerceptionModel","S4EEGEncoder","KinematicGNN","UnifiedTokenizer","build_tokenizer",
    "RSSM","PhysicsAugmentedRSSM","ConsequenceModel","ObservationDecoder",
    "Actor","Critic","ActionEncoder","MCTSPlanner",
    "SequenceReplayBuffer","EpisodicMemory","WorkingMemory",
    "MovementExecutor","KinematicSolver","ObstacleSphere",
    "IntentionFilter","AnomalyDetector","CoordinatePredictor","ArmConfig","JointState",
    "ServoController",
    "NCPEncoder","NCPDecoder","Channel","MsgType",
    "LearningManager","LearningConfig","LearningSignal","StepNFTPolicy","StepNFTLoss",
    "Action","ActionSpace","ActBridge","Executor",
    "NullExecutor","ShellExecutor","ApparatusExecutor",
    "make_apparatus_space","make_shell_space","make_binary_space",
    "Trainer","TrainerConfig","Env","save_checkpoint","load_checkpoint",
]
