"""
Noosphere — Physics-Informed World Model Agent

A single model that learns to perceive, predict, and act across
physical domains using multimodal sensor fusion and hard-coded
physical laws as differentiable constraints.

Quick start
-----------
    from noosphere import NoosphereAgent, AgentConfig

    cfg   = AgentConfig(n_actions=6, n_eeg_ch=64, n_nodes=20)
    agent = NoosphereAgent(cfg, device=torch.device("cuda"))

    obs    = {"rgb": ..., "eeg": ..., "kinematics": ...}
    action, info = agent.step(obs)

Adding a new sensor modality
-----------------------------
    from noosphere.tokenizer import ImagePatchTokenizer
    agent.perception.tokenizer.register_modality(
        "thermal", ImagePatchTokenizer(1, cfg.d_model, patch_size=8)
    )
    # Pass "thermal": tensor in your obs dict — no other changes needed.
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

__version__ = "1.0.0"
__all__ = [
    "NoosphereAgent", "AgentConfig",
    "HybridPerceptionModel",
    "RSSM", "ConsequenceModel", "ObservationDecoder",
    "PhysicsAugmentedRSSM",
    "Actor", "Critic", "ActionEncoder", "MCTSPlanner",
    "SequenceReplayBuffer", "EpisodicMemory", "WorkingMemory",
    "S4EEGEncoder", "KinematicGNN",
    "UnifiedTokenizer", "build_tokenizer",
]
