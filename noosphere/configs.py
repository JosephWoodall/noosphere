from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class PerceptionConfig:
    d_model: int = 256
    n_heads: int = 8
    n_layers: int = 6
    n_eeg_ch: int = 3
    eeg_sfreq: float = 256.0
    s4_d_state: int = 64
    s4_n_blocks: int = 4
    s4_downsample: int = 4
    n_nodes: int = 20
    node_feat_dim: int = 12
    gnn_n_layers: int = 3
    patch_size: int = 8
    max_reach: float = 0.70

@dataclass
class WorldModelConfig:
    action_dim: int = 64
    det_dim: int = 512
    stoch_cats: int = 32
    stoch_cls: int = 32
    hidden_dim: int = 256
    n_bodies: int = 4
    fluid_grid: int = 4
    dt: float = 1 / 60
    lambda_kl: float = 1.0
    lambda_recon: float = 0.5
    lambda_reward: float = 1.0
    lambda_physics: float = 0.5

@dataclass
class ActorCriticConfig:
    lr_actor_critic: float = 3e-4
    entropy_scale: float = 3e-4
    gamma: float = 0.99
    lam: float = 0.95
    bc_weight: float = 2.0
    imag_horizon: int = 15
    ac_updates: int = 5

@dataclass
class PlanningConfig:
    use_mcts: bool = True
    n_mcts_sims: int = 30
    mcts_horizon: int = 10
    max_velocity: float = 1.0

@dataclass
class TrainingConfig:
    batch_size: int = 16
    seq_len: int = 50
    lr_perception: float = 1e-4
    lr_world_model: float = 3e-4
    grad_clip: float = 100.0
    wm_updates: int = 5
    warmup_steps: int = 1000
    replay_capacity: int = 500
    episodic_capacity: int = 5000
    dry_run: bool = False

@dataclass
class BCIConfig:
    min_act_confidence: float = 0.3
    fast_path_threshold: float = 0.85
    momentum_decay: float = 0.05
    n_actions: int = 8
    enable_inter_agent_comms: bool = False # User must explicitly enable "Brain-Phone" features
    allow_collective_learning: bool = False # User must explicitly enable sharing Dynamics Insights

@dataclass
class AgentConfig:
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    world_model: WorldModelConfig = field(default_factory=WorldModelConfig)
    ac: ActorCriticConfig = field(default_factory=ActorCriticConfig)
    planning: PlanningConfig = field(default_factory=PlanningConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    bci: BCIConfig = field(default_factory=BCIConfig)

    @classmethod
    def from_legacy(cls, **kwargs):
        # Helper to map old flat config to nested
        p = PerceptionConfig(
            d_model=kwargs.get('d_model', 256),
            n_eeg_ch=kwargs.get('n_eeg_ch', 3),
            eeg_sfreq=kwargs.get('eeg_sfreq', 256.0),
            n_nodes=kwargs.get('n_nodes', 20),
            node_feat_dim=kwargs.get('node_feat_dim', 12),
            patch_size=kwargs.get('patch_size', 8),
            max_reach=kwargs.get('max_reach', 0.70)
        )
        wm = WorldModelConfig(
            action_dim=kwargs.get('action_dim', 64),
            det_dim=kwargs.get('det_dim', 512),
            stoch_cats=kwargs.get('stoch_cats', 32),
            stoch_cls=kwargs.get('stoch_cls', 32),
            hidden_dim=kwargs.get('hidden_dim', 256),
            n_bodies=kwargs.get('n_bodies', 4),
            fluid_grid=kwargs.get('fluid_grid', 4),
            dt=kwargs.get('dt', 1/60),
            lambda_kl=kwargs.get('lambda_kl', 1.0),
            lambda_recon=kwargs.get('lambda_recon', 0.5),
            lambda_reward=kwargs.get('lambda_reward', 1.0),
            lambda_physics=kwargs.get('lambda_physics', 0.5)
        )
        ac = ActorCriticConfig(
            lr_actor_critic=kwargs.get('lr_actor_critic', 3e-4),
            entropy_scale=kwargs.get('entropy_scale', 3e-4),
            gamma=kwargs.get('gamma', 0.99),
            lam=kwargs.get('lam', 0.95),
            bc_weight=kwargs.get('bc_weight', 2.0),
            imag_horizon=kwargs.get('imag_horizon', 15),
            ac_updates=kwargs.get('ac_updates', 5)
        )
        pl = PlanningConfig(
            use_mcts=kwargs.get('use_mcts', True),
            n_mcts_sims=kwargs.get('n_mcts_sims', 30),
            mcts_horizon=kwargs.get('mcts_horizon', 10),
            max_velocity=kwargs.get('max_velocity', 1.0)
        )
        tr = TrainingConfig(
            batch_size=kwargs.get('batch_size', 16),
            seq_len=kwargs.get('seq_len', 50),
            lr_perception=kwargs.get('lr_perception', 1e-4),
            lr_world_model=kwargs.get('lr_world_model', 3e-4),
            grad_clip=kwargs.get('grad_clip', 100.0),
            wm_updates=kwargs.get('wm_updates', 5),
            warmup_steps=kwargs.get('warmup_steps', 1000),
            replay_capacity=kwargs.get('replay_capacity', 500),
            episodic_capacity=kwargs.get('episodic_capacity', 5000),
            dry_run=kwargs.get('dry_run', False)
        )
        bci = BCIConfig(
            min_act_confidence=kwargs.get('min_act_confidence', 0.3),
            fast_path_threshold=kwargs.get('fast_path_threshold', 0.85),
            momentum_decay=kwargs.get('momentum_decay', 0.05),
            n_actions=kwargs.get('n_actions', 8)
        )
        return cls(perception=p, world_model=wm, ac=ac, planning=pl, training=tr, bci=bci)
