"""Central configuration for the digital-twin prosthetic control system."""

from dataclasses import dataclass, field


@dataclass
class EncoderConfig:
    n_eeg_channels: int = 21
    n_prop_channels: int = 6
    d_model: int = 128
    d_state: int = 64
    n_layers: int = 4
    dropout: float = 0.1


@dataclass
class DecoderConfig:
    n_dof: int = 6
    d_hidden: int = 64


@dataclass
class KalmanConfig:
    # dt is set at runtime to match fs; default: 256 Hz
    dt: float = 1.0 / 256
    process_noise: float = 0.01


@dataclass
class SafetyConfig:
    ern_threshold: float = 0.7
    sigma_threshold: float = 1.5
    ern_halt_duration: float = 0.5
    sigma_halt_duration: float = 0.1
    watchdog_timeout: float = 2.0


@dataclass
class OnlineConfig:
    experience_capacity: int = 512
    adapt_every_n_steps: int = 100
    adapt_n_gradient_steps: int = 5
    adapt_batch_size: int = 32
    adapt_lr: float = 1e-4
    ema_decay: float = 0.995  # parameter EMA to prevent catastrophic forgetting


@dataclass
class JEPAConfig:
    context_fraction: float = 0.75
    ema_decay: float = 0.99
    lr: float = 3e-4
    weight_decay: float = 1e-4
    n_epochs: int = 50
    batch_size: int = 64
    warmup_epochs: int = 5


@dataclass
class SynthConfig:
    fs: int = 256
    n_channels: int = 21
    n_subjects: int = 10
    n_trials: int = 50
    trial_duration_s: float = 4.0


@dataclass
class ZMQConfig:
    pub_port: int = 5555
    sub_port: int = 5556
    heartbeat_ms: int = 100
    watchdog_s: float = 2.0


@dataclass
class V2Config:
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    decoder: DecoderConfig = field(default_factory=DecoderConfig)
    kalman: KalmanConfig = field(default_factory=KalmanConfig)
    safety: SafetyConfig = field(default_factory=SafetyConfig)
    online: OnlineConfig = field(default_factory=OnlineConfig)
    jepa: JEPAConfig = field(default_factory=JEPAConfig)
    synth: SynthConfig = field(default_factory=SynthConfig)
    zmq: ZMQConfig = field(default_factory=ZMQConfig)
    log_dir: str = "v2_digital_self_replication/logs"
    checkpoint_dir: str = "v2_digital_self_replication/checkpoints"


# DOF semantics for the 3-finger robotic arm
DOF_LABELS = [
    "shoulder_yaw",    # horizontal rotation
    "shoulder_pitch",  # elevation
    "shoulder_roll",   # internal/external rotation
    "elbow_flex",      # flexion/extension
    "wrist_rotate",    # pronation/supination
    "grip_aperture",   # open/close 3-finger gripper (0=closed, 1=open)
]

# Standard 21-channel BCI electrode layout (10-20 system)
EEG_CHANNELS_21 = [
    "Fp1", "Fp2",                         # frontal polar   (0-1)
    "F7", "F3", "Fz", "F4", "F8",         # frontal         (2-6)
    "T7", "C3", "Cz", "C4", "T8",         # central         (7-11)
    "P7", "P3", "Pz", "P4", "P8",         # parietal        (12-16)
    "O1", "Oz", "O2",                      # occipital       (17-19)
    "FCz",                                 # fronto-central  (20) — key ERN site
]

# Channel index shortcuts
IDX_C3  = EEG_CHANNELS_21.index("C3")   # 8  — left motor cortex (right-arm ERD)
IDX_CZ  = EEG_CHANNELS_21.index("Cz")   # 9  — vertex
IDX_C4  = EEG_CHANNELS_21.index("C4")   # 10 — right motor cortex (right-arm ERS)
IDX_FCZ = EEG_CHANNELS_21.index("FCz")  # 20 — error negativity site
IDX_T7  = EEG_CHANNELS_21.index("T7")   # 7  — temporal (shoulder)
IDX_T8  = EEG_CHANNELS_21.index("T8")   # 11 — temporal (shoulder)
