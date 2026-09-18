"""PHY-ML: Minimal Verified Uncoded Physical Layer (PHY) Core powered by Sionna 2.0."""

from channel import (
    BlockRNG,
    ChannelOutput,
    make_block_rng,
    generate_channel_coefficient,
    generate_standard_noise,
    apply_channel,
)
from phy_engine import (
    ModulationMode,
    Mode,
    BlockEvaluationResult,
    MODES,
    MODE_BY_ID,
    MODE_BY_NAME,
    constellation,
    modulate,
    demodulate,
    PHYEngine,
)
from metrics import (
    count_errors,
    count_block_errors,
    compute_ber,
    compute_bler,
    RawPHYCounters,
)

__all__ = [
    "BlockRNG",
    "ChannelOutput",
    "make_block_rng",
    "generate_channel_coefficient",
    "generate_standard_noise",
    "apply_channel",
    "ModulationMode",
    "Mode",
    "BlockEvaluationResult",
    "MODES",
    "MODE_BY_ID",
    "MODE_BY_NAME",
    "constellation",
    "modulate",
    "demodulate",
    "PHYEngine",
    "count_errors",
    "count_block_errors",
    "compute_ber",
    "compute_bler",
    "RawPHYCounters",
]
