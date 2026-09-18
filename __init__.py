"""PHY-ML: Minimal Verified Uncoded Physical Layer (PHY) Core."""

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
    RawPHYCounters,
    compute_ber,
    compute_bler,
    compute_raw_goodput,
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
    "RawPHYCounters",
    "compute_ber",
    "compute_bler",
    "compute_raw_goodput",
]
