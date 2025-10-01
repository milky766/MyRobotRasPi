"""Trajectory preset definitions (speed × continuity × synchrony).

Minimal initial version to mirror original project's taxonomy while keeping
logic isolated. These presets define ranges used by RandomTrajectory/segment
sampler (to be ported next step).

Dimensions:
- speed: slow | middle | fast (affects duration & amplitude ranges)
- continuity: step | trapez | mix (profile shape selection)
- sync: sync | async (whether all joints update simultaneously)

Each preset maps to a dict with keys:
  duration_range: (min_s, max_s)    # per segment total duration
  amplitude_range: (min_pct, max_pct)  # relative or absolute target span (placeholder)
  profile: 'step' | 'trapez' | 'mix'
  sync: bool

Note: amplitude_range currently expressed in percent (0-100). Adjust later if
joint-space units differ.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass(frozen=True)
class TrajectoryPreset:
    duration_range: Tuple[float, float]
    amplitude_range: Tuple[float, float]
    profile: str  # 'step', 'trapez', 'mix'
    sync: bool

# Base tables (tunable). Values chosen as placeholders approximating original
# intent; refine after RandomTrajectory port & empirical review.
_DURATION = {
    'slow': (2.0, 4.0),
    'middle': (1.0, 2.0),
    'fast': (0.4, 1.0),
}
_AMPLITUDE = {
    'slow': (5.0, 15.0),
    'middle': (10.0, 25.0),
    'fast': (15.0, 40.0),
}

_CONTINUITY_PROFILES = {
    'step': 'step',
    'trapez': 'trapez',
    'mix': 'mix',
}

_SYNC_FLAG = {
    'sync': True,
    'async': False,
}

def build_preset(speed: str, continuity: str, sync_mode: str) -> TrajectoryPreset:
    if speed not in _DURATION:
        raise ValueError(f'Unknown speed: {speed}')
    if speed not in _AMPLITUDE:
        raise ValueError(f'Unknown speed amplitude: {speed}')
    if continuity not in _CONTINUITY_PROFILES:
        raise ValueError(f'Unknown continuity: {continuity}')
    if sync_mode not in _SYNC_FLAG:
        raise ValueError(f'Unknown sync mode: {sync_mode}')
    return TrajectoryPreset(
        duration_range=_DURATION[speed],
        amplitude_range=_AMPLITUDE[speed],
        profile=_CONTINUITY_PROFILES[continuity],
        sync=_SYNC_FLAG[sync_mode],
    )

# Precompute combinations
PRESETS: Dict[str, TrajectoryPreset] = {}
for speed in _DURATION.keys():
    for continuity in _CONTINUITY_PROFILES.keys():
        for sync_mode in _SYNC_FLAG.keys():
            name = f'{continuity}_{sync_mode}_{speed}'
            PRESETS[name] = build_preset(speed, continuity, sync_mode)

__all__ = ['TrajectoryPreset', 'build_preset', 'PRESETS']
