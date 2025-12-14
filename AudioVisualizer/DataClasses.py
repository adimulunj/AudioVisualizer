from dataclasses import dataclass


@dataclass
class WindowConfig:
    """Window configuration settings."""
    width: int = 600
    height: int = 80
    x: str = "auto"
    y: str = "1710"


@dataclass
class ColorConfig:
    """Color configuration settings."""
    background: str = "#000000"
    line: str = "#00ff88"
    glow: str = "#2bfbe5"
    status_text: str = "#00ff88"


@dataclass
class LineConfig:
    """Line styling configuration."""
    width: float = 4.0
    alpha: float = 1.0
    style: str = "solid"


@dataclass
class GlowConfig:
    """Glow effect configuration."""
    width: float = 10.0
    alpha: float = 0.4


@dataclass
class GradientConfig:
    """Gradient color configuration."""
    enabled: bool = True
    low_color: str = "#0000ff"
    mid_color: str = "#00ff00"
    high_color: str = "#ff0000"


@dataclass
class EffectsConfig:
    """Visual effects configuration."""
    fill_under_curve: bool = False
    fill_alpha: float = 0.3
    mirror_mode: bool = False
    symmetry_mode: bool = False
    thick_on_bass: bool = False


@dataclass
class AudioConfig:
    """Audio processing configuration."""
    smoothing: int = 8
    amplitude_scale: float = 0.8
    amplitude_boost: float = 0.0
    rotation_angle: int = 0
    y_offset: float = 0.0