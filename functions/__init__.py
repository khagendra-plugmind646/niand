"""
Near-Field Integrated Sensing and Communications (NF-ISAC) Package

This package contains the implementation of the NF-ISAC system,
including channel generation, beamforming, and optimization modules.
"""

# Import key functions to make them available at the package level
from .parameters import para_init
from .channel_generation import generate_channel
from .beamforming import beamfocusing
from .rate_calculator import rate_calculator
from .sdr_fully_digital import SDR_fully_digital
from .fim import FIM
from .music_estimation import music_estimation, plot_music_spectrum

__all__ = [
    'para_init',
    'generate_channel',
    'beamfocusing',
    'rate_calculator',
    'SDR_fully_digital',
    'FIM',
    'music_estimation',
    'plot_music_spectrum'
]
