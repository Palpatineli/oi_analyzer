"""A package to analyze the optical imaging results acquired by
[CortexView2](https://github.com/Palpatineli/CortexView2)
"""
__version__ = "0.1.6"
from .imaging import (convert, calculate, circular_transform, circular_free, detrend,
                      get_colormap, online_result, colorize)

__all__ = ["convert", "calculate", "circular_transform", "circular_free", "colorize", "detrend",
           "get_colormap", "online_result"]
