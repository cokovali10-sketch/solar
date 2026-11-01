"""SOLAR - token optimization layer for LLM workflows.

The package is deliberately dependency-free at its core. Token counting falls
back to a calibrated heuristic, the semantic cache uses a local hashing
embedder, and the gateway speaks the OpenAI-compatible chat completions API.
"""

__version__ = "0.1.0"

from .config import SolarConfig
from .models import Message, OptimizationReport, OptimizedRequest, Usage
from .optimizer import SolarOptimizer

__all__ = [
    "Message",
    "OptimizationReport",
    "OptimizedRequest",
    "SolarConfig",
    "SolarOptimizer",
    "Usage",
]
