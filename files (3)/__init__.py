"""
Geostatistics Toolkit for Kriging Analysis
"""

from .core import generate_synthetic_field, run_kriging
from .io_utils import save_to_vtk

__version__ = "1.0.0"
__all__ = ['generate_synthetic_field', 'run_kriging', 'save_to_vtk']
