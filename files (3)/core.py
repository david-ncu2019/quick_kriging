"""
Core geostatistics functions for synthetic field generation and kriging analysis.
"""

import numpy as np
from gstools import SRF, Gaussian, Spherical, Exponential
from pykrige import OrdinaryKriging
from typing import Tuple, Dict, Optional, Any


def generate_synthetic_field(
    x_max: float,
    y_max: float, 
    nx: int,
    ny: int,
    model_type: str = "gaussian",
    variance: float = 1.0,
    length_scale: float = 20.0,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate a synthetic 2D random field using GSTools.
    
    Args:
        x_max: Maximum X coordinate
        y_max: Maximum Y coordinate  
        nx: Number of grid points in X direction
        ny: Number of grid points in Y direction
        model_type: Covariance model ("gaussian", "spherical", "exponential")
        variance: Field variance (sill)
        length_scale: Correlation length (range)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (x_coords, y_coords, field_values)
        x_coords: 1D array of X coordinates
        y_coords: 1D array of Y coordinates  
        field_values: 2D array of field values (ny, nx)
    """
    # Create coordinate arrays
    x = np.linspace(0, x_max, nx)
    y = np.linspace(0, y_max, ny)
    
    # Select covariance model
    model_map = {
        "gaussian": Gaussian,
        "spherical": Spherical,
        "exponential": Exponential
    }
    
    if model_type.lower() not in model_map:
        raise ValueError(f"Unknown model_type: {model_type}")
        
    # Create covariance model
    model = model_map[model_type.lower()](dim=2, var=variance, len_scale=length_scale)
    
    # Generate structured random field
    srf = SRF(model, seed=seed)
    field = srf.structured([x, y])
    
    return x, y, field


def run_kriging(
    sample_x: np.ndarray,
    sample_y: np.ndarray, 
    sample_values: np.ndarray,
    grid_x: np.ndarray,
    grid_y: np.ndarray,
    params: Optional[Dict[str, Any]] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """
    Perform ordinary kriging interpolation.
    
    Args:
        sample_x: X coordinates of sample points
        sample_y: Y coordinates of sample points
        sample_values: Observed values at sample points
        grid_x: X coordinates for prediction grid
        grid_y: Y coordinates for prediction grid
        params: Optional kriging parameters dict with 'model' and 'angle' keys
        
    Returns:
        Tuple of (predictions, variances, final_params)
        predictions: 2D array of interpolated values
        variances: 2D array of kriging variances
        final_params: Dict with final model parameters
    """
    # Set up kriging
    if params is None:
        # Auto-optimize variogram model
        ok = OrdinaryKriging(
            sample_x, sample_y, sample_values,
            variogram_model='gaussian',  # Default fallback
            enable_plotting=False
        )
        final_params = {
            'model': ok.variogram_model,
            'angle': 0.0
        }
    else:
        # Use provided parameters
        model = params.get('model', 'gaussian')
        angle = params.get('angle', 0.0)
        
        ok = OrdinaryKriging(
            sample_x, sample_y, sample_values,
            variogram_model=model,
            anisotropy_angle=angle,
            enable_plotting=False
        )
        final_params = {
            'model': model,
            'angle': angle
        }
    
    # Create prediction grid
    grid_x_2d, grid_y_2d = np.meshgrid(grid_x, grid_y)
    
    # Perform kriging
    predictions, variances = ok.execute('grid', grid_x, grid_y)
    
    return predictions, variances, final_params
