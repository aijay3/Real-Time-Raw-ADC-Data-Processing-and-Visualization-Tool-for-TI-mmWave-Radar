"""
Module for coordinate transformation functions.
"""

import numpy as np
from scipy.ndimage import gaussian_filter
from numba import jit
from typing import Tuple

@jit(nopython=True)
def _interpolate_to_cartesian(r_indices, theta_indices, r_frac, theta_frac, 
                            r_grid_flat, x_grid_flat, z_grid_flat, range_azimuth, x_min, z_min,
                            resolution, x_axis_len, z_axis_len, max_range):
    """JIT-compiled function for the interpolation loop"""
    cartesian_map = np.zeros((z_axis_len, x_axis_len))
    
    for i in range(len(r_indices)):
        if r_grid_flat[i] > max_range:
            continue
            
        r_idx = int(r_indices[i])
        theta_idx = int(theta_indices[i])
        
        if (r_idx >= 0 and r_idx < range_azimuth.shape[0] - 1 and 
            theta_idx >= 0 and theta_idx < range_azimuth.shape[1] - 1):
            
            # Get the four nearest points
            f00 = range_azimuth[r_idx, theta_idx]
            f01 = range_azimuth[r_idx, theta_idx + 1]
            f10 = range_azimuth[r_idx + 1, theta_idx]
            f11 = range_azimuth[r_idx + 1, theta_idx + 1]
            
            # Bilinear interpolation
            r_f = r_frac[i]
            theta_f = theta_frac[i]
            
            # Interpolate along range
            f0 = f00 * (1 - r_f) + f10 * r_f
            f1 = f01 * (1 - r_f) + f11 * r_f
            
            # Interpolate along azimuth
            value = f0 * (1 - theta_f) + f1 * theta_f
            
            # Map to Cartesian grid
            x_idx = int((x_grid_flat[i] - x_min) / resolution)
            z_idx = int((z_grid_flat[i] - z_min) / resolution)
            
            if 0 <= x_idx < x_axis_len and 0 <= z_idx < z_axis_len:
                cartesian_map[z_idx, x_idx] = value
    
    return cartesian_map

# Create a singleton instance of CoordinateTransformer
_transformer = None

def get_transformer():
    """Get or create the singleton CoordinateTransformer instance."""
    global _transformer
    if _transformer is None:
        _transformer = CoordinateTransformer()
    return _transformer

def polar_to_cartesian(range_azimuth: np.ndarray, range_axis: np.ndarray, azimuth_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert polar coordinates (range-azimuth) to Cartesian coordinates (x-z).
    
    Args:
        range_azimuth: Range-Azimuth map data
        range_axis: Range values in meters
        azimuth_angles: Azimuth angles in degrees
        
    Returns:
        Tuple of (cartesian_map, x_axis, z_axis) where:
            cartesian_map: Range-Azimuth data in Cartesian coordinates
            x_axis: X-axis values in meters
            z_axis: Z-axis values in meters (same as range_axis)
    """
    return get_transformer().polar_to_cartesian(range_azimuth, range_axis, azimuth_angles)

class CoordinateTransformer:
    def __init__(self):
        self.range_axis = None
        self.azimuth_angles = None

    def polar_to_cartesian(self, range_azimuth: np.ndarray, range_axis: np.ndarray, azimuth_angles: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Convert polar coordinates (range-azimuth) to Cartesian coordinates (x-z).
        
        Args:
            range_azimuth: Range-Azimuth map data
            range_axis: Range values in meters
            azimuth_angles: Azimuth angles in degrees
            
        Returns:
            Tuple of (cartesian_map, x_axis, z_axis) where:
                cartesian_map: Range-Azimuth data in Cartesian coordinates
                x_axis: X-axis values in meters
                z_axis: Z-axis values in meters (same as range_axis)
        """
        # Store the latest data for future use
        self.range_axis = range_axis
        self.azimuth_angles = azimuth_angles
        # Convert azimuth angles from degrees to radians
        azimuth_rad = np.radians(azimuth_angles)
        
        # Calculate the maximum range for proper scaling
        max_range = np.max(range_axis)
        
        # Define Cartesian grid with appropriate resolution
        resolution = max(0.1, max_range / 500)  # Adaptive resolution based on max range
        
        # Define the Cartesian grid extent
        x_min, x_max = -max_range, max_range
        z_min, z_max = 0, max_range
        
        # Create Cartesian grid axes
        x_axis = np.arange(x_min, x_max + resolution, resolution)
        z_axis = np.arange(z_min, z_max + resolution, resolution)
        
        # Create meshgrid for Cartesian coordinates
        x_grid, z_grid = np.meshgrid(x_axis, z_axis)
        
        # Initialize variables for interpolation
        r_grid = np.sqrt(x_grid**2 + z_grid**2)
        theta_grid = np.arctan2(x_grid, z_grid)
    
        # Normalize theta to match the azimuth_rad range
        min_azimuth = np.min(azimuth_rad)
        max_azimuth = np.max(azimuth_rad)
        theta_grid = np.clip(theta_grid, min_azimuth, max_azimuth)
        
        # Prepare interpolation indices
        r_indices = np.interp(r_grid.flatten(), range_axis, np.arange(len(range_axis)))
        theta_indices = np.interp(theta_grid.flatten(), azimuth_rad, np.arange(len(azimuth_rad)))
        
        # Calculate fractional parts for bilinear interpolation
        r_frac = r_indices - r_indices.astype(int)
        theta_frac = theta_indices - theta_indices.astype(int)
        
        # Clip indices to valid range
        r_indices = np.clip(r_indices, 0, len(range_axis) - 2)
        theta_indices = np.clip(theta_indices, 0, len(azimuth_rad) - 2)
        
        # Call the JIT-compiled interpolation function
        cartesian_map = _interpolate_to_cartesian(
            r_indices, theta_indices, r_frac, theta_frac,
            r_grid.flatten(), x_grid.flatten(), z_grid.flatten(), range_azimuth,
            x_min, z_min, resolution, len(x_axis), len(z_axis), max_range
        )
        
        # Apply Gaussian smoothing
        #cartesian_map = gaussian_filter(cartesian_map, sigma=1.0)
        
        return cartesian_map, x_axis, z_axis
