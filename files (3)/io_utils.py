"""
Input/Output utilities for geostatistics toolkit.
"""

import numpy as np
from typing import Dict
import xml.etree.ElementTree as ET
import base64
import os


def save_to_vtk(
    filepath: str,
    data_fields: Dict[str, np.ndarray],
    x_coords: np.ndarray,
    y_coords: np.ndarray,
    z_coord: float = 0.0
) -> None:
    """
    Save 2D field data to VTK rectilinear grid format.
    
    Args:
        filepath: Output path (without .vtr extension)
        data_fields: Dict mapping field names to 2D numpy arrays
        x_coords: 1D array of X coordinates
        y_coords: 1D array of Y coordinates 
        z_coord: Single Z coordinate value (default 0.0 for 2D)
    """
    # Ensure all fields have same shape
    ny, nx = next(iter(data_fields.values())).shape
    
    if len(x_coords) != nx or len(y_coords) != ny:
        raise ValueError("Coordinate arrays don't match field dimensions")
    
    # Create VTK XML structure
    vtk = ET.Element("VTKFile")
    vtk.set("type", "RectilinearGrid")
    vtk.set("version", "1.0")
    vtk.set("byte_order", "LittleEndian")
    
    # Create grid element
    grid = ET.SubElement(vtk, "RectilinearGrid")
    grid.set("WholeExtent", f"0 {nx-1} 0 {ny-1} 0 0")
    
    # Create piece element
    piece = ET.SubElement(grid, "Piece")
    piece.set("Extent", f"0 {nx-1} 0 {ny-1} 0 0")
    
    # Add coordinates
    coords = ET.SubElement(piece, "Coordinates")
    
    # X coordinates
    x_data = ET.SubElement(coords, "DataArray")
    x_data.set("type", "Float64")
    x_data.set("Name", "X_COORDINATES")
    x_data.set("NumberOfComponents", "1")
    x_data.set("format", "ascii")
    x_data.text = " ".join(map(str, x_coords))
    
    # Y coordinates
    y_data = ET.SubElement(coords, "DataArray")
    y_data.set("type", "Float64")
    y_data.set("Name", "Y_COORDINATES")
    y_data.set("NumberOfComponents", "1")
    y_data.set("format", "ascii")
    y_data.text = " ".join(map(str, y_coords))
    
    # Z coordinates (single value)
    z_data = ET.SubElement(coords, "DataArray")
    z_data.set("type", "Float64")
    z_data.set("Name", "Z_COORDINATES")
    z_data.set("NumberOfComponents", "1")
    z_data.set("format", "ascii")
    z_data.text = str(z_coord)
    
    # Add point data
    point_data = ET.SubElement(piece, "PointData")
    
    for field_name, field_values in data_fields.items():
        # Flatten field in VTK order (k fastest, then j, then i)
        flat_values = field_values.flatten()
        
        field_data = ET.SubElement(point_data, "DataArray")
        field_data.set("type", "Float64")
        field_data.set("Name", field_name)
        field_data.set("NumberOfComponents", "1")
        field_data.set("format", "ascii")
        field_data.text = " ".join(map(str, flat_values))
    
    # Write to file
    output_path = f"{filepath}.vtr"
    tree = ET.ElementTree(vtk)
    tree.write(output_path, xml_declaration=True, encoding="utf-8")
