"""
Vegetation Index Calculator for Sentinel-2 Imagery
NSW Bushfire Risk Assessment Project

This module provides functionality to calculate vegetation indices
from Sentinel-2 satellite imagery for bushfire risk assessment.
"""

import numpy as np
import rasterio
from pathlib import Path
from typing import Tuple, Dict, Optional, Union
import logging

logger = logging.getLogger(__name__)

class VegetationIndexCalculator:
    """Calculate vegetation indices from Sentinel-2 imagery."""
    
    def __init__(self, scene_path: Union[str, Path]):
        """
        Initialize calculator with Sentinel-2 scene path.
        
        Args:
            scene_path: Path to Sentinel-2 scene directory or individual band files
        """
        self.scene_path = Path(scene_path)
        self.bands = {}
        self.metadata = {}
        
    def load_band(self, band_name: str, band_file: Union[str, Path]) -> np.ndarray:
        """
        Load a single band from file.
        
        Args:
            band_name: Name identifier for the band (e.g., 'B04', 'B08')
            band_file: Path to the band file
            
        Returns:
            Band data as numpy array
        """
        try:
            with rasterio.open(band_file) as src:
                band_data = src.read(1).astype(np.float32)
                # Store metadata from first band loaded
                if not self.metadata:
                    self.metadata = {
                        'transform': src.transform,
                        'crs': src.crs,
                        'width': src.width,
                        'height': src.height,
                        'nodata': src.nodata
                    }
                self.bands[band_name] = band_data
                logger.info(f"Loaded band {band_name} with shape {band_data.shape}")
                return band_data
        except Exception as e:
            logger.error(f"Failed to load band {band_name} from {band_file}: {e}")
            raise
    
    def load_bands_from_paths(self, band_paths: Dict[str, Union[str, Path]]) -> None:
        """
        Load multiple bands from file paths.
        
        Args:
            band_paths: Dictionary mapping band names to file paths
                       e.g., {'B04': 'path/to/B04.tif', 'B08': 'path/to/B08.tif'}
        """
        for band_name, band_path in band_paths.items():
            self.load_band(band_name, band_path)
    
    def calculate_ndvi(self, red_band: str = 'B04', nir_band: str = 'B08') -> np.ndarray:
        """
        Calculate Normalized Difference Vegetation Index.
        
        NDVI = (NIR - Red) / (NIR + Red)
        Range: -1 to 1 (higher values indicate healthier vegetation)
        
        Args:
            red_band: Name of red band in loaded bands
            nir_band: Name of NIR band in loaded bands
            
        Returns:
            2D array of NDVI values
        """
        if red_band not in self.bands or nir_band not in self.bands:
            raise ValueError(f"Required bands {red_band} and {nir_band} not loaded")
        
        red = self.bands[red_band]
        nir = self.bands[nir_band]
        
        # Avoid division by zero
        denominator = nir + red
        ndvi = np.where(
            denominator != 0,
            (nir - red) / denominator,
            0
        )
        
        # Clip to valid NDVI range
        ndvi = np.clip(ndvi, -1, 1)
        
        logger.info(f"Calculated NDVI - Min: {ndvi.min():.3f}, Max: {ndvi.max():.3f}, Mean: {ndvi.mean():.3f}")
        return ndvi
    
    def calculate_ndmi(self, nir_band: str = 'B08', swir1_band: str = 'B11') -> np.ndarray:
        """
        Calculate Normalized Difference Moisture Index.
        
        NDMI = (NIR - SWIR1) / (NIR + SWIR1)
        Range: -1 to 1 (higher values indicate higher moisture content)
        
        Args:
            nir_band: Name of NIR band in loaded bands
            swir1_band: Name of SWIR1 band in loaded bands
            
        Returns:
            2D array of NDMI values
        """
        if nir_band not in self.bands or swir1_band not in self.bands:
            raise ValueError(f"Required bands {nir_band} and {swir1_band} not loaded")
        
        nir = self.bands[nir_band]
        swir1 = self.bands[swir1_band]
        
        # Avoid division by zero
        denominator = nir + swir1
        ndmi = np.where(
            denominator != 0,
            (nir - swir1) / denominator,
            0
        )
        
        # Clip to valid NDMI range
        ndmi = np.clip(ndmi, -1, 1)
        
        logger.info(f"Calculated NDMI - Min: {ndmi.min():.3f}, Max: {ndmi.max():.3f}, Mean: {ndmi.mean():.3f}")
        return ndmi
    
    def calculate_nbr(self, nir_band: str = 'B08', swir2_band: str = 'B12') -> np.ndarray:
        """
        Calculate Normalized Burn Ratio.
        
        NBR = (NIR - SWIR2) / (NIR + SWIR2)
        Range: -1 to 1 (used for burn severity assessment)
        
        Args:
            nir_band: Name of NIR band in loaded bands
            swir2_band: Name of SWIR2 band in loaded bands
            
        Returns:
            2D array of NBR values
        """
        if nir_band not in self.bands or swir2_band not in self.bands:
            raise ValueError(f"Required bands {nir_band} and {swir2_band} not loaded")
        
        nir = self.bands[nir_band]
        swir2 = self.bands[swir2_band]
        
        # Avoid division by zero
        denominator = nir + swir2
        nbr = np.where(
            denominator != 0,
            (nir - swir2) / denominator,
            0
        )
        
        # Clip to valid NBR range
        nbr = np.clip(nbr, -1, 1)
        
        logger.info(f"Calculated NBR - Min: {nbr.min():.3f}, Max: {nbr.max():.3f}, Mean: {nbr.mean():.3f}")
        return nbr
    
    def calculate_all_indices(self) -> Dict[str, np.ndarray]:
        """
        Calculate all available vegetation indices.
        
        Returns:
            Dictionary containing all calculated indices
        """
        indices = {}
        
        # Calculate NDVI if red and NIR bands are available
        if 'B04' in self.bands and 'B08' in self.bands:
            indices['NDVI'] = self.calculate_ndvi()
        
        # Calculate NDMI if NIR and SWIR1 bands are available
        if 'B08' in self.bands and 'B11' in self.bands:
            indices['NDMI'] = self.calculate_ndmi()
        
        # Calculate NBR if NIR and SWIR2 bands are available
        if 'B08' in self.bands and 'B12' in self.bands:
            indices['NBR'] = self.calculate_nbr()
        
        logger.info(f"Calculated {len(indices)} vegetation indices")
        return indices
    
    def apply_cloud_mask(self, data: np.ndarray, scl_band: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Apply cloud mask using Scene Classification Layer (SCL).
        
        Args:
            data: Data array to mask
            scl_band: Scene Classification Layer band (if available)
            
        Returns:
            Masked data array
        """
        if scl_band is None and 'SCL' in self.bands:
            scl_band = self.bands['SCL']
        
        if scl_band is not None:
            # SCL values: 0=no_data, 1=saturated, 3=cloud_shadow, 8=cloud_medium, 9=cloud_high, 10=thin_cirrus
            cloud_mask = np.isin(scl_band, [0, 1, 3, 8, 9, 10])
            data_masked = np.where(cloud_mask, np.nan, data)
            logger.info(f"Applied cloud mask, {np.sum(cloud_mask)} pixels masked")
            return data_masked
        
        return data
    
    def save_index_to_file(self, index_data: np.ndarray, output_path: Union[str, Path], 
                          index_name: str = "index") -> None:
        """
        Save calculated index to GeoTIFF file.
        
        Args:
            index_data: Calculated index array
            output_path: Output file path
            index_name: Name of the index for metadata
        """
        if not self.metadata:
            raise ValueError("No metadata available. Load bands first.")
        
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create output profile
        profile = {
            'driver': 'GTiff',
            'height': self.metadata['height'],
            'width': self.metadata['width'],
            'count': 1,
            'dtype': index_data.dtype,
            'crs': self.metadata['crs'],
            'transform': self.metadata['transform'],
            'compress': 'lzw'
        }
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(index_data, 1)
            dst.set_band_description(1, index_name)
        
        logger.info(f"Saved {index_name} to {output_path}")


def quick_calculate_ndvi(red_path: Union[str, Path], nir_path: Union[str, Path]) -> Tuple[np.ndarray, dict]:
    """
    Quick NDVI calculation from file paths.
    
    Args:
        red_path: Path to red band file
        nir_path: Path to NIR band file
        
    Returns:
        Tuple of (NDVI array, metadata dict)
    """
    calc = VegetationIndexCalculator("")
    calc.load_band('B04', red_path)
    calc.load_band('B08', nir_path)
    ndvi = calc.calculate_ndvi()
    return ndvi, calc.metadata 