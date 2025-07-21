"""
Visualization Module for Bushfire Risk Assessment
NSW Bushfire Risk Assessment Project

This module provides functionality to create static and interactive
visualizations of bushfire risk assessment results.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.colors import ListedColormap
import folium
import folium.plugins
import rasterio
import rasterio.plot
from pathlib import Path
from typing import Dict, Tuple, Optional, Union, List
import logging
import geopandas as gpd
from shapely.geometry import box
import base64
from io import BytesIO

logger = logging.getLogger(__name__)

class BushfireRiskVisualizer:
    """
    Create visualizations for bushfire risk assessment data.
    
    Supports both static matplotlib plots and interactive Folium maps
    for displaying risk scores, vegetation indices, and analysis results.
    """
    
    def __init__(self):
        """Initialize visualizer with default settings."""
        self.risk_colors = {
            1: '#2E8B57',  # Very Low - Sea Green
            2: '#FFD700',  # Low - Gold
            3: '#FF8C00',  # Moderate - Dark Orange
            4: '#FF4500',  # High - Orange Red
            5: '#DC143C'   # Very High - Crimson
        }
        
        self.risk_labels = {
            1: 'Very Low',
            2: 'Low', 
            3: 'Moderate',
            4: 'High',
            5: 'Very High'
        }
        
        # Set up matplotlib style
        plt.style.use('default')
        
    def plot_risk_map(self, 
                     risk_array: np.ndarray,
                     risk_categories: np.ndarray,
                     transform: rasterio.Affine,
                     title: str = "Bushfire Risk Assessment",
                     output_path: Optional[Union[str, Path]] = None,
                     figsize: Tuple[int, int] = (12, 8)) -> plt.Figure:
        """
        Create static risk map using matplotlib.
        
        Args:
            risk_array: Risk score array (0-100)
            risk_categories: Risk category array (1-5)
            transform: Rasterio transform for georeferencing
            title: Plot title
            output_path: Optional path to save figure
            figsize: Figure size in inches
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Plot continuous risk scores
        im1 = ax1.imshow(risk_array, cmap='YlOrRd', vmin=0, vmax=100)
        ax1.set_title('Risk Scores (0-100)')
        ax1.set_xlabel('Easting')
        ax1.set_ylabel('Northing')
        
        # Add colorbar for risk scores
        cbar1 = plt.colorbar(im1, ax=ax1, shrink=0.8)
        cbar1.set_label('Risk Score')
        
        # Plot categorical risk levels
        # Create custom colormap for categories
        category_colors = [self.risk_colors[i] for i in range(1, 6)]
        cmap = ListedColormap(category_colors)
        
        im2 = ax2.imshow(risk_categories, cmap=cmap, vmin=1, vmax=5)
        ax2.set_title('Risk Categories')
        ax2.set_xlabel('Easting')
        ax2.set_ylabel('Northing')
        
        # Add colorbar for categories
        cbar2 = plt.colorbar(im2, ax=ax2, shrink=0.8, ticks=range(1, 6))
        cbar2.set_label('Risk Category')
        cbar2.set_ticklabels([self.risk_labels[i] for i in range(1, 6)])
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved risk map to {output_path}")
        
        return fig
    
    def plot_vegetation_indices(self,
                               indices: Dict[str, np.ndarray],
                               title: str = "Vegetation Indices",
                               output_path: Optional[Union[str, Path]] = None,
                               figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
        """
        Plot vegetation indices side by side.
        
        Args:
            indices: Dictionary of vegetation index arrays
            title: Plot title
            output_path: Optional path to save figure
            figsize: Figure size in inches
            
        Returns:
            Matplotlib figure object
        """
        n_indices = len(indices)
        fig, axes = plt.subplots(1, n_indices, figsize=figsize)
        
        if n_indices == 1:
            axes = [axes]
        
        index_configs = {
            'NDVI': {'cmap': 'RdYlGn', 'vmin': -1, 'vmax': 1},
            'NDMI': {'cmap': 'Blues', 'vmin': -1, 'vmax': 1},
            'NBR': {'cmap': 'RdBu', 'vmin': -1, 'vmax': 1}
        }
        
        for i, (index_name, index_data) in enumerate(indices.items()):
            config = index_configs.get(index_name, {'cmap': 'viridis', 'vmin': -1, 'vmax': 1})
            
            im = axes[i].imshow(index_data, cmap=config['cmap'], 
                               vmin=config['vmin'], vmax=config['vmax'])
            axes[i].set_title(f'{index_name}\n(Range: {index_data.min():.3f} to {index_data.max():.3f})')
            axes[i].set_xlabel('Easting')
            axes[i].set_ylabel('Northing')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=axes[i], shrink=0.8)
            cbar.set_label(index_name)
        
        plt.suptitle(title, fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved vegetation indices plot to {output_path}")
        
        return fig
    
    def create_folium_map(self,
                         risk_array: np.ndarray,
                         risk_categories: np.ndarray,
                         bounds: Tuple[float, float, float, float],
                         center_lat: float = -33.75,
                         center_lon: float = 150.4,
                         zoom_start: int = 10) -> folium.Map:
        """
        Create interactive Folium map with risk data.
        
        Args:
            risk_array: Risk score array (0-100)
            risk_categories: Risk category array (1-5)
            bounds: Bounding box (min_lon, min_lat, max_lon, max_lat)
            center_lat: Map center latitude
            center_lon: Map center longitude
            zoom_start: Initial zoom level
            
        Returns:
            Folium map object
        """
        # Create base map
        m = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom_start,
            tiles='OpenStreetMap'
        )
        
        # Add additional tile layers
        # Note: Stamen tiles have been deprecated, using CartoDB alternatives
        folium.TileLayer('CartoDB positron', name='Light').add_to(m)
        folium.TileLayer('CartoDB dark_matter', name='Dark').add_to(m)
        folium.TileLayer('OpenStreetMap', name='OpenStreetMap').add_to(m)
        
        # Convert arrays to base64 encoded images for overlay
        # This is a simplified approach - in production you'd want to use proper raster overlays
        
        # Create risk score overlay
        risk_rgba = self._array_to_rgba(risk_array, cmap='YlOrRd', vmin=0, vmax=100)
        risk_image = self._array_to_base64_image(risk_rgba)
        
        folium.raster_layers.ImageOverlay(
            image=risk_image,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=0.7,
            name='Risk Scores'
        ).add_to(m)
        
        # Create category overlay
        category_rgba = self._categories_to_rgba(risk_categories)
        category_image = self._array_to_base64_image(category_rgba)
        
        folium.raster_layers.ImageOverlay(
            image=category_image,
            bounds=[[bounds[1], bounds[0]], [bounds[3], bounds[2]]],
            opacity=0.7,
            name='Risk Categories'
        ).add_to(m)
        
        # Add layer control
        folium.LayerControl().add_to(m)
        
        # Add risk legend
        self._add_risk_legend(m)
        
        # Add markers for high-risk areas
        self._add_high_risk_markers(m, risk_array, risk_categories, bounds)
        
        logger.info("Created interactive Folium map")
        return m
    
    def _array_to_rgba(self, array: np.ndarray, cmap: str, vmin: float, vmax: float) -> np.ndarray:
        """Convert array to RGBA values using colormap."""
        # Normalize array
        normalized = (array - vmin) / (vmax - vmin)
        normalized = np.clip(normalized, 0, 1)
        
        # Apply colormap
        cmap_obj = plt.cm.get_cmap(cmap)
        rgba = cmap_obj(normalized)
        
        # Handle NaN values
        rgba[np.isnan(array)] = [0, 0, 0, 0]  # Transparent
        
        return (rgba * 255).astype(np.uint8)
    
    def _categories_to_rgba(self, categories: np.ndarray) -> np.ndarray:
        """Convert category array to RGBA values."""
        rgba = np.zeros((*categories.shape, 4), dtype=np.uint8)
        
        for cat_id, color_hex in self.risk_colors.items():
            mask = categories == cat_id
            color_rgb = self._hex_to_rgb(color_hex)
            rgba[mask] = [*color_rgb, 255]  # Full opacity
        
        # Make zero values transparent
        rgba[categories == 0] = [0, 0, 0, 0]
        
        return rgba
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple."""
        hex_color = hex_color.lstrip('#')
        return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    
    def _array_to_base64_image(self, rgba_array: np.ndarray) -> str:
        """Convert RGBA array to base64 encoded PNG."""
        from PIL import Image
        
        # Convert to PIL Image
        img = Image.fromarray(rgba_array, 'RGBA')
        
        # Save to bytes
        buffer = BytesIO()
        img.save(buffer, format='PNG')
        buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.read()).decode()
        return f"data:image/png;base64,{img_base64}"
    
    def _add_risk_legend(self, map_obj: folium.Map) -> None:
        """Add risk category legend to map."""
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 150px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Risk Categories</b></p>
        '''
        
        for cat_id, label in self.risk_labels.items():
            color = self.risk_colors[cat_id]
            legend_html += f'<p><i style="background:{color}; width:15px; height:15px; display:inline-block;"></i> {label}</p>'
        
        legend_html += '</div>'
        
        map_obj.get_root().html.add_child(folium.Element(legend_html))
    
    def _add_high_risk_markers(self, map_obj: folium.Map, risk_array: np.ndarray, 
                              risk_categories: np.ndarray, bounds: Tuple[float, float, float, float]) -> None:
        """Add markers for high-risk areas."""
        # Find high-risk pixels (category 4 or 5)
        high_risk_mask = risk_categories >= 4
        
        if not np.any(high_risk_mask):
            return
        
        # Sample some high-risk locations (avoid too many markers)
        high_risk_coords = np.where(high_risk_mask)
        
        # Subsample to avoid cluttering
        step = max(1, len(high_risk_coords[0]) // 20)  # Max 20 markers
        sampled_indices = slice(None, None, step)
        
        for i, j in zip(high_risk_coords[0][sampled_indices], high_risk_coords[1][sampled_indices]):
            # Convert pixel coordinates to lat/lon
            lat = bounds[1] + (bounds[3] - bounds[1]) * (1 - i / risk_array.shape[0])
            lon = bounds[0] + (bounds[2] - bounds[0]) * (j / risk_array.shape[1])
            
            risk_score = risk_array[i, j]
            risk_cat = risk_categories[i, j]
            
            # Create popup text
            popup_text = f"""
            <b>High Risk Area</b><br>
            Risk Score: {risk_score:.1f}<br>
            Category: {self.risk_labels.get(risk_cat, 'Unknown')}<br>
            Location: {lat:.4f}, {lon:.4f}
            """
            
            # Add marker
            folium.Marker(
                location=[lat, lon],
                popup=folium.Popup(popup_text, max_width=200),
                icon=folium.Icon(color='red', icon='warning-sign')
            ).add_to(map_obj)
    
    def save_folium_map(self, map_obj: folium.Map, output_path: Union[str, Path]) -> None:
        """Save Folium map to HTML file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        map_obj.save(str(output_path))
        logger.info(f"Saved interactive map to {output_path}")
    
    def create_risk_summary_plot(self, stats: Dict[str, Union[int, float]], 
                               output_path: Optional[Union[str, Path]] = None) -> plt.Figure:
        """
        Create summary plot of risk statistics.
        
        Args:
            stats: Dictionary containing risk statistics
            output_path: Optional path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Risk category distribution (pie chart)
        categories = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
        counts = [
            stats.get('very_low_count', 0),
            stats.get('low_count', 0),
            stats.get('moderate_count', 0),
            stats.get('high_count', 0),
            stats.get('very_high_count', 0)
        ]
        
        colors_list = [self.risk_colors[i] for i in range(1, 6)]
        
        # Only include categories with data
        non_zero_data = [(cat, count, color) for cat, count, color in zip(categories, counts, colors_list) if count > 0]
        
        if non_zero_data:
            categories_filtered, counts_filtered, colors_filtered = zip(*non_zero_data)
            
            ax1.pie(counts_filtered, labels=categories_filtered, colors=colors_filtered, 
                   autopct='%1.1f%%', startangle=90)
            ax1.set_title('Risk Category Distribution')
        
        # Risk score histogram
        if 'avg_risk_score' in stats:
            ax2.axvline(stats['avg_risk_score'], color='red', linestyle='--', 
                       label=f"Mean: {stats['avg_risk_score']:.1f}")
            ax2.set_xlabel('Risk Score')
            ax2.set_ylabel('Frequency')
            ax2.set_title('Risk Score Distribution')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved summary plot to {output_path}")
        
        return fig


def quick_visualization(risk_array: np.ndarray, 
                       risk_categories: np.ndarray,
                       output_dir: Union[str, Path] = "./output") -> Dict[str, Path]:
    """
    Quick visualization function for rapid results.
    
    Args:
        risk_array: Risk score array
        risk_categories: Risk category array
        output_dir: Output directory for saving files
        
    Returns:
        Dictionary with paths to generated files
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    visualizer = BushfireRiskVisualizer()
    
    # Create dummy transform (for visualization purposes)
    transform = rasterio.Affine(0.001, 0, 150.0, 0, -0.001, -33.5)
    
    # Generate static risk map
    risk_map_path = output_dir / "risk_map.png"
    fig = visualizer.plot_risk_map(risk_array, risk_categories, transform, 
                                  output_path=risk_map_path)
    plt.close(fig)
    
    # Generate interactive map
    bounds = (150.0, -34.0, 150.8, -33.5)  # Blue Mountains bbox
    folium_map = visualizer.create_folium_map(risk_array, risk_categories, bounds)
    
    interactive_map_path = output_dir / "interactive_map.html"
    visualizer.save_folium_map(folium_map, interactive_map_path)
    
    logger.info("Quick visualization completed")
    
    return {
        'static_map': risk_map_path,
        'interactive_map': interactive_map_path
    } 