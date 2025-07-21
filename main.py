"""
NSW Bushfire Risk Assessment - Main Script
Using Sentinel-2 Satellite Imagery

This is the main entry point for the bushfire risk assessment system.
It demonstrates the complete workflow from data loading to visualization.
"""

import numpy as np
import logging
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Optional, Tuple
from datetime import datetime, date
import argparse
import sys

# Import our modules
from vegetation_indices import VegetationIndexCalculator, quick_calculate_ndvi
from risk_assessment import BushfireRiskAssessor, assess_risk_from_indices, simple_risk_assessment
from visualization import BushfireRiskVisualizer, quick_visualization
from database import BushfireRiskDatabase, setup_database_from_env, test_database_connection

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('bushfire_assessment.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

class BushfireRiskAssessment:
    """
    Main class for bushfire risk assessment workflow.
    
    Coordinates the entire process from data loading to final outputs.
    """
    
    def __init__(self, output_dir: str = "./output"):
        """
        Initialize the assessment system.
        
        Args:
            output_dir: Directory for saving outputs
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.vegetation_calc = None
        self.risk_assessor = BushfireRiskAssessor()
        self.visualizer = BushfireRiskVisualizer()
        
        # Blue Mountains study area (from PRD)
        self.study_area = {
            'min_lon': 150.0,
            'max_lon': 150.8,
            'min_lat': -34.0,
            'max_lat': -33.5
        }
        
        logger.info(f"Initialized bushfire risk assessment system")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Study area: {self.study_area}")
    
    def load_sample_data(self) -> Dict[str, np.ndarray]:
        """
        Load or generate sample Sentinel-2 data for demonstration.
        
        In a real implementation, this would load actual Sentinel-2 imagery.
        For the MVP, we'll create realistic synthetic data.
        
        Returns:
            Dictionary of band data arrays
        """
        logger.info("Loading sample Sentinel-2 data...")
        
        # Create realistic synthetic Sentinel-2 data (200x200 pixels)
        # This simulates a small area within the Blue Mountains
        height, width = 200, 200
        
        # Create spatial patterns that mimic real vegetation
        x = np.linspace(0, 10, width)
        y = np.linspace(0, 10, height)
        X, Y = np.meshgrid(x, y)
        
        # Base terrain pattern
        terrain = np.sin(X * 0.5) * np.cos(Y * 0.3) + 0.3 * np.random.random((height, width))
        
        # Simulate different vegetation types and conditions
        # Higher values in some areas (dense forest), lower in others (sparse/dry vegetation)
        vegetation_density = 0.5 + 0.3 * np.sin(X * 0.8) * np.cos(Y * 0.6) + 0.1 * terrain
        moisture_content = 0.4 + 0.4 * np.cos(X * 0.4) * np.sin(Y * 0.7) + 0.1 * np.random.random((height, width))
        
        # Add some spatial correlation and clamp values
        vegetation_density = np.clip(vegetation_density, 0.1, 0.9)
        moisture_content = np.clip(moisture_content, 0.1, 0.8)
        
        # Simulate Sentinel-2 bands (scaled to typical reflectance values)
        bands = {}
        
        # B04 (Red) - lower values for vegetated areas
        bands['B04'] = (0.3 - 0.2 * vegetation_density + 0.1 * np.random.random((height, width))) * 10000
        
        # B08 (NIR) - higher values for vegetated areas
        bands['B08'] = (0.2 + 0.6 * vegetation_density + 0.1 * np.random.random((height, width))) * 10000
        
        # B11 (SWIR1) - influenced by moisture content
        bands['B11'] = (0.25 - 0.15 * moisture_content + 0.1 * np.random.random((height, width))) * 10000
        
        # B12 (SWIR2) - similar to SWIR1 but slightly different response
        bands['B12'] = (0.2 - 0.1 * moisture_content + 0.05 * vegetation_density + 0.1 * np.random.random((height, width))) * 10000
        
        # Ensure all values are positive
        for band_name in bands:
            bands[band_name] = np.clip(bands[band_name], 100, 8000)
        
        logger.info(f"Generated synthetic data for {len(bands)} bands")
        logger.info(f"Data shape: {bands['B04'].shape}")
        
        return bands
    
    def calculate_vegetation_indices(self, bands: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Calculate vegetation indices from band data.
        
        Args:
            bands: Dictionary of Sentinel-2 band data
            
        Returns:
            Dictionary of calculated vegetation indices
        """
        logger.info("Calculating vegetation indices...")
        
        # Initialize calculator
        self.vegetation_calc = VegetationIndexCalculator("")
        
        # Load bands into calculator
        for band_name, band_data in bands.items():
            self.vegetation_calc.bands[band_name] = band_data.astype(np.float32)
        
        # Set dummy metadata
        self.vegetation_calc.metadata = {
            'transform': None,
            'crs': 'EPSG:4326',
            'width': bands['B04'].shape[1],
            'height': bands['B04'].shape[0],
            'nodata': None
        }
        
        # Calculate all available indices
        indices = self.vegetation_calc.calculate_all_indices()
        
        logger.info(f"Calculated {len(indices)} vegetation indices")
        return indices
    
    def assess_risk(self, indices: Dict[str, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Assess bushfire risk from vegetation indices.
        
        Args:
            indices: Dictionary of vegetation indices
            
        Returns:
            Tuple of (risk_scores, risk_categories)
        """
        logger.info("Assessing bushfire risk...")
        
        risk_scores = self.risk_assessor.calculate_risk_score(indices)
        risk_categories = self.risk_assessor.classify_risk(risk_scores)
        
        # Log summary statistics
        area_stats = self.risk_assessor.calculate_area_statistics(risk_categories)
        logger.info("Risk assessment summary:")
        for category, stats in area_stats.items():
            logger.info(f"  {category}: {stats['area_hectares']:.1f} ha ({stats['percentage']:.1f}%)")
        
        return risk_scores, risk_categories
    
    def create_visualizations(self, 
                            indices: Dict[str, np.ndarray],
                            risk_scores: np.ndarray, 
                            risk_categories: np.ndarray) -> Dict[str, Path]:
        """
        Create all visualizations.
        
        Args:
            indices: Vegetation indices
            risk_scores: Risk score array
            risk_categories: Risk category array
            
        Returns:
            Dictionary of output file paths
        """
        logger.info("Creating visualizations...")
        
        output_paths = {}
        
        # Create vegetation indices plot
        if indices:
            indices_fig = self.visualizer.plot_vegetation_indices(
                indices, 
                output_path=self.output_dir / "vegetation_indices.png"
            )
            output_paths['vegetation_indices'] = self.output_dir / "vegetation_indices.png"
            plt.close(indices_fig)
        
        # Create risk map
        # Create dummy transform for visualization
        import rasterio
        transform = rasterio.Affine(0.004, 0, 150.0, 0, -0.004, -33.5)  # ~400m pixels
        
        risk_fig = self.visualizer.plot_risk_map(
            risk_scores, 
            risk_categories, 
            transform,
            title="NSW Bushfire Risk Assessment - Blue Mountains",
            output_path=self.output_dir / "risk_assessment_map.png"
        )
        output_paths['risk_map'] = self.output_dir / "risk_assessment_map.png"
        plt.close(risk_fig)
        
        # Create interactive map
        bounds = (self.study_area['min_lon'], self.study_area['min_lat'], 
                 self.study_area['max_lon'], self.study_area['max_lat'])
        
        folium_map = self.visualizer.create_folium_map(
            risk_scores, 
            risk_categories, 
            bounds,
            center_lat=-33.75,
            center_lon=150.4
        )
        
        interactive_path = self.output_dir / "interactive_risk_map.html"
        self.visualizer.save_folium_map(folium_map, interactive_path)
        output_paths['interactive_map'] = interactive_path
        
        logger.info(f"Created {len(output_paths)} visualizations")
        return output_paths
    
    def save_to_database(self, 
                        risk_scores: np.ndarray,
                        risk_categories: np.ndarray, 
                        indices: Dict[str, np.ndarray]) -> Optional[int]:
        """
        Save results to PostGIS database.
        
        Args:
            risk_scores: Risk score array
            risk_categories: Risk category array
            indices: Vegetation indices
            
        Returns:
            Database record ID if successful, None if failed
        """
        logger.info("Attempting to save results to database...")
        
        try:
            # Test database connection first
            if not test_database_connection():
                logger.warning("Database connection failed - skipping database save")
                return None
            
            with setup_database_from_env() as db:
                db.create_schema()
                
                # Create polygon for study area
                from shapely.geometry import box
                study_polygon = box(
                    self.study_area['min_lon'], 
                    self.study_area['min_lat'],
                    self.study_area['max_lon'], 
                    self.study_area['max_lat']
                )
                
                # Calculate mean values
                mean_indices = {
                    'NDVI': float(np.nanmean(indices.get('NDVI', []))),
                    'NDMI': float(np.nanmean(indices.get('NDMI', []))),
                    'NBR': float(np.nanmean(indices.get('NBR', [])))
                }
                
                mean_risk_score = float(np.nanmean(risk_scores))
                
                # Determine overall risk category based on mean score
                if mean_risk_score < 20:
                    risk_category = "Very Low"
                elif mean_risk_score < 40:
                    risk_category = "Low"
                elif mean_risk_score < 60:
                    risk_category = "Moderate"
                elif mean_risk_score < 80:
                    risk_category = "High"
                else:
                    risk_category = "Very High"
                
                # Calculate area (approximate)
                area_hectares = ((self.study_area['max_lon'] - self.study_area['min_lon']) * 
                               (self.study_area['max_lat'] - self.study_area['min_lat']) * 
                               111000 * 111000 / 10000)  # Rough conversion to hectares
                
                # Insert record
                record_id = db.insert_risk_assessment(
                    assessment_date=date.today(),
                    geometry=study_polygon,
                    risk_score=mean_risk_score,
                    risk_category=risk_category,
                    indices=mean_indices,
                    area_hectares=area_hectares,
                    metadata={
                        'method': 'synthetic_demo',
                        'data_source': 'simulated_sentinel2',
                        'processing_date': datetime.now().isoformat()
                    }
                )
                
                logger.info(f"Saved assessment to database with ID: {record_id}")
                return record_id
                
        except Exception as e:
            logger.error(f"Failed to save to database: {e}")
            return None
    
    def run_full_assessment(self) -> Dict[str, any]:
        """
        Run the complete bushfire risk assessment workflow.
        
        Returns:
            Dictionary containing all results and output paths
        """
        logger.info("=" * 60)
        logger.info("STARTING BUSHFIRE RISK ASSESSMENT")
        logger.info("=" * 60)
        
        start_time = datetime.now()
        
        try:
            # Step 1: Load data
            bands = self.load_sample_data()
            
            # Step 2: Calculate vegetation indices
            indices = self.calculate_vegetation_indices(bands)
            
            # Step 3: Assess risk
            risk_scores, risk_categories = self.assess_risk(indices)
            
            # Step 4: Create visualizations
            visualization_paths = self.create_visualizations(indices, risk_scores, risk_categories)
            
            # Step 5: Save to database (optional)
            db_record_id = self.save_to_database(risk_scores, risk_categories, indices)
            
            # Calculate processing time
            processing_time = datetime.now() - start_time
            
            # Compile results
            results = {
                'success': True,
                'processing_time': processing_time.total_seconds(),
                'indices': indices,
                'risk_scores': risk_scores,
                'risk_categories': risk_categories,
                'visualization_paths': visualization_paths,
                'database_record_id': db_record_id,
                'summary': {
                    'mean_risk_score': float(np.nanmean(risk_scores)),
                    'high_risk_percentage': float(np.sum(risk_categories >= 4) / risk_categories.size * 100),
                    'total_pixels': int(risk_categories.size)
                }
            }
            
            logger.info("=" * 60)
            logger.info("ASSESSMENT COMPLETED SUCCESSFULLY")
            logger.info("=" * 60)
            logger.info(f"Processing time: {processing_time.total_seconds():.2f} seconds")
            logger.info(f"Mean risk score: {results['summary']['mean_risk_score']:.1f}")
            logger.info(f"High risk areas: {results['summary']['high_risk_percentage']:.1f}%")
            logger.info(f"Outputs saved to: {self.output_dir}")
            
            return results
            
        except Exception as e:
            logger.error(f"Assessment failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(description='NSW Bushfire Risk Assessment using Sentinel-2 Imagery')
    parser.add_argument('--output-dir', '-o', default='./output', 
                       help='Output directory for results (default: ./output)')
    parser.add_argument('--quick', '-q', action='store_true',
                       help='Run quick assessment with minimal outputs')
    parser.add_argument('--no-database', action='store_true',
                       help='Skip database operations')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--parallel', '-p', action='store_true',
                       help='Use parallel processing for large-scale analysis')
    parser.add_argument('--workers', '-w', type=int, default=None,
                       help='Number of parallel workers (default: auto-detect)')
    parser.add_argument('--benchmark', '-b', action='store_true',
                       help='Run performance benchmark')
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    logger.info("NSW Bushfire Risk Assessment System")
    logger.info(f"Version: 1.0.0 (MVP)")
    logger.info(f"Study Area: Blue Mountains, NSW")
    
    # Handle benchmark mode
    if args.benchmark:
        logger.info("Running performance benchmark...")
        from parallel_processor import ParallelProcessor, demonstrate_parallel_processing
        
        results = demonstrate_parallel_processing()
        logger.info("Benchmark completed!")
        return 0
    
    # Handle parallel processing mode
    if args.parallel:
        logger.info("Running in parallel processing mode...")
        from parallel_processor import ParallelProcessor
        
        processor = ParallelProcessor(max_workers=args.workers)
        
        # Generate sample large-scale data
        logger.info("Processing large-scale data with parallel computing...")
        large_data = np.random.rand(4096, 4096, 4).astype(np.float32)
        risk_map = processor.process_large_raster_chunked(large_data, chunk_size=512)
        
        # Save results
        output_path = Path(args.output_dir) / "parallel_risk_map.npy"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, risk_map)
        
        logger.info(f"Parallel processing completed! Results saved to {output_path}")
        return 0
    
    if args.quick:
        # Quick assessment using simple functions
        logger.info("Running quick assessment...")
        
        # Generate minimal sample data
        height, width = 100, 100
        ndvi_sample = np.random.normal(0.5, 0.2, (height, width))
        ndvi_sample = np.clip(ndvi_sample, -1, 1)
        
        # Quick risk calculation
        risk_scores = simple_risk_assessment(ndvi_sample)
        risk_categories = np.zeros_like(risk_scores, dtype=int)
        risk_categories[risk_scores < 20] = 1
        risk_categories[(risk_scores >= 20) & (risk_scores < 40)] = 2
        risk_categories[(risk_scores >= 40) & (risk_scores < 60)] = 3
        risk_categories[(risk_scores >= 60) & (risk_scores < 80)] = 4
        risk_categories[risk_scores >= 80] = 5
        
        # Quick visualization
        output_paths = quick_visualization(risk_scores, risk_categories, args.output_dir)
        
        logger.info("Quick assessment completed!")
        logger.info(f"Static map: {output_paths['static_map']}")
        logger.info(f"Interactive map: {output_paths['interactive_map']}")
        
    else:
        # Full assessment
        assessment = BushfireRiskAssessment(args.output_dir)
        results = assessment.run_full_assessment()
        
        if results['success']:
            print("\n" + "=" * 60)
            print("ASSESSMENT RESULTS SUMMARY")
            print("=" * 60)
            print(f"Mean Risk Score: {results['summary']['mean_risk_score']:.1f}/100")
            print(f"High Risk Areas: {results['summary']['high_risk_percentage']:.1f}%")
            print(f"Processing Time: {results['processing_time']:.2f} seconds")
            print(f"Database Record: {results['database_record_id'] or 'Not saved'}")
            print(f"\nOutput Files:")
            for name, path in results['visualization_paths'].items():
                print(f"  {name}: {path}")
        else:
            print(f"Assessment failed: {results['error']}")
            return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 