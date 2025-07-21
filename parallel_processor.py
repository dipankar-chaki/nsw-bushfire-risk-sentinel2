#!/usr/bin/env python3
"""
Parallel Processing Module for Bushfire Risk Assessment
Demonstrates HPC concepts on multi-core systems (optimized for Apple Silicon)
"""

import os
import time
import logging
import numpy as np
from multiprocessing import Pool, cpu_count, shared_memory
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import partial
import psutil
from typing import List, Tuple, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ParallelProcessor:
    """High-performance parallel processing for satellite imagery analysis"""
    
    def __init__(self, max_workers: int = None):
        """
        Initialize parallel processor
        
        Args:
            max_workers: Maximum number of worker processes (defaults to CPU count - 1)
        """
        self.max_workers = max_workers or max(1, cpu_count() - 1)
        self.performance_stats = {}
        
        # Log system information
        logger.info(f"Initialized ParallelProcessor on {os.uname().sysname}")
        logger.info(f"CPU cores available: {cpu_count()}")
        logger.info(f"Using {self.max_workers} worker processes")
        logger.info(f"Total RAM: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        
    def benchmark_system(self) -> Dict[str, Any]:
        """Benchmark system capabilities"""
        logger.info("Running system benchmark...")
        
        # Test different array sizes
        test_sizes = [100, 500, 1000, 2000]  # Array dimensions
        results = {}
        
        for size in test_sizes:
            # Create test data (simulating satellite imagery bands)
            data = np.random.rand(size, size, 4).astype(np.float32)
            data_size_mb = data.nbytes / (1024**2)
            
            # Benchmark serial processing
            start_time = time.time()
            _ = self._process_indices_serial(data)
            serial_time = time.time() - start_time
            
            # Benchmark parallel processing
            start_time = time.time()
            _ = self._process_indices_parallel(data, self.max_workers)
            parallel_time = time.time() - start_time
            
            speedup = serial_time / parallel_time
            
            results[f"{size}x{size}"] = {
                "data_size_mb": data_size_mb,
                "serial_time": serial_time,
                "parallel_time": parallel_time,
                "speedup": speedup,
                "efficiency": speedup / self.max_workers * 100
            }
            
            logger.info(f"Size {size}x{size}: {speedup:.2f}x speedup, "
                       f"{results[f'{size}x{size}']['efficiency']:.1f}% efficiency")
        
        self.performance_stats = results
        return results
    
    def process_tiles_parallel(self, tile_paths: List[str], 
                             indices_to_calculate: List[str] = ['NDVI', 'NDMI', 'NBR']) -> Dict[str, np.ndarray]:
        """
        Process multiple tiles in parallel
        
        Args:
            tile_paths: List of paths to tile files
            indices_to_calculate: List of vegetation indices to calculate
            
        Returns:
            Dictionary of processed results
        """
        logger.info(f"Processing {len(tile_paths)} tiles in parallel...")
        start_time = time.time()
        
        # Create partial function with fixed indices
        process_func = partial(self._process_single_tile, indices=indices_to_calculate)
        
        # Process tiles in parallel
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_func, tile_paths))
        
        elapsed_time = time.time() - start_time
        tiles_per_second = len(tile_paths) / elapsed_time
        
        logger.info(f"Processed {len(tile_paths)} tiles in {elapsed_time:.2f}s "
                   f"({tiles_per_second:.2f} tiles/second)")
        
        return results
    
    def process_large_raster_chunked(self, data: np.ndarray, chunk_size: int = 1024) -> np.ndarray:
        """
        Process large raster data in chunks using parallel processing
        
        Args:
            data: Large numpy array (height, width, bands)
            chunk_size: Size of chunks to process
            
        Returns:
            Processed risk array
        """
        height, width, bands = data.shape
        logger.info(f"Processing raster of size {height}x{width} in {chunk_size}x{chunk_size} chunks")
        
        # Calculate number of chunks
        n_chunks_y = (height + chunk_size - 1) // chunk_size
        n_chunks_x = (width + chunk_size - 1) // chunk_size
        total_chunks = n_chunks_y * n_chunks_x
        
        logger.info(f"Total chunks to process: {total_chunks}")
        
        # Create output array
        output = np.zeros((height, width), dtype=np.float32)
        
        # Prepare chunk coordinates
        chunks = []
        for i in range(n_chunks_y):
            for j in range(n_chunks_x):
                y_start = i * chunk_size
                y_end = min((i + 1) * chunk_size, height)
                x_start = j * chunk_size
                x_end = min((j + 1) * chunk_size, width)
                chunks.append((y_start, y_end, x_start, x_end))
        
        # Process chunks in parallel
        start_time = time.time()
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all chunk processing tasks
            futures = []
            for chunk_coords in chunks:
                y_start, y_end, x_start, x_end = chunk_coords
                chunk_data = data[y_start:y_end, x_start:x_end, :]
                future = executor.submit(self._process_chunk, chunk_data)
                futures.append((future, chunk_coords))
            
            # Collect results
            for future, chunk_coords in futures:
                y_start, y_end, x_start, x_end = chunk_coords
                chunk_result = future.result()
                output[y_start:y_end, x_start:x_end] = chunk_result
        
        elapsed_time = time.time() - start_time
        pixels_per_second = (height * width) / elapsed_time / 1e6  # Megapixels/second
        
        logger.info(f"Processed {height}x{width} image in {elapsed_time:.2f}s "
                   f"({pixels_per_second:.2f} megapixels/second)")
        
        return output
    
    def parallel_time_series_analysis(self, dates: List[str], 
                                    study_area: Dict[str, float]) -> Dict[str, np.ndarray]:
        """
        Process time series of satellite images in parallel
        
        Args:
            dates: List of dates to process
            study_area: Bounding box dictionary
            
        Returns:
            Time series results
        """
        logger.info(f"Processing time series for {len(dates)} dates")
        
        # Simulate processing multiple dates
        process_func = partial(self._process_date, study_area=study_area)
        
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(process_func, dates))
        
        # Combine results
        time_series_results = {
            'dates': dates,
            'risk_scores': [r['risk_score'] for r in results],
            'statistics': [r['stats'] for r in results]
        }
        
        return time_series_results
    
    # Private helper methods
    def _process_indices_serial(self, data: np.ndarray) -> Dict[str, np.ndarray]:
        """Calculate vegetation indices serially (for benchmarking)"""
        b4 = data[:, :, 0]  # Red
        b8 = data[:, :, 1]  # NIR
        b11 = data[:, :, 2]  # SWIR1
        b12 = data[:, :, 3]  # SWIR2
        
        # Calculate indices
        ndvi = (b8 - b4) / (b8 + b4 + 1e-10)
        ndmi = (b8 - b11) / (b8 + b11 + 1e-10)
        nbr = (b8 - b12) / (b8 + b12 + 1e-10)
        
        return {'NDVI': ndvi, 'NDMI': ndmi, 'NBR': nbr}
    
    def _process_indices_parallel(self, data: np.ndarray, n_workers: int) -> Dict[str, np.ndarray]:
        """Calculate vegetation indices in parallel"""
        height, width = data.shape[:2]
        
        # Split data into chunks for each worker
        chunk_size = height // n_workers
        chunks = []
        
        for i in range(n_workers):
            start_idx = i * chunk_size
            end_idx = start_idx + chunk_size if i < n_workers - 1 else height
            chunks.append(data[start_idx:end_idx])
        
        # Process chunks in parallel
        with Pool(n_workers) as pool:
            chunk_results = pool.map(self._process_indices_serial, chunks)
        
        # Combine results
        combined_results = {}
        for index_name in ['NDVI', 'NDMI', 'NBR']:
            combined_results[index_name] = np.vstack([
                result[index_name] for result in chunk_results
            ])
        
        return combined_results
    
    def _process_single_tile(self, tile_path: str, indices: List[str]) -> Dict[str, Any]:
        """Process a single tile (simulated for demo)"""
        # Simulate loading and processing a tile
        tile_size = 512
        data = np.random.rand(tile_size, tile_size, 4).astype(np.float32)
        
        # Calculate indices
        indices_result = self._process_indices_serial(data)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(indices_result)
        
        return {
            'tile_path': tile_path,
            'risk_score': risk_score,
            'indices': indices_result
        }
    
    def _process_chunk(self, chunk_data: np.ndarray) -> np.ndarray:
        """Process a single chunk of data"""
        # Calculate indices for chunk
        indices = self._process_indices_serial(chunk_data)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(indices)
        
        return risk_score
    
    def _calculate_risk_score(self, indices: Dict[str, np.ndarray]) -> np.ndarray:
        """Calculate bushfire risk score from vegetation indices"""
        # Simplified risk calculation
        ndvi_risk = (1 - indices['NDVI']) * 100 * 0.3
        ndmi_risk = (1 - indices['NDMI']) * 100 * 0.5
        nbr_risk = (1 - indices['NBR']) * 100 * 0.2
        
        risk_score = ndvi_risk + ndmi_risk + nbr_risk
        return np.clip(risk_score, 0, 100)
    
    def _process_date(self, date: str, study_area: Dict[str, float]) -> Dict[str, Any]:
        """Process data for a single date (simulated)"""
        # Simulate processing
        data = np.random.rand(200, 200, 4).astype(np.float32)
        indices = self._process_indices_serial(data)
        risk_score = self._calculate_risk_score(indices)
        
        return {
            'date': date,
            'risk_score': risk_score,
            'stats': {
                'mean_risk': np.mean(risk_score),
                'max_risk': np.max(risk_score),
                'high_risk_area': np.sum(risk_score > 60) / risk_score.size * 100
            }
        }


def demonstrate_parallel_processing():
    """Demonstrate parallel processing capabilities"""
    logger.info("=== Parallel Processing Demonstration ===")
    
    # Initialize processor
    processor = ParallelProcessor()
    
    # 1. Run system benchmark
    logger.info("\n1. System Benchmark:")
    benchmark_results = processor.benchmark_system()
    
    # 2. Simulate processing multiple tiles
    logger.info("\n2. Multi-tile Processing:")
    tile_paths = [f"tile_{i:03d}.tif" for i in range(16)]
    tile_results = processor.process_tiles_parallel(tile_paths)
    
    # 3. Process large raster with chunking
    logger.info("\n3. Large Raster Processing:")
    large_raster = np.random.rand(4096, 4096, 4).astype(np.float32)
    risk_map = processor.process_large_raster_chunked(large_raster, chunk_size=512)
    
    # 4. Time series analysis
    logger.info("\n4. Time Series Analysis:")
    dates = [f"2024-{month:02d}-01" for month in range(1, 13)]
    time_series = processor.parallel_time_series_analysis(dates, 
                                                         {'min_lon': 150.0, 'max_lon': 150.8,
                                                          'min_lat': -34.0, 'max_lat': -33.5})
    
    logger.info("\n=== Demonstration Complete ===")
    
    return {
        'benchmark': benchmark_results,
        'tile_processing': f"Processed {len(tile_paths)} tiles",
        'large_raster': f"Processed {large_raster.shape[0]}x{large_raster.shape[1]} image",
        'time_series': f"Analyzed {len(dates)} dates"
    }


if __name__ == "__main__":
    # Run demonstration
    results = demonstrate_parallel_processing()
    
    # Print summary
    print("\nParallel Processing Summary:")
    print("-" * 50)
    for key, value in results.items():
        if key == 'benchmark':
            print(f"\nBenchmark Results:")
            for size, stats in value.items():
                print(f"  {size}: {stats['speedup']:.2f}x speedup")
        else:
            print(f"{key}: {value}")