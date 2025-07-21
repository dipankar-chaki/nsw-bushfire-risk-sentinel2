#!/bin/bash

# ============================================================================
# Parallel Bushfire Risk Analysis Pipeline
# For HPC environments and multi-core systems (including Apple Silicon)
# ============================================================================

set -e  # Exit on error
set -u  # Exit on undefined variable

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
OUTPUT_DIR="${PROJECT_DIR}/output/parallel_$(date +%Y%m%d_%H%M%S)"
LOG_DIR="${OUTPUT_DIR}/logs"
NUM_WORKERS=${NUM_WORKERS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
CHUNK_SIZE=${CHUNK_SIZE:-1024}

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $(date '+%Y-%m-%d %H:%M:%S') - $1"
}

check_requirements() {
    log_info "Checking system requirements..."
    
    # Check Python
    if ! command -v python3 &> /dev/null; then
        log_error "Python 3 is not installed"
        exit 1
    fi
    
    # Check available memory
    if [[ "$OSTYPE" == "darwin"* ]]; then
        TOTAL_MEM=$(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}')
    else
        TOTAL_MEM=$(free -g | awk '/^Mem:/{print $2}')
    fi
    
    log_info "System has ${TOTAL_MEM} GB of memory"
    
    # Check CPU cores
    log_info "System has ${NUM_WORKERS} CPU cores available"
}

setup_directories() {
    log_info "Setting up output directories..."
    mkdir -p "${OUTPUT_DIR}"
    mkdir -p "${LOG_DIR}"
    mkdir -p "${OUTPUT_DIR}/tiles"
    mkdir -p "${OUTPUT_DIR}/reports"
}

run_benchmark() {
    log_info "Running performance benchmark..."
    
    python3 - <<EOF > "${LOG_DIR}/benchmark.log" 2>&1
import sys
sys.path.append('${PROJECT_DIR}')
from parallel_processor import ParallelProcessor

processor = ParallelProcessor(max_workers=${NUM_WORKERS})
results = processor.benchmark_system()

print("\nBenchmark Results:")
print("-" * 50)
for size, stats in results.items():
    print(f"{size}: {stats['speedup']:.2f}x speedup, {stats['efficiency']:.1f}% efficiency")
EOF
    
    if [ $? -eq 0 ]; then
        log_info "Benchmark completed successfully"
        cat "${LOG_DIR}/benchmark.log"
    else
        log_error "Benchmark failed"
        cat "${LOG_DIR}/benchmark.log"
        exit 1
    fi
}

process_tiles_parallel() {
    log_info "Processing tiles in parallel..."
    
    # Generate list of tiles to process
    TILE_LIST="${OUTPUT_DIR}/tile_list.txt"
    
    # Create sample tile list (in production, this would be actual tile paths)
    for i in {001..016}; do
        echo "tile_${i}.tif" >> "${TILE_LIST}"
    done
    
    # Process tiles using GNU parallel (if available) or Python multiprocessing
    if command -v parallel &> /dev/null; then
        log_info "Using GNU parallel for tile processing"
        
        parallel -j ${NUM_WORKERS} --progress \
            python3 "${PROJECT_DIR}/process_single_tile.py" {} "${OUTPUT_DIR}/tiles" \
            :::: "${TILE_LIST}" \
            2>&1 | tee "${LOG_DIR}/parallel_processing.log"
    else
        log_info "Using Python multiprocessing for tile processing"
        
        python3 - <<EOF 2>&1 | tee "${LOG_DIR}/parallel_processing.log"
import sys
sys.path.append('${PROJECT_DIR}')
from parallel_processor import ParallelProcessor

processor = ParallelProcessor(max_workers=${NUM_WORKERS})
tile_paths = open('${TILE_LIST}').read().strip().split('\n')
results = processor.process_tiles_parallel(tile_paths)
print(f"Processed {len(results)} tiles successfully")
EOF
    fi
}

process_time_series() {
    log_info "Processing time series data..."
    
    python3 - <<EOF > "${LOG_DIR}/time_series.log" 2>&1
import sys
sys.path.append('${PROJECT_DIR}')
from parallel_processor import ParallelProcessor
import json

processor = ParallelProcessor(max_workers=${NUM_WORKERS})
dates = [f"2024-{month:02d}-01" for month in range(1, 13)]
study_area = {'min_lon': 150.0, 'max_lon': 150.8, 'min_lat': -34.0, 'max_lat': -33.5}

results = processor.parallel_time_series_analysis(dates, study_area)

# Save results
with open('${OUTPUT_DIR}/time_series_results.json', 'w') as f:
    json.dump({
        'dates': results['dates'],
        'mean_risk_scores': [float(s['mean_risk']) for s in results['statistics']],
        'max_risk_scores': [float(s['max_risk']) for s in results['statistics']],
        'high_risk_percentages': [float(s['high_risk_area']) for s in results['statistics']]
    }, f, indent=2)

print(f"Time series analysis completed for {len(dates)} dates")
EOF
    
    if [ $? -eq 0 ]; then
        log_info "Time series processing completed"
    else
        log_error "Time series processing failed"
        cat "${LOG_DIR}/time_series.log"
    fi
}

generate_performance_report() {
    log_info "Generating performance report..."
    
    python3 - <<EOF > "${OUTPUT_DIR}/reports/performance_report.txt"
import os
import glob
import datetime

print("=" * 60)
print("PARALLEL PROCESSING PERFORMANCE REPORT")
print("=" * 60)
print(f"Generated: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"System: $(uname -s) $(uname -m)")
print(f"CPU Cores Used: ${NUM_WORKERS}")
print(f"Chunk Size: ${CHUNK_SIZE}")
print()

# Parse benchmark results
if os.path.exists('${LOG_DIR}/benchmark.log'):
    print("BENCHMARK RESULTS:")
    print("-" * 40)
    with open('${LOG_DIR}/benchmark.log', 'r') as f:
        for line in f:
            if 'speedup' in line:
                print(f"  {line.strip()}")
    print()

# Check output files
output_files = glob.glob('${OUTPUT_DIR}/**/*', recursive=True)
print(f"Total output files generated: {len(output_files)}")
print()

print("Processing completed successfully!")
print("=" * 60)
EOF
    
    cat "${OUTPUT_DIR}/reports/performance_report.txt"
}

# Main execution
main() {
    log_info "Starting parallel bushfire risk analysis pipeline"
    log_info "Output directory: ${OUTPUT_DIR}"
    
    check_requirements
    setup_directories
    
    # Run analysis steps
    if [[ "${1:-}" == "--benchmark-only" ]]; then
        run_benchmark
    else
        run_benchmark
        process_tiles_parallel
        process_time_series
        generate_performance_report
    fi
    
    log_info "Pipeline completed successfully!"
    log_info "Results saved to: ${OUTPUT_DIR}"
}

# Handle script arguments
case "${1:-}" in
    --help|-h)
        echo "Usage: $0 [OPTIONS]"
        echo ""
        echo "Options:"
        echo "  --benchmark-only    Run only the performance benchmark"
        echo "  --workers N        Set number of parallel workers (default: auto-detect)"
        echo "  --chunk-size N     Set chunk size for processing (default: 1024)"
        echo "  --help             Show this help message"
        exit 0
        ;;
    --workers)
        NUM_WORKERS=$2
        shift 2
        ;;
    --chunk-size)
        CHUNK_SIZE=$2
        shift 2
        ;;
esac

# Run main function
main "$@"