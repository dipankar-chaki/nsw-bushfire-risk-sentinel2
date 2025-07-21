#!/bin/bash

# ============================================================================
# Batch Processing Script for Large-Scale Tile Analysis
# Demonstrates HPC-style batch job processing on local systems
# ============================================================================

set -e
set -o pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
BATCH_SIZE=${BATCH_SIZE:-4}
MAX_JOBS=${MAX_JOBS:-$(sysctl -n hw.ncpu 2>/dev/null || nproc)}
TILE_DIR="${TILE_DIR:-${PROJECT_DIR}/data/tiles}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_DIR}/output/batch_$(date +%Y%m%d_%H%M%S)}"
LOG_FILE="${OUTPUT_DIR}/batch_processing.log"

# Create output directories
mkdir -p "${OUTPUT_DIR}/processed"
mkdir -p "${OUTPUT_DIR}/logs"
mkdir -p "${OUTPUT_DIR}/reports"

# Logging function
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

# Process a single batch of tiles
process_batch() {
    local batch_id=$1
    local batch_file=$2
    local batch_log="${OUTPUT_DIR}/logs/batch_${batch_id}.log"
    
    log "Processing batch ${batch_id}..."
    
    # Read tile list from batch file
    while IFS= read -r tile; do
        echo "Processing tile: ${tile}"
        
        # Simulate tile processing (replace with actual processing)
        python3 - <<EOF 2>&1
import sys
sys.path.append('${PROJECT_DIR}')
import numpy as np
import time
from vegetation_indices import VegetationIndexCalculator
from risk_assessment import BushfireRiskAssessor

# Simulate processing
print(f"Loading tile: ${tile}")
time.sleep(0.1)  # Simulate I/O

# Generate sample data
data = {
    'B04': np.random.rand(512, 512) * 10000,
    'B08': np.random.rand(512, 512) * 10000,
    'B11': np.random.rand(512, 512) * 10000,
    'B12': np.random.rand(512, 512) * 10000
}

# Calculate indices
calc = VegetationIndexCalculator()
indices = {
    'NDVI': calc.calculate_index(data['B08'], data['B04'], 'difference'),
    'NDMI': calc.calculate_index(data['B08'], data['B11'], 'difference'),
    'NBR': calc.calculate_index(data['B08'], data['B12'], 'difference')
}

# Assess risk
assessor = BushfireRiskAssessor()
risk_score = assessor.calculate_risk_score(indices)

# Save results (simulated)
print(f"Tile ${tile} processed: mean risk = {np.mean(risk_score):.2f}")
EOF
    done < "$batch_file" >> "$batch_log" 2>&1
    
    if [ $? -eq 0 ]; then
        log "Batch ${batch_id} completed successfully"
        return 0
    else
        log "ERROR: Batch ${batch_id} failed"
        return 1
    fi
}

# Monitor job progress
monitor_jobs() {
    local total_jobs=$1
    local completed=0
    
    while [ $completed -lt $total_jobs ]; do
        completed=$(find "${OUTPUT_DIR}/logs" -name "batch_*.log" -exec grep -l "completed successfully" {} \; | wc -l)
        local progress=$((completed * 100 / total_jobs))
        
        printf "\rProgress: [%-50s] %d%% (%d/%d batches)" \
            "$(printf '#%.0s' $(seq 1 $((progress / 2))))" \
            "$progress" "$completed" "$total_jobs"
        
        sleep 2
    done
    echo
}

# Generate performance statistics
generate_stats() {
    log "Generating batch processing statistics..."
    
    python3 - <<EOF > "${OUTPUT_DIR}/reports/batch_stats.txt"
import glob
import re
from datetime import datetime

log_files = glob.glob('${OUTPUT_DIR}/logs/batch_*.log')
total_tiles = 0
total_time = 0
risk_scores = []

print("=" * 60)
print("BATCH PROCESSING STATISTICS")
print("=" * 60)
print(f"Total batches processed: {len(log_files)}")
print(f"Tiles per batch: ${BATCH_SIZE}")
print(f"Maximum parallel jobs: ${MAX_JOBS}")
print()

# Parse log files
for log_file in log_files:
    with open(log_file, 'r') as f:
        content = f.read()
        # Count processed tiles
        tiles = len(re.findall(r'Processing tile:', content))
        total_tiles += tiles
        
        # Extract risk scores
        scores = re.findall(r'mean risk = ([\d.]+)', content)
        risk_scores.extend([float(s) for s in scores])

if risk_scores:
    print(f"Total tiles processed: {total_tiles}")
    print(f"Average risk score: {sum(risk_scores)/len(risk_scores):.2f}")
    print(f"Min risk score: {min(risk_scores):.2f}")
    print(f"Max risk score: {max(risk_scores):.2f}")

print("=" * 60)
EOF
    
    cat "${OUTPUT_DIR}/reports/batch_stats.txt"
}

# Main execution
main() {
    log "Starting batch tile processing"
    log "Configuration:"
    log "  - Tile directory: ${TILE_DIR}"
    log "  - Output directory: ${OUTPUT_DIR}"
    log "  - Batch size: ${BATCH_SIZE}"
    log "  - Max parallel jobs: ${MAX_JOBS}"
    
    # Generate sample tile list (in production, this would scan actual tile directory)
    TILE_LIST="${OUTPUT_DIR}/all_tiles.txt"
    log "Generating tile list..."
    
    for i in {001..100}; do
        echo "tile_${i}.tif" >> "$TILE_LIST"
    done
    
    TOTAL_TILES=$(wc -l < "$TILE_LIST")
    log "Found ${TOTAL_TILES} tiles to process"
    
    # Split tiles into batches
    log "Creating batch files..."
    split -l ${BATCH_SIZE} "$TILE_LIST" "${OUTPUT_DIR}/batch_"
    
    # Count number of batches
    BATCH_FILES=(${OUTPUT_DIR}/batch_*)
    NUM_BATCHES=${#BATCH_FILES[@]}
    log "Created ${NUM_BATCHES} batches"
    
    # Process batches in parallel
    log "Starting parallel batch processing..."
    
    # Launch batch jobs
    job_count=0
    for batch_file in "${BATCH_FILES[@]}"; do
        batch_id=$(basename "$batch_file")
        
        # Wait if we've reached max parallel jobs
        while [ $(jobs -r | wc -l) -ge ${MAX_JOBS} ]; do
            sleep 0.1
        done
        
        # Launch batch job in background
        process_batch "$batch_id" "$batch_file" &
        
        ((job_count++))
        if [ $((job_count % 10)) -eq 0 ]; then
            log "Launched ${job_count}/${NUM_BATCHES} batch jobs"
        fi
    done
    
    log "All batch jobs launched, waiting for completion..."
    
    # Monitor progress
    monitor_jobs $NUM_BATCHES
    
    # Wait for all jobs to complete
    wait
    
    # Generate statistics
    generate_stats
    
    log "Batch processing completed!"
    log "Results saved to: ${OUTPUT_DIR}"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --max-jobs)
            MAX_JOBS="$2"
            shift 2
            ;;
        --tile-dir)
            TILE_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --batch-size N     Number of tiles per batch (default: 4)"
            echo "  --max-jobs N       Maximum parallel jobs (default: CPU count)"
            echo "  --tile-dir PATH    Directory containing tiles (default: ./data/tiles)"
            echo "  --output-dir PATH  Output directory (default: ./output/batch_*)"
            echo "  --help             Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Run main function
main