#!/bin/bash

# ============================================================================
# Environment Setup Script for Bushfire Risk Assessment
# Compatible with HPC environments and local development (including macOS)
# ============================================================================

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect operating system
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        OS="linux"
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        OS="macos"
        # Check if running on Apple Silicon
        if [[ $(uname -m) == "arm64" ]]; then
            ARCH="arm64"
            log_info "Detected Apple Silicon Mac"
        else
            ARCH="x86_64"
            log_info "Detected Intel Mac"
        fi
    else
        OS="unknown"
    fi
}

# Check Python installation
check_python() {
    log_info "Checking Python installation..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        log_info "Found Python ${PYTHON_VERSION}"
        
        # Check if version is 3.8 or higher
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
            log_info "Python version is compatible"
        else
            log_error "Python 3.8 or higher is required"
            exit 1
        fi
    else
        log_error "Python 3 is not installed"
        exit 1
    fi
}

# Create virtual environment
setup_venv() {
    VENV_DIR="${PROJECT_DIR}/venv"
    
    if [ -d "$VENV_DIR" ]; then
        log_warn "Virtual environment already exists"
        read -p "Do you want to recreate it? (y/N) " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            rm -rf "$VENV_DIR"
        else
            return
        fi
    fi
    
    log_info "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
    
    # Activate virtual environment
    source "$VENV_DIR/bin/activate"
    
    # Upgrade pip
    log_info "Upgrading pip..."
    pip install --upgrade pip wheel setuptools
}

# Install dependencies
install_dependencies() {
    log_info "Installing Python dependencies..."
    
    # Install main requirements
    pip install -r "${PROJECT_DIR}/requirements.txt"
    
    # Install additional HPC/parallel processing dependencies
    log_info "Installing additional parallel processing dependencies..."
    pip install dask[complete] joblib tqdm memory_profiler
    
    # Install development dependencies
    if [[ "${1:-}" == "--dev" ]]; then
        log_info "Installing development dependencies..."
        pip install pytest pytest-cov black flake8 mypy jupyter jupyterlab
    fi
}

# Setup GDAL (if needed)
setup_gdal() {
    log_info "Checking GDAL installation..."
    
    if [[ "$OS" == "macos" ]]; then
        if ! command -v gdal-config &> /dev/null; then
            log_warn "GDAL not found. Install with: brew install gdal"
        else
            GDAL_VERSION=$(gdal-config --version)
            log_info "Found GDAL ${GDAL_VERSION}"
        fi
    elif [[ "$OS" == "linux" ]]; then
        if ! command -v gdal-config &> /dev/null; then
            log_warn "GDAL not found. Install with: sudo apt-get install gdal-bin libgdal-dev"
        fi
    fi
}

# Setup PostgreSQL/PostGIS (optional)
check_postgresql() {
    log_info "Checking PostgreSQL/PostGIS (optional)..."
    
    if command -v psql &> /dev/null; then
        PSQL_VERSION=$(psql --version | cut -d' ' -f3)
        log_info "Found PostgreSQL ${PSQL_VERSION}"
        
        # Check if we can connect
        if psql -U postgres -c "SELECT 1;" &> /dev/null; then
            log_info "PostgreSQL connection successful"
        else
            log_warn "PostgreSQL is installed but connection failed"
        fi
    else
        log_warn "PostgreSQL not found (optional for this project)"
    fi
}

# Create environment file
create_env_file() {
    ENV_FILE="${PROJECT_DIR}/.env"
    
    if [ -f "$ENV_FILE" ]; then
        log_warn "Environment file already exists"
        return
    fi
    
    log_info "Creating environment file..."
    cat > "$ENV_FILE" << EOF
# Database Configuration (optional)
DB_HOST=localhost
DB_NAME=bushfire_risk
DB_USER=postgres
DB_PASSWORD=
DB_PORT=5432

# Processing Configuration
NUM_WORKERS=auto
CHUNK_SIZE=1024
BATCH_SIZE=16

# Paths
DATA_DIR=./data
OUTPUT_DIR=./output
LOG_DIR=./logs

# Sentinel Hub Configuration (for real data access)
# SH_CLIENT_ID=your_client_id
# SH_CLIENT_SECRET=your_client_secret
EOF
    
    log_info "Created .env file (update with your settings)"
}

# Download sample data
download_sample_data() {
    DATA_DIR="${PROJECT_DIR}/data/sample"
    
    if [ -d "$DATA_DIR" ]; then
        log_info "Sample data directory already exists"
        return
    fi
    
    log_info "Creating sample data directory..."
    mkdir -p "$DATA_DIR"
    
    # In a real scenario, this would download actual Sentinel-2 data
    log_info "Sample data directory created at: ${DATA_DIR}"
}

# System information
print_system_info() {
    log_info "System Information:"
    echo "  OS: ${OS} ${ARCH:-}"
    echo "  CPU Cores: $(sysctl -n hw.ncpu 2>/dev/null || nproc)"
    
    if [[ "$OS" == "macos" ]]; then
        echo "  Memory: $(sysctl -n hw.memsize | awk '{print $1/1024/1024/1024}') GB"
    else
        echo "  Memory: $(free -g | awk '/^Mem:/{print $2}') GB"
    fi
    
    echo "  Python: $(python3 --version)"
}

# Main setup
main() {
    # Get project directory
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
    
    log_info "Setting up Bushfire Risk Assessment environment"
    log_info "Project directory: ${PROJECT_DIR}"
    
    # Detect OS
    detect_os
    
    # Run setup steps
    check_python
    setup_venv
    install_dependencies "$@"
    setup_gdal
    check_postgresql
    create_env_file
    download_sample_data
    
    # Print summary
    echo
    log_info "Environment setup completed!"
    echo
    print_system_info
    echo
    log_info "To activate the environment, run:"
    echo "  source ${PROJECT_DIR}/venv/bin/activate"
    echo
    log_info "To run parallel processing demo:"
    echo "  python parallel_processor.py"
    echo
    log_info "To run the full pipeline:"
    echo "  ./scripts/run_parallel_analysis.sh"
}

# Run main function
main "$@"