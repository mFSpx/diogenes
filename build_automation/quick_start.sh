#!/usr/bin/env bash
# LUCIDOTA Quick Start Script
# ==========================
# Fast path to get LUCIDOTA development environment running
#
# Usage:
#   ./build_automation/quick_start.sh      # Full setup
#   ./build_automation/quick_start.sh --dev # Development setup only
#   ./build_automation/quick_start.sh --test # Run tests only
#   ./build_automation/quick_start.sh --clean # Clean environment

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="$ROOT/build_automation"
VENV_DIR="$ROOT/.venv"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# =============================================================================
# Help
# =============================================================================

show_help() {
    echo "LUCIDOTA Quick Start Script"
    echo "=========================="
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  --full, -f    Full setup (env + deps + build + test)"
    echo "  --dev, -d     Development setup (env + deps only)"
    echo "  --test, -t    Run tests only"
    echo "  --smoke, -s   Run smoke tests only"
    echo "  --clean, -c   Clean build artifacts"
    echo "  --help, -h    Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  LUCIDOTA_BUILD_JOBS   - Parallel build jobs (default: nproc)"
    echo "  LUCIDOTA_CUDA_ARCH   - CUDA architecture (default: 75)"
    echo "  DBOS_SYSTEM_DATABASE_URL - Database URL for tests"
    echo ""
    echo "Examples:"
    echo "  $0 --dev              # Setup development environment"
    echo "  $0 --test             # Run all tests"
    echo "  $0 --full             # Complete build and test"
    echo "  LUCIDOTA_BUILD_JOBS=4 $0 --full  # Parallel build with 4 jobs"
    exit 0
}

# =============================================================================
# Logging
# =============================================================================

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_step() {
    echo -e "${YELLOW}==> $1${NC}"
}

# =============================================================================
# Setup Functions
# =============================================================================

setup_environment() {
    log_step "Setting up Python environment"
    
    if [ ! -d "$VENV_DIR" ]; then
        log_info "Creating Python virtual environment..."
        python3 -m venv "$VENV_DIR"
        "$VENV_DIR/bin/python" -m pip install --upgrade pip setuptools wheel
    else
        log_info "Virtual environment already exists"
    fi
    
    # Create build directory
    mkdir -p "$BUILD_DIR"
    touch "$BUILD_DIR/.env-stamp"
}

install_dependencies() {
    log_step "Installing dependencies"
    
    if [ ! -f "$BUILD_DIR/.deps-stamp" ]; then
        log_info "Installing Python dependencies..."
        "$VENV_DIR/bin/pip" install -r "$ROOT/requirements-runtime.txt"
        "$VENV_DIR/bin/pip" install pytest pytest-cov pytest-xdist
        
        # Install doggystyle (DBOS kernel)
        if [ -d "$ROOT/01_REPOS/doggystyle" ]; then
            log_info "Installing DBOS kernel..."
            cd "$ROOT/01_REPOS/doggystyle"
            python3 -m venv .venv
            . .venv/bin/activate
            pip install -e .
            cd "$ROOT"
        fi
        
        touch "$BUILD_DIR/.deps-stamp"
        log_info "Dependencies installed"
    else
        log_info "Dependencies already installed"
    fi
}

build_components() {
    log_step "Building components"
    
    # Python build
    log_info "Building Python components..."
    cd "$ROOT"
    "$VENV_DIR/bin/pip" install -e . 2>/dev/null || echo "No setup.py found, skipping"
    
    # CUDA build (optional)
    if command -v nvcc &>/dev/null; then
        log_info "Building CUDA components..."
        LLAMA_CPP="$ROOT/01_REPOS/prismml_llama.cpp"
        BUILD_CUDA="$LLAMA_CPP/build-cuda"
        mkdir -p "$BUILD_CUDA"
        
        cd "$LLAMA_CPP"
        cmake -S . -B "$BUILD_CUDA" -G Ninja \
            -DGGML_CUDA=ON \
            -DGGML_CUDA_FORCE_MMQ=ON \
            -DCMAKE_CUDA_ARCHITECTURES=${LUCIDOTA_CUDA_ARCH:-75} \
            -DGGML_CCACHE=ON \
            -DCMAKE_BUILD_TYPE=Release
        cmake --build "$BUILD_CUDA" --target llama-server llama-cli -j${LUCIDOTA_BUILD_JOBS:-"$(nproc)"}
        cd "$ROOT"
        log_info "CUDA build complete"
    else
        log_warn "CUDA toolkit not found, skipping CUDA build"
    fi
    
    # Rust build (optional)
    if command -v cargo &>/dev/null; then
        log_info "Building Rust components..."
        CLAW_DIR="$ROOT/01_REPOS/claudecode/rust"
        if [ -d "$CLAW_DIR" ]; then
            cd "$CLAW_DIR"
            cargo build --release -p claw-cli
            cargo test --workspace
            cd "$ROOT"
            log_info "Rust build complete"
        fi
    else
        log_warn "Rust toolchain not found, skipping Rust build"
    fi
}

run_tests() {
    log_step "Running tests"
    
    cd "$ROOT"
    
    # Set database URL if provided
    export DBOS_SYSTEM_DATABASE_URL=${DBOS_SYSTEM_DATABASE_URL:-postgresql://mfspx@/lucidota_state}
    
    log_info "Running Python tests..."
    "$VENV_DIR/bin/python" -m pytest tests/ -x -v --tb=short -n auto \
        --ignore=tests/poison_drop \
        --ignore=tests/test_abductive \
        --ignore=tests/test_absurd \
        --timeout=60 \
        -k "not test_full_system_soak_audit and not test_resource_governor"
}

run_smoke_tests() {
    log_step "Running smoke tests"
    
    cd "$ROOT"
    export DBOS_SYSTEM_DATABASE_URL=${DBOS_SYSTEM_DATABASE_URL:-postgresql://mfspx@/lucidota_state}
    
    log_info "Running core smoke tests..."
    "$VENV_DIR/bin/python" scripts/lucidota_runtime_smoke.py
    "$VENV_DIR/bin/python" scripts/lucidota_kernel_api_smoke.py
    "$VENV_DIR/bin/python" scripts/lucidota_model_artifact_readiness.py
    "$VENV_DIR/bin/python" scripts/lucidota_dbos_smoke.py
    
    log_info "Smoke tests passed!"
}

clean_environment() {
    log_step "Cleaning environment"
    
    # Remove Python artifacts
    find "$ROOT" -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find "$ROOT" -type f -name "*.pyc" -delete 2>/dev/null || true
    find "$ROOT" -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    
    # Remove CUDA build
    rm -rf "$ROOT/01_REPOS/prismml_llama.cpp/build-cuda"
    
    # Remove Docker containers
    if command -v docker &>/dev/null; then
        cd "$ROOT/phantom"
        docker compose down -v 2>/dev/null || true
        cd "$ROOT"
    fi
    
    # Remove build stamps
    rm -f "$BUILD_DIR/.env-stamp" "$BUILD_DIR/.deps-stamp"
    
    log_info "Cleanup complete"
}

check_environment() {
    log_step "Checking environment"
    
    if [ -f "$VENV_DIR/bin/python" ]; then
        log_info "Virtual environment: OK ("$("$VENV_DIR/bin/python" --version 2>&1)")"
    else
        log_error "Virtual environment: NOT FOUND"
        echo "Run $0 --dev to create it"
        exit 1
    fi
    
    # Check dependencies
    if [ -f "$BUILD_DIR/.deps-stamp" ]; then
        log_info "Dependencies: OK"
    else
        log_warn "Dependencies: NOT INSTALLED"
        echo "Run $0 --dev to install them"
    fi
    
    log_info "Environment check complete"
}

# =============================================================================
# Main
# =============================================================================

# Parse arguments
TARGET="dev"
while [ $# -gt 0 ]; do
    case "$1" in
        --full|-f)
            TARGET="full"
            shift
            ;;
        --dev|-d)
            TARGET="dev"
            shift
            ;;
        --test|-t)
            TARGET="test"
            shift
            ;;
        --smoke|-s)
            TARGET="smoke"
            shift
            ;;
        --clean|-c)
            TARGET="clean"
            shift
            ;;
        --help|-h)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Execute based on target
case "$TARGET" in
    full)
        log_step "Full Setup (env + deps + build + test)"
        setup_environment
        install_dependencies
        build_components
        run_tests
        log_info "Full setup complete!"
        ;;
    dev)
        log_step "Development Setup (env + deps)"
        setup_environment
        install_dependencies
        log_info "Development setup complete!"
        ;;
    test)
        log_step "Running Tests"
        check_environment
        run_tests
        ;;
    smoke)
        log_step "Running Smoke Tests"
        check_environment
        run_smoke_tests
        ;;
    clean)
        log_step "Cleaning Environment"
        clean_environment
        ;;
esac

log_info "Done!"
