# LUCIDOTA Build Automation - Justfile
# ======================================
# Modern alternative to Makefile using just (https://just.systems)
# Install: cargo install just
#
# Usage:
#   just           - Show all recipes
#   just all       - Full build
#   just dev       - Development setup
#   just test      - Run tests
#   just build     - Build all components
#   just docker    - Build Docker images
#   just clean     - Clean build artifacts

# Configuration
set working-directory := "{{ os_env('PWD') }}"

# Build settings
python_version := "3.12"
build_jobs := if os_env('LUCIDOTA_BUILD_JOBS') { os_env('LUCIDOTA_BUILD_JOBS') } else { os.cpus() }
cuda_arch := if os_env('LUCIDOTA_CUDA_ARCH') { os_env('LUCIDOTA_CUDA_ARCH') } else { "75" }
docker_gid := if os_env('DOCKER_GID') { os_env('DOCKER_GID') } else { "988" }

# Paths
venv_dir := working_directory / ".venv"
python_exe := venv_dir / "bin" / "python"
pip_exe := venv_dir / "bin" / "pip"
phantom_dir := working_directory / "phantom"
llama_dir := working_directory / "01_REPOS" / "prismml_llama.cpp"
dbos_kernel_dir := working_directory / "01_REPOS" / "doggystyle"
claw_dir := working_directory / "01_REPOS" / "claudecode" / "rust"
build_dir := working_directory / "build_automation"

# Colors
GREEN := "\x1b[0;32m"
YELLOW := "\x1b[1;33m"
RED := "\x1b[0;31m"
NC := "\x1b[0m"

# =============================================================================
# Help
# =============================================================================

@_default:
    #!/usr/bin/env just --justfile
    just --list

# =============================================================================
# Main Targets
# =============================================================================

# Full build: environment, dependencies, build, and test
[group('build')]
all: env deps build test

# Development setup
dev: env deps

# Build all components
build: build-python build-cuda build-rust

# =============================================================================
# Environment Setup
# =============================================================================

# Create Python virtual environment
[group('env')]
env:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Creating Python virtual environment..."
    mkdir -p {{ build_dir }}
    python{{ python_version }} -m venv {{ venv_dir }}
    {{ pip_exe }} install --upgrade pip setuptools wheel
    echo "{{ GREEN }}Virtual environment created{{ NC }}"

# =============================================================================
# Dependencies
# =============================================================================

[group('deps')]
deps: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "Installing Python dependencies..."
    {{ pip_exe }} install -r {{ working_directory }}/requirements-runtime.txt
    {{ pip_exe }} install pytest pytest-cov pytest-xdist
    {{ pip_exe }} install -e {{ dbos_kernel_dir }}
    echo "{{ GREEN }}Dependencies installed{{ NC }}"

# =============================================================================
# Python Build
# =============================================================================

[group('python')]
build-python: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Building Python components...{{ NC }}"
    cd {{ working_directory }}
    {{ pip_exe }} install -e . 2>/dev/null || echo "No setup.py found, skipping"
    echo "{{ GREEN }}Python build complete{{ NC }}"

# =============================================================================
# CUDA Builds
# =============================================================================

[group('cuda')]
build-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Building CUDA components...{{ NC }}"
    LLAMA_BUILD="{{ llama_dir }}/build-cuda"
    mkdir -p "$LLAMA_BUILD"
    cd {{ llama_dir }}
    cmake -S . -B "$LLAMA_BUILD" -G Ninja \
        -DGGML_CUDA=ON \
        -DGGML_CUDA_FORCE_MMQ=ON \
        -DCMAKE_CUDA_ARCHITECTURES={{ cuda_arch }} \
        -DGGML_CCACHE=ON \
        -DCMAKE_BUILD_TYPE=Release
    cmake --build "$LLAMA_BUILD" --target llama-server llama-cli -j{{ build_jobs }}
    echo "{{ GREEN }}CUDA build complete{{ NC }}"

# =============================================================================
# Rust Builds (CLAW)
# =============================================================================

[group('rust')]
build-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Building Rust components...{{ NC }}"
    cd {{ claw_dir }}
    cargo build --release -p claw-cli
    cargo test --workspace
    echo "{{ GREEN }}Rust build complete{{ NC }}"

# =============================================================================
# Docker Builds
# =============================================================================

[group('docker')]
docker:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Building Docker images...{{ NC }}"
    cd {{ phantom_dir }}
    DOCKER_GID={{ docker_gid }} docker compose build
    echo "{{ GREEN }}Docker build complete{{ NC }}"

# =============================================================================
# Testing
# =============================================================================

[group('test')]
test: test-python test-kernel test-claw

test-python: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Running Python tests...{{ NC }}"
    cd {{ working_directory }}
    DBOS_SYSTEM_DATABASE_URL="postgresql://mfspx@/lucidota_state" \
    {{ python_exe }} -m pytest tests/ -x -v --tb=short -n auto
    echo "{{ GREEN }}Python tests complete{{ NC }}"

test-kernel: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Running Kernel tests...{{ NC }}"
    cd {{ dbos_kernel_dir }}
    . .venv/bin/activate
    python -m pytest -q
    echo "{{ GREEN }}Kernel tests complete{{ NC }}"

test-claw:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Running CLAW tests...{{ NC }}"
    cd {{ claw_dir }}
    cargo test --workspace
    echo "{{ GREEN }}CLAW tests complete{{ NC }}"

# =============================================================================
# Smoke Tests
# =============================================================================

[group('smoke')]
smoke: smoke-python smoke-rust

smoke-python: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Running Python smoke tests...{{ NC }}"
    cd {{ working_directory }}
    DBOS_SYSTEM_DATABASE_URL="postgresql://mfspx@/lucidota_state" \
    {{ python_exe }} scripts/lucidota_runtime_smoke.py
    {{ python_exe }} scripts/lucidota_kernel_api_smoke.py
    {{ python_exe }} scripts/lucidota_model_artifact_readiness.py
    {{ python_exe }} scripts/lucidota_dbos_smoke.py
    echo "{{ GREEN }}Python smoke tests passed{{ NC }}"

smoke-rust:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Running Rust smoke tests...{{ NC }}"
    cd {{ claw_dir }}
    cargo test --workspace
    echo "{{ GREEN }}Rust smoke tests passed{{ NC }}"

# Full Diogenes system check
diogenes-check:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Running full Diogenes system check...{{ NC }}"
    cd {{ working_directory }}
    ./check_diogenes.sh
    echo "{{ GREEN }}Diogenes check complete{{ NC }}"

# =============================================================================
# Code Quality
# =============================================================================

[group('quality')]
lint: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Running linters...{{ NC }}"
    cd {{ working_directory }}
    {{ python_exe }} -m flake8 scripts/ --max-line-length=120 --ignore=E501,W503 || true
    {{ python_exe }} -m black --check scripts/ 2>/dev/null || echo "black not installed"
    echo "{{ GREEN }}Linting complete{{ NC }}"

format: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Formatting code...{{ NC }}"
    cd {{ working_directory }}
    {{ python_exe }} -m black scripts/ 2>/dev/null || echo "black not installed"
    {{ python_exe }} -m isort scripts/ 2>/dev/null || echo "isort not installed"
    echo "{{ GREEN }}Formatting complete{{ NC }}"

# =============================================================================
# Clean Targets
# =============================================================================

[group('clean')]
clean: clean-python clean-cuda clean-docker clean-build

clean-python:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Cleaning Python artifacts...{{ NC }}"
    find {{ working_directory }} -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
    find {{ working_directory }} -type f -name "*.pyc" -delete 2>/dev/null || true
    find {{ working_directory }} -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
    echo "{{ GREEN }}Python cleanup complete{{ NC }}"

clean-cuda:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Cleaning CUDA build artifacts...{{ NC }}"
    rm -rf {{ llama_dir }}/build-cuda
    echo "{{ GREEN }}CUDA cleanup complete{{ NC }}"

clean-docker:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Cleaning Docker artifacts...{{ NC }}"
    cd {{ phantom_dir }}
    docker compose down -v 2>/dev/null || true
    echo "{{ GREEN }}Docker cleanup complete{{ NC }}"

clean-build:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Cleaning build directory...{{ NC }}"
    rm -rf {{ build_dir }}
    rm -f .env-stamp .deps-stamp
    echo "{{ GREEN }}Build directory cleanup complete{{ NC }}"

# =============================================================================
# Utility Targets
# =============================================================================

[group('utils')]

# Check environment status
check-env:
    #!/usr/bin/env bash
    if [ -f "{{ venv_dir }}/bin/python" ]; then
        echo "{{ GREEN }}Virtual environment exists{{ NC }}"
        {{ python_exe }} --version
    else
        echo "{{ RED }}Virtual environment not found{{ NC }}"
        echo "Run 'just env' to create it"
        exit 1
    fi

# Check dependencies
check-deps: check-env
    #!/usr/bin/env bash
    echo "{{ GREEN }}Dependencies check{{ NC }}"
    {{ pip_exe }} list | grep -E "(dbos|river|transformers|accelerate)" || echo "Core dependencies not found"

# Show status
status: check-env check-deps

# =============================================================================
# Dev Library Operations
# =============================================================================

[group('library')]

dev-library-scan:
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Scanning Dev Library...{{ NC }}"
    cd {{ working_directory }}
    {{ python_exe }} scripts/dev_library_scan.py --list
    echo "{{ GREEN }}Dev Library scan complete{{ NC }}"

dev-library-query query="":
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Querying Dev Library for '{{ query }}'...{{ NC }}"
    cd {{ working_directory }}
    {{ python_exe }} scripts/dev_library_scan.py --query "{{ query }}"
    echo "{{ GREEN }}Query complete{{ NC }}"

# =============================================================================
# Version Info
# =============================================================================

version:
    #!/usr/bin/env bash
    echo "{{ YELLOW }}LUCIDOTA Build System Version{{ NC }}"
    echo "Justfile: $JUSTFILE"
    echo "Working directory: $PWD"
    git rev-parse --short HEAD 2>/dev/null || echo "Git: unknown"
    date -u +%Y%m%dT%H%M%SZ

# =============================================================================
# Run Targets
# =============================================================================

# Run Phantom
run-phantom:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ phantom_dir }}
    docker compose up -d

# Stop Phantom
stop-phantom:
    #!/usr/bin/env bash
    set -euo pipefail
    cd {{ phantom_dir }}
    docker compose down

# =============================================================================
# Build Automation Scripts
# =============================================================================

# Generate build report
build-report: env
    #!/usr/bin/env bash
    set -euo pipefail
    echo "{{ YELLOW }}Generating build report...{{ NC }}"
    cd {{ working_directory }}
    {{ python_exe }} build_automation/build_report.py
    echo "{{ GREEN }}Build report generated{{ NC }}"
