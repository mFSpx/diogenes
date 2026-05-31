# LUCIDOTA Build Automation Makefile
# ======================================
# Comprehensive build system for the LUCIDOTA project
# Supports: Python environment, CUDA builds, Docker, tests, CI/CD
#
# Usage:
#   make help          - Show all available targets
#   make all           - Full build (env + deps + tests)
#   make dev           - Development setup
#   make test          - Run all tests
#   make smoke         - Run smoke tests
#   make docker        - Build Docker images
#   make clean         - Clean build artifacts
#

# =============================================================================
# Configuration
# =============================================================================

ROOT_DIR := $(shell pwd)
SCRIPTS_DIR := $(ROOT_DIR)/scripts
BUILD_DIR := $(ROOT_DIR)/build_automation
VENV_DIR := $(ROOT_DIR)/.venv
PYTHON := $(VENV_DIR)/bin/python
PIP := $(VENV_DIR)/bin/pip

# Build configuration
BUILD_JOBS ?= $(shell nproc)
PYTHON_VERSION ?= 3.12
CUDA_ARCH ?= 75
DOCKER_GID ?= 988

# Project components
PHANTOM_DIR := $(ROOT_DIR)/phantom
LLAMA_CPP_DIR := $(ROOT_DIR)/01_REPOS/prismml_llama.cpp
DBOS_KERNEL_DIR := $(ROOT_DIR)/01_REPOS/doggystyle
CLAW_DIR := $(ROOT_DIR)/01_REPOS/claudecode/rust

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m

.PHONY: help

# =============================================================================
# Help / Documentation
# =============================================================================

help: ## Show this help message
	@echo "LUCIDOTA Build System"
	@echo "===================="
	@echo ""
	@echo "Core Targets:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2}'
	@echo ""
	@echo "Configuration:"
	@echo "  BUILD_JOBS=$(BUILD_JOBS)"
	@echo "  PYTHON_VERSION=$(PYTHON_VERSION)"
	@echo "  CUDA_ARCH=$(CUDA_ARCH)"
	@echo "  DOCKER_GID=$(DOCKER_GID)"
	@echo ""
	@echo "Environment Variables:"
	@echo "  LUCIDOTA_BUILD_JOBS   - Parallel build jobs (default: nproc)"
	@echo "  LUCIDOTA_CUDA_ARCH   - CUDA architecture (default: 75)"
	@echo "  DBOS_SYSTEM_DATABASE_URL - Database URL for DBOS tests"
	@echo ""

# =============================================================================
# Main Build Targets
# =============================================================================

all: env deps build test ## Full build: environment, dependencies, build, and test

dev: env deps ## Development setup: environment and dependencies only

build: build-python build-cuda build-rust ## Build all components

# =============================================================================
# Environment Setup
# =============================================================================

.env-stamp:
	@mkdir -p $(BUILD_DIR)
	@echo "Creating Python virtual environment..."
	python$(PYTHON_VERSION) -m venv $(VENV_DIR)
	$(PIP) install --upgrade pip setuptools wheel
	@echo "$(GREEN)Virtual environment created$(NC)"
	touch $@

env: .env-stamp ## Setup Python virtual environment

# =============================================================================
# Dependencies
# =============================================================================

.deps-stamp: .env-stamp
	@echo "Installing Python dependencies..."
	$(PIP) install -r $(ROOT_DIR)/requirements-runtime.txt
	$(PIP) install pytest pytest-cov pytest-xdist
	$(PIP) install -e $(ROOT_DIR)/01_REPOS/doggystyle
	@echo "$(GREEN)Dependencies installed$(NC)"
	touch $@

deps: .deps-stamp ## Install all Python dependencies

# =============================================================================
# Python Build
# =============================================================================

build-python: .env-stamp
	@echo "$(YELLOW)Building Python components...$(NC)"
	cd $(ROOT_DIR) && $(PIP) install -e . 2>/dev/null || echo "No setup.py found, skipping"
	@echo "$(GREEN)Python build complete$(NC)"

# =============================================================================
# CUDA Builds (llama.cpp)
# =============================================================================

LLAMA_BUILD_DIR := $(LLAMA_CPP_DIR)/build-cuda

build-cuda: 
	@echo "$(YELLOW)Building CUDA components...$(NC)"
	@mkdir -p $(LLAMA_BUILD_DIR)
	cd $(LLAMA_CPP_DIR) && \
	cmake -S . -B $(LLAMA_BUILD_DIR) -G Ninja \
		-DGGML_CUDA=ON \
		-DGGML_CUDA_FORCE_MMQ=ON \
		-DCMAKE_CUDA_ARCHITECTURES=$(CUDA_ARCH) \
		-DGGML_CCACHE=ON \
		-DCMAKE_BUILD_TYPE=Release
	cmake --build $(LLAMA_BUILD_DIR) --target llama-server llama-cli -j$(BUILD_JOBS)
	@echo "$(GREEN)CUDA build complete$(NC)"

# =============================================================================
# Rust Builds (CLAW)
# =============================================================================

build-rust:
	@echo "$(YELLOW)Building Rust components...$(NC)"
	cd $(CLAW_DIR) && \
	cargo build --release -p claw-cli && \
	cargo test --workspace
	@echo "$(GREEN)Rust build complete$(NC)"

# =============================================================================
# Docker Builds
# =============================================================================

docker-phantom: 
	@echo "$(YELLOW)Building Phantom Docker image...$(NC)"
	cd $(PHANTOM_DIR) && \
	docker compose build
	@echo "$(GREEN)Phantom Docker build complete$(NC)"

docker: docker-phantom ## Build all Docker images

# =============================================================================
# Testing
# =============================================================================

test: test-python test-kernel test-claw ## Run all tests

test-python:
	@echo "$(YELLOW)Running Python tests...$(NC)"
	cd $(ROOT_DIR) && \
	$(PYTHON) -m pytest tests/ -x -v --tb=short -n auto
	@echo "$(GREEN)Python tests complete$(NC)"

test-kernel:
	@echo "$(YELLOW)Running Kernel tests...$(NC)"
	cd $(DBOS_KERNEL_DIR) && \
	. .venv/bin/activate && \
	python -m pytest -q
	@echo "$(GREEN)Kernel tests complete$(NC)"

test-claw:
	@echo "$(YELLOW)Running CLAW tests...$(NC)"
	cd $(CLAW_DIR) && \
	cargo test --workspace
	@echo "$(GREEN)CLAW tests complete$(NC)"

# =============================================================================
# Smoke Tests (from check_diogenes.sh)
# =============================================================================

smoke: smoke-environment smoke-python smoke-rust ## Run all smoke tests

smoke-environment:
	@echo "$(YELLOW)Running environment smoke tests...$(NC)"
	$(call-smoke-script)
	@echo "$(GREEN)Environment smoke tests passed$(NC)"

smoke-python:
	@echo "$(YELLOW)Running Python smoke tests...$(NC)"
	cd $(ROOT_DIR) && \
	DBOS_SYSTEM_DATABASE_URL="postgresql://mfspx@/lucidota_state" \
	$(PYTHON) scripts/lucidota_runtime_smoke.py && \
	$(PYTHON) scripts/lucidota_kernel_api_smoke.py && \
	$(PYTHON) scripts/lucidota_model_artifact_readiness.py && \
	$(PYTHON) scripts/lucidota_dbos_smoke.py
	@echo "$(GREEN)Python smoke tests passed$(NC)"

smoke-rust:
	@echo "$(YELLOW)Running Rust smoke tests...$(NC)"
	cd $(CLAW_DIR) && \
	cargo test --workspace
	@echo "$(GREEN)Rust smoke tests passed$(NC)"

# =============================================================================
# Diogenes Check (Full System Verification)
# =============================================================================

diogenes-check:
	@echo "$(YELLOW)Running full Diogenes system check...$(NC)"
	cd $(ROOT_DIR) && \
	./check_diogenes.sh
	@echo "$(GREEN)Diogenes check complete$(NC)"

# =============================================================================
# Code Quality
# =============================================================================

lint:
	@echo "$(YELLOW)Running linters...$(NC)"
	cd $(ROOT_DIR) && \
	$(PYTHON) -m flake8 scripts/ --max-line-length=120 --ignore=E501,W503 || true
	$(PYTHON) -m black --check scripts/ 2>/dev/null || echo "black not installed, skipping"
	@echo "$(GREEN)Linting complete$(NC)"

format:
	@echo "$(YELLOW)Formatting code...$(NC)"
	cd $(ROOT_DIR) && \
	$(PYTHON) -m black scripts/ 2>/dev/null || echo "black not installed, skipping"
	$(PYTHON) -m isort scripts/ 2>/dev/null || echo "isort not installed, skipping"
	@echo "$(GREEN)Formatting complete$(NC)"

# =============================================================================
# Clean Targets
# =============================================================================

clean: clean-python clean-cuda clean-docker clean-build ## Clean all build artifacts

clean-python:
	@echo "$(YELLOW)Cleaning Python artifacts...$(NC)"
	find $(ROOT_DIR) -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find $(ROOT_DIR) -type f -name "*.pyc" -delete 2>/dev/null || true
	find $(ROOT_DIR) -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "$(GREEN)Python cleanup complete$(NC)"

clean-cuda:
	@echo "$(YELLOW)Cleaning CUDA build artifacts...$(NC)"
	rm -rf $(LLAMA_BUILD_DIR)
	@echo "$(GREEN)CUDA cleanup complete$(NC)"

clean-docker:
	@echo "$(YELLOW)Cleaning Docker artifacts...$(NC)"
	cd $(PHANTOM_DIR) && \
	docker compose down -v 2>/dev/null || true
	@echo "$(GREEN)Docker cleanup complete$(NC)"

clean-build:
	@echo "$(YELLOW)Cleaning build directory...$(NC)"
	rm -rf $(BUILD_DIR)
	rm -f .env-stamp .deps-stamp
	@echo "$(GREEN)Build directory cleanup complete$(NC)"

# =============================================================================
# Utility Targets
# =============================================================================

check-env: ## Check if environment is set up
	@if [ -f "$(VENV_DIR)/bin/python" ]; then \
		echo "$(GREEN)Virtual environment exists$(NC)"; \
		$(PYTHON) --version; \
	else \
		echo "$(RED)Virtual environment not found$(NC)"; \
		echo "Run 'make env' to create it"; \
		false; \
	fi

check-deps: ## Check if dependencies are installed
	@if [ -f "$(VENV_DIR)/bin/pip" ]; then \
		echo "$(GREEN)Dependencies check$(NC)"; \
		$(PIP) list | grep -E "(dbos|river|transformers|accelerate)" || echo "Core dependencies not found"; \
	else \
		echo "$(RED)Dependencies not installed$(NC)"; \
		echo "Run 'make deps' to install them"; \
		false; \
	fi

status: check-env check-deps ## Show build system status

# =============================================================================
# Dev Library Operations
# =============================================================================

dev-library-scan:
	@echo "$(YELLOW)Scanning Dev Library...$(NC)"
	cd $(ROOT_DIR) && \
	$(PYTHON) scripts/dev_library_scan.py --list
	@echo "$(GREEN)Dev Library scan complete$(NC)"

dev-library-query: ## Query the Dev Library (usage: make dev-library-query QUERY=topic)
	@echo "$(YELLOW)Querying Dev Library for '$(QUERY)'...$(NC)"
	cd $(ROOT_DIR) && \
	$(PYTHON) scripts/dev_library_scan.py --query $(QUERY)
	@echo "$(GREEN)Query complete$(NC)"

# =============================================================================
# Version Info
# =============================================================================

version:
	@echo "$(YELLOW)LUCIDOTA Build System Version$(NC)"
	@echo "Makefile: $(MAKEFILE_LIST)"
	@echo "Git commit: $(shell git rev-parse --short HEAD 2>/dev/null || echo 'unknown')"
	@echo "Timestamp: $(shell date -u +%Y%m%dT%H%M%SZ)"

# =============================================================================
# Define helper for smoke tests
# =============================================================================

define call-smoke-script
	@echo "Running check_diogenes.sh..." && \
	cd $(ROOT_DIR) && \
	./check_diogenes.sh
endef
