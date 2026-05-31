# LUCIDOTA Build Automation

Comprehensive build system for the LUCIDOTA project. This system provides multiple interfaces for building, testing, and deploying LUCIDOTA components.

## Overview

The LUCIDOTA build automation system includes:

| Component | File | Purpose |
|-----------|------|---------|
| Makefile | `Makefile` | Traditional build system (GNU Make) |
| Justfile | `justfile` | Modern build system (just) |
| CI/CD | `.github/workflows/lucidota-build.yml` | GitHub Actions workflow |
| Quick Start | `build_automation/quick_start.sh` | Shell script for rapid setup |
| Report Generator | `build_automation/build_report.py` | System status reporting |

## Quick Start

### 1. Development Setup

```bash
# Using Make
make dev

# Using Just (requires: cargo install just)
just dev

# Using Quick Start script
./build_automation/quick_start.sh --dev
```

### 2. Full Build and Test

```bash
# Using Make
make all

# Using Just
just all

# Using Quick Start script
./build_automation/quick_start.sh --full
```

### 3. Run Tests

```bash
# Using Make
make test

# Using Just
just test

# Using Quick Start script
./build_automation/quick_start.sh --test
```

## Build Targets

### Core Targets

| Target | Description |
|--------|-------------|
| `all` | Full build (environment + dependencies + build + test) |
| `dev` | Development setup (environment + dependencies only) |
| `build` | Build all components |
| `test` | Run all tests |
| `smoke` | Run smoke tests |
| `docker` | Build Docker images |
| `clean` | Clean all build artifacts |

### Environment Targets

| Target | Description |
|--------|-------------|
| `env` | Setup Python virtual environment |
| `deps` | Install all Python dependencies |
| `check-env` | Check if environment is set up |
| `check-deps` | Check if dependencies are installed |
| `status` | Show build system status |

### Component Build Targets

| Target | Description | Requirements |
|--------|-------------|--------------|
| `build-python` | Build Python components | Python 3.12+ |
| `build-cuda` | Build CUDA components (llama.cpp) | CUDA Toolkit, cmake, ninja |
| `build-rust` | Build Rust components (CLAW) | Rust toolchain |
| `docker` | Build all Docker images | Docker, Docker Compose |
| `docker-phantom` | Build Phantom Docker image | Docker, Docker Compose |

### Test Targets

| Target | Description |
|--------|-------------|
| `test-python` | Run Python tests |
| `test-kernel` | Run Kernel tests |
| `test-claw` | Run CLAW tests |
| `smoke-python` | Run Python smoke tests |
| `smoke-rust` | Run Rust smoke tests |
| `diogenes-check` | Full Diogenes system check |

### Quality Targets

| Target | Description |
|--------|-------------|
| `lint` | Run linters (flake8, black) |
| `format` | Format code (black, isort) |

### Clean Targets

| Target | Description |
|--------|-------------|
| `clean-python` | Clean Python artifacts |
| `clean-cuda` | Clean CUDA build artifacts |
| `clean-docker` | Clean Docker artifacts |
| `clean-build` | Clean build directory |

## Justfile Features

The `justfile` provides additional features beyond the Makefile:

```bash
# Show version info
just version

# Run Phantom Docker container
just run-phantom
just stop-phantom

# Dev Library operations
just dev-library-scan
just dev-library-query query="krampus"

# Generate build report
just build-report
```

## CI/CD Pipeline

The GitHub Actions workflow `.github/workflows/lucidota-build.yml` provides:

### Triggers
- **Push to main/diogenes/main**: Runs full build pipeline
- **Pull Requests**: Runs core tests and validation
- **Manual Trigger**: Full build with CUDA support via workflow_dispatch

### Jobs

1. **build-config**: Determines build matrix based on event
2. **build-core**: Python environment, dependencies, core tests
3. **build-cuda**: CUDA builds (llama.cpp) - requires GPU runner
4. **build-rust**: Rust builds (CLAW)
5. **build-docker**: Phantom Docker image build and push
6. **integration-test**: Full system integration tests
7. **docs**: Build report generation

### Manual Trigger

```bash
# Trigger via GitHub CLI
gh workflow run lucidota-build.yml --ref main --field force-full=true

# Or via API
curl -X POST \
  -H "Authorization: token YOUR_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/OWNER/REPO/actions/workflows/lucidota-build.yml/dispatches \
  -d '{"ref":"main","inputs":{"force-full":"true"}}'
```

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `LUCIDOTA_BUILD_JOBS` | `nproc` | Parallel build jobs |
| `LUCIDOTA_CUDA_ARCH` | `75` | CUDA architecture for llama.cpp |
| `DOCKER_GID` | `988` | Docker group ID for Phantom |
| `DBOS_SYSTEM_DATABASE_URL` | `postgresql://mfspx@/lucidota_state` | Database URL for DBOS |

### Python Dependencies

Core runtime dependencies are defined in `requirements-runtime.txt`:
- dbos
- river
- treelite
- transformers
- accelerate
- peft
- safetensors
- sentencepiece
- datasets
- bitsandbytes

## System Requirements

### Required Tools

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.12+ | Primary language |
| pip | latest | Package manager |
| venv | built-in | Virtual environment |
| make | any | Build system (optional with Just) |
| just | 1.0+ | Alternative build system (optional) |
| git | any | Version control |

### Optional Tools (for full builds)

| Tool | Version | Purpose |
|------|---------|---------|
| cmake | 3.20+ | CUDA builds |
| ninja | 1.10+ | CUDA builds |
| ccache | any | Build caching |
| Rust | 1.70+ | CLAW builds |
| Docker | 24+ | Containerization |
| Docker Compose | 2+ | Container orchestration |
| CUDA Toolkit | 11+ | GPU acceleration |
| PostgreSQL | 15+ | Database (with AGE extension) |

## Project Structure

```
LUCIDOTA/
├── Makefile                    # GNU Make build system
├── justfile                    # Just build system
├── .github/
│   └── workflows/
│       └── lucidota-build.yml  # GitHub Actions CI/CD
├── build_automation/
│   ├── quick_start.sh          # Quick start script
│   ├── build_report.py         # Build report generator
│   └── README.md               # This file
├── scripts/                    # 484+ Python scripts
│   ├── check_diogenes.sh       # Full system verification
│   └── ...
├── phantom/
│   ├── Dockerfile              # Phantom Docker image
│   └── docker-compose.yaml     # Docker Compose config
├── 01_REPOS/                   # External repositories
│   ├── doggystyle/             # DBOS kernel
│   ├── claudecode/
│   │   └── rust/               # CLAW (Rust)
│   └── prismml_llama.cpp/      # llama.cpp (CUDA)
├── tests/                      # Test suite
└── requirements-runtime.txt    # Python dependencies
```

## Dev Library Integration

The build system respects the [LUCIDOTA Dev Library Reuse Law](00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md):

- Before writing new build scripts, check existing tools in the Dev Library
- Use `python3 scripts/dev_library_scan.py --query <topic>` to find reusable components
- Sovereign artifacts in the proof hoard are preserved
- Harden production copies through contracts/tests/receipts

## Usage Examples

### Local Development

```bash
# Create virtual environment
make env

# Install dependencies
make deps

# Check status
make status

# Run specific tests
make test-python

# Clean up
make clean
```

### Docker Development

```bash
# Build Phantom Docker image
make docker

# Or with Just
just docker

# Start Phantom services
cd phantom && docker compose up -d

# Stop Phantom services
cd phantom && docker compose down
```

### CI/CD Operations

```bash
# Check build report
just build-report

# Query Dev Library
just dev-library-query query="build"

# Full system verification
make diogenes-check
```

### Custom Builds

```bash
# CUDA build with specific architecture
LUCIDOTA_CUDA_ARCH=80 make build-cuda

# Parallel build with 8 jobs
LUCIDOTA_BUILD_JOBS=8 make all

# Custom database URL
DBOS_SYSTEM_DATABASE_URL=postgresql://user:pass@host/db make test
```

## Troubleshooting

### Common Issues

1. **Python version not found**:
   ```bash
   # Install Python 3.12
   sudo apt-get install python3.12 python3.12-venv python3.12-dev
   ```

2. **Missing build tools**:
   ```bash
   # Install on Ubuntu/Debian
   sudo apt-get install build-essential cmake ninja-build ccache
   ```

3. **CUDA not available**:
   ```bash
   # CUDA builds are optional, skip with:
   make build-python  # Skip CUDA and Rust
   ```

4. **Docker permission issues**:
   ```bash
   # Add user to docker group
   sudo usermod -aG docker $USER
   newgrp docker
   ```

5. **PostgreSQL not running**:
   ```bash
   # Start PostgreSQL
   sudo systemctl start postgresql
   # Create database
   createdb lucidota_state
   ```

### Debugging

```bash
# Verbose Make output
make VERBOSE=1 all

# Dry run with Just
just --dry-run all

# Check environment variables
make status
```

## Build Report

Generate a comprehensive build report:

```bash
# Using Python directly
python3 build_automation/build_report.py

# Using Just
just build-report
```

The report includes:
- Git information (commit, branch, status)
- Python environment status
- Dependency checks
- Project statistics
- Available build targets
- CI/CD pipeline information
- Health check requirements

## Contributing

When adding new build functionality:

1. **Follow the Blueprint-First Law**: Explicit build paths over hidden workflows
2. **Respect Dev Library Reuse**: Check for existing tools before creating new ones
3. **Provide receipts**: Build outputs should be verifiable
4. **Add to all interfaces**: Update Makefile, Justfile, and CI/CD workflow
5. **Document**: Update this README with new targets and usage

## License

This build automation system is part of the LUCIDOTA project and follows the same licensing terms.

## References

- [LUCIDOTA Dev Library Reuse Law](00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md)
- [Blueprint-First / PocketFlow Pseudolaw](00_PROJECT_BRAIN/BLUEPRINT_FIRST_MODEL_SECOND_PSEUDOLAW.md)
- [TICKLETRUNK Manifest](00_PROJECT_BRAIN/TICKLETRUNK.md)
- [AGENTS.md](../AGENTS.md) - Agent operation guidelines
