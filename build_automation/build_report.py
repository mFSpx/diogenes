#!/usr/bin/env python3
"""
LUCIDOTA Build Report Generator
=================================
Generates a comprehensive build report for the LUCIDOTA project.
"""

import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ROOT_DIR = Path(__file__).parent.parent.absolute()
BUILD_DIR = ROOT_DIR / "build_automation"
REPORT_FILE = BUILD_DIR / "report.md"


def get_git_info() -> Dict[str, str]:
    """Get Git repository information."""
    info = {}
    try:
        info["commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["branch"] = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
        info["status"] = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL
        ).decode().strip()
        if not info["status"]:
            info["status"] = "clean"
    except Exception:
        info["commit"] = "unknown"
        info["branch"] = "unknown"
        info["status"] = "unknown"
    return info


def check_python_environment() -> Dict[str, Any]:
    """Check Python environment status."""
    venv_path = ROOT_DIR / ".venv"
    result = {
        "venv_exists": venv_path.exists(),
        "python_version": None,
        "pip_packages": [],
    }
    
    if venv_path.exists():
        python_exe = str(venv_path / "bin" / "python")
        try:
            result["python_version"] = subprocess.check_output(
                [python_exe, "--version"]
            ).decode().strip()
            
            # Get installed packages
            output = subprocess.check_output(
                [str(venv_path / "bin" / "pip"), "list", "--format=json"]
            ).decode()
            packages = json.loads(output)
            result["pip_packages"] = [p["name"] for p in packages]
        except Exception:
            pass
    
    return result


def check_dependencies() -> Dict[str, bool]:
    """Check if core dependencies are installed."""
    requirements_file = ROOT_DIR / "requirements-runtime.txt"
    deps = {
        "dbos": False,
        "river": False,
        "transformers": False,
        "accelerate": False,
        "peft": False,
        "safetensors": False,
    }
    
    if requirements_file.exists():
        venv_path = ROOT_DIR / ".venv"
        if venv_path.exists():
            try:
                output = subprocess.check_output(
                    [str(venv_path / "bin" / "pip"), "list", "--format=freeze"]
                ).decode().lower()
                for dep in deps:
                    deps[dep] = dep.lower() in output
            except Exception:
                pass
    
    return deps


def count_files(directory: Path, extensions: List[str]) -> int:
    """Count files with given extensions in a directory."""
    count = 0
    for ext in extensions:
        count += len(list(directory.rglob(f"*.{ext}")))
    return count


def check_project_stats() -> Dict[str, int]:
    """Get project statistics."""
    return {
        "python_scripts": count_files(ROOT_DIR / "scripts", ["py"]),
        "tests": count_files(ROOT_DIR / "tests", ["py"]),
        "repos": count_files(ROOT_DIR / "01_REPOS", []),
        "algos": count_files(ROOT_DIR / "ALGOS", ["py"]),
    }


def check_docker_status() -> Dict[str, bool]:
    """Check Docker status."""
    phantom_dir = ROOT_DIR / "phantom"
    dockerfile = phantom_dir / "Dockerfile"
    docker_compose = phantom_dir / "docker-compose.yaml"
    
    return {
        "dockerfile_exists": dockerfile.exists(),
        "docker_compose_exists": docker_compose.exists(),
        "docker_running": False,
    }


def check_build_stamps() -> Dict[str, bool]:
    """Check if build stamps exist."""
    return {
        "env_stamp": (BUILD_DIR / ".env-stamp").exists(),
        "deps_stamp": (BUILD_DIR / ".deps-stamp").exists(),
    }


def run_safe_command(cmd: List[str], cwd: Optional[Path] = None) -> Optional[str]:
    """Run a command safely and return output or None."""
    try:
        result = subprocess.check_output(cmd, cwd=cwd, stderr=subprocess.DEVNULL, timeout=10)
        return result.decode().strip()
    except Exception:
        return None


def generate_report() -> str:
    """Generate the build report in markdown format."""
    git_info = get_git_info()
    python_env = check_python_environment()
    dependencies = check_dependencies()
    stats = check_project_stats()
    docker = check_docker_status()
    stamps = check_build_stamps()
    
    timestamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    
    # Header
    report = f"""# LUCIDOTA Build Report

**Generated:** {timestamp}  
**Commit:** {git_info.get('commit', 'unknown')}  
**Branch:** {git_info.get('branch', 'unknown')}  
**Status:** {git_info.get('status', 'unknown')}

---

## 📊 Build System Status

| Component | Status | Details |
|-----------|--------|---------|
| Python Environment | {'✅' if python_env['venv_exists'] else '❌'} | {python_env['python_version'] or 'Not created'} |
| Dependencies | {'✅' if all(dependencies.values()) else '⚠️'} | Core deps: {sum(dependencies.values())}/{len(dependencies)} |
| Build Stamps | {'✅' if stamps['env_stamp'] and stamps['deps_stamp'] else '❌'} | ENV: {stamps['env_stamp']}, DEPS: {stamps['deps_stamp']} |
| Docker | {'✅' if docker['dockerfile_exists'] else '❌'} | Dockerfile: {docker['dockerfile_exists']} |

---

## 🔧 Dependencies Check

| Package | Installed |
|---------|-----------|
"""
    
    for dep, installed in dependencies.items():
        status = "✅" if installed else "❌"
        report += f"| {dep} | {status} |\n"
    
    report += """\n---

## 📈 Project Statistics

| Category | Count |
|----------|-------|
| Python Scripts | {stats['python_scripts']} |
| Tests | {stats['tests']} |
| External Repos | {stats['repos']} |
| Algorithms | {stats['algos']} |

---

## 🏗️ Build Targets

### Available Makefile Targets
```bash
# Core
make all           # Full build (env + deps + build + test)
make dev           # Development setup
make build         # Build all components
make test          # Run all tests
make smoke         # Run smoke tests
make clean         # Clean build artifacts

# Environment
make env           # Setup Python virtual environment
make deps          # Install all Python dependencies
make check-env     # Check if environment is set up
make check-deps    # Check if dependencies are installed
make status        # Show build system status

# Build Components
make build-python  # Build Python components
make build-cuda    # Build CUDA components (llama.cpp)
make build-rust    # Build Rust components (CLAW)

# Docker
make docker        # Build all Docker images
make docker-phantom # Build Phantom Docker image

# Testing
make test-python   # Run Python tests
make test-kernel    # Run Kernel tests
make test-claw      # Run CLAW tests
make smoke          # Run all smoke tests
make diogenes-check # Full Diogenes system check

# Quality
make lint          # Run linters
make format        # Format code

# Clean
make clean          # Clean all build artifacts
make clean-python   # Clean Python artifacts
make clean-cuda     # Clean CUDA build artifacts
make clean-docker   # Clean Docker artifacts
```

### Available Justfile Targets
```bash
# Same targets as Makefile, with improved syntax
just all           # Full build
just dev           # Development setup
just build         # Build all components
just test          # Run all tests
just docker        # Build Docker images
just clean         # Clean build artifacts

# Additional features
just status        # Show build system status
just version       # Show version info
just run-phantom   # Run Phantom Docker container
just stop-phantom  # Stop Phantom Docker container

# Dev Library
just dev-library-scan          # Scan the Dev Library
just dev-library-query query=x # Query the Dev Library

# Utilities
just check-env     # Check environment
just check-deps    # Check dependencies
just lint          # Run linters
just format        # Format code
```

---

## 🚀 CI/CD Pipeline

The GitHub Actions workflow `.github/workflows/lucidota-build.yml` provides:

- **Push to main/diogenes/main**: Runs full build pipeline
- **Pull Requests**: Runs core tests and validation
- **Manual Trigger**: Full build with CUDA support
- **Matrix Builds**: Python 3.12, Ubuntu latest
- **Artifact Caching**: Pip and Cargo caches for faster builds

### CI Jobs
1. **build-config**: Determines build matrix based on event
2. **build-core**: Python environment, dependencies, core tests
3. **build-cuda**: CUDA builds (llama.cpp) - requires GPU runner
4. **build-rust**: Rust builds (CLAW)
5. **build-docker**: Phantom Docker image build and push
6. **integration-test**: Full system integration tests
7. **docs**: Build report generation

---

## 🔍 Health Checks

### Database Requirements
- PostgreSQL 15+ with AGE extension
- Connection URL: `postgresql://mfspx@/lucidota_state`

### Required Tools
- Python {sys.version.split()[0]}+
- pip
- venv
- make
- cmake (for CUDA builds)
- ninja (for CUDA builds)
- ccache (for CUDA builds)
- Rust toolchain (for CLAW)
- Docker and Docker Compose (for Phantom)
- CUDA Toolkit (for GPU builds)

---

## 📝 Notes

- This report is generated by `build_automation/build_report.py`
- The build system respects LUCIDOTA Dev Library Reuse Law
- All sovereign artifacts in the proof hoard are preserved
- See `00_PROJECT_BRAIN/ACTIVE_SPEC/04_DEV_LIBRARY_REUSE_LAW.md` for reuse guidelines
"""
    
    return report


def main() -> None:
    """Main entry point."""
    print("Generating LUCIDOTA build report...")
    
    # Create build directory
    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Generate report
    report = generate_report()
    
    # Write report to file
    REPORT_FILE.write_text(report)
    print(f"Report written to: {REPORT_FILE}")
    
    # Also print to stdout
    print("\n" + report)


if __name__ == "__main__":
    main()
