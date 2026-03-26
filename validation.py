import subprocess
import sys
from pathlib import Path

import pytest


def validate_openenv_yaml():
    path = Path("openenv.yaml")
    assert path.exists(), "openenv.yaml is missing"


def validate_docker_build():
    cmd = ["docker", "build", "-t", "openenv-email-test", "."]
    print("Running docker build...")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    assert proc.returncode == 0, f"docker build failed: {proc.stderr}"
    print("docker build successful")


def run_pytests():
    result = pytest.main(["-q", "tests"])
    assert result == 0, "pytest tests failed"


def main():
    print("Validating openenv metadata...")
    validate_openenv_yaml()

    print("Running pytest suite...")
    run_pytests()

    print("Running docker build check...")
    try:
        validate_docker_build()
    except AssertionError as e:
        print("Docker build check failed (possible missing docker runtime in environment).", e)
        raise

    print("All validations passed")


if __name__ == "__main__":
    main()
