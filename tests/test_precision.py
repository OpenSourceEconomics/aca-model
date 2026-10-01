"""Process-level floating-point precision selection."""

import os
import subprocess
import sys


def test_aca_precision_environment_selects_fp32() -> None:
    """`ACA_JAX_ENABLE_X64=0` makes ACA computations use 32-bit floats."""
    env = {
        **os.environ,
        "JAX_ENABLE_X64": "1",
        "ACA_JAX_ENABLE_X64": "0",
    }

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import aca_model, jax; print(jax.config.jax_enable_x64)",
        ],
        check=True,
        capture_output=True,
        env=env,
        text=True,
    )

    assert result.stdout.strip() == "False"
