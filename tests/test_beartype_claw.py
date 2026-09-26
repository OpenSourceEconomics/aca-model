"""The beartype claw is live on the `aca_model` package.

Registering `beartype_package("aca_model", ...)` in `aca_model/__init__.py`
instruments every `aca_model` module at import time, so a type violation in
any aca_model function — including the numerical DAG leaf functions fed into
pylcm — is caught at the call boundary rather than slipping through against
a dishonest annotation.

The test calls a real model-builder with one argument of the wrong type; the
`BeartypeCallHintViolation` is what proves the claw is installed.
"""

import pytest
from beartype.roar import BeartypeCallHintViolation

from aca_model.benchmark import create_benchmark_model


def test_claw_checks_aca_model() -> None:
    """An ill-typed argument to an `aca_model` function is rejected by beartype.

    The benchmark factory requires a DiscreteGrid for preference types.
    Invalid types are rejected before model construction.
    """
    with pytest.raises(BeartypeCallHintViolation):
        create_benchmark_model(pref_type_grid="not a grid")  # ty: ignore[invalid-argument-type]
