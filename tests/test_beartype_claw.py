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
from helpers.model import make_baseline_model  # ty: ignore[unresolved-import]


def test_claw_checks_aca_model() -> None:
    """An ill-typed argument to an `aca_model` function is rejected by beartype.

    `create_model` annotates `n_subjects` as `int`; passing a string is caught
    by the claw before the value reaches pylcm's own `Model` perimeter.
    """
    with pytest.raises(BeartypeCallHintViolation):
        make_baseline_model(n_subjects="not an int")
