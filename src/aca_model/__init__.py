import jax

jax.config.update("jax_enable_x64", True)

# Import lcm before installing the claw so its `_jaxtyping_patch` (picklable
# jaxtyping sentinel) and `MappingProxyType` pytree registration are in place.
import lcm  # noqa: E402, F401

# Install beartype's AST-rewriting claw on the whole `aca_model` package before
# any submodule is imported. The claw transforms each module's AST at first
# import to insert runtime type checks against its annotations; aca_model's
# numerical DAG/transition/utility functions are otherwise unchecked, since
# pylcm's own claw is scoped to `lcm.*`. Violations surface as beartype's
# `BeartypeCallHintViolation` — aca_model is an application, not a library with
# a documented exception contract.
from beartype import BeartypeConf, BeartypeStrategy  # noqa: E402
from beartype.claw import beartype_package  # noqa: E402

beartype_package(
    "aca_model",
    conf=BeartypeConf(strategy=BeartypeStrategy.On, is_pep484_tower=True),
)
