"""Economic grids and numerical solver choices for the aca_model package."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

SRC = Path(__file__).parent.resolve()
ROOT = SRC.parents[1]
BLD = ROOT / "bld"


@dataclass(frozen=True)
class ModelConfig:
    start_age: int = 51
    last_start_age: int = 60
    end_age: int = 96
    """First age at which everybody is dead; the model's last period."""
    ss_early_age: int = 62
    ss_forced_age: int = 70
    work_forced_out_age: int = 72
    medicare_age: int = 65
    # Pension benefits must be drawn from this age on; pension wealth no longer
    # accrues from earnings.
    pension_must_receive_age: int = 71


@dataclass(frozen=True)
class GridConfig:
    """Economic resolution and numerical choices, independent of device execution."""

    n_assets_gridpoints: int = 24
    n_aime_gridpoints: int = 12
    # AIME piecewise grid: number of points per segment between the PIA bend
    # points (0 → kink_0 → kink_1 → kink_2 → extension), at least 2 each. The
    # fourth segment covers the sparse delayed-retirement-credit region above
    # the taxable max, so it carries fewer points than the dense lower segments.
    aime_points_per_segment: tuple[int, int, int, int] = (10, 11, 11, 6)
    n_consumption_dollars_gridpoints: int = 70
    n_wage_res_gridpoints: int = 5
    n_hcc_persistent_gridpoints: int = 3
    n_hcc_transitory_gridpoints: int = 5
    # Fold the transitory medical-cost shock out of the state space. Nothing a
    # period computes reads last year's transitory shock, so with the fold it is
    # drawn on the transition only and no value carries its axis.
    fold_hcc_transitory: bool = False
    # Declare the asset law toward living targets as resources minus the
    # out-of-pocket bill, so the solve averages each target's value over the
    # transitory shock once per period instead of at every source point. Needs
    # `fold_hcc_transitory` and the brute-force asset law.
    subtract_hcc_bill: bool = False
    # The post-decision savings grid is cubically clustered toward the borrowing
    # constraint. Its node count controls resolution under DC-EGM and NB-EGM.
    n_savings_gridpoints: int = 200
    # Arithmetic used to compare upper-envelope candidates:
    # - "certified": double-double comparisons publish only separated winners.
    # - "ordinary": comparisons use the working floating-point format.
    nbegm_envelope_arithmetic: Literal["certified", "ordinary"] = "certified"
    # How NB-EGM parents read institutional cliffs in the child value:
    # - "one_sided": duplicated abscissae preserve one-sided limits.
    # - "bridged": interpolation may bridge finite-grid discontinuities.
    # Live labor choices require "bridged" because branch-dependent income
    # thresholds cannot share one query grid of one-sided cliff limits.
    nbegm_jump_read: Literal["one_sided", "bridged"] = "one_sided"


MODEL_CONFIG = ModelConfig()


def model_ages(*, final_age: int) -> tuple[int, ...]:
    """Return the model's ages, from `start_age` through `final_age`.

    `final_age` is the first age at which everybody is dead, so the model has
    `final_age - start_age` nonterminal periods. The age-indexed data cover
    `start_age` through `MODEL_CONFIG.end_age`; a shorter horizon reads a prefix
    of them.

    Raises:
        ValueError: If `final_age` leaves no nonterminal period or lies beyond
            `MODEL_CONFIG.end_age`.

    """
    _fail_if_invalid_final_age(final_age)
    return tuple(range(MODEL_CONFIG.start_age, final_age + 1))


def _fail_if_invalid_final_age(final_age: int) -> None:
    if not MODEL_CONFIG.start_age < final_age <= MODEL_CONFIG.end_age:
        msg = (
            f"final_age must lie in ({MODEL_CONFIG.start_age}, "
            f"{MODEL_CONFIG.end_age}], got {final_age}."
        )
        raise ValueError(msg)


# ACA is calibrated annually. Keep the exact clock on the host until Model build;
# importing a task or regime builder must not initialize a JAX device allocator.
MODEL_AGES: tuple[int, ...] = model_ages(final_age=MODEL_CONFIG.end_age)
GRID_CONFIG = GridConfig()

BENCHMARK_GRID_CONFIG = GridConfig(
    fold_hcc_transitory=True,
    subtract_hcc_bill=True,
    n_assets_gridpoints=3,
    n_aime_gridpoints=3,
    n_consumption_dollars_gridpoints=5,
    n_wage_res_gridpoints=3,
    n_hcc_persistent_gridpoints=3,
    n_hcc_transitory_gridpoints=3,
)
