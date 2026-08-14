"""Cost ledger of splitting marital status out of the state space.

Moving marital status from a three-code state onto the regime axis leaves the
work per period unchanged but redistributes it: twice as many regimes, each
carrying a smaller product of state cardinalities. These are the exact
identities the resource claims rest on, so they are asserted with `Fraction`
arithmetic rather than floats — a ledger that only holds approximately is not
a ledger.

The measured quantities (device memory, solve and compile wall time) come from
the benchmark run; what is pinned here is the arithmetic those measurements are
read against.
"""

from fractions import Fraction

from lcm import DiscreteGrid

from aca_model.agent.labor_market import SpousalIncome
from aca_model.baseline.regimes import REGIME_SPECS
from aca_model.baseline.regimes._common import MARITAL_STATUSES

# Cardinality of the pre-split `spousal_income` state, which every living
# regime carried: `(single, married_no_inc, married_has_inc)`.
_PRE_SPLIT_SPOUSAL_CODES = 3

_WITHIN_MARRIAGE_CODES = len(DiscreteGrid(SpousalIncome).categories)
_LIVING_AXES = len(REGIME_SPECS) // len(MARITAL_STATUSES)


def test_the_split_doubles_the_living_regime_count() -> None:
    """Each within-marriage axis combination gets a single and a married copy."""
    assert len(REGIME_SPECS) == 2 * _LIVING_AXES


def test_the_fixed_regime_overhead_term_is_positive() -> None:
    """`T_split - T_base` carries `+(n_split - n_base) * omega`.

    Per-regime overhead is a cost, so doubling the regime count raises it. The
    sign is stated under the new-minus-old convention and is positive there;
    the total delta may still be negative, because constant folding can lower
    the per-cell cost, but the two effects must be reported separately rather
    than netted.
    """
    omega = Fraction(7, 13)  # any positive per-regime overhead
    delta = (len(REGIME_SPECS) - _LIVING_AXES) * omega
    assert delta == _LIVING_AXES * omega
    assert delta > 0


def test_total_state_cells_are_conserved() -> None:
    """Summed over both copies, a regime spans the same cells as before.

    The single copy carries no `spousal_income` and the married copy carries
    the two within-marriage codes, so `1 + 2` replaces the original `3`.
    """
    per_axis_before = Fraction(_PRE_SPLIT_SPOUSAL_CODES)
    per_axis_after = Fraction(1 + _WITHIN_MARRIAGE_CODES)
    assert per_axis_after == per_axis_before


def test_the_largest_regime_loses_a_third_of_its_cells() -> None:
    """The binding regime's `Q` transient shrinks by exactly `3/2`.

    The largest regime is the married copy, whose `spousal_income` axis is two
    codes wide where the pre-split regime's was three. This is a per-regime
    cell claim, not a claim about total peak device memory: retained
    continuation storage is unaffected.
    """
    headroom = Fraction(_PRE_SPLIT_SPOUSAL_CODES, _WITHIN_MARRIAGE_CODES)
    assert headroom == Fraction(3, 2)


def test_continuation_node_count_is_unchanged() -> None:
    """A source still reaches four continuation nodes per within-marriage target.

    Before: one living target carrying three spousal codes, plus `dead`.
    After: a single target with no spousal axis, a married target with two
    codes, plus `dead`.
    """
    before = _PRE_SPLIT_SPOUSAL_CODES + 1
    after = 1 + _WITHIN_MARRIAGE_CODES + 1
    assert after == before == 4
