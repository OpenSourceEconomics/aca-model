"""ACA function overrides applied on top of baseline regimes.

Replaces baseline functions with ACA-aware versions based on the active
PolicyVariant. Unlike the previous stub-based approach, baseline regimes
have no ACA placeholders — the override adds new DAG nodes and swaps
consuming functions that need them.
"""

from aca_model.aca import health_insurance as aca_hi
from aca_model.aca.health_insurance import PolicyVariant


def apply_aca_overrides(
    functions: dict,
    spec: dict[str, str],
    policy: PolicyVariant,
) -> None:
    """Override baseline functions with ACA versions in-place.

    Three orthogonal feature flags derived from the policy variant:

    - **Medicaid expansion**: ACA-style income-only eligibility (all regimes).
    - **Subsidies**: premium credits, cost-sharing reductions, and their
      consuming functions (nongroup+nomc only).
    - **Mandate**: individual mandate penalty (nongroup+nomc only, requires
      subsidies).
    """
    has_medicaid_expansion = policy not in (
        PolicyVariant.ACA_NO_MEDICAID_EXPANSION,
        PolicyVariant.ACA_NO_MEDICAID_EXPANSION_NO_MANDATE,
    )
    has_subsidies = policy != PolicyVariant.ACA_ONLY_MEDICAID_EXPANSION
    has_mandate = policy in (
        PolicyVariant.ACA,
        PolicyVariant.ACA_NO_MEDICAID_EXPANSION,
    )

    if has_medicaid_expansion:
        functions["is_medicaid_eligible"] = aca_hi.is_medicaid_eligible

    if has_subsidies and spec["his"] == "nongroup" and spec["mc"] == "nomc":
        if has_mandate:
            functions["mandate_penalty"] = aca_hi.mandate_penalty
        # No mandate: mandate_penalty is a fixed param (0.0) in the params
        # dict, not a DAG function — no entry needed here.
        functions["hic_premium_subsidy"] = aca_hi.premium_subsidy
        functions["cost_sharing_scale"] = aca_hi.cost_sharing
        functions["cash_on_hand"] = aca_hi.cash_on_hand
        functions["primary_oop"] = aca_hi.primary_oop
