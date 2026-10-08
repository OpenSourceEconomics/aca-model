"""ACA function overrides applied on top of baseline regimes.

Replaces baseline functions with ACA-aware versions based on the active
PolicyVariant. Unlike the previous stub-based approach, baseline regimes
have no ACA placeholders — the override adds new DAG nodes and swaps
consuming functions that need them.
"""

from aca_model.aca import health_insurance as aca_hi
from aca_model.aca.health_insurance import PolicyVariant
from aca_model.baseline.regimes._common import RegimeSpec
from aca_model.environment import taxes


def apply_aca_overrides(
    functions: dict,
    spec: RegimeSpec,
    policy: PolicyVariant,
) -> None:
    """Override baseline functions with ACA versions in-place.

    Every variant charges the ACA surtaxes in every regime: the 3.8% surtax
    on unearned income above 200,000, plus, under the full ACA only, the
    0.9% surtax on earnings above 200,000 (struct-ret's composition).

    Three orthogonal feature flags derived from the policy variant:

    - **Medicaid expansion**: two-track eligibility (categorical SSI plus the
      under-65 MAGI expansion) installed on all regimes. The expansion arm is
      internally scoped to the under-65 population, disabled or not, so only
      post-65 households keep the categorical track with its asset test.
    - **Subsidies**: the reformed non-group market (nongroup+nomc only): the
      community-rated plan premium, premium credits, cost-sharing
      reductions, and their consuming functions. Credits and cost-sharing
      mask to their neutral value when Medicaid-eligible (minimum-essential
      coverage).
    - **Mandate**: individual mandate penalty (nongroup+nomc only, requires
      subsidies), waived when Medicaid-eligible.
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

    functions["after_tax_income_before_aca_surtaxes"] = taxes.after_tax_income
    functions["after_tax_income"] = (
        taxes.after_tax_income_with_aca_surtaxes
        if policy == PolicyVariant.ACA
        else taxes.after_tax_income_with_aca_investment_surtax
    )

    if has_medicaid_expansion:
        functions["is_medicaid_eligible"] = aca_hi.is_medicaid_eligible

    if has_subsidies and spec["his"] == "nongroup" and spec["mc"] == "nomc":
        if has_mandate:
            functions["mandate_penalty"] = aca_hi.mandate_penalty
        # No mandate: mandate_penalty is a fixed param (0.0) in the params
        # dict, not a DAG function — no entry needed here.
        functions["plan_premium"] = aca_hi.community_rated_premium
        # The community-rated premium replaces the risk-rated one, so the
        # risk-rated premium's zero-profit intercept has no consumer left.
        functions.pop("private_premium_intercept", None)
        functions["hic_premium_subsidy"] = aca_hi.premium_subsidy
        functions["cost_sharing_scale"] = aca_hi.cost_sharing
        functions["premium_default"] = aca_hi.premium_default
        functions["cash_on_hand"] = aca_hi.cash_on_hand
        functions["primary_oop"] = aca_hi.primary_oop
