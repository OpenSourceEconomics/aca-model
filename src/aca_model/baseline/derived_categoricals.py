"""Base-layer derived categoricals shared by baseline + ACA model factories."""

from types import MappingProxyType

from lcm import DiscreteGrid

from aca_model.baseline.health_insurance import HealthInsuranceState

# `target_his` is a state subsumed into regimes.
BASE_DERIVED_CATEGORICALS = MappingProxyType(
    {"target_his": DiscreteGrid(HealthInsuranceState)}
)
