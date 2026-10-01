"""Economic-grid configuration rejects hardware execution controls."""

import pytest

from aca_model.config import GridConfig


@pytest.mark.parametrize(
    "field",
    [
        "n_assets_batch_size",
        "n_aime_batch_size",
        "pref_type_distributed",
        "spousal_income_distributed",
        "n_health_batch_size",
        "n_spousal_income_batch_size",
        "n_lagged_labor_supply_batch_size",
        "n_claimed_ss_batch_size",
        "n_pref_type_batch_size",
        "n_wage_res_batch_size",
        "n_savings_batch_size",
        "n_stochastic_node_batch_size",
        "n_nbegm_stochastic_node_batch_size",
        "n_nbegm_envelope_segment_block_size",
        "n_nbegm_interval_batch_size",
        "n_nbegm_max_device_workspace_bytes",
        "n_nbegm_cell_block_size",
        "n_nbegm_branch_batch_size",
    ],
)
def test_grid_config_refuses_execution_controls(field):
    """A misplaced execution control fails visibly instead of being ignored."""
    value = True if field.endswith("distributed") else 1
    with pytest.raises(TypeError, match=field):
        GridConfig(**{field: value})  # ty: ignore[invalid-argument-type]
