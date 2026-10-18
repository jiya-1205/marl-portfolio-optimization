import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from app.simulation_runner import (
    SCENARIOS,
    FRICTION_PROFILES,
    check_checkpoint_status,
    run_interactive_simulation,
)


def test_checkpoint_integrity():
    status = check_checkpoint_status()
    assert len(status) == 4, "Expected 4 checkpoints to be tested"
    for name, info in status.items():
        assert info["exists"], f"Checkpoint {name} must exist on disk"
        assert info["verified"], f"Checkpoint {name} must pass forward-pass inference verification"


def test_marl_simulation_run():
    outcome = run_interactive_simulation(
        scenario_key="2023_bull",
        initial_capital=100_000.0,
        friction_key="standard",
        policy_type="marl",
    )
    assert outcome.initial_capital == 100_000.0
    assert outcome.ending_capital > 0.0
    assert len(outcome.first_target_allocation) == 6  # 5 assets + cash
    total_alloc = sum(outcome.first_target_allocation.values())
    assert abs(total_alloc - 1.0) < 1e-4, "Allocations must sum to 100%"
    assert len(outcome.daily_values) > 10, "Should have daily values"
    assert outcome.is_checkpoint_verified is True


def test_all_scenarios_and_policies():
    policies = ["marl", "ppo", "sac", "equal_weight"]
    for p in policies:
        out = run_interactive_simulation(
            scenario_key="2024_test",
            initial_capital=50_000.0,
            friction_key="low",
            policy_type=p,
        )
        assert out.ending_capital > 0
        assert len(out.daily_values) > 5


if __name__ == "__main__":
    print("Running checkpoint integrity test...")
    test_checkpoint_integrity()
    print("Checkpoint integrity test PASSED!")

    print("Running MARL simulation test...")
    test_marl_simulation_run()
    print("MARL simulation test PASSED!")

    print("Running all policies test...")
    test_all_scenarios_and_policies()
    print("All policies test PASSED!")
    print("\nALL BACKEND & UI INTEGRATION TESTS PASSED CLEANLY!")
