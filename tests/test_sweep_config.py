"""Expansion of a sweep configuration into cases.

The table asks one method to appear on more than one row -- LDF with and
without the final projection -- and one problem to appear under more than one
scene. Both are config-level ideas that never reach a generator, so they are
checked here rather than by running anything.
"""

import pytest

from cfm import sweep


def _rows(cases, problem):
    return [c for c in cases if c.problem == problem]


def test_flat_config_still_expands():
    """A config with no [[block]] is itself the single block."""
    cases = sweep.build_cases({
        "problems": ["star"], "methods": ["ldf"], "steps": [10, 100],
    })
    assert [c.steps for c in cases] == [10, 100]
    assert all(c.variant == "ldf" and c.method == "ldf" for c in cases)


def test_variants_give_one_method_two_rows():
    cases = sweep.build_cases({
        "variants": [
            {"name": "ldf", "label": "LDF (no projection)",
             "gains": {"num_projection_iters": 0}},
            {"name": "ldf_projected", "method": "ldf",
             "label": "LDF + projection",
             "gains": {"num_projection_iters": 2}},
        ],
        "problems": ["star"], "methods": ["ldf", "ldf_projected"],
        "steps": [100],
    })
    assert [c.method for c in cases] == ["ldf", "ldf"]
    assert [c.gains["num_projection_iters"] for c in cases] == [0, 2]
    assert len({c.key for c in cases}) == 2, "rows must not share a file"
    assert [c.row_label for c in cases] == [
        "LDF (no projection)", "LDF + projection",
    ]


def test_variant_gains_survive_an_override_aimed_at_the_method():
    """A gain aimed at `ldf` tunes both rows without collapsing them."""
    cases = sweep.build_cases({
        "variants": [
            {"name": "ldf", "gains": {"num_projection_iters": 0}},
            {"name": "ldf_projected", "method": "ldf",
             "gains": {"num_projection_iters": 2}},
        ],
        "gains": [{"method": "ldf", "penalty_weight": 7.0,
                   "num_projection_iters": 99}],
        "problems": ["star"], "methods": ["ldf", "ldf_projected"],
        "steps": [100],
    })
    assert all(c.gains["penalty_weight"] == 7.0 for c in cases)
    assert [c.gains["num_projection_iters"] for c in cases] == [0, 2]


def test_a_gain_selector_may_name_several_methods():
    cases = sweep.build_cases({
        "gains": [{"method": ["ldf", "penalty"], "penalty_weight": 3.0}],
        "problems": ["star"], "methods": ["ldf", "penalty", "pigdm"],
        "steps": [100],
    })
    weights = {c.method: c.gains.get("penalty_weight") for c in cases}
    assert weights == {"ldf": 3.0, "penalty": 3.0, "pigdm": None}


def test_blocks_carry_their_own_grid_and_scene():
    cases = sweep.build_cases({
        "num_samples": 20,
        "block": [
            {"problems": ["star"], "methods": ["ldf"], "steps": [100]},
            {"problems": ["obstacles"], "methods": ["ldf"], "steps": [500],
             "problem_label": "Obstacle avoidance (6 obstacles)",
             "num_samples": 5,
             "problem_options": {"obstacles": {"num_obstacles": 6}}},
        ],
    })
    star, obstacles = _rows(cases, "star"), _rows(cases, "obstacles")
    assert len(star) == len(obstacles) == 1
    assert star[0].num_samples == 20 and obstacles[0].num_samples == 5
    assert obstacles[0].steps == 500
    assert obstacles[0].problem_options == {"num_obstacles": 6}
    assert obstacles[0].problem_label == "Obstacle avoidance (6 obstacles)"
    assert star[0].problem_label == "Star"


def test_the_same_problem_under_two_scenes_gets_two_files():
    cases = sweep.build_cases({
        "block": [
            {"problems": ["obstacles"], "methods": ["ldf"], "steps": [500],
             "problem_options": {"obstacles": {"num_obstacles": n}}}
            for n in (2, 6)
        ],
    })
    assert len({c.key for c in cases}) == 2


def test_a_block_only_runs_methods_the_constraint_admits():
    """Obstacle avoidance is an inequality, so PCFM drops out on its own."""
    cases = sweep.build_cases({
        "problems": ["obstacles"], "methods": ["cbf", "pcfm", "ldf"],
        "steps": [500],
    })
    assert [c.method for c in cases] == ["cbf", "ldf"]


def test_renaming_a_problem_needs_a_block_of_one():
    with pytest.raises(ValueError, match="only one problem"):
        sweep.build_cases({
            "block": [{
                "problems": ["star", "mnist"], "methods": ["ldf"],
                "steps": [100], "problem_label": "ambiguous",
            }],
        })


def test_the_shipped_table1_config_expands():
    """The config the paper's table is built from stays loadable."""
    cases = sweep.build_cases(
        sweep.load_config("experiments/table1.toml")
    )
    rows = {(c.problem_label, c.row_label) for c in cases}
    assert ("Star", "LDF (no projection)") in rows
    assert ("Star", "LDF + projection") in rows
    assert ("MNIST", "Penalty only") in rows
    assert ("Obstacle avoidance (2 obstacles)", "LDF + projection") in rows
    assert ("Obstacle avoidance (6 obstacles)",
            "CBF safety filter (SafeFlow)") in rows
    # Every obstacle case runs at dt = 0.002 and every CBF one relaxes the QP.
    obstacles = _rows(cases, "obstacles")
    assert {c.steps for c in obstacles} == {500}
    assert all(c.gains["qp"] == "elastic"
               for c in obstacles if c.method == "cbf")
