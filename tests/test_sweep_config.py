"""Expansion of a sweep configuration into cases.

The table asks one method to appear on more than one row -- LDF with and
without the final projection -- and one problem to appear under more than one
scene. Both are config-level ideas that never reach a generator, so they are
checked here rather than by running anything.
"""

import json

import pytest

from cfm import sweep


def _rows(cases, problem):
    return [c for c in cases if c.problem == problem]


def _ldf_weights(cases):
    """The penalty weight each LDF case runs at, keyed by step count."""
    return {(c.steps, c.gains["penalty_weight"])
            for c in cases if c.method == "ldf"}


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
    assert ("Walker2D (medium-expert)",
            "CBF safety filter (SafeFlow)") in rows
    assert ("Hopper (medium-expert)", "LDF + projection") in rows
    assert ("Star (inequality)", "CBF safety filter (SafeFlow)") in rows
    # The star appears under both of its constraints, at the same two
    # resolutions and the same penalty weights, so the two blocks differ in
    # the constraint and nothing else.
    star = _rows(cases, "star")
    equality = [c for c in star if not c.problem_options]
    inequality = [c for c in star if c.problem_options]
    assert all(c.problem_options == {"constraint": "right_half"}
               for c in inequality)
    assert {c.steps for c in inequality} == {10, 100}
    assert _ldf_weights(inequality) == _ldf_weights(equality)
    # PCFM and PiGDM are equality-only and drop out of the inequality block.
    assert {c.method for c in inequality} == {"cbf", "ldf"}
    # Every obstacle case runs at dt = 0.002 and every CBF one relaxes the QP.
    obstacles = _rows(cases, "obstacles")
    assert {c.steps for c in obstacles} == {500}
    assert all(c.gains["qp"] == "elastic"
               for c in obstacles if c.method == "cbf")
    # The locomotion rows run at the example's dt = 0.01, and relax the QP
    # too: the exact solve fails part-way through a Hopper sweep.
    locomotion = _rows(cases, "walker2d") + _rows(cases, "hopper")
    assert {c.steps for c in locomotion} == {100}
    assert all(c.gains["qp"] == "elastic"
               for c in locomotion if c.method == "cbf")


def _baseline(tmp_path, params, mean_violation=1e-3, mean_time_ms=10.0):
    """A one-entry baseline file for the star, as the recorder wrote them.

    Keyed ``star/ours``, since the baseline predates the ours -> ldf rename
    and ``compare_to_baseline`` maps the name back before looking it up.
    """
    path = tmp_path / "baseline.json"
    path.write_text(json.dumps({"results": {"star/ours": {
        "mean_violation": mean_violation,
        "mean_time_ms": mean_time_ms,
        "params": params,
    }}}))
    return path


def _record(**overrides):
    record = {
        "problem": "star", "method": "ldf", "steps": 100,
        "mean_violation": 1.0, "mean_time_ms": 10.0,
        "config": {}, "problem_options": {},
    }
    record.update(overrides)
    return record


def test_baseline_ignores_a_scenario_it_never_ran(tmp_path):
    """The star under its inequality is not the star the baseline recorded.

    The baseline keys on problem and method alone, so without this the
    inequality rows would be measured against the unit-norm equality's
    violation and read as a regression.
    """
    path = _baseline(tmp_path, params={"dt": 0.01})
    record = _record(problem_options={"constraint": "right_half"})
    assert sweep.compare_to_baseline([record], path) == []


def test_baseline_still_compares_the_scenario_it_did_run(tmp_path):
    path = _baseline(tmp_path, params={"num_obstacles": 2, "scene_seed": 0})
    record = _record(problem_options={"num_obstacles": 2, "scene_seed": 0})
    assert sweep.compare_to_baseline([record], path) == [
        "star/ours: violation 1.000e+00 > baseline 1.000e-03"
    ]


def test_baseline_ignores_a_scene_it_never_ran(tmp_path):
    """The crowded obstacle scene is not the one the baseline recorded."""
    path = _baseline(tmp_path, params={"num_obstacles": 2, "scene_seed": 0})
    record = _record(problem_options={"num_obstacles": 6, "scene_seed": 0})
    assert sweep.compare_to_baseline([record], path) == []
