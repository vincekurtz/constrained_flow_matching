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


def _ldf_weights(cases, method="ldf"):
    """The penalty weight each LDF-family case runs at, keyed by step count."""
    return {(c.steps, c.gains["penalty_weight"])
            for c in cases if c.method == method}


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


def test_renaming_a_problem_by_string_needs_a_block_of_one():
    with pytest.raises(ValueError, match="only one problem"):
        sweep.build_cases({
            "block": [{
                "problems": ["star", "mnist"], "methods": ["ldf"],
                "steps": [100], "problem_label": "ambiguous",
            }],
        })


def test_a_table_problem_label_renames_each_problem():
    cases = sweep.build_cases({
        "block": [{
            "problems": ["star", "mnist"], "methods": ["ldf"],
            "steps": [100], "problem_label": {"star": "Star (eq.)"},
        }],
    })
    labels = {c.problem: c.problem_label for c in cases}
    # A problem the table leaves out keeps its registered label.
    assert labels == {"star": "Star (eq.)", "mnist": "MNIST"}


def test_the_shipped_table1_config_expands():
    """The config the paper's table is built from stays loadable."""
    cases = sweep.build_cases(
        sweep.load_config("experiments/table1.toml")
    )
    rows = {(c.problem_label, c.row_label) for c in cases}
    assert ("Star (eq.)", "LDF (no projection)") in rows
    assert ("Star (eq.)", "LDF + projection") in rows
    assert ("MNIST", "Penalty only") in rows
    assert ("Obstacles (2)", "LDF + projection") in rows
    assert ("Obstacles (6)", "CBF safety filter (SafeFlow)") in rows
    assert ("Walker2D", "CBF safety filter (SafeFlow)") in rows
    assert ("Hopper", "LDF + projection") in rows
    assert ("Star (ineq.)", "CBF safety filter (SafeFlow)") in rows
    # Every scenario runs at one resolution, since the table has no Steps
    # column to tell two apart.
    assert len({(c.problem_label, c.name) for c in cases}) == len(cases)
    # The star appears under both of its constraints, at the same
    # resolution and the same penalty weight, so the two blocks differ in
    # the constraint and nothing else.
    star = _rows(cases, "star")
    equality = [c for c in star if not c.problem_options]
    inequality = [c for c in star if c.problem_options]
    assert all(c.problem_options == {"constraint": "right_half"}
               for c in inequality)
    assert {c.steps for c in star} == {100}
    assert _ldf_weights(inequality) == _ldf_weights(equality)
    # The penalty-only ablation runs on the inequality at LDF's weights.
    assert ("Star (ineq.)", "Penalty only") in rows
    assert (_ldf_weights(inequality, "penalty")
            == _ldf_weights(inequality))
    # PCFM and PiGDM are equality-only and drop out of the inequality block.
    assert {c.method for c in inequality} == {"cbf", "penalty", "ldf"}
    # The penalty-only ablation runs on every inequality problem, at the
    # penalty weight LDF runs at there.
    for problem in ("obstacles", "walker2d", "hopper"):
        weights = {c.method: c.gains["penalty_weight"]
                   for c in _rows(cases, problem)
                   if c.method in ("ldf", "penalty")}
        assert weights["penalty"] == weights["ldf"]
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


def _table_record(variant, time_ms, violation, problem_options=None,
                  num_nan=0, label="Star", steps=100):
    return {
        "problem": "star", "problem_label": label,
        "problem_options": problem_options or {},
        "method": variant.removesuffix("_projected"), "variant": variant,
        "method_label": variant, "steps": steps,
        "mean_time_ms": time_ms, "mean_violation": violation,
        "num_nan": num_nan,
    }


def test_latex_table_pivots_methods_into_column_pairs():
    """One row per scenario, equality above inequality, bolded by rule."""
    ineq = {"constraint": "right_half"}
    records = [
        _table_record("cbf", 318.0, 0.0, ineq, num_nan=1,
                      label="Star (ineq.)"),
        _table_record("ldf", 7.7, 2.6e-3),
        _table_record("pcfm", 29.6, 4.5e-8),
        _table_record("ldf_projected", 7.9, 0.0, ineq,
                      label="Star (ineq.)"),
    ]
    lines = sweep.render_latex_table(records).splitlines()
    rows = [l for l in lines if l.startswith("Star")]
    assert rows == [
        r"Star & --- & --- & --- & --- & 29.6 & \scib{4.5}{-8} & --- & --- "
        r"& \textbf{7.7} & \sci{2.6}{-3} & --- & --- \\",
        r"Star (ineq.) & --- & --- & 318.0 & \textbf{0}~$\dagger$ "
        r"& --- & --- & --- & --- & --- & --- & \textbf{7.9} & \textbf{0} \\",
    ]
    # The two halves are split by a rule.
    assert lines.index(rows[1]) == lines.index(rows[0]) + 2


def test_latex_table_refuses_a_scenario_at_two_step_counts():
    records = [_table_record("ldf", 1.0, 1e-2, steps=10),
               _table_record("ldf", 8.0, 1e-3, steps=100)]
    with pytest.raises(ValueError, match="more than one step count"):
        sweep.render_latex_table(records)


def test_latex_table_refuses_two_results_for_one_cell():
    records = [_table_record("ldf", 1.0, 1e-2), _table_record("ldf", 2.0, 1e-2)]
    with pytest.raises(ValueError, match="stale"):
        sweep.render_latex_table(records)
