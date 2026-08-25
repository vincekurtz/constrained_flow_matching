"""Integrity of the method and problem registries.

Cheap checks that catch the kind of typo -- a gain keyed on a method that
does not exist, a problem whose constraint no method can handle -- that would
otherwise surface as a confusing failure halfway through a sweep.
"""

import pytest

from cfm import methods
import problems
from cfm.core.constraints import KINDS


@pytest.mark.parametrize("method", methods.METHODS.values(),
                         ids=lambda m: m.name)
def test_method_is_well_formed(method):
    assert method.name and method.label
    assert callable(method.generate)
    assert method.supports, f"{method.name} supports no constraint kind"
    assert set(method.supports) <= set(KINDS)


def test_method_names_match_their_keys():
    for key, method in methods.METHODS.items():
        assert key == method.name


def test_get_reports_unknown_methods():
    with pytest.raises(KeyError, match="unknown method"):
        methods.get("definitely-not-a-method")


def test_locked_values_appear_in_defaults():
    """A locked parameter must also be the default, or config() contradicts."""
    for method in methods.METHODS.values():
        for key, value in method.locked.items():
            assert method.defaults.get(key) == value, (
                f"{method.name}: locked {key}={value} but default is "
                f"{method.defaults.get(key)!r}"
            )


def test_penalty_is_ldf_with_frozen_multipliers():
    """The ablation must delegate, not reimplement."""
    assert methods.PENALTY.generate is methods.LDF.generate
    assert methods.PENALTY.locked == {"rescale_factor": 0.0}
    assert methods.PENALTY.defaults["rescale_factor"] == 0.0
    # Everything else should match LDF, so the comparison is like-for-like.
    shared = set(methods.LDF.defaults) - {"rescale_factor"}
    for key in shared:
        assert methods.PENALTY.defaults[key] == methods.LDF.defaults[key]


def test_config_applies_overrides_then_locked():
    cfg = methods.LDF.config(penalty_weight=2.0)
    assert cfg["penalty_weight"] == 2.0
    assert cfg["rescale_factor"] == methods.LDF.defaults["rescale_factor"]

    # None means "unset" so argparse defaults do not clobber the registry.
    assert methods.LDF.config(penalty_weight=None)["penalty_weight"] == 5.0


def test_locked_parameter_cannot_be_overridden():
    with pytest.raises(ValueError, match="cannot be overridden"):
        methods.PENALTY.config(rescale_factor=1.0)


def test_locked_parameter_accepts_a_matching_value():
    """Passing the locked value explicitly is harmless, not an error."""
    assert methods.PENALTY.config(rescale_factor=0.0)["rescale_factor"] == 0.0


@pytest.mark.parametrize("name", problems.names())
def test_problem_is_well_formed(name):
    problem = problems.get(name)
    assert problem.label
    assert callable(problem.make_dataset)
    assert callable(problem.make_model)
    assert callable(problem.plot)
    assert problem.checkpoint_path.endswith(".pkl")
    assert problem.train.num_epochs > 0
    assert problem.default_num_samples > 0


@pytest.mark.parametrize("name", problems.names())
def test_problem_gains_name_registered_methods(name):
    """A gain keyed on a nonexistent method would silently never apply."""
    problem = problems.get(name)
    for method_name in problem.method_gains:
        assert method_name in methods.METHODS, (
            f"problem {name!r} has gains for unknown method {method_name!r}"
        )


@pytest.mark.parametrize("name", problems.names())
def test_constrained_problems_have_a_usable_method(name):
    problem = problems.get(name)
    if not problem.constrained:
        pytest.skip(f"{name} is unconstrained")
    constraint = problem.make_constraint()
    assert methods.supporting(constraint), (
        f"no registered method handles {name}'s {constraint.kind} constraint"
    )


@pytest.mark.parametrize("name", problems.names())
def test_gains_are_accepted_by_their_methods(name):
    """Every gain override must survive config() for that method."""
    problem = problems.get(name)
    for method_name, gains in problem.method_gains.items():
        methods.get(method_name).config(**gains)


def test_problem_names_match_their_keys():
    for key, problem in problems.all_problems().items():
        assert key == problem.name


def test_get_reports_unknown_problems():
    with pytest.raises(KeyError, match="unknown problem"):
        problems.get("not-a-problem")
