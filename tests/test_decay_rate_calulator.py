import math
import pytest
from tic_tac_learn.control.learning_rate_decay.decay_rate_calulator import (e_decay,
                                                                            linear_decay)

def test_e_decay_step_zero_returns_starting_rate():
    assert e_decay(0, 0.75, 5.0) == pytest.approx(0.75)


@pytest.mark.parametrize(
    "step, starting_lr, decay_rate",
    [
        (3, 1.0, -0.5),
        (2, 0.5, 0.1),
        (10, 0.2, 0.3),
        (5, 0.01, -0.01),
    ],
)
def test_e_decay_matches_math_exp(step, starting_lr, decay_rate):
    # e_decay always uses a negative exponent, so the sign of decay_rate doesn't matter
    expected = float(starting_lr * math.exp(-abs(decay_rate) * step))
    assert e_decay(step, starting_lr, decay_rate) == pytest.approx(expected)


@pytest.mark.parametrize("decay_rate", [0.1, 0.5, 2.0])
def test_e_decay_positive_rate_still_decays(decay_rate):
    starting_lr = 0.8
    rates = [e_decay(step, starting_lr, decay_rate) for step in range(5)]
    assert rates[0] == pytest.approx(starting_lr)
    assert all(later < earlier for earlier, later in zip(rates, rates[1:]))
    assert e_decay(3, starting_lr, decay_rate) == pytest.approx(e_decay(3, starting_lr, -decay_rate))


def test_e_decay_returns_float_type():
    result = e_decay(1, 0.33, 0.1)
    assert isinstance(result, float)