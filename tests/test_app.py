import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import math
import pytest


def test_sigmoid_midpoint():
    from app import _sigmoid
    assert abs(_sigmoid(0.0) - 0.5) < 1e-6


def test_sigmoid_positive():
    from app import _sigmoid
    assert _sigmoid(2.0) > 0.5
    assert _sigmoid(2.0) < 1.0


def test_sigmoid_negative():
    from app import _sigmoid
    assert _sigmoid(-2.0) < 0.5
    assert _sigmoid(-2.0) > 0.0
