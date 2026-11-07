from benchmark.continuous_problem import ContinuousProblem
import pytest

class DummyContinuousProblem(ContinuousProblem):

    def evaluate(self, candidate):
        return 0.0

    def random_solution(self, rng):
        return []

def test_valid_candidate():
    prob = DummyContinuousProblem(3, -5.12, 5.12)
    assert prob.validate([1.0, 2.0, 3.0])

def test_valid_candidate_at_bounds():
    prob = DummyContinuousProblem(3, -5.12, 5.12)
    assert prob.validate([0.0, -5.12, 5.12])

def test_invalid_candidate_out_of_bounds():
    prob = DummyContinuousProblem(3, -5.12, 5.12)
    assert not prob.validate([1.0, 2.0, 7.0])

def test_invalid_candidate_out_of_bounds_with_negative_value():
    prob = DummyContinuousProblem(3, -5.12, 5.12)
    assert not prob.validate([1.0, 2.0, -7.0])

def test_invalid_candidate_dimension():
    prob = DummyContinuousProblem(3, -5.12, 5.12)
    assert not prob.validate([1.0, 2.0])

def test_cannot_instantiate_abstract_continuous_problem():
    with pytest.raises(TypeError):
        ContinuousProblem(3, -5.12, 5.12)