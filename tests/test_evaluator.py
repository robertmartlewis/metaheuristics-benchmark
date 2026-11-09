from benchmark.evaluator import Evaluator
from benchmark.problem import Problem
import pytest

class DummyProblem(Problem):

    def validate(self, candidate):
        return True

    def evaluate(self, candidate):
        return candidate[0] ** 2

    def random_solution(self, rng):
        return [0.0]

def test_evaluator_count_calls():
    evaluator = Evaluator(DummyProblem(), max_evaluations=3)
    assert evaluator.evaluations == 0
    evaluation1 = evaluator.evaluate([2.0])
    assert evaluator.evaluations == 1
    assert evaluation1 == 4.0
    evaluation2 = evaluator.evaluate([3.0])
    assert evaluator.evaluations == 2
    assert evaluation2 == 9.0

def test_evaluator_budget_limit():
    evaluator = Evaluator(DummyProblem(), max_evaluations=3)

    evaluator.evaluate([1.0])
    evaluator.evaluate([2.0])
    evaluator.evaluate([3.0])

    with pytest.raises(RuntimeError):
        evaluator.evaluate([4.0])

    assert evaluator.evaluations == 3

def test_evaluator_rejects_invalid_budget():
    with pytest.raises(ValueError):
        Evaluator(DummyProblem(), max_evaluations=0)
    with pytest.raises(ValueError):
        Evaluator(DummyProblem(), max_evaluations=-10)

def test_evaluator_rejects_invalid_budget_type():
    with pytest.raises(TypeError):
        Evaluator(DummyProblem(), max_evaluations=3.5)
    with pytest.raises(TypeError):
        Evaluator(DummyProblem(), max_evaluations=True)
    with pytest.raises(TypeError):
        Evaluator(DummyProblem(), max_evaluations="100")

def test_evaluator_tracks_best_fitness():
    evaluator = Evaluator(DummyProblem(), max_evaluations=4)

    assert evaluator.best_fitness == float("inf")

    evaluator.evaluate([3.0])
    assert evaluator.best_fitness == 9.0

    evaluator.evaluate([2.0])
    assert evaluator.best_fitness == 4.0

    evaluator.evaluate([4.0])
    assert evaluator.best_fitness == 4.0

    evaluator.evaluate([1.0])
    assert evaluator.best_fitness == 1.0

def test_evaluator_rejects_nan_fitness():
    evaluator = Evaluator(DummyProblem(), max_evaluations=4)

    with pytest.raises(ValueError):
        evaluator.evaluate([float("nan")])

    assert evaluator.evaluations == 1
    assert evaluator.best_fitness == float("inf")