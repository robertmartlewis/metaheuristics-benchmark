from benchmark.run_result import RunResult
import pytest

def test_run_result_stores_execution_data():
    result = RunResult(
        algorithm="GA",
        problem="Sphere",
        seed=202700,
        best_fitness=1.5,
        evaluations=3,
        best_history=(9.0, 4.0, 1.5)
    )

    assert result.algorithm == "GA"
    assert result.problem == "Sphere"
    assert result.seed == 202700
    assert result.best_fitness == 1.5
    assert result.evaluations == 3
    assert result.best_history == (9.0, 4.0, 1.5)

def test_run_result_preserves_history_snapshot():
    history = [9.0, 4.0, 1.5]
    result = RunResult(
        algorithm="GA",
        problem="Sphere",
        seed=202700,
        best_fitness=1.5,
        evaluations=3,
        best_history=tuple(history)
    )
    history.append(0.5)
    assert result.best_history == (9.0, 4.0, 1.5)
    assert len(history) == 4

def test_run_result_rejects_inconsistent_evaluation_count():
    with pytest.raises(ValueError):
        result = RunResult(
            algorithm="GA",
            problem="Sphere",
            seed=202700,
            best_fitness=1.5,
            evaluations=3,
            best_history=(9.0, 4.0)
        )

def test_run_result_rejects_inconsistent_best_fitness():
    with pytest.raises(ValueError):
        RunResult(
            algorithm="GA",
            problem="Sphere",
            seed=202700,
            best_fitness=1.5,
            evaluations=3,
            best_history=(9.0, 4.0, 2.0)
        )

def test_run_result_rejects_empty_history():
    with pytest.raises(ValueError):
        RunResult(
            algorithm="GA",
            problem="Sphere",
            seed=202700,
            best_fitness=float("inf"),
            evaluations=0,
            best_history=()
        )