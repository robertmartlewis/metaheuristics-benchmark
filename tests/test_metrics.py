from benchmark.metrics import evaluations_to_convergence


def test_convergence_after_last_improvement():
    history = (10, 9, 9, 9, 9)

    assert evaluations_to_convergence(
        history, patience=3
    ) == 5


def test_convergence_ignores_earlier_plateau():
    history = (10, 10, 10, 10, 8, 8, 8, 8)

    assert evaluations_to_convergence(
        history, patience=3
    ) == 8


def test_no_convergence_before_budget_exhausted():
    history = (10, 9, 8, 7, 6)

    assert evaluations_to_convergence(
        history, patience=3
    ) is None