from benchmark.problem import Problem

def test_valid_candidate():
    prob = Problem(3, -5.12, 5.12)
    assert prob.validate([1.0, 2.0, 3.0])

def test_valid_candidate_at_bounds():
    prob = Problem(3, -5.12, 5.12)
    assert prob.validate([0.0, -5.12, 5.12])

def test_invalid_candidate_out_of_bounds():
    prob = Problem(3, -5.12, 5.12)
    assert not prob.validate([1.0, 2.0, 7.0])

def test_invalid_candidate_out_of_bounds_with_negative_value():
    prob = Problem(3, -5.12, 5.12)
    assert not prob.validate([1.0, 2.0, -7.0])

def test_invalid_candidate_dimension():
    prob = Problem(3, -5.12, 5.12)
    assert not prob.validate([1.0, 2.0])