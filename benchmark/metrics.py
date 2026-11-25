def evaluations_to_convergence(best_history, patience=1000):
    last_improvement = 1

    for i in range(1, len(best_history)):
        if best_history[i] < best_history[i - 1]:
            last_improvement = i + 1

    evaluations_without_improvement = len(best_history) - last_improvement

    if evaluations_without_improvement >= patience:
        return last_improvement + patience

    return None