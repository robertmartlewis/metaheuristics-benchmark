from dataclasses import dataclass

@dataclass
class RunResult:
    algorithm: str
    problem: str
    seed: int
    best_fitness: float
    evaluations: int
    best_history: tuple[float,...]

    def __post_init__(self):
        if self.evaluations <= 0:
            raise ValueError("Evaluation count must be positive")
        if len(self.best_history) != self.evaluations:
            raise ValueError("History length must match evaluation count")
        if self.best_fitness != self.best_history[-1]:
            raise ValueError("Best fitness must match final history value")