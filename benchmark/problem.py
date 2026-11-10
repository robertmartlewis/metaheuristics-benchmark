from abc import ABC, abstractmethod

class Problem(ABC):

    @abstractmethod
    def validate(self, candidate):
        pass

    @abstractmethod
    def evaluate(self, candidate):
        pass

    @abstractmethod
    def random_solution(self, rng):
        pass