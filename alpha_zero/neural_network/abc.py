from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class NeuralNetwork(ABC):
    @abstractmethod
    def load(self, filepath: Path) -> None:
        ...

    @abstractmethod
    def predict(self, canonical_state: np.ndarray) -> tuple[np.ndarray, float]:
        ...

    @abstractmethod
    def save(self, filepath: Path) -> None:
        ...

    @abstractmethod
    def train(self, samples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        ...
