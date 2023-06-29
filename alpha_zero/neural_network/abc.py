from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np


class NeuralNetwork(ABC):
    @abstractmethod
    def checkpoint_load(self, filepath: Path) -> None:
        ...

    @abstractmethod
    def checkpoint_save(self, filepath: Path) -> None:
        ...

    @abstractmethod
    def predict(self, board: np.ndarray) -> tuple[np.ndarray, float]:
        ...

    @abstractmethod
    def train(self, samples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        ...
