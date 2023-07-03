import logging
from pathlib import Path

import numpy as np

from .abc import NeuralNetwork as AbstractNeuralNetwork


class NeuralNetwork(AbstractNeuralNetwork):
    action_size: int

    def __init__(self, action_size: int, board_size: int) -> None:
        self.action_size = action_size

    def load(self, filepath: Path) -> None:
        logging.warning("Loading is not implemented for Random Neural Network")

    def predict(self, canonical_state: np.ndarray) -> tuple[np.ndarray, float]:
        return np.ones(shape=(self.action_size,)) / self.action_size, 0.0

    def save(self, filepath: Path) -> None:
        logging.warning("Saving is not implemented for Random Neural Network")

    def train(self, samples: list[tuple[np.ndarray, np.ndarray, float]]) -> None:
        logging.warning("Training is not implemented for Random Neural Network")
