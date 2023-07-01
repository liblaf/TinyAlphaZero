from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Self

import numpy as np

CANONICAL_PLAYER: int = 1


class Board(ABC):
    @property
    @abstractmethod
    def action_size(self) -> int:
        ...

    @abstractmethod
    def canonicalize(self, player: int) -> None:
        ...

    @abstractmethod
    def check_terminated(self, player: int) -> bool:
        ...

    @abstractmethod
    def copy(self) -> Self:
        ...

    @abstractmethod
    def encode(self) -> np.ndarray:
        ...


class Game(ABC):
    board_size: int

    @abstractmethod
    def __init__(self, board_size: int) -> None:
        ...

    @property
    @abstractmethod
    def action_size(self) -> int:
        ...

    @abstractmethod
    def canonicalize(self, board: Board, player: int) -> Board:
        ...

    @abstractmethod
    def check_terminated(self, board: Board, player: int) -> tuple[bool, float]:
        ...

    @abstractmethod
    def get_equivalent_boards(
        self, board: Board, policy: np.ndarray
    ) -> Iterable[tuple[Board, np.ndarray]]:
        ...

    @abstractmethod
    def get_init_board(self) -> Board:
        ...

    @abstractmethod
    def get_valid_actions(self, board: Board, player: int) -> np.ndarray:
        ...

    @abstractmethod
    def play(self, board: Board, action: int, player: int) -> Board:
        ...
