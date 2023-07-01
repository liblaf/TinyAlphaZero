from abc import ABC, abstractmethod

from alpha_zero.game import Board, Game


class Player(ABC):
    game: Game

    @abstractmethod
    def __repr__(self) -> str:
        ...

    @abstractmethod
    def play(self, board: Board, player: int) -> int:
        ...
