from abc import ABC, abstractmethod

from alpha_zero.game import Board, Game


class Player(ABC):
    game: Game

    @abstractmethod
    def __init__(self, game: Game) -> None:
        ...

    @abstractmethod
    def __str__(self) -> str:
        ...

    @abstractmethod
    def play(self, board: Board, player: int) -> int:
        ...
