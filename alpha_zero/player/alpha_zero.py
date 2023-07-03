import numpy as np

from alpha_zero.game import Board, Game
from alpha_zero.mcts import MCTS

from .abc import Player as AbstractPlayer


class Player(AbstractPlayer):
    game: Game
    mcts: MCTS
    random: bool = False

    def __init__(self, game: Game, mcts: MCTS, random: bool = True) -> None:
        self.game = game
        self.mcts = mcts
        self.random = random

    def __repr__(self) -> str:
        return "AlphaZero"

    def play(self, board: Board, player: int) -> int:
        self.mcts.eval()
        policy: np.ndarray = self.mcts.search(board=board, player=player)
        if self.random:
            return np.random.choice(self.game.action_size, p=policy)
        else:
            return int(np.argmax(policy))
