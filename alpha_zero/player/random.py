import numpy as np

from alpha_zero.game import Board, Game
from alpha_zero.utils.policy import mask as mask_policy

from .abc import Player as AbstractPlayer


class Player(AbstractPlayer):
    game: Game

    def __init__(self, game: Game) -> None:
        self.game = game

    def __repr__(self) -> str:
        return "Random"

    def play(self, board: Board, player: int) -> int:
        valid_actions: np.ndarray = self.game.get_valid_actions(
            board=board, player=player
        )
        policy: np.ndarray = mask_policy(
            policy=valid_actions, valid_actions=valid_actions
        )
        return np.random.choice(self.game.action_size, p=policy)
