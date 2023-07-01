import pickle

from alpha_zero.game import Game
from alpha_zero.player.random import Player


def test_player_random_pickle(board_size: int = 5) -> None:
    game: Game = Game(board_size=board_size)
    player: Player = Player(game=game)
    pickle.dumps(player)
