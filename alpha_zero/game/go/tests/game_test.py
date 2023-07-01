import pickle

from alpha_zero.game.go.game import Game


def test_go_game_pickle(size: int = 5) -> None:
    game: Game = Game(board_size=5)
    pickle.dumps(game)


def test_go_game_action_size() -> None:
    game: Game = Game(board_size=5)
    assert game.action_size == 26
