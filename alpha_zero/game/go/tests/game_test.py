from ..game import Game


def test_go_game_action_size() -> None:
    game: Game = Game(board_size=5)
    assert game.action_size == 26
