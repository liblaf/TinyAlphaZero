from alpha_zero.game import Game
from alpha_zero.match import multi_match, single_match
from alpha_zero.player import PlayerRandom


def test_single_match(board_size: int = 5):
    game: Game = Game(board_size=board_size)
    player_1 = PlayerRandom(game=game)
    player_2 = PlayerRandom(game=game)
    result = single_match(game=game, player_1=player_1, player_2=player_2)
    assert result in [-1, 0, 1]


def test_multi_match(board_size: int = 5, num_match: int = 8):
    game: Game = Game(board_size=board_size)
    player_1 = PlayerRandom(game=game)
    player_2 = PlayerRandom(game=game)
    win, loss, draw = multi_match(
        game=game, player_1=player_1, player_2=player_2, num_match=8
    )
    assert win + loss + draw == (num_match // 2 * 2)
