from typing import Optional

from alpha_zero.game import Game
from alpha_zero.match import multi_match, single_match
from alpha_zero.mcts import MCTS, MCTSConfig
from alpha_zero.neural_network.random import NeuralNetwork as RandomNetwork
from alpha_zero.player import PlayerAlphaZero, PlayerRandom
from alpha_zero.trainer import DRAW_VALUE


def test_single_match(board_size: int = 5) -> None:
    game: Game = Game(board_size=board_size)
    player_1 = PlayerRandom(game=game)
    player_2 = PlayerRandom(game=game)
    result = single_match(game=game, player_1=player_1, player_2=player_2)
    assert result in [-1, 0, 1]


def test_multi_match(board_size: int = 5, num_match: int = 8) -> None:
    game: Game = Game(board_size=board_size)
    player_1 = PlayerRandom(game=game)
    player_2 = PlayerRandom(game=game)
    win, loss, draw = multi_match(
        game=game, player_1=player_1, player_2=player_2, num_match=8
    )
    assert win + loss + draw == (num_match // 2 * 2)


def test_mcts_random(
    board_size: int = 5, num_match: int = 100, processes: Optional[int] = 8
) -> None:
    game: Game = Game(board_size=board_size)
    net: RandomNetwork = RandomNetwork(
        action_size=game.action_size, board_size=game.board_size
    )
    mcts: MCTS = MCTS(game=game, net=net, config=MCTSConfig(num_simulation_eval=128))
    win, loss, draw = multi_match(
        game=game,
        player_1=PlayerAlphaZero(game=game, mcts=mcts),
        player_2=PlayerRandom(game=game),
        num_match=num_match,
        processes=processes,
    )
    assert (win + DRAW_VALUE * draw) / (win + loss + 2 * DRAW_VALUE * draw) > 0.7
