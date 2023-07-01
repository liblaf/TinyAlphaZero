import pickle

from alpha_zero.game import Game
from alpha_zero.mcts import MCTS
from alpha_zero.neural_network import NeuralNetwork


def test_mcts_pickle(board_size: int = 5) -> None:
    game: Game = Game(board_size=board_size)
    net: NeuralNetwork = NeuralNetwork(
        action_size=game.action_size, board_size=game.board_size
    )
    mcts: MCTS = MCTS(game=game, net=net)
    pickle.dumps(mcts)
