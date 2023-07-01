import numbers
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np

from alpha_zero.game import Board, Game
from alpha_zero.neural_network.residual import NeuralNetworkWrapper as NetworkWrapper


def test_network_wrapper_pickle(board_size: int = 5) -> None:
    game: Game = Game(board_size=board_size)
    net: NetworkWrapper = NetworkWrapper(
        action_size=game.action_size, board_size=game.board_size
    )
    pickle.dumps(net)


def test_network_wrapper_checkpoint(board_size: int = 5) -> None:
    game: Game = Game(board_size=board_size)
    net: NetworkWrapper = NetworkWrapper(
        action_size=game.action_size, board_size=game.board_size
    )
    tmp_dir: Path = Path(tempfile.mkdtemp())
    try:
        checkpoint_filepath: Path = tmp_dir / "checkpoint.pth"
        net.save(filepath=checkpoint_filepath)
        assert checkpoint_filepath.exists()
        net.load(filepath=checkpoint_filepath)
    finally:
        shutil.rmtree(tmp_dir)


def test_network_wrapper_predict(board_size: int = 5) -> None:
    game: Game = Game(board_size=board_size)
    net: NetworkWrapper = NetworkWrapper(
        action_size=game.action_size, board_size=game.board_size
    )
    board: Board = game.get_init_board()
    policy, value = net.predict(board.encode())
    assert policy.shape == (game.action_size,)
    assert np.all((0 <= policy) & (policy <= 1))
    np.testing.assert_almost_equal(np.sum(policy), 1.0)
    assert isinstance(value, numbers.Real)
    assert -1 <= value <= 1
