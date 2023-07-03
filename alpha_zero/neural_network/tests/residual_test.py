import itertools
import numbers
import pickle
import shutil
import tempfile
from pathlib import Path

import numpy as np

from alpha_zero.game import CANONICAL_PLAYER, Board, Game
from alpha_zero.neural_network.residual import NeuralNetworkWrapper as NetworkWrapper
from alpha_zero.utils.board import random as random_board
from alpha_zero.utils.policy import mask as mask_policy


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
    np.testing.assert_allclose(np.sum(policy), 1.0, rtol=1e-6, atol=1e-6)
    assert isinstance(value, numbers.Real)
    assert -1 <= value <= 1


def test_network_wrapper_train(board_size: int = 5) -> None:
    game: Game = Game(board_size=board_size)
    net: NetworkWrapper = NetworkWrapper(
        action_size=game.action_size,
        board_size=game.board_size,
        num_features=32,
        num_residual_blocks=1,
    )
    board: Board = random_board(game=game)
    valid_actions: np.ndarray = game.get_valid_actions(
        board=board, player=CANONICAL_PLAYER
    )
    policy: np.ndarray = mask_policy(
        policy=np.random.rand(game.action_size), valid_actions=valid_actions
    )
    value: float = np.random.rand()
    dataset: list[tuple[np.ndarray, np.ndarray, float]] = list(
        itertools.repeat((board.encode(), policy, value), times=65536)
    )
    loss: float = net.train(samples=dataset, lr=1e-3)
    predict_policy, predict_value = net.predict(board.encode())
    predict_policy = mask_policy(policy=predict_policy, valid_actions=valid_actions)
    np.testing.assert_allclose(predict_policy, policy, rtol=1e-2, atol=1e-2)
    np.testing.assert_allclose(predict_value, value, rtol=1e-2, atol=1e-2)
