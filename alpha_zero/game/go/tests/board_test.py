import numpy as np

from ..board import Board


def test_go_board_copy(size: int = 5) -> None:
    board: Board = Board(size=size)
    new_board: Board = board.copy()
    assert new_board is not board


def test_go_board_encode(size: int = 5) -> None:
    board: Board = Board(size=size)
    encoded: np.ndarray = board.encode()
    assert encoded.shape == (3, size, size)
    assert np.all((encoded == 0) | (encoded == 1))
