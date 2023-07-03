import pickle

import numpy as np

from alpha_zero.game.go.board import Board


def test_go_bard_pickle(size: int = 5) -> None:
    board: Board = Board(size=size)
    pickle.dumps(board)


def test_go_board_copy(size: int = 5) -> None:
    board: Board = Board(size=size)
    new_board: Board = board.copy()
    assert new_board is not board


def test_go_board_encode(size: int = 5) -> None:
    board: Board = Board(size=size)
    encoded: np.ndarray = board.encode()
    assert encoded.shape == (3, size, size)
    assert np.all((encoded == 0) | (encoded == 1))


def test_go_board_valid_moves_0() -> None:
    board: Board = Board(size=5)
    board.from_numpy(
        np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, 1, -1, 0],
                [-1, 1, 0, 1, -1],
                [0, -1, 1, -1, 0],
                [0, 0, -1, 0, 0],
            ]
        )
    )
    assert set(board.get_valid_moves(player=1)) == set(
        [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 4),
            (3, 0),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
        ]
    )

    board.place_stone(2, 2, player=-1)
    np.testing.assert_allclose(
        board.to_numpy(),
        np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, 0, -1, 0],
                [-1, 0, -1, 0, -1],
                [0, -1, 0, -1, 0],
                [0, 0, -1, 0, 0],
            ]
        ),
    )
    assert set(board.get_valid_moves(player=1)) == set(
        [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 4),
            (3, 0),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
        ]
    )
    assert set(board.get_valid_moves(player=-1)) == set(
        [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 2),
            (1, 4),
            (2, 1),
            (2, 3),
            (3, 0),
            (3, 2),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 3),
            (4, 4),
        ]
    )


def test_go_board_valid_moves_1() -> None:
    board: Board = Board(size=5)
    board.from_numpy(
        np.array(
            [
                [0, -1, 1, -1, 0],
                [-1, 1, 1, 1, -1],
                [1, 1, 1, 1, 1],
                [1, 1, 1, 1, 0],
                [1, 1, 1, -1, 0],
            ]
        )
    )
    assert set(board.get_valid_moves(player=1)) == set([(0, 0), (0, 4), (3, 4), (4, 4)])
    assert set(board.get_valid_moves(player=-1)) == set([(3, 4), (4, 4)])

    board.place_stone(3, 4, player=-1)
    np.testing.assert_allclose(
        board.to_numpy(),
        np.array(
            [
                [0, -1, 0, -1, 0],
                [-1, 0, 0, 0, -1],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, -1],
                [0, 0, 0, -1, 0],
            ]
        ),
    )


def test_go_board_ko_rule() -> None:
    board: Board = Board(size=5)
    board.from_numpy(
        np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 0, -1, 1, 0],
                [0, -1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        )
    )
    assert set(board.get_valid_moves(player=1)) == set(
        [
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 4),
            (2, 0),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )
    assert set(board.get_valid_moves(player=-1)) == set(
        [
            (0, 0),
            (0, 3),
            (0, 4),
            (1, 1),
            (1, 4),
            (2, 0),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )

    board.place_stone(1, 1, player=1)
    np.testing.assert_allclose(
        board.to_numpy(),
        np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 1, 0, 1, 0],
                [0, -1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
    )
    assert set(board.get_valid_moves(player=1)) == set(
        [
            (0, 0),
            (0, 3),
            (0, 4),
            (1, 4),
            (2, 0),
            (2, 3),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )
    assert set(board.get_valid_moves(player=1)) == set(board.get_valid_moves(player=-1))

    board.place_stone(1, 4, player=-1)
    board.place_stone(2, 3, player=1)
    assert set(board.get_valid_moves(player=1)) == set(
        [
            (0, 0),
            (0, 3),
            (0, 4),
            (1, 2),
            (2, 0),
            (2, 4),
            (3, 0),
            (3, 1),
            (3, 2),
            (3, 3),
            (3, 4),
            (4, 0),
            (4, 1),
            (4, 2),
            (4, 3),
        ]
    )
    assert set(board.get_valid_moves(player=1)) == set(board.get_valid_moves(player=-1))

    board.place_stone(1, 2, player=-1)
    np.testing.assert_allclose(
        board.to_numpy(),
        np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 0, -1, 1, -1],
                [0, -1, 1, 1, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        ),
    )


def test_go_board_scores_0() -> None:
    board: Board = Board(size=5)
    board.from_numpy(
        np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 0, -1, 1, 0],
                [0, -1, 1, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1],
            ]
        )
    )
    assert board.scores == (4, 6)

    board.place_stone(1, 1, player=1)
    assert board.scores == (6, 4)


def test_go_board_scores_1() -> None:
    board: Board = Board(size=5)
    board.from_numpy(
        np.array(
            [
                [0, -1, 1, 0, 0],
                [-1, 0, 0, 1, 0],
                [0, -1, 0, 1, 0],
                [0, -1, 0, 1, 0],
                [0, 0, -1, 0, 1],
            ]
        )
    )
    assert board.scores == (10, 10)


def test_go_board_scores_2() -> None:
    board: Board = Board(size=5)
    board.from_numpy(
        np.array(
            [
                [0, -1, 1, -1, 0],
                [-1, 0, 1, 1, -1],
                [0, 1, 0, 1, 0],
                [1, 0, 1, 1, 0],
                [0, 1, -1, 0, 1],
            ]
        )
    )
    assert board.scores == (13, 7)
