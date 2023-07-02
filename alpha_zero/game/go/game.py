from collections.abc import Iterable
from functools import cached_property, lru_cache

import numpy as np

from alpha_zero.game.abc import Game as AbstractGame

from . import const
from .board import Board


class Game(AbstractGame):
    board_size: int

    def __init__(self, board_size: int) -> None:
        self.board_size = board_size

    @cached_property
    def action_size(self) -> int:
        return self.board_size * self.board_size + 1

    def canonicalize(self, board: Board, player: int) -> Board:
        new_board: Board = board.copy()
        new_board.canonicalize(player=player)
        return new_board

    def canonicalize_value(self, value: float, player: int) -> float:
        if abs(value) < 0.5:
            return abs(value)
        if player == const.CANONICAL_PLAYER:
            return value
        else:
            return -value

    def check_terminated(self, board: Board, player: int) -> tuple[bool, float]:
        if not board.check_terminated(player=player):
            return False, 0
        black_score, white_score = board.scores
        if black_score > white_score:
            return True, player
        elif black_score < white_score:
            return True, -player
        elif black_score == white_score:
            return True, const.DRAW_VALUE
        else:
            assert False, "Unreachable"  # pragma: no cover

    def get_equivalent_boards(
        self, board: Board, policy: np.ndarray
    ) -> Iterable[tuple[Board, np.ndarray]]:
        return [(board, policy)]
        # policy_matrix: np.ndarray = np.resize(
        #     policy, new_shape=(self.board_size, self.board_size)
        # )
        # policy_pass: float = policy[-1]
        # next_board: np.ndarray = board.data.copy()
        # next_policy: np.ndarray = policy_matrix.copy()

        # def get_board(matrix: np.ndarray) -> Board:
        #     new_board: Board = board.copy()
        #     new_board.from_numpy(matrix)
        #     return new_board

        # def get_policy(matrix: np.ndarray) -> np.ndarray:
        #     return np.append(matrix.flatten(), policy_pass)

        # for _ in range(4):
        #     yield get_board(next_board), get_policy(next_policy)
        #     yield get_board(np.fliplr(next_board)), get_policy(np.fliplr(next_policy))
        #     next_board = np.rot90(next_board)
        #     next_policy = np.rot90(next_policy)

        # np.testing.assert_array_almost_equal(next_board, board.data)
        # np.testing.assert_array_almost_equal(next_policy, policy_matrix)

    def get_init_board(self) -> Board:
        return Board(size=self.board_size)

    def get_next_player(self, player: int) -> int:
        assert player in [+1, -1]
        return -player

    @lru_cache(maxsize=65536)
    def get_valid_actions(self, board: Board, player: int) -> np.ndarray:
        ret: np.ndarray = np.zeros(shape=(self.action_size,), dtype=bool)
        for x, y in board.get_valid_moves(player=player):
            ret[x * self.board_size + y] = True
        return ret

    def play(self, board: Board, action: int, player: int) -> Board:
        new_board: Board = board.copy()
        new_board.place_stone(
            action // self.board_size, action % self.board_size, player=player
        )
        return new_board
