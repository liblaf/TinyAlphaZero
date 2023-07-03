from collections.abc import Iterable
from enum import IntEnum
from typing import Self

import numpy as np

from alpha_zero.game.abc import Board as AbstractBoard

from . import const

ADJACENT: list[tuple[int, int]] = [(-1, 0), (+1, 0), (0, -1), (0, +1)]


class Stone(IntEnum):
    BLACK = 1
    EMPTY = 0
    WHITE = -1


STONE_DISPLAY: dict[Stone, str] = {
    Stone.BLACK: "\N{Black Circle}",  # ●
    Stone.EMPTY: "+",
    Stone.WHITE: "\N{White Circle}",  # ○
}


class Board(AbstractBoard):
    data: np.ndarray
    last_captured: tuple[int, int] = (-1, -1)
    last_move_is_pass: tuple[bool, bool] = (False, False)
    move_count: int = 0
    size: int

    def __init__(self, size: int) -> None:
        self.size = size
        self.data = np.zeros(shape=(size, size), dtype=np.int8)

    def __repr__(self) -> str:
        ret: str = ""
        ret += f"Move Count   : {self.move_count}\n"
        ret += f"Last Captured: {self.last_captured}\n"
        for i in range(self.size):
            for j in range(self.size):
                ret += STONE_DISPLAY.get(self.data[i, j], "?")
            ret += "\n"
        return ret

    def __eq__(self, other: Self) -> bool:
        if not np.array_equal(self.data, other.data):
            return False
        if False and self.move_count != other.move_count:
            return False
        if self.last_captured != other.last_captured:
            return False
        if self.last_move_is_pass != other.last_move_is_pass:
            return False
        return True

    def __hash__(self) -> int:
        """Hashing is used to cache the results of get_valid_actions()"""
        return hash((self.last_captured, self.last_move_is_pass, self.data.dumps()))

    @property
    def action_size(self) -> int:
        return self.size * self.size + 1

    def canonicalize(self, player: int) -> None:
        if player != const.CANONICAL_PLAYER:
            self.data *= -1

    def check_terminated(self, player: int) -> bool:
        if self.last_move_is_pass == (True, True):
            return True
        if self.move_count >= 4 * self.size * self.size:
            return True
        return False

    def copy(self) -> Self:
        ret: Self = Board(size=self.size)
        ret.data = self.data.copy()
        ret.last_captured = self.last_captured
        ret.last_move_is_pass = self.last_move_is_pass
        ret.move_count = self.move_count
        return ret

    def encode(self) -> np.ndarray:
        return np.stack(
            [
                self.data == Stone.BLACK,
                self.data == Stone.EMPTY,
                self.data == Stone.WHITE,
            ],
            dtype=float,
        )

    def from_numpy(self, data: np.ndarray) -> None:
        assert data.shape == (self.size, self.size)
        self.data = data

    def get_valid_moves(self, player: int) -> Iterable[tuple[int, int]]:
        for x, y in np.ndindex(self.data.shape):
            if self._validate_move(x, y, player=player):
                yield x, y

    def place_stone(self, x: int, y: int, player: int) -> None:
        assert self._validate_move(x, y, player=player)
        self._place_stone_unsafe(x, y, player=player)

    @property
    def scores(self) -> tuple[int, int]:
        black_score: int = 0
        white_score: int = 0
        visited: np.ndarray = np.zeros(shape=(self.size, self.size), dtype=bool)
        for x, y in np.ndindex(self.size, self.size):
            if self.data[x, y] == Stone.EMPTY:
                if visited[x, y]:
                    continue
                color: Stone = Stone.EMPTY
                connected: list[tuple[int, int]] = list(self._get_connected(x, y))
                for nx, ny in self._get_bounds(x, y):
                    if self.data[nx, ny] == Stone.EMPTY:
                        pass
                    else:
                        if color == Stone.EMPTY:
                            color = self.data[nx, ny]
                        elif color == self.data[nx, ny]:
                            pass
                        elif color != self.data[nx, ny]:
                            color = Stone.EMPTY
                            break
                        else:
                            assert False, "Unreachable"  # pragma: no cover
                if color == Stone.EMPTY:
                    pass
                elif color == Stone.BLACK:
                    black_score += len(connected)
                elif color == Stone.WHITE:
                    white_score += len(connected)
                else:
                    assert False, "Unreachable"  # pragma: no cover
                for x, y in connected:
                    visited[x, y] = True
            elif self.data[x, y] == Stone.BLACK:
                black_score += 1
            elif self.data[x, y] == Stone.WHITE:
                white_score += 1
            else:
                assert False, "Unreachable"  # pragma: no cover
        return black_score, white_score

    def to_numpy(self) -> np.ndarray:
        return self.data

    def _get_adjacent(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        for dx, dy in ADJACENT:
            nx, ny = x + dx, y + dy
            if self._within_board(nx, ny):
                yield nx, ny

    def _get_bounds(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        color: Stone = Stone(self.data[x, y])
        visited: set[tuple[int, int]] = set()
        for nx, ny in self._get_connected(x, y):
            for nnx, nny in self._get_adjacent(nx, ny):
                if self.data[nnx, nny] != color and (nnx, nny) not in visited:
                    yield nnx, nny
                    visited.add((nnx, nny))

    def _get_connected(self, x: int, y: int) -> Iterable[tuple[int, int]]:
        color: Stone = Stone(self.data[x, y])
        stack: list[tuple[int, int]] = [(x, y)]
        visited: set[tuple[int, int]] = set()
        while len(stack) > 0:
            x, y = stack.pop()
            if (x, y) in visited:
                continue
            yield x, y
            visited.add((x, y))
            for nx, ny in self._get_adjacent(x, y):
                if self.data[nx, ny] == color:
                    stack.append((nx, ny))

    def _get_connected_liberty(
        self, x: int, y: int
    ) -> tuple[Iterable[tuple[int, int]], bool]:
        color: Stone = Stone(self.data[x, y])
        connected: set[tuple[int, int]] = set()
        stack: list[tuple[int, int]] = [(x, y)]
        liberty: bool = False
        while len(stack) > 0:
            x, y = stack.pop()
            if (x, y) in connected:
                continue
            connected.add((x, y))
            for nx, ny in self._get_adjacent(x, y):
                if self.data[nx, ny] == Stone.EMPTY:
                    liberty = True
                elif self.data[nx, ny] == color:
                    stack.append((nx, ny))
        return connected, liberty

    def _get_liberty(self, x: int, y: int) -> bool:
        for nx, ny in self._get_connected(x, y):
            for nnx, nny in self._get_adjacent(nx, ny):
                if self.data[nnx, nny] == Stone.EMPTY:
                    return True
        return False

    def _place_stone_unsafe(self, x: int, y: int, player: int) -> None:
        color: Stone = Stone(player)
        if (x, y) == (self.size, 0):
            self.last_move_is_pass = (self.last_move_is_pass[-1], True)
        else:
            self.data[x, y] = color
            capture: list[tuple[int, int]] = []
            for nx, ny in self._get_adjacent(x, y):
                if self.data[nx, ny] == -color:
                    connected, liberty = self._get_connected_liberty(nx, ny)
                    if not liberty:
                        capture.extend(connected)
            for nnx, nny in capture:
                self.data[nnx, nny] = Stone.EMPTY
            if len(capture) == 1:
                self.last_captured = capture[0]
            else:
                self.last_captured = (-1, -1)
            self.last_move_is_pass = (self.last_move_is_pass[-1], False)
        self.move_count += 1

    def _validate_move(self, x: int, y: int, player: int) -> bool:
        if (x, y) == (self.size, 0):
            return True
        assert self._within_board(x, y)
        if self.data[x, y] != Stone.EMPTY:
            return False
        if (x, y) == self.last_captured:
            return False
        board: Self = self.copy()
        board._place_stone_unsafe(x, y, player=player)
        if not board._get_liberty(x, y):
            return False
        return True

    def _within_board(self, x: int, y: int) -> bool:
        return 0 <= x < self.size and 0 <= y < self.size
