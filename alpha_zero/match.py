import itertools
from typing import Optional

import torch.multiprocessing as mp
from rich import print
from rich.panel import Panel
from rich.text import Text

from .game import Board, Game
from .player.abc import Player


def display_board(board: Board, move_count: int = 0) -> None:
    content: str = str(board)
    if "Move Count" not in content:
        content = f"Move Count: {move_count}\n" + content
    print(Panel(content, expand=False))


def single_match(
    game: Game, player_1: Player, player_2: Player, display: bool = True
) -> int:
    board: Board = game.get_init_board()
    move_count: int = 0
    value: float = 0
    while True:
        action: int = player_1.play(board=board, player=1)
        board = game.play(board=board, action=action, player=1)
        move_count += 1
        if display:
            display_board(board=board, move_count=move_count)
        terminated, value = game.check_terminated(board=board, player=1)
        if terminated:
            break

        action: int = player_2.play(board=board, player=-1)
        board = game.play(board=board, action=action, player=-1)
        move_count += 1
        if display:
            display_board(board=board, move_count=move_count)
        terminated, value = game.check_terminated(board=board, player=-1)
        if terminated:
            value = -value
            break
    if value > 0.5:
        if display:
            print(Text(text=f"{player_1} defeated {player_2}", style="bold green"))
        return +1
    elif value < -0.5:
        if display:
            print(Text(text=f"{player_1} defeated by {player_2}", style="bold red"))
        return -1
    else:
        if display:
            print(Text(text=f"{player_1} and {player_2} drew", style="bold yellow"))
        return 0


def multi_match(
    game: Game,
    player_1: Player,
    player_2: Player,
    num_match: int = 100,
    processes: Optional[int] = None,
    display: bool = True,
) -> tuple[int, int, int]:
    num_match = num_match // 2 * 2
    with mp.Pool(processes=processes) as pool:
        results: list[float] = pool.starmap(
            single_match,
            itertools.repeat((game, player_1, player_2, False), times=num_match // 2),
        )
        win, loss, draw = 0, 0, 0
        for value in results:
            if value > +0.5:
                win += 1
            elif value < -0.5:
                loss += 1
            else:
                draw += 1
        results: list[float] = pool.starmap(
            single_match,
            itertools.repeat((game, player_2, player_1, False), times=num_match // 2),
        )
        for value in results:
            if value < -0.5:
                win += 1
            elif value > +0.5:
                loss += 1
            else:
                draw += 1
        if display:
            print(f"{str(player_1):>10} Win : {win:>3} ({win / num_match:>6.2%})")
            print(f"{str(player_2):>10} Win : {loss:>3} ({loss / num_match:>6.2%})")
            print(f"          Draw : {draw:>3} ({draw / num_match:>6.2%})")
        return win, loss, draw
