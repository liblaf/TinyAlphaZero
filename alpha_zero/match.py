from rich import print
from rich.panel import Panel
from rich.text import Text

from .game import Board, Game
from .player.abc import Player


def display_board(board: Board, move_count: int = 0) -> None:
    content: str = str(board)
    if "Move Count" not in content:
        content = f"Move Count: {move_count}\n" + content
    print(Panel(content))


def single_match(
    game: Game, player_1: Player, player_2: Player, display: bool = True
) -> float:
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
    if display:
        if value > 0:
            print(Text(text=f"{player_1} defeated {player_2}", style="bold green"))
        elif value < 0:
            print(Text(text=f"{player_1} defeated by {player_2}", style="bold red"))
        else:
            print(Text(text=f"{player_1} and {player_2} drew", style="bold yellow"))
    return value
