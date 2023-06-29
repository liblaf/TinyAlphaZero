from pathlib import Path
from typing import Annotated, Optional

import click
import typer

from alpha_zero.game import Game
from alpha_zero.match import single_match
from alpha_zero.player import PLAYER_LIST, Player, PlayerRandom


def construct_player(
    game: Game, player: str, checkpoint: Optional[Path] = None
) -> Player:
    if player == "Random":
        return PlayerRandom(game=game)
    elif player == "AlphaZero":
        raise NotImplementedError()
    raise NotImplementedError()


def main(
    player_1: Annotated[
        str,
        typer.Argument(
            envvar="PLAYER_1",
            click_type=click.Choice(choices=PLAYER_LIST, case_sensitive=False),
            case_sensitive=False,
        ),
    ],
    player_2: Annotated[
        str,
        typer.Argument(
            envvar="PLAYER_2",
            click_type=click.Choice(choices=PLAYER_LIST, case_sensitive=False),
            case_sensitive=False,
        ),
    ],
    board_size: Annotated[int, typer.Option("--board-size", envvar="BOARD_SIZE")] = 9,
    match_count: Annotated[
        int, typer.Option("--match-count", envvar="MATCH_COUNT")
    ] = 1,
    player_1_checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            "--player-1-checkpoint",
            envvar="PLAYER_1_CHECKPOINT",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=False,
        ),
    ] = None,
    player_2_checkpoint: Annotated[
        Optional[Path],
        typer.Option(
            "--player-2-checkpoint",
            envvar="PLAYER_1_CHECKPOINT",
            exists=True,
            file_okay=True,
            dir_okay=False,
            readable=True,
            writable=False,
        ),
    ] = None,
) -> None:
    game: Game = Game(board_size=board_size)
    player_1_instance: Player = construct_player(
        game=game, player=player_1, checkpoint=player_1_checkpoint
    )
    player_2_instance: Player = construct_player(
        game=game, player=player_2, checkpoint=player_2_checkpoint
    )
    if match_count == 1:
        single_match(
            game=game,
            player_1=player_1_instance,
            player_2=player_2_instance,
            display=True,
        )


if __name__ == "__main__":
    typer.run(main)
