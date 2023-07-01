from pathlib import Path
from typing import Annotated, Optional

import click
import typer

from alpha_zero.game import Game
from alpha_zero.match import multi_match, single_match
from alpha_zero.mcts import MCTS
from alpha_zero.neural_network import NeuralNetwork
from alpha_zero.player import PLAYER_LIST, Player, PlayerAlphaZero, PlayerRandom


def construct_player(
    game: Game, player: str, checkpoint: Optional[Path] = None
) -> Player:
    if player == "Random":
        return PlayerRandom(game=game)
    elif player == "AlphaZero":
        net: NeuralNetwork = NeuralNetwork(
            action_size=game.action_size, board_size=game.board_size
        )
        if checkpoint:
            net.load(filepath=checkpoint)
        mcts: MCTS = MCTS(game=game, net=net)
        return PlayerAlphaZero(game=game, mcts=mcts)
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
    num_match: Annotated[int, typer.Option("--num-match", envvar="NUM_MATCH")] = 1,
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
    if num_match > 1:
        multi_match(
            game=game,
            player_1=player_1_instance,
            player_2=player_2_instance,
            num_match=num_match,
            display=True,
        )
    else:
        single_match(
            game=game,
            player_1=player_1_instance,
            player_2=player_2_instance,
            display=True,
        )


if __name__ == "__main__":
    typer.run(main)
