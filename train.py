from typing import Annotated, Optional

import typer

from alpha_zero.game import Game
from alpha_zero.trainer import Trainer


def main(
    board_size: Annotated[int, typer.Option("--board-size", envvar="BOARD_SIZE")] = 9,
    processes: Annotated[
        Optional[int], typer.Option("--processes", envvar="PROCESSES")
    ] = None,
) -> None:
    game: Game = Game(board_size=board_size)
    trainer: Trainer = Trainer(game=game)
    trainer.train(processes=processes)


if __name__ == "__main__":
    typer.run(main)
