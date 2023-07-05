from typing import Annotated, Optional

import typer

from alpha_zero.game import Game
from alpha_zero.mcts import MCTSConfig
from alpha_zero.trainer import Trainer


def main(
    board_size: Annotated[int, typer.Option("--board-size", envvar="BOARD_SIZE")] = 9,
    epochs: Annotated[int, typer.Option("--epochs", envvar="EPOCHS")] = 8,
    num_simulation_eval: Annotated[
        int, typer.Option("--num-simulation-eval", envvar="NUM_SIMULATION_EVAL")
    ] = 0,
    num_simulation_train: Annotated[
        int, typer.Option("--num-simulation-train", envvar="NUM_SIMULATION_TRAIN")
    ] = 128,
    processes: Annotated[
        Optional[int], typer.Option("--processes", envvar="PROCESSES")
    ] = 8,
    update_threshold: Annotated[
        float, typer.Option("--update-threshold", envvar="UPDATE_THRESHOLD")
    ] = 0.55,
) -> None:
    game: Game = Game(board_size=board_size)
    mcts_config: MCTSConfig = MCTSConfig(
        num_simulation_eval=num_simulation_eval,
        num_simulation_train=num_simulation_train,
    )
    trainer: Trainer = Trainer(game=game, mcts_config=mcts_config)
    trainer.train(epochs=epochs, processes=processes, update_threshold=update_threshold)


if __name__ == "__main__":
    typer.run(main)
