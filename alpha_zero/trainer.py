import copy
import itertools
import logging
from collections import deque
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch.multiprocessing as mp

from alpha_zero.game import CANONICAL_PLAYER, Board, Game
from alpha_zero.match import multi_match
from alpha_zero.mcts import MCTS, MCTSConfig
from alpha_zero.neural_network import NeuralNetwork
from alpha_zero.player import PlayerAlphaZero, PlayerRandom
from alpha_zero.plot import plot_loss, plot_update_frequency, plot_win_rate

DRAW_VALUE: float = 0.1


def self_play_single_yield(
    game: Game, mcts: MCTS
) -> Iterable[tuple[Board, np.ndarray, float]]:
    board: Board = game.get_init_board()
    history: list[tuple[Board, np.ndarray, int]] = []
    player: int = CANONICAL_PLAYER
    value: float = np.nan
    while True:
        policy: np.ndarray = mcts.search(board=board, player=player)
        action: int = np.random.choice(game.action_size, p=policy)
        board = game.play(board=board, action=action, player=player)
        terminated, value = game.check_terminated(board=board, player=player)
        history.append((board, policy, player))
        if terminated:
            value = game.canonicalize_value(value=value, player=player)
            break
        player = game.get_next_player(player=player)
    assert not np.isnan(value)
    for board, policy, player in history:
        canonical_value: float = game.canonicalize_value(value=value, player=player)
        for b, p in game.get_equivalent_boards(
            board=game.canonicalize(board=board, player=player), policy=policy
        ):
            yield b, p, canonical_value


def self_play_single(
    game: Game, mcts: MCTS
) -> Iterable[tuple[Board, np.ndarray, float]]:
    return list(self_play_single_yield(game=game, mcts=mcts))


class Trainer:
    game: Game
    mcts_config: MCTSConfig
    net: NeuralNetwork

    begin_time: datetime
    loss: list[float]
    match_results: list[tuple[int, int, int]]
    time: list[datetime]
    updated: list[bool]

    def __init__(self, game: Game, mcts_config: MCTSConfig = MCTSConfig()) -> None:
        self.game = game
        self.mcts_config = mcts_config
        self.net = NeuralNetwork(
            action_size=self.game.action_size, board_size=self.game.board_size
        )

        self.loss = []
        self.match_results = []
        self.time = []
        self.updated = []

    def self_play_multi(
        self, mcts: MCTS, num_match: int = 4, *, processes: Optional[int] = None
    ) -> Iterable[Iterable[tuple[Board, np.ndarray, float]]]:
        with mp.Pool(processes=processes) as pool:
            return list(
                pool.starmap(
                    self_play_single,
                    itertools.repeat((self.game, mcts), times=num_match),
                )
            )

    def train(
        self,
        num_iter: int = 128,
        *,
        maxlen: int = 1048576,
        output: Path = Path.cwd() / "output",
        processes: Optional[int] = None,
        update_threshold: float = 0.55,
    ) -> None:
        self.begin_time: datetime = datetime.now()
        dataset: deque[tuple[np.ndarray, np.ndarray, float]] = deque([], maxlen=maxlen)
        for i in range(num_iter):
            logging.info(f"Training Iter {i:>3} ...")
            for match in self.self_play_multi(
                mcts=MCTS(game=self.game, net=self.net, config=self.mcts_config),
                processes=processes,
            ):
                dataset.extend(
                    (board.encode(), policy, value) for board, policy, value in match
                )
            logging.info(f"Length of Dataset: {len(dataset):>6}")
            last_net: NeuralNetwork = copy.deepcopy(self.net)
            self._train(dataset=dataset)
            self.time.append(datetime.now())
            update: bool = self._play_with_last(
                last_net=last_net,
                update_threshold=update_threshold,
                processes=processes,
            )
            if update:
                self._play_with_random(processes=processes)
            else:
                if len(self.match_results) > 0:
                    self.match_results.append(self.match_results[-1])
                else:
                    self._play_with_random(processes=processes)
            self._plot(output=output)

    def _play_with_last(
        self,
        last_net: NeuralNetwork,
        update_threshold: float = 0.55,
        *,
        processes: Optional[int] = None,
    ) -> bool:
        win, loss, draw = multi_match(
            game=self.game,
            player_1=PlayerAlphaZero(
                game=self.game,
                mcts=MCTS(self.game, net=self.net, config=self.mcts_config),
                random=True,
            ),
            player_2=PlayerAlphaZero(
                game=self.game,
                mcts=MCTS(self.game, net=last_net, config=self.mcts_config),
                random=True,
            ),
            processes=processes,
        )
        if (win + DRAW_VALUE * draw) / (
            win + loss + 2 * DRAW_VALUE * draw
        ) > update_threshold:
            logging.info("Accept New Neural Network")
            self.updated.append(True)
            return True
        else:
            logging.info("Reject New Neural Network")
            self.updated.append(False)
            self.net = last_net
            return False

    def _play_with_random(self, processes: Optional[int] = None) -> None:
        win, loss, draw = multi_match(
            game=self.game,
            player_1=PlayerAlphaZero(
                game=self.game,
                mcts=MCTS(game=self.game, net=self.net, config=self.mcts_config),
                random=True,
            ),
            player_2=PlayerRandom(game=self.game),
            processes=processes,
        )
        self.match_results.append((win, loss, draw))

    def _plot(self, output: Path = Path.cwd() / "output") -> None:
        plot_loss(
            begin_time=self.begin_time,
            loss=self.loss,
            time=self.time,
            updated=self.updated,
            output=output / "loss.png",
        )
        plot_update_frequency(
            begin_time=self.begin_time,
            time=self.time,
            updated=self.updated,
            output=output / "update.png",
        )
        plot_win_rate(
            begin_time=self.begin_time,
            match_results=self.match_results,
            time=self.time,
            updated=self.updated,
            output=output / "win-rate.png",
        )

    def _train(self, dataset: Sequence[tuple[np.ndarray, np.ndarray, float]]) -> None:
        self.loss.append(self.net.train(samples=dataset))
        logging.info(f"Loss: {self.loss[-1]}")
