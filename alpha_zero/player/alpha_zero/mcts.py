from typing import Optional, Self

import numpy as np

from alpha_zero.game import Board, Game
from alpha_zero.neural_network import NeuralNetwork
from alpha_zero.utils.policy import mask as mask_array

from . import const


def mask_policy(
    game: Game, state: Board, policy: np.ndarray, player: int = 1
) -> np.ndarray:
    valid_actions: np.ndarray = game.get_valid_actions(board=state, player=player)
    return mask_array(policy=policy, valid_actions=valid_actions)


class Node:
    children: list[Self]
    game: Game
    last_action: int
    parent: Optional[Self] = None
    state: Board

    prior_probability: float = 0
    total_action_value: float = 0
    visit_count: int = 0

    def __init__(
        self,
        game: Game,
        state: Board,
        last_action: int,
        parent: Optional[Self] = None,
        prior_probability: float = 0,
    ) -> None:
        self.children = []
        self.game = game
        self.last_action = last_action
        self.parent = parent
        self.state = state

        self.prior_probability = prior_probability

    def backward(self, value: float) -> None:
        self.total_action_value += value
        self.visit_count += 1
        if self.parent is not None:
            self.parent.backward(-value)

    def expand(self, policy: np.ndarray) -> None:
        for action, probability in enumerate(policy):
            if probability > 0:
                next_state = self.game.play(board=self.state, action=action, player=1)
                next_state = self.game.change_perspective(board=next_state, player=-1)
                self.children.append(
                    Node(
                        game=self.game,
                        state=next_state,
                        last_action=action,
                        prior_probability=probability,
                    )
                )

    def get_upper_bound_confidence(self, child: Self) -> float:
        if child.visit_count > 0:
            mean_action_value: float = child.total_action_value / child.visit_count
        else:
            mean_action_value: float = 0
        if np.isnan(const.EXPLORATION_RATE_BASE):
            exploration_rate: float = const.EXPLORATION_RATE_INIT
        else:
            exploration_rate: float = (
                1 + self.visit_count + const.EXPLORATION_RATE_BASE
            ) / const.EXPLORATION_RATE_BASE + const.EXPLORATION_RATE_INIT
        assert self.prior_probability
        return mean_action_value + exploration_rate * child.prior_probability * np.sqrt(
            self.visit_count
        ) / (1 + child.visit_count)

    @property
    def is_expanded(self):
        return len(self.children) > 0

    def select_child(self) -> Self:
        assert len(self.children) > 0
        return max(self.children, key=self.get_upper_bound_confidence)


class MCTS:
    num_search_eval: int = const.NUM_SEARCH_EVAL
    num_search_train: int = const.NUM_SEARCH_TRAIN
    training: bool = True

    total_action_value: dict[tuple[Board, int], float] = {}
    visit_count_state: dict[Board, int] = {}
    visit_count: dict[tuple[Board, int], int] = {}

    game: Game
    net: NeuralNetwork

    def __init__(self, game: Game, net: NeuralNetwork) -> None:
        self.game = game
        self.net = net

    def eval(self) -> None:
        self.training = False

    def search(self, state: Board, player: int) -> np.ndarray:
        state = self.game.change_perspective(board=state, player=player)
        num_search: int = (
            self.num_search_train if self.training else self.num_search_eval
        )
        root: Node = Node(game=self.game, state=state, last_action=-1)
        for _ in range(num_search):
            current: Node = root
            while current.is_expanded:
                current = current.select_child()
            terminated, value = self.game.check_terminated(
                board=current.state, player=1
            )
            if not terminated:
                policy, value = self.net.predict(current.state.encode())
                mask_policy(game=self.game, state=current.state, policy=policy)
                current.expand(policy)
            current.backward(value)
        if num_search > 0:
            policy: np.ndarray = np.zeros(shape=(self.game.action_size))
            for child in root.children:
                policy[child.last_action] = child.visit_count
        else:
            policy, value = self.net.predict(state.encode())
        policy = mask_policy(game=self.game, state=state, policy=policy)
        return policy

    def train(self) -> None:
        self.training = True
