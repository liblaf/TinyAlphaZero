import dataclasses

import numpy as np

from .game import CANONICAL_PLAYER, Board, Game
from .neural_network import NeuralNetwork
from .utils.policy import mask as mask_policy


@dataclasses.dataclass(kw_only=True)
class MCTSConfig:
    exploration_rate_base: float = np.nan
    exploration_rate_init: float = np.sqrt(2)
    num_simulation_eval: int = 0
    num_simulation_train: int = 128


class MCTS:
    action_visit_count: dict[tuple[Board, int], int]
    mean_action_value: dict[tuple[Board, int], float]
    prior_probability: dict[Board, np.ndarray]
    state_visit_count: dict[Board, int]

    config: MCTSConfig = MCTSConfig()
    game: Game
    net: NeuralNetwork
    training: bool = True

    def __init__(
        self, game: Game, net: NeuralNetwork, config: MCTSConfig = MCTSConfig()
    ) -> None:
        self.action_visit_count = {}
        self.mean_action_value = {}
        self.prior_probability = {}
        self.state_visit_count = {}

        self.config = config
        self.game = game
        self.net = net

    def eval(self) -> None:
        self.training = False

    def search(self, board: Board, player: int) -> np.ndarray:
        canonical_state: Board = self.game.canonicalize(board=board, player=player)
        num_simulations: int = (
            self.config.num_simulation_train
            if self.training
            else self.config.num_simulation_eval
        )
        for _ in range(num_simulations):
            self.simulate(canonical_state=canonical_state)
        if num_simulations > 1:
            policy: np.ndarray = np.array(
                [
                    self.action_visit_count.get((canonical_state, action), 0)
                    for action in range(self.game.action_size)
                ]
            )
        else:
            policy, value = self.net.predict(canonical_state.encode())
        valid_actions: np.ndarray = self.game.get_valid_actions(
            board=canonical_state, player=CANONICAL_PLAYER
        )
        policy: np.ndarray = mask_policy(policy=policy, valid_actions=valid_actions)
        return policy

    def simulate(self, canonical_state: Board) -> float:
        terminated, value = self.game.check_terminated(
            board=canonical_state, player=CANONICAL_PLAYER
        )
        if terminated:
            return value
        if canonical_state not in self.prior_probability:
            policy, value = self.net.predict(canonical_state.encode())
            valid_actions: np.ndarray = self.game.get_valid_actions(
                board=canonical_state, player=CANONICAL_PLAYER
            )
            policy: np.ndarray = mask_policy(policy=policy, valid_actions=valid_actions)
            self.prior_probability[canonical_state] = policy
            return value
        action: int = self._select_action(canonical_state=canonical_state)
        next_board: Board = self.game.play(
            board=canonical_state, action=action, player=CANONICAL_PLAYER
        )
        next_player: int = self.game.get_next_player(player=CANONICAL_PLAYER)
        next_canonical_state: Board = self.game.canonicalize(
            board=next_board, player=next_player
        )
        next_value: float = self.simulate(canonical_state=next_canonical_state)
        value: float = self.game.canonicalize_value(
            value=next_value, player=next_player
        )
        action_visit_count: int = self.action_visit_count.get(
            (canonical_state, action), 0
        )
        mean_action_value: float = self.mean_action_value.get(
            (canonical_state, action), 0
        )
        state_visit_count: int = self.state_visit_count.get(canonical_state, 0)
        self.action_visit_count[(canonical_state, action)] = action_visit_count + 1
        self.mean_action_value[(canonical_state, action)] = (
            mean_action_value * action_visit_count + value
        ) / (action_visit_count + 1)
        self.state_visit_count[canonical_state] = state_visit_count + 1
        return value

    def train(self) -> None:
        self.training = True

    def _get_upper_bound_confidence(self, canonical_state: Board, action: int) -> float:
        action_visit_count: int = self.action_visit_count.get(
            (canonical_state, action), 0
        )
        mean_action_value: float = self.mean_action_value.get(
            (canonical_state, action), 0
        )
        prior_probability: float = self.prior_probability[canonical_state][action]
        state_visit_count: int = self.state_visit_count.get(canonical_state, 0)
        if np.isnan(self.config.exploration_rate_base):
            exploration_rate: float = self.config.exploration_rate_init
        else:
            exploration_rate: float = (
                np.log(
                    (1 + state_visit_count + self.config.exploration_rate_base)
                    / self.config.exploration_rate_base
                )
                / (1 + state_visit_count)
                + self.config.exploration_rate_init
            )
        return mean_action_value + exploration_rate * prior_probability * np.sqrt(
            state_visit_count
        ) / (1 + action_visit_count)

    def _select_action(self, canonical_state: Board) -> int:
        valid_actions: np.ndarray = self.game.get_valid_actions(
            board=canonical_state, player=CANONICAL_PLAYER
        )
        best_action: int = -1
        best_ucb: float = -np.inf
        for action in np.where(valid_actions)[0]:
            ucb: float = self._get_upper_bound_confidence(
                canonical_state=canonical_state, action=action
            )
            if ucb > best_ucb:
                best_action = action
                best_ucb = ucb
        assert best_action != -1
        return best_action
