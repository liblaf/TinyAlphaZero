from .abc import Player
from .alpha_zero import Player as PlayerAlphaZero
from .random import Player as PlayerRandom

PLAYER_LIST: list[str] = ["AlphaZero", "Random"]
