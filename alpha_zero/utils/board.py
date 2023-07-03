from alpha_zero.game import CANONICAL_PLAYER, Board, Game
from alpha_zero.player import Player, PlayerRandom


def random(game: Game, num_moves: int = 32) -> Board:
    board: Board = game.get_init_board()
    player: Player = PlayerRandom(game=game)
    current_player: int = CANONICAL_PLAYER
    for _ in range(num_moves):
        action: int = player.play(board=board, player=current_player)
        board = game.play(board=board, action=action, player=current_player)
        current_player = game.get_next_player(player=current_player)
    return board
