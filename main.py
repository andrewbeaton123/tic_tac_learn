from game.game import TicTacToe
import numpy as np

def main():
    env = TicTacToe()
    while not env.is_game_over():
        env.print_board()
        valid_moves = env.get_valid_moves()
        print("Valid moves:", valid_moves)
        row, col = valid_moves[np.random.choice(len(valid_moves))]
        env.make_move(row, col)

    winner = env.check_winner()
    if winner == 0:
        print("It's a draw!")
    else:
        print(f"Player {winner} wins!")

if __name__ == "__main__":
    main()