import numpy as np
import random 


class TicTacToe:
    def __init__(self):
        self.board = np.zeros((3, 3))  # 3x3 Tic Tac Toe board
        self.current_player = random.choice([0,1])  # Player 1 starts

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.current_player = 3 - self.current_player  # Switch players (1 -> 2, 2 -> 1)
        else:
            raise ValueError("Invalid move")

    def check_winner(self):
        for player in [1, 2]:
            # Check rows, columns, and diagonals for a win
            if np.any(np.all(self.board == player, axis=0)) or \
               np.any(np.all(self.board == player, axis=1)) or \
               np.all(np.diag(self.board) == player) or \
               np.all(np.diag(np.fliplr(self.board)) == player):
                return player
        if len(self.get_valid_moves()) == 0:
            return 0  # Draw
        return None  # Game is ongoing

    def is_game_over(self):
        return self.check_winner() is not None

    def print_board(self):
        for row in self.board:
            print(" | ".join(["X" if cell == 1 else "O" if cell == 2 else " " for cell in row]))
            print("-" * 9)

    def step(self, action):
        if not self.is_game_over() and action in self.get_valid_moves():
            row, col = action
            self.make_move(row, col)
            if self.is_game_over():
                if self.winner == 1:
                    return self.board, 1, True  # Player 1 wins
                elif self.winner == 2:
                    return self.board, -1, True  # Player 2 wins
                else:
                    return self.board, 0, True  # Draw
            else:
                return self.board, 0, False  # Game continues
        else:
            return self.board, -1, True  # Invalid move or game already over


# Example usage:
if __name__ == "__main__":
    
    winner_log = {"0":0, "1":0,"2":0}
    for  x in range((10000)):
        env = TicTacToe()
        while not env.is_game_over():
            env.print_board()
            valid_moves = env.get_valid_moves()
            print("Valid moves:", valid_moves)
            row, col = valid_moves[np.random.choice(len(valid_moves))]
            env.make_move(row, col)

        winner = env.check_winner()
        #if winner == 0:
        #    print("It's a draw!")
        #else:
        #    print(f"Player {winner} wins!")
        
        winner_log[str(winner)] +=1
        env.reset()
print(winner_log)

    
