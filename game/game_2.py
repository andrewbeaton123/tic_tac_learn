import numpy as np

class TicTacToe:
    def __init__(self,starting_player:int, board =None):
        
        self.current_player = starting_player # Player 1 starts
        self.winner =0 # currently a draw 
        if board is None:
            self.board = np.zeros((3, 3))  # 3x3 Tic Tac Toe board
        else :
            self.board = board

    def reset(self):
        self.board = np.zeros((3, 3))
        self.current_player = 1

    def get_valid_moves(self):
        return np.argwhere(self.board == 0)

    def make_move(self, row, col):
        if self.board[row, col] == 0:
            self.board[row, col] = self.current_player
            self.check_winner()
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
                self.winner = player
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
                    return self.winner