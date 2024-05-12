from src.game.game_2 import TicTacToe
import streamlit as st

game = TicTacToe(1)  # Create a new Tic Tac Toe game

# Streamlit App
st.title("Tic Tac Toe")

# Display the board
for row in range(3):
  for col in range(3):
    if game.board[row, col] == 0:
      # Create a button for each empty space on the board
      if st.button(f"{row},{col}", key=f"{row}-{col}"):
        # Get the row and column from the button text
        row, col = map(int, st.session_state[f"{row}-{col}"].split(","))
        try:
          game.make_move(row, col)
        except ValueError:
          pass  # Handle invalid moves silently
    else:
      # Display the X or O on the board
      st.write(
          "X" if game.board[row, col] == 1 else "O"
      )
  st.write("\n")

# Check for a winner or draw
if game.is_game_over():
  if game.winner == 1:
    st.write("Player 1 wins!")
  elif game.winner == 2:
    st.write("Player 2 wins!")
  else:
    st.write("It's a draw!")

# Reset button
if st.button("Reset Game"):
  game.reset()