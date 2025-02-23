# Print the board
def print_board(board):
    for row in board:
        print(" | ".join(row))
        print("-" * 5)

# Check if a player has won
def check_winner(board, player):
    # Check rows
    for row in board:
        if all([cell == player for cell in row]):
            return True
    # Check columns
    for col in range(3):
        if all([board[row][col] == player for row in range(3)]):
            return True
    # Check diagonals
    if all([board[i][i] == player for i in range(3)]):
        return True
    if all([board[i][2 - i] == player for i in range(3)]):
        return True
    return False

# Check if the board is full
def is_full(board):
    for row in board:
        if " " in row:
            return False
    return True

# Main game function
def play_game():
    board = [[" " for _ in range(3)] for _ in range(3)]
    current_player = "X"
    
    while True:
        print_board(board)
        print(f"Player {current_player}'s turn:")
        
        # Get player input
        try:
            row = int(input("Choose row (0-2): "))
            col = int(input("Choose column (0-2): "))
        except ValueError:
            print("Please enter valid numbers.")
            continue
        
        # Validate input
        if row < 0 or row > 2 or col < 0 or col > 2 or board[row][col] != " ":
            print("Cell is already occupied or input is invalid. Try again.")
            continue
        
        # Update the board
        board[row][col] = current_player
        
        # Check for a winner
        if check_winner(board, current_player):
            print_board(board)
            print(f"Player {current_player} wins!")
            break
        
        # Check for a draw
        if is_full(board):
            print_board(board)
            print("It's a draw!")
            break
        
        # Switch player
        current_player = "O" if current_player == "X" else "X"

# Run the game
play_game()