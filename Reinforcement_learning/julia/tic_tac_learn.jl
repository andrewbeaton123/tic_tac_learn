using Random

# Tic Tac Toe struct and functions (same as before)
mutable struct TicTacToe
    board::Matrix{Int}
    current_player::Int
    winner::Int
end

function TicTacToe()::TicTacToe
    return TicTacToe(zeros(Int, 3, 3), 1, 0)
end

function reset!(game::TicTacToe)
    game.board .= 0
    game.current_player = 1
    game.winner = 0
end

function get_valid_moves(game::TicTacToe)
    return findall(x -> x == 0, game.board)
end

function make_move!(game::TicTacToe, row, col)
    if game.board[row, col] == 0
        game.board[row, col] = game.current_player
        check_winner(game.board)
        game.current_player = 3 - game.current_player
    else
        throw(ArgumentError("Invalid move"))
    end
end


function check_winner(board)
    n = size(board, 1)
    for player in [1, 2]
        # Check rows, columns, and main diagonal for a win
        println(board)
        println(board[1:4:end])
        println(board[n:n-1:end-n+1])
        if any(all(board .== player, dims=1)) ||  # Check rows
           any(all(board .== player, dims=2)) ||  # Check columns
           all(board[1:n+1:end] .== player)  ||
           all(board[n:n-1:end-n+1] .== player) # Check main diagonal
            return player
        end
    end
    return 0  # Draw
end

function is_game_over(game::TicTacToe)
    return game.winner != 0
end

function print_board(game::TicTacToe)
    for row in 1:3
        println(" | " * join([cell == 1 ? "X" : cell == 2 ? "O" : " " for cell in game.board[row, :]]))


        println("-" ^ 17)
    end
end

function step!(game::TicTacToe, action)
    if !is_game_over(game) && action in get_valid_moves(game)
        println(action)
        row, col = Tuple(action)
        println(row,col)
        make_move!(game, row, col)
        if is_game_over(game) && game.winner == 1
            return
        end
    end
end

# Q-learning agent
struct QAgent
    q_values::Dict{Tuple{NTuple{2,Int}, Int}, Float64}
    epsilon::Float64
    alpha::Float64
    gamma::Float64
end

function QAgent(epsilon::Float64 = 0.1, alpha::Float64 = 0.1, gamma::Float64 = 0.9)::QAgent
    q_values = Dict{Tuple{NTuple{2,Int}, Int}, Float64}()
    return QAgent(q_values, epsilon, alpha, gamma)
end

function get_q_value(agent::QAgent, state, action)
    state_action = (state, action)
    return get(agent.q_values, state_action, 0.0)
end

function update_q_value!(agent::QAgent, state, action, new_value)
    state_action = (state, action)
    agent.q_values[state_action] = new_value
end

function epsilon_greedy_policy(agent::QAgent, state, valid_moves)
    if rand() < agent.epsilon
        return rand(valid_moves)
    else
        q_values = [get_q_value(agent, state, action) for action in valid_moves]
        return valid_moves[argmax(q_values)]
    end
end

# Monte Carlo training
function train_monte_carlo(agent::QAgent, episodes)
    for _ in 1:episodes
        game = TicTacToe()
        episode_data = []
        while !is_game_over(game)
            state = copy(game.board)
            valid_moves = get_valid_moves(game)
            action = epsilon_greedy_policy(agent, state, valid_moves)
            step!(game, action)
            push!(episode_data, (state, action))
        end
        update_q_values!(agent, episode_data, game.winner)
    end
end

function update_q_values!(agent::QAgent, episode_data, winner)
    G = winner == 1 ? 1.0 : 0.0
    for (state, action) in reverse(episode_data)
        G = agent.gamma * G
        current_value = get_q_value(agent, state, action)
        new_value = current_value + agent.alpha * (G - current_value)
        update_q_value!(agent, state, action, new_value)
    end
end

# Play a game against the trained agent
function play_game(agent::QAgent)
    game = TicTacToe()
    while !is_game_over(game)
        print_board(game)
        if game.current_player == 1
            println("Your turn! Enter row and column (1-3) separated by space:")
            action = tuple(parse(Int, x) for x in split(readline(), ' '))
        else
            println("AI's turn...")
            action = play_ai_move(agent, game)
        end
        step!(game, action)
    end
    print_board(game)
    if game.winner == 1
        println("You win!")
    elseif game.winner == 2
        println("AI wins!")
    else
        println("It's a draw!")
    end
end

function play_ai_move(agent::QAgent, game::TicTacToe)
    valid_moves = get_valid_moves(game)
    action = epsilon_greedy_policy(agent, game.board, valid_moves)
    return action
end

# Main training and testing
function main()
    agent = QAgent()

    println("Training the agent...")
    train_monte_carlo(agent, 10000)

    println("Test of your training logic...")
    play_game(agent)
end

main()
