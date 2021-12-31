import chess
import chess.engine
import random
import numpy

STOCKFISH_PATH = 'C:/Users/alexp/Desktop/chess-NN/stockfish/stockfish.exe'


# Generate a random board
def random_board():
    # Create board
    board = chess.Board()
    # Set number of turns to randomly play
    depth = random.randrange(0, 200)
    # Play a random move until either the game ends or the depth is reached
    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = random.choice(all_moves)
        board.push(random_move)
        if board.is_game_over():
            break

    return board


# Evaluate the board using Stockfish at the specified depth
def stockfish(board, depth):
    # Load Stockfish engine
    with chess.engine.SimpleEngine.popen_uci(STOCKFISH_PATH) as sf:
        # Analyze board
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        # Get evaluation score
        score = result['score'].white().score()

        return score


# Board indices
squares_index = {
    'a': 0,
    'b': 1,
    'c': 2,
    'd': 3,
    'e': 4,
    'f': 5,
    'g': 6,
    'h': 7
}


# Map chessboard square to bitboard index, i.e. h3 -> 17
def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


# Format chessboard object
# board3d array contains 14 8x8 bitboards, one for each piece of each color (12) and then the possible moves for each
# player.
# For info on bitboards: https://www.chessprogramming.org/Bitboards
def format_board(board):
    # Initialize matrix in memory
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)

    # Generate bitboards for each piece on the board
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # Find all moves for white and black and add them to the 3D array
    current_turn = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = current_turn

    return board3d
