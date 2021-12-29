import chess
import chess.engine
import random
import numpy


# this function will create our x (board)
def random_board(max_depth=200):
    board = chess.Board()
    depth = random.randrange(0, max_depth)

    for _ in range(depth):
        all_moves = list(board.legal_moves)
        random_move = random.choice(all_moves)
        board.push(random_move)
        if board.is_game_over():
            break

    return board


# this function will create our f(x) (score)
def stockfish(board, depth):
    with chess.engine.SimpleEngine.popen_uci('C:/Users/alexp/PycharmProjects/Chess-CNN-Eval/stockfish\stockfish_14.1_win_x64_avx2.exe') as sf:
        result = sf.analyse(board, chess.engine.Limit(depth=depth))
        score = result['score'].white().score()
        return score


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


# example: h3 -> 17
def square_to_index(square):
    letter = chess.square_name(square)
    return 8 - int(letter[1]), squares_index[letter[0]]


def split_dims(board):
    # this is the 3d matrix
    board3d = numpy.zeros((14, 8, 8), dtype=numpy.int8)

    # here we add the pieces view on the matrix
    for piece in chess.PIECE_TYPES:
        for square in board.pieces(piece, chess.WHITE):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece - 1][7 - idx[0]][idx[1]] = 1
        for square in board.pieces(piece, chess.BLACK):
            idx = numpy.unravel_index(square, (8, 8))
            board3d[piece + 5][7 - idx[0]][idx[1]] = 1

    # add attacks and valid moves too
    # so the network knows what is being attacked
    aux = board.turn
    board.turn = chess.WHITE
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[12][i][j] = 1
    board.turn = chess.BLACK
    for move in board.legal_moves:
        i, j = square_to_index(move.to_square)
        board3d[13][i][j] = 1
    board.turn = aux

    return board3d

NUM_TO_SIM = 1000000
STOCKFISH_DEPTH = 0

x_train = numpy.full((NUM_TO_SIM, 14, 8, 8), 0)
y_train = numpy.full((NUM_TO_SIM), 0)
for i in range(NUM_TO_SIM):
    board = random_board()
    eval = stockfish(board, STOCKFISH_DEPTH)
    if eval is not None:
        board = split_dims(board)
        x_train[i, :, :, :] = board
        y_train[i] = eval
    print((i / NUM_TO_SIM) * 100)

numpy.savez_compressed('new_dataset.npz', x_train=x_train, y_train=y_train)