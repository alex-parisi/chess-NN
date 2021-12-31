from inc.dataFnc import *


# Set length of dataset and depth of Stockfish evaluation
NUM_TO_SIM = 1000000
STOCKFISH_DEPTH = 0

# Initialize matrices in memory
x_train = numpy.full((NUM_TO_SIM, 14, 8, 8), 0)
y_train = numpy.full(NUM_TO_SIM, 0)

# Generate data
for i in range(NUM_TO_SIM):
    # Generate a random board that has a valid evaluation
    while 1:
        board = random_board()
        evaluation = stockfish(board, STOCKFISH_DEPTH)
        if evaluation is not None:
            board = format_board(board)
            x_train[i, :, :, :] = board
            y_train[i] = evaluation
            break
    # Print progress
    print((i / NUM_TO_SIM) * 100)

# Save as compressed .npz file
numpy.savez_compressed('../new_dataset.npz', x_train=x_train, y_train=y_train)
