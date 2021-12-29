# chess-NN

A project with the goal of training a deep neural network, i.e. a convolutional neural network (CNN),
to play the game of chess. Currently, I am working on implementing a board evaluation function using a CNN.

The board evaluation CNN was trained using randomly generated legal positions and Stockfish's evaluation.

The next step is to train a separate CNN on the likelihood that a move is "good", again using Stockfish's evaluation.

### Config

In order to run this project, execute the following inside the chess-NN directory:

```
pip install -r requirements.txt
```

This will install the required packages. You must also configure TensorFlow to work properly with a GPU (if desired):

