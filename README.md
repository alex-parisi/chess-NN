# chess-NN

A project with the goal of training a deep neural network, i.e. a convolutional neural network (CNN),
to play the game of chess. Currently, I am working on implementing a board evaluation function using a CNN.

The board evaluation CNN was trained using randomly generated legal positions and Stockfish's evaluation.

The next step is to train a separate CNN on the likelihood that a move is "good", again using Stockfish's evaluation. This likelihood will be used as the policy
metric for a Monte Carlo tree search.

### Config

This project was written using Python 3.7.9 (64-bit), and has not been tested with other versions of Python. Note that *all* 32-bit installations of Python will not work.

Development Machine:
- Windows 10 64-bit
- Intel Core i7-7700K CPU @ 4.2 GHz
- 16 GB RAM @ 2400 MHz
- NVIDIA GeForce GTX 970

I imagine this project should be compatible with Linux, given that the correct TensorFlow installation instructions are followed.

In order to run this project, first execute the following inside the chess-NN directory:
```
pip install -r requirements.txt
```

This will install the required packages. You must also configure TensorFlow to work properly with a GPU (if desired).
Follow [this link for more information](https://www.tensorflow.org/install/gpu).

- Scroll to "Software requirements" and follow **ALL** installation instructions for the following:
  - NVIDIA GPU drivers -- latest
  - CUDA Toolkit -- version 11.0
  - CUPTI -- not required, installed with CUDA Toolkit
  - cuDNN SDK -- version 8.2.0
  - TensorRT 7 -- not required
- Scroll to "Linux setup" or "Windows setup" and follow the instructions to complete the installation.

This option can be skipped, but the neural network will train very slowly without access to a GPU.

### Running

To generate the dataset, first you must specify the path of the Stockfish engine. You can [download the engine here](https://stockfishchess.org/download/).
Then, in the dataFnc.py file, change the "STOCKFISH_PATH" value to the location of the Stockfish executable.

Then run:
```
generateData.py
```
The result will be a very large .npz dataset containing 1,000,000 chess board positions and their Stockfish evaluation at the specified depth (default = 0).

The process to generate the dataset takes quite a bit of time (~= 20 hours), so you can instead use the dataset provided in the repository.

Then run:
```
trainModel.py
```
This will create the neural network, train it using the dataset generated above, and then evaluate 100 test cases to validate accuracy. 
It will also export the trained model for future use. 
