import numpy
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers


# Build the CNN model
def build_model(conv_size, conv_depth):
    # Input layer
    board3d = layers.Input(shape=(8, 8, 14))

    # Convolution Layers
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu',
                          data_format='channels_last')(x)

    # Flatten to reduce dimensionality
    x = layers.Flatten()(x)

    # Fully connected layer
    x = layers.Dense(64, 'relu')(x)

    # Sigmoid evaluation layer
    x = layers.Dense(1, 'sigmoid')(x)

    return models.Model(inputs=board3d, outputs=x)


# Load dataset from .npz file, normalize evaluation, and pull out data for testing after training
def get_dataset():
    # Load dataset
    container = numpy.load('new_dataset.npz')
    # Extract variables
    x, y = container['x_train'], container['y_train']
    # Normalize evaluation values to be between 0 and 1
    y = numpy.asarray(y / abs(y).max() / 2 + 0.5, dtype=numpy.float32)
    # Move axis to fit input format
    x = numpy.moveaxis(x, 1, 3)
    # Pull out 1000 values for testing after training
    indices = numpy.random.randint(0, x.shape[0], 1000)
    a = x[indices, :, :, :]
    b = y[indices]
    # Remove those indices from the training data
    x = numpy.delete(x, indices, axis=0)
    y = numpy.delete(y, indices, axis=0)

    return x, y, a, b
