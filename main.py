import numpy
import tensorflow as tf
import tensorflow.keras.models as models
import tensorflow.keras.layers as layers
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
from matplotlib import pyplot as plt


def build_model(conv_size, conv_depth):
    board3d = layers.Input(shape=(8, 8, 14))

    # adding the convolutional layers
    x = board3d
    for _ in range(conv_depth):
        x = layers.Conv2D(filters=conv_size, kernel_size=3, padding='same', activation='relu',
                          data_format='channels_last')(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, 'relu')(x)
    x = layers.Dense(1, 'sigmoid')(x)

    return models.Model(inputs=board3d, outputs=x)


def get_dataset():
    container = numpy.load('dataset.npz')
    b, v = container['b'], container['v']
    v = numpy.asarray(v / abs(v).max() / 2 + 0.5, dtype=numpy.float32)  # normalization (0 - 1)
    return b, v


physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

model = build_model(32, 4)
# utils.plot_model(model, to_file='model_plot.png', show_shapes=True, show_layer_names=False)
x_train, y_train = get_dataset()
x_train = numpy.moveaxis(x_train, 1, 3)

numOfValidationSet = int(x_train.shape[0] * 0.1)
indices = numpy.random.randint(0, x_train.shape[0], numOfValidationSet)
x_validation = x_train[indices, :, :, :]
y_validation = y_train[indices]
x_train = numpy.delete(x_train, indices, axis=0)
y_train = numpy.delete(y_train, indices, axis=0)

indices = numpy.random.randint(0, x_train.shape[0], 1000)
x_test = x_train[indices, :, :, :]
y_test = y_train[indices]
x_train = numpy.delete(x_train, indices, axis=0)
y_train = numpy.delete(y_train, indices, axis=0)

print(x_train.shape)
print(y_train.shape)

print(x_validation.shape)
print(y_validation.shape)

# model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error', metrics='accuracy')
# model.summary()
# history = model.fit(x_train, y_train,
#                     batch_size=2048,
#                     epochs=1000,
#                     verbose=1,
#                     validation_split=0.1,
#                     validation_data=(x_validation, y_validation),
#                     callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
#                                callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-4)])

# model.save('model.h5')

model = models.load_model('model.h5')

y_pred = model.predict(x_test[:100, :, :, :])

plt.figure(0)
plt.plot(y_test[:100])
plt.plot(y_pred)
plt.show()