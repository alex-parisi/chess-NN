import tensorflow as tf
import tensorflow.keras.optimizers as optimizers
import tensorflow.keras.callbacks as callbacks
from matplotlib import pyplot as plt
from inc.nnFnc import *


# Enable memory growth if using a GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

# Build model
model = build_model(32, 4)

# Load dataset
x_train, y_train, x_test, y_test = get_dataset()

# Compile model
model.compile(optimizer=optimizers.Adam(5e-4), loss='mean_squared_error')

# Train model
history = model.fit(x_train, y_train,
                    batch_size=2048,
                    epochs=1000,
                    verbose=1,
                    validation_split=0.1,
                    callbacks=[callbacks.ReduceLROnPlateau(monitor='loss', patience=10),
                               callbacks.EarlyStopping(monitor='loss', patience=15, min_delta=1e-5)])

# Save model
model.save('model.h5')
# model = models.load_model('model.h5')

# Use model to predict board values
y_prediction = model.predict(x_test[:100, :, :, :])

# Plot results
plt.figure(0)
plt.plot(y_test[:100])
plt.plot(y_prediction)
plt.show()

# Plot model training
plt.figure(1)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training', 'Validation'], loc='upper left')
plt.show()
