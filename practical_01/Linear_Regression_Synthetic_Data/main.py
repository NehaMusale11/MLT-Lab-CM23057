import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data
np.random.seed(42)
X = np.linspace(0, 10, 100)
y = 3 * X + 7 + np.random.randn(*X.shape) * 2

# Define simple linear regression model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,))
])

model.compile(optimizer='adam', loss='mse')

# Train model
history = model.fit(X, y, epochs=100, verbose=0)

# Plot predicted vs actual
y_pred = model.predict(X)
plt.scatter(X, y, label="Actual")
plt.plot(X, y_pred, color='r', label="Predicted")
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.show()

print("Training complete. Model weights:", model.get_weights())
