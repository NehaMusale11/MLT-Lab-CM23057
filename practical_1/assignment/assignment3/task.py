import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
X = np.linspace(0, 20, 200)
y = 5 * X + 12 + np.random.randn(*X.shape) * 4

# Train model
model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=200, verbose=0)

# Predictions on training data
y_pred_train = model.predict(X)

plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, y_pred_train, color='red', label='Predicted')
plt.xlabel("X")
plt.ylabel("y")
plt.title("Predicted vs Actual Values")
plt.legend()
plt.show()

# Predict on unseen data
X_new = np.array([25, 28, 30])
y_pred_new = model.predict(X_new)
expected = 5 * X_new + 12  # expected values based on synthetic formula

for x, pred, exp in zip(X_new, y_pred_new.flatten(), expected):
    print(f"Input: {x}, Predicted: {pred:.2f}, Expected: {exp}")
