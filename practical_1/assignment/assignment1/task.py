import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
X = np.linspace(0, 10, 100)
y = 2.5 * X + 5 + np.random.randn(*X.shape) * 1.5

# Function to train model with different learning rates
def train_model(lr):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr), loss='mse')
    history = model.fit(X, y, epochs=100, verbose=0)
    return model, history

learning_rates = [0.001, 0.01, 0.1]
for lr in learning_rates:
    model, history = train_model(lr)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, label=f"LR={lr}")

plt.scatter(X, y, color='black', label='Actual')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.title("Effect of Learning Rate on Convergence")
plt.show()

# Predict on new unseen data
X_new = np.array([12, 15, 20])
y_new_pred = model.predict(X_new)
print("Predictions for unseen inputs:", y_new_pred.flatten())
