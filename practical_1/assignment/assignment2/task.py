import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Synthetic data
X = np.linspace(0, 15, 150)
y = 4 * X + 10 + np.random.randn(*X.shape) * 3

# Function to train and plot predictions for a given learning rate
def train_and_plot(lr):
    model = tf.keras.Sequential([tf.keras.layers.Dense(1, input_shape=(1,))])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss='mse')
    history = model.fit(X, y, epochs=150, verbose=0)
    y_pred = model.predict(X)
    plt.plot(X, y_pred, label=f"LR={lr}")
    return model

learning_rates = [0.001, 0.01, 0.05, 0.1]

plt.scatter(X, y, color='black', label="Actual Data")

models = []
for lr in learning_rates:
    model = train_and_plot(lr)
    models.append(model)

plt.xlabel("X")
plt.ylabel("y")
plt.title("Learning Rate Effect on Linear Regression")
plt.legend()
plt.show()

# Predictions for unseen inputs using the last trained model
X_new = np.array([16, 18, 22])
y_pred_new = models[-1].predict(X_new)
print("Predictions for new inputs:", y_pred_new.flatten())
