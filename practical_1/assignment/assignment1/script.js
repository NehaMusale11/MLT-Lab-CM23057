async function run() {
  console.log("Assignment 1: Linear Regression using TensorFlow.js");

  // Simple dataset (y = 2x)
  const X = tf.tensor2d([1, 2, 3, 4], [4, 1]);
  const y = tf.tensor2d([2, 4, 6, 8], [4, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  });

  await model.fit(X, y, { epochs: 100 });

  console.log("Training completed");

  // Prediction
  const testInput = tf.tensor2d([5], [1, 1]);
  const prediction = model.predict(testInput);

  prediction.print();
}

run();
