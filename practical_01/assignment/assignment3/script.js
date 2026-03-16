async function run() {
  console.log("Assignment 3: Model Evaluation using TFJS");

  // Dataset: y = 4x + 10
  const X = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
  const y = tf.tensor2d([14, 18, 22, 26, 30], [5, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  });

  await model.fit(X, y, { epochs: 150 });

  console.log("Training completed");

  // Test inputs
  const testInputs = [6, 7, 8];
  const expected = testInputs.map(x => 4 * x + 10);

  const predictions = model.predict(
    tf.tensor2d(testInputs, [testInputs.length, 1])
  ).dataSync();

  console.log("\nPredicted vs Expected vs Error:");

  testInputs.forEach((x, i) => {
    const error = Math.abs(predictions[i] - expected[i]);
    console.log(
      `Input: ${x}, Predicted: ${predictions[i].toFixed(2)}, Expected: ${expected[i]}, Error: ${error.toFixed(2)}`
    );
  });
}

run();
