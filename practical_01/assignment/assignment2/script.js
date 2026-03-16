async function trainWithLR(lr) {
  const X = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
  const y = tf.tensor2d([17, 22, 27, 32, 37], [5, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: tf.train.sgd(lr),
    loss: 'meanSquaredError'
  });

  console.log(`\nTraining with Learning Rate = ${lr}`);

  await model.fit(X, y, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 25 === 0) {
          console.log(
            `Epoch ${epoch} | Loss = ${logs.loss.toFixed(4)}`
          );
        }
      }
    }
  });

  return model;
}

async function run() {
  // Learning rate analysis
  await trainWithLR(0.001);
  const model = await trainWithLR(0.01);
  await trainWithLR(0.1);

  // Predicted vs Actual
  const X = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
  const y = tf.tensor2d([17, 22, 27, 32, 37], [5, 1]);

  const predictions = model.predict(X);

  const actual = Array.from(y.dataSync());
  const predicted = Array.from(predictions.dataSync());

  console.log("\nPredicted vs Actual:");
  actual.forEach((v, i) => {
    console.log(`Actual: ${v}, Predicted: ${predicted[i].toFixed(2)}`);
  });

  // Unseen data prediction
  const unseen = tf.tensor2d([10, 12], [2, 1]);
  const unseenPred = model.predict(unseen).dataSync();

  console.log("\nUnseen Input Prediction:");
  console.log(`Input: 10 → Predicted: ${unseenPred[0].toFixed(2)}`);
  console.log(`Input: 12 → Predicted: ${unseenPred[1].toFixed(2)}`);

  // Plot
  const ctx = document.getElementById("chart").getContext("2d");
  new Chart(ctx, {
    type: "scatter",
    data: {
      datasets: [
        {
          label: "Actual",
          data: actual.map((y, x) => ({ x: x + 1, y })),
          backgroundColor: "blue"
        },
        {
          label: "Predicted",
          data: predicted.map((y, x) => ({ x: x + 1, y })),
          backgroundColor: "red"
        }
      ]
    },
    options: {
      scales: {
        x: { title: { display: true, text: "Input X" } },
        y: { title: { display: true, text: "Output Y" } }
      }
    }
  });
}

run();
