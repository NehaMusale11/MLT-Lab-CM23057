async function trainModel(learningRate) {
  const X = tf.tensor2d([1, 2, 3, 4, 5], [5, 1]);
  const y = tf.tensor2d([17, 22, 27, 32, 37], [5, 1]);

  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 1, inputShape: [1] }));

  model.compile({
    optimizer: tf.train.sgd(learningRate),
    loss: 'meanSquaredError'
  });

  console.log(`\nTraining with learning rate = ${learningRate}`);

  await model.fit(X, y, {
    epochs: 100,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        if (epoch % 20 === 0) {
          console.log(
            `LR ${learningRate} | Epoch ${epoch} | Loss ${logs.loss.toFixed(4)}`
          );
        }
      }
    }
  });
}

async function run() {
  await trainModel(0.001);
  await trainModel(0.01);
  await trainModel(0.1);
}

run();
