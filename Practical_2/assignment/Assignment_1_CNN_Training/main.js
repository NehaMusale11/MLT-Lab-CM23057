async function loadMNIST() {
    const mnist = await tf.data.mnist();
    const data = mnist.nextTrainBatch(60000);

    return {
        images: data.xs.reshape([60000, 28, 28, 1]),
        labels: data.labels
    };
}

function createCNNModel() {
    const model = tf.sequential();

    model.add(tf.layers.conv2d({
        inputShape: [28, 28, 1],
        filters: 32,
        kernelSize: 3,
        activation: "relu"
    }));

    model.add(tf.layers.maxPooling2d({ poolSize: 2 }));
    model.add(tf.layers.flatten());

    model.add(tf.layers.dense({ units: 128, activation: "relu" }));
    model.add(tf.layers.dense({ units: 10, activation: "softmax" }));

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    return model;
}

async function trainCNN() {
    const data = await loadMNIST();
    const model = createCNNModel();

    console.log("Training started...");

    await model.fit(data.images, data.labels, {
        epochs: 5,
        batchSize: 128,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(
                    `Epoch ${epoch + 1} → Accuracy: ${(logs.acc * 100).toFixed(2)}%`
                );
            }
        }
    });

    console.log("Training completed.");
}

trainCNN();
