async function loadMNIST() {
    const mnist = await tf.data.mnist();
    const data = mnist.nextTrainBatch(60000);

    return {
        images: data.xs.reshape([60000, 28, 28, 1]),
        flatImages: data.xs.reshape([60000, 784]),
        labels: data.labels
    };
}

/* CNN MODEL */
function createCNN() {
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

/* DENSE MODEL */
function createDense() {
    const model = tf.sequential();

    model.add(tf.layers.dense({
        inputShape: [784],
        units: 128,
        activation: "relu"
    }));

    model.add(tf.layers.dense({
        units: 10,
        activation: "softmax"
    }));

    model.compile({
        optimizer: "adam",
        loss: "categoricalCrossentropy",
        metrics: ["accuracy"]
    });

    return model;
}

async function compareModels() {
    const data = await loadMNIST();

    /* Train CNN */
    const cnn = createCNN();
    console.log("Training CNN...");

    await cnn.fit(data.images, data.labels, {
        epochs: 5,
        batchSize: 128
    });

    const cnnAcc = cnn.history.history.acc.pop();

    /* Train Dense */
    const dense = createDense();
    console.log("Training Dense Network...");

    await dense.fit(data.flatImages, data.labels, {
        epochs: 5,
        batchSize: 128
    });

    const denseAcc = dense.history.history.acc.pop();

    console.log("===== ACCURACY COMPARISON =====");
    console.log(`CNN Accuracy   : ${(cnnAcc * 100).toFixed(2)}%`);
    console.log(`Dense Accuracy : ${(denseAcc * 100).toFixed(2)}%`);
}

compareModels();
