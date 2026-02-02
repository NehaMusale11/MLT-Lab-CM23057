let model;
const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");

let drawing = false;

ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);

/* Canvas Drawing */
canvas.addEventListener("mousedown", () => drawing = true);
canvas.addEventListener("mouseup", () => drawing = false);
canvas.addEventListener("mousemove", draw);

function draw(e) {
    if (!drawing) return;

    ctx.fillStyle = "white";
    ctx.beginPath();
    ctx.arc(e.offsetX, e.offsetY, 10, 0, Math.PI * 2);
    ctx.fill();
}

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    document.getElementById("result").innerText = "";
}

/* Simple CNN (pretrained simulation) */
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

/* Load Model */
async function loadModel() {
    model = createCNNModel();
}

/* Predict Digit */
function predictDigit() {
    if (!model) {
        alert("Model not loaded!");
        return;
    }

    const imageData = ctx.getImageData(0, 0, 280, 280);

    const tensor = tf.browser.fromPixels(imageData, 1)
        .resizeNearestNeighbor([28, 28])
        .toFloat()
        .div(255)
        .reshape([1, 28, 28, 1]);

    const prediction = model.predict(tensor);
    const digit = prediction.argMax(1).dataSync()[0];

    document.getElementById("result").innerText =
        "Predicted Digit: " + digit;
}

loadModel();
