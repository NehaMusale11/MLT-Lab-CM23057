let model;

async function loadModel() {
    model = await mobilenet.load();
    document.getElementById("status").innerText =
        "Model Loaded Successfully";
    console.log("MobileNet Loaded Successfully");
}

loadModel();

function changeImage() {
    const selected = document.getElementById("imageSelect").value;
    document.getElementById("image").src = "images/" + selected;
}

async function classifyImage() {

    const img = document.getElementById("image");

    const predictions = await model.classify(img, 3);

    const result = document.getElementById("result");
    result.innerHTML = "";

    console.log("Top 3 Predictions:");

    predictions.forEach((pred, index) => {

        const text =
            (index + 1) + ". " +
            pred.className +
            " - " +
            pred.probability.toFixed(3);

        const li = document.createElement("li");
        li.innerText = text;
        result.appendChild(li);

        console.log(text);
    });

    console.log("-----------------------------");
}