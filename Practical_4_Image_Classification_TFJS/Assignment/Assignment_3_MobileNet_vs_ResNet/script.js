let modelV1;
let modelV2;

async function loadModels() {

    modelV1 = await mobilenet.load({
        version: 1,
        alpha: 1.0   // REQUIRED FIX
    });

    modelV2 = await mobilenet.load({
        version: 2,
        alpha: 1.0
    });

    document.getElementById("status").innerText =
        "Models Loaded Successfully ✅";

    console.log("MobileNet V1 and V2 Loaded Successfully");
}

loadModels();

function changeImage() {
    const selected = document.getElementById("imageSelect").value;
    document.getElementById("image").src = "images/" + selected;
}

async function compareModels() {

    const img = document.getElementById("image");

    document.getElementById("resultV1").innerHTML = "";
    document.getElementById("resultV2").innerHTML = "";

    // ⏱ Measure V1
    const startV1 = performance.now();
    const predictionsV1 = await modelV1.classify(img, 3);
    const endV1 = performance.now();
    const timeV1 = (endV1 - startV1).toFixed(2);

    // ⏱ Measure V2
    const startV2 = performance.now();
    const predictionsV2 = await modelV2.classify(img, 3);
    const endV2 = performance.now();
    const timeV2 = (endV2 - startV2).toFixed(2);

    console.log("MobileNet V1 Predictions:");
    predictionsV1.forEach((pred, i) => {

        const text = `${i+1}. ${pred.className} - ${pred.probability.toFixed(3)}`;
        console.log(text);

        const li = document.createElement("li");
        li.innerText = text;
        document.getElementById("resultV1").appendChild(li);
    });

    console.log("MobileNet V2 Predictions:");
    predictionsV2.forEach((pred, i) => {

        const text = `${i+1}. ${pred.className} - ${pred.probability.toFixed(3)}`;
        console.log(text);

        const li = document.createElement("li");
        li.innerText = text;
        document.getElementById("resultV2").appendChild(li);
    });

    document.getElementById("speedV1").innerText =
        "⏱ Time Taken: " + timeV1 + " ms";

    document.getElementById("speedV2").innerText =
        "⏱ Time Taken: " + timeV2 + " ms";

    console.log("V1 Time:", timeV1, "ms");
    console.log("V2 Time:", timeV2, "ms");
}