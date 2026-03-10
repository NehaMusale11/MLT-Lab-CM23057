function runDense(){

const model = tf.sequential()

model.add(tf.layers.dense({
units:4,
inputShape:[3],
activation:'relu'
}))

model.add(tf.layers.dense({
units:1,
activation:'sigmoid'
}))

document.getElementById("output").innerText=
"Dense Model: Fast training but less context understanding"

console.log("Dense model executed")
}

function runRNN(){

const model = tf.sequential()

model.add(tf.layers.simpleRNN({
units:4,
inputShape:[3,1]
}))

model.add(tf.layers.dense({
units:1,
activation:'sigmoid'
}))

document.getElementById("output").innerText=
"RNN Model: Better for sequence data like sentences"

console.log("RNN model executed")
}