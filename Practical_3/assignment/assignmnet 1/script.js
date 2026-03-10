let model

async function trainModel(){

const xs = tf.tensor2d([
[1,1,1],
[1,0,1],
[0,1,0],
[0,0,1]
])

const ys = tf.tensor2d([
[1],
[1],
[0],
[0]
])

model = tf.sequential()

model.add(tf.layers.dense({
units:4,
inputShape:[3],
activation:'relu'
}))

model.add(tf.layers.dense({
units:1,
activation:'sigmoid'
}))

model.compile({
optimizer:'adam',
loss:'binaryCrossentropy',
metrics:['accuracy']
})

document.getElementById("status").innerText="Training..."

await model.fit(xs,ys,{epochs:100})

document.getElementById("status").innerText="Training Completed ✅"

console.log("Model trained successfully")
}