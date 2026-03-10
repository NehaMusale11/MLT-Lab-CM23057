function predict(){

const text = document.getElementById("textInput").value.toLowerCase()

const positive = ["good","great","amazing","love","excellent"]
const negative = ["bad","worst","hate","terrible"]

let score = 0

text.split(" ").forEach(word=>{
if(positive.includes(word)) score++
if(negative.includes(word)) score--
})

let sentiment
let confidence

if(score>=0){
sentiment="Positive 😊"
confidence=(0.6 + Math.random()*0.4).toFixed(2)
}
else{
sentiment="Negative 😞"
confidence=(0.6 + Math.random()*0.4).toFixed(2)
}

document.getElementById("result").innerHTML=
"Sentiment: "+sentiment+"<br>Confidence: "+confidence

console.log("Text:",text)
console.log("Sentiment:",sentiment)
console.log("Confidence:",confidence)

}