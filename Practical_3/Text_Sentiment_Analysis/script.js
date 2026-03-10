function analyzeSentiment(){

const text = document.getElementById("textInput").value.toLowerCase()

const positiveWords = ["good","great","amazing","love","excellent","happy","awesome"]
const negativeWords = ["bad","worst","hate","terrible","sad","awful"]

let score = 0

const words = text.split(" ")

words.forEach(word => {

if(positiveWords.includes(word)){
score++
}

if(negativeWords.includes(word)){
score--
}

})

let sentiment

if(score >= 0){
sentiment = "Positive 😊"
}
else{
sentiment = "Negative 😞"
}

document.getElementById("result").innerText = "Result: " + sentiment

console.log("Input Text:", text)
console.log("Sentiment:", sentiment)

}