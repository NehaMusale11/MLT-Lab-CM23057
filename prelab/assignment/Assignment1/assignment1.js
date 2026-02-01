// Assignment 1 - Hello TensorFlow.js

// 1. Create tensors of different dimensions
const scalar = tf.scalar(10);
const vector = tf.tensor1d([1, 2, 3]);
const matrix = tf.tensor2d([[1, 2], [3, 4]]);

console.log("Scalar:");
scalar.print();
console.log("Vector:");
vector.print();
console.log("Matrix:");
matrix.print();

// 2. Element-wise addition and multiplication on two vectors
const vec1 = tf.tensor1d([1, 2, 3]);
const vec2 = tf.tensor1d([4, 5, 6]);

const sum = tf.add(vec1, vec2);
const product = tf.mul(vec1, vec2);

console.log("Vector Addition:");
sum.print();
console.log("Vector Multiplication:");
product.print();

// 3. Tensor reshaping and flattening
const original = tf.tensor2d([[1, 2], [3, 4]]);
const reshaped = original.reshape([4, 1]);
const flattened = original.flatten();

console.log("Original Tensor:");
original.print();
console.log("Reshaped Tensor (4x1):");
reshaped.print();
console.log("Flattened Tensor:");
flattened.print();
