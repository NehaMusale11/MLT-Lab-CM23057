// Assignment 2 - TensorFlow.js Starter

// Example: Create a 3x3 matrix and print
const matrix3x3 = tf.tensor2d([[1,2,3],[4,5,6],[7,8,9]]);
console.log("3x3 Matrix:");
matrix3x3.print();

// Example: Perform addition with another matrix
const matrixB = tf.tensor2d([[9,8,7],[6,5,4],[3,2,1]]);
const addedMatrix = tf.add(matrix3x3, matrixB);
console.log("Added Matrix:");
addedMatrix.print();

// Example: Reshape 3x3 to 1x9
const reshapedMatrix = matrix3x3.reshape([1,9]);
console.log("Reshaped 1x9 Matrix:");
reshapedMatrix.print();

// Example: Flatten matrix
const flattenedMatrix = matrix3x3.flatten();
console.log("Flattened Matrix:");
flattenedMatrix.print();
