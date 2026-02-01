// Assignment 3 - TensorFlow.js Starter

// Example: Create scalar, vector, and matrix
const scalarA = tf.scalar(15);
const vectorA = tf.tensor1d([10,20,30]);
const matrixA = tf.tensor2d([[1,2],[3,4]]);

console.log("Scalar:");
scalarA.print();
console.log("Vector:");
vectorA.print();
console.log("Matrix:");
matrixA.print();

// Example: Element-wise multiplication of vectors
const vectorB = tf.tensor1d([2,3,4]);
const vectorProduct = tf.mul(vectorA, vectorB);
console.log("Vector Multiplication:");
vectorProduct.print();

// Example: Reshape matrix
const reshapedMatrixA = matrixA.reshape([4,1]);
console.log("Reshaped 4x1 Matrix:");
reshapedMatrixA.print();
