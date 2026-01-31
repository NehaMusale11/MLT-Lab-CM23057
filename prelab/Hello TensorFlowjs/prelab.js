// Create tensors
const scalar = tf.scalar(5);
const vector = tf.tensor1d([1, 2, 3]);
const matrix = tf.tensor2d([[1, 2], [3, 4]]);

// Perform addition
const a = tf.tensor1d([1, 2, 3]);
const b = tf.tensor1d([4, 5, 6]);
const added = tf.add(a, b);

// Print results in console
console.log("Scalar:");
scalar.print();   // prints in console
console.log("Vector:");
vector.print();
console.log("Matrix:");
matrix.print();
console.log("Added Vector:");
added.print();
