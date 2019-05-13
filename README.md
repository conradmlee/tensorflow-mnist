# Artificial Neural Networks Using TensorFlow
A Multilayer Neural Network implementation using the TensorFlow library to recognize handwritten digits from the MNIST database.

### Introduction

The MNIST is a well-known dataset for testing machine learning algorithms in image processing, and consists of 60k training samples and 10k testing samples. Further details about the dataset can be found in: http://yann.lecun.com/exdb/mnist/

### Inputs
The MNIST test set consists of 10,000 sample images. Each image consists of 784 pixels. The inputs are the activation values for the 784 neurons in the input layer, which represent the grayscale values of each individual pixel in a 28 pixel x 28 pixel image. The input values are given by the tensor x, which consists of a 784 x 1 vector.

### Outputs

The MNIST data set is a database of handwritten digits. Thus, the possible output values are the ten numerical digits, from 0 to 9.
The outputs are the values for the ten neurons in the output layer, represented by a 10 x 1 vector. The dimensions of the output layer are 10 x 1, representing the ten numerical digits (0 to 9). 

### Mapping Function

The mapping function used is the ReLU (Rectified Linear Unit) Activation Function. The ReLU applies the function: R(z) = max(0, z). ReLU is linear for all positive values, and zero for all negative inputs. The ReLU function is relatively simpler than the sigmoidal function, thus it is usually easier and faster to train with neural networks.

### Cost Function

The cost function used is Softmax Cross Entropy with Logits. This is similar to regular Cross Entropy function, except that it uses the Softmax function to normalize the output.

### Learning Algorithm

The learning algorithm here is the Adam algorithm (“adaptive moment estimation”). The Adam algorithm makes use of both the average of the first moment of the gradients (the mean) as well as the average of the second moment of the gradients (the variance). Specifically, it calculates an exponential moving average of the gradient and the squared gradient, with the parameters beta1 and beta2 controlling the decay rates of the moving averages.

Source: https://machinelearningmastery.com/adam-optimization-algorithm-for-deep-learning/

### Conclusions

We modified the number of hidden units in both hidden layers, using different values such as 64, 128, or 256 hidden units. Then we compared the results (based on time and accuracy). We also plot a confusion matrix for each run of the neural network.

We notice that increasing the number of units in the hidden layers increases the computation time but does not increases the accuracy. This is probably because the neural network is overfitting. Furthermore, in some cases, the neural network does not recognize certain digits at all (represented by a single column consisting of all zeroes in the confusion matrix). This is another likely indicator of overfitting.

We may further increase accuracy by testing some other parameters. For example: changing the number of hidden layers (using just one hidden layer, or using more than two hidden layers), lowering the number of units in the hidden layers (or using different numbers of units for each hidden layer), varying the number of training epochs (perhaps we are not allowing the model to train for long enough), varying the learning rate, or using ensemble methods (aggregating results from different neural network models).
