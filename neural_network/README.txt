README
written by: Lital Shoshani

In this exercise we will implement, train and evaluate our neural network using PyTorch
package.

Implementation:
In this exercise we will implement fully connected neural networks via PyTorch.
We will need to implement several settings and report the effect of each setting in terms of
loss and accuracy.

We will explore the following in our model:

a). Neural Network with two hidden layers, the first layer will have a size of
	100 and the second layer will have a size of 50, both should be followed
	by ReLU activation function

b). Dropout – we will add dropout layers to our model. The dropout will be place on 
	the output of the hidden layers.

c). Batch Normalization - we will add Batch Normalization layers to our model. 
	The Batch Normalization will be placed before the activation functions.

d). We use log_softmax as the output of the network and nll_loss function.


Training:
We will train our models using FashionMNIST dataset.
Our models will be trained for 10 epochs each.
We will split the training set to train and validation (80:20).

Evaluation:
In models.png we can see that the best results are in model 3.

Learning rate parameter: We started with learning rate = 0.1, and changed it to 0.01, and at last to 0.001 (which gave us the best results).

The next parameter is the batch for the train: we tried to change the batch values from 1 to 64.
values lower than 64 gave us the success rate in the range of 9% to 13% which is very low. Changing
the batch to 64 gave us success rate in the range of 80% and above.
(note: in the validation and test, the batch is 1).

The next parameter is Optimizer: we chose the optimizer adam, which ave us the best results, and 
the learning process was continuous. Continuous learning process was very important: by using differents
optimizers the train and validation jumped from 60% to 80% and down to 70% and then back up, meaning that
the learning process was not continuous.


Notes:
(a) In this exercise we were allowed to use numpy PyTorch framework.
(b) In this exercise we were restricted to Python 2.7 only.

In this exercise are two files attached:
1): ex_8_code.py - the code file.
2): test.pred - the prediction for the test file.