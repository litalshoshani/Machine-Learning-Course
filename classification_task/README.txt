README

In this exercise we will solve a clothing picture classification task using neural
networks. We will use the Fashion-MNIST, a well known dataset.

Content:
Each image is 28 pixels in height and 28 pixels in width, for a total of 784
pixels in total. Each pixel has a single pixel-value associated with it, indicating
 the lightness or darkness of that pixel, with higher numbers meaning
darker. This pixel-value is an integer between 0 and 255.

The possible labels are:
0. T-shirt/top
1. Trouser
2. Pullover
3. Dress
4. Coat
5. Sandal
6. Shirt
7. Sneaker
8. Bag
9. Ankle boot

Instructions:
We will train a multi-class NN with one hidden layer. The loss function is the negative
log likelihood.
We will split the train into train and validation with 80:20 ratio, and use the validation set
for hyper-parameters tuning.

Using our trained classifier, we will create a test.pred file. This file will store
our model's predictions for 'test x'.
The prediction file contains 5000 lines. Each line is our prediction for the corresponding
example in the test file.

Our model and parameters:
First, we initialize the parameters - w1, w2, b1, b2 - with random values in the range of (-0.8, 0.8).
the reason for that is that we want to work with the activation function sigmoid, and for that it is better to work with smaller values.

One of the parameters we chose was H - hidden layer. the hidden layer should be between 10 (which is the number of classes) and 784 (which is the number of pixels).

At the beggining, we started with a H=16, which is relatively small, and when we saw that the success rate is pretty low (around 20%) we changed the H parameter to 50 and then 100. Our conclusion: H=100 was the best.

The second parameter we chose was the learning rate - lr. we used the learning rate when we needed to update the weights and the bias. The weight's upadte accurd after every step - meaning, after we calculated the gradients for every x and its tag.
We chose the learing rate to be lr=0.01, which gave us the best results.

The last parameter was the number of epoches - which is the number of times we ran over the train_x and its tags. We chose epoches=100. 

The results: success rate were around 80%-95%.

Notes:
(a) In this exercise we were not allowed to use any machine learning packages or tools
	(e.g. scikit-learn, PyBrain, PyML, etc.).
(b) I used numpy package
(c) In this exercise we were restricted to Python 2.7 only.

In this exercise are two files attached:
1): ex3.py - the code file.
2): test.pred - the prediction for the test file.
