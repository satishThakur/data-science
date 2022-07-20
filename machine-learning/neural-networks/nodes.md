# Neural Networks

Neural Networks are also called Deep Learning systems. Historically Neural network started with mimicing biological brain (neurons which are basic units). Though the motivation has been human btains but neural networks have differed from biological brains. 
Few areas where neural network has shown significant promise are:
* Speech recognision.
* Computer vision
* NLP

## Biological Motivation

Biological Neoron has inputs called dendrites (through which it gets the input signal or activation). Neuron has a cell body. The output of a neuron is transmitted via axon which would be input to other neurons. 

![Biological Neuron](images/bio-neuron.png)

The mathematical neuron is extremely simplified model of human neuron. Infact it has shifted a lot from the biological counterpart. The neuron can be represented as:

![neuron](images/enuron.png)


## Why Deep Learning Now?
* Traditional AI does not sclae with data. What it means is traditional AI does not become better as more and more data is available.
* Deep learning scales with data. As more data is available, medium and large neural networks keep performing better.
* Development in computation power - with powerful GPU/CPU the ability to process large deep leanrning networks.


## How Neural Network Works?

Lets take an example of demand prediction. Given input features like - price, material, color etc we need to predict if the shirt would have high demand. The typical neural network for the same might look like:

![Layers](images/neuron-layers.png)

The way to think about is:
* Input Layers are the input features to the model.
* The hidden layers are the one where model does feature engineering. Which means it combines one of more input features (with appropriate weight) and create high level features which are better predictors of the output.
* The output layer is the one which decides on prediction. 

Same intution can be applied to image processing where the goal is to find if image is for person xyz. Here:
* Input Layer would be invidual pixel value of the image.
* Multiple hidden layers which build on last layer.
* Output Layer which predicts.

