# Introduction to Machine Learning

### What is Machine learning?
Ability for computers to learn on their own without explicitly being programmed. 

Types of Machine Learning:
* Supervised Machine Learning
* Unsupervised Machine Learning



### Supervised Machine Learning
In supervised learning the system leans from the examples. The examples which map input X to Y (label) - using this data where we have already identified the right answer (Y or the label) the algorithm would learn to predict Y from unseen X.
In supervised learning when the goal is to predict Y (which can have infinite possible values) is called Regression. 
Few examples:
* Predicting price of a house in a location, provided data which has prices of already sold house.

When the goal is to predict Categories which are finite in number the Supervised learning is called "Classification". Note that the categories does not need to be numeric. Few examples:

* Predicting weather cancer is beneign or not based on the data.
* Spam filter - classifying email as a span or not.
* Content moderation - claddifying content as fit or not fit.

### Unsupervised Machine Learning
As in supervised learning the data is already labeled with y (the output), in contrast in unsupervised learning data is not labeled. The algorightm tries to find pattern within the data. One typical class of unsupervised learning is "Clustering" for example: Google news which cluster the realted articles together. 
Another class of unsupervised leatning is "Anomaly detection". Examples could be "Froud detection".
Unsupervised learning is also used for "Dimentionality Reduction" - compressing data to fewer data set without compromising the kowledge. 

![Machine Learning](images/ml-intro.png)

## Example Linear Regression

### Common Terminology in ML
* ***Training Set***  - Data used to train the model. In case of supervised the data would have both x and y.
* ***Input Variable or Feature*** The input part of data (x) which is used to predict output. Generally denoted by x.
* ***Output or Target Variable*** Y - which we are trying to predict
* ***m*** Number of training examples.

Just to get our intution lets take an example of linear regression (a type of regression):
A regression model takes the training set as input and train the algorithm which outputs a function f which is of shape:
$$ f_{w,b}(x) = wx + b \tag{1}$$
Which means we would try to fit a linear line to predict the output values. We have taken example where we have price of house given the size of the house. We will use linear regression to train the model and get the function f.
Code sample for the same can be found at - [link](https://github.com/satishThakur/data-science/blob/main/machine-learning/ml-intro/linear-regression.ipynb)


## Cost Function

As we have seen in linear regression we try to fit a line which is close to the training set. Cost function measures how well our model works or how well the line fits. As different values of w and b would produce different lines our goal is to find the value of w and b which is the best fit.
What does it mean to be best fit? Thats where cost function comes in to play. The value of w and b which minimizes the cost function would best fit the model.
There are multiple cost functions used but one of the common one is called ***Square Error Cost Function*** and is defined as:

$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 
where 
  $$f_{w,b}(x^{(i)}) = wx^{(i)} + b \tag{2}$$
  
- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$.  
- $(f_{w,b}(x^{(i)}) -y^{(i)})^2$ is the squared difference between the target value and the prediction.   
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w,b)$.  

To summarize:

our model is $$f_{w,b}(x) = wx + b $$

our parameters are w and b

Cost function is $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 \tag{1}$$ 

Out goal - find value of w and x which minimizes cost function.

## Gradient Decent
Gradient Decent is an algorithm to find parameters which minimizes the Cost Function. In other words for linear regression - Gradient Decent can be used to train the model (compute w and b) which provides us the hypothesis or the function to predict output variable.
For "Cost Function" like ***Square Error*** with linear regression - Gradient Decent will always find global minima. If we use a different cost function - Gradient decent might find local minima (which means depending upon where we start we might get different result). 
The algorithm for Gradient Decent is as below:
* Start with some arbitary values of w and b.
* Keep changing values of w and b to minimumze cost function.

In case of linear regression:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w} \tag{3}  \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*}$$
where, parameters $w$, $b$ are updated simultaneously.  
The gradient is defined as:
$$
\begin{align}
\frac{\partial J(w,b)}{\partial w}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} \tag{4}\\
  \frac{\partial J(w,b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) \tag{5}\\
\end{align}
$$

Here *simultaniously* means that you calculate the partial derivatives for all the parameters before updating any of the parameters.
Lets understand the "Gradient Decent" parameter update little more.
* $$ \alpha $$ - This is called learning rate. Learning rate controls how bigger steps we take for convergence of w and b. If $$ \alpha $$ is too small the algorithm would take more time to converge and if too large it might oscillate and rather diverge. 
*  $$ \frac{\partial J(w,b)}{\partial w} $$ - The derivative or partial derivate represents the direction. To have a better intution for a fixed b the function J(w) would look like a u-shaped parabola as showin in picture below:

![Cost Function](images/cost_func_linear.png)
As this is intuative the derivative at any point represent the slope of the tangent. Hence at any point the derivate will move the new w towards the global minima. The logic is same for w and b - where instead of 2-d we get a 3-d graph. But the shape of the surface still remains convex and hence it converges to gobal minima. 

Now to put the theory in pratice we would take the same regression example (Housing Prices) but this time we will use gradient decent to train our model. The code for the same is [here](https://github.com/satishThakur/data-science/blob/main/machine-learning/ml-intro/gradient-decent.ipynb). This completes our introduction to ML. In next week we would deep dive into "Linear Regression" more.

