## Linear Regression

### Common Terminology in ML
* ***Training Set***  - Data used to train the model. In case of supervised the data would have both x and y.
* ***Input Variable or Feature*** The input part of data (x) which is used to predict output. Generally denoted by x.
* ***Output or Target Variable*** Y - which we are trying to predict
* ***m***  - Number of training examples.

Just to get our intution lets take an example of linear regression (a type of regression):
A regression model takes the training set as input and train the algorithm which outputs a function f which is of shape:
$$f_{w,b}(x) = w*x + b $$
Which means we would try to fit a linear line to predict the output values. We have taken example where we have price of house given the size of the house. We will use linear regression to train the model and get the function f.
Code sample for the same can be found at - [link](https://github.com/satishThakur/data-science/blob/main/machine-learning/uni-regression/linear-regression.ipynb)


## Cost Function

As we have seen in linear regression we try to fit a line which is close to the training set. Cost function measures how well our model works or how well the line fits. As different values of w and b would produce different lines our goal is to find the value of w and b which is the best fit.
What does it mean to be best fit? Thats where cost function comes in to play. The value of w and b which minimizes the cost function would best fit the model.
There are multiple cost functions used but one of the common one is called ***Square Error Cost Function*** and is defined as:

$$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$ 
where 
  $$f_{w,b}(x^{(i)}) = wx^{(i)} + b $$
  
- $f_{w,b}(x^{(i)})$ is our prediction for example $i$ using parameters $w,b$.  
- $(f_{w,b}(x^{(i)}) -y^{(i)})^2$ is the squared difference between the target value and the prediction.   
- These differences are summed over all the $m$ examples and divided by `2m` to produce the cost, $J(w,b)$.  

To summarize:

our model is $$f_{w,b}(x) = wx + b $$

our parameters are w and b

Cost function is $$J(w,b) = \frac{1}{2m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})^2 $$ 

Our goal - find value of w and x which minimizes cost function.

## Gradient Decent
Gradient Decent is an algorithm to find parameters which minimizes the Cost Function. In other words for linear regression - Gradient Decent can be used to train the model (compute w and b) which provides us the hypothesis or the function to predict output variable.
For "Cost Function" like ***Square Error*** with linear regression - Gradient Decent will always find global minima. If we use a different cost function - Gradient decent might find local minima (which means depending upon where we start we might get different result). 
The algorithm for Gradient Decent is as below:
* Start with some arbitary values of w and b.
* Keep changing values of w and b to minimimze cost function.

In case of linear regression:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline
\;  w &= w -  \alpha \frac{\partial J(w,b)}{\partial w}   \; \newline 
 b &= b -  \alpha \frac{\partial J(w,b)}{\partial b}  \newline \rbrace
\end{align*} $$


where, parameters w, b are updated simultaneously.  
The gradient is defined as:
$$\frac{\partial J(w,b)}{\partial w}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)})x^{(i)} $$

$$\frac{\partial J(w,b)}{\partial b}  = \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{w,b}(x^{(i)}) - y^{(i)}) $$
Here *simultaniously* means that you calculate the partial derivatives for all the parameters before updating any of the parameters.


Lets understand the "Gradient Decent" parameter update little more.
* $$\alpha $$ This is called learning rate. Learning rate controls how bigger steps we take for convergence of w and b. If $$\alpha $$ is too small the algorithm would take more time to converge and if too large it might oscillate and rather diverge. 
*  $$\frac{\partial J(w,b)}{\partial w} $$ The derivative or partial derivate represents the direction. To have a better intution for a fixed b the function J(w) would look like a u-shaped parabola as showin in picture below:

![Cost Function](images/cost_func_linear.png)


As this is intutive the derivative at any point represent the slope of the tangent. Hence at any point the derivative will move the new w towards the global minima. The logic is same for w and b - where instead of 2-d we get a 3-d graph. But the shape of the surface still remains convex and hence it converges to gobal minima. 

Now to put the theory in pratice we would take the same regression example (Housing Prices) but this time we will use gradient decent to train our model. The code for the same is [here](https://github.com/satishThakur/data-science/blob/main/machine-learning/uni-regression/gradient-decent.ipynb). This completes our introduction to ML. In next week we would deep dive into "Linear Regression" more.