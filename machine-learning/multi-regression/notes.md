# Multiple Linear Regression
In this section we would continue our journey of ***Linear Regression***. In previous section we have gone through Linear Regression with single Feature. We would now explore if we have multiple features and that is what is *Multiple Linear Regression*.

***Note*** Do not confuse with ***Multivariate Regression*** which is different than ***Multiple Linear Regression***. ***Multivariate Regression*** means when we are trying to predict multiple outcome variable. In ***Multiple Linear Regression*** we are still trying to predict single outcome but with multiple input features.

### Terminologies
Lets take an example of housing price prediction. The training dataset contains three examples with four features (size, bedrooms, floors and, age) shown in the table below.
| Size (sqft) | Number of Bedrooms  | Number of floors | Age of  Home | Price (1000s dollars)  |   
| ----------------| ------------------- |----------------- |--------------|-------------- |  
| 2104            | 5                   | 1                | 45           | 460           |  
| 1416            | 3                   | 2                | 40           | 232           |  
| 852             | 2                   | 1                | 35           | 178           |  

We will build a linear regression model using these values so you can then predict the price for other houses. This is example of ***Multiple Linear Regression*** model.

* $\mathbf{x}^{(i)}$, $y^{(i)}$ $i_{th}$ Training Example  `X[i]`, `y[i]` in Python.
* $\mathbf{w}$   parameter: weight
* $b$           parameter: bias
* $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ The result of the model evaluation at $\mathbf{x}^{(i)}$

The approach and algorithm remains pretty much similar to univariate regression. In multi regression we would have N features and hence we would rather have a W vector:

Gradient descent for multiple variables:

$$\begin{align*} \text{repeat}&\text{ until convergence:} \; \lbrace \newline\;
& w_j = w_j -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial w_j} \tag{5}  \; & \text{for j = 0..n-1}\newline
&b\ \ = b -  \alpha \frac{\partial J(\mathbf{w},b)}{\partial b}  \newline \rbrace
\end{align*}$$

where, n is the number of features, parameters $w_j$,  $b$, are updated simultaneously and where  

$$
\begin{align}
\frac{\partial J(\mathbf{w},b)}{\partial w_j}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)})x_{j}^{(i)} \tag{6}  \\
\frac{\partial J(\mathbf{w},b)}{\partial b}  &= \frac{1}{m} \sum\limits_{i = 0}^{m-1} (f_{\mathbf{w},b}(\mathbf{x}^{(i)}) - y^{(i)}) \tag{7}
\end{align}
$$
* m is the number of training examples in the data set

    
*  $f_{\mathbf{w},b}(\mathbf{x}^{(i)})$ is the model's prediction, while $y^{(i)}$ is the target value

The code for the same is [here]()

### Vectorization
The model's prediction with multiple variables is given by the linear model:

$$ f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b $$
or in vector notation:
$$ f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b $$ 
where $\cdot$ is a vector `dot product`

When using Python libraries like numpy the vector algebra (for example dot product) would be much faster than doing the same in for loop. These libraries do take advantage of modern GPU/CPU to parallelize the computation. This is what basically is Vectorization. 

### Feature Scaling
As we saw in the previous example multi regression our w and b were still not converging. One of the issue could be where the range of different features varies significantly. For example the size of house vary from 852 to 2104 but number of bedrooms only vary from 2 to 5.
To bring the features to similar ranges there are multiple ways:
* ***Mean Normalization*** - Here we subtract the mean from feature and devide by (max - min). 
* ***Z-Score Normalization***  - 

To implement z-score normalization, adjust your input values as shown in this formula:
$$x^{(i)}_j = \dfrac{x^{(i)}_j - \mu_j}{\sigma_j}$$ 
where $j$ selects a feature or a column in the $\mathbf{X}$ matrix. $µ_j$ is the mean of all the values for feature (j) and $\sigma_j$ is the standard deviation of feature (j).
$$
\begin{align}
\mu_j &= \frac{1}{m} \sum_{i=0}^{m-1} x^{(i)}_j\\
\sigma^2_j &= \frac{1}{m} \sum_{i=0}^{m-1} (x^{(i)}_j - \mu_j)^2
\end{align}
$$

As intuive from the formula after Z-Score normalization the distribution mean is centered around 0. As we are deviding the values by standard deviation, most of the values should lie between -3 and 3. 

### Choosing the right value of Learning Rate
As we already know:
* A small learning rate would make the gradient descent run slower.
* A large learning rate would lead to non-convergence.

Ways to select the right learning rate:

* Plot the `cost function` for few iteratins with initial `alpha`.
* If the `cost` is increasing or creating saw-tooth then the value of `alpha` is large.
* Iterate the same process with smaller value of `alpha`.

### How do we know Gradient Descent is Converging?
1. Plot the cost with iterations. If the cost has stabalized towards end of iteration - this means we have converged.
2. Automated way. Choose a value e (ephsilon - lets say 0.0001). If the successive values of cost have delta less than e we can assume converged.


## Polynomial Regression

### Feature Engineering
Feature engineering  - Using Intution to design new features by transforming or combining original Features. Few examples:
* If lengh and width are being used as input features, a new features called area could be derieved.
* If number of SKU sold and SKU price are featues - total sale which is number of SKU x Price could be new feature.

For cases where Linenar regression does not fit our needs we could also try out Polynomial regression. This could easily achived via Feature Engineering:

* If for the feature $$x_i$$ we see that $$x_i^2$$ has better correlation with the output, we can create a new feature whose data is $$x_i^2$$
* We can use the same logic to see if we need new feature - $$x_i^3$$
* As we are multiplying the feature values - ***Feature Scaling*** becomes more important in case of Polynomial Regression.
* Rest of the procedure remains same as multi regression.

Code with example of ***Polynomila Regression*** is linked [here](). The code conatains example of both feature engineering and scaling, then uses gradient descent to train the polynomial regression Model.

## Using Scikit-learn

