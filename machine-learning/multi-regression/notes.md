# Multiple Linear Regression
In this section we would continue our journey of ***Linear Regression***. In previous section we have gone through Linear Regression with single Feature. We would not explore if we have multiple features and that is what is *Multiple Linear Regression*.
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

### Vectorization
The model's prediction with multiple variables is given by the linear model:

$$ f_{\mathbf{w},b}(\mathbf{x}) =  w_0x_0 + w_1x_1 +... + w_{n-1}x_{n-1} + b \tag{1}$$
or in vector notation:
$$ f_{\mathbf{w},b}(\mathbf{x}) = \mathbf{w} \cdot \mathbf{x} + b  \tag{2} $$ 
where $\cdot$ is a vector `dot product`

