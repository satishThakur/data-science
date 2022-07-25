"""
Python Module which contains functions needs to carry on univariate regression.
In univariate regression we would have single feature and single output variable.
"""

import numpy as np

def hypothesis(x, w, b):
    """
        Input:
            x: Scalar Value - Input or feature value.
            w: Scalar Value - Model Parameter
            b : Scalar Value - Model parameter - Bias
        Output:
            predicted value.
                
    """
    return w * x + b

#Helper function which takes array of x and output model output
#This is bulk version of Hypothesis function, it takes N features and returns 
#predicted value for all of them. 
def model_output(xs, w, b):
    """
        Input:
            xs: NDArray(m,) - Input or feature values array.
            w: Scalar Value - Model Parameter
            b : Scalar Value - Model parameter - Bias
        Output:
            NDArray(m,) - predicted values in .
    """
    m = xs.size
    output = np.zeros((m,))
    for i in range(m):
        output[i] = hypothesis(xs[i], w, b)
    return output


# This function can take advantage of parallel computation.
def model_output_fast(xs, w, b):
    """
        Input:
            xs: NDArray(m,) - Input or feature values array.
            w: Scalar Value - Model Parameter
            b : Scalar Value - Model parameter - Bias
        Output:
            NDArray(m,) - predicted values in .
    """
    return xs * w + b



#compute_cost calculate cost from current training set.
#x and y both are vetors here.
def compute_cost(x, y, w, b):
    """
        Input:
            x: NDArray(m,) - Input or feature values array.
            y: NDArray(m,) - output labled values array.
            w: Scalar Value - Model Parameter
            b: Scalar Value - Model parameter - Bias
        Output:
            Scalar - For given model parameters cost for the trainint set.
    """
    cost = 0
    m = x.size
    for i in range(m):
        y_hat = hypothesis(x[i], w, b)
        cost = cost + (y_hat - y[i])**2
    return cost/2*m   


#compute_gradient computes the gradient for both w and b parameter, 
#given the training set. 
def compute_gradient(x, y, w, b):
    """
        Input:
            x: NDArray(m,) - Input or feature values array.
            y: NDArray(m,) - output labled values array.
            w: Scalar Value - Model Parameter
            b: Scalar Value - Model parameter - Bias
        Output:
            w_grad: Scalar - Gradient for W parameter.
            b_grad: Scalar - Gradient for b parameter.
    """

    w_grad = 0.0
    b_grad = 0.0
    m = x.size
    for i in range(m):
        f_w_b = hypothesis(x[i], w, b)
        w_grad = w_grad + (f_w_b - y[i]) * x[i]
        b_grad = b_grad + (f_w_b - y[i])
    return (w_grad/m, b_grad/m)    


#gradient_descent rungs gradient descent algorithm to train the model.
#The function takes initial values of w and b and computes the values of w and b 
# which minimizes the cost.
def gradient_descent(x_train, y_train,w, b,alpha,num_iter, debug=False):
    """
        Input:
            x_train: NDArray(m,n) - Input or feature values array from the training set.
            y_train: NDArray(m,) - output labled values array.
            w: Scalar Value - Model Parameter inital value
            b: Scalar Value - Model parameter - Bias - intial value.
            alpha: Scalar - Learning rate
            num_iter: Scalar - Number of iterations to be performed 
            debug: Boolean - if true prints debug info.
        Output:
            w: Scalar - Model parameter final value.
            b: Scalar - Model parameter b's final value.
            cost_history: Array - All the cost history values in a array.
    """

    cost_history = []
    for i in range(num_iter):
        (w_delta, b_delta) = compute_gradient(x_train,y_train,w,b)
        w = w - alpha * w_delta
        b = b - alpha * b_delta
        cost_history.append(compute_cost(x_train, y_train, w, b))
        if debug and i % 100 == 0:
            print(f'Iteration - {i}, cost - {cost_history[-1]}')                  
    
    return (w,b, cost_history)
