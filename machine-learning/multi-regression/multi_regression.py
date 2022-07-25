"""
Python modules which contains functions to implement supervised regression for multiple features.
The code would have functions to carry out both Linear and Polynomial regression with N input featues and one output prediction. 
"""

#Here x is the feature vector, w is the parameter vector and b is bias.
def predict(x,w,b):
	"""
	Input:
		x: ndarray(n,) - Vector of n input features.
		w: ndarray(n,) - Vector of model parameters.
		b: Scalar - Model Parameter - Bias b.
	Output:
		Scalar value which is y hat - Prediction from the model. 	

	"""
    return np.dot(x,w) + b


#The cost signifies how accurate the model is - w.r.t. training data.
#There are multiple cost functions here we are using "Mean Squared Error" as cost function. 
def compute_cost(Xs, y, w, b):
	"""
	Input:
		Xs: ndarray(m,n) - m training data containing n features each.
		y : ndarray(m,)  - m outpot attribute of Label for the training set.
		w : ndarray(n,)   - Vector of model parameters.
		b : Scalar        - Model Parameter - Bias b.
	Output:
		Scalar value which is y hat - Prediction from the model. 	

	"""
    m = Xs.shape[0]
    cost = 0.0
    for i in range(m):
        cost = cost + ( predict(Xs[i], w, b) - y[i])**2
    return cost / (2 * m)


 
# Gradient finds the delta in w and b which would reduce the overall cost and hence
# move us one step towards local minima. The way this is done is by finding the partial derivates
# for each parameter w w.r.t. the cost function.      

 def gradient(X_train, y_train, w, b):
 	"""
	Input:
		X_train: ndarray(m,n) - m training data containing n features each.
		y_train: ndarray(m,)  - m outpot attribute of Label for the training set.
		w : ndarray(n,)       - Vector of model parameters.
		b : Scalar            - Model Parameter - Bias b.
	Output:
		w_delta: ndarray(n,) - Vector of delta for w patameters.
		w_delta: Scalar      - delta for b parameter
		
	"""

    m,n = X_train.shape # n is number of features.
    
    w_delta = np.zeros((n,))
    b_delta = 0.0
    
    for i in range(m):
        err = predict(X_train[i], w, b) - y_train[i]
        b_delta = b_delta + err 
        for j in range(n):
            w_delta[j] = w_delta[j] + err * x_train[i][j]

            
    w_delta = w_delta/m
    b_delta = b_delta/m
    return w_delta, b_delta
    



 def gradient_descent(X_train, y_train, w_in, b_in, a, num_iter):
 	"""
	Input:
		X_train: ndarray(m,n) - m training data containing n features each.
		y_train: ndarray(m,)  - m outpot attribute of Label for the training set.
		w_in   : ndarray(n,)    - Vector of model parameters - initial value.
		b_in   : Scalar            - Model Parameter - Bias b - initial value.
	Output:
		w: ndarray(n,) - Vector of  final w patameters.
		b: Scalar      -  b parameter final value
		j_history: Array - History of the cost - has num_iter entries.
		
	"""
    w = copy.deepcopy(w_in)  #avoid modifying global w within function
    b = b_in
    j_history = []
    
    for i in range(num_iter):
        w_delta, b_delta = gradient(X_train, y_train, w, b)
        w = w - a * w_delta
        b = b - a * b_delta
        cost = compute_cost(X_train, y_train, w, b)
        j_history.append(cost)
        
        if i % 10 == 0:
            print(f"Iteration {i:4d}: Cost {j_history[-1]:8.2f}   ")
    
    return w, b, j_history             