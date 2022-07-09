## Classification
In linear regression we have seen examples where what we are trying to predict has infitite values. In classification we would see examples where the output variable has few fixed categories. Lets look at few of the examples:

* Classifing an email as spam or no spam.
* Fraud detection. Marking every transaction as fraud or no fraud.
* Tumor - Classifying tumor as benign or non benign. 

When we are trying to find true/false - also called ***Binary Classifier***. 

From what we have learnt we can try to use linear regression for classification. One way would be:

When `f(x) < 0.5 - Prediction is 0 -> y^ = 0`

When `f(x) >= 0.5 - Prediction is 1 -> y^ = 1`

This might work for few of the examples but in general linear regression is not a very good model for classification. The problem is with every additional data (might be to far right) Linear regression would lead to shift over (changing of decision boundary) which would lead to bad classification.

### Logictic Regression
As we saw Linear regression is not a very good model to be used for classification. Lets intruduce `Sigmoid Function` or `Logistic Function` which would help us to model classification better. 

Sigmoid function is is a s-shape graph which maps every input value to a output value from 0 to 1. The function is defined as:

$g(z) = \frac{1}{1+e^{-z}}\ $

As we can see for a very large z the g(z) would approach near 1 (but never exactly 1) and for very small z values it would approach 0. 


![Sigmoid Function](images/sigmoid.png)

So the process for Logistic Regression is as below:

* We would find the w and b which minimize the cost (this would be explaned later - for now lets assume we have found w and b)
* We would feed it to logistic function for prediction:

$$ f_{\mathbf{w},b}(\mathbf{x}^{(i)}) = g(\mathbf{w} \cdot \mathbf{x}^{(i)} + b ) $$ 

  where

  $g(z) = \frac{1}{1+e^{-z}}\$


#### Interpretation
The way we can interpret the output prediction is - Probability. Lets assume we are using Logistic regression to find if given a size of tumour is it malignant or not. If we get output as 0.7 we can interpret as  - Model has predicted 70% chance of tumor being malignant. 

Here is link of code to see how Sigmoud function looks - [Code Link](https://github.com/satishThakur/data-science/blob/main/machine-learning/classification/sigmoid_function.ipynb) .

### Decision Boundary

