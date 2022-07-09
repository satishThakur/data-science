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

