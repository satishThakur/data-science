# Recommender Systems
Recommender systems are at the heart of most e-commerce businesses for example Netflix recommending you what to watch or Amazon recommending you products to buy. Recommender systems drive economic value for the business and hence are in wideppread use. 
Few examples for the recommender systems:
* Recommending  restaurant by a food delivery application.
* Recommending article/news to read.
* Recommending products on e-commerce website.

## Example
Lets take an example which would help us to build intuition for "Collaborative Filter" based recommender systems. We would like to predict movies to the users. We have data about the movies and the ratings users have provided them.
We can think of this problem as predicting what ratings users would give to a movie which he/she has not yet watched. If we are able to do that then we can recommend movies to the user which he/she would have rated high.

### Using Per Item Features
Lets assume with movies we also have access to some of its featues, like genre etc. For example we have 2 features called "Action" and "Rmance" and values for every movie. Now we can build a liner regression model to fit the already rated movies for these features. This would let us predict the ratings for moving not rated by user.
This is very similar to regression model except we would have different model being trained for every user.
*What would be the Cost Function?*
Again we can use *root mean square* as the cost function with only difference that we would be summing it for all users. Now we run gradient descent and find the values of W and b for every user we can use it for prediction.

But what if we do not have these features available? 

### Collaborative Filter
