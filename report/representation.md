In terms of representation, we are going to opt for the bag of words approach, 
as our dataset is not considered very large. We are also not going to consider 
n-grams with n > 2 for similar reasons. We are going to use `sklearn`'s implementation 
of `CountVectorizer` to vectorize our titles. 
