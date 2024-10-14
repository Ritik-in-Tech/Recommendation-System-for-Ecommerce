# Ecommerce-Product Recommendation System

Product Recommendation System is a machine learning project that analyzes users' browsing and buying records to give them personalized product suggestions. It looks at how people use the system and makes suggestions based on that information using joint filtering and content-based filtering algorithms. Users will have a better shopping experience generally, and e-commerce businesses will make more money.

## Dataset

We have used an amazon dataset on user ratings for electronic products, this dataset doesn't have any headers. To avoid biases, each product and user is assigned a unique identifier instead of using their name or any other potentially biased information.

You can find the dataset: [dataset](https://www.kaggle.com/datasets/vibivij/amazon-electronics-rating-datasetrecommendation/download?datasetVersionNumber=1)

## Approach

#### 1) Rank Based Product Recommendation

##### Objective -

1. Recommend products with highest number of ratings.
2. Target new customers with most popular products.

##### Outputs -

1. Recommend top 5 products with 50/100 minimum ratings/interactions.

##### Approach -

1. Calculate average rating for each product.
2. Calculate total number of ratings for each product.
3. Create a DataFrame using these values and sort it by average.
4. Write a function to get 'n' top products with specified minimum number of interactions.

#### 2) Similarity based collaborative filtering

##### Objective -

1. Provide personalized and relevant recommendations to users.

##### Outputs -

1. Recommend top 5 products based on interactions of similar users.

##### Approach -

1. Here, user_id is of object, for our convenience we convert it to value of 0 to 1539(integer type).
2. We write a function to find similar users -
   i) Find the similarity score of the desired user with each user in the interaction matrix using cosine_similarity and append to an empty list and sort it.
   ii) extract the similar user and similarity scores from the sorted list.
   iii) remove original user and its similarity score and return the rest.
3. We write a function to recommend users -
   i) Call the previous similar users function to get the similar users for the desired user_id.
   ii) Find prod_ids with which the original user has interacted -> observed_interactions
   iii) For each similar user Find 'n' products with which the similar user has interacted with but not the actual user.
   iv) return the specified number of products.

## License

[MIT](https://choosealicense.com/licenses/mit/)
