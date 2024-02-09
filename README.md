# Problem Statement
Given the data of multiple user ratings for multiple vendors, develop a recommendation system to suggest top 5 vendors for an input user id.

# Approach
- Clean the data to address missing values, outliers, and unformatted data
- Join Vendor dataframe and Orders dataframe to transform data to obtain maximum information
- Transform dataframe to represent user-vendor information in a matrix with users as rows, vendors as columns and ratings as the values
- Use item-item collaborative filtering to obtain the top 5 suggested vendors
- Use embedding layers to reduce sparsity of matrix and compress data
- Use dense layers to learn and infer user patterns
- Predict the maximum ratings the user would give to vendors
- Extract the top 5 predicted-ratings vendors

# Tools and Libraries
- Python 3.11
- PyData (NumPy, Pandas): Data Handling
- Matplotlib, Seaborn: Data Visualization
- Tensorflow: Embeddings, Model Building, Predictions
