# recommender_systems

## Summary of Approaches

| **MODEL**&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;                                                         | **DETAILS**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
|-------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| Content Based                                                     | **Cosine Similarity** between one-hot encoding vector for 18 genres<br/><br/>    multiplied by<br/><br/>   **Exponential Decay Similarity** for years **math.exp(-diff / 10.0)**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| User-Based Collaborative Filtering (for Top-N)           | Recommend the highest rated movies by the Top-10 similar users                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| User-Based KNN (for Rating) | Predict every missing rating using the Top-10 similar users who have watched that movie<br/><br/> **using Cosine Similarity**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| Item-Based KNN (for Rating) | Predict every missing rating using the Top-10 similar movies watched by the user<br/><br/> **using Cosine Similarity**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  |
| SVD Based Matrix Factorization (for Rating)              | # **Principal Component Analysis**<br/>  * PCA on R   = U (Users x Latent Features)<br/>  * PCA on R<sup>T</sup> = M (Movies x Latent Features)<br/><br/>    # **R = U ∑ M<sup>T</sup>**<br/>   * U = User Matrix<br/>  * ∑ = Diagonal Matrix (which tells the strength of latent factors)<br/>  * M<sup>T</sup> = Movie Matrix<br/><br/>    # **Singular Value Decomposition**<br/>   * It's a way of computing  **U ∑ M<sup>T</sup>** in one shot<br/>  * Null Values - fill missing values with certain defaults like mean<br/>  * SGD or ALS to calculate SVD inspired algorithm<br/>      * Since, SVD doesn't works with missing values<br/>    * Rating = dot product of **(User x Latent Factors)** & **(Movie x Latent Factors)**<br/>  |
| SVD++ Based Matrix Factorization (for Rating)            | The SVD++ algorithm, an extension of SVD taking into account implicit ratings                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| Deep Learning (for Rating)                              | Generate User & Movie embeddinggs to model Ratings using ANN                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
