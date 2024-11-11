# Collaborative Filtering Benchmarking

This repository contains the solution for collaborative filtering tasks, implemented with Python’s Surprise library. The project includes optimizing user-based K-NN by tuning the number of neighbors (K) to minimize Mean Absolute Error (MAE) under different levels of sparsity, addressing sparsity issues with SVD (Funk variant), generating Top-N recommendations with evaluations of precision, recall, and F1 metrics across varying values of NN.

## Setup

1. Clone the repository:

```bash
git clone git@github.com:ikajdan/collaborative_filtering_benchmarking.git
cd collaborative_filtering_benchmarking
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install the required packages:

```bash
pip install -r requirements.txt
```

4. Download the data:

```bash
curl -O https://files.grouplens.org/datasets/movielens/ml-100k.zip
unzip ml-100k.zip
```

## Task 1

<details>
  <summary>Task Description</summary><br>

  Given the [dataset](https://grouplens.org/datasets/movielens/100k/) and the algorithm of K-NN (K-Nearest Neighbors), for user-based CF (Collaborative Filtering):

  1. Find out the value for K that minimizes the MAE (Mean Absolute Error) with 25% of missing ratings.
  2. Sparsity problem: find out the value for K that minimizes the MAE with 75% of missing ratings.
</details>

---

<!-- <br>
<div align="center">
  <img src="figures/task1.png" width="900" height="auto"/>
  <br><br>
  <em>MAE vs. K for 25% and 75% missing ratings.</em>
</div>
<br> -->

For both sparsity levels, it seems that MAE generally decreases as K increases, reaching a minimum before leveling off or slightly increasing. For this dataset and KNN-based user collaborative filtering:

- with 25% missing ratings: The lowest MAE occurs around K=22 with an MAE of approximately 0.7784,
- with 75% missing ratings: The lowest MAE occurs around K=24 with an MAE of approximately 0.8216.

These values indicate that optimal K values for minimizing MAE slightly differ with the degree of sparsity, with both tending to stabilize around K=25 to K=30.

The MAE is consistently lower for the 25% missing ratings case across all K values. This is expected, as more data generally leads to better predictive accuracy. In the second case algorithm struggles with sparsity since fewer neighbors have overlapping ratings with the target user, leading to less reliable recommendations. Higher data sparsity may benefit from larger neighborhoods to aggregate more data points and reduce error.

## Task 2

<details>
  <summary>Task Description</summary><br>

  Mitigation of sparsity problem: show how SVD (Singular Value Decomposition), the Funk variant, can provide a better MAE than user-based K-NN using the provided [data](data.csv).
</details>

---

The results for the two algorithms and the given dataset are as follows:

- MAE for Funk SVD: 0.9950
- MAE for User-based K-NN: 1.143

Funk SVD yields a lower MAE than user-based K-NN by capturing latent factors that reveal hidden relationships between users and items, enabling it to predict ratings for sparse data where users and items don’t overlap directly. By approximating missing values and filtering out noise, Funk SVD creates a more accurate, generalized model that’s computationally efficient at prediction time, while K-NN’s reliance on direct similarity and overlap limits its accuracy and scalability in sparse datasets.

## Task 3

<details>
  <summary>Task Description</summary><br>

  Top-N recommendations: calculate the precision, recall, and F1 with different values for N (10 to 100) using user-based K-NN (with the best Ks) and SVD. To do this, you must suppose that the relevant recommendations for a specific user are those rated with 4 or 5 stars in the data set. Perform the calculations for both 25% and 75% of missing ratings.

  Explain why you think that the results reported in the three tasks make sense.
</details>

---

```
   Algorithm  Missing %    N  Precision    Recall        F1
0        KNN       0.25   10   0.000000  0.000000  0.000000
1        KNN       0.25   20   0.000000  0.000000  0.000000
2        KNN       0.25   30   0.000000  0.000000  0.000000
3        KNN       0.25   40   0.000000  0.000000  0.000000
4        KNN       0.25   50   0.000000  0.000000  0.000000
5        KNN       0.25   60   0.000000  0.000000  0.000000
6        KNN       0.25   70   0.000000  0.000000  0.000000
7        KNN       0.25   80   0.000000  0.000000  0.000000
8        KNN       0.25   90   0.000000  0.000000  0.000000
9        KNN       0.25  100   0.000000  0.000000  0.000000
10       SVD       0.25   10   1.000000  0.350000  0.518519
11       SVD       0.25   20   1.000000  0.350000  0.518519
12       SVD       0.25   30   1.000000  0.350000  0.518519
13       SVD       0.25   40   1.000000  0.350000  0.518519
14       SVD       0.25   50   1.000000  0.350000  0.518519
15       SVD       0.25   60   1.000000  0.350000  0.518519
16       SVD       0.25   70   1.000000  0.350000  0.518519
17       SVD       0.25   80   1.000000  0.350000  0.518519
18       SVD       0.25   90   1.000000  0.350000  0.518519
19       SVD       0.25  100   1.000000  0.350000  0.518519
20       KNN       0.75   10   0.583333  0.124972  0.205845
21       KNN       0.75   20   0.583333  0.124972  0.205845
22       KNN       0.75   30   0.583333  0.124972  0.205845
23       KNN       0.75   40   0.583333  0.124972  0.205845
24       KNN       0.75   50   0.583333  0.124972  0.205845
25       KNN       0.75   60   0.583333  0.124972  0.205845
26       KNN       0.75   70   0.583333  0.124972  0.205845
27       KNN       0.75   80   0.583333  0.124972  0.205845
28       KNN       0.75   90   0.583333  0.124972  0.205845
29       KNN       0.75  100   0.583333  0.124972  0.205845
30       SVD       0.75   10   0.000000  0.000000  0.000000
31       SVD       0.75   20   0.000000  0.000000  0.000000
32       SVD       0.75   30   0.000000  0.000000  0.000000
33       SVD       0.75   40   0.000000  0.000000  0.000000
34       SVD       0.75   50   0.000000  0.000000  0.000000
35       SVD       0.75   60   0.000000  0.000000  0.000000
36       SVD       0.75   70   0.000000  0.000000  0.000000
37       SVD       0.75   80   0.000000  0.000000  0.000000
38       SVD       0.75   90   0.000000  0.000000  0.000000
39       SVD       0.75  100   0.000000  0.000000  0.000000
```

The results show that SVD outperforms KNN in recommending relevant items when only 25% of ratings are missing, achieving high precision and recall due to its strength in generalizing across moderately sparse data. However, with 75% missing data, KNN maintains moderate precision but low recall, as it struggles with limited user similarity, while SVD fails entirely to produce relevant recommendations due to insufficient data for building latent factors. These outcomes highlight that SVD performs best with moderate sparsity, whereas KNN can still retrieve some relevant items even under extreme sparsity, though with limited effectiveness.

For KNN at 25% missing data, the algorithm likely failed to find relevant items due to insufficient overlap or similarity among users, as it depends directly on user-to-user comparisons. For SVD at 75% missing data, the high sparsity means there wasn’t enough data to train reliable latent factors, leading it to fail in producing meaningful recommendations.
