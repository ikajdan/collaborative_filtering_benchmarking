import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader, accuracy
from surprise.model_selection import train_test_split


def main():
    data = pd.read_csv("data.csv", sep=";", header=None)

    # Get the movie titles from the first row and ratings from the rest
    movies = data.iloc[0, 1:].values
    ratings_data = data.iloc[1:, 1:].fillna(0)

    long_format = pd.DataFrame()

    for user_id in range(len(ratings_data)):
        for movie_id in range(len(movies)):
            long_format = long_format._append(
                {
                    "user_id": user_id,
                    "movie_id": movies[movie_id],
                    "rating": ratings_data.iloc[user_id, movie_id],
                },
                ignore_index=True,
            )

    reader = Reader(rating_scale=(1, 5))

    data = Dataset.load_from_df(long_format[["user_id", "movie_id", "rating"]], reader)

    trainset, testset = train_test_split(data, test_size=0.2)

    svd = SVD()
    svd.fit(trainset)
    predictions_svd = svd.test(testset)

    mae_svd = accuracy.mae(predictions_svd, verbose=True)

    knn = KNNBasic(sim_options={"name": "cosine", "user_based": True})
    knn.fit(trainset)
    predictions_knn = knn.test(testset)

    mae_knn = accuracy.mae(predictions_knn, verbose=True)

    print(f"MAE for Funk SVD: {mae_svd:.4f}")
    print(f"MAE for User-based K-NN: {mae_knn:.4f}")


if __name__ == "__main__":
    main()
