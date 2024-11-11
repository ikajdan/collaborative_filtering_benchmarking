from collections import defaultdict

import pandas as pd
from surprise import SVD, Dataset, KNNBasic, Reader
from surprise.model_selection import train_test_split


# Load data from file
def load_data(filename):
    data = pd.read_csv(filename, sep=";", index_col=0)
    df = data.stack().reset_index()
    df.columns = ["item", "user", "rating"]
    return df


# Generate Top-N recommendations
def get_top_n(predictions, n=10, threshold=4.0):
    top_n = defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        if est >= threshold:
            top_n[uid].append((iid, est))
    for uid, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[uid] = user_ratings[:n]
    return top_n


# Precision, Recall, and F1 calculations
def precision_recall_at_n(top_n, relevant_ratings):
    precisions = []
    recalls = []

    for uid, user_ratings in top_n.items():
        n_rel = len(relevant_ratings[uid])
        n_rec_k = len(user_ratings)
        n_rel_and_rec_k = sum(
            (iid in relevant_ratings[uid]) for (iid, _) in user_ratings
        )

        precision = n_rel_and_rec_k / n_rec_k if n_rec_k else 0
        recall = n_rel_and_rec_k / n_rel if n_rel else 0

        precisions.append(precision)
        recalls.append(recall)

    if precisions and recalls:
        avg_precision = sum(precisions) / len(precisions)
        avg_recall = sum(recalls) / len(recalls)
        f1 = (
            2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
            if (avg_precision + avg_recall)
            else 0
        )
    else:
        avg_precision, avg_recall, f1 = 0, 0, 0

    return avg_precision, avg_recall, f1


# Main function
def evaluate_models(filename):
    df = load_data(filename)

    # Define a reader to handle the rating scale
    reader = Reader(rating_scale=(1, 5))
    data = Dataset.load_from_df(df[["user", "item", "rating"]], reader)

    results = []
    for missing_percentage in [0.25, 0.75]:
        trainset, testset = train_test_split(data, test_size=missing_percentage)

        # Models
        algorithms = {
            "KNN": KNNBasic(sim_options={"user_based": True}),
            "SVD": SVD(),
        }

        # Relevant items for each user (ground truth)
        relevant_ratings = defaultdict(set)
        for uid, iid, true_r in testset:
            if true_r >= 4.0:
                relevant_ratings[uid].add(iid)

        # Calculate precision, recall, and F1 for each model
        for algo_name, algo in algorithms.items():
            algo.fit(trainset)
            predictions = algo.test(testset)

            for n in range(10, 101, 10):
                top_n = get_top_n(predictions, n=n)
                precision, recall, f1 = precision_recall_at_n(top_n, relevant_ratings)

                results.append(
                    {
                        "Algorithm": algo_name,
                        "Missing %": missing_percentage,
                        "N": n,
                        "Precision": precision,
                        "Recall": recall,
                        "F1": f1,
                    }
                )

    # Display results
    results_df = pd.DataFrame(results)
    print(results_df)


# Run the evaluation
evaluate_models("data.csv")
