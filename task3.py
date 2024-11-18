from collections import defaultdict
from tkinter import N

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from surprise import SVD, Dataset, KNNWithMeans, Reader
from surprise.model_selection import train_test_split


def precision_recall_at_n(predictions, n=10, threshold=3.5):
    """
    Calculate precision and recall at N for each user.
    """
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions, recalls = {}, {}
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rel = sum((true_r >= threshold) for _, true_r in user_ratings)
        n_rel_and_rec = sum((true_r >= threshold) for _, true_r in user_ratings[:n])

        precisions[uid] = n_rel_and_rec / n
        recalls[uid] = n_rel_and_rec / n_rel if n_rel != 0 else 0

    return precisions, recalls


def load_csv():
    """
    Load data from CSV and prepare it for the Surprise library.
    """
    csv_file = pd.read_csv("data.csv", delimiter=";")
    temp = np.delete(csv_file.to_numpy(), np.s_[0], axis=1)
    ratings = temp.T.flatten()

    users, movies = [], []
    for i in range(50):
        for j in range(20):
            movies.append(j)
            users.append(i)

    movies = np.array(movies)
    users = np.array(users)

    ratings_dict = {"userID": users, "itemID": movies, "rating": ratings}
    df = pd.DataFrame(ratings_dict)

    reader = Reader(rating_scale=(1, 5))
    return Dataset.load_from_df(df[["userID", "itemID", "rating"]], reader)


def evaluate_model(model, trainset, testset, N_list=range(10, 101, 5)):
    """
    Evaluate a model using precision, recall, and F1-score for different N values.
    """
    model.fit(trainset)
    predictions = model.test(testset)

    precision_list, recall_list, f1_list = [], [], []

    for n in N_list:
        precisions, recalls = precision_recall_at_n(predictions, n=n, threshold=4)
        avg_precision = sum(prec for prec in precisions.values()) / len(precisions)
        avg_recall = sum(rec for rec in recalls.values()) / len(recalls)
        f1 = 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

        precision_list.append(avg_precision)
        recall_list.append(avg_recall)
        f1_list.append(f1)

    return precision_list, recall_list, f1_list


def plot_results(results_25, results_75, N_list):
    """
    Plot precision, recall, and F1 scores for both 25% and 75% missing ratings.
    """
    precision_list_25, recall_list_25, f1_list_25 = results_25
    precision_list_75, recall_list_75, f1_list_75 = results_75

    sns.set_theme(style="whitegrid")
    palette = sns.color_palette("muted")

    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    axes[0].plot(
        N_list, precision_list_25, marker="o", color=palette[0], label="Precision (25%)"
    )
    axes[0].plot(
        N_list, precision_list_75, marker="s", color=palette[1], label="Precision (75%)"
    )
    axes[0].set_title("Precision at N", fontsize=14)
    axes[0].set_xlabel("N", fontsize=12)
    axes[0].set_ylabel("Precision", fontsize=12)
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(
        N_list, recall_list_25, marker="o", color=palette[2], label="Recall (25%)"
    )
    axes[1].plot(
        N_list, recall_list_75, marker="s", color=palette[3], label="Recall (75%)"
    )
    axes[1].set_title("Recall at N", fontsize=14)
    axes[1].set_xlabel("N", fontsize=12)
    axes[1].set_ylabel("Recall", fontsize=12)
    axes[1].legend()
    axes[1].grid(True)

    axes[2].plot(
        N_list, f1_list_25, marker="o", color=palette[4], label="F1 Score (25%)"
    )
    axes[2].plot(
        N_list, f1_list_75, marker="s", color=palette[5], label="F1 Score (75%)"
    )
    axes[2].set_title("F1 Score at N", fontsize=14)
    axes[2].set_xlabel("N", fontsize=12)
    axes[2].set_ylabel("F1 Score", fontsize=12)
    axes[2].legend()
    axes[2].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()


def main():
    data = load_csv()

    N_list = range(1, 101, 5)

    # 25% missing ratings
    train_set, test_set = train_test_split(data, test_size=0.25, random_state=22)

    # Best K
    best_k_25 = None
    best_f1_25 = 0
    for k in range(5, 51, 5):
        knn_model = KNNWithMeans(
            k=k, sim_options={"name": "pearson", "user_based": True}, verbose=False
        )
        knn_results_25 = evaluate_model(knn_model, train_set, test_set, N_list=N_list)
        avg_f1_25 = np.mean(knn_results_25[2])

        if avg_f1_25 > best_f1_25:
            best_f1_25 = avg_f1_25
            best_k_25 = k

    print(
        f"Best k for KNN model with 25% missing ratings: {best_k_25} with F1 Score: {best_f1_25}"
    )

    svd_model = SVD(random_state=3)
    svd_results_25 = evaluate_model(svd_model, train_set, test_set, N_list=N_list)

    # 75% missing ratings
    train_set, test_set = train_test_split(data, test_size=0.75, random_state=22)

    # Best K
    best_k_75 = None
    best_f1_75 = 0
    for k in range(5, 51, 5):
        knn_model = KNNWithMeans(
            k=k, sim_options={"name": "pearson", "user_based": True}, verbose=False
        )
        knn_results_75 = evaluate_model(knn_model, train_set, test_set, N_list=N_list)
        avg_f1_75 = np.mean(knn_results_75[2])

        if avg_f1_75 > best_f1_75:
            best_f1_75 = avg_f1_75
            best_k_75 = k

    print(
        f"Best k for KNN model with 75% missing ratings: {best_k_75} with F1 Score: {best_f1_75}"
    )

    svd_model = SVD(random_state=3)
    svd_results_75 = evaluate_model(svd_model, train_set, test_set, N_list=N_list)

    # Plot results for both 25% and 75% missing ratings
    plot_results(knn_results_25, knn_results_75, N_list)
    plot_results(svd_results_25, svd_results_75, N_list)


if __name__ == "__main__":
    main()
