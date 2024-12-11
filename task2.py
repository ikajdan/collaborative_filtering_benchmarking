from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from surprise import SVD, Dataset, KNNWithMeans, Reader
from surprise.accuracy import mae
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
        n_rel = sum(true_r >= threshold for _, true_r in user_ratings)
        n_rel_and_rec = sum(true_r >= threshold for _, true_r in user_ratings[:n])

        precisions[uid] = n_rel_and_rec / n
        recalls[uid] = n_rel_and_rec / n_rel if n_rel else 0

    return precisions, recalls


def calculate_f1_score(precision, recall):
    """
    Calculate F1 score.
    """
    return (
        2 * precision * recall / (precision + recall) if precision + recall != 0 else 0
    )


def plot_comparison_metrics(metrics_25, metrics_75):
    """
    Plot comparison metrics.
    """
    metrics_titles = ["MAE", "Precision", "Recall", "F1 Score"]
    svd_25 = metrics_25["SVD"]
    knn_25 = metrics_25["KNN"]
    svd_75 = metrics_75["SVD"]
    knn_75 = metrics_75["KNN"]

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(12, 8))

    x = np.arange(len(metrics_titles))
    width = 0.2

    palette = sns.color_palette("Set2")
    plt.bar(x - 1.5 * width, svd_25, width, label="SVD (25%)", color=palette[0])
    plt.bar(x - 0.5 * width, knn_25, width, label="K-NN (25%)", color=palette[1])
    plt.bar(x + 0.5 * width, svd_75, width, label="SVD (75%)", color=palette[2])
    plt.bar(x + 1.5 * width, knn_75, width, label="K-NN (75%)", color=palette[3])

    plt.xlabel("Metrics", fontsize=14)
    plt.ylabel("Score", fontsize=14)
    plt.xticks(ticks=x, labels=metrics_titles, fontsize=12)
    plt.legend(title="Models and Test Ratios", title_fontsize="13", fontsize=11)

    plt.tight_layout()
    plt.show()


def evaluate_model(algo, train_set, test_set, n=5, threshold=4):
    """
    Evaluate model using MAE, precision, recall, and F1 score.
    """
    algo.fit(train_set)
    predictions = algo.test(test_set)

    mae_score = mae(predictions, verbose=False)
    precisions, recalls = precision_recall_at_n(predictions, n=n, threshold=threshold)

    precision_avg = np.mean(list(precisions.values()))
    recall_avg = np.mean(list(recalls.values()))
    f1_score = calculate_f1_score(precision_avg, recall_avg)

    return mae_score, precision_avg, recall_avg, f1_score


def main():
    reader = Reader(line_format="user item rating timestamp", sep="\t")
    data = Dataset.load_from_file("./ml-100k/u.data", reader=reader)
    # data = Dataset.load_builtin("ml-100k")

    test_sizes = [0.25, 0.75]
    metrics_25 = {"SVD": [], "KNN": []}
    metrics_75 = {"SVD": [], "KNN": []}

    for test_size, metrics in zip(test_sizes, [metrics_25, metrics_75]):
        train_set, test_set = train_test_split(
            data, test_size=test_size, random_state=42
        )

        # SVD Model Evaluation
        svd = SVD(random_state=3)
        mae_svd, pre_svd, recall_svd, f1_svd = evaluate_model(svd, train_set, test_set)
        metrics["SVD"] = [mae_svd, pre_svd, recall_svd, f1_svd]

        # KNN Model Evaluation
        sim_options_knn = {"name": "pearson", "user_based": True}
        knn = KNNWithMeans(k=10, sim_options=sim_options_knn, verbose=False)
        mae_knn, pre_knn, recall_knn, f1_knn = evaluate_model(knn, train_set, test_set)
        metrics["KNN"] = [mae_knn, pre_knn, recall_knn, f1_knn]

    plot_comparison_metrics(metrics_25, metrics_75)


if __name__ == "__main__":
    main()
