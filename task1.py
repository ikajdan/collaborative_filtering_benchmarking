import matplotlib.pyplot as plt
import seaborn as sns
from surprise import Dataset, KNNWithMeans, Reader, accuracy
from surprise.model_selection import train_test_split


def plot_results(k_values, mae_results):
    """
    Plot the MAE results for different K values.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 7))

    sns.lineplot(
        x=k_values,
        y=mae_results[0.25],
        label="25% Missing Ratings",
        marker="o",
        color="b",
    )
    sns.lineplot(
        x=k_values,
        y=mae_results[0.75],
        label="75% Missing Ratings",
        marker="s",
        color="orange",
    )

    plt.xlabel("K (Number of Neighbors)", fontsize=14)
    plt.ylabel("MAE (Mean Absolute Error)", fontsize=14)
    plt.legend(title="Dataset Split", loc="upper right", fontsize=12)
    plt.grid(visible=True)
    plt.show()


def main():
    reader = Reader(line_format="user item rating timestamp", sep="\t")
    data = Dataset.load_from_file("./ml-100k/u.data", reader=reader)
    # data = Dataset.load_builtin("ml-100k")

    test_sizes = [0.25, 0.75]
    k_values = range(1, 101)
    mae_results = {0.25: [], 0.75: []}

    for test_size in test_sizes:
        train_set, test_set = train_test_split(
            data, test_size=test_size, random_state=42
        )

        for k in k_values:
            knn = KNNWithMeans(k=k, sim_options={"user_based": True}, verbose=False)
            knn.fit(train_set)
            predictions = knn.test(test_set)
            mae = accuracy.mae(predictions, verbose=False)
            mae_results[test_size].append(mae)

    plot_results(k_values, mae_results)

    for test_size in test_sizes:
        missing_percent = int(test_size * 100)
        print(f"\nMAE values for {missing_percent}% missing ratings:")
        for k, mae in zip(k_values, mae_results[test_size]):
            print(f"K = {k}: MAE = {mae:.4f}")


if __name__ == "__main__":
    main()
