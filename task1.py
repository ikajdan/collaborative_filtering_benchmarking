import matplotlib.pyplot as plt
from surprise import Dataset, KNNBasic, accuracy
from surprise.model_selection import train_test_split
from surprise.reader import Reader


def main():
    reader = Reader(line_format="user item rating timestamp", sep="\t")
    data = Dataset.load_from_file("./ml-100k/u.data", reader=reader)
    # data = Dataset.load_builtin("ml-100k")

    test_sizes = [0.25, 0.75]
    mae_results = {0.25: [], 0.75: []}

    k_values = range(1, 50)

    for test_size in test_sizes:
        train_set, test_set = train_test_split(data, test_size=test_size)

        for k in k_values:
            algo = KNNBasic(k=k, sim_options={"user_based": True})
            algo.fit(train_set)

            predictions = algo.test(test_set)

            mae = accuracy.mae(predictions, verbose=False)
            mae_results[test_size].append(mae)

    plt.figure(figsize=(12, 6))

    plt.plot(k_values, mae_results[0.25], label="25% Missing Ratings", marker="o")
    plt.plot(k_values, mae_results[0.75], label="75% Missing Ratings", marker="s")

    plt.xlabel("K (Number of Neighbors)")
    plt.ylabel("MAE (Mean Absolute Error)")
    plt.legend()
    plt.grid()

    plt.show()

    for test_size in test_sizes:
        missing_percent = int(test_size * 100)
        print(f"\nMAE values for {missing_percent}% missing ratings:")
        for k, mae in zip(k_values, mae_results[test_size]):
            print(f"K = {k}: MAE = {mae:.4f}")


if __name__ == "__main__":
    main()
