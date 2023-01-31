import pandas as pd

from tabulate import tabulate

# Below are all the styles that you can use for tabulate:

#     “plain”
#     “simple”
#     “github”
#     “grid”
#     “fancy_grid”
#     “pipe”
#     “orgtbl”
#     “jira”
#     “presto”
#     “pretty”
#     “psql”
#     “rst”
#     “mediawiki”
#     “moinmoin”
#     “youtrack”
#     “html”
#     “latex”
#     “latex_raw”
#     “latex_booktabs”
#     “textile”

def main():
    model_types = ["inn", "pinn", "lstm"]

    for model_type in model_types:
        print("Results for model type: ", model_type)
        inn_df = pd.read_csv(f"./gridsearch_results/{model_type}_score.csv")
        inn_df = inn_df.sort_values("weighted_score")

        inn_df = inn_df.drop("weighted_score", axis=1)
        inn_df = inn_df.drop("weighted_score_std", axis=1)

        if model_type != "lstm":
            inn_df = inn_df.drop("MSE_forward", axis=1)
            inn_df = inn_df.drop("MSE_forward_std", axis=1)
            inn_df.rename(columns={"MSE_backward": "localization_error"}, inplace=True)
            inn_df.rename(columns={"MSE_backward_std": "localization_error_std"}, inplace=True)

            inn_df["localization_error"] = (inn_df["localization_error"] * 0.1).round(2).astype(str) + " (±" + inn_df["localization_error_std"].round(2).astype(str) + ")"
            inn_df["volume_error"] = (inn_df["volume_error"] * 0.1).round(2).astype(str) + " (±" + inn_df["volume_error_std"].round(2).astype(str) + ")"
            inn_df["speed_error"] = (inn_df["speed_error"] * 0.1).round(2).astype(str) + " (±" + inn_df["speed_error_std"].round(2).astype(str) + ")"
        else:
            inn_df["localization_error"] = inn_df["localization_error"].round(2).astype(str) + " (±" + inn_df["localization_error_std"].round(2).astype(str) + ")"
            inn_df["volume_error"] = inn_df["volume_error"].round(2).astype(str) + " (±" + inn_df["volume_error_std"].round(2).astype(str) + ")"
            inn_df["speed_error"] = inn_df["speed_error"].round(2).astype(str) + " (±" + inn_df["speed_error_std"].round(2).astype(str) + ")"


        inn_df = inn_df.drop("volume_error_std", axis=1)
        inn_df = inn_df.drop("localization_error_std", axis=1)
        inn_df = inn_df.drop("speed_error_std", axis=1)

        print(tabulate(inn_df, headers="keys", tablefmt="latex_booktabs", showindex=False))
        print('\n')

    # lstm_df = pd.read_csv("./gridsearch_results/lstm_score.csv")
    # print(lstm_df.head())
    # lstm_df = lstm_df.sort_values("weighted_score")

    # lstm_df = lstm_df.drop("weighted_score", axis=1)

    # print(tabulate(lstm_df, headers="keys", tablefmt="latex"))

if __name__ == '__main__':
    main()
