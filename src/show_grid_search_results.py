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
    inn_df = pd.read_csv("./inn_score.csv")
    inn_df = inn_df.sort_values("weighted_score")

    inn_df = inn_df.drop("weighted_score", axis=1)

    print(tabulate(inn_df, headers="keys", tablefmt="latex"))

    lstm_df = pd.read_csv("./lstm_score.csv")
    print(lstm_df.head())
    lstm_df = lstm_df.sort_values("weighted_score")

    lstm_df = lstm_df.drop("weighted_score", axis=1)

    print(tabulate(lstm_df, headers="keys", tablefmt="latex"))

if __name__ == '__main__':
    main()
