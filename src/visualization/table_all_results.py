if __name__ == "__main__":
    import sys
    sys.path.append("../")

from translation_key import translation_key
from pathlib import Path

import json
import pandas as pd


def table_data(dt="speed"):
    models = ["INN", "PINN", "LSTM"]

    subsets = [
        "parallel", "offset", "offset_inverse", "far_off_parallel", "mult_path"
    ]

    df = pd.DataFrame()
    for model in models:
        s = []
        for subset in subsets:
            with open(Path(
                    f"../../results/{dt}_{model}_{subset}_results.json")) as f:
                results = json.load(f)

            # Add element to Series
            s.append(
                f"{results['combined'][0]:.2f} (± {results['combined'][1]:.2f})"
            )
        df = pd.concat([df, pd.Series(s, name=model)], axis=1)
    df.rename({x: translation_key[subsets[x]]
               for x in range(len(subsets))},
              axis=0,
              inplace=True)

    print(df.to_latex(bold_rows=True, escape=False, caption=f"Results for {dt} data. The mean and standard deviation of the combined error is shown for each model and subset.", label=f"tab:{dt}_MSE_results"))


def main():
    data_types = ["speed", "volume", "angle", "location"]
    for dt in data_types:
        try:
            print("\n\n------")
            print(f"Getting {dt} latex...\n")
            table_data(dt=dt)
        except Exception as e:
            print(f"Error for {dt}: {e}")
            continue


if __name__ == '__main__':
    main()
