if __name__ == "__main__":
    import sys
    sys.path.append("../")

from translation_key import translation_key
from pathlib import Path

import json
import pandas as pd


def table_data(dt="speed", noise_experiment=False):
    models = ["INN", "PINN", "LSTM"]

    if noise_experiment:
        subsets = [
            "low_noise_parallel", "high_noise_parallel",
            "low_noise_saw", "high_noise_saw",
        ]
    else:
        subsets = [
            "parallel", "offset", "offset_inverse", "far_off_parallel", "mult_path", "sine"
        ]

    # Find best mean value per path between the models
    best_model_idx = {}

    for subset in subsets:
        best_model = None
        best_value = 999999999999999999
        for idx, model in enumerate(models):
            with open(Path(
                    f"../../results/{dt}_{model}_{subset}_results.json")) as f:
                results = json.load(f)
            if results['combined'][0] < best_value:
                best_value = results['combined'][0]
                best_model = idx
        if best_model != None:
            best_model_idx[f"{model}_{subset}"] = best_model
        else:
            print("Something might have gone wrong")
            best_model_idx[f"{model}_{subset}"] = 0


    df = pd.DataFrame()
    for idx, model in enumerate(models):
        s = []
        for subset in subsets:
            with open(Path(
                    f"../../results/{dt}_{model}_{subset}_results.json")) as f:
                results = json.load(f)

            # Add element to Series
            if f"{model}_{subset}" in best_model_idx and best_model_idx[f"{model}_{subset}"] == idx:
                s.append(
                    "\\textbf{"+ f"{results['combined'][0]:.2f}" + "(±" + f"{results['combined'][1]:.2f}" + ")}"
                )
            else:
                s.append(
                    f"{results['combined'][0]:.2f} (±{results['combined'][1]:.2f})"
                )
        df = pd.concat([df, pd.Series(s, name=model)], axis=1)
    df.rename({x: translation_key[subsets[x]]
               for x in range(len(subsets))},
              axis=0,
              inplace=True)

    print(df.style.to_latex(position_float="centering", hrules=True, caption="Results for \\textbf{" + dt + "} estimation. The mean and standard deviation of the combined RMS error over 25 runs is shown for each model and path. The units are in degrees. The best mean per path is shown in bold, lower is better.", label=f"tab:{dt}_MSE_results", ))


def main():
    data_types = ["speed", "volume", "angle", "location"]
    for dt in data_types:
        # try:
        print("\n\n------")
        print(f"Getting {dt} latex...\n")
        table_data(dt=dt, noise_experiment=False)
        # except Exception as e:
        #     print(f"Error for {dt}: {e}")
        #     continue


if __name__ == '__main__':
    main()
