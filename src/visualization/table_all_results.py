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


    df = pd.DataFrame()
    for model in models:
        s = []
        for subset in subsets:
            with open(Path(
                    f"../../results/{dt}_{model}_{subset}_results.json")) as f:
                results = json.load(f)

            # Add element to Series
            s.append(
                f"{results['combined'][0]:.2f} (±{results['combined'][1]:.2f})"
            )
        df = pd.concat([df, pd.Series(s, name=model)], axis=1)
    df.rename({x: translation_key[subsets[x]]
               for x in range(len(subsets))},
              axis=0,
              inplace=True)

    print(df.style.to_latex(position_float="centering", hrules=True, caption=f"Results for {dt} estimation under varying noise conditions. The mean and standard deviation of the combined RMS error is shown for the INN and PI/INN models for separate paths..", label=f"tab:{dt}_MSE_noisy_results", ))


def main():
    data_types = ["speed", "volume", "angle", "location"]
    for dt in data_types:
        try:
            print("\n\n------")
            print(f"Getting {dt} latex...\n")
            table_data(dt=dt, noise_experiment=False)
        except Exception as e:
            print(f"Error for {dt}: {e}")
            continue


if __name__ == '__main__':
    main()
