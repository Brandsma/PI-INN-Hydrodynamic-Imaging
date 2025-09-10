from pathlib import Path
import pandas


def estimation_accuracy():
    ## TODO: Cross-validation training

    # ELM?

    inverse_sine_data = get_sine_data()
    models = [LSTM(), INN(), INN_PINN()]

    ## Inverse Sine
    # LSTM, INN, INN+PINN, (INN+PINN+LSTM)
    # Get polarity, derivative, original angle
    for model in models:
        result: pandas.DataFrame = model.predict_data()

        result_filename = Path(
            f"../results/inverse_sine/{model.__class__.__name__}_experimentI.csv"
        )
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        result.to_csv(result_filename)

    ALL_vp_sphere_data = get_ALL_vp_data()
    models = [LSTM(), INN(), INN_PINN()]

    ## ALL Velocity Potential Sphere
    # LSTM, INN, INN+PINN, (INN+PINN+LSTM)

    for model in models:
        result: pandas.DataFrame = model.predict_data()

        result_filename = Path(
            f"../results/ALL_vp/{model.__class__.__name__}_experimentI.csv"
        )
        result_filename.parent.mkdir(parents=True, exist_ok=True)

        result.to_csv(result_filename)
        # Get volume calculations

        # Get speed calculations

        # Get localization calculations

    ## ALL Stream Function Sphere
    # LSTM, INN, INN+PINN, (INN+PINN+LSTM)
    pass


def sensitivity_analysis():
    pass


def main():
    estimation_accuracy()
    sensitivity_analysis()


if __name__ == "__main__":
    main()
