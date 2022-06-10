import pandas as pd


def main():
    vx_data = pd.read_csv('./data/a1_normw1_theta0/simdata_vx.csv')
    vy_data = pd.read_csv('./data/a1_normw1_theta0/simdata_vy.csv')

    print(vx_data)


if __name__ == "__main__":
    main()
