if __name__ == "__main__":
    import sys

    sys.path.append("..")

from matplotlib import pyplot as plt

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["text.usetex"] = True

import numpy as np

from LSTM.simulation.simulation import v_x, v_y


def main():

    s_list = list(range(1024))

    x = 700
    y = 200

    vx_list = []
    vy_list = []

    for s in s_list:
        vx_list.append(v_x(s, x, y, 0, 20, 20))
        vy_list.append(v_y(s, x, y, 0, 20, 20))

    fig, ax = plt.subplots(3, 1, gridspec_kw={"height_ratios": [5, 1, 1]})

    # For each subplot
    for elem in ax:
        elem.spines["top"].set_visible(False)
        elem.spines["right"].set_visible(False)
        elem.spines["bottom"].set_visible(False)
        elem.spines["left"].set_visible(False)

    ## Sphere sim
    ax[0].set_title("Simulation Environment")
    # Sphere
    ax[0].scatter(x, y, s=150, label="Sphere", color="#E9C46A", zorder=1)

    # Sensors
    num_sensors = 8
    sensors_x = list(
        np.linspace(
            int(512 - (200 / 500 * 512)), int(512 + (200 / 500 * 512)), num=num_sensors
        )
    )
    sensors_y = [190] * num_sensors
    ax[0].scatter(
        sensors_x, sensors_y, s=60, label="Sphere", color="#265650CA", zorder=1
    )

    # Guide lines
    ax[0].vlines(x, 190, y, linestyle="dashed", zorder=0, color="#2646532F")
    ax[0].vlines(512, 190, 210, linestyle="solid", zorder=0, color="#264653AA")
    ax[0].hlines(y, 512, x, linestyle="dashed", zorder=0, color="#2646532F")

    # Axes
    ax[0].xaxis.set_visible(False)
    ax[0].spines["top"].set_visible(True)
    ax[0].spines["right"].set_visible(True)
    ax[0].spines["bottom"].set_visible(True)
    ax[0].spines["left"].set_visible(True)
    ax[0].set_xlim((0, 1024))
    ax[0].set_ylim((190, 210))
    ax[0].set_yticks([190, 200, 210], [0, 75, 150])
    ax[0].set_ylabel("$d$")

    ## Vx profile
    ax[1].set_title("$v_x$", loc="left")
    ax[1].plot(vx_list, label="$v_x$", color="#2A9D8F")
    ax[1].xaxis.set_visible(False)
    ax[1].yaxis.set_visible(False)

    ## Vy profile
    ax[2].set_title("$v_y$", loc="left")
    ax[2].plot(vy_list, label="$v_y$", color="#E76F51")
    ax[2].yaxis.set_visible(False)
    ax[2].set_xticks([0, 512, 1024], [-500, 0, 500])
    ax[2].spines["bottom"].set_visible(True)
    ax[2].set_xlabel("$s$")

    plt.savefig("./simulation_environment.pdf")


if __name__ == "__main__":
    main()
