if __name__ == "__main__":
    import sys

    sys.path.append("../")

import numpy as np
from lib.params import Data, Settings

from LSTM.simulation.simulation import simulate_single_step
from colour import Color

from matplotlib.colors import ListedColormap

from matplotlib import pyplot as plt

plt.rcParams["axes.axisbelow"] = True
plt.rcParams["text.usetex"] = True


def get_data(subset="parallel"):
    train_location = f"../../data/simulation_data/{subset}/combined.npy"
    trained_model_location = "../../data/trained_models/window_size:16&stride:2&n_nodes:128&alpha:0.05&decay:1e-09&n_epochs:8&shuffle_data:True&data_split:0.8&dropout_ratio:0&ac_fun:tanh"
    settings = Settings.from_model_location(
        trained_model_location, data_location=train_location
    )

    settings.shuffle_data = True
    settings.num_sensors = 8
    settings.seed = 42

    # Load data
    return Data(settings, train_location)


def sensor_difference_visualization():
    for num_sensors in [8, 64]:
        input_sensors = np.linspace(-200, 200, num_sensors)
        # input_sensors = input_sensors[(num_sensors // 2 - (wanted_num_sensors // 2)):(num_sensors // 2 + (wanted_num_sensors // 2))]
        data = simulate_single_step(
            input_sensors,
            x=0,
            y=75,
            theta=0,
            a=10,
            norm_w=10,
            add_noise=True,
            noise_power=1.5e-5,
        )

        vx = data[0::2]
        vy = data[1::2]

        plt.figure()
        plt.xlim((-200, 200))

        plt.gca().hlines(0, -200, 200, linestyle="solid", zorder=0, color="#26465333")
        plot_velocity_profiles(
            input_sensors, vx, vy, title=f"Velocity Profiles for {num_sensors} Sensors"
        )
    plt.show()


def speed_difference_visualization():
    blue = Color("#2A9D8F")
    orange = Color("#E76F51")

    # Show the effect of changing speed
    plt.figure()
    a_list = [10, 20, 30, 40, 50]
    color_gradient = list(blue.range_to(orange, len(a_list)))
    for a in reversed(a_list):
        input_sensors = np.linspace(-200, 200, 64)
        data = simulate_single_step(
            input_sensors,
            x=0,
            y=75,
            theta=0,
            a=10,
            norm_w=a,
            add_noise=True,
            noise_power=1.5e-5,
        )

        vx = data[0::2]
        vy = data[1::2]

        plot_velocity_profiles(
            input_sensors,
            vx,
            vy,
            title=f"$v_y$ for Various Speeds",
            plot_vx=False,
            vy_label=r"$\left\Vert W \right\Vert={}$".format(a),
            vy_color=color_gradient.pop().hex_l,
        )
    plt.ylim((-0.15, 0.15))
    plt.xlim((-200, 200))

    plt.savefig(
        "../../Thesis/images/speed_difference_visualization_vy.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.figure()
    a_list = [10, 20, 30, 40, 50]
    color_gradient = list(blue.range_to(orange, len(a_list)))
    for a in reversed(a_list):
        input_sensors = np.linspace(-200, 200, 64)
        data = simulate_single_step(
            input_sensors,
            x=0,
            y=75,
            theta=0,
            a=10,
            norm_w=a,
            add_noise=True,
            noise_power=1.5e-5,
        )

        vx = data[0::2]
        vy = data[1::2]

        plot_velocity_profiles(
            input_sensors,
            vx,
            vy,
            title=f"$v_x$ for Various Speeds",
            plot_vy=False,
            vx_label=r"$\left\Vert W \right\Vert={}$".format(a),
            vx_color=color_gradient.pop().hex_l,
        )
    plt.ylim((-0.15, 0.15))
    plt.xlim((-200, 200))

    plt.savefig(
        "../../Thesis/images/speed_difference_visualization_vx.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    # plt.show()


def volume_difference_visualization():
    blue = Color("#2A9D8F")
    orange = Color("#E76F51")

    # Show the effect of changing volume
    plt.figure()
    a_list = [10, 20, 30, 40, 50]
    color_gradient = list(blue.range_to(orange, len(a_list)))
    for a in reversed(a_list):
        input_sensors = np.linspace(-200, 200, 64)
        data = simulate_single_step(
            input_sensors,
            x=0,
            y=75,
            theta=0,
            a=a,
            norm_w=10,
            add_noise=True,
            noise_power=1.5e-5,
        )

        vx = data[0::2]
        vy = data[1::2]

        plot_velocity_profiles(
            input_sensors,
            vx,
            vy,
            title=f"$v_y$ for Various Volumes",
            plot_vx=False,
            vy_label=r"$a={}$".format(a),
            vy_color=color_gradient.pop().hex_l,
        )
    plt.ylim((-2, 2))
    plt.xlim((-200, 200))

    plt.savefig(
        "../../Thesis/images/volume_difference_visualization_vy.pdf",
        dpi=300,
        bbox_inches="tight",
    )

    plt.figure()
    a_list = [10, 20, 30, 40, 50]
    color_gradient = list(blue.range_to(orange, len(a_list)))
    for a in reversed(a_list):
        input_sensors = np.linspace(-200, 200, 64)
        data = simulate_single_step(
            input_sensors,
            x=0,
            y=75,
            theta=0,
            a=a,
            norm_w=10,
            add_noise=True,
            noise_power=1.5e-5,
        )

        vx = data[0::2]
        vy = data[1::2]

        plot_velocity_profiles(
            input_sensors,
            vx,
            vy,
            title=f"$v_x$ for Various Volumes",
            plot_vy=False,
            vx_label=r"$a={}$".format(a),
            vx_color=color_gradient.pop().hex_l,
        )
    plt.ylim((-2, 2))
    plt.xlim((-200, 200))

    plt.savefig(
        "../../Thesis/images/volume_difference_visualization_vx.pdf",
        dpi=300,
        bbox_inches="tight",
    )


def plot_velocity_profiles(
    input_sensors,
    vx,
    vy,
    title="Velocity Profiles",
    plot_vx=True,
    plot_vy=True,
    vx_label="$v_x$",
    vy_label="$v_y$",
    vx_color="#2A9D8F",
    vy_color="#E76F51",
):
    if plot_vx:
        plt.plot(input_sensors, vx, label=vx_label, color=vx_color)
    if plot_vy:
        plt.plot(input_sensors, vy, label=vy_label, color=vy_color)
    plt.xlabel("Sensor Position (mm)")
    plt.legend()
    ax = plt.gca()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(True)
    ax.spines["left"].set_visible(True)
    # remove left label
    ax.yaxis.set_ticks_position("none")
    plt.title(title)


def signaltonoise_dB(a, axis=0, ddof=0):
    a = np.asanyarray(a)
    m = a.mean(axis)
    sd = a.std(axis=axis, ddof=ddof)
    return 20 * np.log10(abs(np.where(sd == 0, 0, m / sd)))


def contour_plot():
    input_sensors = [0]
    x = np.linspace(-500, 500, num=256)
    y = np.linspace(0.005, 250, num=256)
    X, Y = np.meshgrid(x, y)

    # Calculate the Vx and Vy for each point
    data = simulate_single_step(
        input_sensors,
        x=X,
        y=Y,
        theta=0,
        a=30,
        norm_w=30,
        add_noise=True,
        noise_power=1.5e-5,
    )
    # for idx_x, x_val in enumerate(X):
    #     data.append([])
    #     for y_val in Y:
    #         data[idx_x].append(simulate_single_step(input_sensors, x=x_val, y=y_val, theta=0, a=30, norm_w=30, add_noise=True, noise_power=1.5e-5))

    data = np.array(data)

    noise_level = np.std(data)
    SNR = data / noise_level

    vx = SNR[0, :, :]
    vy = SNR[1, :, :]

    print(vy.shape)
    print(vx.shape)

    fig, ax = plt.subplots()

    # Plot the contours
    blue = Color("#2A9D8F")
    orange = Color("#E76F51")

    # Show the effect of changing volume
    num_levels = 20
    color_gradient = list(orange.range_to(blue, num_levels))
    thesis_cmap = ListedColormap([color.hex_l for color in color_gradient])
    cs = ax.contourf(X, Y, vy, num_levels, cmap=thesis_cmap)
    cbar = fig.colorbar(cs)

    # plot_velocity_profiles(input_sensors, vx[0], vy[0], title=f"$v_y$ for Volume", plot_vx=False, vy_label="a=10", vy_color="#2A9D8F")
    # plot_velocity_profiles(input_sensors, vx[-1], vy[-1], title=f"$v_y$ for Volume", plot_vx=False, vy_label="a=50", vy_color="#E76F51")

    plt.show()

    # data = simulate_single_step(input_sensors, x=x, y=y, theta=0, a=10, norm_w=10, add_noise=True, noise_power=1.5e-5)

    # vx = data[0::2]
    # vy = data[1::2]

    # print(vx.shape)


if __name__ == "__main__":
    # sensor_difference_visualization()
    speed_difference_visualization()
    volume_difference_visualization()
