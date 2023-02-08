if __name__=="__main__":
    import sys
    sys.path.append('..')

from matplotlib import pyplot as plt
from sine_path import get_sine_path
plt.rcParams['axes.axisbelow'] = True
plt.rcParams['text.usetex'] = True

import numpy as np

from LSTM.simulation.simulation import v_x, v_y, calculate_path

def draw_arrow(plt, arr_start, arr_end):
       dx = arr_end[0] - arr_start[0]
       dy = arr_end[1] - arr_start[1]
       plt.arrow(arr_start[0], arr_start[1], dx, dy, head_width=10, head_length=25, length_includes_head=True, color='#264653AA')


def main():

    s_list = list(range(1024))


    fig, ax = plt.subplots(1,1)

    paths=[]
    labels = []

    points = [[-500, 0], [500, 0]]
    p = calculate_path(points, 1024, simulation_area_offset=75)
    paths.append(p)
    labels.append("Parallel")

    points = [[-500, 0], [500, 75]]
    p = calculate_path(points, 1024, simulation_area_offset=75)
    paths.append(p)
    labels.append("Offset")

    points = [[-500, 75], [500, 0]]
    p = calculate_path(points, 1024, simulation_area_offset=75)
    paths.append(p)
    labels.append("Inverse Offset")

    points = [[-500, 150], [500, 150]]
    p = calculate_path(points, 1024, simulation_area_offset=75)
    paths.append(p)
    labels.append("Far Off Parallel")

    points = [[-500, 0], [-300, 75], [-100, 0], [100, 75], [300,0], [500, 75]]
    p = calculate_path(points, 1024, simulation_area_offset=75)
    paths.append(p)
    labels.append("Saw")

    x, y = get_sine_path()
    points = list(zip(x,y))
    p = calculate_path(points, 1024, simulation_area_offset=75)
    paths.append(p)
    labels.append("Sine")



    # For each subplot
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['bottom'].set_visible(False)
    # ax.spines['left'].set_visible(False)

    ## Sphere sim
    ax.set_title("Sphere Paths")
    # Sphere

    colors = ['#2A9D8FFF', '#E9C46AFF', '#0496FFFF', '#F4A261FF', '#E76F51FF', "#550527FF"]
    assert len(paths) == len(colors), "length color and paths needs to be the same"
    for idx, path in enumerate(paths):
        # color="#E9C46A"
        ax.plot([x[0] for x in path], [y[1] for y in path], zorder=1, linestyle="dashed", color=colors[idx], label=labels[idx])

    draw_arrow(plt, [-500, 33], [500,33])

    # Sensors
    num_sensors = 8
    sensors_x = list(np.linspace(-200, 200, num=num_sensors))
    sensors_y = [0] * num_sensors
    ax.scatter(sensors_x, sensors_y, s=60, color="#2A9D8F", zorder=1)
    ax.set_ylim((0,250))
    ax.set_xticks([-500, -250, 0, 250, 500])
    ax.set_yticks([0, 75, 150, 225])

    # Axes
    ax.set_ylabel("$d$ (mm)")
    ax.set_xlabel("$s$ (mm)")

    plt.legend(loc="lower left", prop={"size": 7})

    plt.savefig("./path_example.pdf")


if __name__ == '__main__':
    main()
