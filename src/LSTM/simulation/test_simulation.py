import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], ".."))

import numpy as np

from simulation import calculate_angle, calculate_path, v_x, v_y


def test_calculate_angle_0():
    start_point = [50, 50]
    terminal_point = [100, 50]
    a = calculate_angle(start_point, terminal_point)
    assert a == 0


def test_calculate_angle_90():
    start_point = [50, 50]
    terminal_point = [50, 100]
    a = calculate_angle(start_point, terminal_point)
    assert a == 90


def test_calculate_angle_180():
    start_point = [50, 50]
    terminal_point = [-50, 50]
    a = calculate_angle(start_point, terminal_point)
    assert a == 0


def test_calculate_angle_45():
    start_point = [50, 50]
    terminal_point = [75, 75]
    a = calculate_angle(start_point, terminal_point)
    assert a == 45


def test_calculate_path_simple_with_y_change():
    points = [[-500, 0], [500, 200]]
    path = calculate_path(points, 1024, simulation_area_offset=75)
    assert len(path) == 1024
    dist = np.linalg.norm(np.array(path[0]) - np.array(path[-1]))
    assert dist == 1019.803902718557


def test_calculate_path_multiple_points_with_y_change():
    points = [[-500, 0], [-200, 30], [500, 900]]
    path = calculate_path(points, 1024, simulation_area_offset=75)
    assert len(path) == 1024
    dist = np.linalg.norm(np.array(path[0]) - np.array(path[-1]))
    assert dist == 1345.362404707371


def test_calculate_path_simple():
    points = [[-500, 0], [500, 0]]
    path = calculate_path(points, 1024, simulation_area_offset=75)
    assert len(path) == 1024
    dist = np.linalg.norm(np.array(path[0]) - np.array(path[-1]))
    assert dist == 1000.0


def test_calculate_path_multiple_points():
    points = [[-500, 0], [-200, 0], [500, 0]]
    path = calculate_path(points, 1024, simulation_area_offset=75)
    assert len(path) == 1024
    dist = np.linalg.norm(np.array(path[0]) - np.array(path[-1]))
    assert dist == 1000.0


def test_vx():
    assert v_x(0, 50, 75, 0, 10, 10) == -0.0005251599490889745


def test_vy():
    assert v_y(0, 50, 75, 0, 10, 10) == -0.009452879083601538
