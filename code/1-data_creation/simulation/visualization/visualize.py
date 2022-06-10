import numpy as np
import pandas as pd
from manim import *

Y_OFFSET = -3.25


class VisualizeVelocityProfiles(Scene):
    def construct(self):
        vx_arr = pd.read_csv("../data/a1_normw3_theta0/simdata_vx.csv")
        vy_arr = pd.read_csv("../data/a1_normw3_theta0/simdata_vy.csv")
        print(len(vx_arr))

        x_range = (-300, 300)
        y_range = (10, 40)

        # Add axes
        ax = Axes(
            x_range=[-300, 300, 300],
            y_range=[10, 100, 90],
            tips=False,
            axis_config={"include_numbers": True},
        )
        self.add(ax)

        # Move Sphere through Tank
        circle = Circle(radius=0.1).set_fill(RED, opacity=1.0)
        self.add(circle)
        sensor_list = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]
        for x in range(x_range[0], x_range[1], 200):
            for y in range(y_range[0], y_range[1], 10):
                # Move sphere
                circle2 = Circle(radius=0.1).set_fill(RED, opacity=1.0)
                circle2.set_x(x)

                # Move velocity profile vectors
                vecs = []
                for sensor in sensor_list:
                    # vy
                    vy = vy_arr.iloc[y * (x_range[1] * 2) + x +
                                     300][sensor] * 1000000
                    print(vy)
                    new_arrow = Arrow([sensor, Y_OFFSET, 0],
                                      [sensor, Y_OFFSET + vy, 0],
                                      stroke_width=4,
                                      max_tip_length_to_length_ratio=0.25,
                                      color=RED)
                    vecs.append(new_arrow)

                    # vx
                    vx = vx_arr.iloc[y * (x_range[1] * 2) + x +
                                     300][sensor] * 1000000
                    new_arrow = Arrow([sensor - 0.25, Y_OFFSET + 0.25, 0],
                                      [sensor + vx, Y_OFFSET + 0.25, 0],
                                      stroke_width=4,
                                      max_tip_length_to_length_ratio=0.23,
                                      color=BLUE)
                    vecs.append(new_arrow)
                for vec in vecs:
                    self.add(vec)

                # Run animation
                self.play(FadeTransform(circle, circle2))

                # Clean up
                self.remove(circle2)
                circle.set_x(x)
                for vec in vecs:
                    self.remove(vec)
