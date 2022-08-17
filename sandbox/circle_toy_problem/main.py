import deepxde as dde
import numpy as np
import math

K = 4
UNKNOWN_K = dde.Variable(3.5)

def circle_func(x, y):
    x1, y1 = x[:, 0:1], y[:, 0:1]

    return x1**2 + y1**2 - UNKNOWN_K

def sol(x):
    if x > 0:
        x *= -1
    return np.sqrt(-x**2 + K)

def gen_traindata(samples=100):
    angle = np.linspace( 0 , 2 * np.pi , samples ).reshape(samples,1)
    x = K * np.cos(angle)
    y = K * np.sin(angle)
    return x, y

def main():
    geom = dde.geometry.Interval(-2,2)

    ob_x, ob_y = gen_traindata()
    observed_pointset = dde.icbc.PointSetBC(ob_x, ob_y, component=0)

    bc1 = dde.icbc.DirichletBC(geom, sol, lambda _, on_boundary: on_boundary, component=0)

    data = dde.data.PDE(geom, circle_func, [bc1, observed_pointset], num_domain=400, num_boundary=30, anchors=ob_x)

    net = dde.nn.FNN([1] + [40] * 3 + [3], "tanh", "Glorot uniform")
    model = dde.Model(data, net)
    model.compile("adam", lr=0.001, external_trainable_variables=[UNKNOWN_K])
    variable = dde.callbacks.VariableValue(
        [UNKNOWN_K], period=600, filename="variables.dat"
    )
    losshistory, train_state = model.train(iterations=60000, callbacks=[variable])
    dde.saveplot(losshistory, train_state, issave=True, isplot=True)



if __name__ == '__main__':
    main()
