import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt


def system(y, t):
    y1, y2 = y
    dydt = [y1 - y2, y2]
    return dydt


def odeint_interpolate(system, init_cond, interval_low, interval_high, samples):
    t = np.linspace(interval_low, interval_high, samples)
    sol = odeint(system, init_cond, t)
    return [interp1d(t, sol[:, i], kind="cubic") for i in range(len(sol[0]))]
    # return sol


def main():
    t = np.linspace(0, 10, 100)
    y1, y2 = odeint_interpolate(system, [3, 1], 0, 10, 100)
    y1_eval = [y1(ti) for ti in t]
    y2_eval = [y2(ti) for ti in t]
    plt.plot(t, y1_eval, "r", label="y1")
    plt.plot(t, y2_eval, "b", label="y2")

    # sol = odeint_interpolate(system, [3, 1], 0, 10, 100)
    # plt.plot(t, sol[:, 0], "b", label="y1(t)")
    # plt.plot(t, sol[:, 1], "g", label="y2(t)")

    plt.legend(loc="best")
    plt.xlabel("t")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
