import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# |                 Main function here is odeint_interpolate:                 |
# |           1- Define a system                                              |
# |           2- Solve with odeint                                            |
# |           3- Interpolate the solution                                     |
# |           4- Return the interpolation                                     |
# ----------------------------------------------------------------------------


def system(y, t):
    S, I, R = y
    dSdt = -0.2 * S * I / 1000
    dIdt = 0.2 * S * I / 1000 - 0.1 * I
    dRdt = 0.1 * I
    return dSdt, dIdt, dRdt


def odeint_interpolate(system, init_cond, interval_low, interval_high, samples):
    t = np.linspace(interval_low, interval_high, samples)
    sol = odeint(system, init_cond, t)
    return [interp1d(t, sol[:, i], kind="cubic") for i in range(len(sol[0]))]
    # return sol


def main():
    t = np.linspace(0, 150, 1000)
    S, I, R = odeint_interpolate(system, [1000, 1, 0], 0, 150, 1000)
    S1_eval = [S(ti) for ti in t]
    I2_eval = [I(ti) for ti in t]
    R2_eval = [R(ti) for ti in t]
    plt.plot(t, S1_eval, "b", label="S")
    plt.plot(t, I2_eval, "r", label="I")
    plt.plot(t, R2_eval, "g", label="R")

    # sol = odeint_interpolate(system, [1000, 1, 0], 0, 100, 1000)
    # plt.plot(t, sol[:, 0], "b", label="S(t)")
    # plt.plot(t, sol[:, 1], "r", label="I(t)")
    # plt.plot(t, sol[:, 2], "g", label="R(t)")

    plt.legend(loc="best")
    plt.xlabel("t")
    plt.grid()
    plt.show()


if __name__ == "__main__":
    main()
